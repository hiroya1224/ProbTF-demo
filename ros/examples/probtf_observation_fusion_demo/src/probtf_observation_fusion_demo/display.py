"""Render queries of one immutable core snapshot; never propagate transforms here."""

from dataclasses import dataclass
import io
import json

import numpy as np
import rospy
from scipy.special import ndtri
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from probtf.spherical_law.support import covariance_support


AXIS_COLORS = {
    "origin": (0.18, 0.20, 0.25),
    "x": (0.90, 0.12, 0.12),
    "y": (0.10, 0.65, 0.20),
    "z": (0.12, 0.30, 0.95),
    "observed": (0.60, 0.15, 0.75),
}


@dataclass(frozen=True)
class DisplayPoint:
    """A reference to a core result, with rendering metadata only."""

    query: object
    frame_label: str
    axis: str
    component_id: str
    weight: float
    sample_count: int
    log_weight: object = None


@dataclass(frozen=True)
class DisplaySnapshot:
    points: tuple
    frame_id: str
    stamp: float
    command: tuple
    version: int
    stage: str
    observation_ids: tuple
    summary: str
    show_labels: bool = True
    stiffness_record: object = None
    goal_memory: object = None
    behavior: object = None


def query_record(point):
    """Numbers sent to the adapter, also recorded beside the actual wire data."""
    query = point.query
    return {
        "query_id": query.query_id,
        "source_component_id": query.component_id,
        "source_state_id": query.state_id,
        "source_frame": query.frame,
        "source_stamp": query.stamp,
        "source_version": query.version,
        "source_command": list(query.command),
        "approximation_order": query.approximation_order,
        "frame_label": point.frame_label,
        "axis": point.axis,
        "component_id": point.component_id,
        "weight": point.weight,
        "log_weight": point.log_weight,
        "sample_count": point.sample_count,
        "mean": query.mean.tolist(),
        "covariance": query.covariance.tolist(),
        "within": query.within.tolist(),
        "between": query.between.tolist(),
    }


def snapshot_record(snapshot):
    record = {
        "frame_id": snapshot.frame_id,
        "stamp": snapshot.stamp,
        "command": list(snapshot.command),
        "version": snapshot.version,
        "stage": snapshot.stage,
        "summary": snapshot.summary,
        "show_labels": snapshot.show_labels,
        "observation_ids": list(snapshot.observation_ids),
        "render_rule": "deterministic Halton Gaussian quantiles; display queries only",
        "queries": [query_record(point) for point in snapshot.points],
    }
    if snapshot.stiffness_record is not None:
        record["stiffness"] = snapshot.stiffness_record
    if snapshot.goal_memory is not None:
        record['fixed_goal_memory'] = snapshot.goal_memory
    if snapshot.behavior is not None:
        record['behavior'] = snapshot.behavior
    return record


def _normal_points(query, count, offset):
    # Zero normal variance stays zero. Only round-off negative eigenvalues are
    # treated as zero; significant negative variance is an error, not a fallback.
    support = covariance_support(query.covariance, "display query " + query.query_id)
    # A deterministic equal-mass quantile sequence is a rendering query, not
    # a Monte Carlo approximation of the source law. Its empirical moments
    # never enter inference, control, stiffness, or another propagation step.
    indices = np.arange(int(count), dtype=np.int64) + int(offset) + 1
    coordinates = np.empty((int(count), support.rank))
    for axis, base in enumerate((2, 3, 5)[:support.rank]):
        digits, factor = indices.copy(), 1. / base
        uniform = np.zeros(int(count))
        while np.any(digits):
            uniform += factor * (digits % base)
            digits //= base
            factor /= base
        coordinates[:, axis] = ndtri(uniform)
    return query.mean + coordinates @ (
        support.basis * np.sqrt(support.eigenvalues)
    ).T


def _marker(snapshot, marker_id, namespace, marker_type):
    marker = Marker()
    marker.header.frame_id = snapshot.frame_id
    marker.header.stamp = rospy.Time.from_sec(snapshot.stamp)
    marker.ns = namespace
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    return marker


def make_markers(snapshot, seed=29, point_size=0.0074999998323619365):
    """Render deterministic Gaussian quantiles of the actual core queries.

    The historical ``seed`` setting is retained as the sequence's integer
    offset. No random generator is used, and no source parameter is altered.
    """
    if int(seed) < 0:
        raise ValueError("The Gaussian quantile sequence offset must be nonnegative.")
    clear = _marker(snapshot, 0, "clear", Marker.SPHERE_LIST)
    clear.action = Marker.DELETEALL
    output = [clear]
    groups = {}
    for index, point in enumerate(snapshot.points):
        marker = _marker(
            snapshot, index, "gaussian/" + point.frame_label + "/" + point.axis,
            Marker.SPHERE_LIST,
        )
        marker.scale.x = marker.scale.y = marker.scale.z = float(point_size)
        # Each component retains its exact weight in the input snapshot. Alpha
        # encodes that weight; its covariance and Gaussian coordinates are intact.
        marker.color = ColorRGBA(*AXIS_COLORS[point.axis], 0.85 * point.weight)
        marker.points = [Point(*value) for value in _normal_points(
            point.query, point.sample_count, int(seed)
        )]
        output.append(marker)
        groups.setdefault((point.frame_label, point.component_id), {})[point.axis] = point
    offset = len(snapshot.points)
    component_ids = tuple(dict.fromkeys(point.component_id for point in snapshot.points))
    for index, ((label, component), group) in enumerate(groups.items()):
        origin = group["origin"].query.mean
        marker = _marker(snapshot, offset + index, "axes/" + label, Marker.LINE_LIST)
        marker.scale.x = 0.00225
        for axis in (axis for axis in ("x", "y", "z", "observed") if axis in group):
            marker.points.extend((Point(*origin), Point(*group[axis].query.mean)))
            color = ColorRGBA(*AXIS_COLORS[axis], 0.95 * group[axis].weight)
            marker.colors.extend((color, color))
        output.append(marker)
        text = _marker(snapshot, offset + len(groups) + index, "labels", Marker.TEXT_VIEW_FACING)
        text.pose.position = Point(*(origin + np.array([0., 0., 0.045])))
        text.scale.z = 0.025
        text.color = ColorRGBA(0.10, 0.12, 0.17, 1.0)
        text.text = label + (" [k{}, w={:.3f}]".format(
            component_ids.index(component), group["origin"].weight)
            if len(component_ids) > 1 else "")
        if snapshot.show_labels:
            output.append(text)
    status = _marker(snapshot, 0, "status", Marker.TEXT_VIEW_FACING)
    status.pose.position = Point(0.75, -0.32, 1.05)
    status.scale.z = 0.028
    status.color = ColorRGBA(0.08, 0.10, 0.16, 1.0)
    status.text = snapshot.summary
    output.append(status)
    return MarkerArray(output)


def serialized_marker_array(message):
    buffer = io.BytesIO()
    message.serialize(buffer)
    return buffer.getvalue()


def verify_wire_identity(snapshot, actual, *, seed, point_size):
    """Regenerate from the same source and compare the real ROS wire message."""
    payload = serialized_marker_array(actual)
    received = MarkerArray().deserialize(payload)
    expected = make_markers(snapshot, seed, point_size)
    if serialized_marker_array(expected) != serialized_marker_array(received):
        raise AssertionError("Published MarkerArray differs from the core snapshot query.")
    return payload


def save_snapshot(path, snapshot, message, *, seed, point_size):
    payload = verify_wire_identity(snapshot, message, seed=seed, point_size=point_size)
    path.with_suffix(".markers.bin").write_bytes(payload)
    record = snapshot_record(snapshot)
    record.update({"render_seed": seed, "point_size": point_size,
                   "display_roundoff_tolerance": "core covariance_support: atol+rtol*spectral_radius",
                   "wire_identity_verified": True})
    path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
