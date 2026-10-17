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
from probtf.geometry import rotmat_to_quat
from probtf.spherical_law.support import covariance_support


AXIS_COLORS = {
    "origin": (0.18, 0.20, 0.25),
    "x": (0.90, 0.12, 0.12),
    "y": (0.10, 0.65, 0.20),
    "z": (0.12, 0.30, 0.95),
    "observed": (0.60, 0.15, 0.75),
}

GOAL_AXIS_SHAFT_DIAMETER = 0.012
GOAL_AXIS_HEAD_DIAMETER = 0.024
GOAL_AXIS_HEAD_LENGTH = 0.024


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
    surprise_record: object = None
    candidate_measure: object = None
    sensor_event: object = None


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
        "mode": query.mode.tolist(),
        "mean": query.mean.tolist(),
        "covariance": query.covariance.tolist(),
        "within": query.within.tolist(),
        "between": query.between.tolist(),
    }


def goal_axis_records(snapshot):
    """Describe bold axes from the same goal origin/endpoint queries.

    Each goal component has its own three modal arrows. They are queries of
    that component, not a selected mixture representative or a new law. The
    preexisting mean axes, Gaussian particles and all masses stay intact.
    """
    groups = {}
    for point in snapshot.points:
        if point.frame_label == 'switch_goal':
            group = groups.setdefault(point.component_id, {})
            if point.axis in group:
                raise ValueError('A bold goal axis has a duplicate source endpoint.')
            group[point.axis] = point
    records = []
    memory = snapshot.goal_memory or {}
    for component_index, (component, group) in enumerate(groups.items()):
        if set(group) != {'origin', 'x', 'y', 'z'}:
            raise ValueError('Bold goal axes require the existing origin and XYZ queries for every mode.')
        origin = group['origin']
        source = origin.query
        for index, axis in enumerate(('x', 'y', 'z')):
            endpoint = group[axis]
            query = endpoint.query
            if (query.state_id != source.state_id or query.component_id != source.component_id
                    or query.version != snapshot.version or source.version != snapshot.version
                    or query.stamp != snapshot.stamp or source.stamp != snapshot.stamp
                    or tuple(query.command) != tuple(snapshot.command)
                    or tuple(source.command) != tuple(snapshot.command)
                    or query.frame != snapshot.frame_id or source.frame != snapshot.frame_id
                    or endpoint.weight != origin.weight or endpoint.log_weight != origin.log_weight):
                raise ValueError('Bold goal axes must reference the same goal mode, mass and source snapshot.')
            records.append(dict(namespace='goal_axes/'+component, marker_id=index,
                component_id=component, component_index=component_index,
                child_frame_id='estimated_switch_goal_'+str(component_index), axis=axis,
                weight=endpoint.weight, log_weight=endpoint.log_weight,
                source_component_id=query.component_id, source_state_id=query.state_id,
                source_frame=query.frame, source_stamp=query.stamp,
                source_version=query.version, source_command=list(query.command),
                origin_query_id=source.query_id, endpoint_query_id=query.query_id,
                start=source.mode.tolist(), end=query.mode.tolist(),
                shaft_diameter_m=GOAL_AXIS_SHAFT_DIAMETER,
                head_diameter_m=GOAL_AXIS_HEAD_DIAMETER,
                head_length_m=GOAL_AXIS_HEAD_LENGTH,
                color_rgb=list(AXIS_COLORS[axis]), alpha=.95*endpoint.weight,
                goal_memory_available=bool(memory.get('available', False)),
                goal_observation_stamp=memory.get('observation_stamp'),
                goal_observation_version=memory.get('observation_version'),
                goal_observation_command=memory.get('observation_command'),
                goal_observation_ids=list(memory.get('observation_ids', ())),
                operation='modal frame query of same goal law',
                render_rule='same core goal-origin and axis-endpoint modes; all goal mixture components retained'))
    return records


def goal_pose_records(snapshot):
    """Query each component's modal TF from its same four core endpoints.

    A common positive axis length and an orthonormal right-handed modal basis
    are required. No mean-frame approximation, mixture selection, covariance
    change or projection of a nonorthogonal mean basis is performed here.
    """
    groups = {}
    for axis in goal_axis_records(snapshot):
        groups.setdefault(axis['component_id'], {})[axis['axis']] = axis
    records = []
    for group in groups.values():
        source = group['x']
        origin = np.asarray(source['start'])
        columns = np.column_stack([np.asarray(group[axis]['end'])-origin for axis in ('x', 'y', 'z')])
        lengths = np.linalg.norm(columns, axis=0)
        if (not np.all(np.isfinite(columns)) or lengths[0] <= 0.
                or not np.allclose(lengths, lengths[0], rtol=1e-10, atol=1e-12)):
            raise ValueError('Modal goal TF queries require a common positive axis length.')
        rotation = columns/lengths[0]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-10)
                or not np.isclose(np.linalg.det(rotation), 1., rtol=0., atol=1e-10)):
            raise ValueError('Modal goal TF queries must form a right-handed orthonormal frame.')
        common_keys = ('component_id', 'component_index', 'child_frame_id', 'weight', 'log_weight',
            'source_component_id', 'source_state_id', 'source_frame', 'source_stamp',
            'source_version', 'source_command', 'goal_memory_available',
            'goal_observation_stamp', 'goal_observation_version', 'goal_observation_command',
            'goal_observation_ids', 'operation')
        records.append(dict({key: source[key] for key in common_keys},
            translation=origin.tolist(), rotation_wxyz=rotmat_to_quat(rotation).tolist(),
            rotation_matrix=rotation.tolist(), axis_length_m=float(lengths[0]),
            origin_query_id=source['origin_query_id'],
            endpoint_query_ids={axis: group[axis]['endpoint_query_id'] for axis in ('x', 'y', 'z')},
            modal_endpoints={axis: group[axis]['end'] for axis in ('x', 'y', 'z')}))
    return records


def verify_measure_identity(snapshot):
    """Match the measure's normalized shape to the actual plotted body law."""
    measure = snapshot.candidate_measure
    if measure is None:
        return
    sources = {}
    for point in snapshot.points:
        if point.frame_label in ('marker', 'switch_goal'):
            continue
        query = point.query
        source = (query.state_id, query.version, query.stamp,
                  tuple(query.command), point.log_weight)
        if point.component_id in sources and sources[point.component_id] != source:
            raise ValueError('Candidate-measure display queries have inconsistent body provenance.')
        sources[point.component_id] = source
    actual = list(sources.values())
    expected = list(zip(measure['shape_source_state_ids'], measure['shape_source_versions'],
        measure['shape_source_stamps'], map(tuple, measure['shape_source_commands']),
        measure['shape_log_weights']))
    lengths = [len(measure[key]) for key in ('shape_source_state_ids',
        'shape_source_versions', 'shape_source_stamps', 'shape_source_commands', 'shape_log_weights')]
    if not actual or len(set(lengths)) != 1 or actual != expected:
        raise ValueError('The displayed candidate measure does not belong to the plotted body law.')
    if not np.isfinite(measure['log_total_mass']):
        raise ValueError('A displayed finite candidate measure needs its actual finite log mass.')
    if any(item[1] != snapshot.version or item[2] != snapshot.stamp or
           item[3] != tuple(snapshot.command) for item in actual):
        raise ValueError('Candidate measure and displayed snapshot have different command times.')


def snapshot_record(snapshot):
    verify_measure_identity(snapshot)
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
    if snapshot.surprise_record is not None:
        record['bayes_factor_surprise'] = snapshot.surprise_record
    if snapshot.candidate_measure is not None:
        record['candidate_measure'] = snapshot.candidate_measure
    if snapshot.sensor_event is not None:
        record['sensor_event'] = snapshot.sensor_event
    axes = goal_axis_records(snapshot)
    if axes:
        record['goal_axes'] = axes
        record['goal_poses'] = goal_pose_records(snapshot)
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
    verify_measure_identity(snapshot)
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
    # Append the requested bold target axes, preserving every preexisting
    # particle, thin axis, label, ID, size and ordering byte for byte.
    for record in goal_axis_records(snapshot):
        marker = _marker(snapshot, record['marker_id'], record['namespace'], Marker.ARROW)
        marker.points = [Point(*record['start']), Point(*record['end'])]
        marker.scale.x = record['shaft_diameter_m']
        marker.scale.y = record['head_diameter_m']
        marker.scale.z = record['head_length_m']
        marker.color = ColorRGBA(*record['color_rgb'], record['alpha'])
        # ARROW ignores text for drawing. Keep the complete branch provenance
        # on the actual wire as well as in the snapshot's sidecar record.
        marker.text = json.dumps(record, sort_keys=True, separators=(',', ':'))
        output.append(marker)
    return MarkerArray(output)


def serialized_marker_array(message):
    buffer = io.BytesIO()
    message.serialize(buffer)
    return buffer.getvalue()


def verify_wire_identity(snapshot, actual, *, seed, point_size):
    """Compare the real wire, including goal arrows and their provenance."""
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
