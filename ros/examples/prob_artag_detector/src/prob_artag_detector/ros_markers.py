"""RViz markers for every retained planar pose-mixture component."""

import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from probtf.geometry import rotmat_to_quat


_AXIS_COLORS = (
    ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0),
    ColorRGBA(r=0.1, g=1.0, b=0.1, a=1.0),
    ColorRGBA(r=0.1, g=0.35, b=1.0, a=1.0),
)


def _point(values):
    value = np.asarray(values, dtype=float).reshape(3)
    return Point(x=float(value[0]), y=float(value[1]), z=float(value[2]))


def _assign_pose(marker, translation, rotation):
    translation = np.asarray(translation, dtype=float).reshape(3)
    quaternion = rotmat_to_quat(np.asarray(rotation, dtype=float).reshape(3, 3))
    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
        float(value) for value in translation
    )
    marker.pose.orientation.w = float(quaternion[0])
    marker.pose.orientation.x = float(quaternion[1])
    marker.pose.orientation.y = float(quaternion[2])
    marker.pose.orientation.z = float(quaternion[3])


def _weighted_color(weight, primary):
    alpha = max(0.25, min(0.9, 0.25 + 0.65 * float(weight)))
    if primary:
        return ColorRGBA(r=0.08, g=0.85, b=0.42, a=alpha)
    return ColorRGBA(r=0.95, g=0.48, b=0.12, a=alpha)


def _base_marker(header, namespace, marker_id, marker_type, lifetime):
    marker = Marker()
    marker.header = header
    marker.ns = namespace
    marker.id = int(marker_id)
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    if lifetime is not None:
        marker.lifetime = lifetime
    return marker


def _covariance_pose(covariance, maximum_scale):
    covariance = np.asarray(covariance, dtype=float).reshape(3, 3)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues[0]) <= 0.0:
        raise ValueError("marker translation covariance must be positive definite.")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if np.linalg.det(eigenvectors) < 0.0:
        eigenvectors[:, -1] *= -1.0
    raw_scale = 4.0 * np.sqrt(eigenvalues)  # full diameter of a 2-sigma ellipsoid
    clipped = bool(np.any(raw_scale > float(maximum_scale)))
    scale = np.clip(raw_scale, 0.002, float(maximum_scale))
    return eigenvectors, scale, clipped


def build_pose_mixture_markers(
    results,
    header,
    tag_size_m,
    axis_length_m=None,
    tag_thickness_m=0.003,
    maximum_uncertainty_scale_m=0.75,
    lifetime=None,
    include_axes=True,
):
    """Build a clear-first MarkerArray retaining all accepted IPPE modes."""

    tag_size_m = float(tag_size_m)
    axis_length_m = (
        0.6 * tag_size_m if axis_length_m is None else float(axis_length_m)
    )
    tag_thickness_m = float(tag_thickness_m)
    maximum_uncertainty_scale_m = float(maximum_uncertainty_scale_m)
    if (
        not np.isfinite(tag_size_m)
        or not np.isfinite(axis_length_m)
        or not np.isfinite(tag_thickness_m)
        or not np.isfinite(maximum_uncertainty_scale_m)
        or min(
            tag_size_m,
            axis_length_m,
            tag_thickness_m,
            maximum_uncertainty_scale_m,
        )
        <= 0.0
    ):
        raise ValueError("marker dimensions must be positive and finite.")

    output = MarkerArray()
    clear = Marker()
    clear.header = header
    clear.action = Marker.DELETEALL
    output.markers.append(clear)
    marker_id = 0
    for result in results:
        if result is None:
            continue
        weights = np.asarray(result.weights, dtype=float)
        if weights.ndim != 1 or weights.size == 0 or not np.all(np.isfinite(weights)):
            raise ValueError("pose-mixture weights must be a non-empty finite vector.")
        primary_index = int(np.argmax(weights))
        child_frame = str(result.record.child_frame_id).strip().strip("/") or "apriltag"
        components = tuple(result.record.distribution.components)
        if not (
            len(result.rotations)
            == len(result.translations)
            == len(components)
            == len(weights)
        ):
            raise ValueError("pose-mixture result arrays must have matching lengths.")
        for mode_index, (rotation, translation, component, weight) in enumerate(
            zip(result.rotations, result.translations, components, weights)
        ):
            namespace = "{}/mode_{}".format(child_frame, mode_index)
            primary = mode_index == primary_index
            color = _weighted_color(weight, primary)

            plane = _base_marker(
                header, namespace + "/tag", marker_id, Marker.CUBE, lifetime
            )
            marker_id += 1
            _assign_pose(plane, translation, rotation)
            plane.scale.x = tag_size_m
            plane.scale.y = tag_size_m
            plane.scale.z = tag_thickness_m
            plane.color = color
            output.markers.append(plane)

            origin = np.asarray(translation, dtype=float).reshape(3)
            rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
            if include_axes:
                axes = _base_marker(
                    header,
                    namespace + "/axes",
                    marker_id,
                    Marker.LINE_LIST,
                    lifetime,
                )
                marker_id += 1
                axes.scale.x = max(0.0015, 0.025 * tag_size_m)
                for axis_index, axis_color in enumerate(_AXIS_COLORS):
                    endpoint = origin + axis_length_m * rotation[:, axis_index]
                    axes.points.extend((_point(origin), _point(endpoint)))
                    axes.colors.extend((axis_color, axis_color))
                output.markers.append(axes)

            covariance_rotation, covariance_scale, covariance_clipped = _covariance_pose(
                component.translation.residual_covariance,
                maximum_uncertainty_scale_m,
            )
            ellipsoid = _base_marker(
                header,
                namespace + "/conditional_translation_2sigma_at_mode",
                marker_id,
                Marker.SPHERE,
                lifetime,
            )
            marker_id += 1
            _assign_pose(ellipsoid, translation, covariance_rotation)
            ellipsoid.scale.x, ellipsoid.scale.y, ellipsoid.scale.z = (
                float(value) for value in covariance_scale
            )
            ellipsoid.color = (
                ColorRGBA(r=1.0, g=0.05, b=0.5, a=0.42)
                if covariance_clipped
                else ColorRGBA(r=color.r, g=color.g, b=color.b, a=min(0.28, color.a))
            )
            output.markers.append(ellipsoid)

            label = _base_marker(
                header, namespace + "/label", marker_id, Marker.TEXT_VIEW_FACING, lifetime
            )
            marker_id += 1
            label_position = origin + rotation @ np.array([0.0, 0.7 * tag_size_m, 0.0])
            label.pose.position = _point(label_position)
            label.scale.z = max(0.018, 0.22 * tag_size_m)
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
            label.text = "{}  mode={}  w={:.3f}{}{}".format(
                child_frame,
                mode_index,
                float(weight),
                "  [TF]" if primary else "",
                "  [conditional 2sigma clipped]" if covariance_clipped else "",
            )
            output.markers.append(label)
    return output
