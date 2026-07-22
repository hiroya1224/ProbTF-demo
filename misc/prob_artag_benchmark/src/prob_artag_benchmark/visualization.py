"""OpenCV-only overlays for ground truth, observations, and IPPE axes."""

from typing import Dict, Iterable

import cv2
import numpy as np

from .metrics import ippe_object_points, project_pose


GT_COLOR = (210, 170, 0)
DETECTION_COLOR = (220, 0, 220)
AXIS_COLORS = ((30, 30, 230), (30, 190, 30), (230, 90, 20))
CANDIDATE_COLORS = ((20, 20, 20), (0, 120, 220), (170, 40, 120), (130, 100, 0))


def _polyline(image, corners, color, label, label_offset_y):
    points = np.rint(np.asarray(corners, dtype=float)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
    for index, point in enumerate(points.reshape(-1, 2)):
        cv2.circle(image, tuple(point), 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            str(index),
            (int(point[0]) + 4, int(point[1]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    origin = tuple(points.reshape(-1, 2)[0])
    cv2.putText(
        image,
        label,
        (int(origin[0]), max(12, int(origin[1]) + int(label_offset_y))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_axes(
    image,
    seed,
    camera_matrix,
    distortion,
    tag_size_m,
    axis_length_ratio,
    candidate_index,
    label,
):
    length = float(axis_length_ratio) * float(tag_size_m)
    axes = np.array(
        [[0.0, 0.0, 0.0], [length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
        dtype=float,
    )
    try:
        pixels = project_pose(
            axes,
            seed.rotation,
            seed.translation,
            camera_matrix,
            distortion,
        )
    except Exception:
        return
    if not np.all(np.isfinite(pixels)):
        return
    points = np.rint(pixels).astype(int)
    origin = tuple(points[0])
    for endpoint, color in zip(points[1:], AXIS_COLORS):
        cv2.line(image, origin, tuple(endpoint), color, 2, cv2.LINE_AA)
    candidate_color = CANDIDATE_COLORS[candidate_index % len(CANDIDATE_COLORS)]
    cv2.circle(image, origin, 5 + candidate_index % 3, candidate_color, 1, cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (origin[0] + 5, origin[1] + 15 + 14 * candidate_index),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        candidate_color,
        1,
        cv2.LINE_AA,
    )


def draw_overlay(
    image,
    ground_truth: Iterable[dict],
    detections: Iterable[object],
    seeds_by_detection: Dict[int, tuple],
    tag_sizes_by_detection: Dict[int, float],
    camera_matrix,
    distortion,
    axis_length_ratio=0.5,
):
    output = np.asarray(image).copy()
    for tag in ground_truth:
        marker_id = int(tag.get("id", tag.get("marker_id")))
        _polyline(
            output, tag["corners_px"], GT_COLOR, "GT:{}".format(marker_id), -10
        )
    for detection_index, detection in enumerate(detections):
        marker_id = int(detection.marker_id)
        _polyline(
            output,
            detection.corners_px,
            DETECTION_COLOR,
            "DET:{}".format(marker_id),
            20,
        )
        for candidate_index, seed in enumerate(seeds_by_detection.get(detection_index, ())):
            _draw_axes(
                output,
                seed,
                camera_matrix,
                distortion,
                tag_sizes_by_detection[detection_index],
                axis_length_ratio,
                candidate_index,
                "{}:{}".format(marker_id, candidate_index),
            )
    cv2.putText(
        output, "GT", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GT_COLOR, 2, cv2.LINE_AA
    )
    cv2.putText(
        output,
        "DETECTED",
        (62, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        DETECTION_COLOR,
        2,
        cv2.LINE_AA,
    )
    return output
