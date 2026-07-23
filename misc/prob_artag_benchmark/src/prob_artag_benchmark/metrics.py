"""Deterministic association and pose-error metrics."""

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def corners_array(value):
    corners = np.asarray(value, dtype=float)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise ValueError("corners must be a finite 4x2 array")
    return corners


def corner_metrics(reference, observed):
    difference = corners_array(observed) - corners_array(reference)
    norms = np.linalg.norm(difference, axis=1)
    return {
        "mean_px": float(np.mean(norms)),
        "rmse_px": float(np.sqrt(np.mean(norms * norms))),
        "max_px": float(np.max(norms)),
        "per_corner_px": [float(value) for value in norms],
    }


def rotation_error_rad(estimate, truth):
    first = np.asarray(estimate, dtype=float).reshape(3, 3)
    second = np.asarray(truth, dtype=float).reshape(3, 3)
    relative = first.T @ second
    cosine = float(np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0))
    return float(math.acos(cosine))


def pose_errors(rotation, translation, truth_transform):
    truth = np.asarray(truth_transform, dtype=float)
    if truth.shape != (4, 4) or not np.all(np.isfinite(truth)):
        raise ValueError("T_C_M must be a finite 4x4 matrix")
    translation_error = float(
        np.linalg.norm(np.asarray(translation, dtype=float).reshape(3) - truth[:3, 3])
    )
    angle = rotation_error_rad(rotation, truth[:3, :3])
    return {
        "translation_m": translation_error,
        "rotation_rad": angle,
        "rotation_deg": float(np.degrees(angle)),
    }


def reprojection_rmse(projected, observed):
    difference = corners_array(projected) - corners_array(observed)
    return float(np.sqrt(np.mean(difference * difference)))


def _marker_id(value):
    if isinstance(value, dict):
        return int(value.get("id", value.get("marker_id")))
    return int(getattr(value, "marker_id"))


def _corners(value):
    return value["corners_px"] if isinstance(value, dict) else value.corners_px


def _family(value):
    if isinstance(value, dict):
        return str(value.get("family", ""))
    return str(getattr(value, "family", ""))


def _minimum_cost_pairs(costs, ground_truth_indices, detection_indices):
    ground_truth_indices = tuple(sorted(ground_truth_indices))
    detection_indices = tuple(sorted(detection_indices))
    if not costs or not ground_truth_indices or not detection_indices:
        return ()
    cost_by_pair = {
        (int(gt_index), int(det_index)): float(cost)
        for cost, gt_index, det_index in costs
    }
    maximum = max(cost_by_pair.values())
    penalty = max(1.0, maximum) * (len(ground_truth_indices) + len(detection_indices) + 1)
    matrix = np.full(
        (len(ground_truth_indices), len(detection_indices)), penalty, dtype=float
    )
    for row, gt_index in enumerate(ground_truth_indices):
        for column, det_index in enumerate(detection_indices):
            pair = (gt_index, det_index)
            if pair in cost_by_pair:
                matrix[row, column] = cost_by_pair[pair] + 1e-12 * (
                    row * (len(detection_indices) + 1) + column
                )
    rows, columns = linear_sum_assignment(matrix)
    selected = []
    for row, column in zip(rows, columns):
        pair = (ground_truth_indices[int(row)], detection_indices[int(column)])
        if pair in cost_by_pair:
            selected.append((cost_by_pair[pair], pair[0], pair[1]))
    return tuple(sorted(selected, key=lambda item: (item[1], item[2])))


def associate_by_id_and_geometry(
    ground_truth: Iterable[dict],
    detections: Iterable[object],
    max_corner_rmse_px: float,
):
    """Match exact IDs first, then classify plausible wrong-ID detections.

    The second pass exists only to measure ID decoding failures.  Distant
    unmatched observations remain false positives/misses instead of being
    forced into a misleading pair.
    """

    truth = tuple(ground_truth)
    observed = tuple(detections)
    available_gt = set(range(len(truth)))
    available_det = set(range(len(observed)))
    pairs: List[Dict[str, object]] = []
    threshold = float(max_corner_rmse_px)

    exact_costs: List[Tuple[float, int, int]] = []
    for gt_index, tag in enumerate(truth):
        for det_index, detection in enumerate(observed):
            if (
                _family(tag) == _family(detection)
                and _marker_id(tag) == _marker_id(detection)
            ):
                cost = corner_metrics(_corners(tag), _corners(detection))["rmse_px"]
                if cost <= threshold:
                    exact_costs.append((cost, gt_index, det_index))
    for cost, gt_index, det_index in _minimum_cost_pairs(
        exact_costs, available_gt, available_det
    ):
        available_gt.remove(gt_index)
        available_det.remove(det_index)
        pairs.append(
            {
                "gt_index": gt_index,
                "detection_index": det_index,
                "id_correct": True,
                "association": "id_match",
                "association_corner_rmse_px": float(cost),
            }
        )

    mismatch_costs: List[Tuple[float, int, int]] = []
    for gt_index in available_gt:
        for det_index in available_det:
            if _family(truth[gt_index]) != _family(observed[det_index]):
                continue
            cost = corner_metrics(_corners(truth[gt_index]), _corners(observed[det_index]))[
                "rmse_px"
            ]
            if cost <= threshold:
                mismatch_costs.append((cost, gt_index, det_index))
    for cost, gt_index, det_index in _minimum_cost_pairs(
        mismatch_costs, available_gt, available_det
    ):
        if cost > threshold:
            continue
        available_gt.remove(gt_index)
        available_det.remove(det_index)
        pairs.append(
            {
                "gt_index": gt_index,
                "detection_index": det_index,
                "id_correct": False,
                "association": "geometric_wrong_id",
                "association_corner_rmse_px": float(cost),
            }
        )

    return {
        "pairs": tuple(
            sorted(
                pairs,
                key=lambda item: (item["gt_index"], item["detection_index"]),
            )
        ),
        "unmatched_ground_truth": tuple(sorted(available_gt)),
        "unmatched_detections": tuple(sorted(available_det)),
    }


def project_pose(object_points, rotation, translation, camera_matrix, distortion):
    import cv2

    rotation_vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=float).reshape(3, 3))
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=float),
        rotation_vector,
        np.asarray(translation, dtype=float).reshape(3, 1),
        np.asarray(camera_matrix, dtype=float).reshape(3, 3),
        np.asarray(distortion, dtype=float).reshape(-1),
    )
    return projected.reshape(-1, 2)


def ippe_object_points(tag_size_m):
    half = 0.5 * float(tag_size_m)
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=float,
    )
