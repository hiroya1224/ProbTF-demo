"""End-to-end offline renderer/detector benchmark orchestration."""

from collections import Counter
import math
from numbers import Integral
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .adapter import DefaultPipelineAdapter
from .io import discover_frames, write_reports
from .metrics import (
    associate_by_id_and_geometry,
    corner_metrics,
    ippe_object_points,
    pose_errors,
    project_pose,
    reprojection_rmse,
    rotation_error_rad,
)
from .models import BenchmarkConfig, SeedPose
from .visualization import draw_overlay


def _tag_id(tag):
    return int(tag.get("id", tag.get("marker_id")))


def _finite_or_none(value):
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _camera_arrays(camera_metadata):
    matrix = camera_metadata.get("camera_matrix")
    if matrix is None:
        matrix = [
            [camera_metadata["fx"], 0.0, camera_metadata["cx"]],
            [0.0, camera_metadata["fy"], camera_metadata["cy"]],
            [0.0, 0.0, 1.0],
        ]
    return (
        np.asarray(matrix, dtype=float).reshape(3, 3),
        np.asarray(camera_metadata.get("distortion", ()), dtype=float).reshape(-1),
    )


def _validate_frame_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("metadata root must be an object")
    camera = metadata.get("camera")
    if not isinstance(camera, dict):
        raise ValueError("metadata.camera must be an object")
    width = int(camera.get("width", 0))
    height = int(camera.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("camera width and height must be positive")
    matrix, distortion = _camera_arrays(camera)
    if not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("camera intrinsics must be finite with positive focal lengths")
    if not np.all(np.isfinite(distortion)):
        raise ValueError("camera distortion must be finite")
    tags = metadata.get("tags")
    if not isinstance(tags, list):
        raise ValueError("metadata.tags must be an array")
    for index, tag in enumerate(tags):
        prefix = "metadata.tags[{}]".format(index)
        if not isinstance(tag, dict):
            raise ValueError("{} must be an object".format(prefix))
        marker_id = _tag_id(tag)
        if marker_id < 0:
            raise ValueError("{}.id must be non-negative".format(prefix))
        family = tag.get("family")
        if not isinstance(family, str) or not family:
            raise ValueError("{}.family must be a non-empty string".format(prefix))
        tag_size = float(tag.get("size_m", float("nan")))
        if not math.isfinite(tag_size) or tag_size <= 0.0:
            raise ValueError("{}.size_m must be positive and finite".format(prefix))
        transform = np.asarray(tag.get("T_C_M"), dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("{}.T_C_M must be a finite 4x4 matrix".format(prefix))
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
            raise ValueError("{}.T_C_M must have homogeneous last row".format(prefix))
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-6
        ):
            raise ValueError("{}.T_C_M rotation must be in SO(3)".format(prefix))
        corners = np.asarray(tag.get("corners_px"), dtype=float)
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            raise ValueError("{}.corners_px must be a finite 4x2 array".format(prefix))
        if not isinstance(tag.get("front_facing"), bool):
            raise ValueError("{}.front_facing must be boolean".format(prefix))
        visible_fraction = float(tag.get("visible_fraction", float("nan")))
        if not math.isfinite(visible_fraction) or not 0.0 <= visible_fraction <= 1.0:
            raise ValueError("{}.visible_fraction must be in [0,1]".format(prefix))
        projected_size = float(tag.get("projected_size_px", float("nan")))
        if not math.isfinite(projected_size) or projected_size <= 0.0:
            raise ValueError("{}.projected_size_px must be positive".format(prefix))


def _eligible_tags(metadata, config):
    tags = []
    for tag in metadata.get("tags", ()):
        if config.front_facing_only and not bool(tag.get("front_facing", True)):
            continue
        if float(tag.get("visible_fraction", 1.0)) < config.min_visible_fraction:
            continue
        corners = np.asarray(tag["corners_px"], dtype=float).reshape(4, 2)
        copied = dict(tag)
        copied["corners_px"] = corners
        tags.append(copied)
    return tuple(
        sorted(
            tags,
            key=lambda tag: (
                _tag_id(tag),
                int(tag.get("instance_id", 0)),
                tuple(np.asarray(tag["corners_px"]).reshape(-1)),
            ),
        )
    )


def _sorted_observations(observations):
    return tuple(
        sorted(
            observations,
            key=lambda item: (
                int(item.marker_id),
                tuple(np.asarray(item.corners_px, dtype=float).reshape(-1)),
            ),
        )
    )


def _diagnostics_by_seed(result):
    output = {}
    diagnostics = getattr(result, "diagnostics", None)
    for item in getattr(diagnostics, "candidates", ()):
        output[int(item.seed_index)] = item
    return output


_SEED_PATTERN = re.compile(r"candidate\s+(\d+)")


def _valid_seed_indices(values, mode_count, seed_count):
    try:
        values = tuple(values)
    except TypeError:
        return None
    if len(values) != mode_count:
        return None
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        return None
    output = [int(value) for value in values]
    if any(value < 0 or value >= seed_count for value in output):
        return None
    if len(set(output)) != len(output):
        return None
    return output


def _component_seed_indices(result, mode_count, seed_count):
    direct = getattr(result, "seed_indices", None)
    if direct is not None:
        validated = _valid_seed_indices(direct, mode_count, seed_count)
        if validated is not None:
            return validated

    output = [None] * mode_count
    record = getattr(result, "record", None)
    distribution = getattr(record, "distribution", None)
    components = tuple(getattr(distribution, "components", ()))
    for mode_index, component in enumerate(components[:mode_count]):
        provenance = getattr(component, "provenance", None)
        detail = str(getattr(provenance, "detail", ""))
        match = _SEED_PATTERN.search(detail)
        if match:
            seed_index = int(match.group(1))
            if 0 <= seed_index < seed_count:
                output[mode_index] = seed_index
    present = [value for value in output if value is not None]
    if len(set(present)) != len(present):
        return [None] * mode_count
    return output


def _complete_mode_seed_mapping(result, seeds):
    rotations = tuple(getattr(result, "rotations", ()))
    translations = tuple(getattr(result, "translations", ()))
    weights = tuple(getattr(result, "weights", ()))
    log_masses = tuple(getattr(result, "log_masses", ()))
    count = len(rotations)
    if not (
        len(translations) == count
        and len(weights) == count
        and len(log_masses) == count
    ):
        raise ValueError("Phase-2 result mode arrays have inconsistent lengths")
    record = getattr(result, "record", None)
    distribution = getattr(record, "distribution", None)
    components = getattr(distribution, "components", None)
    if components is not None and len(tuple(components)) != count:
        raise ValueError("Phase-2 component count does not match returned modes")
    if count > len(seeds):
        raise ValueError("Phase-2 returned more modes than IPPE seeds")
    mapping = _component_seed_indices(result, count, len(seeds))
    used = {index for index in mapping if index is not None and 0 <= index < len(seeds)}
    for mode_index in range(count):
        if mapping[mode_index] is not None and mapping[mode_index] < len(seeds):
            continue
        choices = []
        for seed_index, seed in enumerate(seeds):
            if seed_index in used:
                continue
            translation_distance = float(
                np.linalg.norm(np.asarray(translations[mode_index]) - seed.translation)
            )
            rotation_distance = rotation_error_rad(rotations[mode_index], seed.rotation)
            choices.append((translation_distance + rotation_distance, seed_index))
        if choices:
            selected = min(choices)[1]
            mapping[mode_index] = selected
            used.add(selected)
    if any(index is None for index in mapping) or len(set(mapping)) != len(mapping):
        raise ValueError("Phase-2 modes could not be mapped one-to-one to IPPE seeds")
    return tuple(mapping)


def _pose_metric(rotation, translation, truth_transform):
    if truth_transform is None:
        return None
    try:
        return pose_errors(rotation, translation, truth_transform)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None


def _candidate_evaluation(
    seeds,
    result,
    estimation_error,
    observation,
    camera_matrix,
    distortion,
    tag_size_m,
    truth_tag,
):
    diagnostics = {} if result is None else _diagnostics_by_seed(result)
    mode_by_seed = {}
    if result is not None:
        rotations = tuple(getattr(result, "rotations", ()))
        translations = tuple(getattr(result, "translations", ()))
        weights = tuple(getattr(result, "weights", ()))
        log_masses = tuple(getattr(result, "log_masses", ()))
        mapping = _complete_mode_seed_mapping(result, seeds)
        for mode_index, seed_index in enumerate(mapping):
            if seed_index is None:
                continue
            mode_by_seed[int(seed_index)] = {
                "rotation": np.asarray(rotations[mode_index], dtype=float),
                "translation": np.asarray(translations[mode_index], dtype=float),
                "weight": _finite_or_none(weights[mode_index])
                if mode_index < len(weights)
                else None,
                "log_mass": _finite_or_none(log_masses[mode_index])
                if mode_index < len(log_masses)
                else None,
            }

    truth_transform = None if truth_tag is None else truth_tag.get("T_C_M")
    object_points = ippe_object_points(tag_size_m)
    records = []
    for seed_index, seed in enumerate(seeds):
        diagnostic = diagnostics.get(seed_index)
        refined = mode_by_seed.get(seed_index)
        try:
            initial_projection = project_pose(
                object_points,
                seed.rotation,
                seed.translation,
                camera_matrix,
                distortion,
            )
            initial_reprojection = reprojection_rmse(
                initial_projection, observation.corners_px
            )
        except Exception:
            initial_reprojection = None
        initial_pose_error = _pose_metric(
            seed.rotation, seed.translation, truth_transform
        )
        final_reprojection = None
        final_pose_error = None
        if refined is not None:
            try:
                final_projection = project_pose(
                    object_points,
                    refined["rotation"],
                    refined["translation"],
                    camera_matrix,
                    distortion,
                )
                final_reprojection = reprojection_rmse(
                    final_projection, observation.corners_px
                )
            except Exception:
                final_reprojection = None
            final_pose_error = _pose_metric(
                refined["rotation"], refined["translation"], truth_transform
            )

        selected_rotation = seed.rotation if refined is None else refined["rotation"]
        selected_translation = seed.translation if refined is None else refined["translation"]
        selected_pose_error = initial_pose_error if refined is None else final_pose_error
        reason = (
            str(getattr(diagnostic, "reason", ""))
            if diagnostic is not None
            else ("estimator_error" if estimation_error else "seed_only")
        )
        accepted = refined is not None
        record = {
            "candidate_index": seed_index,
            "accepted": accepted,
            "status": "accepted" if accepted else "rejected",
            "reason": reason,
            "pose_stage": "refined" if accepted else "ippe_seed",
            "reported_initial_reprojection_error_px": _finite_or_none(
                seed.reported_reprojection_error_px
            ),
            "initial_reprojection_rmse_px": initial_reprojection,
            "final_reprojection_rmse_px": final_reprojection,
            "final_objective": _finite_or_none(
                getattr(diagnostic, "final_objective", None)
            ),
            "iterations": int(getattr(diagnostic, "iterations", 0)),
            "weight": None if refined is None else refined["weight"],
            "log_mass": None if refined is None else refined["log_mass"],
            "initial_pose": {
                "rotation": seed.rotation,
                "translation": seed.translation,
            },
            "final_pose": None
            if refined is None
            else {
                "rotation": refined["rotation"],
                "translation": refined["translation"],
            },
            "initial_pose_error": initial_pose_error,
            "final_pose_error": final_pose_error,
            "translation": selected_translation,
            "rotation": selected_rotation,
            "translation_error_m": None
            if selected_pose_error is None
            else selected_pose_error["translation_m"],
            "rotation_error_deg": None
            if selected_pose_error is None
            else selected_pose_error["rotation_deg"],
            "reprojection_rmse_px": final_reprojection
            if accepted
            else initial_reprojection,
        }
        records.append(record)
    return tuple(records)


def _candidate_csv_row(frame_key, frame_id, gt_id, detected_id, candidate):
    translation = np.asarray(candidate["translation"], dtype=float).reshape(3)
    rotation_vector, _ = cv2.Rodrigues(
        np.asarray(candidate["rotation"], dtype=float).reshape(3, 3)
    )
    rotation_vector = rotation_vector.reshape(3)
    initial_error = candidate["initial_pose_error"] or {}
    final_error = candidate["final_pose_error"] or {}
    return {
        "frame_key": frame_key,
        "frame_id": frame_id,
        "gt_id": "" if gt_id is None else gt_id,
        "detected_id": detected_id,
        "candidate_index": candidate["candidate_index"],
        "accepted": candidate["accepted"],
        "status": candidate["status"],
        "reason": candidate["reason"],
        "pose_stage": candidate["pose_stage"],
        "reported_initial_reprojection_error_px": candidate[
            "reported_initial_reprojection_error_px"
        ],
        "initial_reprojection_rmse_px": candidate["initial_reprojection_rmse_px"],
        "final_reprojection_rmse_px": candidate["final_reprojection_rmse_px"],
        "final_objective": candidate["final_objective"],
        "weight": candidate["weight"],
        "log_mass": candidate["log_mass"],
        "translation_x_m": translation[0],
        "translation_y_m": translation[1],
        "translation_z_m": translation[2],
        "rotation_vector_x_rad": rotation_vector[0],
        "rotation_vector_y_rad": rotation_vector[1],
        "rotation_vector_z_rad": rotation_vector[2],
        "translation_error_m": candidate["translation_error_m"],
        "rotation_error_deg": candidate["rotation_error_deg"],
        "initial_translation_error_m": initial_error.get("translation_m"),
        "initial_rotation_error_deg": initial_error.get("rotation_deg"),
        "final_translation_error_m": final_error.get("translation_m"),
        "final_rotation_error_deg": final_error.get("rotation_deg"),
        "iterations": candidate["iterations"],
    }


def _nearest_mode_candidate(candidates, config):
    usable = [
        item
        for item in candidates
        if item["accepted"]
        and item["translation_error_m"] is not None
        and item["rotation_error_deg"] is not None
    ]
    if not usable:
        return None
    return min(
        usable,
        key=lambda item: (
            (
                item["translation_error_m"]
                / config.gt_near_translation_threshold_m
            )
            ** 2
            + (
                item["rotation_error_deg"]
                / config.gt_near_rotation_threshold_deg
            )
            ** 2,
            item["candidate_index"],
        ),
    )


def _ippe_candidate_metrics(candidates, config):
    initial = [
        item
        for item in candidates
        if item.get("initial_pose_error") is not None
    ]
    nearest = None
    if initial:
        nearest = min(
            initial,
            key=lambda item: (
                (
                    item["initial_pose_error"]["translation_m"]
                    / config.gt_near_translation_threshold_m
                )
                ** 2
                + (
                    item["initial_pose_error"]["rotation_deg"]
                    / config.gt_near_rotation_threshold_deg
                )
                ** 2,
                item["candidate_index"],
            ),
        )
    two_candidates = len(candidates) == 2 and len(initial) == 2
    near = None
    if two_candidates:
        near = any(
            item["initial_pose_error"]["translation_m"]
            <= config.gt_near_translation_threshold_m
            and item["initial_pose_error"]["rotation_deg"]
            <= config.gt_near_rotation_threshold_deg
            for item in initial
        )
    reprojection = [
        float(item["initial_reprojection_rmse_px"])
        for item in candidates
        if item.get("initial_reprojection_rmse_px") is not None
    ]
    gap = None
    if two_candidates and len(reprojection) == 2:
        gap = float(abs(reprojection[0] - reprojection[1]))
    return {
        "two_ippe_candidates_available": bool(two_candidates),
        "gt_near_ippe_candidate_exists": near,
        "nearest_ippe_translation_error_m": None
        if nearest is None
        else nearest["initial_pose_error"]["translation_m"],
        "nearest_ippe_rotation_error_deg": None
        if nearest is None
        else nearest["initial_pose_error"]["rotation_deg"],
        "minimum_ippe_translation_error_m": None
        if not initial
        else min(item["initial_pose_error"]["translation_m"] for item in initial),
        "minimum_ippe_rotation_error_deg": None
        if not initial
        else min(item["initial_pose_error"]["rotation_deg"] for item in initial),
        "ippe_reprojection_rmse_gap_px": gap,
    }


def _safe_overlay_name(frame_key):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(frame_key)).strip("._")
    return (value or "frame") + ".png"


def _frame_failure(frame, frame_id, scenario, error, ground_truth=()):
    ground_truth = tuple(ground_truth)
    gt_count = len(ground_truth)
    row = {
        "frame_key": frame.key,
        "frame_id": frame_id,
        "scenario": scenario,
        "status": "error",
        "error": str(error),
        "ground_truth_count": gt_count,
        "detection_count": 0,
        "correct_id_count": 0,
        "wrong_id_count": 0,
        "missed_count": gt_count,
        "false_positive_count": 0,
        "seed_error_count": 0,
        "estimation_error_count": 0,
        "overlay_error_count": 0,
        "recall": 0.0 if gt_count else None,
        "overlay": "",
    }
    report = {
        "frame_key": frame.key,
        "frame_id": frame_id,
        "scenario": scenario,
        "status": "error",
        "error": str(error),
        "ground_truth": [
            {
                "marker_id": _tag_id(tag),
                "association": "frame_error",
                "detected_id": None,
                "id_correct": None,
                "estimation_status": "not_run",
                "estimation_error": str(error),
            }
            for tag in ground_truth
        ],
        "detections": [],
    }
    tag_rows = [
        {
            "frame_key": frame.key,
            "frame_id": frame_id,
            "gt_id": _tag_id(tag),
            "detected_id": "",
            "association": "frame_error",
            "id_correct": "",
            "front_facing": bool(tag.get("front_facing", True)),
            "visible_fraction": float(tag.get("visible_fraction", 1.0)),
            "projected_size_px": _finite_or_none(tag.get("projected_size_px")),
            "estimation_status": "not_run",
            "estimation_error": str(error),
            "candidate_count": 0,
            "accepted_candidate_count": 0,
            "two_ippe_candidates_available": False,
            "gt_near_ippe_candidate_exists": None,
            "nearest_ippe_translation_error_m": None,
            "nearest_ippe_rotation_error_deg": None,
            "minimum_ippe_translation_error_m": None,
            "minimum_ippe_rotation_error_deg": None,
            "ippe_reprojection_rmse_gap_px": None,
        }
        for tag in ground_truth
    ]
    return row, report, tag_rows


def _process_frame(frame, output_dir, config, adapter):
    metadata = frame.metadata
    _validate_frame_metadata(metadata)
    frame_id = metadata.get("frame_id", frame.key)
    scenario = str(metadata.get("scenario", "unknown"))
    ground_truth = _eligible_tags(metadata, config)
    image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
    if image is None:
        logical_image = "frames/{}/{}".format(frame.key, frame.image_path.name)
        row, report, tag_rows = _frame_failure(
            frame,
            frame_id,
            scenario,
            "cannot read {}".format(logical_image),
            ground_truth,
        )
        return row, report, tag_rows, []
    try:
        camera_metadata = metadata["camera"]
        camera_matrix, distortion = _camera_arrays(camera_metadata)
        camera_model = adapter.camera_from_metadata(camera_metadata)
    except Exception as exc:
        row, report, tag_rows = _frame_failure(
            frame, frame_id, scenario, "camera: {}".format(exc), ground_truth
        )
        return row, report, tag_rows, []

    detection_error = None
    try:
        observations = _sorted_observations(adapter.detect(image))
    except Exception as exc:
        observations = ()
        detection_error = "{}: {}".format(type(exc).__name__, exc)

    association = associate_by_id_and_geometry(
        ground_truth, observations, config.association_max_corner_rmse_px
    )
    pair_by_detection = {
        int(item["detection_index"]): item for item in association["pairs"]
    }
    pair_by_truth = {int(item["gt_index"]): item for item in association["pairs"]}
    detection_results = []
    candidate_rows = []
    seeds_by_detection = {}
    sizes_by_detection = {}
    stamp = metadata.get("stamp", frame_id)
    try:
        stamp_value = float(stamp)
    except (TypeError, ValueError):
        stamp_value = 0.0

    for detection_index, observation in enumerate(observations):
        pair = pair_by_detection.get(detection_index)
        truth_tag = None if pair is None else ground_truth[int(pair["gt_index"])]
        tag_size_m = float(
            config.default_tag_size_m
            if truth_tag is None
            else truth_tag.get("size_m", config.default_tag_size_m)
        )
        sizes_by_detection[detection_index] = tag_size_m
        seed_error = None
        try:
            seeds = tuple(adapter.solve_candidates(observation, camera_model, tag_size_m))
        except Exception as exc:
            seeds = ()
            seed_error = "{}: {}".format(type(exc).__name__, exc)
        seeds_by_detection[detection_index] = seeds

        estimation_result = None
        estimation_error = None
        try:
            estimation_result = adapter.estimate(
                observation,
                camera_model,
                tag_size_m,
                config.camera_frame_id,
                "{}{}".format(config.tag_frame_prefix, int(observation.marker_id)),
                stamp_value,
                "{}__{}{}".format(
                    config.camera_frame_id,
                    config.tag_frame_prefix,
                    int(observation.marker_id),
                ),
                config.authority,
                seeds=seeds,
            )
        except Exception as exc:
            estimation_error = "{}: {}".format(type(exc).__name__, exc)

        candidates = _candidate_evaluation(
            seeds,
            estimation_result,
            estimation_error,
            observation,
            camera_matrix,
            distortion,
            tag_size_m,
            truth_tag,
        )
        gt_id = None if truth_tag is None else _tag_id(truth_tag)
        for candidate in candidates:
            candidate_rows.append(
                _candidate_csv_row(
                    frame.key,
                    frame_id,
                    gt_id,
                    int(observation.marker_id),
                    candidate,
                )
            )
        diagnostics = getattr(estimation_result, "diagnostics", None)
        detection_results.append(
            {
                "detection_index": detection_index,
                "marker_id": int(observation.marker_id),
                "family": str(getattr(observation, "family", config.family)),
                "corners_px": np.asarray(observation.corners_px, dtype=float),
                "association": None if pair is None else dict(pair),
                "tag_size_m": tag_size_m,
                "seed_status": "ok" if seed_error is None else "error",
                "seed_error": seed_error,
                "estimation_status": "ok" if estimation_result is not None else "error",
                "estimation_error": estimation_error,
                "seed_count": len(seeds),
                "accepted_count": int(getattr(diagnostics, "accepted_count", 0)),
                "deduplicated_count": int(getattr(diagnostics, "deduplicated_count", 0)),
                "candidates": candidates,
            }
        )

    tag_rows = []
    truth_reports = []
    for gt_index, tag in enumerate(ground_truth):
        pair = pair_by_truth.get(gt_index)
        detection = None if pair is None else observations[int(pair["detection_index"])]
        evaluation = None if pair is None else detection_results[int(pair["detection_index"])]
        metrics = (
            None
            if detection is None
            else corner_metrics(tag["corners_px"], detection.corners_px)
        )
        candidates = () if evaluation is None else evaluation["candidates"]
        nearest = _nearest_mode_candidate(candidates, config)
        ippe_metrics = _ippe_candidate_metrics(candidates, config)
        estimation_status = (
            "not_detected" if evaluation is None else evaluation["estimation_status"]
        )
        estimation_error = None if evaluation is None else evaluation["estimation_error"]
        truth_report = {
            "marker_id": _tag_id(tag),
            "instance_id": int(tag.get("instance_id", 0)),
            "front_facing": bool(tag.get("front_facing", True)),
            "visible_fraction": float(tag.get("visible_fraction", 1.0)),
            "projected_size_px": _finite_or_none(tag.get("projected_size_px")),
            "association": "missed" if pair is None else pair["association"],
            "detected_id": None if detection is None else int(detection.marker_id),
            "id_correct": None if pair is None else bool(pair["id_correct"]),
            "corner_error": metrics,
            "estimation_status": estimation_status,
            "estimation_error": estimation_error,
            "nearest_mode_candidate": nearest,
            "ippe_metrics": ippe_metrics,
        }
        truth_reports.append(truth_report)
        tag_rows.append(
            {
                "frame_key": frame.key,
                "frame_id": frame_id,
                "gt_id": _tag_id(tag),
                "detected_id": "" if detection is None else int(detection.marker_id),
                "association": truth_report["association"],
                "id_correct": "" if pair is None else bool(pair["id_correct"]),
                "front_facing": truth_report["front_facing"],
                "visible_fraction": truth_report["visible_fraction"],
                "projected_size_px": truth_report["projected_size_px"],
                "corner_mean_px": None if metrics is None else metrics["mean_px"],
                "corner_rmse_px": None if metrics is None else metrics["rmse_px"],
                "corner_max_px": None if metrics is None else metrics["max_px"],
                "estimation_status": estimation_status,
                "estimation_error": estimation_error,
                "candidate_count": len(candidates),
                "accepted_candidate_count": sum(bool(item["accepted"]) for item in candidates),
                **ippe_metrics,
                "nearest_mode_translation_error_m": None
                if nearest is None
                else nearest["translation_error_m"],
                "nearest_mode_rotation_error_deg": None
                if nearest is None
                else nearest["rotation_error_deg"],
                "nearest_mode_reprojection_rmse_px": None
                if nearest is None
                else nearest["reprojection_rmse_px"],
            }
        )

    for detection_index in association["unmatched_detections"]:
        evaluation = detection_results[int(detection_index)]
        tag_rows.append(
            {
                "frame_key": frame.key,
                "frame_id": frame_id,
                "gt_id": "",
                "detected_id": evaluation["marker_id"],
                "association": "false_positive",
                "id_correct": False,
                "estimation_status": evaluation["estimation_status"],
                "estimation_error": evaluation["estimation_error"],
                "candidate_count": len(evaluation["candidates"]),
                "accepted_candidate_count": sum(
                    bool(item["accepted"]) for item in evaluation["candidates"]
                ),
                "two_ippe_candidates_available": None,
                "gt_near_ippe_candidate_exists": None,
                "nearest_ippe_translation_error_m": None,
                "nearest_ippe_rotation_error_deg": None,
                "minimum_ippe_translation_error_m": None,
                "minimum_ippe_rotation_error_deg": None,
                "ippe_reprojection_rmse_gap_px": None,
            }
        )

    overlay_name = _safe_overlay_name(frame.key)
    overlay_path = Path(output_dir) / "overlays" / overlay_name
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = draw_overlay(
        image,
        ground_truth,
        observations,
        seeds_by_detection,
        sizes_by_detection,
        camera_matrix,
        distortion,
        config.axis_length_ratio,
    )
    overlay_ok = bool(cv2.imwrite(str(overlay_path), overlay))
    overlay_relative = "overlays/{}".format(overlay_name) if overlay_ok else ""
    correct_count = sum(bool(item["id_correct"]) for item in association["pairs"])
    wrong_count = len(association["pairs"]) - correct_count
    gt_count = len(ground_truth)
    seed_error_count = sum(item["seed_status"] != "ok" for item in detection_results)
    estimation_error_count = sum(
        item["estimation_status"] != "ok" for item in detection_results
    )
    overlay_error_count = 0 if overlay_ok else 1
    if detection_error:
        frame_status = "detection_error"
        frame_error = detection_error
    elif seed_error_count or estimation_error_count:
        frame_status = "pose_error"
        frame_error = "{} seed error(s), {} estimation error(s)".format(
            seed_error_count, estimation_error_count
        )
    elif overlay_error_count:
        frame_status = "overlay_error"
        frame_error = "overlay image could not be written"
    else:
        frame_status = "ok"
        frame_error = ""
    frame_row = {
        "frame_key": frame.key,
        "frame_id": frame_id,
        "scenario": scenario,
        "status": frame_status,
        "error": frame_error,
        "ground_truth_count": gt_count,
        "detection_count": len(observations),
        "correct_id_count": correct_count,
        "wrong_id_count": wrong_count,
        "missed_count": len(association["unmatched_ground_truth"]),
        "false_positive_count": len(association["unmatched_detections"]),
        "seed_error_count": seed_error_count,
        "estimation_error_count": estimation_error_count,
        "overlay_error_count": overlay_error_count,
        "recall": None if gt_count == 0 else float(correct_count) / gt_count,
        "overlay": overlay_relative,
    }
    frame_report = {
        "frame_key": frame.key,
        "frame_id": frame_id,
        "scenario": scenario,
        "status": frame_status,
        "error": frame_error or None,
        "image": "frames/{}/{}".format(frame.key, frame.image_path.name),
        "overlay": overlay_relative,
        "ground_truth": truth_reports,
        "detections": detection_results,
        "unmatched_ground_truth_indices": association["unmatched_ground_truth"],
        "unmatched_detection_indices": association["unmatched_detections"],
    }
    return frame_row, frame_report, tag_rows, candidate_rows


def _ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator) / float(denominator)


def _summary(frame_rows, tag_rows, candidate_rows):
    gt_count = sum(int(row["ground_truth_count"]) for row in frame_rows)
    detections = sum(int(row["detection_count"]) for row in frame_rows)
    correct = sum(int(row["correct_id_count"]) for row in frame_rows)
    wrong = sum(int(row["wrong_id_count"]) for row in frame_rows)
    missed = sum(int(row["missed_count"]) for row in frame_rows)
    false_positive = sum(int(row["false_positive_count"]) for row in frame_rows)
    seed_errors = sum(int(row["seed_error_count"]) for row in frame_rows)
    estimation_errors = sum(int(row["estimation_error_count"]) for row in frame_rows)
    overlay_errors = sum(int(row["overlay_error_count"]) for row in frame_rows)
    corner_values = [
        float(row["corner_rmse_px"])
        for row in tag_rows
        if row.get("corner_rmse_px") is not None
    ]
    nearest_mode_translation_values = [
        float(row["nearest_mode_translation_error_m"])
        for row in tag_rows
        if row.get("nearest_mode_translation_error_m") is not None
    ]
    nearest_mode_rotation_values = [
        float(row["nearest_mode_rotation_error_deg"])
        for row in tag_rows
        if row.get("nearest_mode_rotation_error_deg") is not None
    ]
    nearest_mode_reprojection_values = [
        float(row["nearest_mode_reprojection_rmse_px"])
        for row in tag_rows
        if row.get("nearest_mode_reprojection_rmse_px") is not None
    ]
    nearest_ippe_translation_values = [
        float(row["nearest_ippe_translation_error_m"])
        for row in tag_rows
        if row.get("nearest_ippe_translation_error_m") is not None
    ]
    nearest_ippe_rotation_values = [
        float(row["nearest_ippe_rotation_error_deg"])
        for row in tag_rows
        if row.get("nearest_ippe_rotation_error_deg") is not None
    ]
    minimum_ippe_translation_values = [
        float(row["minimum_ippe_translation_error_m"])
        for row in tag_rows
        if row.get("minimum_ippe_translation_error_m") is not None
    ]
    minimum_ippe_rotation_values = [
        float(row["minimum_ippe_rotation_error_deg"])
        for row in tag_rows
        if row.get("minimum_ippe_rotation_error_deg") is not None
    ]
    ippe_reprojection_gap_values = [
        float(row["ippe_reprojection_rmse_gap_px"])
        for row in tag_rows
        if row.get("ippe_reprojection_rmse_gap_px") is not None
    ]
    two_ippe_count = sum(
        row.get("two_ippe_candidates_available") is True for row in tag_rows
    )
    near_ippe_count = sum(
        row.get("gt_near_ippe_candidate_exists") is True for row in tag_rows
    )
    status_counts = Counter(str(row.get("reason", "")) for row in candidate_rows)
    return {
        "frame_count": len(frame_rows),
        "successful_frame_count": sum(row["status"] == "ok" for row in frame_rows),
        "error_frame_count": sum(row["status"] != "ok" for row in frame_rows),
        "ground_truth_count": gt_count,
        "detection_count": detections,
        "correct_id_count": correct,
        "wrong_id_count": wrong,
        "missed_count": missed,
        "false_positive_count": false_positive,
        "seed_error_count": seed_errors,
        "estimation_error_count": estimation_errors,
        "overlay_error_count": overlay_errors,
        "recall": _ratio(correct, gt_count),
        "geometric_recall": _ratio(correct + wrong, gt_count),
        "id_accuracy_on_associated": _ratio(correct, correct + wrong),
        "precision": _ratio(correct, detections),
        "two_ippe_candidate_tag_count": two_ippe_count,
        "two_ippe_candidate_coverage": _ratio(two_ippe_count, gt_count),
        "gt_near_ippe_candidate_count": near_ippe_count,
        "gt_near_ippe_candidate_rate": _ratio(near_ippe_count, two_ippe_count),
        "gt_near_ippe_candidate_coverage": _ratio(near_ippe_count, gt_count),
        "corner_rmse_px_mean": None if not corner_values else float(np.mean(corner_values)),
        "corner_rmse_px_max": None if not corner_values else float(np.max(corner_values)),
        "nearest_mode_translation_error_m_mean": None
        if not nearest_mode_translation_values
        else float(np.mean(nearest_mode_translation_values)),
        "nearest_mode_translation_error_m_max": None
        if not nearest_mode_translation_values
        else float(np.max(nearest_mode_translation_values)),
        "nearest_mode_rotation_error_deg_mean": None
        if not nearest_mode_rotation_values
        else float(np.mean(nearest_mode_rotation_values)),
        "nearest_mode_rotation_error_deg_max": None
        if not nearest_mode_rotation_values
        else float(np.max(nearest_mode_rotation_values)),
        "nearest_mode_reprojection_rmse_px_mean": None
        if not nearest_mode_reprojection_values
        else float(np.mean(nearest_mode_reprojection_values)),
        "nearest_mode_reprojection_rmse_px_max": None
        if not nearest_mode_reprojection_values
        else float(np.max(nearest_mode_reprojection_values)),
        "nearest_ippe_translation_error_m_mean": None
        if not nearest_ippe_translation_values
        else float(np.mean(nearest_ippe_translation_values)),
        "nearest_ippe_translation_error_m_max": None
        if not nearest_ippe_translation_values
        else float(np.max(nearest_ippe_translation_values)),
        "nearest_ippe_rotation_error_deg_mean": None
        if not nearest_ippe_rotation_values
        else float(np.mean(nearest_ippe_rotation_values)),
        "nearest_ippe_rotation_error_deg_max": None
        if not nearest_ippe_rotation_values
        else float(np.max(nearest_ippe_rotation_values)),
        "minimum_ippe_translation_error_m_mean": None
        if not minimum_ippe_translation_values
        else float(np.mean(minimum_ippe_translation_values)),
        "minimum_ippe_translation_error_m_max": None
        if not minimum_ippe_translation_values
        else float(np.max(minimum_ippe_translation_values)),
        "minimum_ippe_rotation_error_deg_mean": None
        if not minimum_ippe_rotation_values
        else float(np.mean(minimum_ippe_rotation_values)),
        "minimum_ippe_rotation_error_deg_max": None
        if not minimum_ippe_rotation_values
        else float(np.max(minimum_ippe_rotation_values)),
        "ippe_reprojection_rmse_gap_px_mean": None
        if not ippe_reprojection_gap_values
        else float(np.mean(ippe_reprojection_gap_values)),
        "ippe_reprojection_rmse_gap_px_max": None
        if not ippe_reprojection_gap_values
        else float(np.max(ippe_reprojection_gap_values)),
        "candidate_status_counts": dict(sorted(status_counts.items())),
    }


def evaluate_dataset(
    dataset_root,
    output_dir,
    config: Optional[BenchmarkConfig] = None,
    adapter=None,
):
    """Evaluate every Phase-1 frame and write deterministic JSON/CSV/PNG output."""

    config = BenchmarkConfig() if config is None else config
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be BenchmarkConfig")
    adapter = DefaultPipelineAdapter(config) if adapter is None else adapter
    frames = discover_frames(dataset_root)
    overlay_root = Path(output_dir) / "overlays"
    if overlay_root.is_dir():
        for stale_overlay in sorted(overlay_root.glob("*.png")):
            stale_overlay.unlink()
    frame_rows = []
    tag_rows = []
    candidate_rows = []
    frame_reports = []
    for frame in frames:
        if frame.load_error:
            frame_id = frame.key
            scenario = "unknown"
            frame_row, frame_report, frame_tags = _frame_failure(
                frame,
                frame_id,
                scenario,
                frame.load_error,
            )
            frame_rows.append(frame_row)
            frame_reports.append(frame_report)
            tag_rows.extend(frame_tags)
            continue
        try:
            frame_row, frame_report, frame_tags, frame_candidates = _process_frame(
                frame, output_dir, config, adapter
            )
        except Exception as exc:
            try:
                tags = _eligible_tags(frame.metadata, config)
            except Exception:
                tags = ()
            frame_id = frame.metadata.get("frame_id", frame.key)
            scenario = str(frame.metadata.get("scenario", "unknown"))
            frame_row, frame_report, frame_tags = _frame_failure(
                frame,
                frame_id,
                scenario,
                "{}: {}".format(type(exc).__name__, exc),
                tags,
            )
            frame_candidates = []
        frame_rows.append(frame_row)
        frame_reports.append(frame_report)
        tag_rows.extend(frame_tags)
        candidate_rows.extend(frame_candidates)
    report = {
        "schema_version": 1,
        "benchmark": "prob_artag_benchmark",
        "config": config.to_dict(),
        "api_notes": tuple(getattr(adapter, "api_notes", ())),
        "summary": _summary(frame_rows, tag_rows, candidate_rows),
        "frames": frame_reports,
    }
    write_reports(output_dir, report, frame_rows, tag_rows, candidate_rows)
    return report
