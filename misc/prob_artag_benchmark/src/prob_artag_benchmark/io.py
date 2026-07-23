"""Phase-1 frame discovery and deterministic report serialization."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from .models import FrameInput


FRAME_FIELDS = (
    "frame_key",
    "frame_id",
    "scenario",
    "status",
    "error",
    "ground_truth_count",
    "detection_count",
    "correct_id_count",
    "wrong_id_count",
    "missed_count",
    "false_positive_count",
    "seed_error_count",
    "estimation_error_count",
    "overlay_error_count",
    "recall",
    "overlay",
)

TAG_FIELDS = (
    "frame_key",
    "frame_id",
    "gt_id",
    "detected_id",
    "association",
    "id_correct",
    "front_facing",
    "visible_fraction",
    "projected_size_px",
    "corner_mean_px",
    "corner_rmse_px",
    "corner_max_px",
    "estimation_status",
    "estimation_error",
    "candidate_count",
    "accepted_candidate_count",
    "two_ippe_candidates_available",
    "gt_near_ippe_candidate_exists",
    "nearest_ippe_translation_error_m",
    "nearest_ippe_rotation_error_deg",
    "minimum_ippe_translation_error_m",
    "minimum_ippe_rotation_error_deg",
    "ippe_reprojection_rmse_gap_px",
    "nearest_mode_translation_error_m",
    "nearest_mode_rotation_error_deg",
    "nearest_mode_reprojection_rmse_px",
)

CANDIDATE_FIELDS = (
    "frame_key",
    "frame_id",
    "gt_id",
    "detected_id",
    "candidate_index",
    "accepted",
    "status",
    "reason",
    "pose_stage",
    "reported_initial_reprojection_error_px",
    "initial_reprojection_rmse_px",
    "final_reprojection_rmse_px",
    "final_objective",
    "weight",
    "log_mass",
    "translation_x_m",
    "translation_y_m",
    "translation_z_m",
    "rotation_vector_x_rad",
    "rotation_vector_y_rad",
    "rotation_vector_z_rad",
    "translation_error_m",
    "rotation_error_deg",
    "initial_translation_error_m",
    "initial_rotation_error_deg",
    "final_translation_error_m",
    "final_rotation_error_deg",
    "iterations",
)


def discover_frames(dataset_root):
    root = Path(dataset_root)
    if not root.exists():
        raise ValueError("dataset does not exist: {}".format(root))
    frames_root = root / "frames" if (root / "frames").is_dir() else root
    if (frames_root / "metadata.json").is_file():
        frame_directories = [frames_root]
    else:
        frame_directories = sorted(
            (path for path in frames_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
    if not frame_directories:
        raise ValueError("dataset contains no frame metadata: {}".format(root))
    frames = []
    for directory in frame_directories:
        metadata_path = directory / "metadata.json"
        load_error = ""
        if not metadata_path.is_file():
            metadata = {}
            load_error = "FileNotFoundError: missing frames/{}/metadata.json".format(
                directory.name
            )
        else:
            try:
                with metadata_path.open("r", encoding="utf-8") as stream:
                    metadata = json.load(stream)
                if not isinstance(metadata, dict):
                    raise ValueError("metadata root must be a JSON object")
            except (OSError, TypeError, ValueError) as exc:
                metadata = {}
                load_error = "{}: {}".format(type(exc).__name__, exc)
        image_name = str(metadata.get("rgb_file", "rgb.png"))
        frames.append(
            FrameInput(
                key=directory.name,
                directory=directory,
                metadata_path=metadata_path,
                image_path=directory / image_name,
                metadata=metadata,
                load_error=load_error,
            )
        )
    return tuple(frames)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            _jsonable(value),
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def write_csv(path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def write_reports(output_dir, report, frame_rows, tag_rows, candidate_rows):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "metrics.json", report)
    write_csv(root / "frames.csv", frame_rows, FRAME_FIELDS)
    write_csv(root / "tags.csv", tag_rows, TAG_FIELDS)
    write_csv(root / "candidates.csv", candidate_rows, CANDIDATE_FIELDS)
