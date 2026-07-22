"""Small, serialization-friendly benchmark models."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class BenchmarkConfig:
    family: str = "DICT_APRILTAG_36h11"
    corner_sigma_px: float = 0.5
    corner_refinement: bool = True
    default_tag_size_m: float = 0.12
    camera_frame_id: str = "camera_optical_frame"
    tag_frame_prefix: str = "apriltag_"
    authority: str = "prob_artag_benchmark"
    min_visible_fraction: float = 0.0
    front_facing_only: bool = True
    association_max_corner_rmse_px: float = 25.0
    gt_near_translation_threshold_m: float = 0.02
    gt_near_rotation_threshold_deg: float = 5.0
    axis_length_ratio: float = 0.5
    estimator_max_iterations: int = 30
    estimator_convergence_tolerance: float = 1e-9
    estimator_min_depth: float = 1e-6
    estimator_dedup_translation_tolerance: float = 1e-7
    estimator_dedup_rotation_tolerance_rad: float = 1e-7
    estimator_finite_difference_step: float = 1e-6
    estimator_verify_jacobian: bool = False

    def __post_init__(self):
        if not self.family:
            raise ValueError("family must not be empty")
        for name in (
            "corner_sigma_px",
            "default_tag_size_m",
            "estimator_convergence_tolerance",
            "estimator_min_depth",
            "estimator_dedup_translation_tolerance",
            "estimator_dedup_rotation_tolerance_rad",
            "estimator_finite_difference_step",
            "axis_length_ratio",
            "gt_near_translation_threshold_m",
            "gt_near_rotation_threshold_deg",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be positive and finite".format(name))
        if (
            not np.isfinite(self.min_visible_fraction)
            or not 0.0 <= self.min_visible_fraction <= 1.0
        ):
            raise ValueError("min_visible_fraction must be in [0,1]")
        if (
            not np.isfinite(self.association_max_corner_rmse_px)
            or self.association_max_corner_rmse_px < 0.0
        ):
            raise ValueError(
                "association_max_corner_rmse_px must be finite and non-negative"
            )
        if self.estimator_max_iterations < 1:
            raise ValueError("estimator_max_iterations must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameInput:
    key: str
    directory: Path
    metadata_path: Path
    image_path: Path
    metadata: Dict[str, Any]
    load_error: str = ""


@dataclass(frozen=True)
class SeedPose:
    rotation: np.ndarray
    translation: np.ndarray
    reported_reprojection_error_px: float

    def __post_init__(self):
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float).reshape(-1)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("seed rotation/translation must have shapes (3,3)/(3,)")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("seed pose must be finite")
        object.__setattr__(self, "rotation", rotation.copy())
        object.__setattr__(self, "translation", translation.copy())
        object.__setattr__(
            self,
            "reported_reprojection_error_px",
            float(self.reported_reprojection_error_px),
        )
