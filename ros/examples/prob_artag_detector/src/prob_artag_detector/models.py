"""ROS-independent inputs and diagnostics for probabilistic AprilTag poses."""

from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Tuple

import numpy as np


def _finite_array(values, shape, name):
    value = np.asarray(values, dtype=float)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError("{} must be a finite array with shape {}.".format(name, shape))
    output = value.copy()
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class CameraModel:
    """Calibrated OpenCV optical camera model."""

    camera_matrix: np.ndarray
    distortion: np.ndarray
    width: int = 0
    height: int = 0

    def __post_init__(self):
        matrix = _finite_array(self.camera_matrix, (3, 3), "camera_matrix")
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise ValueError("camera focal lengths must be positive.")
        if not np.allclose(matrix[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-12):
            raise ValueError("camera_matrix must have the calibrated pinhole last row [0,0,1].")
        distortion = np.asarray(self.distortion, dtype=float).reshape(-1)
        if distortion.size not in (0, 4, 5, 8, 12, 14) or not np.all(np.isfinite(distortion)):
            raise ValueError("distortion must contain 0, 4, 5, 8, 12, or 14 finite coefficients.")
        distortion = distortion.copy()
        distortion.setflags(write=False)
        width = int(self.width)
        height = int(self.height)
        if width < 0 or height < 0:
            raise ValueError("camera image dimensions must be non-negative.")
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "distortion", distortion)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    @property
    def has_distortion(self):
        return bool(self.distortion.size and np.any(np.abs(self.distortion) > 1e-15))

    @classmethod
    def from_camera_info(cls, message):
        distortion_model = str(getattr(message, "distortion_model", "")).strip()
        if distortion_model not in ("", "plumb_bob", "rational_polynomial"):
            raise ValueError(
                "Unsupported CameraInfo distortion_model {!r}.".format(
                    distortion_model
                )
            )
        width = int(message.width)
        height = int(message.height)
        if width <= 0 or height <= 0:
            raise ValueError("CameraInfo width and height must be positive.")
        binning_x = int(getattr(message, "binning_x", 0))
        binning_y = int(getattr(message, "binning_y", 0))
        if binning_x not in (0, 1) or binning_y not in (0, 1):
            raise ValueError(
                "Binned CameraInfo is unsupported; publish matching unbinned image_raw."
            )
        roi = getattr(message, "roi", None)
        if roi is not None and (
            int(getattr(roi, "x_offset", 0)) != 0
            or int(getattr(roi, "y_offset", 0)) != 0
            or int(getattr(roi, "width", 0)) != 0
            or int(getattr(roi, "height", 0)) != 0
            or bool(getattr(roi, "do_rectify", False))
        ):
            raise ValueError(
                "ROI CameraInfo is unsupported; publish the full matching image_raw."
            )
        return cls(
            np.asarray(message.K, dtype=float).reshape(3, 3),
            np.asarray(message.D, dtype=float),
            width,
            height,
        )


@dataclass(frozen=True)
class MarkerObservation:
    """One decoded marker with OpenCV's canonical clockwise corner order."""

    marker_id: int
    corners_px: np.ndarray
    image_covariance: np.ndarray
    family: str = "DICT_APRILTAG_36h11"

    def __post_init__(self):
        marker_id = int(self.marker_id)
        if marker_id < 0:
            raise ValueError("marker_id must be non-negative.")
        corners = _finite_array(self.corners_px, (4, 2), "corners_px")
        covariance = _finite_array(self.image_covariance, (8, 8), "image_covariance")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-10):
            raise ValueError("image_covariance must be symmetric.")
        object.__setattr__(self, "marker_id", marker_id)
        object.__setattr__(self, "corners_px", corners)
        object.__setattr__(self, "image_covariance", covariance)
        object.__setattr__(self, "family", str(self.family))


@dataclass(frozen=True)
class CandidateDiagnostic:
    seed_index: int
    accepted: bool
    reason: str
    initial_error: float
    final_objective: float
    iterations: int


@dataclass(frozen=True)
class EstimationDiagnostics:
    seed_count: int
    accepted_count: int
    deduplicated_count: int
    rejected_cheirality: int
    rejected_refinement: int
    rejected_spd: int
    candidates: Tuple[CandidateDiagnostic, ...]


@dataclass(frozen=True)
class PoseMixtureResult:
    record: object
    diagnostics: EstimationDiagnostics
    rotations: Tuple[np.ndarray, ...]
    translations: Tuple[np.ndarray, ...]
    log_masses: Tuple[float, ...]
    weights: Tuple[float, ...]
    seed_indices: Tuple[int, ...] = ()

    def __post_init__(self):
        rotations = tuple(self.rotations)
        translations = tuple(self.translations)
        log_masses = tuple(float(value) for value in self.log_masses)
        weights = tuple(float(value) for value in self.weights)
        mode_count = len(rotations)
        if not (
            len(translations) == mode_count
            and len(log_masses) == mode_count
            and len(weights) == mode_count
        ):
            raise ValueError("all PoseMixtureResult mode arrays must have equal length.")
        if not np.all(np.isfinite(log_masses)) or not np.all(np.isfinite(weights)):
            raise ValueError("PoseMixtureResult masses and weights must be finite.")

        distribution = getattr(self.record, "distribution", None)
        components = getattr(distribution, "components", None)
        if components is not None and len(tuple(components)) != mode_count:
            raise ValueError("record component count must match returned mode count.")

        raw_seed_indices = tuple(self.seed_indices)
        if raw_seed_indices:
            if len(raw_seed_indices) != mode_count:
                raise ValueError("seed_indices must match returned mode count.")
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in raw_seed_indices
            ):
                raise TypeError("seed_indices must contain integers.")
            seed_indices = tuple(int(value) for value in raw_seed_indices)
            if len(set(seed_indices)) != len(seed_indices):
                raise ValueError("seed_indices must be unique.")
            seed_count = getattr(self.diagnostics, "seed_count", None)
            if seed_count is not None and any(
                value < 0 or value >= int(seed_count) for value in seed_indices
            ):
                raise ValueError("seed_indices must refer to diagnostics seeds.")
        else:
            seed_indices = ()

        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "translations", translations)
        object.__setattr__(self, "log_masses", log_masses)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "seed_indices", seed_indices)
