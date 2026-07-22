"""Camera calibration loading and explicit uncalibrated-camera fallback."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import yaml

from prob_artag_detector.models import CameraModel


SUPPORTED_DISTORTION_MODELS = ("", "plumb_bob", "rational_polynomial")


@dataclass(frozen=True)
class CameraCalibration:
    """A validated camera model plus ROS calibration metadata."""

    model: CameraModel
    distortion_model: str = "plumb_bob"
    camera_name: str = "camera"

    def __post_init__(self):
        if not isinstance(self.model, CameraModel):
            raise TypeError("model must be CameraModel.")
        distortion_model = str(self.distortion_model).strip()
        if distortion_model not in SUPPORTED_DISTORTION_MODELS:
            raise ValueError(
                "Unsupported distortion model {!r}; expected plumb_bob or "
                "rational_polynomial.".format(distortion_model)
            )
        object.__setattr__(self, "distortion_model", distortion_model or "plumb_bob")
        object.__setattr__(self, "camera_name", str(self.camera_name).strip() or "camera")


def approximate_camera_model(
    width,
    height,
    horizontal_fov_deg=60.0,
    fx_px=None,
    fy_px=None,
    cx_px=None,
    cy_px=None,
):
    """Construct a zero-distortion pinhole fallback for an image size.

    The fallback is deliberately explicit and approximate.  It is useful for a
    live demo when no calibration YAML or valid ``CameraInfo`` is available;
    metric depth should not be treated as calibrated until a real model is used.
    """

    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("fallback image width and height must be positive.")
    horizontal_fov_deg = float(horizontal_fov_deg)
    if not math.isfinite(horizontal_fov_deg) or not 1.0 <= horizontal_fov_deg < 179.0:
        raise ValueError("horizontal_fov_deg must be finite and in [1, 179).")

    inferred_focal = 0.5 * float(width) / math.tan(
        0.5 * math.radians(horizontal_fov_deg)
    )

    def positive_or_default(value, default, name):
        if value is None:
            return float(default)
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("{} must be positive and finite when supplied.".format(name))
        return value

    def finite_or_default(value, default, name):
        if value is None:
            return float(default)
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("{} must be finite when supplied.".format(name))
        return value

    fx_value = positive_or_default(fx_px, inferred_focal, "fx_px")
    fy_value = positive_or_default(fy_px, fx_value, "fy_px")
    cx_value = finite_or_default(cx_px, 0.5 * (width - 1.0), "cx_px")
    cy_value = finite_or_default(cy_px, 0.5 * (height - 1.0), "cy_px")
    return CameraModel(
        np.array(
            [
                [fx_value, 0.0, cx_value],
                [0.0, fy_value, cy_value],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        np.zeros(5, dtype=float),
        width,
        height,
    )


def _data_vector(mapping, key, required=True):
    if key not in mapping:
        if required:
            raise ValueError("Calibration YAML is missing {!r}.".format(key))
        return np.zeros(0, dtype=float)
    value = mapping[key]
    if isinstance(value, dict):
        value = value.get("data")
    if value is None:
        raise ValueError("Calibration YAML field {!r} has no data.".format(key))
    vector = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("Calibration YAML field {!r} must be finite.".format(key))
    return vector


def load_camera_calibration(path):
    """Load the standard ROS camera-calibration YAML representation."""

    calibration_path = Path(path).expanduser()
    with calibration_path.open("r", encoding="utf-8") as stream:
        try:
            mapping = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            raise ValueError("Calibration YAML could not be parsed: {}.".format(error)) from error
    if not isinstance(mapping, dict):
        raise ValueError("Calibration YAML root must be a mapping.")
    width = int(mapping.get("image_width", 0))
    height = int(mapping.get("image_height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Calibration YAML requires positive image_width and image_height.")
    camera_matrix = _data_vector(mapping, "camera_matrix")
    if camera_matrix.size != 9:
        raise ValueError("camera_matrix must contain exactly 9 values.")
    distortion = _data_vector(mapping, "distortion_coefficients", required=False)
    distortion_model = str(mapping.get("distortion_model", "plumb_bob")).strip()
    return CameraCalibration(
        CameraModel(camera_matrix.reshape(3, 3), distortion, width, height),
        distortion_model,
        mapping.get("camera_name", "camera"),
    )
