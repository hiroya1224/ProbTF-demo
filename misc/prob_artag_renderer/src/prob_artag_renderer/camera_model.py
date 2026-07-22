"""Calibrated OpenCV optical camera model."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

import numpy as np


@dataclass
class CameraModel:
    width: int = 640
    height: int = 480
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0
    distortion: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)
        self.fx = float(self.fx)
        self.fy = float(self.fy)
        self.cx = float(self.cx)
        self.cy = float(self.cy)
        self.distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        if self.width <= 0 or self.height <= 0 or self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera dimensions and focal lengths must be positive")
        if self.distortion.size not in (0, 4, 5, 8, 12, 14):
            raise ValueError("distortion must follow an OpenCV coefficient layout")

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def is_distorted(self) -> bool:
        return bool(self.distortion.size and np.any(self.distortion != 0.0))

    def project(self, points_C: np.ndarray) -> np.ndarray:
        points = np.asarray(points_C, dtype=np.float64).reshape(-1, 3)
        if np.any(points[:, 2] <= 0.0):
            raise ValueError("all projected points must have positive camera depth")
        if self.is_distorted:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("OpenCV is required for distorted projection") from exc
            image, _ = cv2.projectPoints(
                points, np.zeros(3), np.zeros(3), self.matrix, self.distortion
            )
            return image.reshape(-1, 2)
        normalized = points[:, :2] / points[:, 2:3]
        return normalized * np.array([self.fx, self.fy]) + np.array([self.cx, self.cy])

    def unproject(self, pixels: np.ndarray, depth_m: Iterable[float]) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        depth = np.asarray(depth_m, dtype=np.float64).reshape(-1)
        if depth.size == 1:
            depth = np.repeat(depth, pixels.shape[0])
        if depth.shape[0] != pixels.shape[0] or np.any(depth <= 0.0):
            raise ValueError("depth must be positive and match the number of pixels")
        if self.is_distorted:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("OpenCV is required for distorted unprojection") from exc
            normalized = cv2.undistortPoints(
                pixels.reshape(-1, 1, 2), self.matrix, self.distortion
            ).reshape(-1, 2)
        else:
            normalized = (pixels - np.array([self.cx, self.cy])) / np.array([self.fx, self.fy])
        return np.column_stack((normalized * depth[:, None], depth))

    def with_distortion(self, coefficients: Iterable[float]) -> "CameraModel":
        return CameraModel(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy,
            np.asarray(coefficients, dtype=np.float64),
        )

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "CameraModel":
        return cls(
            width=value.get("width", 640), height=value.get("height", 480),
            fx=value.get("fx", 600.0), fy=value.get("fy", 600.0),
            cx=value.get("cx", 320.0), cy=value.get("cy", 240.0),
            distortion=value.get("distortion", [0, 0, 0, 0, 0]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width, "height": self.height,
            "camera_matrix": self.matrix.tolist(),
            "distortion": self.distortion.tolist(),
        }
