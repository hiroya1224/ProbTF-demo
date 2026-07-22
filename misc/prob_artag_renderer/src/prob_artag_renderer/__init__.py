"""Synthetic AprilTag rendering and exact geometric annotations."""

from .camera_model import CameraModel
from .scene_sampler import SCENARIOS, SceneSampler

__all__ = ["CameraModel", "SCENARIOS", "SceneSampler"]
__version__ = "0.1.0"
