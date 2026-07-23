"""Offline evaluation for synthetic probabilistic AprilTag datasets."""

from .adapter import ApiMismatchError, DefaultPipelineAdapter
from .models import BenchmarkConfig, FrameInput, SeedPose
from .runner import evaluate_dataset

__all__ = [
    "ApiMismatchError",
    "BenchmarkConfig",
    "DefaultPipelineAdapter",
    "FrameInput",
    "SeedPose",
    "evaluate_dataset",
]

__version__ = "0.1.0"
