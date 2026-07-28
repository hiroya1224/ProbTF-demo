"""Lazy compatibility view of held-out probability calibration evidence."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name != "ProbabilityCalibrationReport":
        raise AttributeError(name)
    from grape_param_estim.counterfactual import ProbabilityCalibrationReport

    return ProbabilityCalibrationReport


def __dir__():
    return sorted(set(globals()) | {"ProbabilityCalibrationReport"})


__all__ = ["ProbabilityCalibrationReport"]
