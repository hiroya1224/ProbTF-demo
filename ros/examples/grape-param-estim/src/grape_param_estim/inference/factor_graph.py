"""Compatibility boundary for the optional factor-graph smoother.

The implementation remains an explicitly non-default comparison backend while
callers migrate away from the legacy all-in-one registry.
"""

from grape_param_estim.alternative_backends import (
    BatchImuPreintegrationSmoother,
    FactorGraphSmootherConfig,
)

__all__ = [
    "BatchImuPreintegrationSmoother",
    "FactorGraphSmootherConfig",
]
