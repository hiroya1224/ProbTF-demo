"""Lazy compatibility view of the established target-tube implementation."""

from typing import Any


_NAMES = frozenset(
    ("TargetTrajectory", "TargetTube", "TubeEvaluation", "evaluate_target_tube")
)


def __getattr__(name: str) -> Any:
    if name not in _NAMES:
        raise AttributeError(name)
    from grape_param_estim import counterfactual

    return getattr(counterfactual, name)


def __dir__():
    return sorted(set(globals()) | _NAMES)


__all__ = sorted(_NAMES)
