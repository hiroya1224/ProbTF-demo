"""Derivative-free continuous command-delay profile optimization."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable, Optional, Tuple

import numpy as np

from grape_param_estim.batch.state import BatchState


LagEvaluator = Callable[[float, Optional[BatchState]], "LagObjectiveResult"]


@dataclass(frozen=True)
class LagObjectiveResult:
    """One fixed-lag inner MAP result returned to the profile optimizer."""

    objective: Optional[float]
    converged: bool
    state: Optional[BatchState]
    inner_iterations: int
    termination_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.converged, (bool, np.bool_)):
            raise TypeError("converged must be boolean")
        if self.objective is not None:
            objective = float(self.objective)
            if not np.isfinite(objective):
                raise ValueError("objective must be finite when present")
            object.__setattr__(self, "objective", objective)
        if self.converged and self.objective is None:
            raise ValueError("a converged lag evaluation requires an objective")
        if self.converged and self.state is None:
            raise ValueError("a converged lag evaluation requires a state")
        if self.state is not None and not isinstance(self.state, BatchState):
            raise TypeError("state must be a BatchState or None")
        if (
            isinstance(self.inner_iterations, bool)
            or not isinstance(self.inner_iterations, Integral)
            or self.inner_iterations < 0
        ):
            raise ValueError("inner_iterations must be a non-negative integer")
        if (
            type(self.termination_reason) is not str
            or not self.termination_reason
            or self.termination_reason.strip() != self.termination_reason
        ):
            raise ValueError("termination_reason must be a canonical string")
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "inner_iterations", int(self.inner_iterations))


@dataclass(frozen=True)
class LagProfileSettings:
    """Bounded coarse-grid and golden-section refinement settings."""

    minimum_lag: float
    maximum_lag: float
    coarse_grid_points: int = 9
    refinement_tolerance: float = 1.0e-5
    maximum_refinement_evaluations: int = 32

    def __post_init__(self) -> None:
        minimum = float(self.minimum_lag)
        maximum = float(self.maximum_lag)
        tolerance = float(self.refinement_tolerance)
        if (
            not np.isfinite(minimum)
            or not np.isfinite(maximum)
            or minimum < 0.0
            or maximum <= minimum
        ):
            raise ValueError("lag bounds must be finite, non-negative, and ordered")
        if (
            isinstance(self.coarse_grid_points, bool)
            or not isinstance(self.coarse_grid_points, Integral)
            or self.coarse_grid_points < 3
        ):
            raise ValueError("coarse_grid_points must be an integer at least three")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("refinement_tolerance must be finite and positive")
        if tolerance >= maximum - minimum:
            raise ValueError("refinement_tolerance must be smaller than the bounds")
        if (
            isinstance(self.maximum_refinement_evaluations, bool)
            or not isinstance(self.maximum_refinement_evaluations, Integral)
            or self.maximum_refinement_evaluations < 2
        ):
            raise ValueError(
                "maximum_refinement_evaluations must be an integer at least two"
            )
        object.__setattr__(self, "minimum_lag", minimum)
        object.__setattr__(self, "maximum_lag", maximum)
        object.__setattr__(self, "coarse_grid_points", int(self.coarse_grid_points))
        object.__setattr__(self, "refinement_tolerance", tolerance)
        object.__setattr__(
            self,
            "maximum_refinement_evaluations",
            int(self.maximum_refinement_evaluations),
        )


@dataclass(frozen=True)
class LagProfilePoint:
    """One chronological objective evaluation and warm-start provenance."""

    lag: float
    phase: str
    objective: Optional[float]
    converged: bool
    inner_iterations: int
    termination_reason: str
    warm_start_lag: Optional[float]


@dataclass(frozen=True)
class LagProfileResult:
    """Best converged continuous lag and all objective evaluations."""

    best_lag: float
    best_objective: float
    best_state: Optional[BatchState]
    initial_refinement_bracket: Tuple[float, float]
    final_refinement_bracket: Tuple[float, float]
    points: Tuple[LagProfilePoint, ...]


class LagProfileFailure(RuntimeError):
    """No converged fixed-lag inner solve was available in the bounds."""


def optimize_lag_profile(
    evaluator: LagEvaluator,
    settings: LagProfileSettings,
    initial_warm_start: Optional[BatchState] = None,
) -> LagProfileResult:
    """Run a bounded coarse grid followed by continuous golden refinement."""

    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if not isinstance(settings, LagProfileSettings):
        raise TypeError("settings must be LagProfileSettings")
    if initial_warm_start is not None and not isinstance(
        initial_warm_start, BatchState
    ):
        raise TypeError("initial_warm_start must be a BatchState or None")

    cache = {}
    points = []

    def evaluate(lag: float, phase: str) -> LagObjectiveResult:
        lag_value = float(lag)
        if lag_value in cache:
            return cache[lag_value]
        successful = [
            (known_lag, result)
            for known_lag, result in cache.items()
            if result.converged and result.state is not None
        ]
        if successful:
            warm_lag, warm_result = min(
                successful,
                key=lambda item: (abs(item[0] - lag_value), item[0]),
            )
            warm_state = warm_result.state
        else:
            warm_lag = None
            warm_state = initial_warm_start
        result = evaluator(lag_value, warm_state)
        if not isinstance(result, LagObjectiveResult):
            raise TypeError("evaluator must return a LagObjectiveResult")
        cache[lag_value] = result
        points.append(
            LagProfilePoint(
                lag=lag_value,
                phase=phase,
                objective=result.objective,
                converged=result.converged,
                inner_iterations=result.inner_iterations,
                termination_reason=result.termination_reason,
                warm_start_lag=warm_lag,
            )
        )
        return result

    coarse_lags = np.linspace(
        settings.minimum_lag,
        settings.maximum_lag,
        settings.coarse_grid_points,
    )
    coarse_results = [evaluate(float(lag), "coarse") for lag in coarse_lags]
    usable_indices = [
        index
        for index, result in enumerate(coarse_results)
        if result.converged and result.objective is not None
    ]
    if not usable_indices:
        raise LagProfileFailure("no coarse-grid lag evaluation converged")
    best_coarse_index = min(
        usable_indices,
        key=lambda index: (
            coarse_results[index].objective,
            float(coarse_lags[index]),
        ),
    )
    left_index = max(0, best_coarse_index - 1)
    right_index = min(len(coarse_lags) - 1, best_coarse_index + 1)
    if left_index == right_index:
        raise LagProfileFailure("coarse grid did not produce a refinement bracket")
    left = float(coarse_lags[left_index])
    right = float(coarse_lags[right_index])
    initial_bracket = (left, right)

    golden_ratio = 0.5 * (np.sqrt(5.0) - 1.0)
    inner_left = right - golden_ratio * (right - left)
    inner_right = left + golden_ratio * (right - left)
    left_result = evaluate(inner_left, "refinement")
    right_result = evaluate(inner_right, "refinement")
    refinement_count = 2

    def profile_value(result: LagObjectiveResult) -> float:
        return (
            result.objective
            if result.converged and result.objective is not None
            else np.inf
        )

    while (
        right - left > settings.refinement_tolerance
        and refinement_count < settings.maximum_refinement_evaluations
    ):
        if profile_value(left_result) <= profile_value(right_result):
            right = inner_right
            inner_right = inner_left
            right_result = left_result
            inner_left = right - golden_ratio * (right - left)
            left_result = evaluate(inner_left, "refinement")
        else:
            left = inner_left
            inner_left = inner_right
            left_result = right_result
            inner_right = left + golden_ratio * (right - left)
            right_result = evaluate(inner_right, "refinement")
        refinement_count += 1

    successful = [
        (lag, result)
        for lag, result in cache.items()
        if result.converged and result.objective is not None
    ]
    best_lag, best_result = min(
        successful,
        key=lambda item: (item[1].objective, item[0]),
    )
    return LagProfileResult(
        best_lag=float(best_lag),
        best_objective=float(best_result.objective),
        best_state=best_result.state,
        initial_refinement_bracket=initial_bracket,
        final_refinement_bracket=(left, right),
        points=tuple(points),
    )


__all__ = [
    "LagEvaluator",
    "LagObjectiveResult",
    "LagProfileFailure",
    "LagProfilePoint",
    "LagProfileResult",
    "LagProfileSettings",
    "optimize_lag_profile",
]
