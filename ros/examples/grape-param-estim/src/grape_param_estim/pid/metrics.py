"""Separated physical metrics and robust posterior summaries for PID search."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from grape_param_estim.controller_config import PidGainConfiguration


PHYSICAL_ERROR_METRICS = (
    "position_rmse",
    "orientation_rmse",
    "maximum_position_error",
    "maximum_orientation_error",
)
FORECAST_COST_METRICS = PHYSICAL_ERROR_METRICS + (
    "numerical_failure_count",
    "actuator_saturation_duration",
    "actuator_saturation_rate",
)
DEFAULT_SELECTION_POLICY = (
    "Pareto non-dominated candidates that weakly improve the current PID in "
    "every completion, failure, physical-error, and saturation tail metric, "
    "with at least one strict improvement; gain change remains a separate "
    "Pareto objective"
)


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("{} must be finite and non-negative".format(name))
    return result


@dataclass(frozen=True)
class ForecastMetrics:
    """Unit-preserving outcome from one closed-loop forecast."""

    position_rmse: float
    orientation_rmse: float
    maximum_position_error: float
    maximum_orientation_error: float
    forecast_completion: float
    numerical_failure_count: int
    actuator_saturation_duration: float
    actuator_saturation_rate: float

    def __post_init__(self) -> None:
        for name in PHYSICAL_ERROR_METRICS + (
            "actuator_saturation_duration",
            "actuator_saturation_rate",
        ):
            object.__setattr__(
                self, name, _finite_nonnegative(getattr(self, name), name)
            )
        completion = float(self.forecast_completion)
        if not np.isfinite(completion) or not 0.0 <= completion <= 1.0:
            raise ValueError("forecast_completion must be in [0, 1]")
        failures = self.numerical_failure_count
        if (
            isinstance(failures, (bool, np.bool_))
            or not isinstance(failures, (int, np.integer))
            or failures < 0
        ):
            raise ValueError("numerical_failure_count must be non-negative")
        if self.actuator_saturation_rate > 1.0:
            raise ValueError("actuator_saturation_rate must be in [0, 1]")
        object.__setattr__(self, "forecast_completion", completion)
        object.__setattr__(self, "numerical_failure_count", int(failures))

    def cost_values(self) -> np.ndarray:
        result = np.asarray(
            tuple(getattr(self, name) for name in FORECAST_COST_METRICS),
            dtype=float,
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class ForecastMetricRecord:
    """Audited identity of one candidate/sample/bag/discrepancy forecast."""

    candidate_id: str
    sample_id: str
    bag_id: str
    replicate_index: int
    discrepancy_seed: int
    metrics: ForecastMetrics

    def __post_init__(self) -> None:
        for name in ("candidate_id", "sample_id", "bag_id"):
            value = str(getattr(self, name))
            if not value or value.strip() != value:
                raise ValueError("{} must be a canonical string".format(name))
            object.__setattr__(self, name, value)
        for name in ("replicate_index", "discrepancy_seed"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError("{} must be non-negative".format(name))
            object.__setattr__(self, name, int(value))
        if not isinstance(self.metrics, ForecastMetrics):
            raise TypeError("metrics must be ForecastMetrics")


def _tail_inputs(value: Sequence[float], level: float) -> Tuple[np.ndarray, float]:
    values = np.asarray(value, dtype=float)
    selected_level = float(level)
    if (
        values.ndim != 1
        or values.size < 1
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("tail values must be a finite non-empty vector")
    if not np.isfinite(selected_level) or not 0.0 <= selected_level < 1.0:
        raise ValueError("tail level must be in [0, 1)")
    return values, selected_level


def empirical_upper_cvar(value: Sequence[float], level: float) -> float:
    """Equal-weight upper CVaR with an exact fractional boundary draw."""

    values, selected_level = _tail_inputs(value, level)
    ordered = np.sort(values)[::-1]
    tail_count = (1.0 - selected_level) * ordered.size
    whole = int(np.floor(tail_count))
    fraction = tail_count - whole
    total = float(np.sum(ordered[:whole]))
    if fraction > 0.0:
        total += fraction * float(ordered[whole])
    return total / tail_count


def empirical_lower_cvar(value: Sequence[float], level: float) -> float:
    """Equal-weight lower-tail CVaR for forecast completion."""

    values, selected_level = _tail_inputs(value, level)
    return -empirical_upper_cvar(-values, selected_level)


@dataclass(frozen=True)
class CandidateMetricSummary:
    """Posterior/bag/replicate summaries without combining physical units."""

    candidate_id: str
    record_count: int
    metric_names: Tuple[str, ...]
    mean: np.ndarray
    quantile: np.ndarray
    upper_cvar: np.ndarray
    forecast_completion_mean: float
    forecast_completion_lower_quantile: float
    forecast_completion_lower_cvar: float
    gain_change_magnitude: float
    quantile_level: float
    cvar_level: float

    def __post_init__(self) -> None:
        identifier = str(self.candidate_id)
        names = tuple(str(value) for value in self.metric_names)
        if not identifier:
            raise ValueError("candidate_id cannot be empty")
        if names != FORECAST_COST_METRICS:
            raise ValueError("metric names must preserve physical separation")
        if (
            isinstance(self.record_count, (bool, np.bool_))
            or not isinstance(self.record_count, (int, np.integer))
            or self.record_count < 1
        ):
            raise ValueError("record_count must be positive")
        for name in ("mean", "quantile", "upper_cvar"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (len(names),) or np.any(~np.isfinite(value)):
                raise ValueError("{} must align with metric_names".format(name))
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        for name in (
            "forecast_completion_mean",
            "forecast_completion_lower_quantile",
            "forecast_completion_lower_cvar",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("{} must be in [0, 1]".format(name))
            object.__setattr__(self, name, value)
        change = _finite_nonnegative(
            self.gain_change_magnitude, "gain_change_magnitude"
        )
        quantile_level = float(self.quantile_level)
        cvar_level = float(self.cvar_level)
        if not np.isfinite(quantile_level) or not 0.0 < quantile_level < 1.0:
            raise ValueError("quantile_level must be in (0, 1)")
        if not np.isfinite(cvar_level) or not 0.0 <= cvar_level < 1.0:
            raise ValueError("cvar_level must be in [0, 1)")
        object.__setattr__(self, "candidate_id", identifier)
        object.__setattr__(self, "record_count", int(self.record_count))
        object.__setattr__(self, "metric_names", names)
        object.__setattr__(self, "gain_change_magnitude", change)
        object.__setattr__(self, "quantile_level", quantile_level)
        object.__setattr__(self, "cvar_level", cvar_level)

    def performance_cost_vector(self) -> np.ndarray:
        """Return lower-is-better objectives, excluding gain change."""

        result = np.concatenate(
            (
                np.asarray(
                    (
                        -self.forecast_completion_mean,
                        -self.forecast_completion_lower_quantile,
                        -self.forecast_completion_lower_cvar,
                    )
                ),
                self.mean,
                self.quantile,
                self.upper_cvar,
            )
        )
        result.setflags(write=False)
        return result

    def pareto_cost_vector(self) -> np.ndarray:
        result = np.concatenate(
            (
                self.performance_cost_vector(),
                np.asarray((self.gain_change_magnitude,)),
            )
        )
        result.setflags(write=False)
        return result

    def metric_index(self, metric_name: str) -> int:
        try:
            return self.metric_names.index(str(metric_name))
        except ValueError as error:
            raise KeyError("unknown metric") from error


def gain_change_magnitude(
    current: PidGainConfiguration, candidate: PidGainConfiguration
) -> float:
    """Return an RMS symmetric relative gain change as a separate objective."""

    if not isinstance(current, PidGainConfiguration) or not isinstance(
        candidate, PidGainConfiguration
    ):
        raise TypeError("gain configurations have the wrong type")
    denominator = np.maximum(
        np.maximum(np.abs(current.values), np.abs(candidate.values)), 1.0e-12
    )
    relative = (candidate.values - current.values) / denominator
    return float(np.sqrt(np.mean(relative * relative)))


def summarize_forecast_records(
    records: Sequence[ForecastMetricRecord],
    current: PidGainConfiguration,
    candidate: PidGainConfiguration,
    *,
    quantile_level: float = 0.95,
    cvar_level: float = 0.90,
) -> CandidateMetricSummary:
    """Summarize a complete Cartesian forecast set for one candidate."""

    selected = tuple(records)
    if not selected or any(
        not isinstance(value, ForecastMetricRecord) for value in selected
    ):
        raise ValueError("records must contain forecast metric records")
    candidate_ids = {value.candidate_id for value in selected}
    if len(candidate_ids) != 1:
        raise ValueError("one summary cannot mix candidate IDs")
    values = np.vstack(tuple(value.metrics.cost_values() for value in selected))
    completion = np.asarray(
        tuple(value.metrics.forecast_completion for value in selected)
    )
    selected_quantile = float(quantile_level)
    selected_cvar = float(cvar_level)
    if not np.isfinite(selected_quantile) or not 0.0 < selected_quantile < 1.0:
        raise ValueError("quantile_level must be in (0, 1)")
    if not np.isfinite(selected_cvar) or not 0.0 <= selected_cvar < 1.0:
        raise ValueError("cvar_level must be in [0, 1)")
    return CandidateMetricSummary(
        candidate_id=selected[0].candidate_id,
        record_count=len(selected),
        metric_names=FORECAST_COST_METRICS,
        mean=np.mean(values, axis=0),
        quantile=np.quantile(values, selected_quantile, axis=0),
        upper_cvar=np.asarray(
            tuple(
                empirical_upper_cvar(values[:, index], selected_cvar)
                for index in range(values.shape[1])
            )
        ),
        forecast_completion_mean=float(np.mean(completion)),
        forecast_completion_lower_quantile=float(
            np.quantile(completion, 1.0 - selected_quantile)
        ),
        forecast_completion_lower_cvar=empirical_lower_cvar(
            completion, selected_cvar
        ),
        gain_change_magnitude=gain_change_magnitude(current, candidate),
        quantile_level=selected_quantile,
        cvar_level=selected_cvar,
    )


def _dominates(
    first: np.ndarray,
    second: np.ndarray,
    tolerance: float,
) -> bool:
    return bool(
        np.all(first <= second + tolerance)
        and np.any(first < second - tolerance)
    )


def pareto_nondominated_candidate_ids(
    summaries: Sequence[CandidateMetricSummary],
    *,
    tolerance: float = 1.0e-12,
) -> Tuple[str, ...]:
    """Return candidates not dominated across separated robust objectives."""

    selected = tuple(summaries)
    if not selected or any(
        not isinstance(value, CandidateMetricSummary) for value in selected
    ):
        raise ValueError("summaries must contain candidate metric summaries")
    identifiers = tuple(value.candidate_id for value in selected)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate summaries must have unique IDs")
    selected_tolerance = float(tolerance)
    if not np.isfinite(selected_tolerance) or selected_tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    costs = tuple(value.pareto_cost_vector() for value in selected)
    return tuple(
        identifiers[index]
        for index in range(len(selected))
        if not any(
            other != index
            and _dominates(costs[other], costs[index], selected_tolerance)
            for other in range(len(selected))
        )
    )


def componentwise_improves_current(
    candidate: CandidateMetricSummary,
    current: CandidateMetricSummary,
    *,
    tolerance: float = 1.0e-12,
) -> bool:
    """Require no worse performance objective and one strict improvement."""

    if not isinstance(candidate, CandidateMetricSummary) or not isinstance(
        current, CandidateMetricSummary
    ):
        raise TypeError("candidate summaries have the wrong type")
    selected_tolerance = float(tolerance)
    if not np.isfinite(selected_tolerance) or selected_tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    return _dominates(
        candidate.performance_cost_vector(),
        current.performance_cost_vector(),
        selected_tolerance,
    )


@dataclass(frozen=True)
class RecommendationDecision:
    """Default Pareto recommendation set, or an explicit unavailable result."""

    nondominated_candidate_ids: Tuple[str, ...]
    recommended_candidate_ids: Tuple[str, ...]
    recommendation_available: bool
    rejection_reason: str
    selection_policy: str = DEFAULT_SELECTION_POLICY

    def __post_init__(self) -> None:
        nondominated = tuple(str(value) for value in self.nondominated_candidate_ids)
        recommended = tuple(str(value) for value in self.recommended_candidate_ids)
        available = bool(self.recommendation_available)
        reason = str(self.rejection_reason)
        policy = str(self.selection_policy)
        if (
            not nondominated
            or len(set(nondominated)) != len(nondominated)
            or len(set(recommended)) != len(recommended)
            or any(value not in nondominated for value in recommended)
            or available != bool(recommended)
            or (available and reason)
            or (not available and not reason)
            or not policy
        ):
            raise ValueError("recommendation decision is inconsistent")
        object.__setattr__(self, "nondominated_candidate_ids", nondominated)
        object.__setattr__(self, "recommended_candidate_ids", recommended)
        object.__setattr__(self, "recommendation_available", available)
        object.__setattr__(self, "rejection_reason", reason)
        object.__setattr__(self, "selection_policy", policy)


def decide_recommendation(
    summaries: Sequence[CandidateMetricSummary],
    *,
    current_candidate_id: str = "current",
    tolerance: float = 1.0e-12,
) -> RecommendationDecision:
    """Return improved Pareto candidates without inventing a weighted score."""

    selected = tuple(summaries)
    identifiers = tuple(value.candidate_id for value in selected)
    current_id = str(current_candidate_id)
    if identifiers.count(current_id) != 1:
        raise ValueError("exactly one current baseline summary is required")
    current = selected[identifiers.index(current_id)]
    nondominated = pareto_nondominated_candidate_ids(
        selected, tolerance=tolerance
    )
    recommended = tuple(
        value.candidate_id
        for value in selected
        if value.candidate_id != current_id
        and value.candidate_id in nondominated
        and componentwise_improves_current(
            value, current, tolerance=tolerance
        )
    )
    return RecommendationDecision(
        nondominated_candidate_ids=nondominated,
        recommended_candidate_ids=recommended,
        recommendation_available=bool(recommended),
        rejection_reason=(
            ""
            if recommended
            else "recommendation unavailable: no Pareto candidate improves current"
        ),
    )


__all__ = [
    "CandidateMetricSummary",
    "DEFAULT_SELECTION_POLICY",
    "FORECAST_COST_METRICS",
    "ForecastMetricRecord",
    "ForecastMetrics",
    "PHYSICAL_ERROR_METRICS",
    "RecommendationDecision",
    "componentwise_improves_current",
    "decide_recommendation",
    "empirical_lower_cvar",
    "empirical_upper_cvar",
    "gain_change_magnitude",
    "pareto_nondominated_candidate_ids",
    "summarize_forecast_records",
]
