"""Multi-bag full closed-loop evaluation of exact PID gain candidates.

The physical posterior is never averaged.  Every exact candidate is flown
against every raw static member and every selected bag.  Position and
orientation errors retain their physical units and are never combined into a
weighted tracking loss.  Candidate comparison is Pareto based; no automatic
representative is selected from the member-derived proposal ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.artifact_io import (
    PID_PROPOSAL_EVALUATION_SCHEMA,
    WRITING_STATUS,
    begin_bundle,
    mark_bundle_cancelled,
    mark_bundle_complete,
    read_manifest,
)
from grape_param_estim.controller import ControllerConfig, GrapeController
from grape_param_estim.controller_config import (
    PID_GROUPS,
    PidGainComparison,
    PidGainConfiguration,
    apply_pid_gain_configuration,
    render_pid_diff_yaml,
    render_proposed_pid_yaml,
)
from grape_param_estim.dynamics import FullSixDofPlant, simulate_closed_loop
from grape_param_estim.geometry import (
    correction_transform_path,
    euler_xyz_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_vector_from_matrix,
)
from grape_param_estim.pid_proposal import (
    PidGainCandidate,
    PidProposalEnsemble,
    current_pid_candidate,
    derive_pid_proposal_ensemble,
)
from grape_param_estim.progress import (
    CancellationToken,
    ProgressCallback,
    ProgressCancelled,
    ProgressTracker,
)
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    ClosedLoopTrajectory,
    ControllerState,
    GrapeGeometry,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


PHYSICAL_METRICS = (
    "position_rmse",
    "orientation_rmse",
    "maximum_position_error",
    "maximum_orientation_error",
)
POSITION_THRESHOLD_METRICS = (
    "position_rmse",
    "maximum_position_error",
)
ORIENTATION_THRESHOLD_METRICS = (
    "orientation_rmse",
    "maximum_orientation_error",
)
RESIDUAL_POLICIES = ("posterior_replay", "zero")
NOT_CONFIGURED = "Not configured"
CORRECTION_COVERAGE_INTERVAL = 0.95
COMPONENTWISE_IMPROVEMENT_RULE = (
    "The explicitly selected proposal must have no lower forecast completion, "
    "no more numerical failures, and no worse bag-equal mean or upper-CVaR "
    "for any separately reported position/orientation RMSE or maximum-error "
    "metric (and no worse configured threshold exceedance), with at least one "
    "strict improvement. It must also be Pareto non-dominated. Gain-change "
    "magnitude is a Pareto objective but is not traded against physical error "
    "through a weighted score."
)
COUNTERFACTUAL_ASSUMPTION_PREFIX = (
    "same recorded reference; same posterior member initial state; same "
    "estimated static plant member and constant delay"
)


def _finite_vector(value: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or np.any(~np.isfinite(result)):
        raise ValueError("{} must be a finite {} array".format(name, shape))
    return result.copy()


def _normalised_provenance(
    provenance: Optional[Mapping[str, str]],
) -> Tuple[Tuple[str, str], ...]:
    if provenance is None:
        return tuple()
    result = tuple(
        sorted((str(key), str(value)) for key, value in provenance.items())
    )
    if any(not key for key, _value in result):
        raise ValueError("provenance keys cannot be empty")
    if len({key for key, _value in result}) != len(result):
        raise ValueError("provenance keys must be unique")
    return result


@dataclass(frozen=True)
class ErrorThresholds:
    """Optional, unit-preserving threshold definitions.

    Thresholds default to RMSE.  A caller may explicitly select the matching
    maximum-error metric instead.  An absent value remains ``Not configured``
    and is excluded from Pareto and recommendation decisions.
    """

    position: Optional[float] = None
    orientation: Optional[float] = None
    position_metric: str = "position_rmse"
    orientation_metric: str = "orientation_rmse"

    def __post_init__(self) -> None:
        if self.position_metric not in POSITION_THRESHOLD_METRICS:
            raise ValueError("invalid position threshold metric")
        if self.orientation_metric not in ORIENTATION_THRESHOLD_METRICS:
            raise ValueError("invalid orientation threshold metric")
        for name in ("position", "orientation"):
            value = getattr(self, name)
            if value is None:
                continue
            selected = float(value)
            if not np.isfinite(selected) or selected <= 0.0:
                raise ValueError("{} threshold must be positive".format(name))
            object.__setattr__(self, name, selected)

    @property
    def position_configured(self) -> bool:
        return self.position is not None

    @property
    def orientation_configured(self) -> bool:
        return self.orientation is not None

    def position_display(self) -> str:
        return (
            NOT_CONFIGURED
            if self.position is None
            else "{} {} m".format(self.position_metric, self.position)
        )

    def orientation_display(self) -> str:
        return (
            NOT_CONFIGURED
            if self.orientation is None
            else "{} {} rad".format(self.orientation_metric, self.orientation)
        )


@dataclass(frozen=True)
class CounterfactualBagScenario:
    """One bag-specific scenario sharing the static physical member law."""

    bag_id: str
    times: np.ndarray
    references: Tuple[ReferenceState, ...]
    initial_states: Tuple[RigidBodyState, ...]
    initial_controller_states: Tuple[ControllerState, ...]
    initial_actuator_states: Tuple[Optional[ActuatorState], ...]
    posterior_residual_wrench: np.ndarray
    controller_configuration: ControllerConfig
    controller_nominal_parameters: VehicleParameters
    controller_geometry: GrapeGeometry
    plant_geometry: GrapeGeometry
    actuator_parameters: ActuatorParameters
    residual_policy: str = "posterior_replay"
    provenance: Tuple[Tuple[str, str], ...] = tuple()

    def __post_init__(self) -> None:
        bag_id = str(self.bag_id)
        times = np.asarray(self.times, dtype=float)
        references = tuple(self.references)
        states = tuple(self.initial_states)
        controller_states = tuple(self.initial_controller_states)
        actuator_states = tuple(self.initial_actuator_states)
        residual = np.asarray(self.posterior_residual_wrench, dtype=float)
        policy = str(self.residual_policy)
        member_count = len(states)
        if not bag_id:
            raise ValueError("bag_id cannot be empty")
        if (
            times.ndim != 1
            or times.size < 2
            or np.any(~np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
            or len(references) != times.size
            or any(not isinstance(value, ReferenceState) for value in references)
        ):
            raise ValueError("scenario times and references must align")
        if (
            member_count < 1
            or len(controller_states) != member_count
            or len(actuator_states) != member_count
            or any(not isinstance(value, RigidBodyState) for value in states)
            or any(
                not isinstance(value, ControllerState)
                for value in controller_states
            )
            or any(
                value is not None and not isinstance(value, ActuatorState)
                for value in actuator_states
            )
            or residual.shape != (member_count, times.size - 1, 6)
            or np.any(~np.isfinite(residual))
        ):
            raise ValueError("scenario member-local paths must stay aligned")
        if policy not in RESIDUAL_POLICIES:
            raise ValueError("unknown residual policy")
        for value, expected, name in (
            (self.controller_configuration, ControllerConfig, "controller"),
            (
                self.controller_nominal_parameters,
                VehicleParameters,
                "controller nominal parameters",
            ),
            (self.controller_geometry, GrapeGeometry, "controller geometry"),
            (self.plant_geometry, GrapeGeometry, "plant geometry"),
            (self.actuator_parameters, ActuatorParameters, "actuator parameters"),
        ):
            if not isinstance(value, expected):
                raise TypeError("{} has the wrong type".format(name))
        provenance = tuple((str(key), str(value)) for key, value in self.provenance)
        if any(not key for key, _value in provenance):
            raise ValueError("scenario provenance keys cannot be empty")
        object.__setattr__(self, "bag_id", bag_id)
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "initial_states", states)
        object.__setattr__(self, "initial_controller_states", controller_states)
        object.__setattr__(self, "initial_actuator_states", actuator_states)
        object.__setattr__(self, "posterior_residual_wrench", residual.copy())
        object.__setattr__(self, "residual_policy", policy)
        object.__setattr__(self, "provenance", provenance)

    @property
    def member_count(self) -> int:
        return len(self.initial_states)

    @property
    def reference_position(self) -> np.ndarray:
        return np.asarray([value.position for value in self.references])

    @property
    def reference_rpy(self) -> np.ndarray:
        return np.asarray([value.rpy for value in self.references])

    @property
    def reference_orientation_xyzw(self) -> np.ndarray:
        return np.asarray(
            [
                matrix_to_quaternion(euler_xyz_to_matrix(value.rpy))
                for value in self.references
            ]
        )


@dataclass(frozen=True)
class PosteriorPredictiveInput:
    """Shared physical posterior plus independent variable-length bag paths."""

    selected_mode_id: str
    physical_parameter_members: Tuple[VehicleParameters, ...]
    proposal_ensemble: PidProposalEnsemble
    bags: Tuple[CounterfactualBagScenario, ...]
    provenance: Tuple[Tuple[str, str], ...] = tuple()

    def __post_init__(self) -> None:
        mode = str(self.selected_mode_id)
        physical = tuple(self.physical_parameter_members)
        bags = tuple(self.bags)
        if not mode:
            raise ValueError("selected_mode_id cannot be empty")
        if not isinstance(self.proposal_ensemble, PidProposalEnsemble):
            raise TypeError("proposal_ensemble has the wrong type")
        member_count = self.proposal_ensemble.member_id.size
        if (
            len(physical) != member_count
            or any(not isinstance(value, VehicleParameters) for value in physical)
        ):
            raise ValueError("physical members must align with PID proposals")
        if any(value != mode for value in self.proposal_ensemble.source_mode_id):
            raise ValueError(
                "posterior predictive input cannot average across modes"
            )
        if (
            not bags
            or len({value.bag_id for value in bags}) != len(bags)
            or any(value.member_count != member_count for value in bags)
        ):
            raise ValueError("bags must be unique and share the raw member law")
        provenance = tuple((str(key), str(value)) for key, value in self.provenance)
        if any(not key for key, _value in provenance):
            raise ValueError("input provenance keys cannot be empty")
        object.__setattr__(self, "selected_mode_id", mode)
        object.__setattr__(self, "physical_parameter_members", physical)
        object.__setattr__(self, "bags", bags)
        object.__setattr__(self, "provenance", provenance)

    @property
    def member_id(self) -> np.ndarray:
        return self.proposal_ensemble.member_id.copy()

    @property
    def current(self) -> PidGainConfiguration:
        return self.proposal_ensemble.current

    @property
    def scenario_assumption(self) -> str:
        policies = ", ".join(
            "{}={}".format(value.bag_id, value.residual_policy)
            for value in self.bags
        )
        return (
            "{}; residual policy by bag: {}; this is not a forecast of a new "
            "disturbance realization"
        ).format(COUNTERFACTUAL_ASSUMPTION_PREFIX, policies)


@dataclass(frozen=True)
class TrajectoryMetricValues:
    position_rmse: float
    orientation_rmse: float
    maximum_position_error: float
    maximum_orientation_error: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.position_rmse,
                self.orientation_rmse,
                self.maximum_position_error,
                self.maximum_orientation_error,
            ),
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("trajectory metrics must be finite and non-negative")
        for name, value in zip(PHYSICAL_METRICS, values):
            object.__setattr__(self, name, float(value))

    def as_array(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in PHYSICAL_METRICS])


@dataclass(frozen=True)
class MetricSummary:
    mean: Optional[float]
    upper_cvar: Optional[float]
    count: int

    def __post_init__(self) -> None:
        count = int(self.count)
        if count < 0:
            raise ValueError("metric count cannot be negative")
        if count == 0:
            if self.mean is not None or self.upper_cvar is not None:
                raise ValueError("empty metric summaries must be unavailable")
        else:
            values = np.asarray((self.mean, self.upper_cvar), dtype=float)
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError("metric summaries must be finite")
            object.__setattr__(self, "mean", float(values[0]))
            object.__setattr__(self, "upper_cvar", float(values[1]))
        object.__setattr__(self, "count", count)


@dataclass(frozen=True)
class CandidateSummary:
    metrics: Mapping[str, MetricSummary]
    position_threshold_exceedance: Optional[float]
    orientation_threshold_exceedance: Optional[float]
    forecast_completion: float
    numerical_failure_count: int

    def __post_init__(self) -> None:
        metrics = dict(self.metrics)
        if set(metrics) != set(PHYSICAL_METRICS) or any(
            not isinstance(value, MetricSummary) for value in metrics.values()
        ):
            raise ValueError("summary must retain all four physical metrics")
        for name in (
            "position_threshold_exceedance",
            "orientation_threshold_exceedance",
        ):
            value = getattr(self, name)
            if value is not None:
                selected = float(value)
                if not np.isfinite(selected) or not 0.0 <= selected <= 1.0:
                    raise ValueError("threshold exceedance must be in [0, 1]")
                object.__setattr__(self, name, selected)
        completion = float(self.forecast_completion)
        failures = int(self.numerical_failure_count)
        if not np.isfinite(completion) or not 0.0 <= completion <= 1.0:
            raise ValueError("forecast completion must be in [0, 1]")
        if failures < 0:
            raise ValueError("numerical failure count cannot be negative")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "forecast_completion", completion)
        object.__setattr__(self, "numerical_failure_count", failures)


@dataclass(frozen=True)
class BagCandidateEvaluation:
    bag_id: str
    member_id: np.ndarray
    trajectories: Tuple[Optional[ClosedLoopTrajectory], ...]
    prediction_position: np.ndarray
    prediction_orientation_xyzw: np.ndarray
    correction_translation: np.ndarray
    correction_rotation_vector: np.ndarray
    position_error: np.ndarray
    orientation_error_rotation_vector: np.ndarray
    position_rmse: np.ndarray
    orientation_rmse: np.ndarray
    maximum_position_error: np.ndarray
    maximum_orientation_error: np.ndarray
    forecast_completed: np.ndarray
    failure_reason: Tuple[str, ...]
    position_threshold_exceeded: Optional[np.ndarray]
    orientation_threshold_exceeded: Optional[np.ndarray]
    summary: CandidateSummary
    residual_policy: str
    correction_translation_zero_coverage: Optional[float]
    correction_rotation_zero_coverage: Optional[float]
    correction_transform_zero_coverage: Optional[float]

    def __post_init__(self) -> None:
        bag_id = str(self.bag_id)
        member_id = np.asarray(self.member_id, dtype=np.int64)
        completed = np.asarray(self.forecast_completed, dtype=bool)
        reasons = tuple(str(value) for value in self.failure_reason)
        members = member_id.size
        if (
            not bag_id
            or members < 1
            or member_id.shape != (members,)
            or np.unique(member_id).size != members
            or completed.shape != (members,)
            or len(reasons) != members
            or len(self.trajectories) != members
            or self.residual_policy not in RESIDUAL_POLICIES
            or not isinstance(self.summary, CandidateSummary)
        ):
            raise ValueError("bag candidate identity/status arrays are invalid")
        prediction_position = np.asarray(self.prediction_position, dtype=float)
        if prediction_position.ndim != 3 or prediction_position.shape[
            :1
        ] != (members,) or prediction_position.shape[2] != 3:
            raise ValueError("prediction_position must have shape (M, T, 3)")
        samples = prediction_position.shape[1]
        path_fields = (
            ("prediction_position", prediction_position, 3),
            (
                "prediction_orientation_xyzw",
                np.asarray(self.prediction_orientation_xyzw, dtype=float),
                4,
            ),
            (
                "correction_translation",
                np.asarray(self.correction_translation, dtype=float),
                3,
            ),
            (
                "correction_rotation_vector",
                np.asarray(self.correction_rotation_vector, dtype=float),
                3,
            ),
            ("position_error", np.asarray(self.position_error, dtype=float), 3),
            (
                "orientation_error_rotation_vector",
                np.asarray(self.orientation_error_rotation_vector, dtype=float),
                3,
            ),
        )
        for name, value, width in path_fields:
            if value.shape != (members, samples, width):
                raise ValueError("{} has invalid shape".format(name))
            object.__setattr__(self, name, value.copy())
        metric_arrays = {}
        for name in PHYSICAL_METRICS:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (members,):
                raise ValueError("{} must have shape (M,)".format(name))
            metric_arrays[name] = value
            object.__setattr__(self, name, value.copy())
        for index in range(members):
            paths = [value[index] for _name, value, _width in path_fields]
            metrics = np.asarray(
                [metric_arrays[name][index] for name in PHYSICAL_METRICS]
            )
            trajectory = self.trajectories[index]
            if completed[index]:
                if (
                    trajectory is None
                    or reasons[index]
                    or any(np.any(~np.isfinite(value)) for value in paths)
                    or np.any(~np.isfinite(metrics))
                ):
                    raise ValueError(
                        "completed forecasts need finite paths/metrics and no reason"
                    )
            elif (
                trajectory is not None
                or not reasons[index]
                or any(not np.all(np.isnan(value)) for value in paths)
                or not np.all(np.isnan(metrics))
            ):
                raise ValueError(
                    "failed forecasts need missing paths/metrics and a reason"
                )
        for name in (
            "position_threshold_exceeded",
            "orientation_threshold_exceeded",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            selected = np.asarray(value, dtype=float)
            if selected.shape != (members,):
                raise ValueError("{} must have shape (M,)".format(name))
            if np.any(~np.isnan(selected[~completed])) or np.any(
                ~np.isin(selected[completed], (0.0, 1.0))
            ):
                raise ValueError(
                    "threshold flags must exclude numerical failures"
                )
            object.__setattr__(self, name, selected.copy())
        for name in (
            "correction_translation_zero_coverage",
            "correction_rotation_zero_coverage",
            "correction_transform_zero_coverage",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            selected = float(value)
            if not np.isfinite(selected) or not 0.0 <= selected <= 1.0:
                raise ValueError("correction coverage must be in [0, 1]")
            object.__setattr__(self, name, selected)
        object.__setattr__(self, "bag_id", bag_id)
        object.__setattr__(self, "member_id", member_id.copy())
        object.__setattr__(self, "trajectories", tuple(self.trajectories))
        object.__setattr__(self, "forecast_completed", completed.copy())
        object.__setattr__(self, "failure_reason", reasons)


@dataclass(frozen=True)
class PidCandidateEvaluation:
    candidate: PidGainCandidate
    bags: Tuple[BagCandidateEvaluation, ...]
    aggregate: CandidateSummary
    log_gain_change: float
    pareto_dominated: bool = False
    improves_current: bool = False

    def __post_init__(self) -> None:
        bags = tuple(self.bags)
        change = float(self.log_gain_change)
        if (
            not isinstance(self.candidate, PidGainCandidate)
            or not bags
            or len({value.bag_id for value in bags}) != len(bags)
            or any(not isinstance(value, BagCandidateEvaluation) for value in bags)
            or not isinstance(self.aggregate, CandidateSummary)
            or np.isnan(change)
            or change < 0.0
        ):
            raise ValueError("PID candidate evaluation is invalid")
        object.__setattr__(self, "bags", bags)
        object.__setattr__(self, "log_gain_change", change)
        object.__setattr__(self, "pareto_dominated", bool(self.pareto_dominated))
        object.__setattr__(self, "improves_current", bool(self.improves_current))

    @property
    def pareto_non_dominated(self) -> bool:
        return not self.pareto_dominated

    def bag(self, bag_id: str) -> BagCandidateEvaluation:
        for value in self.bags:
            if value.bag_id == bag_id:
                return value
        raise KeyError("unknown candidate bag: {}".format(bag_id))


@dataclass(frozen=True)
class PosteriorPredictiveDecision:
    predictive_input: PosteriorPredictiveInput
    evaluations: Tuple[PidCandidateEvaluation, ...]
    cvar_level: float
    thresholds: ErrorThresholds
    selected_candidate_id: Optional[str]
    recommendation_available: bool
    recommended_candidate_id: str
    rejection_reason: str
    improvement_rule: str = COMPONENTWISE_IMPROVEMENT_RULE

    def __post_init__(self) -> None:
        evaluations = tuple(self.evaluations)
        level = float(self.cvar_level)
        if (
            not isinstance(self.predictive_input, PosteriorPredictiveInput)
            or not evaluations
            or any(
                not isinstance(value, PidCandidateEvaluation)
                for value in evaluations
            )
            or len({value.candidate.candidate_id for value in evaluations})
            != len(evaluations)
            or evaluations[0].candidate.candidate_id != "current"
            or not 0.0 <= level < 1.0
            or not isinstance(self.thresholds, ErrorThresholds)
        ):
            raise ValueError("posterior-predictive decision is invalid")
        selected = (
            None
            if self.selected_candidate_id is None
            else str(self.selected_candidate_id)
        )
        ids = {value.candidate.candidate_id for value in evaluations}
        if selected is not None and selected not in ids:
            raise ValueError("selected candidate was not evaluated")
        recommended = str(self.recommended_candidate_id)
        if self.recommendation_available:
            if not recommended or recommended != selected:
                raise ValueError("recommendation must be the explicit selection")
            evaluation = next(
                value
                for value in evaluations
                if value.candidate.candidate_id == recommended
            )
            if not evaluation.improves_current or evaluation.pareto_dominated:
                raise ValueError("recommended candidate is not eligible")
        elif recommended:
            raise ValueError("unavailable recommendation cannot have an ID")
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "cvar_level", level)
        object.__setattr__(self, "selected_candidate_id", selected)
        object.__setattr__(
            self, "recommendation_available", bool(self.recommendation_available)
        )
        object.__setattr__(self, "recommended_candidate_id", recommended)
        object.__setattr__(self, "rejection_reason", str(self.rejection_reason))

    @property
    def current_evaluation(self) -> PidCandidateEvaluation:
        return self.evaluations[0]

    @property
    def selected_evaluation(self) -> PidCandidateEvaluation:
        if self.selected_candidate_id is None:
            raise RuntimeError("no candidate was explicitly selected")
        return self.evaluation(self.selected_candidate_id)

    def evaluation(self, candidate_id: str) -> PidCandidateEvaluation:
        for value in self.evaluations:
            if value.candidate.candidate_id == str(candidate_id):
                return value
        raise KeyError("unknown candidate: {}".format(candidate_id))


def empirical_upper_cvar(values: Sequence[float], level: float) -> float:
    """Exact equal-weight empirical upper-tail CVaR, including fractional mass."""

    samples = np.asarray(values, dtype=float)
    selected_level = float(level)
    if (
        samples.ndim != 1
        or samples.size < 1
        or np.any(~np.isfinite(samples))
        or not np.isfinite(selected_level)
        or not 0.0 <= selected_level < 1.0
    ):
        raise ValueError("CVaR inputs are invalid")
    ordered = np.sort(samples)
    count = ordered.size
    left = np.arange(count, dtype=float) / count
    right = np.arange(1, count + 1, dtype=float) / count
    mass = np.maximum(0.0, right - np.maximum(left, selected_level))
    return float(np.dot(mass, ordered) / (1.0 - selected_level))


def time_integrated_error_metrics(
    times: Sequence[float],
    position_error: np.ndarray,
    orientation_error_rotation_vector: np.ndarray,
) -> TrajectoryMetricValues:
    """Compute four separate physical metrics on an irregular time grid."""

    selected_times = np.asarray(times, dtype=float)
    position = np.asarray(position_error, dtype=float)
    orientation = np.asarray(orientation_error_rotation_vector, dtype=float)
    if (
        selected_times.ndim != 1
        or selected_times.size < 2
        or np.any(~np.isfinite(selected_times))
        or np.any(np.diff(selected_times) <= 0.0)
        or position.shape != (selected_times.size, 3)
        or orientation.shape != (selected_times.size, 3)
        or np.any(~np.isfinite(position))
        or np.any(~np.isfinite(orientation))
    ):
        raise ValueError("time and error paths must be finite and aligned")
    duration = selected_times[-1] - selected_times[0]
    position_norm_squared = np.sum(position * position, axis=1)
    orientation_norm_squared = np.sum(orientation * orientation, axis=1)
    return TrajectoryMetricValues(
        position_rmse=float(
            np.sqrt(np.trapz(position_norm_squared, selected_times) / duration)
        ),
        orientation_rmse=float(
            np.sqrt(np.trapz(orientation_norm_squared, selected_times) / duration)
        ),
        maximum_position_error=float(np.sqrt(np.max(position_norm_squared))),
        maximum_orientation_error=float(
            np.sqrt(np.max(orientation_norm_squared))
        ),
    )


def correction_zero_coverage(
    correction_translation: np.ndarray,
    correction_rotation_vector: np.ndarray,
    forecast_completed: np.ndarray,
    interval: float = CORRECTION_COVERAGE_INTERVAL,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return zero-correction coverage for raw-member component intervals.

    For each time/component, the central ``interval`` raw-member percentile
    interval is formed from completed forecasts only.  Coverage is the
    fraction of those component intervals containing the desired correction
    value zero.  It is a calibration diagnostic, never a Pareto objective.
    """

    translation = np.asarray(correction_translation, dtype=float)
    rotation = np.asarray(correction_rotation_vector, dtype=float)
    completed = np.asarray(forecast_completed, dtype=bool)
    selected_interval = float(interval)
    if (
        translation.ndim != 3
        or translation.shape[2] != 3
        or rotation.shape != translation.shape
        or completed.shape != (translation.shape[0],)
        or not np.isfinite(selected_interval)
        or not 0.0 < selected_interval < 1.0
    ):
        raise ValueError("correction coverage arrays are invalid")
    if np.any(~np.isfinite(translation[completed])) or np.any(
        ~np.isfinite(rotation[completed])
    ):
        raise ValueError("completed correction paths must be finite")
    if np.any(~np.isnan(translation[~completed])) or np.any(
        ~np.isnan(rotation[~completed])
    ):
        raise ValueError("failed correction paths must be NaN")
    if not np.any(completed):
        return None, None, None
    tail = 50.0 * (1.0 - selected_interval)

    def component_coverage(path: np.ndarray) -> Tuple[float, np.ndarray]:
        lower, upper = np.percentile(
            path[completed], (tail, 100.0 - tail), axis=0
        )
        covered = (lower <= 0.0) & (upper >= 0.0)
        return float(np.mean(covered)), covered

    translation_coverage, translation_covered = component_coverage(
        translation
    )
    rotation_coverage, rotation_covered = component_coverage(rotation)
    combined_coverage = float(
        np.mean(
            np.concatenate(
                (translation_covered, rotation_covered), axis=1
            )
        )
    )
    return translation_coverage, rotation_coverage, combined_coverage


def _pose_error_paths(
    scenario: CounterfactualBagScenario,
    trajectory: ClosedLoopTrajectory,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference_position = scenario.reference_position
    reference_orientation = scenario.reference_orientation_xyzw
    position_error = trajectory.position - reference_position
    orientation_error = np.empty_like(position_error)
    for index in range(scenario.times.size):
        reference_rotation = quaternion_to_matrix(reference_orientation[index])
        predicted_rotation = quaternion_to_matrix(
            trajectory.orientation_xyzw[index]
        )
        orientation_error[index] = rotation_vector_from_matrix(
            reference_rotation.T @ predicted_rotation
        )
    correction_translation, correction_rotation = correction_transform_path(
        reference_position,
        reference_orientation,
        trajectory.position,
        trajectory.orientation_xyzw,
    )
    return (
        position_error,
        orientation_error,
        correction_translation,
        correction_rotation,
    )


def _metric_summary(values: np.ndarray, level: float) -> MetricSummary:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return MetricSummary(None, None, 0)
    return MetricSummary(
        float(np.mean(finite)), empirical_upper_cvar(finite, level), finite.size
    )


def summarize_member_metrics(
    metric_values: Mapping[str, np.ndarray],
    forecast_completed: np.ndarray,
    cvar_level: float,
    position_threshold_exceeded: Optional[np.ndarray] = None,
    orientation_threshold_exceeded: Optional[np.ndarray] = None,
) -> CandidateSummary:
    """Summarize members without folding numerical failures into thresholds."""

    completed = np.asarray(forecast_completed, dtype=bool)
    if completed.ndim != 1 or completed.size < 1:
        raise ValueError("forecast_completed must be a non-empty vector")
    metrics = {}
    for name in PHYSICAL_METRICS:
        if name not in metric_values:
            raise ValueError("missing physical metric {}".format(name))
        values = np.asarray(metric_values[name], dtype=float)
        if values.shape != completed.shape:
            raise ValueError("physical metric member axes must align")
        if np.any(~np.isfinite(values[completed])) or np.any(
            ~np.isnan(values[~completed])
        ):
            raise ValueError("failed member metrics must be NaN only")
        metrics[name] = _metric_summary(values[completed], cvar_level)

    def threshold_rate(value: Optional[np.ndarray]) -> Optional[float]:
        if value is None:
            return None
        selected = np.asarray(value, dtype=float)
        if selected.shape != completed.shape:
            raise ValueError("threshold member axes must align")
        if np.any(~np.isnan(selected[~completed])) or np.any(
            ~np.isin(selected[completed], (0.0, 1.0))
        ):
            raise ValueError("thresholds cannot include numerical failures")
        if not np.any(completed):
            return None
        return float(np.mean(selected[completed]))

    return CandidateSummary(
        metrics=metrics,
        position_threshold_exceedance=threshold_rate(
            position_threshold_exceeded
        ),
        orientation_threshold_exceedance=threshold_rate(
            orientation_threshold_exceeded
        ),
        forecast_completion=float(np.mean(completed)),
        numerical_failure_count=int(np.count_nonzero(~completed)),
    )


def bag_equal_aggregate(
    bag_summaries: Sequence[CandidateSummary],
) -> CandidateSummary:
    """Average only same-name, same-unit bag summaries with one vote per bag."""

    summaries = tuple(bag_summaries)
    if not summaries or any(
        not isinstance(value, CandidateSummary) for value in summaries
    ):
        raise ValueError("at least one bag summary is required")
    metrics = {}
    for name in PHYSICAL_METRICS:
        available = [
            value.metrics[name]
            for value in summaries
            if value.metrics[name].count > 0
        ]
        if not available:
            metrics[name] = MetricSummary(None, None, 0)
        else:
            metrics[name] = MetricSummary(
                float(np.mean([value.mean for value in available])),
                float(np.mean([value.upper_cvar for value in available])),
                len(available),
            )

    def aggregate_threshold(name: str) -> Optional[float]:
        values = [
            getattr(value, name)
            for value in summaries
            if getattr(value, name) is not None
        ]
        return None if not values else float(np.mean(values))

    return CandidateSummary(
        metrics=metrics,
        position_threshold_exceedance=aggregate_threshold(
            "position_threshold_exceedance"
        ),
        orientation_threshold_exceedance=aggregate_threshold(
            "orientation_threshold_exceedance"
        ),
        forecast_completion=float(
            np.mean([value.forecast_completion for value in summaries])
        ),
        numerical_failure_count=int(
            sum(value.numerical_failure_count for value in summaries)
        ),
    )


def log_gain_change(
    current: PidGainConfiguration, proposed: PidGainConfiguration
) -> float:
    """L2 log-ratio change; changing a zero/nonzero pattern is infinite."""

    if not isinstance(current, PidGainConfiguration) or not isinstance(
        proposed, PidGainConfiguration
    ):
        raise TypeError("gain change requires PID configurations")
    before = current.values
    after = proposed.values
    if np.any((before == 0.0) != (after == 0.0)):
        return float("inf")
    positive = before > 0.0
    if not np.any(positive):
        return 0.0
    return float(np.linalg.norm(np.log(after[positive] / before[positive])))


def _candidate_configuration(
    predictive_input: PosteriorPredictiveInput,
    candidates: Optional[Sequence[PidGainCandidate]],
) -> Tuple[PidGainCandidate, ...]:
    current = current_pid_candidate(predictive_input.current)
    supplied = tuple(() if candidates is None else candidates)
    if any(not isinstance(value, PidGainCandidate) for value in supplied):
        raise TypeError("candidates must be PidGainCandidate instances")
    by_id = {value.candidate_id: value for value in supplied}
    if len(by_id) != len(supplied):
        raise ValueError("candidate IDs must be unique")
    if "current" in by_id:
        declared = by_id.pop("current")
        if declared.source != "current" or not np.array_equal(
            declared.configuration.values, predictive_input.current.values
        ):
            raise ValueError("current candidate must equal the recorded baseline")
    ordered = (current,) + tuple(
        value for value in supplied if value.candidate_id != "current"
    )
    proposals = predictive_input.proposal_ensemble
    for candidate in ordered[1:]:
        if candidate.source == "current":
            raise ValueError("only candidate ID current may have current source")
        if candidate.source == "member-derived":
            index = proposals.member_index(candidate.source_member_id)
            if (
                candidate.source_mode_id != predictive_input.selected_mode_id
                or not np.array_equal(
                    candidate.configuration.values,
                    proposals.exact_gain_values[index],
                )
            ):
                raise ValueError(
                    "member-derived candidate must match its raw proposal member"
                )
    return ordered


def _threshold_flags(
    thresholds: ErrorThresholds,
    metrics: Mapping[str, np.ndarray],
    completed: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    def selected(value: Optional[float], metric: str) -> Optional[np.ndarray]:
        if value is None:
            return None
        result = np.full(completed.shape, np.nan, dtype=float)
        result[completed] = (
            np.asarray(metrics[metric])[completed] > float(value)
        ).astype(float)
        return result

    return (
        selected(thresholds.position, thresholds.position_metric),
        selected(thresholds.orientation, thresholds.orientation_metric),
    )


def _evaluate_candidate_bag(
    predictive_input: PosteriorPredictiveInput,
    candidate: PidGainCandidate,
    scenario: CounterfactualBagScenario,
    cvar_level: float,
    thresholds: ErrorThresholds,
    member_boundary,
    cancellation_checkpoint,
) -> BagCandidateEvaluation:
    members = predictive_input.member_id.size
    samples = scenario.times.size
    prediction_position = np.full((members, samples, 3), np.nan)
    prediction_orientation = np.full((members, samples, 4), np.nan)
    correction_translation = np.full((members, samples, 3), np.nan)
    correction_rotation = np.full((members, samples, 3), np.nan)
    position_error = np.full((members, samples, 3), np.nan)
    orientation_error = np.full((members, samples, 3), np.nan)
    metric_values = {
        name: np.full((members,), np.nan) for name in PHYSICAL_METRICS
    }
    completed = np.zeros(members, dtype=bool)
    reasons = [""] * members
    trajectories = []
    controller_configuration = apply_pid_gain_configuration(
        scenario.controller_configuration, candidate.configuration
    )
    for member_index, member_id in enumerate(predictive_input.member_id):
        cancellation_checkpoint()
        try:
            residual = (
                scenario.posterior_residual_wrench[member_index]
                if scenario.residual_policy == "posterior_replay"
                else np.zeros((samples - 1, 6), dtype=float)
            )
            actuator_parameters = replace(
                scenario.actuator_parameters,
                delay=float(
                    predictive_input.proposal_ensemble.constant_delay[
                        member_index
                    ]
                ),
            )
            trajectory = simulate_closed_loop(
                times=scenario.times,
                references=scenario.references,
                initial_state=scenario.initial_states[member_index],
                initial_controller_state=(
                    scenario.initial_controller_states[member_index]
                ),
                controller=GrapeController(
                    controller_configuration,
                    scenario.controller_nominal_parameters,
                    scenario.controller_geometry,
                    articulated_model=GrapeArticulatedModel(),
                ),
                plant=FullSixDofPlant(
                    predictive_input.physical_parameter_members[member_index],
                    scenario.plant_geometry,
                ),
                actuator_parameters=actuator_parameters,
                initial_actuator_state=(
                    scenario.initial_actuator_states[member_index]
                ),
                interval_residual_wrench=residual,
            )
            (
                member_position_error,
                member_orientation_error,
                member_correction_translation,
                member_correction_rotation,
            ) = _pose_error_paths(scenario, trajectory)
            metric = time_integrated_error_metrics(
                scenario.times,
                member_position_error,
                member_orientation_error,
            )
            prediction_position[member_index] = trajectory.position
            prediction_orientation[member_index] = trajectory.orientation_xyzw
            correction_translation[member_index] = member_correction_translation
            correction_rotation[member_index] = member_correction_rotation
            position_error[member_index] = member_position_error
            orientation_error[member_index] = member_orientation_error
            for name in PHYSICAL_METRICS:
                metric_values[name][member_index] = getattr(metric, name)
            completed[member_index] = True
            trajectories.append(trajectory)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:  # forecast failures are primary output
            trajectories.append(None)
            reasons[member_index] = "{}: {}".format(
                type(error).__name__, str(error)
            )
        finally:
            member_boundary(candidate, scenario, int(member_id))
    position_threshold, orientation_threshold = _threshold_flags(
        thresholds, metric_values, completed
    )
    summary = summarize_member_metrics(
        metric_values,
        completed,
        cvar_level,
        position_threshold,
        orientation_threshold,
    )
    (
        translation_coverage,
        rotation_coverage,
        transform_coverage,
    ) = correction_zero_coverage(
        correction_translation,
        correction_rotation,
        completed,
    )
    return BagCandidateEvaluation(
        bag_id=scenario.bag_id,
        member_id=predictive_input.member_id,
        trajectories=tuple(trajectories),
        prediction_position=prediction_position,
        prediction_orientation_xyzw=prediction_orientation,
        correction_translation=correction_translation,
        correction_rotation_vector=correction_rotation,
        position_error=position_error,
        orientation_error_rotation_vector=orientation_error,
        position_rmse=metric_values["position_rmse"],
        orientation_rmse=metric_values["orientation_rmse"],
        maximum_position_error=metric_values["maximum_position_error"],
        maximum_orientation_error=metric_values[
            "maximum_orientation_error"
        ],
        forecast_completed=completed,
        failure_reason=tuple(reasons),
        position_threshold_exceeded=position_threshold,
        orientation_threshold_exceeded=orientation_threshold,
        summary=summary,
        residual_policy=scenario.residual_policy,
        correction_translation_zero_coverage=translation_coverage,
        correction_rotation_zero_coverage=rotation_coverage,
        correction_transform_zero_coverage=transform_coverage,
    )


def _summary_objectives(
    summary: CandidateSummary,
    thresholds: ErrorThresholds,
) -> Tuple[float, ...]:
    values = []
    for name in PHYSICAL_METRICS:
        metric = summary.metrics[name]
        values.extend(
            (
                float("inf") if metric.mean is None else metric.mean,
                (
                    float("inf")
                    if metric.upper_cvar is None
                    else metric.upper_cvar
                ),
            )
        )
    if thresholds.position_configured:
        values.append(
            float("inf")
            if summary.position_threshold_exceedance is None
            else summary.position_threshold_exceedance
        )
    if thresholds.orientation_configured:
        values.append(
            float("inf")
            if summary.orientation_threshold_exceedance is None
            else summary.orientation_threshold_exceedance
        )
    values.extend(
        (1.0 - summary.forecast_completion, summary.numerical_failure_count)
    )
    return tuple(values)


def _pareto_objectives(
    evaluation: PidCandidateEvaluation, thresholds: ErrorThresholds
) -> np.ndarray:
    return np.asarray(
        _summary_objectives(evaluation.aggregate, thresholds)
        + (evaluation.log_gain_change,),
        dtype=float,
    )


def pareto_dominated_flags(
    evaluations: Sequence[PidCandidateEvaluation],
    thresholds: ErrorThresholds,
) -> np.ndarray:
    """Return minimization Pareto flags without cross-metric weighting."""

    values = tuple(evaluations)
    if not values:
        raise ValueError("at least one candidate evaluation is required")
    objectives = np.asarray(
        [_pareto_objectives(value, thresholds) for value in values]
    )
    dominated = np.zeros(len(values), dtype=bool)
    for candidate_index in range(len(values)):
        for other_index in range(len(values)):
            if candidate_index == other_index:
                continue
            if np.all(objectives[other_index] <= objectives[candidate_index]) and (
                np.any(objectives[other_index] < objectives[candidate_index])
            ):
                dominated[candidate_index] = True
                break
    return dominated


def _componentwise_improves_current(
    candidate: PidCandidateEvaluation,
    current: PidCandidateEvaluation,
    thresholds: ErrorThresholds,
) -> bool:
    proposed = np.asarray(
        _summary_objectives(candidate.aggregate, thresholds), dtype=float
    )
    baseline = np.asarray(
        _summary_objectives(current.aggregate, thresholds), dtype=float
    )
    tolerance = 1.0e-12
    return bool(
        np.all(proposed <= baseline + tolerance)
        and np.any(proposed < baseline - tolerance)
    )


def evaluate_pid_proposals(
    predictive_input: PosteriorPredictiveInput,
    candidates: Optional[Sequence[PidGainCandidate]] = None,
    cvar_level: float = 0.90,
    thresholds: Optional[ErrorThresholds] = None,
    selected_candidate_id: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
    progress_run_id: str = "pid-proposal-evaluation",
    cancellation_token: Optional[CancellationToken] = None,
) -> PosteriorPredictiveDecision:
    """Run every exact candidate × bag × raw-member closed-loop forecast."""

    if not isinstance(predictive_input, PosteriorPredictiveInput):
        raise TypeError("predictive_input has the wrong type")
    level = float(cvar_level)
    if not np.isfinite(level) or not 0.0 <= level < 1.0:
        raise ValueError("cvar_level must be in [0, 1)")
    selected_thresholds = ErrorThresholds() if thresholds is None else thresholds
    if not isinstance(selected_thresholds, ErrorThresholds):
        raise TypeError("thresholds has the wrong type")
    selected_candidates = _candidate_configuration(
        predictive_input, candidates
    )
    total_units = (
        len(selected_candidates)
        * len(predictive_input.bags)
        * predictive_input.member_id.size
    )
    tracker = ProgressTracker(
        progress_run_id,
        total_units,
        callback=progress_callback,
        cancellation_token=cancellation_token,
    )
    completed_units = 0

    def member_boundary(candidate, scenario, member_id):
        nonlocal completed_units
        completed_units += 1
        tracker.emit(
            completed_units,
            stage_id="candidate_member_bag_forecast",
            stage_label="Evaluating PID candidate {}".format(
                candidate.candidate_id
            ),
            bag_id=scenario.bag_id,
            member_id=member_id,
            message="candidate {} | bag {} | member {}".format(
                candidate.candidate_id, scenario.bag_id, member_id
            ),
        )

    evaluations = []
    for candidate in selected_candidates:
        bag_evaluations = tuple(
            _evaluate_candidate_bag(
                predictive_input,
                candidate,
                scenario,
                level,
                selected_thresholds,
                member_boundary,
                tracker.checkpoint,
            )
            for scenario in predictive_input.bags
        )
        evaluations.append(
            PidCandidateEvaluation(
                candidate=candidate,
                bags=bag_evaluations,
                aggregate=bag_equal_aggregate(
                    [value.summary for value in bag_evaluations]
                ),
                log_gain_change=log_gain_change(
                    predictive_input.current, candidate.configuration
                ),
            )
        )
    dominated = pareto_dominated_flags(evaluations, selected_thresholds)
    current = evaluations[0]
    evaluations = [
        replace(
            value,
            pareto_dominated=bool(dominated[index]),
            improves_current=(
                index > 0
                and _componentwise_improves_current(
                    value, current, selected_thresholds
                )
            ),
        )
        for index, value in enumerate(evaluations)
    ]
    selected_id = (
        None if selected_candidate_id is None else str(selected_candidate_id)
    )
    if selected_id is not None and selected_id not in {
        value.candidate.candidate_id for value in evaluations
    }:
        raise ValueError("selected candidate was not evaluated")
    recommendation = False
    recommended_id = ""
    if selected_id is None:
        rejection = "No proposal was explicitly selected; no automatic representative is used."
    elif selected_id == "current":
        rejection = "The current baseline is not a new proposal."
    else:
        selected_evaluation = next(
            value
            for value in evaluations
            if value.candidate.candidate_id == selected_id
        )
        if selected_evaluation.pareto_dominated:
            rejection = "The explicitly selected proposal is Pareto dominated."
        elif not selected_evaluation.improves_current:
            rejection = (
                "The explicitly selected proposal does not improve current "
                "under the documented componentwise rule."
            )
        else:
            recommendation = True
            recommended_id = selected_id
            rejection = ""
    return PosteriorPredictiveDecision(
        predictive_input=predictive_input,
        evaluations=tuple(evaluations),
        cvar_level=level,
        thresholds=selected_thresholds,
        selected_candidate_id=selected_id,
        recommendation_available=recommendation,
        recommended_candidate_id=recommended_id,
        rejection_reason=rejection,
    )


def input_from_joint_assimilation(
    result,
    current: PidGainConfiguration,
    selected_mode_id: str = "actuator_wiring_nominal",
    residual_policy: Any = "posterior_replay",
    provenance: Optional[Mapping[str, str]] = None,
) -> PosteriorPredictiveInput:
    """Build member-aligned scenarios directly from one joint smoothing result."""

    from grape_param_estim.joint_assimilation import JointAssimilationResult

    if not isinstance(result, JointAssimilationResult):
        raise TypeError("result must be JointAssimilationResult")
    if not isinstance(current, PidGainConfiguration):
        raise TypeError("current must be PidGainConfiguration")
    posterior = result.posterior
    shared = posterior.shared_parameter_ensemble
    first_problem = result.problem.bags[0].problem.strong_problem
    physical = tuple(
        first_problem.parameter_chart.decode(value)
        for value in shared.physical_parameter_coordinates
    )
    modes = tuple(str(selected_mode_id) for _value in posterior.member_id)
    proposals = derive_pid_proposal_ensemble(
        member_id=posterior.member_id,
        physical_parameter_members=physical,
        constant_delay=shared.constant_delay,
        source_mode_id=modes,
        controller_nominal_parameters=first_problem.controller_parameters,
        geometry=first_problem.geometry,
        current=current,
    )
    if isinstance(residual_policy, Mapping):
        policies = {
            str(key): str(value) for key, value in residual_policy.items()
        }
    else:
        policies = {
            value.bag_id: str(residual_policy)
            for value in result.prepared_flights
        }
    scenarios = []
    for prepared in result.prepared_flights:
        bag = posterior.bag(prepared.bag_id)
        strong = prepared.joint_bag_problem.problem.strong_problem
        scenarios.append(
            CounterfactualBagScenario(
                bag_id=prepared.bag_id,
                times=prepared.episode.observations.times,
                references=prepared.episode.references,
                initial_states=bag.initial_states,
                initial_controller_states=bag.initial_controller_states,
                initial_actuator_states=bag.initial_actuator_states,
                posterior_residual_wrench=bag.residual_wrench_ensemble,
                controller_configuration=strong.controller_configuration,
                controller_nominal_parameters=strong.controller_parameters,
                controller_geometry=strong.geometry,
                plant_geometry=strong.geometry,
                actuator_parameters=strong.actuator_parameters,
                residual_policy=policies[prepared.bag_id],
                provenance=(
                    ("bag_id", prepared.bag_id),
                    (
                        "bag_sha256",
                        prepared.episode.provenance.bag_sha256,
                    ),
                ),
            )
        )
    return PosteriorPredictiveInput(
        selected_mode_id=str(selected_mode_id),
        physical_parameter_members=physical,
        proposal_ensemble=proposals,
        bags=tuple(scenarios),
        provenance=_normalised_provenance(provenance),
    )


def _optional_scalar(value: Optional[float]) -> np.ndarray:
    return np.asarray((np.nan if value is None else float(value),))


def _summary_arrays(
    evaluations: Sequence[PidCandidateEvaluation],
    bag_ids: Sequence[str],
) -> Dict[str, np.ndarray]:
    candidates = len(evaluations)
    bags = len(bag_ids)
    result: Dict[str, np.ndarray] = {}
    for metric_name in PHYSICAL_METRICS:
        per_bag_mean = np.full((candidates, bags), np.nan)
        per_bag_cvar = np.full((candidates, bags), np.nan)
        aggregate_mean = np.full(candidates, np.nan)
        aggregate_cvar = np.full(candidates, np.nan)
        for candidate_index, evaluation in enumerate(evaluations):
            for bag_index, bag_id in enumerate(bag_ids):
                summary = evaluation.bag(bag_id).summary.metrics[metric_name]
                if summary.mean is not None:
                    per_bag_mean[candidate_index, bag_index] = summary.mean
                    per_bag_cvar[candidate_index, bag_index] = summary.upper_cvar
            aggregate = evaluation.aggregate.metrics[metric_name]
            if aggregate.mean is not None:
                aggregate_mean[candidate_index] = aggregate.mean
                aggregate_cvar[candidate_index] = aggregate.upper_cvar
        result["per_bag_{}_mean".format(metric_name)] = per_bag_mean
        result["per_bag_{}_upper_cvar".format(metric_name)] = per_bag_cvar
        result["aggregate_{}_mean".format(metric_name)] = aggregate_mean
        result["aggregate_{}_upper_cvar".format(metric_name)] = aggregate_cvar
    for field_name in (
        "correction_translation_zero_coverage",
        "correction_rotation_zero_coverage",
        "correction_transform_zero_coverage",
    ):
        coverage = np.full((candidates, bags), np.nan)
        for candidate_index, evaluation in enumerate(evaluations):
            for bag_index, bag_id in enumerate(bag_ids):
                value = getattr(evaluation.bag(bag_id), field_name)
                if value is not None:
                    coverage[candidate_index, bag_index] = value
        result["per_bag_{}".format(field_name)] = coverage
    result["per_bag_forecast_completion"] = np.asarray(
        [
            [
                evaluation.bag(bag_id).summary.forecast_completion
                for bag_id in bag_ids
            ]
            for evaluation in evaluations
        ]
    )
    result["per_bag_numerical_failure_count"] = np.asarray(
        [
            [
                evaluation.bag(bag_id).summary.numerical_failure_count
                for bag_id in bag_ids
            ]
            for evaluation in evaluations
        ],
        dtype=np.int64,
    )
    for threshold_name in ("position", "orientation"):
        field_name = "{}_threshold_exceedance".format(threshold_name)
        per_bag = np.full((candidates, bags), np.nan)
        aggregate = np.full(candidates, np.nan)
        for candidate_index, evaluation in enumerate(evaluations):
            aggregate_value = getattr(evaluation.aggregate, field_name)
            if aggregate_value is not None:
                aggregate[candidate_index] = aggregate_value
            for bag_index, bag_id in enumerate(bag_ids):
                value = getattr(evaluation.bag(bag_id).summary, field_name)
                if value is not None:
                    per_bag[candidate_index, bag_index] = value
        result["per_bag_{}".format(field_name)] = per_bag
        result["aggregate_{}".format(field_name)] = aggregate
    return result


def pid_proposal_evaluation_manifest(
    predictive_input: PosteriorPredictiveInput,
    evaluation_id: str,
    source_run_id: str,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the writing manifest so a worker can publish it before forecast."""

    if not isinstance(predictive_input, PosteriorPredictiveInput):
        raise TypeError("predictive_input has the wrong type")
    evaluation_identifier = str(evaluation_id)
    source_identifier = str(source_run_id)
    created = (
        datetime.now(timezone.utc).isoformat()
        if created_at is None
        else str(created_at)
    )
    if not evaluation_identifier or not source_identifier or not created:
        raise ValueError("evaluation, source run, and creation IDs cannot be empty")
    bag_ids = tuple(value.bag_id for value in predictive_input.bags)
    return {
        "schema": PID_PROPOSAL_EVALUATION_SCHEMA,
        "evaluation_id": evaluation_identifier,
        "source_run_id": source_identifier,
        "created_at": created,
        "selected_bag_ids": list(bag_ids),
        "artifacts": {
            "proposal_ensemble": "proposal_ensemble.npz",
            "summary": "summary.npz",
            "proposed_yaml": "proposed_GimbalrotorControl.yaml",
            "proposed_diff_yaml": "proposed_GimbalrotorControl.diff.yaml",
            "bags": {
                bag_id: "bags/{}.npz".format(bag_id) for bag_id in bag_ids
            },
        },
    }


def save_pid_proposal_evaluation(
    root: str,
    decision: PosteriorPredictiveDecision,
    evaluation_id: str,
    source_run_id: str,
    created_at: Optional[str] = None,
    yaml_candidate_id: Optional[str] = None,
    bundle_started: bool = False,
    cancellation_token: Optional[CancellationToken] = None,
) -> Path:
    """Write the plan-v1 directory bundle and atomically publish completion."""

    if not isinstance(decision, PosteriorPredictiveDecision):
        raise TypeError("decision has the wrong type")
    evaluation_identifier = str(evaluation_id)
    source_identifier = str(source_run_id)
    if not evaluation_identifier or not source_identifier:
        raise ValueError("evaluation and source run IDs cannot be empty")
    destination = Path(root).expanduser().resolve()
    bag_ids = tuple(value.bag_id for value in decision.predictive_input.bags)
    current_provenance = decision.predictive_input.current.provenance
    if current_provenance is None:
        raise ValueError(
            "current PID must retain recorded controller snapshot provenance"
        )
    if current_provenance.bag_id not in bag_ids:
        raise ValueError("current PID baseline bag must be a selected bag")
    cancellation = (
        CancellationToken()
        if cancellation_token is None
        else cancellation_token
    )
    if not isinstance(cancellation, CancellationToken):
        raise TypeError("cancellation_token must be CancellationToken")
    if not isinstance(bundle_started, bool):
        raise TypeError("bundle_started must be boolean")
    if bundle_started:
        existing = read_manifest(destination)
        manifest = pid_proposal_evaluation_manifest(
            decision.predictive_input,
            evaluation_identifier,
            source_identifier,
            existing.get("created_at") if created_at is None else created_at,
        )
        if existing.get("status") != WRITING_STATUS or any(
            existing.get(key) != value for key, value in manifest.items()
        ):
            raise ValueError(
                "existing writing manifest does not match PID evaluation"
            )
    else:
        manifest = pid_proposal_evaluation_manifest(
            decision.predictive_input,
            evaluation_identifier,
            source_identifier,
            created_at,
        )
        begin_bundle(destination, manifest)

    def checkpoint() -> None:
        try:
            cancellation.raise_if_cancelled()
        except ProgressCancelled as error:
            mark_bundle_cancelled(destination, error.reason)
            raise

    checkpoint()
    (destination / "bags").mkdir(parents=True, exist_ok=True)
    proposals = decision.predictive_input.proposal_ensemble
    range_50, range_95 = proposals.percentile_ranges()
    np.savez_compressed(
        str(destination / "proposal_ensemble.npz"),
        source_run_id=np.asarray((source_identifier,)),
        source_member_id=proposals.member_id,
        proposal_source_member_id=proposals.source_member_id,
        source_mode_id=np.asarray(proposals.source_mode_id),
        xy_scale=proposals.group_scales[:, 0],
        z_scale=proposals.group_scales[:, 1],
        roll_pitch_scale=proposals.group_scales[:, 2],
        yaw_scale=proposals.group_scales[:, 3],
        proposed_pid=proposals.exact_gain_values,
        current_pid=proposals.current.values,
        constant_delay=proposals.constant_delay,
        acceleration_response=proposals.acceleration_response,
        proposal_range_50=range_50,
        proposal_range_95=range_95,
    )
    checkpoint()
    evaluations = decision.evaluations
    candidate_ids = np.asarray(
        [value.candidate.candidate_id for value in evaluations]
    )
    candidate_sources = np.asarray(
        [value.candidate.source for value in evaluations]
    )
    candidate_pid = np.asarray(
        [value.candidate.configuration.values for value in evaluations]
    )
    comparison = [
        PidGainComparison.from_configurations(
            decision.predictive_input.current, value.candidate.configuration
        )
        for value in evaluations
    ]
    candidate_count = len(evaluations)
    bag_count = len(bag_ids)
    member_count = decision.predictive_input.member_id.size
    raw_metrics = {
        name: np.asarray(
            [
                [getattr(value.bag(bag_id), name) for bag_id in bag_ids]
                for value in evaluations
            ]
        )
        for name in PHYSICAL_METRICS
    }
    completion = np.asarray(
        [
            [value.bag(bag_id).forecast_completed for bag_id in bag_ids]
            for value in evaluations
        ]
    )
    reasons = np.asarray(
        [
            [value.bag(bag_id).failure_reason for bag_id in bag_ids]
            for value in evaluations
        ]
    )
    position_threshold = np.full(
        (candidate_count, bag_count, member_count), np.nan
    )
    orientation_threshold = np.full_like(position_threshold, np.nan)
    for candidate_index, evaluation in enumerate(evaluations):
        for bag_index, bag_id in enumerate(bag_ids):
            bag = evaluation.bag(bag_id)
            if bag.position_threshold_exceeded is not None:
                position_threshold[candidate_index, bag_index] = (
                    bag.position_threshold_exceeded
                )
            if bag.orientation_threshold_exceeded is not None:
                orientation_threshold[candidate_index, bag_index] = (
                    bag.orientation_threshold_exceeded
                )
    summary_payload = {
        "source_run_id": np.asarray((source_identifier,)),
        "source_member_id": decision.predictive_input.member_id,
        "source_mode_id": np.asarray(
            decision.predictive_input.proposal_ensemble.source_mode_id
        ),
        "bag_id": np.asarray(bag_ids),
        "candidate_id": candidate_ids,
        "candidate_source": candidate_sources,
        "candidate_source_member_id": np.asarray(
            [value.candidate.source_member_id for value in evaluations],
            dtype=np.int64,
        ),
        "candidate_source_mode_id": np.asarray(
            [value.candidate.source_mode_id for value in evaluations]
        ),
        "current_pid": decision.predictive_input.current.values,
        "current_pid_baseline_bag_id": np.asarray(
            (current_provenance.bag_id,)
        ),
        "current_pid_snapshot_group": np.asarray(PID_GROUPS),
        "current_pid_snapshot_topic": np.asarray(current_provenance.topics),
        "current_pid_snapshot_record_time": (
            current_provenance.record_times
        ),
        "current_pid_snapshot_source_kind": np.asarray(
            current_provenance.source_kinds
        ),
        "proposed_pid": candidate_pid,
        "difference": np.asarray([value.difference for value in comparison]),
        "ratio": np.asarray([value.ratio for value in comparison]),
        "ratio_configured": np.asarray(
            [value.ratio_configured for value in comparison]
        ),
        "member_bag_forecast_completion": completion,
        "member_bag_failure_reason": reasons,
        "member_bag_position_threshold_exceeded": position_threshold,
        "member_bag_orientation_threshold_exceeded": orientation_threshold,
        "position_threshold": _optional_scalar(decision.thresholds.position),
        "orientation_threshold": _optional_scalar(
            decision.thresholds.orientation
        ),
        "position_threshold_configured": np.asarray(
            (decision.thresholds.position_configured,), dtype=bool
        ),
        "orientation_threshold_configured": np.asarray(
            (decision.thresholds.orientation_configured,), dtype=bool
        ),
        "position_threshold_metric": np.asarray(
            (decision.thresholds.position_metric,)
        ),
        "orientation_threshold_metric": np.asarray(
            (decision.thresholds.orientation_metric,)
        ),
        "cvar_level": np.asarray((decision.cvar_level,)),
        "correction_coverage_interval": np.asarray(
            (CORRECTION_COVERAGE_INTERVAL,)
        ),
        "log_gain_change": np.asarray(
            [value.log_gain_change for value in evaluations]
        ),
        "forecast_completion": np.asarray(
            [value.aggregate.forecast_completion for value in evaluations]
        ),
        "numerical_failure_count": np.asarray(
            [
                value.aggregate.numerical_failure_count
                for value in evaluations
            ],
            dtype=np.int64,
        ),
        "pareto_dominated": np.asarray(
            [value.pareto_dominated for value in evaluations], dtype=bool
        ),
        "pareto_non_dominated": np.asarray(
            [value.pareto_non_dominated for value in evaluations], dtype=bool
        ),
        "improves_current": np.asarray(
            [value.improves_current for value in evaluations], dtype=bool
        ),
        "candidate_eligible": np.asarray(
            [
                index > 0
                and value.improves_current
                and not value.pareto_dominated
                for index, value in enumerate(evaluations)
            ],
            dtype=bool,
        ),
        "candidate_rejection_reason": np.asarray(
            [
                (
                    "current baseline"
                    if index == 0
                    else (
                        "Pareto dominated"
                        if value.pareto_dominated
                        else (
                            "does not improve current"
                            if not value.improves_current
                            else ""
                        )
                    )
                )
                for index, value in enumerate(evaluations)
            ]
        ),
        "selected_candidate_id": np.asarray(
            ("" if decision.selected_candidate_id is None else decision.selected_candidate_id,)
        ),
        "recommendation_available": np.asarray(
            (decision.recommendation_available,), dtype=bool
        ),
        "recommended_candidate_id": np.asarray(
            (decision.recommended_candidate_id,)
        ),
        "rejection_reason": np.asarray((decision.rejection_reason,)),
        "improvement_rule": np.asarray((decision.improvement_rule,)),
        "scenario_assumption": np.asarray(
            (decision.predictive_input.scenario_assumption,)
        ),
    }
    for name, values in raw_metrics.items():
        summary_payload["member_bag_{}".format(name)] = values
    summary_payload.update(_summary_arrays(evaluations, bag_ids))
    np.savez_compressed(str(destination / "summary.npz"), **summary_payload)
    checkpoint()
    for scenario in decision.predictive_input.bags:
        bag_evaluations = [value.bag(scenario.bag_id) for value in evaluations]
        np.savez_compressed(
            str(destination / "bags" / "{}.npz".format(scenario.bag_id)),
            member_id=decision.predictive_input.member_id,
            candidate_id=candidate_ids,
            times=scenario.times,
            reference_position=scenario.reference_position,
            reference_rpy=scenario.reference_rpy,
            prediction_position=np.asarray(
                [value.prediction_position for value in bag_evaluations]
            ),
            prediction_orientation_xyzw=np.asarray(
                [
                    value.prediction_orientation_xyzw
                    for value in bag_evaluations
                ]
            ),
            correction_translation=np.asarray(
                [value.correction_translation for value in bag_evaluations]
            ),
            correction_rotation_vector=np.asarray(
                [
                    value.correction_rotation_vector
                    for value in bag_evaluations
                ]
            ),
            position_error=np.asarray(
                [value.position_error for value in bag_evaluations]
            ),
            orientation_error_rotation_vector=np.asarray(
                [
                    value.orientation_error_rotation_vector
                    for value in bag_evaluations
                ]
            ),
            position_rmse=np.asarray(
                [value.position_rmse for value in bag_evaluations]
            ),
            orientation_rmse=np.asarray(
                [value.orientation_rmse for value in bag_evaluations]
            ),
            maximum_position_error=np.asarray(
                [
                    value.maximum_position_error
                    for value in bag_evaluations
                ]
            ),
            maximum_orientation_error=np.asarray(
                [
                    value.maximum_orientation_error
                    for value in bag_evaluations
                ]
            ),
            forecast_success=np.asarray(
                [value.forecast_completed for value in bag_evaluations]
            ),
            forecast_failure_reason=np.asarray(
                [value.failure_reason for value in bag_evaluations]
            ),
            residual_policy=np.asarray(
                [value.residual_policy for value in bag_evaluations]
            ),
            correction_coverage_interval=np.asarray(
                (CORRECTION_COVERAGE_INTERVAL,)
            ),
            correction_translation_zero_coverage=np.asarray(
                [
                    np.nan
                    if value.correction_translation_zero_coverage is None
                    else value.correction_translation_zero_coverage
                    for value in bag_evaluations
                ]
            ),
            correction_rotation_zero_coverage=np.asarray(
                [
                    np.nan
                    if value.correction_rotation_zero_coverage is None
                    else value.correction_rotation_zero_coverage
                    for value in bag_evaluations
                ]
            ),
            correction_transform_zero_coverage=np.asarray(
                [
                    np.nan
                    if value.correction_transform_zero_coverage is None
                    else value.correction_transform_zero_coverage
                    for value in bag_evaluations
                ]
            ),
        )
        checkpoint()
    selected_yaml_id = (
        yaml_candidate_id
        if yaml_candidate_id is not None
        else decision.selected_candidate_id
    )
    if selected_yaml_id is None:
        selected_yaml_id = "current"
    yaml_evaluation = decision.evaluation(str(selected_yaml_id))
    yaml_configuration = yaml_evaluation.candidate.configuration
    comparison_for_yaml = PidGainComparison.from_configurations(
        decision.predictive_input.current, yaml_configuration
    )
    (destination / "proposed_GimbalrotorControl.yaml").write_text(
        render_proposed_pid_yaml(yaml_configuration), encoding="utf-8"
    )
    checkpoint()
    (destination / "proposed_GimbalrotorControl.diff.yaml").write_text(
        render_pid_diff_yaml(comparison_for_yaml), encoding="utf-8"
    )
    checkpoint()
    mark_bundle_complete(destination)
    return destination


__all__ = [
    "COMPONENTWISE_IMPROVEMENT_RULE",
    "CORRECTION_COVERAGE_INTERVAL",
    "COUNTERFACTUAL_ASSUMPTION_PREFIX",
    "CounterfactualBagScenario",
    "ErrorThresholds",
    "MetricSummary",
    "NOT_CONFIGURED",
    "PHYSICAL_METRICS",
    "PidCandidateEvaluation",
    "PosteriorPredictiveDecision",
    "PosteriorPredictiveInput",
    "TrajectoryMetricValues",
    "bag_equal_aggregate",
    "correction_zero_coverage",
    "empirical_upper_cvar",
    "evaluate_pid_proposals",
    "input_from_joint_assimilation",
    "log_gain_change",
    "pareto_dominated_flags",
    "pid_proposal_evaluation_manifest",
    "save_pid_proposal_evaluation",
    "summarize_member_metrics",
    "time_integrated_error_metrics",
]
