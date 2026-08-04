"""Full closed-loop forecasts for posterior-driven PID evaluation.

The forecast starts once from a bag-specific initial condition and then runs
continuously over the complete selected interval.  It never replaces the
simulated state with observations.  Future model discrepancy is either zero
or a fresh realization from the estimated diagonal Q; an estimated historical
dynamics-residual sequence is not accepted by this API.
"""

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.closed_loop_stepper import (
    ClosedLoopStepper,
    ClosedLoopStepperState,
)
from grape_param_estim.controller import ControllerConfig, GrapeController
from grape_param_estim.controller_config import apply_pid_gain_configuration
from grape_param_estim.dynamics import FullSixDofPlant
from grape_param_estim.geometry import (
    euler_xyz_to_matrix,
    quaternion_to_matrix,
    so3_log,
)
from grape_param_estim.pid.metrics import ForecastMetrics
from grape_param_estim.pid.particle_search import (
    BODY_WRENCH_MODEL_DISCREPANCY,
    ModelDiscrepancyRealization,
    SPECIFIC_ACCELERATION_MODEL_DISCREPANCY,
)
from grape_param_estim.pid.proposal import PhysicalPlantSample, PidCandidate
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    ControllerState,
    GrapeGeometry,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


_NUMERICAL_FORECAST_ERRORS = (
    ArithmeticError,
    RuntimeError,
    ValueError,
    np.linalg.LinAlgError,
)


def _canonical_identifier(value: object, name: str) -> str:
    selected = str(value)
    if not selected or selected.strip() != selected or "\x00" in selected:
        raise ValueError("{} must be a canonical non-empty string".format(name))
    return selected


@dataclass(frozen=True)
class PidForecastInitialCondition:
    """One sample-aligned dynamic initial condition for one bag."""

    sample_id: str
    rigid_body_state: RigidBodyState
    controller_state: ControllerState
    actuator_state: Optional[ActuatorState]
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_id",
            _canonical_identifier(self.sample_id, "sample_id"),
        )
        object.__setattr__(
            self, "source", _canonical_identifier(self.source, "source")
        )
        if not isinstance(self.rigid_body_state, RigidBodyState):
            raise TypeError("rigid_body_state must be RigidBodyState")
        if not isinstance(self.controller_state, ControllerState):
            raise TypeError("controller_state must be ControllerState")
        if self.actuator_state is not None and not isinstance(
            self.actuator_state, ActuatorState
        ):
            raise TypeError("actuator_state must be ActuatorState or None")


@dataclass(frozen=True)
class PidForecastScenario:
    """One recorded-reference scenario evaluated against every plant sample.

    ``initial_conditions`` is explicitly sample-aligned.  A caller that uses
    the same selected-mode MAP initial state for all posterior draws must
    duplicate it with the explicit ``shared_selected_mode_map_initial``
    provenance rather than relying on an implicit fallback.
    """

    bag_id: str
    times: np.ndarray
    references: Tuple[ReferenceState, ...]
    initial_conditions: Tuple[PidForecastInitialCondition, ...]
    controller_configuration: ControllerConfig
    controller_nominal_parameters: VehicleParameters
    controller_geometry: GrapeGeometry
    plant_geometry: GrapeGeometry
    actuator_parameters: ActuatorParameters
    provenance: Tuple[Tuple[str, str], ...] = tuple()
    articulated_model: Optional[GrapeArticulatedModel] = None

    def __post_init__(self) -> None:
        bag_id = _canonical_identifier(self.bag_id, "bag_id")
        times = np.asarray(self.times, dtype=float)
        references = tuple(self.references)
        initial = tuple(self.initial_conditions)
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
            not initial
            or any(
                not isinstance(value, PidForecastInitialCondition)
                for value in initial
            )
            or len({value.sample_id for value in initial}) != len(initial)
        ):
            raise ValueError(
                "initial_conditions must have unique posterior sample IDs"
            )
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
        if self.articulated_model is not None and not isinstance(
            self.articulated_model, GrapeArticulatedModel
        ):
            raise TypeError("articulated_model has the wrong type")
        provenance = tuple(
            (
                _canonical_identifier(key, "provenance key"),
                str(value),
            )
            for key, value in self.provenance
        )
        if len({key for key, _value in provenance}) != len(provenance):
            raise ValueError("scenario provenance keys must be unique")
        selected_times = times.copy()
        selected_times.setflags(write=False)
        object.__setattr__(self, "bag_id", bag_id)
        object.__setattr__(self, "times", selected_times)
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "initial_conditions", initial)
        object.__setattr__(self, "provenance", provenance)

    def initial_condition(self, sample_id: str) -> PidForecastInitialCondition:
        identifier = str(sample_id)
        matches = tuple(
            value
            for value in self.initial_conditions
            if value.sample_id == identifier
        )
        if len(matches) != 1:
            raise KeyError(
                "scenario has no unique initial condition for sample {!r}".format(
                    identifier
                )
            )
        return matches[0]


@dataclass(frozen=True)
class PidForecastTrace:
    """Finite trajectory prefix and actuator-limit activity from one forecast."""

    times: np.ndarray
    position: np.ndarray
    orientation_xyzw: np.ndarray
    thrust_saturated: np.ndarray
    gimbal_saturated: np.ndarray
    completed_intervals: int
    requested_intervals: int
    failure_reason: str

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        position = np.asarray(self.position, dtype=float)
        orientation = np.asarray(self.orientation_xyzw, dtype=float)
        thrust = np.asarray(self.thrust_saturated, dtype=bool)
        gimbal = np.asarray(self.gimbal_saturated, dtype=bool)
        count = times.size
        if (
            times.ndim != 1
            or count < 1
            or np.any(~np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
            or position.shape != (count, 3)
            or orientation.shape != (count, 4)
            or thrust.shape != (max(count - 1, 0), 4)
            or gimbal.shape != (max(count - 1, 0), 4)
            or np.any(~np.isfinite(position))
            or np.any(~np.isfinite(orientation))
            or not np.allclose(
                np.linalg.norm(orientation, axis=1),
                1.0,
                rtol=1.0e-7,
                atol=1.0e-9,
            )
        ):
            raise ValueError("forecast trace arrays are not aligned")
        completed = int(self.completed_intervals)
        requested = int(self.requested_intervals)
        reason = str(self.failure_reason)
        if (
            isinstance(self.completed_intervals, (bool, np.bool_))
            or isinstance(self.requested_intervals, (bool, np.bool_))
            or requested < 1
            or completed < 0
            or completed > requested
            or count != completed + 1
            or bool(reason) != (completed < requested)
        ):
            raise ValueError("forecast completion and failure reason disagree")
        for name, value in (
            ("times", times),
            ("position", position),
            ("orientation_xyzw", orientation),
            ("thrust_saturated", thrust),
            ("gimbal_saturated", gimbal),
        ):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        object.__setattr__(self, "completed_intervals", completed)
        object.__setattr__(self, "requested_intervals", requested)
        object.__setattr__(self, "failure_reason", reason)

    @property
    def completed(self) -> bool:
        return self.completed_intervals == self.requested_intervals


@dataclass(frozen=True)
class PidForecastOutcome:
    """One forecast trace and its unit-preserving metrics."""

    candidate_id: str
    sample_id: str
    bag_id: str
    discrepancy_seed: int
    trace: PidForecastTrace
    metrics: ForecastMetrics

    def __post_init__(self) -> None:
        for name in ("candidate_id", "sample_id", "bag_id"):
            object.__setattr__(
                self,
                name,
                _canonical_identifier(getattr(self, name), name),
            )
        seed = self.discrepancy_seed
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("discrepancy_seed must be non-negative")
        if not isinstance(self.trace, PidForecastTrace):
            raise TypeError("trace must be PidForecastTrace")
        if not isinstance(self.metrics, ForecastMetrics):
            raise TypeError("metrics must be ForecastMetrics")
        object.__setattr__(self, "discrepancy_seed", int(seed))


def _actuator_saturation(
    actuator_state: ActuatorState,
    parameters: ActuatorParameters,
) -> Tuple[np.ndarray, np.ndarray]:
    thrust_scale = max(1.0, abs(parameters.maximum_thrust))
    gimbal_scale = max(1.0, abs(parameters.maximum_gimbal_angle))
    thrust_tolerance = 1.0e-10 * thrust_scale
    gimbal_tolerance = 1.0e-10 * gimbal_scale
    thrust = (
        actuator_state.thrust <= parameters.minimum_thrust + thrust_tolerance
    ) | (actuator_state.thrust >= parameters.maximum_thrust - thrust_tolerance)
    gimbal = (
        np.abs(actuator_state.gimbal_angle)
        >= parameters.maximum_gimbal_angle - gimbal_tolerance
    )
    return thrust, gimbal


def _error_metrics(
    trace: PidForecastTrace,
    scenario: PidForecastScenario,
) -> Tuple[float, float, float, float]:
    count = trace.times.size
    position_error = trace.position - np.asarray(
        tuple(value.position for value in scenario.references[:count])
    )
    orientation_error = np.empty((count, 3), dtype=float)
    for index in range(count):
        actual = quaternion_to_matrix(trace.orientation_xyzw[index])
        reference = euler_xyz_to_matrix(scenario.references[index].rpy)
        orientation_error[index] = so3_log(reference.T @ actual)
    position_norm = np.linalg.norm(position_error, axis=1)
    orientation_norm = np.linalg.norm(orientation_error, axis=1)
    return (
        float(np.sqrt(np.mean(position_norm * position_norm))),
        float(np.sqrt(np.mean(orientation_norm * orientation_norm))),
        float(np.max(position_norm)),
        float(np.max(orientation_norm)),
    )


def _metrics_from_trace(
    trace: PidForecastTrace,
    scenario: PidForecastScenario,
) -> ForecastMetrics:
    (
        position_rmse,
        orientation_rmse,
        maximum_position_error,
        maximum_orientation_error,
    ) = _error_metrics(trace, scenario)
    horizon = float(scenario.times[-1] - scenario.times[0])
    completed_duration = float(trace.times[-1] - trace.times[0])
    interval_duration = np.diff(trace.times)
    if interval_duration.size:
        any_saturated = np.any(
            np.concatenate(
                (trace.thrust_saturated, trace.gimbal_saturated), axis=1
            ),
            axis=1,
        )
        saturation_duration = float(
            np.sum(interval_duration[any_saturated])
        )
        channel_activity = np.concatenate(
            (trace.thrust_saturated, trace.gimbal_saturated), axis=1
        )
        saturation_rate = float(
            np.sum(interval_duration[:, None] * channel_activity)
            / (8.0 * completed_duration)
        )
    else:
        saturation_duration = 0.0
        saturation_rate = 0.0
    return ForecastMetrics(
        position_rmse=position_rmse,
        orientation_rmse=orientation_rmse,
        maximum_position_error=maximum_position_error,
        maximum_orientation_error=maximum_orientation_error,
        forecast_completion=completed_duration / horizon,
        numerical_failure_count=0 if trace.completed else 1,
        actuator_saturation_duration=saturation_duration,
        actuator_saturation_rate=saturation_rate,
    )


def _body_wrench_discrepancy(
    realization: ModelDiscrepancyRealization,
    time_step: np.ndarray,
    parameters: VehicleParameters,
) -> np.ndarray:
    statistical = realization.interval_average_residual(time_step)
    if realization.residual_quantity == BODY_WRENCH_MODEL_DISCREPANCY:
        return statistical
    if realization.residual_quantity == SPECIFIC_ACCELERATION_MODEL_DISCREPANCY:
        wrench = np.empty_like(statistical)
        wrench[:, :3] = parameters.mass * statistical[:, :3]
        wrench[:, 3:] = np.einsum(
            "ij,nj->ni", parameters.inertia, statistical[:, 3:]
        )
        return wrench
    raise AssertionError("validated discrepancy quantity is unreachable")


def run_pid_forecast(
    candidate: PidCandidate,
    sample: PhysicalPlantSample,
    scenario: PidForecastScenario,
    discrepancy: ModelDiscrepancyRealization,
) -> PidForecastOutcome:
    """Run one continuous candidate/sample/bag/replicate forecast."""

    if not isinstance(candidate, PidCandidate):
        raise TypeError("candidate must be PidCandidate")
    if not isinstance(sample, PhysicalPlantSample):
        raise TypeError("sample must be PhysicalPlantSample")
    if not isinstance(scenario, PidForecastScenario):
        raise TypeError("scenario must be PidForecastScenario")
    if not isinstance(discrepancy, ModelDiscrepancyRealization):
        raise TypeError("discrepancy must be ModelDiscrepancyRealization")
    initial = scenario.initial_condition(sample.sample_id)
    configured = apply_pid_gain_configuration(
        scenario.controller_configuration, candidate.configuration
    )
    controller = GrapeController(
        configured,
        scenario.controller_nominal_parameters,
        scenario.controller_geometry,
        scenario.articulated_model,
    )
    actuator_parameters = replace(
        scenario.actuator_parameters, delay=sample.delay
    )
    stepper = ClosedLoopStepper(
        controller=controller,
        plant=FullSixDofPlant(sample.parameters, scenario.plant_geometry),
        actuator_parameters=actuator_parameters,
        initial_state=ClosedLoopStepperState(
            time=float(scenario.times[0]),
            rigid_body_state=initial.rigid_body_state,
            controller_state=initial.controller_state,
            actuator_state=initial.actuator_state,
        ),
    )
    discrepancy_path = _body_wrench_discrepancy(
        discrepancy,
        np.diff(scenario.times),
        sample.parameters,
    )
    position = [initial.rigid_body_state.position]
    orientation = [initial.rigid_body_state.orientation_xyzw]
    thrust_saturated = []
    gimbal_saturated = []
    completed = 0
    failure_reason = ""
    for index in range(scenario.times.size - 1):
        try:
            emitted = stepper.advance_interval(
                float(scenario.times[index + 1]),
                scenario.references[index],
                discrepancy_path[index],
            )
        except _NUMERICAL_FORECAST_ERRORS as error:
            failure_reason = "{}: {}".format(type(error).__name__, error)
            break
        thrust, gimbal = _actuator_saturation(
            emitted.actuator_state, actuator_parameters
        )
        thrust_saturated.append(thrust)
        gimbal_saturated.append(gimbal)
        completed += 1
        position.append(stepper.state.rigid_body_state.position)
        orientation.append(stepper.state.rigid_body_state.orientation_xyzw)

    trace = PidForecastTrace(
        times=scenario.times[: completed + 1],
        position=np.asarray(position),
        orientation_xyzw=np.asarray(orientation),
        thrust_saturated=np.asarray(thrust_saturated, dtype=bool).reshape(
            (completed, 4)
        ),
        gimbal_saturated=np.asarray(gimbal_saturated, dtype=bool).reshape(
            (completed, 4)
        ),
        completed_intervals=completed,
        requested_intervals=scenario.times.size - 1,
        failure_reason=failure_reason,
    )
    return PidForecastOutcome(
        candidate_id=candidate.candidate_id,
        sample_id=sample.sample_id,
        bag_id=scenario.bag_id,
        discrepancy_seed=discrepancy.seed,
        trace=trace,
        metrics=_metrics_from_trace(trace, scenario),
    )


class ClosedLoopPidForecastEvaluator:
    """Callable adapter for :func:`evaluate_pid_candidates`."""

    def __init__(self, scenarios: Sequence[PidForecastScenario]) -> None:
        selected = tuple(scenarios)
        if (
            not selected
            or any(not isinstance(value, PidForecastScenario) for value in selected)
            or len({value.bag_id for value in selected}) != len(selected)
        ):
            raise ValueError("scenarios must have unique non-empty bag IDs")
        self._scenarios: Mapping[str, PidForecastScenario] = {
            value.bag_id: value for value in selected
        }

    @property
    def bag_ids(self) -> Tuple[str, ...]:
        return tuple(self._scenarios)

    def outcome(
        self,
        candidate: PidCandidate,
        sample: PhysicalPlantSample,
        bag_id: str,
        discrepancy: ModelDiscrepancyRealization,
    ) -> PidForecastOutcome:
        try:
            scenario = self._scenarios[str(bag_id)]
        except KeyError as error:
            raise KeyError("unknown PID forecast bag") from error
        return run_pid_forecast(candidate, sample, scenario, discrepancy)

    def __call__(
        self,
        candidate: PidCandidate,
        sample: PhysicalPlantSample,
        bag_id: str,
        discrepancy: ModelDiscrepancyRealization,
    ) -> ForecastMetrics:
        return self.outcome(candidate, sample, bag_id, discrepancy).metrics


__all__ = [
    "ClosedLoopPidForecastEvaluator",
    "PidForecastInitialCondition",
    "PidForecastOutcome",
    "PidForecastScenario",
    "PidForecastTrace",
    "run_pid_forecast",
]
