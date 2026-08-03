"""One-bag Q-only ensemble filtering and fixed-interval smoothing.

The fixed controller, plant, and actuator model are propagated by one
``ClosedLoopStepper`` per member.  The stochastic state is the 32-D Grape
chart state, including a six-axis body-frame OU residual wrench.  Static
physical parameters and EM orchestration deliberately remain outside this
module.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.closed_loop_stepper import (
    ClosedLoopStepper,
    ClosedLoopStepperState,
)
from grape_param_estim.controller import GrapeController
from grape_param_estim.diagonal_q import (
    BODY_WRENCH_DIMENSION,
    BodyWrenchDiagonalCovariance,
    OuTransitionFactors,
)
from grape_param_estim.dynamics import FullSixDofPlant
from grape_param_estim.ensemble_state_smoother import (
    deterministic_square_root_update,
    ensemble_rts_smoothing_step,
    exact_gaussian_ensemble,
)
from grape_param_estim.filter_state import (
    FILTER_STATE_DIMENSION,
    GrapeFilterState,
    GrapeFilterStateChart,
)
from grape_param_estim.geometry import normalise_quaternion
from grape_param_estim.progress import (
    CancellationToken,
    ProgressCallback,
    ProgressTracker,
)
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    ReferenceState,
)


POSE_DIMENSION = 6


def _finite_matrix(value, shape, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or np.any(~np.isfinite(result)):
        raise ValueError(
            "{} must have finite shape {}".format(name, shape)
        )
    return result.copy()


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or np.any(~np.isfinite(result)):
        raise ValueError(
            "{} must contain {} finite values".format(name, size)
        )
    return result.copy()


def _positive_definite_block(value, name: str) -> np.ndarray:
    result = _finite_matrix(value, (3, 3), name)
    if not np.allclose(result, result.T, rtol=1.0e-12, atol=1.0e-14):
        raise ValueError("{} must be symmetric".format(name))
    result = 0.5 * (result + result.T)
    if np.any(np.linalg.eigvalsh(result) <= 0.0):
        raise ValueError("{} must be positive definite".format(name))
    return result


def _increasing_times(value) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.ndim != 1
        or result.size < 1
        or np.any(~np.isfinite(result))
        or np.any(np.diff(result) <= 0.0)
    ):
        raise ValueError("times must be a non-empty increasing vector")
    return result.copy()


def _seed(value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("seed must be an unsigned 32-bit integer")
    result = int(value)
    if result < 0 or result >= 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    return result


def _state_ensemble(
    values: Iterable[GrapeFilterState], name: str
) -> Tuple[GrapeFilterState, ...]:
    result = tuple(values)
    if len(result) < 2:
        raise ValueError(
            "{} must contain at least two GrapeFilterState members".format(
                name
            )
        )
    if any(not isinstance(value, GrapeFilterState) for value in result):
        raise TypeError(
            "{} must contain only GrapeFilterState members".format(name)
        )
    return result


def _copy_state(value: GrapeFilterState) -> GrapeFilterState:
    return GrapeFilterState(
        rigid=value.rigid,
        controller=value.controller,
        actuator=value.actuator,
        residual_wrench=value.residual_wrench,
    )


def _copy_state_ensemble(
    values: Iterable[GrapeFilterState], name: str
) -> Tuple[GrapeFilterState, ...]:
    return tuple(_copy_state(value) for value in _state_ensemble(values, name))


@dataclass(frozen=True)
class PoseObservationCovariance:
    """Pose ``R`` with no translation/rotation cross covariance."""

    translation: np.ndarray
    rotation_tangent: np.ndarray

    def __post_init__(self) -> None:
        translation = _positive_definite_block(
            self.translation, "translation covariance"
        )
        rotation = _positive_definite_block(
            self.rotation_tangent, "rotation tangent covariance"
        )
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "rotation_tangent", rotation)

    @property
    def matrix(self) -> np.ndarray:
        result = np.zeros((POSE_DIMENSION, POSE_DIMENSION), dtype=float)
        result[:3, :3] = self.translation
        result[3:, 3:] = self.rotation_tangent
        return result

    @classmethod
    def isotropic(
        cls,
        translation_standard_deviation: float,
        rotation_standard_deviation: float,
    ) -> "PoseObservationCovariance":
        translation = float(translation_standard_deviation)
        rotation = float(rotation_standard_deviation)
        if (
            not np.isfinite(translation)
            or translation <= 0.0
            or not np.isfinite(rotation)
            or rotation <= 0.0
        ):
            raise ValueError(
                "pose standard deviations must be finite and positive"
            )
        return cls(
            np.eye(3) * translation**2,
            np.eye(3) * rotation**2,
        )


def ou_wrench_transition(
    current_wrench: np.ndarray,
    rho: float,
    innovation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the next OU wrench and its trapezoidal interval average."""

    current = np.asarray(current_wrench, dtype=float)
    noise = np.asarray(innovation, dtype=float)
    selected_rho = float(rho)
    if (
        current.ndim != 2
        or current.shape[0] < 1
        or current.shape[1] != BODY_WRENCH_DIMENSION
        or noise.shape != current.shape
        or np.any(~np.isfinite(current))
        or np.any(~np.isfinite(noise))
    ):
        raise ValueError(
            "wrench and innovation must have finite member-first shape "
            "(M, 6)"
        )
    if (
        not np.isfinite(selected_rho)
        or selected_rho < 0.0
        or selected_rho > 1.0
    ):
        raise ValueError("rho must be finite and lie in [0, 1]")
    next_wrench = selected_rho * current + noise
    interval_average = 0.5 * (current + next_wrench)
    if np.any(~np.isfinite(next_wrench)):
        raise ValueError("OU wrench transition is not representable")
    return next_wrench, interval_average


def _clip_actuator_state(
    state: GrapeFilterState,
    parameters: ActuatorParameters,
) -> GrapeFilterState:
    actuator = ActuatorState(
        thrust=np.clip(
            state.actuator.thrust,
            parameters.minimum_thrust,
            parameters.maximum_thrust,
        ),
        gimbal_angle=np.clip(
            state.actuator.gimbal_angle,
            -parameters.maximum_gimbal_angle,
            parameters.maximum_gimbal_angle,
        ),
    )
    return GrapeFilterState(
        rigid=state.rigid,
        controller=state.controller,
        actuator=actuator,
        residual_wrench=state.residual_wrench,
    )


def _state_series(
    values: Sequence[Sequence[GrapeFilterState]],
    time_count: int,
    member_count: int,
    name: str,
) -> Tuple[Tuple[GrapeFilterState, ...], ...]:
    selected = tuple(values)
    if len(selected) != time_count:
        raise ValueError("{} must align with times".format(name))
    result = []
    for ensemble in selected:
        copied = _copy_state_ensemble(ensemble, name)
        if len(copied) != member_count:
            raise ValueError("{} must be member-aligned".format(name))
        result.append(copied)
    return tuple(result)


@dataclass(frozen=True)
class StochasticClosedLoopEStepResult:
    """Member-aligned filtering and smoothing output for one bag."""

    times: np.ndarray
    smoothed_wrench_ensemble: np.ndarray
    filter_log_likelihood: float
    filter_log_likelihood_by_time: np.ndarray
    filter_nis: np.ndarray
    forecast_state_ensembles: Tuple[Tuple[GrapeFilterState, ...], ...]
    analysis_state_ensembles: Tuple[Tuple[GrapeFilterState, ...], ...]
    smoothed_state_ensembles: Tuple[Tuple[GrapeFilterState, ...], ...]
    smoothing_gains: Tuple[np.ndarray, ...]
    command_issue_times: Tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        times = _increasing_times(self.times)
        wrench = np.asarray(self.smoothed_wrench_ensemble, dtype=float)
        if (
            wrench.ndim != 3
            or wrench.shape[0] < 2
            or wrench.shape[1:] != (times.size, BODY_WRENCH_DIMENSION)
            or np.any(~np.isfinite(wrench))
        ):
            raise ValueError(
                "smoothed_wrench_ensemble must have finite shape (M, N, 6)"
            )
        member_count = int(wrench.shape[0])
        forecast = _state_series(
            self.forecast_state_ensembles,
            times.size,
            member_count,
            "forecast_state_ensembles",
        )
        analysis = _state_series(
            self.analysis_state_ensembles,
            times.size,
            member_count,
            "analysis_state_ensembles",
        )
        smoothed = _state_series(
            self.smoothed_state_ensembles,
            times.size,
            member_count,
            "smoothed_state_ensembles",
        )
        expected_wrench = np.transpose(
            np.asarray(
                [
                    [state.residual_wrench for state in ensemble]
                    for ensemble in smoothed
                ]
            ),
            (1, 0, 2),
        )
        if not np.array_equal(wrench, expected_wrench):
            raise ValueError(
                "smoothed_wrench_ensemble must match smoothed states"
            )
        likelihood = _finite_vector(
            self.filter_log_likelihood_by_time,
            times.size,
            "filter_log_likelihood_by_time",
        )
        nis = _finite_vector(self.filter_nis, times.size, "filter_nis")
        if np.any(nis < 0.0):
            raise ValueError("filter_nis cannot be negative")
        total = float(self.filter_log_likelihood)
        expected_total = math.fsum(float(value) for value in likelihood)
        if not np.isfinite(total) or not np.isclose(
            total, expected_total, rtol=1.0e-12, atol=1.0e-12
        ):
            raise ValueError(
                "filter_log_likelihood must sum the time contributions"
            )
        gains = tuple(
            np.asarray(value, dtype=float) for value in self.smoothing_gains
        )
        if len(gains) != max(0, times.size - 1) or any(
            value.shape != (FILTER_STATE_DIMENSION, FILTER_STATE_DIMENSION)
            or np.any(~np.isfinite(value))
            for value in gains
        ):
            raise ValueError("smoothing_gains are not time-aligned 32-D maps")
        histories = tuple(
            np.asarray(value, dtype=float) for value in self.command_issue_times
        )
        if len(histories) != member_count or any(
            value.shape != (max(0, times.size - 1),)
            or np.any(~np.isfinite(value))
            or not np.array_equal(value, times[:-1])
            for value in histories
        ):
            raise ValueError("command_issue_times are not member-aligned")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "smoothed_wrench_ensemble", wrench.copy())
        object.__setattr__(self, "filter_log_likelihood", total)
        object.__setattr__(
            self, "filter_log_likelihood_by_time", likelihood.copy()
        )
        object.__setattr__(self, "filter_nis", nis.copy())
        object.__setattr__(self, "forecast_state_ensembles", forecast)
        object.__setattr__(self, "analysis_state_ensembles", analysis)
        object.__setattr__(self, "smoothed_state_ensembles", smoothed)
        object.__setattr__(
            self, "smoothing_gains", tuple(value.copy() for value in gains)
        )
        object.__setattr__(
            self,
            "command_issue_times",
            tuple(value.copy() for value in histories),
        )

    @property
    def log_likelihood(self) -> float:
        return self.filter_log_likelihood

    @property
    def nis(self) -> np.ndarray:
        return self.filter_nis.copy()


def _validate_inputs(
    *,
    times,
    references,
    observed_position,
    observed_orientation_xyzw,
    initial_state_ensemble,
    controller,
    plant,
    actuator_parameters,
    wrench_covariance,
    observation_covariance,
):
    selected_times = _increasing_times(times)
    selected_references = tuple(references)
    if len(selected_references) != selected_times.size:
        raise ValueError("references must contain one ReferenceState per time")
    if any(
        not isinstance(value, ReferenceState) for value in selected_references
    ):
        raise TypeError("references must contain only ReferenceState values")
    position = _finite_matrix(
        observed_position,
        (selected_times.size, 3),
        "observed_position",
    )
    orientation = _finite_matrix(
        observed_orientation_xyzw,
        (selected_times.size, 4),
        "observed_orientation_xyzw",
    )
    orientation = np.asarray(
        [normalise_quaternion(value) for value in orientation]
    )
    ensemble = _copy_state_ensemble(
        initial_state_ensemble, "initial_state_ensemble"
    )
    if not isinstance(controller, GrapeController):
        raise TypeError("controller must be a GrapeController")
    if not isinstance(plant, FullSixDofPlant):
        raise TypeError("plant must be a FullSixDofPlant")
    if not isinstance(actuator_parameters, ActuatorParameters):
        raise TypeError("actuator_parameters must be ActuatorParameters")
    if not isinstance(wrench_covariance, BodyWrenchDiagonalCovariance):
        raise TypeError(
            "wrench_covariance must be BodyWrenchDiagonalCovariance"
        )
    if not isinstance(observation_covariance, PoseObservationCovariance):
        raise TypeError(
            "observation_covariance must be PoseObservationCovariance"
        )
    return (
        selected_times,
        selected_references,
        position,
        orientation,
        ensemble,
    )


def _transition_seed(seed: int, interval_index: int) -> int:
    return int((seed + 2654435761 * (interval_index + 1)) % (2**32))


def _chart_aware_smoothing_step(
    analysis: Tuple[GrapeFilterState, ...],
    next_forecast: Tuple[GrapeFilterState, ...],
    next_smoothed: Tuple[GrapeFilterState, ...],
    covariance_rcond: float,
):
    chart = GrapeFilterStateChart.from_ensemble(
        analysis + next_forecast + next_smoothed
    )
    smoothed_coordinates, gain = ensemble_rts_smoothing_step(
        chart.encode_ensemble(analysis),
        chart.encode_ensemble(next_forecast),
        chart.encode_ensemble(next_smoothed),
        covariance_rcond=covariance_rcond,
    )
    return chart.decode_ensemble(smoothed_coordinates, analysis), gain


def run_stochastic_closed_loop_e_step(
    *,
    times: Sequence[float],
    references: Sequence[ReferenceState],
    observed_position: np.ndarray,
    observed_orientation_xyzw: np.ndarray,
    initial_state_ensemble: Sequence[GrapeFilterState],
    controller: GrapeController,
    plant: FullSixDofPlant,
    actuator_parameters: ActuatorParameters,
    wrench_covariance: BodyWrenchDiagonalCovariance,
    correlation_time: float,
    observation_covariance: PoseObservationCovariance,
    seed: int,
    covariance_rcond: float = 1.0e-12,
    progress_callback: Optional[ProgressCallback] = None,
    cancellation_token: Optional[CancellationToken] = None,
    progress_run_id: str = "q-only-e-step",
    bag_id: Optional[str] = None,
) -> StochasticClosedLoopEStepResult:
    """Filter and smooth one bag with fixed model parameters and diagonal Q."""

    (
        selected_times,
        selected_references,
        position,
        orientation,
        initial_ensemble,
    ) = _validate_inputs(
        times=times,
        references=references,
        observed_position=observed_position,
        observed_orientation_xyzw=observed_orientation_xyzw,
        initial_state_ensemble=initial_state_ensemble,
        controller=controller,
        plant=plant,
        actuator_parameters=actuator_parameters,
        wrench_covariance=wrench_covariance,
        observation_covariance=observation_covariance,
    )
    selected_seed = _seed(seed)
    selected_rcond = float(covariance_rcond)
    if not np.isfinite(selected_rcond) or selected_rcond <= 0.0:
        raise ValueError("covariance_rcond must be finite and positive")
    if not isinstance(progress_run_id, str) or not progress_run_id:
        raise ValueError("progress_run_id must be a non-empty string")
    if bag_id is not None and (
        not isinstance(bag_id, str) or not bag_id
    ):
        raise ValueError("bag_id must be None or a non-empty string")

    factors = OuTransitionFactors(selected_times, correlation_time)
    innovation_variance = factors.innovation_variance(wrench_covariance)
    member_count = len(initial_ensemble)
    time_count = int(selected_times.size)
    total_units = (
        time_count
        + member_count * max(0, time_count - 1)
        + max(0, time_count - 1)
    )
    tracker = ProgressTracker(
        progress_run_id,
        total_units,
        callback=progress_callback,
        cancellation_token=cancellation_token,
    )
    tracker.emit(
        0,
        "q_only_filter",
        "Q-only ensemble filtering",
        bag_id=bag_id,
        message="starting filter",
    )

    steppers = tuple(
        ClosedLoopStepper(
            controller=controller,
            plant=plant,
            actuator_parameters=actuator_parameters,
            initial_state=ClosedLoopStepperState(
                time=selected_times[0],
                rigid_body_state=state.rigid,
                controller_state=state.controller,
                actuator_state=state.actuator,
            ),
        )
        for state in initial_ensemble
    )
    forecast_ensembles = [initial_ensemble]
    analysis_ensembles = []
    log_likelihood = np.empty(time_count, dtype=float)
    nis = np.empty(time_count, dtype=float)
    completed = 0

    for time_index in range(time_count):
        tracker.checkpoint()
        forecast = forecast_ensembles[time_index]
        chart = GrapeFilterStateChart.from_ensemble(forecast)
        forecast_coordinates = chart.encode_ensemble(forecast)
        predicted_pose = chart.predicted_pose_ensemble(forecast)
        observed_pose = chart.observed_pose_coordinates(
            position[time_index], orientation[time_index]
        )
        update = deterministic_square_root_update(
            forecast_coordinates,
            predicted_pose,
            observed_pose,
            observation_covariance.matrix,
        )
        decoded = chart.decode_ensemble(
            update.analysis_ensemble, forecast
        )
        analysis = tuple(
            _clip_actuator_state(state, actuator_parameters)
            for state in decoded
        )
        analysis_ensembles.append(analysis)
        for stepper, state in zip(steppers, analysis):
            stepper.replace_dynamic_state(
                rigid_body_state=state.rigid,
                controller_state=state.controller,
                actuator_state=state.actuator,
            )
        log_likelihood[time_index] = update.approximate_log_likelihood
        nis[time_index] = float(
            update.innovation
            @ np.linalg.solve(
                update.innovation_covariance, update.innovation
            )
        )
        completed += 1
        tracker.emit(
            completed,
            "q_only_filter",
            "Q-only ensemble filtering",
            bag_id=bag_id,
            message="analysed observation {}/{}".format(
                time_index + 1, time_count
            ),
        )

        if time_index + 1 == time_count:
            continue
        analysis_chart = GrapeFilterStateChart.from_ensemble(analysis)
        analysis_coordinates = analysis_chart.encode_ensemble(analysis)
        innovation_covariance = np.diag(innovation_variance[time_index])
        try:
            noise = exact_gaussian_ensemble(
                np.zeros(BODY_WRENCH_DIMENSION),
                innovation_covariance,
                member_count,
                _transition_seed(selected_seed, time_index),
                orthogonal_to=analysis_coordinates,
            )
        except ValueError as error:
            raise ValueError(
                "ensemble cannot provide six process-noise directions "
                "orthogonal to the current 32-D analysis anomalies"
            ) from error
        current_wrench = np.asarray(
            [state.residual_wrench for state in analysis]
        )
        next_wrench, interval_wrench = ou_wrench_transition(
            current_wrench, factors.rho[time_index], noise
        )
        next_forecast = []
        for member_index, stepper in enumerate(steppers):
            tracker.checkpoint()
            stepper.advance_interval(
                selected_times[time_index + 1],
                selected_references[time_index],
                interval_wrench[member_index],
            )
            stepper_state = stepper.state
            next_forecast.append(
                GrapeFilterState(
                    rigid=stepper_state.rigid_body_state,
                    controller=stepper_state.controller_state,
                    actuator=stepper_state.actuator_state,
                    residual_wrench=next_wrench[member_index],
                )
            )
            completed += 1
            tracker.emit(
                completed,
                "q_only_forecast",
                "Q-only closed-loop forecast",
                bag_id=bag_id,
                member_id=member_index,
                message="propagated interval {}/{}".format(
                    time_index + 1, time_count - 1
                ),
            )
        forecast_ensembles.append(tuple(next_forecast))

    smoothed_ensembles = [None] * time_count
    smoothing_gains = [None] * max(0, time_count - 1)
    smoothed_ensembles[-1] = analysis_ensembles[-1]
    for time_index in range(time_count - 2, -1, -1):
        tracker.checkpoint()
        smoothed, gain = _chart_aware_smoothing_step(
            analysis_ensembles[time_index],
            forecast_ensembles[time_index + 1],
            smoothed_ensembles[time_index + 1],
            selected_rcond,
        )
        smoothed_ensembles[time_index] = smoothed
        smoothing_gains[time_index] = gain
        completed += 1
        tracker.emit(
            completed,
            "q_only_smoother",
            "Q-only fixed-interval smoothing",
            bag_id=bag_id,
            message="smoothed boundary {}/{}".format(
                time_index + 1, time_count
            ),
        )

    smoothed_wrench = np.transpose(
        np.asarray(
            [
                [state.residual_wrench for state in ensemble]
                for ensemble in smoothed_ensembles
            ]
        ),
        (1, 0, 2),
    )
    return StochasticClosedLoopEStepResult(
        times=selected_times,
        smoothed_wrench_ensemble=smoothed_wrench,
        filter_log_likelihood=math.fsum(
            float(value) for value in log_likelihood
        ),
        filter_log_likelihood_by_time=log_likelihood,
        filter_nis=nis,
        forecast_state_ensembles=tuple(forecast_ensembles),
        analysis_state_ensembles=tuple(analysis_ensembles),
        smoothed_state_ensembles=tuple(smoothed_ensembles),
        smoothing_gains=tuple(smoothing_gains),
        command_issue_times=tuple(
            stepper.command_issue_times for stepper in steppers
        ),
    )


__all__ = [
    "POSE_DIMENSION",
    "PoseObservationCovariance",
    "StochasticClosedLoopEStepResult",
    "ou_wrench_transition",
    "run_stochastic_closed_loop_e_step",
]
