"""Single-bag augmented EnKF/EnRTS for shared static parameters.

Each ensemble member carries 19 Euclidean static coordinates followed by the
32-D Grape dynamic chart state.  Plant parameters and a smoothly bounded
actuator delay are decoded after every analysis, while the controller's
nominal model remains fixed.  The bounded causal command buffer is auxiliary
state: it receives the same ETKF member analysis and is trimmed to commands
that can still affect a future interval.  OU draws are exactly orthogonal to
the 51-D primary state, not to every auxiliary-history anomaly, and EnRTS
smooths the 51-D marginal.  This module intentionally contains no multi-bag
or EM orchestration.
"""

from dataclasses import dataclass, replace
import math
from typing import Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.augmented_forecast_pool import (
    PersistentAugmentedForecastPool,
    validated_forecast_worker_count,
)
from grape_param_estim.augmented_parameter_state import (
    AUGMENTED_FILTER_DIMENSION,
    MINIMUM_PROCESS_NOISE_MEMBER_COUNT,
    SHARED_STATIC_DIMENSION,
    AugmentedInitialEnsemble,
    decode_shared_static_coordinates,
)
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
    GrapeFilterState,
    GrapeFilterStateChart,
)
from grape_param_estim.progress import (
    CancellationToken,
    ProgressCallback,
    ProgressTracker,
)
from grape_param_estim.stochastic_closed_loop import (
    PoseObservationCovariance,
    ou_wrench_transition,
)
from grape_param_estim.strong_constraint import StrongConstraintProblem
from grape_param_estim.system import (
    ActuatorCommand,
    ActuatorParameters,
    ActuatorState,
)
from grape_param_estim.timing import BoundedDelayChart


COMMAND_COORDINATE_DIMENSION = 22


def _finite_array(value, shape, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or np.any(~np.isfinite(result)):
        raise ValueError(
            "{} must have finite shape {}".format(name, shape)
        )
    return result.copy()


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    return _finite_array(value, (size,), name)


def _seed(value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("seed must be an unsigned 32-bit integer")
    result = int(value)
    if result < 0 or result >= 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    return result


def _copy_state(value: GrapeFilterState) -> GrapeFilterState:
    if not isinstance(value, GrapeFilterState):
        raise TypeError("dynamic ensemble must contain GrapeFilterState")
    return GrapeFilterState(
        rigid=value.rigid,
        controller=value.controller,
        actuator=value.actuator,
        residual_wrench=value.residual_wrench,
    )


def _copy_dynamic_series(
    values: Sequence[Sequence[GrapeFilterState]],
    time_count: int,
    member_count: int,
    name: str,
) -> Tuple[Tuple[GrapeFilterState, ...], ...]:
    selected = tuple(tuple(value) for value in values)
    if len(selected) != time_count or any(
        len(value) != member_count for value in selected
    ):
        raise ValueError("{} must be time/member aligned".format(name))
    return tuple(
        tuple(_copy_state(state) for state in ensemble)
        for ensemble in selected
    )


def _clip_dynamic_state(
    state: GrapeFilterState, limits: ActuatorParameters
) -> GrapeFilterState:
    return GrapeFilterState(
        rigid=state.rigid,
        controller=state.controller,
        actuator=ActuatorState(
            np.clip(
                state.actuator.thrust,
                limits.minimum_thrust,
                limits.maximum_thrust,
            ),
            np.clip(
                state.actuator.gimbal_angle,
                -limits.maximum_gimbal_angle,
                limits.maximum_gimbal_angle,
            ),
        ),
        residual_wrench=state.residual_wrench,
    )


def _member_models(
    problem: StrongConstraintProblem,
    static_coordinates: np.ndarray,
    delay_chart: BoundedDelayChart,
):
    parameters = []
    actuators = []
    for coordinates in static_coordinates:
        vehicle, delay = decode_shared_static_coordinates(
            problem, coordinates, delay_chart
        )
        parameters.append(vehicle)
        actuators.append(
            replace(problem.actuator_parameters, delay=delay)
        )
    return tuple(parameters), tuple(actuators)


def _encode_augmented(
    static_coordinates: np.ndarray,
    dynamic_states: Sequence[GrapeFilterState],
    chart: GrapeFilterStateChart,
) -> np.ndarray:
    static = np.asarray(static_coordinates, dtype=float)
    dynamic = chart.encode_ensemble(dynamic_states)
    if static.shape != (dynamic.shape[0], SHARED_STATIC_DIMENSION):
        raise ValueError("static and dynamic ensembles must be aligned")
    result = np.concatenate((static, dynamic), axis=1)
    if result.shape[1] != AUGMENTED_FILTER_DIMENSION:
        raise AssertionError("augmented state layout is inconsistent")
    return result


def _transition_seed(seed: int, interval_index: int) -> int:
    return int((seed + 2654435761 * (interval_index + 1)) % (2**32))


def _encode_command(command: ActuatorCommand) -> np.ndarray:
    if not isinstance(command, ActuatorCommand):
        raise TypeError("command history must contain ActuatorCommand values")
    return np.concatenate(
        (
            command.thrust,
            command.gimbal_angle,
            command.virtual_force,
            command.desired_acceleration,
        )
    )


def _decode_command(coordinates: np.ndarray) -> ActuatorCommand:
    value = _finite_vector(
        coordinates, COMMAND_COORDINATE_DIMENSION, "command coordinates"
    )
    return ActuatorCommand(
        value[0:4],
        value[4:8],
        value[8:16],
        value[16:22],
    )


def _analyse_auxiliary_ensemble(
    forecast_auxiliary: np.ndarray,
    predicted_observation: np.ndarray,
    update,
) -> np.ndarray:
    """Apply an existing ETKF analysis to aligned Euclidean auxiliaries."""

    forecast = np.asarray(forecast_auxiliary, dtype=float)
    predicted = np.asarray(predicted_observation, dtype=float)
    members = update.analysis_ensemble.shape[0]
    if (
        forecast.ndim != 2
        or forecast.shape[0] != members
        or predicted.ndim != 2
        or predicted.shape[0] != members
        or predicted.shape[1] != update.innovation.size
        or np.any(~np.isfinite(forecast))
        or np.any(~np.isfinite(predicted))
    ):
        raise ValueError("auxiliary ensemble is not member aligned")
    mean = np.mean(forecast, axis=0)
    anomalies = forecast - mean[None, :]
    observation_anomalies = (
        predicted - update.forecast_observation_mean[None, :]
    )
    cross_covariance = (
        anomalies.T @ observation_anomalies / float(members - 1)
    )
    gain = np.linalg.solve(
        update.innovation_covariance, cross_covariance.T
    ).T
    analysis_mean = mean + gain @ update.innovation
    analysis_anomalies = update.member_transform @ anomalies
    analysis_anomalies -= np.mean(
        analysis_anomalies, axis=0, keepdims=True
    )
    result = analysis_mean[None, :] + analysis_anomalies
    if np.any(~np.isfinite(result)):
        raise ValueError("auxiliary analysis is not representable")
    return result


def _analyse_command_histories(
    steppers: Sequence[ClosedLoopStepper],
    predicted_observation: np.ndarray,
    update,
) -> None:
    """Keep causal delay buffers member-aligned through an ETKF analysis."""

    if not steppers:
        raise ValueError("at least one stepper is required")
    issue_times = steppers[0].command_issue_times
    for stepper in steppers[1:]:
        if not np.array_equal(stepper.command_issue_times, issue_times):
            raise ValueError("command histories must share one issue-time grid")
    history_size = int(issue_times.size)
    if history_size == 0:
        return
    forecast = np.asarray(
        [
            [
                _encode_command(command)
                for command in stepper.command_history_commands
            ]
            for stepper in steppers
        ],
        dtype=float,
    )
    expected_shape = (
        len(steppers),
        history_size,
        COMMAND_COORDINATE_DIMENSION,
    )
    if forecast.shape != expected_shape:
        raise ValueError("command histories are not member aligned")
    analysed = _analyse_auxiliary_ensemble(
        forecast.reshape(len(steppers), -1),
        predicted_observation,
        update,
    ).reshape(expected_shape)
    for stepper, member_history in zip(steppers, analysed):
        stepper.replace_command_history(
            tuple(_decode_command(value) for value in member_history)
        )


def _chart_aware_smoothing_step(
    analysis_static: np.ndarray,
    analysis_dynamic: Tuple[GrapeFilterState, ...],
    next_forecast_static: np.ndarray,
    next_forecast_dynamic: Tuple[GrapeFilterState, ...],
    next_smoothed_static: np.ndarray,
    next_smoothed_dynamic: Tuple[GrapeFilterState, ...],
    covariance_rcond: float,
):
    chart = GrapeFilterStateChart.from_ensemble(
        analysis_dynamic + next_forecast_dynamic + next_smoothed_dynamic
    )
    smoothed, gain = ensemble_rts_smoothing_step(
        _encode_augmented(analysis_static, analysis_dynamic, chart),
        _encode_augmented(
            next_forecast_static, next_forecast_dynamic, chart
        ),
        _encode_augmented(
            next_smoothed_static, next_smoothed_dynamic, chart
        ),
        covariance_rcond=covariance_rcond,
    )
    static = smoothed[:, :SHARED_STATIC_DIMENSION]
    dynamic = chart.decode_ensemble(
        smoothed[:, SHARED_STATIC_DIMENSION:], analysis_dynamic
    )
    return static, dynamic, gain


@dataclass(frozen=True)
class AugmentedParameterFilterResult:
    """Complete member-aligned result of one augmented filtering pass."""

    member_id: np.ndarray
    times: np.ndarray
    prior_static_ensemble: np.ndarray
    final_static_ensemble: np.ndarray
    static_forecast_ensemble: np.ndarray
    static_analysis_ensemble: np.ndarray
    static_smoothed_ensemble: np.ndarray
    dynamic_forecast_state_ensembles: Tuple[
        Tuple[GrapeFilterState, ...], ...
    ]
    dynamic_analysis_state_ensembles: Tuple[
        Tuple[GrapeFilterState, ...], ...
    ]
    dynamic_smoothed_state_ensembles: Tuple[
        Tuple[GrapeFilterState, ...], ...
    ]
    smoothed_wrench_ensemble: np.ndarray
    filter_log_likelihood: float
    filter_log_likelihood_by_time: np.ndarray
    filter_nis: np.ndarray
    smoothing_gains: Tuple[np.ndarray, ...]
    command_issue_times: Tuple[np.ndarray, ...]
    final_command_history_ensemble: np.ndarray
    maximum_delay: float
    applied_model_mass: np.ndarray
    applied_model_delay: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        if (
            times.ndim != 1
            or times.size < 2
            or np.any(~np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("times must be a finite increasing vector")
        member_id = np.asarray(self.member_id, dtype=np.int64)
        members = int(member_id.size)
        if (
            member_id.shape != (members,)
            or np.unique(member_id).size != members
            or members < MINIMUM_PROCESS_NOISE_MEMBER_COUNT
        ):
            raise ValueError("member_id is invalid for a 51-D process")
        static_shape = (members, times.size, SHARED_STATIC_DIMENSION)
        prior = _finite_array(
            self.prior_static_ensemble,
            (members, SHARED_STATIC_DIMENSION),
            "prior_static_ensemble",
        )
        final = _finite_array(
            self.final_static_ensemble,
            (members, SHARED_STATIC_DIMENSION),
            "final_static_ensemble",
        )
        static_forecast = _finite_array(
            self.static_forecast_ensemble,
            static_shape,
            "static_forecast_ensemble",
        )
        static_analysis = _finite_array(
            self.static_analysis_ensemble,
            static_shape,
            "static_analysis_ensemble",
        )
        static_smoothed = _finite_array(
            self.static_smoothed_ensemble,
            static_shape,
            "static_smoothed_ensemble",
        )
        if not np.array_equal(prior, static_forecast[:, 0, :]):
            raise ValueError("prior static ensemble must be first forecast")
        if not np.array_equal(final, static_analysis[:, -1, :]):
            raise ValueError("final static ensemble must be final analysis")
        dynamic_forecast = _copy_dynamic_series(
            self.dynamic_forecast_state_ensembles,
            times.size,
            members,
            "dynamic_forecast_state_ensembles",
        )
        dynamic_analysis = _copy_dynamic_series(
            self.dynamic_analysis_state_ensembles,
            times.size,
            members,
            "dynamic_analysis_state_ensembles",
        )
        dynamic_smoothed = _copy_dynamic_series(
            self.dynamic_smoothed_state_ensembles,
            times.size,
            members,
            "dynamic_smoothed_state_ensembles",
        )
        wrench = _finite_array(
            self.smoothed_wrench_ensemble,
            (members, times.size, BODY_WRENCH_DIMENSION),
            "smoothed_wrench_ensemble",
        )
        expected_wrench = np.transpose(
            np.asarray(
                [
                    [state.residual_wrench for state in ensemble]
                    for ensemble in dynamic_smoothed
                ]
            ),
            (1, 0, 2),
        )
        if not np.array_equal(wrench, expected_wrench):
            raise ValueError("smoothed wrench must match dynamic states")
        likelihood = _finite_vector(
            self.filter_log_likelihood_by_time,
            times.size,
            "filter_log_likelihood_by_time",
        )
        nis = _finite_vector(self.filter_nis, times.size, "filter_nis")
        if np.any(nis < 0.0):
            raise ValueError("filter_nis cannot be negative")
        total = float(self.filter_log_likelihood)
        if not np.isfinite(total) or not np.isclose(
            total,
            math.fsum(float(value) for value in likelihood),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("filter likelihood total is inconsistent")
        gains = tuple(
            np.asarray(value, dtype=float) for value in self.smoothing_gains
        )
        if len(gains) != times.size - 1 or any(
            value.shape
            != (AUGMENTED_FILTER_DIMENSION, AUGMENTED_FILTER_DIMENSION)
            or np.any(~np.isfinite(value))
            for value in gains
        ):
            raise ValueError("smoothing gains must be finite 51-D maps")
        histories = tuple(
            np.asarray(value, dtype=float) for value in self.command_issue_times
        )
        maximum_delay = float(self.maximum_delay)
        if not np.isfinite(maximum_delay) or maximum_delay <= 0.0:
            raise ValueError("maximum_delay must be finite and positive")
        all_issue_times = times[:-1]
        first_retained = max(
            int(
                np.searchsorted(
                    all_issue_times,
                    times[-1] - maximum_delay,
                    side="right",
                )
                - 1
            ),
            0,
        )
        expected_issue_times = all_issue_times[first_retained:]
        if len(histories) != members or any(
            not np.array_equal(value, expected_issue_times)
            for value in histories
        ):
            raise ValueError("command histories are not member aligned")
        command_history = _finite_array(
            self.final_command_history_ensemble,
            (
                members,
                expected_issue_times.size,
                COMMAND_COORDINATE_DIMENSION,
            ),
            "final_command_history_ensemble",
        )
        masses = _finite_array(
            self.applied_model_mass,
            (members, times.size),
            "applied_model_mass",
        )
        delays = _finite_array(
            self.applied_model_delay,
            (members, times.size),
            "applied_model_delay",
        )
        if np.any(masses <= 0.0) or np.any(delays < 0.0):
            raise ValueError("applied model mass/delay are not physical")
        for name, value in (
            ("times", times),
            ("member_id", member_id),
            ("prior_static_ensemble", prior),
            ("final_static_ensemble", final),
            ("static_forecast_ensemble", static_forecast),
            ("static_analysis_ensemble", static_analysis),
            ("static_smoothed_ensemble", static_smoothed),
            ("smoothed_wrench_ensemble", wrench),
            ("filter_log_likelihood_by_time", likelihood),
            ("filter_nis", nis),
            ("final_command_history_ensemble", command_history),
            ("applied_model_mass", masses),
            ("applied_model_delay", delays),
        ):
            object.__setattr__(self, name, value.copy())
        object.__setattr__(self, "filter_log_likelihood", total)
        object.__setattr__(self, "maximum_delay", maximum_delay)
        object.__setattr__(
            self,
            "dynamic_forecast_state_ensembles",
            dynamic_forecast,
        )
        object.__setattr__(
            self,
            "dynamic_analysis_state_ensembles",
            dynamic_analysis,
        )
        object.__setattr__(
            self,
            "dynamic_smoothed_state_ensembles",
            dynamic_smoothed,
        )
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


def run_augmented_parameter_filter(
    *,
    problem: StrongConstraintProblem,
    initial_ensemble: AugmentedInitialEnsemble,
    wrench_covariance: BodyWrenchDiagonalCovariance,
    correlation_time: float,
    observation_covariance: PoseObservationCovariance,
    seed: int,
    delay_chart: Optional[BoundedDelayChart] = None,
    covariance_rcond: float = 1.0e-12,
    forecast_workers: int = 1,
    progress_callback: Optional[ProgressCallback] = None,
    cancellation_token: Optional[CancellationToken] = None,
    progress_run_id: str = "augmented-parameter-filter",
    bag_id: Optional[str] = None,
) -> AugmentedParameterFilterResult:
    """Update one bag's shared 19-D static ensemble sequentially.

    ``delay_chart`` supplies the physical maximum used both by the bijective
    delay coordinate and by causal command-buffer truncation.
    ``forecast_workers=1`` preserves the in-process reference path; larger
    values use a persistent spawned pool while the parent retains history.
    """

    if not isinstance(problem, StrongConstraintProblem):
        raise TypeError("problem must be a StrongConstraintProblem")
    if not isinstance(initial_ensemble, AugmentedInitialEnsemble):
        raise TypeError("initial_ensemble must be AugmentedInitialEnsemble")
    if initial_ensemble.member_count < MINIMUM_PROCESS_NOISE_MEMBER_COUNT:
        raise ValueError(
            "augmented process noise requires at least {} members".format(
                MINIMUM_PROCESS_NOISE_MEMBER_COUNT
            )
        )
    if not isinstance(wrench_covariance, BodyWrenchDiagonalCovariance):
        raise TypeError(
            "wrench_covariance must be BodyWrenchDiagonalCovariance"
        )
    if not isinstance(observation_covariance, PoseObservationCovariance):
        raise TypeError(
            "observation_covariance must be PoseObservationCovariance"
        )
    selected_delay_chart = delay_chart or BoundedDelayChart()
    if not isinstance(selected_delay_chart, BoundedDelayChart):
        raise TypeError("delay_chart must be a BoundedDelayChart")
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
    cancellation = cancellation_token or CancellationToken()
    if not isinstance(cancellation, CancellationToken):
        raise TypeError("cancellation_token must be CancellationToken")

    observations = problem.observations
    times = observations.times.copy()
    factors = OuTransitionFactors(times, correlation_time)
    innovation_variance = factors.innovation_variance(wrench_covariance)
    member_count = initial_ensemble.member_count
    selected_workers = validated_forecast_worker_count(
        forecast_workers, member_count
    )
    time_count = int(times.size)
    total_units = (
        time_count
        + member_count * (time_count - 1)
        + time_count - 1
    )
    tracker = ProgressTracker(
        progress_run_id,
        total_units,
        callback=progress_callback,
        cancellation_token=cancellation,
    )
    tracker.emit(
        0,
        "augmented_filter",
        "Augmented parameter filtering",
        bag_id=bag_id,
        message="starting augmented filter",
    )

    nominal_controller = GrapeController(
        problem.controller_configuration,
        problem.controller_parameters,
        problem.geometry,
        articulated_model=GrapeArticulatedModel(),
    )
    prior_static = initial_ensemble.shared_coordinates.copy()
    prior_parameters, prior_actuators = _member_models(
        problem, prior_static, selected_delay_chart
    )
    steppers = tuple(
        ClosedLoopStepper(
            controller=nominal_controller,
            plant=FullSixDofPlant(parameters, problem.geometry),
            actuator_parameters=actuators,
            initial_state=ClosedLoopStepperState(
                time=times[0],
                rigid_body_state=state.rigid,
                controller_state=state.controller,
                actuator_state=state.actuator,
            ),
        )
        for state, parameters, actuators in zip(
            initial_ensemble.filter_states,
            prior_parameters,
            prior_actuators,
        )
    )
    static_forecast = [prior_static]
    static_analysis = []
    dynamic_forecast = [tuple(initial_ensemble.filter_states)]
    dynamic_analysis = []
    log_likelihood = np.empty(time_count, dtype=float)
    nis = np.empty(time_count, dtype=float)
    model_mass = np.empty((time_count, member_count), dtype=float)
    model_delay = np.empty((time_count, member_count), dtype=float)
    completed = 0

    forecast_pool = None
    if selected_workers > 1:
        forecast_pool = PersistentAugmentedForecastPool(
            controller=nominal_controller,
            geometry=problem.geometry,
            initial_time=times[0],
            initial_states=initial_ensemble.filter_states,
            initial_vehicle_parameters=prior_parameters,
            initial_actuator_parameters=prior_actuators,
            worker_count=selected_workers,
        )
    try:
        for time_index in range(time_count):
            tracker.checkpoint()
            forecast_static = static_forecast[time_index]
            forecast_dynamic = dynamic_forecast[time_index]
            chart = GrapeFilterStateChart.from_ensemble(forecast_dynamic)
            forecast_coordinates = _encode_augmented(
                forecast_static, forecast_dynamic, chart
            )
            predicted_pose = chart.predicted_pose_ensemble(forecast_dynamic)
            observed_pose = chart.observed_pose_coordinates(
                observations.position[time_index],
                observations.orientation_xyzw[time_index],
            )
            update = deterministic_square_root_update(
                forecast_coordinates,
                predicted_pose,
                observed_pose,
                observation_covariance.matrix,
            )
            for stepper in steppers:
                stepper.trim_command_history(
                    times[time_index], selected_delay_chart.maximum_delay
                )
            _analyse_command_histories(steppers, predicted_pose, update)
            analysed_static = update.analysis_ensemble[
                :, :SHARED_STATIC_DIMENSION
            ].copy()
            decoded_dynamic = chart.decode_ensemble(
                update.analysis_ensemble[:, SHARED_STATIC_DIMENSION:],
                forecast_dynamic,
            )
            analysed_dynamic = tuple(
                _clip_dynamic_state(state, problem.actuator_parameters)
                for state in decoded_dynamic
            )
            parameters, actuators = _member_models(
                problem, analysed_static, selected_delay_chart
            )
            for member_index, (
                stepper,
                state,
                vehicle,
                actuator,
            ) in enumerate(
                zip(steppers, analysed_dynamic, parameters, actuators)
            ):
                stepper.replace_static_model(
                    controller=nominal_controller,
                    plant=FullSixDofPlant(vehicle, problem.geometry),
                    actuator_parameters=actuator,
                )
                stepper.replace_dynamic_state(
                    rigid_body_state=state.rigid,
                    controller_state=state.controller,
                    actuator_state=state.actuator,
                )
                model_mass[time_index, member_index] = vehicle.mass
                model_delay[time_index, member_index] = actuator.delay
            static_analysis.append(analysed_static)
            dynamic_analysis.append(analysed_dynamic)
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
                "augmented_filter",
                "Augmented parameter filtering",
                bag_id=bag_id,
                message="analysed observation {}/{}".format(
                    time_index + 1, time_count
                ),
            )

            if time_index + 1 == time_count:
                continue
            analysis_chart = GrapeFilterStateChart.from_ensemble(
                analysed_dynamic
            )
            analysis_coordinates = _encode_augmented(
                analysed_static, analysed_dynamic, analysis_chart
            )
            try:
                noise = exact_gaussian_ensemble(
                    np.zeros(BODY_WRENCH_DIMENSION),
                    np.diag(innovation_variance[time_index]),
                    member_count,
                    _transition_seed(selected_seed, time_index),
                    orthogonal_to=analysis_coordinates,
                )
            except ValueError as error:
                raise ValueError(
                    "ensemble cannot provide six process-noise directions "
                    "orthogonal to the current 51-D analysis anomalies"
                ) from error
            current_wrench = np.asarray(
                [state.residual_wrench for state in analysed_dynamic]
            )
            next_wrench, interval_wrench = ou_wrench_transition(
                current_wrench, factors.rho[time_index], noise
            )
            next_dynamic = []
            if forecast_pool is None:
                for member_index, stepper in enumerate(steppers):
                    tracker.checkpoint()
                    stepper.advance_interval(
                        times[time_index + 1],
                        problem.references[time_index],
                        interval_wrench[member_index],
                    )
                    state = stepper.state
                    next_dynamic.append(
                        GrapeFilterState(
                            rigid=state.rigid_body_state,
                            controller=state.controller_state,
                            actuator=state.actuator_state,
                            residual_wrench=next_wrench[member_index],
                        )
                    )
                    completed += 1
                    tracker.emit(
                        completed,
                        "augmented_forecast",
                        "Augmented closed-loop forecast",
                        bag_id=bag_id,
                        member_id=member_index,
                        message="propagated interval {}/{}".format(
                            time_index + 1, time_count - 1
                        ),
                    )
            else:
                (
                    next_stepper_states,
                    issued_commands,
                ) = forecast_pool.advance_interval(
                    start_time=times[time_index],
                    end_time=times[time_index + 1],
                    reference=problem.references[time_index],
                    analysis_states=analysed_dynamic,
                    vehicle_parameters=parameters,
                    actuator_parameters=actuators,
                    command_issue_times=tuple(
                        stepper.command_issue_times for stepper in steppers
                    ),
                    command_histories=tuple(
                        stepper.command_history_commands
                        for stepper in steppers
                    ),
                    interval_wrench=interval_wrench,
                    cancellation_token=cancellation,
                )
                for stepper, state, command in zip(
                    steppers, next_stepper_states, issued_commands
                ):
                    stepper.accept_external_interval_advance(state, command)
                for member_index, state in enumerate(next_stepper_states):
                    tracker.checkpoint()
                    next_dynamic.append(
                        GrapeFilterState(
                            rigid=state.rigid_body_state,
                            controller=state.controller_state,
                            actuator=state.actuator_state,
                            residual_wrench=next_wrench[member_index],
                        )
                    )
                    completed += 1
                    tracker.emit(
                        completed,
                        "augmented_forecast",
                        "Augmented closed-loop forecast",
                        bag_id=bag_id,
                        member_id=member_index,
                        message="propagated interval {}/{}".format(
                            time_index + 1, time_count - 1
                        ),
                    )
            static_forecast.append(analysed_static.copy())
            dynamic_forecast.append(tuple(next_dynamic))
    except BaseException:
        if forecast_pool is not None:
            forecast_pool.abort()
        raise
    else:
        if forecast_pool is not None:
            forecast_pool.close()

    static_smoothed = [None] * time_count
    dynamic_smoothed = [None] * time_count
    smoothing_gains = [None] * (time_count - 1)
    static_smoothed[-1] = static_analysis[-1]
    dynamic_smoothed[-1] = dynamic_analysis[-1]
    for time_index in range(time_count - 2, -1, -1):
        tracker.checkpoint()
        smoothed_static, smoothed_dynamic, gain = (
            _chart_aware_smoothing_step(
                static_analysis[time_index],
                dynamic_analysis[time_index],
                static_forecast[time_index + 1],
                dynamic_forecast[time_index + 1],
                static_smoothed[time_index + 1],
                dynamic_smoothed[time_index + 1],
                selected_rcond,
            )
        )
        static_smoothed[time_index] = smoothed_static
        dynamic_smoothed[time_index] = smoothed_dynamic
        smoothing_gains[time_index] = gain
        completed += 1
        tracker.emit(
            completed,
            "augmented_smoother",
            "Augmented fixed-interval smoothing",
            bag_id=bag_id,
            message="smoothed boundary {}/{}".format(
                time_index + 1, time_count
            ),
        )

    static_forecast_array = np.transpose(
        np.asarray(static_forecast), (1, 0, 2)
    )
    static_analysis_array = np.transpose(
        np.asarray(static_analysis), (1, 0, 2)
    )
    static_smoothed_array = np.transpose(
        np.asarray(static_smoothed), (1, 0, 2)
    )
    smoothed_wrench = np.transpose(
        np.asarray(
            [
                [state.residual_wrench for state in ensemble]
                for ensemble in dynamic_smoothed
            ]
        ),
        (1, 0, 2),
    )
    final_command_history = np.asarray(
        [
            [
                _encode_command(command)
                for command in stepper.command_history_commands
            ]
            for stepper in steppers
        ],
        dtype=float,
    )
    return AugmentedParameterFilterResult(
        member_id=initial_ensemble.member_id,
        times=times,
        prior_static_ensemble=prior_static,
        final_static_ensemble=static_analysis_array[:, -1, :],
        static_forecast_ensemble=static_forecast_array,
        static_analysis_ensemble=static_analysis_array,
        static_smoothed_ensemble=static_smoothed_array,
        dynamic_forecast_state_ensembles=tuple(dynamic_forecast),
        dynamic_analysis_state_ensembles=tuple(dynamic_analysis),
        dynamic_smoothed_state_ensembles=tuple(dynamic_smoothed),
        smoothed_wrench_ensemble=smoothed_wrench,
        filter_log_likelihood=math.fsum(
            float(value) for value in log_likelihood
        ),
        filter_log_likelihood_by_time=log_likelihood,
        filter_nis=nis,
        smoothing_gains=tuple(smoothing_gains),
        command_issue_times=tuple(
            stepper.command_issue_times for stepper in steppers
        ),
        final_command_history_ensemble=final_command_history,
        maximum_delay=selected_delay_chart.maximum_delay,
        applied_model_mass=model_mass.T,
        applied_model_delay=model_delay.T,
    )


__all__ = [
    "AugmentedParameterFilterResult",
    "COMMAND_COORDINATE_DIMENSION",
    "run_augmented_parameter_filter",
]
