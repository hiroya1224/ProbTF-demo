"""Audited real-rosbag execution of the sparse batch estimation backend."""

from dataclasses import dataclass
import time
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.batch.dynamics_moments import (
    evaluate_prepared_dynamics_intervals,
)
from grape_param_estim.batch.em_loop import (
    LaplaceEmResult,
    LaplaceEmSettings,
    run_laplace_em,
)
from grape_param_estim.batch.evidence import StaticLaplaceGeometry
from grape_param_estim.batch.graph_builder import build_initial_batch_state
from grape_param_estim.batch.lag_profile import (
    LagProfileResult,
    LagProfileSettings,
)
from grape_param_estim.batch.lm import LMSettings
from grape_param_estim.batch.preparation import (
    PreparationSelection,
    prepare_fixed_batch_graph_data,
)
from grape_param_estim.batch.state import StateScaling
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.batch_request import BatchEstimationRequest
from grape_param_estim.estimation import (
    FixedGraphLaplaceSolution,
    SparseLaplaceEStepSolver,
    make_fixed_q_laplace_problem_factory,
)
from grape_param_estim.initialization import (
    FlightInitialization,
    build_flight_initialization,
)
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.posterior.delayed_acceptance import (
    DelayedAcceptanceSampler,
    PosteriorPoint,
    QuadraticSurrogate,
    build_ridge_aware_proposal,
)
from grape_param_estim.posterior.laplace_target import LaplaceMarginalTarget
from grape_param_estim.posterior.run import (
    McmcRunResult,
    McmcRunSettings,
    initialize_mcmc_chains,
    run_mcmc_chains,
)
from grape_param_estim.real_rosbag import load_flight_data
from grape_param_estim.sensor_models import FlightData
from grape_param_estim.system import (
    ActuatorParameters,
    GrapeGeometry,
    VehicleParameters,
)


ProgressCallback = Callable[[str, int, int, str], None]
CancellationCheck = Callable[[], bool]
FlightLoader = Callable[
    [Mapping[str, object], bool, Optional[Callable[[], None]]], FlightData
]


@dataclass(frozen=True)
class RealEstimationInputs:
    """Authenticated sensor data and fixed physical-model conventions."""

    request: BatchEstimationRequest
    flight_data: Tuple[FlightData, ...]
    initializations: Tuple[FlightInitialization, ...]
    parameter_chart: VehicleParameterChart
    geometry: GrapeGeometry
    actuator_parameters: ActuatorParameters
    scaling: StateScaling
    loading_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.request, BatchEstimationRequest):
            raise TypeError("request must be BatchEstimationRequest")
        if (
            type(self.flight_data) is not tuple
            or tuple(value.bag_id for value in self.flight_data)
            != self.request.bag_ids
        ):
            raise ValueError("flight_data must match request bag order")
        if (
            type(self.initializations) is not tuple
            or tuple(value.bag_id for value in self.initializations)
            != self.request.bag_ids
        ):
            raise ValueError("initializations must match request bag order")
        for value, expected, name in (
            (self.parameter_chart, VehicleParameterChart, "parameter_chart"),
            (self.geometry, GrapeGeometry, "geometry"),
            (self.actuator_parameters, ActuatorParameters, "actuator_parameters"),
            (self.scaling, StateScaling, "scaling"),
        ):
            if not isinstance(value, expected):
                raise TypeError("{} has an invalid type".format(name))
        elapsed = float(self.loading_seconds)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("loading_seconds must be finite and non-negative")
        object.__setattr__(self, "loading_seconds", elapsed)


@dataclass(frozen=True)
class DelayUncertaintyEstimate:
    standard_deviation_seconds: float
    source: str
    curvature: Optional[float]

    def __post_init__(self) -> None:
        standard_deviation = float(self.standard_deviation_seconds)
        if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
            raise ValueError("delay standard deviation must be positive")
        if (
            not isinstance(self.source, str)
            or not self.source
            or self.source.strip() != self.source
        ):
            raise ValueError("delay uncertainty source must be canonical text")
        curvature = self.curvature
        if curvature is not None:
            curvature = float(curvature)
            if not np.isfinite(curvature) or curvature <= 0.0:
                raise ValueError("delay curvature must be positive when present")
        object.__setattr__(
            self, "standard_deviation_seconds", standard_deviation
        )
        object.__setattr__(self, "curvature", curvature)


@dataclass(frozen=True)
class ModeEstimationResult:
    mode_id: str
    em: LaplaceEmResult
    final_solution: FixedGraphLaplaceSolution
    static_geometry: StaticLaplaceGeometry
    lag_profile_history: Tuple[LagProfileResult, ...]
    final_q_lag_profile_history: Tuple[LagProfileResult, ...]
    delay_uncertainty: DelayUncertaintyEstimate
    nonlinear_iteration_seconds: Tuple[float, ...]
    em_iteration_seconds: Tuple[float, ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.mode_id, str) or not self.mode_id:
            raise ValueError("mode_id must be non-empty")
        if not isinstance(self.em, LaplaceEmResult):
            raise TypeError("em must be LaplaceEmResult")
        if not isinstance(self.final_solution, FixedGraphLaplaceSolution):
            raise TypeError("final_solution must be FixedGraphLaplaceSolution")
        if not isinstance(self.static_geometry, StaticLaplaceGeometry):
            raise TypeError("static_geometry must be StaticLaplaceGeometry")
        if (
            type(self.lag_profile_history) is not tuple
            or not self.lag_profile_history
            or any(
                not isinstance(value, LagProfileResult)
                for value in self.lag_profile_history
            )
        ):
            raise ValueError("lag_profile_history cannot be empty")
        if (
            type(self.final_q_lag_profile_history) is not tuple
            or any(
                not isinstance(value, LagProfileResult)
                for value in self.final_q_lag_profile_history
            )
        ):
            raise TypeError(
                "final_q_lag_profile_history must contain LagProfileResult values"
            )
        if any(
            all(value is not known for known in self.lag_profile_history)
            for value in self.final_q_lag_profile_history
        ):
            raise ValueError(
                "final-Q lag profiles must be part of chronological history"
            )
        if not isinstance(self.delay_uncertainty, DelayUncertaintyEstimate):
            raise TypeError(
                "delay_uncertainty must be DelayUncertaintyEstimate"
            )
        for name in (
            "nonlinear_iteration_seconds",
            "em_iteration_seconds",
        ):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or not values
                or not np.all(np.isfinite(values))
                or any(float(value) < 0.0 for value in values)
            ):
                raise ValueError(
                    "{} must contain measured non-negative timings".format(
                        name
                    )
                )
            object.__setattr__(
                self, name, tuple(float(value) for value in values)
            )
        elapsed = float(self.elapsed_seconds)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        object.__setattr__(self, "elapsed_seconds", elapsed)


@dataclass(frozen=True)
class RealEstimationResult:
    inputs: RealEstimationInputs
    modes: Tuple[ModeEstimationResult, ...]
    selected_mode_id: str
    mcmc: Optional[McmcRunResult]
    mcmc_target_seconds: Tuple[float, ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, RealEstimationInputs):
            raise TypeError("inputs must be RealEstimationInputs")
        if (
            type(self.modes) is not tuple
            or not self.modes
            or any(not isinstance(value, ModeEstimationResult) for value in self.modes)
        ):
            raise ValueError("modes cannot be empty")
        identifiers = tuple(value.mode_id for value in self.modes)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("mode IDs must be unique")
        if self.selected_mode_id not in identifiers:
            raise ValueError("selected_mode_id is absent")
        if self.mcmc is not None and not isinstance(self.mcmc, McmcRunResult):
            raise TypeError("mcmc must be McmcRunResult or None")
        enabled = bool(self.inputs.request.payload["mcmc_settings"]["enabled"])
        if enabled != (self.mcmc is not None):
            raise ValueError("MCMC result presence disagrees with request")
        timings = self.mcmc_target_seconds
        if (
            type(timings) is not tuple
            or not np.all(np.isfinite(timings))
            or any(float(value) < 0.0 for value in timings)
        ):
            raise ValueError(
                "mcmc_target_seconds must contain non-negative timings"
            )
        if enabled != bool(timings):
            raise ValueError(
                "MCMC target timings presence disagrees with request"
            )
        object.__setattr__(
            self,
            "mcmc_target_seconds",
            tuple(float(value) for value in timings),
        )
        elapsed = float(self.elapsed_seconds)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        object.__setattr__(self, "elapsed_seconds", elapsed)

    @property
    def selected_mode(self) -> ModeEstimationResult:
        return next(
            value
            for value in self.modes
            if value.mode_id == self.selected_mode_id
        )


def production_state_scaling() -> StateScaling:
    """Return fixed physical-per-scaled-coordinate units for real runs."""

    return StateScaling(
        {
            VariableKind.STATIC_PARAMETERS: 0.1,
            VariableKind.GYRO_BIAS: 0.01,
            VariableKind.ACCELEROMETER_BIAS: 0.1,
            VariableKind.POSITION: 0.1,
            VariableKind.ORIENTATION_TANGENT: 0.1,
            VariableKind.LINEAR_VELOCITY: 0.5,
            VariableKind.ANGULAR_VELOCITY: 0.5,
            VariableKind.CONTROLLER_INTEGRAL: 0.1,
            VariableKind.ACTUATOR_THRUST: 5.0,
            VariableKind.GIMBAL_ANGLE: 0.2,
        }
    )


def _default_flight_loader(
    bag: Mapping[str, object],
    include_accelerometer: bool,
    checkpoint: Optional[Callable[[], None]],
) -> FlightData:
    interval = tuple(float(value) for value in bag["interval_seconds"])
    return load_flight_data(
        path=str(bag["path"]),
        start_local=interval[0],
        end_local=interval[1],
        include_fc_specific_force=include_accelerometer,
        compute_sha256=True,
        checkpoint=checkpoint,
        bag_id=str(bag["bag_id"]),
    )


def prepare_real_estimation_inputs(
    request: BatchEstimationRequest,
    *,
    flight_loader: Optional[FlightLoader] = None,
    cancellation_requested: Optional[CancellationCheck] = None,
    progress: Optional[ProgressCallback] = None,
) -> RealEstimationInputs:
    """Authenticate bags and build initialization without resampling factors."""

    if not isinstance(request, BatchEstimationRequest):
        raise TypeError("request must be BatchEstimationRequest")
    loader = _default_flight_loader if flight_loader is None else flight_loader
    if not callable(loader):
        raise TypeError("flight_loader must be callable")
    for callback, name in (
        (cancellation_requested, "cancellation_requested"),
        (progress, "progress"),
    ):
        if callback is not None and not callable(callback):
            raise TypeError("{} must be callable".format(name))

    def checkpoint() -> None:
        if cancellation_requested is not None and cancellation_requested():
            raise RuntimeError("estimation_cancelled")

    started = time.perf_counter()
    flights = []
    initializations = []
    period = float(request.payload["knot_policy"]["period_seconds"])
    bags = tuple(request.payload["bags"])
    for index, bag in enumerate(bags):
        checkpoint()
        factor = bag["observation_factors"]["accelerometer"]
        flight = loader(bag, bool(factor["enabled"]), checkpoint)
        if not isinstance(flight, FlightData):
            raise TypeError("flight_loader must return FlightData")
        expected_digest = str(bag["sha256"])
        actual_digest = flight.provenance.bag_sha256
        labelled_actual = (
            actual_digest
            if actual_digest.startswith("sha256:")
            else "sha256:" + actual_digest
        )
        if labelled_actual != expected_digest:
            raise ValueError("loaded FlightData SHA-256 differs from request")
        initialization = build_flight_initialization(
            flight,
            period,
            allow_zero_integral_fallback=False,
        )
        flights.append(flight)
        initializations.append(initialization)
        if progress is not None:
            progress(
                "preparing_trajectory",
                index + 1,
                len(bags),
                "prepared {}".format(flight.bag_id),
            )
    return RealEstimationInputs(
        request=request,
        flight_data=tuple(flights),
        initializations=tuple(initializations),
        parameter_chart=VehicleParameterChart(VehicleParameters.nominal()),
        geometry=GrapeGeometry.grape(),
        # Constant command transport delay is an outer parameter.  The
        # actuator model's legacy delay field remains exactly zero, while all
        # response and saturation parameters are explicit request inputs.
        actuator_parameters=ActuatorParameters(
            thrust_time_constant=float(
                request.payload["actuator_model"][
                    "thrust_time_constant_seconds"
                ]
            ),
            gimbal_time_constant=float(
                request.payload["actuator_model"][
                    "gimbal_time_constant_seconds"
                ]
            ),
            delay=0.0,
            minimum_thrust=float(
                request.payload["actuator_model"][
                    "minimum_thrust_newtons"
                ]
            ),
            maximum_thrust=float(
                request.payload["actuator_model"][
                    "maximum_thrust_newtons"
                ]
            ),
            maximum_gimbal_angle=float(
                request.payload["actuator_model"][
                    "maximum_gimbal_angle_radians"
                ]
            ),
            maximum_gimbal_rate=float(
                request.payload["actuator_model"][
                    "maximum_gimbal_rate_radians_per_second"
                ]
            ),
        ),
        scaling=production_state_scaling(),
        loading_seconds=time.perf_counter() - started,
    )


def _mode_ids(request: BatchEstimationRequest) -> Tuple[str, ...]:
    return tuple(
        str(value["mode_id"])
        for value in request.payload["mode_hypotheses"]
    )


def _lm_settings(request: BatchEstimationRequest) -> LMSettings:
    return LMSettings(**dict(request.payload["solver_settings"]))


def _em_settings(request: BatchEstimationRequest) -> LaplaceEmSettings:
    return LaplaceEmSettings(**dict(request.payload["em_settings"]))


def _lag_settings(request: BatchEstimationRequest) -> LagProfileSettings:
    value = request.payload["delay"]
    bounds = value["bounds_seconds"]
    return LagProfileSettings(
        minimum_lag=float(bounds[0]),
        maximum_lag=float(bounds[1]),
        coarse_grid_points=int(value["coarse_grid_points"]),
        refinement_tolerance=float(
            value["refinement_tolerance_seconds"]
        ),
        maximum_refinement_evaluations=int(
            value["maximum_refinement_evaluations"]
        ),
    )


def _graph_factory(inputs: RealEstimationInputs, mode_id: str):
    def factory(q, delay, static_coordinate):
        return prepare_fixed_batch_graph_data(
            request=inputs.request,
            flight_data=inputs.flight_data,
            initializations=inputs.initializations,
            parameter_chart=inputs.parameter_chart,
            geometry=inputs.geometry,
            actuator_parameters=inputs.actuator_parameters,
            scaling=inputs.scaling,
            selection=PreparationSelection(
                mode_id=mode_id,
                fixed_delay_seconds=delay,
                q_diagonal=q,
                initial_parameter_coordinates=static_coordinate,
            ),
        )

    return factory


def estimate_delay_uncertainty(
    profiles: Sequence[LagProfileResult],
    bounds: Tuple[float, float],
) -> DelayUncertaintyEstimate:
    """Fit local profile curvature, falling back explicitly to its prior."""

    selected_profiles = tuple(profiles)
    if any(
        not isinstance(profile, LagProfileResult)
        for profile in selected_profiles
    ):
        raise TypeError("profiles must contain LagProfileResult values")
    points = {}
    for profile in selected_profiles:
        for point in profile.points:
            if point.converged and point.objective is not None:
                points[float(point.lag)] = float(point.objective)
    if len(points) >= 3:
        best_lag = min(points, key=lambda value: (points[value], value))
        nearest = sorted(points, key=lambda value: abs(value - best_lag))[:5]
        x = np.asarray(nearest, dtype=float) - best_lag
        y = np.asarray(tuple(points[value] for value in nearest), dtype=float)
        design = np.column_stack((np.ones(x.size), x, 0.5 * x * x))
        coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        curvature = float(coefficients[2])
        if rank == 3 and np.isfinite(curvature) and curvature > 0.0:
            return DelayUncertaintyEstimate(
                standard_deviation_seconds=float(np.sqrt(1.0 / curvature)),
                source="positive local quadratic profile curvature",
                curvature=curvature,
            )
    lower, upper = (float(value) for value in bounds)
    if not np.all(np.isfinite((lower, upper))) or lower >= upper:
        raise ValueError("delay bounds must be finite and increasing")
    return DelayUncertaintyEstimate(
        standard_deviation_seconds=(upper - lower) / np.sqrt(12.0),
        source="uniform delay prior because local profile curvature is unavailable",
        curvature=None,
    )


def estimate_mode(
    inputs: RealEstimationInputs,
    mode_id: str,
    *,
    cancellation_requested: Optional[CancellationCheck] = None,
    progress: Optional[ProgressCallback] = None,
) -> ModeEstimationResult:
    """Run lag-profiled Laplace-EM and final undamped geometry for one mode."""

    if mode_id not in _mode_ids(inputs.request):
        raise ValueError("mode_id is absent from the request")
    started = time.perf_counter()
    request = inputs.request
    q_value = request.payload["q"]
    delay_value = request.payload["delay"]
    initial_q = np.asarray(q_value["initial_diagonal"], dtype=float)
    q_floor = np.asarray(q_value["floor_diagonal"], dtype=float)
    initial_delay = float(delay_value["initial_seconds"])
    initial_static = np.asarray(
        request.payload["parameter_prior"]["mean_coordinate"], dtype=float
    )
    factory = _graph_factory(inputs, mode_id)
    prepared_initial = factory(initial_q, initial_delay, initial_static)
    initial_state = build_initial_batch_state(prepared_initial)
    dynamics = evaluate_prepared_dynamics_intervals(
        prepared_initial, initial_state
    )
    if not dynamics.intervals:
        raise ValueError("mode has no valid free-flight dynamics interval")
    lag_settings = _lag_settings(request)
    nonlinear_timings = []
    em_timings = []
    last_em_boundary = [time.perf_counter()]

    def lm_progress(record) -> None:
        nonlinear_timings.append(float(record.elapsed_seconds))
        if progress is not None:
            maximum = int(request.payload["solver_settings"]["maximum_iterations"])
            progress(
                "optimizing_full_trajectory",
                record.iteration + 1,
                maximum,
                "mode={} objective={:.9g}".format(
                    mode_id, record.objective_before
                ),
            )

    e_step = SparseLaplaceEStepSolver(
        factory,
        initial_static,
        _lm_settings(request),
        lag_settings,
        cancellation_requested=cancellation_requested,
        lm_progress=lm_progress,
    )

    def em_progress(record) -> None:
        now = time.perf_counter()
        em_timings.append(now - last_em_boundary[0])
        last_em_boundary[0] = now
        if progress is not None:
            maximum = int(request.payload["em_settings"]["maximum_iterations"])
            progress(
                "updating_model_error_covariance",
                record.iteration + 1,
                maximum,
                "mode={} lag={:.6f}s".format(
                    mode_id, record.output_step.lag
                ),
            )

    em = run_laplace_em(
        definition=prepared_initial.dynamics.q_definition,
        initial_q=initial_q,
        q_floor=q_floor,
        interval_time_steps=dynamics.time_step,
        initial_lag=initial_delay,
        solver=e_step,
        settings=_em_settings(request),
        initial_warm_start=initial_state,
        cancellation_requested=cancellation_requested,
        progress=em_progress,
    )
    final_step = em.final_step
    # Reuse the exact solve selected by EM.  Re-solving from its converged
    # state can legitimately improve the objective further, but then the MAP,
    # marginal objective, delay profile, and EM record no longer describe one
    # common posterior point.
    final_solution = e_step.take_solution_for_result(final_step)
    geometry = final_solution.static_geometry()
    profiles = tuple(e_step.profile_history)
    final_q_profiles = tuple(
        profile
        for profile_q, profile in zip(
            e_step.profile_q_history, e_step.profile_history
        )
        if np.array_equal(profile_q, final_step.q)
    )
    uncertainty = estimate_delay_uncertainty(
        final_q_profiles[-1:],
        (
            lag_settings.minimum_lag,
            lag_settings.maximum_lag,
        ),
    )
    return ModeEstimationResult(
        mode_id=mode_id,
        em=em,
        final_solution=final_solution,
        static_geometry=geometry,
        lag_profile_history=profiles,
        final_q_lag_profile_history=final_q_profiles[-1:],
        delay_uncertainty=uncertainty,
        nonlinear_iteration_seconds=tuple(nonlinear_timings),
        em_iteration_seconds=tuple(em_timings),
        elapsed_seconds=time.perf_counter() - started,
    )


def _uniform_delay_log_prior(bounds: Tuple[float, float]):
    lower, upper = bounds
    log_density = -float(np.log(upper - lower))

    def evaluate(delay: float) -> float:
        value = float(delay)
        return log_density if lower <= value <= upper else float("-inf")

    return evaluate


def sample_selected_mode(
    inputs: RealEstimationInputs,
    mode: ModeEstimationResult,
    *,
    cancellation_requested: Optional[CancellationCheck] = None,
    progress: Optional[ProgressCallback] = None,
    target_timing_callback: Optional[Callable[[float], None]] = None,
) -> McmcRunResult:
    """Run ridge-aware delayed-acceptance MCMC for one selected mode."""

    raw = inputs.request.payload["mcmc_settings"]
    if not bool(raw["enabled"]):
        raise ValueError("request does not enable MCMC")
    if target_timing_callback is not None and not callable(
        target_timing_callback
    ):
        raise TypeError("target_timing_callback must be callable")
    delay_raw = inputs.request.payload["delay"]
    bounds = tuple(float(value) for value in delay_raw["bounds_seconds"])
    final = mode.final_solution
    static_map = final.lm.state.value(final.lm.state.layout.variable_keys[0])
    map_point = PosteriorPoint(static_map, final.prepared.fixed_delay)
    information = np.zeros((19, 19), dtype=float)
    information[:18, :18] = (
        mode.static_geometry.information.posterior.hessian
    )
    information[-1, -1] = (
        1.0 / mode.delay_uncertainty.standard_deviation_seconds**2
    )
    target_factory = make_fixed_q_laplace_problem_factory(
        _graph_factory(inputs, mode.mode_id), final.prepared.dynamics.q
    )
    target = LaplaceMarginalTarget(
        target_factory,
        _uniform_delay_log_prior(bounds),
        _lm_settings(inputs.request),
    )
    map_started = time.perf_counter()
    try:
        map_evaluation = target(map_point, None)
    finally:
        if target_timing_callback is not None:
            target_timing_callback(time.perf_counter() - map_started)
    if not map_evaluation.successful:
        raise ValueError(
            "MCMC MAP target evaluation failed: {}".format(
                map_evaluation.failure_reason
            )
        )
    surrogate = QuadraticSurrogate(
        map_point, map_evaluation.log_density, information
    )
    proposal = build_ridge_aware_proposal(
        mode.static_geometry.information.likelihood.hessian,
        mode.static_geometry.exact_ridge_direction,
        delay_scale=float(raw["delay_scale_seconds"]),
        local_scale=float(raw["local_scale"]),
        exact_ridge_scale=float(raw["exact_ridge_scale"]),
        near_ridge_scale=float(raw["near_ridge_scale"]),
        identified_scale=float(raw["identified_scale"]),
        near_relative_threshold=float(raw["near_relative_threshold"]),
    )
    settings = McmcRunSettings(
        mode_id=mode.mode_id,
        chain_count=int(raw["chain_count"]),
        warmup_steps=int(raw["warmup_steps"]),
        retained_draws=int(raw["retained_draws"]),
        thinning=int(raw["thinning"]),
        random_seed=int(raw["random_seed"]),
        rhat_threshold=float(raw["rhat_threshold"]),
        minimum_effective_sample_size=float(
            raw["minimum_effective_sample_size"]
        ),
    )
    initializations = initialize_mcmc_chains(
        map_point,
        mode.static_geometry.covariance,
        mode.delay_uncertainty.standard_deviation_seconds,
        mode.static_geometry.exact_ridge_direction,
        bounds,
        settings.chain_count,
        settings.random_seed,
    )

    def sampler_factory(_chain_id: str) -> DelayedAcceptanceSampler:
        chain_target = LaplaceMarginalTarget(
            target_factory,
            _uniform_delay_log_prior(bounds),
            _lm_settings(inputs.request),
        )
        def timed_target(point, warm_start):
            target_started = time.perf_counter()
            try:
                return chain_target(point, warm_start)
            finally:
                if target_timing_callback is not None:
                    target_timing_callback(
                        time.perf_counter() - target_started
                    )

        return DelayedAcceptanceSampler(
            surrogate, proposal, bounds, timed_target
        )

    def mcmc_progress(
        chain_index, chain_count, completed, total, _step
    ) -> None:
        if progress is not None:
            overall_total = chain_count * total
            progress(
                "sampling_parameter_posterior",
                chain_index * total + completed,
                overall_total,
                "chain {}/{}".format(chain_index + 1, chain_count),
            )

    return run_mcmc_chains(
        sampler_factory,
        settings,
        initializations,
        mode.static_geometry.exact_ridge_direction,
        cancellation_requested=cancellation_requested,
        progress=mcmc_progress,
    )


def run_real_estimation(
    inputs: RealEstimationInputs,
    *,
    cancellation_requested: Optional[CancellationCheck] = None,
    progress: Optional[ProgressCallback] = None,
) -> RealEstimationResult:
    """Estimate every explicit mode, select by marginal objective, then sample."""

    if not isinstance(inputs, RealEstimationInputs):
        raise TypeError("inputs must be RealEstimationInputs")
    started = time.perf_counter()
    modes = tuple(
        estimate_mode(
            inputs,
            mode_id,
            cancellation_requested=cancellation_requested,
            progress=progress,
        )
        for mode_id in _mode_ids(inputs.request)
    )
    selected = min(
        modes,
        key=lambda value: (
            value.em.final_step.approximate_marginal_objective,
            value.mode_id,
        ),
    )
    if progress is not None:
        progress(
            "computing_local_posterior_geometry",
            0,
            1,
            "validating selected reduced Hessian",
        )
        progress(
            "computing_local_posterior_geometry",
            1,
            1,
            "selected mode={} local geometry ready".format(
                selected.mode_id
            ),
        )
    mcmc_target_seconds = []
    mcmc = (
        sample_selected_mode(
            inputs,
            selected,
            cancellation_requested=cancellation_requested,
            progress=progress,
            target_timing_callback=mcmc_target_seconds.append,
        )
        if bool(inputs.request.payload["mcmc_settings"]["enabled"])
        else None
    )
    return RealEstimationResult(
        inputs=inputs,
        modes=modes,
        selected_mode_id=selected.mode_id,
        mcmc=mcmc,
        mcmc_target_seconds=tuple(mcmc_target_seconds),
        elapsed_seconds=time.perf_counter() - started,
    )


__all__ = [
    "CancellationCheck",
    "DelayUncertaintyEstimate",
    "FlightLoader",
    "ModeEstimationResult",
    "ProgressCallback",
    "RealEstimationInputs",
    "RealEstimationResult",
    "estimate_delay_uncertainty",
    "estimate_mode",
    "prepare_real_estimation_inputs",
    "production_state_scaling",
    "run_real_estimation",
    "sample_selected_mode",
]
