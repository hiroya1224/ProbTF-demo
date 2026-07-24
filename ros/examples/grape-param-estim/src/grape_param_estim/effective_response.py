"""Low-dimensional effective plant response and posterior fitting.

The model describes the response that matters to closed-loop control, not a
claim about uniquely identified physical mass or thrust coefficients.  It
supports axis effectiveness/cross coupling, actuator delay and first-order
lag, velocity damping, bias, and episode random effects.

Fitting consumes velocity transitions from trajectory-posterior samples.  A
batch explicitly marked as a raw mocap numerical derivative is rejected so
that the old errors-in-variables likelihood cannot be reintroduced silently.
"""

from dataclasses import dataclass
from math import lgamma, pi
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


RESPONSE_DIMENSION = 6


def _finite_vector(values, name, positive=False, nonnegative=False):
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(RESPONSE_DIMENSION, float(array))
    if array.shape != (RESPONSE_DIMENSION,) or not np.all(np.isfinite(array)):
        raise ValueError("{} must be a finite scalar or six-vector".format(name))
    if positive and np.any(array <= 0.0):
        raise ValueError("{} must be strictly positive".format(name))
    if nonnegative and np.any(array < 0.0):
        raise ValueError("{} must be non-negative".format(name))
    output = np.array(array, copy=True)
    output.setflags(write=False)
    return output


def _finite_matrix(values, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (RESPONSE_DIMENSION, RESPONSE_DIMENSION) or not np.all(
        np.isfinite(array)
    ):
        raise ValueError("{} must be a finite 6x6 matrix".format(name))
    output = np.array(array, copy=True)
    output.setflags(write=False)
    return output


def _logsumexp(values):
    array = np.asarray(values, dtype=float)
    maximum = float(np.max(array))
    if not np.isfinite(maximum):
        return maximum
    return maximum + float(np.log(np.sum(np.exp(array - maximum))))


@dataclass(frozen=True)
class ResponseState:
    generalized_position: np.ndarray
    generalized_velocity: np.ndarray
    actuator_state: np.ndarray

    def __post_init__(self):
        for name in (
            "generalized_position",
            "generalized_velocity",
            "actuator_state",
        ):
            object.__setattr__(
                self, name, _finite_vector(getattr(self, name), name)
            )


@dataclass(frozen=True)
class EffectiveResponseParameters:
    effectiveness: np.ndarray
    time_constant_s: np.ndarray
    delay_s: np.ndarray
    damping: np.ndarray
    bias: np.ndarray
    episode_id: str = ""

    def __post_init__(self):
        object.__setattr__(
            self, "effectiveness", _finite_matrix(self.effectiveness, "effectiveness")
        )
        object.__setattr__(
            self,
            "time_constant_s",
            _finite_vector(self.time_constant_s, "time_constant_s", positive=True),
        )
        object.__setattr__(
            self, "delay_s", _finite_vector(self.delay_s, "delay_s", nonnegative=True)
        )
        object.__setattr__(
            self, "damping", _finite_vector(self.damping, "damping", nonnegative=True)
        )
        object.__setattr__(self, "bias", _finite_vector(self.bias, "bias"))

    def as_vector(self):
        result = np.concatenate(
            (
                self.effectiveness.reshape(-1),
                self.time_constant_s,
                self.delay_s,
                self.damping,
                self.bias,
            )
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class TransitionResult:
    state: ResponseState
    generalized_acceleration: np.ndarray
    delayed_command: np.ndarray

    def __post_init__(self):
        if not isinstance(self.state, ResponseState):
            raise TypeError("state must be ResponseState")
        object.__setattr__(
            self,
            "generalized_acceleration",
            _finite_vector(self.generalized_acceleration, "generalized_acceleration"),
        )
        object.__setattr__(
            self, "delayed_command", _finite_vector(self.delayed_command, "delayed_command")
        )


def delayed_command(
    command_times: np.ndarray,
    commands: np.ndarray,
    requested_time: float,
    delay_s: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate only commands available by ``requested_time``."""

    times = np.asarray(command_times, dtype=float).reshape(-1)
    values = np.asarray(commands, dtype=float)
    delays = _finite_vector(delay_s, "delay_s", nonnegative=True)
    requested = float(requested_time)
    if (
        times.size == 0
        or values.shape != (times.size, RESPONSE_DIMENSION)
        or not np.all(np.isfinite(times))
        or not np.all(np.isfinite(values))
        or np.any(np.diff(times) <= 0.0)
        or not np.isfinite(requested)
    ):
        raise ValueError("command history must be finite, aligned, and increasing")
    output = np.empty(RESPONSE_DIMENSION)
    for axis in range(RESPONSE_DIMENSION):
        query = requested - delays[axis]
        # np.interp uses boundary values outside the sampled interval.  The
        # upper boundary is restricted to commands already sent by requested.
        available = times <= requested
        if not np.any(available):
            raise ValueError("requested time precedes all command history")
        causal_times = times[available]
        causal_values = values[available, axis]
        output[axis] = np.interp(
            query,
            causal_times,
            causal_values,
            left=causal_values[0],
            right=causal_values[-1],
        )
    return output


class LowDimensionalEffectiveResponse:
    model_id = "low_dimensional_effective_response/v1"

    def transition(
        self,
        state: ResponseState,
        command_times: np.ndarray,
        commands: np.ndarray,
        timestamp: float,
        delta: float,
        parameters: EffectiveResponseParameters,
        process_noise: Optional[np.ndarray] = None,
    ) -> TransitionResult:
        if not isinstance(state, ResponseState):
            raise TypeError("state must be ResponseState")
        if not isinstance(parameters, EffectiveResponseParameters):
            raise TypeError("parameters must be EffectiveResponseParameters")
        dt = float(delta)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("delta must be finite and positive")
        delayed = delayed_command(
            command_times, commands, timestamp, parameters.delay_s
        )
        decay = np.exp(-dt / parameters.time_constant_s)
        actuator = decay * state.actuator_state + (1.0 - decay) * delayed
        acceleration = (
            parameters.effectiveness @ actuator
            - parameters.damping * state.generalized_velocity
            + parameters.bias
        )
        if process_noise is not None:
            acceleration = acceleration + _finite_vector(
                process_noise, "process_noise"
            )
        velocity = state.generalized_velocity + acceleration * dt
        position = (
            state.generalized_position
            + state.generalized_velocity * dt
            + 0.5 * acceleration * dt * dt
        )
        return TransitionResult(
            state=ResponseState(position, velocity, actuator),
            generalized_acceleration=acceleration,
            delayed_command=delayed,
        )

    def transition_log_likelihood(
        self,
        predicted: ResponseState,
        observed: ResponseState,
        position_sigma: np.ndarray,
        velocity_sigma: np.ndarray,
        degrees_of_freedom: float = 5.0,
    ) -> float:
        """Student-t one-step pose/velocity likelihood."""

        if not isinstance(predicted, ResponseState) or not isinstance(observed, ResponseState):
            raise TypeError("predicted and observed must be ResponseState")
        position_scale = _finite_vector(
            position_sigma, "position_sigma", positive=True
        )
        velocity_scale = _finite_vector(
            velocity_sigma, "velocity_sigma", positive=True
        )
        dof = float(degrees_of_freedom)
        if not np.isfinite(dof) or dof <= 0.0:
            raise ValueError("degrees_of_freedom must be finite and positive")
        residual = np.concatenate(
            (
                observed.generalized_position - predicted.generalized_position,
                observed.generalized_velocity - predicted.generalized_velocity,
            )
        )
        sigma = np.concatenate((position_scale, velocity_scale))
        scaled = residual / sigma
        normalizer = (
            lgamma(0.5 * (dof + 1.0))
            - lgamma(0.5 * dof)
            - 0.5 * np.log(dof * pi)
        )
        return float(
            np.sum(
                normalizer
                - np.log(sigma)
                - 0.5 * (dof + 1.0) * np.log1p(scaled * scaled / dof)
            )
        )


@dataclass(frozen=True)
class TrajectoryTransitionBatch:
    timestamps: np.ndarray
    generalized_position: np.ndarray
    generalized_velocity: np.ndarray
    commands: np.ndarray
    episode_id: str
    trajectory_sample_id: int = 0
    trajectory_weight: float = 1.0
    state_source: str = "trajectory_posterior"
    raw_mocap_derivative: bool = False

    def __post_init__(self):
        times = np.asarray(self.timestamps, dtype=float).reshape(-1)
        count = times.size
        if (
            count < 3
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("timestamps must contain at least three increasing samples")
        for name in ("generalized_position", "generalized_velocity", "commands"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (count, RESPONSE_DIMENSION) or not np.all(
                np.isfinite(values)
            ):
                raise ValueError("{} must have finite shape (N, 6)".format(name))
            copy = np.array(values, copy=True)
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)
        weight = float(self.trajectory_weight)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("trajectory_weight must be finite and positive")
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if self.raw_mocap_derivative:
            raise ValueError(
                "raw mocap derivatives are diagnostic-only and cannot enter the response likelihood"
            )
        if self.state_source not in (
            "trajectory_posterior",
            "synthetic_known_truth",
        ):
            raise ValueError(
                "state_source must be trajectory_posterior or synthetic_known_truth"
            )
        times_copy = np.array(times, copy=True)
        times_copy.setflags(write=False)
        object.__setattr__(self, "timestamps", times_copy)
        object.__setattr__(self, "trajectory_weight", weight)


@dataclass(frozen=True)
class EffectiveResponseFitConfig:
    delay_grid_s: np.ndarray
    time_constant_grid_s: np.ndarray
    ridge: float = 1.0e-6
    prior_scale: float = 10.0
    residual_sigma_floor: float = 1.0e-3
    posterior_sample_count: int = 256
    seed: int = 7

    def __post_init__(self):
        delays = np.asarray(self.delay_grid_s, dtype=float).reshape(-1)
        constants = np.asarray(self.time_constant_grid_s, dtype=float).reshape(-1)
        if (
            delays.size == 0
            or constants.size == 0
            or not np.all(np.isfinite(delays))
            or not np.all(np.isfinite(constants))
            or np.any(delays < 0.0)
            or np.any(constants <= 0.0)
        ):
            raise ValueError("delay/time-constant grids must be finite and valid")
        for name in ("ridge", "prior_scale", "residual_sigma_floor"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be finite and positive".format(name))
            object.__setattr__(self, name, value)
        count = int(self.posterior_sample_count)
        if count < 1:
            raise ValueError("posterior_sample_count must be positive")
        delay_copy = np.unique(delays)
        constant_copy = np.unique(constants)
        delay_copy.setflags(write=False)
        constant_copy.setflags(write=False)
        object.__setattr__(self, "delay_grid_s", delay_copy)
        object.__setattr__(self, "time_constant_grid_s", constant_copy)
        object.__setattr__(self, "posterior_sample_count", count)


@dataclass(frozen=True)
class IdentifiabilityReport:
    design_rank: int
    parameter_count: int
    condition_number: float
    per_axis_design_rank: np.ndarray
    per_axis_condition_number: np.ndarray
    singular_values: np.ndarray
    null_directions: np.ndarray
    posterior_maximum_absolute_correlation: float
    identifiable: bool
    scope: str


@dataclass(frozen=True)
class EffectiveResponsePosterior:
    samples: Tuple[EffectiveResponseParameters, ...]
    weights: np.ndarray
    grid_delay_s: np.ndarray
    grid_time_constant_s: np.ndarray
    grid_weights: np.ndarray
    identifiability: IdentifiabilityReport
    log_evidence: float
    approximation: str
    source_sample_ids: Tuple[int, ...]

    def __post_init__(self):
        samples = tuple(self.samples)
        if not samples or not all(
            isinstance(item, EffectiveResponseParameters) for item in samples
        ):
            raise ValueError("posterior requires response-parameter samples")
        weights = np.asarray(self.weights, dtype=float)
        if (
            weights.shape != (len(samples),)
            or np.any(weights < 0.0)
            or not np.isclose(np.sum(weights), 1.0)
        ):
            raise ValueError("posterior weights must be a probability vector")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "source_sample_ids", tuple(self.source_sample_ids))

    def mean_parameters(self) -> EffectiveResponseParameters:
        effectiveness = np.average(
            np.asarray([item.effectiveness for item in self.samples]),
            axis=0,
            weights=self.weights,
        )
        time_constant = np.average(
            np.asarray([item.time_constant_s for item in self.samples]),
            axis=0,
            weights=self.weights,
        )
        delay = np.average(
            np.asarray([item.delay_s for item in self.samples]),
            axis=0,
            weights=self.weights,
        )
        damping = np.average(
            np.asarray([item.damping for item in self.samples]),
            axis=0,
            weights=self.weights,
        )
        bias = np.average(
            np.asarray([item.bias for item in self.samples]),
            axis=0,
            weights=self.weights,
        )
        return EffectiveResponseParameters(
            effectiveness, time_constant, delay, damping, bias
        )

    def vector_credible_interval(self, probability=0.95):
        level = float(probability)
        if not 0.0 < level < 1.0:
            raise ValueError("probability must be in (0, 1)")
        values = np.asarray([item.as_vector() for item in self.samples])
        lower = np.empty(values.shape[1])
        upper = np.empty(values.shape[1])
        tail = 0.5 * (1.0 - level)
        for column in range(values.shape[1]):
            order = np.argsort(values[:, column])
            cumulative = np.cumsum(self.weights[order])
            lower[column] = np.interp(tail, cumulative, values[order, column])
            upper[column] = np.interp(
                1.0 - tail, cumulative, values[order, column]
            )
        return lower, upper


@dataclass(frozen=True)
class _GridFit:
    delay: float
    time_constant: float
    coefficients: np.ndarray
    coefficient_covariances: np.ndarray
    log_score: float
    design: np.ndarray
    residual_variance: np.ndarray


def _actuator_history(times, commands, delay, time_constant):
    state = np.array(commands[0], copy=True)
    output = np.empty_like(commands)
    delay_vector = np.full(RESPONSE_DIMENSION, delay)
    # output[i] is the actuator state used for transition i -> i+1.  The
    # exact discrete first-order update therefore occurs before storing the
    # row that is paired with (v[i+1] - v[i]) / dt.
    for index in range(len(times) - 1):
        dt = float(times[index + 1] - times[index])
        delayed = delayed_command(
            times[: index + 1], commands[: index + 1], times[index],
            delay_vector,
        )
        decay = np.exp(-dt / time_constant)
        state = decay * state + (1.0 - decay) * delayed
        output[index] = state
    output[-1] = state
    return output


def _fit_grid_point(
    batches: Sequence[TrajectoryTransitionBatch],
    delay: float,
    time_constant: float,
    config: EffectiveResponseFitConfig,
) -> _GridFit:
    designs = []
    targets = []
    weights = []
    for batch in batches:
        actuator = _actuator_history(
            batch.timestamps, batch.commands, delay, time_constant
        )
        delta = np.diff(batch.timestamps)
        acceleration = np.diff(batch.generalized_velocity, axis=0) / delta[:, None]
        velocity = batch.generalized_velocity[:-1]
        # Axis-specific damping is represented later by selecting the matching
        # velocity column; all actuator axes remain available for cross coupling.
        base = np.column_stack(
            (actuator[:-1], velocity, np.ones(len(delta)))
        )
        designs.append(base)
        targets.append(acceleration)
        weights.append(np.full(len(delta), batch.trajectory_weight))
    base_design = np.concatenate(designs, axis=0)
    target = np.concatenate(targets, axis=0)
    sample_weight = np.concatenate(weights)
    sample_weight = sample_weight / np.mean(sample_weight)
    coefficient_count = RESPONSE_DIMENSION + 2
    coefficients = np.empty((RESPONSE_DIMENSION, coefficient_count))
    covariances = np.empty(
        (RESPONSE_DIMENSION, coefficient_count, coefficient_count)
    )
    residual_variance = np.empty(RESPONSE_DIMENSION)
    score = 0.0
    axis_designs = []
    for axis in range(RESPONSE_DIMENSION):
        design = np.column_stack(
            (
                base_design[:, :RESPONSE_DIMENSION],
                -base_design[:, RESPONSE_DIMENSION + axis],
                base_design[:, -1],
            )
        )
        axis_designs.append(design)
        weighted_design = design * np.sqrt(sample_weight)[:, None]
        weighted_target = target[:, axis] * np.sqrt(sample_weight)
        precision = (
            weighted_design.T @ weighted_design
            + np.eye(coefficient_count)
            * (config.ridge + 1.0 / config.prior_scale ** 2)
        )
        mean = np.linalg.solve(
            precision, weighted_design.T @ weighted_target
        )
        residual = target[:, axis] - design @ mean
        variance = max(
            float(np.average(residual * residual, weights=sample_weight)),
            config.residual_sigma_floor ** 2,
        )
        covariance = _positive_semidefinite(
            np.linalg.inv(precision) * variance
        )
        coefficients[axis] = mean
        covariances[axis] = covariance
        residual_variance[axis] = variance
        # Integrated Gaussian score with a weak complexity penalty.  All grid
        # points have the same number of coefficients.
        score += -0.5 * float(
            np.sum(
                sample_weight
                * (
                    residual * residual / variance
                    + np.log(2.0 * np.pi * variance)
                )
            )
        )
        score += -0.5 * float(np.linalg.slogdet(precision)[1])
    return _GridFit(
        delay=float(delay),
        time_constant=float(time_constant),
        coefficients=coefficients,
        coefficient_covariances=covariances,
        log_score=score,
        design=np.stack(axis_designs, axis=0),
        residual_variance=residual_variance,
    )


def _positive_semidefinite(matrix):
    symmetric = 0.5 * (np.asarray(matrix) + np.asarray(matrix).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, 1.0e-12)) @ eigenvectors.T


def _identifiability(design, posterior_vectors):
    axis_designs = np.asarray(design, dtype=float)
    if axis_designs.ndim != 3 or axis_designs.shape[0] != RESPONSE_DIMENSION:
        raise ValueError("identifiability design must have one matrix per axis")
    coefficient_count = axis_designs.shape[2]
    per_axis_rank = np.empty(RESPONSE_DIMENSION, dtype=int)
    per_axis_condition = np.empty(RESPONSE_DIMENSION)
    all_singular_values = []
    embedded_null = []
    for axis in range(RESPONSE_DIMENSION):
        singular, right = None, None
        _, singular, right = np.linalg.svd(
            axis_designs[axis], full_matrices=True
        )
        tolerance = (
            singular[0]
            * max(axis_designs[axis].shape)
            * np.finfo(float).eps
            if singular.size
            else 0.0
        )
        axis_rank = int(np.sum(singular > tolerance))
        per_axis_rank[axis] = axis_rank
        per_axis_condition[axis] = (
            float(singular[0] / singular[-1])
            if axis_rank == coefficient_count and singular[-1] > 0.0
            else float("inf")
        )
        all_singular_values.extend(float(item) for item in singular)
        for direction in right[axis_rank:]:
            embedded = np.zeros(RESPONSE_DIMENSION * coefficient_count)
            start = axis * coefficient_count
            embedded[start : start + coefficient_count] = direction
            embedded_null.append(embedded)
    singular_values = np.asarray(sorted(all_singular_values, reverse=True))
    rank = int(np.sum(per_axis_rank))
    parameter_count = RESPONSE_DIMENSION * coefficient_count
    condition = (
        float(np.max(per_axis_condition))
        if np.all(np.isfinite(per_axis_condition))
        else float("inf")
    )
    null = (
        np.asarray(embedded_null)
        if embedded_null
        else np.empty((0, parameter_count))
    )
    if posterior_vectors.shape[0] > 1:
        covariance = np.cov(posterior_vectors, rowvar=False)
        standard = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        denominator = np.outer(standard, standard)
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0.0,
        )
        np.fill_diagonal(correlation, 0.0)
        maximum_correlation = float(np.max(np.abs(correlation)))
    else:
        maximum_correlation = 0.0
    return IdentifiabilityReport(
        design_rank=rank,
        parameter_count=parameter_count,
        condition_number=condition,
        per_axis_design_rank=per_axis_rank,
        per_axis_condition_number=per_axis_condition,
        singular_values=singular_values,
        null_directions=null,
        posterior_maximum_absolute_correlation=maximum_correlation,
        identifiable=rank == parameter_count and np.isfinite(condition),
        scope="conditional_linear_coefficients_given_delay_and_lag",
    )


def fit_effective_response(
    batches: Sequence[TrajectoryTransitionBatch],
    config: EffectiveResponseFitConfig,
) -> EffectiveResponsePosterior:
    """Fit a delay/lag mixture with Bayesian linear conditional posteriors."""

    data = tuple(batches)
    if not data or not all(
        isinstance(item, TrajectoryTransitionBatch) for item in data
    ):
        raise ValueError("at least one TrajectoryTransitionBatch is required")
    if not isinstance(config, EffectiveResponseFitConfig):
        raise TypeError("config must be EffectiveResponseFitConfig")
    grid_fits = []
    for delay in config.delay_grid_s:
        for time_constant in config.time_constant_grid_s:
            grid_fits.append(
                _fit_grid_point(data, float(delay), float(time_constant), config)
            )
    log_scores = np.asarray([item.log_score for item in grid_fits])
    log_normalizer = _logsumexp(log_scores)
    grid_weights = np.exp(log_scores - log_normalizer)
    grid_weights /= np.sum(grid_weights)
    rng = np.random.default_rng(int(config.seed))
    selected = rng.choice(
        len(grid_fits),
        size=config.posterior_sample_count,
        replace=True,
        p=grid_weights,
    )
    parameter_samples = []
    for fit_index in selected:
        fit = grid_fits[int(fit_index)]
        coefficients = np.empty_like(fit.coefficients)
        for axis in range(RESPONSE_DIMENSION):
            coefficients[axis] = rng.multivariate_normal(
                fit.coefficients[axis], fit.coefficient_covariances[axis]
            )
        effectiveness = coefficients[:, :RESPONSE_DIMENSION]
        damping = np.maximum(coefficients[:, RESPONSE_DIMENSION], 0.0)
        bias = coefficients[:, RESPONSE_DIMENSION + 1]
        parameter_samples.append(
            EffectiveResponseParameters(
                effectiveness=effectiveness,
                time_constant_s=np.full(
                    RESPONSE_DIMENSION, fit.time_constant
                ),
                delay_s=np.full(RESPONSE_DIMENSION, fit.delay),
                damping=damping,
                bias=bias,
            )
        )
    sample_weights = np.full(
        config.posterior_sample_count, 1.0 / config.posterior_sample_count
    )
    best_fit = grid_fits[int(np.argmax(grid_weights))]
    posterior_vectors = np.asarray(
        [item.as_vector() for item in parameter_samples]
    )
    report = _identifiability(best_fit.design, posterior_vectors)
    return EffectiveResponsePosterior(
        samples=tuple(parameter_samples),
        weights=sample_weights,
        grid_delay_s=np.asarray(
            [item.delay for item in grid_fits], dtype=float
        ),
        grid_time_constant_s=np.asarray(
            [item.time_constant for item in grid_fits], dtype=float
        ),
        grid_weights=grid_weights,
        identifiability=report,
        log_evidence=float(log_normalizer),
        approximation="delay_lag_grid_with_conditional_bayesian_linear_response",
        source_sample_ids=tuple(
            sorted(set(item.trajectory_sample_id for item in data))
        ),
    )


@dataclass(frozen=True)
class HierarchicalEffectiveResponsePosterior:
    population: EffectiveResponsePosterior
    episode_parameter_means: Mapping[str, EffectiveResponseParameters]
    episode_random_effect_covariance: np.ndarray
    approximation: str


def fit_hierarchical_effective_response(
    batches: Sequence[TrajectoryTransitionBatch],
    config: EffectiveResponseFitConfig,
    shrinkage_observations: float = 100.0,
) -> HierarchicalEffectiveResponsePosterior:
    """Empirical-Bayes partial pooling across episodes."""

    data = tuple(batches)
    population = fit_effective_response(data, config)
    population_mean = population.mean_parameters()
    grouped: Dict[str, list] = {}
    for item in data:
        grouped.setdefault(item.episode_id, []).append(item)
    episode_means = {}
    random_effects = []
    shrinkage = float(shrinkage_observations)
    if not np.isfinite(shrinkage) or shrinkage <= 0.0:
        raise ValueError("shrinkage_observations must be finite and positive")
    for episode_id, episode_batches in sorted(grouped.items()):
        local = fit_effective_response(episode_batches, config).mean_parameters()
        transitions = sum(item.timestamps.size - 1 for item in episode_batches)
        local_weight = transitions / (transitions + shrinkage)
        effectiveness = (
            local_weight * local.effectiveness
            + (1.0 - local_weight) * population_mean.effectiveness
        )
        time_constant = (
            local_weight * local.time_constant_s
            + (1.0 - local_weight) * population_mean.time_constant_s
        )
        delay = (
            local_weight * local.delay_s
            + (1.0 - local_weight) * population_mean.delay_s
        )
        damping = (
            local_weight * local.damping
            + (1.0 - local_weight) * population_mean.damping
        )
        bias = (
            local_weight * local.bias
            + (1.0 - local_weight) * population_mean.bias
        )
        pooled = EffectiveResponseParameters(
            effectiveness,
            time_constant,
            delay,
            damping,
            bias,
            episode_id=episode_id,
        )
        episode_means[episode_id] = pooled
        random_effects.append(pooled.as_vector() - population_mean.as_vector())
    dimension = population_mean.as_vector().size
    covariance = (
        np.cov(np.asarray(random_effects), rowvar=False)
        if len(random_effects) > 1
        else np.zeros((dimension, dimension))
    )
    covariance = np.atleast_2d(covariance)
    return HierarchicalEffectiveResponsePosterior(
        population=population,
        episode_parameter_means=episode_means,
        episode_random_effect_covariance=covariance,
        approximation="empirical_bayes_partial_pooling",
    )


__all__ = [
    "EffectiveResponseFitConfig",
    "EffectiveResponseParameters",
    "EffectiveResponsePosterior",
    "HierarchicalEffectiveResponsePosterior",
    "IdentifiabilityReport",
    "LowDimensionalEffectiveResponse",
    "RESPONSE_DIMENSION",
    "ResponseState",
    "TrajectoryTransitionBatch",
    "TransitionResult",
    "delayed_command",
    "fit_effective_response",
    "fit_hierarchical_effective_response",
]
