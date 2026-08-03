"""Filter-agnostic EM orchestration for stationary diagonal wrench ``Q``.

The injected expectation step owns filtering and fixed-interval smoothing.
This module only validates its per-bag output, computes the existing OU
sufficient statistics, performs the shared closed-form M-step, and records a
small deterministic iteration trace.
"""

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

from grape_param_estim.diagonal_q import (
    BODY_WRENCH_DIMENSION,
    BodyWrenchDiagonalCovariance,
    DiagonalQEmUpdate,
    diagonal_q_em_sufficient_statistics,
    shared_diagonal_q_m_step,
)
from grape_param_estim.progress import (
    CancellationToken,
    ProgressCallback,
    ProgressTracker,
)


LOG_Q_TOLERANCE_TERMINATION = "log_q_tolerance"
MAXIMUM_ITERATIONS_TERMINATION = "maximum_iterations"


def _bag_id(value, name: str = "bag_id") -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("{} must be a non-empty string".format(name))
    return value


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a positive integer".format(name))
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "{} must be a positive integer".format(name)
        ) from error
    if result != value or result <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return result


def _finite_scalar(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("{} must be finite".format(name)) from error
    if not np.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _finite_sum(values, name: str) -> float:
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError("{} is not representable".format(name)) from error
    if not np.isfinite(result):
        raise ValueError("{} is not representable".format(name))
    return result


def _component_vector(value, name: str, *, strictly_positive: bool) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        result = np.full(BODY_WRENCH_DIMENSION, float(result), dtype=float)
    invalid_sign = result <= 0.0 if strictly_positive else result < 0.0
    if (
        result.shape != (BODY_WRENCH_DIMENSION,)
        or np.any(~np.isfinite(result))
        or np.any(invalid_sign)
    ):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(
            "{} must contain one or six finite {} values".format(
                name, qualifier
            )
        )
    return result.copy()


@dataclass(frozen=True)
class DiagonalQInitialPilot:
    """One bag's explicit pose-pilot scale used only to initialise EM.

    ``stationary_standard_deviation`` is supplied by the pilot calibration;
    it is not estimated from an EM smoother output.  Its first three entries
    are in N and its final three entries are in N*m.
    """

    bag_id: str
    boundary_count: int
    stationary_standard_deviation: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "bag_id", _bag_id(self.bag_id))
        object.__setattr__(
            self,
            "boundary_count",
            _positive_integer(self.boundary_count, "boundary_count"),
        )
        object.__setattr__(
            self,
            "stationary_standard_deviation",
            _component_vector(
                self.stationary_standard_deviation,
                "stationary_standard_deviation",
                strictly_positive=False,
            ),
        )

    @property
    def stationary_variance(self) -> np.ndarray:
        with np.errstate(over="raise", invalid="raise"):
            try:
                result = self.stationary_standard_deviation**2
            except FloatingPointError as error:
                raise ValueError(
                    "pilot stationary variance is not representable"
                ) from error
        if np.any(~np.isfinite(result)):
            raise ValueError("pilot stationary variance is not representable")
        return result.copy()


@dataclass(frozen=True)
class DiagonalQBagExpectation:
    """One bag's E-step output with a member-aligned smoothed wrench path."""

    bag_id: str
    times: np.ndarray
    correlation_time: float
    smoothed_wrench: np.ndarray
    approx_log_likelihood: float

    def __post_init__(self) -> None:
        # The existing statistic builder is the single source of truth for
        # time grid, correlation time, member alignment, shape, and finite
        # value checks.
        statistics = diagonal_q_em_sufficient_statistics(
            self.bag_id,
            self.times,
            self.correlation_time,
            self.smoothed_wrench,
        )
        likelihood = _finite_scalar(
            self.approx_log_likelihood, "approx_log_likelihood"
        )
        wrench = np.asarray(self.smoothed_wrench, dtype=float)
        object.__setattr__(self, "bag_id", statistics.bag_id)
        object.__setattr__(self, "times", statistics.times.copy())
        object.__setattr__(
            self, "correlation_time", statistics.correlation_time
        )
        object.__setattr__(self, "smoothed_wrench", wrench.copy())
        object.__setattr__(self, "approx_log_likelihood", likelihood)

    @property
    def boundary_count(self) -> int:
        return int(self.times.size)

    @property
    def member_count(self) -> int:
        return int(self.smoothed_wrench.shape[0])

    @property
    def sufficient_statistics(self):
        """Build the audited OU sufficient statistics for this output."""

        return diagonal_q_em_sufficient_statistics(
            self.bag_id,
            self.times,
            self.correlation_time,
            self.smoothed_wrench,
        )


@dataclass(frozen=True)
class DiagonalQEmConfig:
    """Explicit EM iteration, convergence, and component-floor settings."""

    maximum_iterations: int
    log_q_tolerance: float
    component_floor: np.ndarray

    def __post_init__(self) -> None:
        maximum = _positive_integer(
            self.maximum_iterations, "maximum_iterations"
        )
        tolerance = _finite_scalar(
            self.log_q_tolerance, "log_q_tolerance"
        )
        if tolerance <= 0.0:
            raise ValueError("log_q_tolerance must be positive")
        floor = _component_vector(
            self.component_floor,
            "component_floor",
            strictly_positive=True,
        )
        object.__setattr__(self, "maximum_iterations", maximum)
        object.__setattr__(self, "log_q_tolerance", tolerance)
        object.__setattr__(self, "component_floor", floor)


def _ordered_pilots(
    pilots: Iterable[DiagonalQInitialPilot],
) -> Tuple[DiagonalQInitialPilot, ...]:
    try:
        values = tuple(pilots)
    except TypeError as error:
        raise TypeError(
            "pilots must be an iterable of DiagonalQInitialPilot"
        ) from error
    if not values:
        raise ValueError("at least one diagonal-Q pilot is required")
    if any(not isinstance(value, DiagonalQInitialPilot) for value in values):
        raise TypeError("pilots must contain DiagonalQInitialPilot values")
    ordered = tuple(sorted(values, key=lambda value: value.bag_id))
    identifiers = tuple(value.bag_id for value in ordered)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("diagonal-Q pilots must have unique bag IDs")
    return ordered


def _initial_covariance_from_ordered_pilots(
    pilots: Tuple[DiagonalQInitialPilot, ...], component_floor: np.ndarray
) -> BodyWrenchDiagonalCovariance:
    total_boundaries = sum(value.boundary_count for value in pilots)
    raw_variance = np.asarray(
        [
            _finite_sum(
                (
                    float(value.boundary_count)
                    * float(value.stationary_variance[component])
                    for value in pilots
                ),
                "pilot variance numerator",
            )
            / float(total_boundaries)
            for component in range(BODY_WRENCH_DIMENSION)
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(raw_variance)) or np.any(raw_variance < 0.0):
        raise ValueError("pilot stationary variance is not representable")
    return BodyWrenchDiagonalCovariance(
        np.maximum(raw_variance, component_floor)
    )


def initial_diagonal_q_from_pilots(
    pilots: Iterable[DiagonalQInitialPilot], component_floor
) -> BodyWrenchDiagonalCovariance:
    """Return ``sum(N_b * sigma_b**2) / sum(N_b)``, then apply the floor."""

    ordered = _ordered_pilots(pilots)
    floor = _component_vector(
        component_floor, "component_floor", strictly_positive=True
    )
    return _initial_covariance_from_ordered_pilots(ordered, floor)


def _maximum_absolute_log_q_change(
    before: BodyWrenchDiagonalCovariance,
    after: BodyWrenchDiagonalCovariance,
) -> float:
    change = np.abs(
        np.log(after.stationary_variance)
        - np.log(before.stationary_variance)
    )
    result = float(np.max(change))
    if not np.isfinite(result):
        raise ValueError("log-Q change is not representable")
    return result


@dataclass(frozen=True)
class DiagonalQEmIteration:
    """Compact deterministic trace of one completed E/M iteration."""

    iteration: int
    input_covariance: BodyWrenchDiagonalCovariance
    update: DiagonalQEmUpdate
    bag_approx_log_likelihoods: Tuple[Tuple[str, float], ...]
    approx_log_likelihood: float
    maximum_absolute_log_q_change: float
    converged: bool

    def __post_init__(self) -> None:
        selected_iteration = _positive_integer(self.iteration, "iteration")
        if not isinstance(
            self.input_covariance, BodyWrenchDiagonalCovariance
        ):
            raise TypeError(
                "input_covariance must be BodyWrenchDiagonalCovariance"
            )
        if not isinstance(self.update, DiagonalQEmUpdate):
            raise TypeError("update must be DiagonalQEmUpdate")
        try:
            likelihood_items = tuple(self.bag_approx_log_likelihoods)
        except TypeError as error:
            raise TypeError(
                "bag_approx_log_likelihoods must be an iterable"
            ) from error
        normalised_items = []
        for item in likelihood_items:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(
                    "bag likelihood items must be (bag_id, value) tuples"
                )
            normalised_items.append(
                (
                    _bag_id(item[0]),
                    _finite_scalar(item[1], "bag approx_log_likelihood"),
                )
            )
        normalised = tuple(normalised_items)
        identifiers = tuple(item[0] for item in normalised)
        if (
            not normalised
            or identifiers != tuple(sorted(identifiers))
            or len(set(identifiers)) != len(identifiers)
            or identifiers != self.update.bag_ids
        ):
            raise ValueError(
                "bag likelihoods must match the sorted M-step bag IDs"
            )
        total_likelihood = _finite_scalar(
            self.approx_log_likelihood, "approx_log_likelihood"
        )
        expected_likelihood = _finite_sum(
            (item[1] for item in normalised),
            "combined approx_log_likelihood",
        )
        if total_likelihood != expected_likelihood:
            raise ValueError(
                "approx_log_likelihood must equal the per-bag sum"
            )
        log_change = _finite_scalar(
            self.maximum_absolute_log_q_change,
            "maximum_absolute_log_q_change",
        )
        if log_change < 0.0:
            raise ValueError(
                "maximum_absolute_log_q_change cannot be negative"
            )
        expected_change = _maximum_absolute_log_q_change(
            self.input_covariance, self.update.covariance
        )
        if log_change != expected_change:
            raise ValueError(
                "maximum_absolute_log_q_change does not match the update"
            )
        if not isinstance(self.converged, (bool, np.bool_)):
            raise TypeError("converged must be boolean")
        object.__setattr__(self, "iteration", selected_iteration)
        object.__setattr__(
            self, "bag_approx_log_likelihoods", normalised
        )
        object.__setattr__(self, "approx_log_likelihood", total_likelihood)
        object.__setattr__(
            self, "maximum_absolute_log_q_change", log_change
        )
        object.__setattr__(self, "converged", bool(self.converged))

    @property
    def output_covariance(self) -> BodyWrenchDiagonalCovariance:
        return self.update.covariance


def _ordered_expectations(
    values,
    expected_pilots: Tuple[DiagonalQInitialPilot, ...],
    fixed_layout,
):
    try:
        expectations = tuple(values)
    except TypeError as error:
        raise TypeError(
            "expectation_step must return an iterable of "
            "DiagonalQBagExpectation"
        ) from error
    if any(
        not isinstance(value, DiagonalQBagExpectation)
        for value in expectations
    ):
        raise TypeError(
            "expectation_step must return DiagonalQBagExpectation values"
        )
    ordered = tuple(sorted(expectations, key=lambda value: value.bag_id))
    identifiers = tuple(value.bag_id for value in ordered)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("E-step outputs must have unique bag IDs")
    expected_ids = tuple(value.bag_id for value in expected_pilots)
    if identifiers != expected_ids:
        missing = tuple(sorted(set(expected_ids) - set(identifiers)))
        unexpected = tuple(sorted(set(identifiers) - set(expected_ids)))
        raise ValueError(
            "E-step bag set changed; missing={}, unexpected={}".format(
                missing, unexpected
            )
        )
    pilot_by_id = {value.bag_id: value for value in expected_pilots}
    for expectation in ordered:
        expected_count = pilot_by_id[expectation.bag_id].boundary_count
        if expectation.boundary_count != expected_count:
            raise ValueError(
                "E-step boundary count changed for bag {!r}".format(
                    expectation.bag_id
                )
            )
    if fixed_layout is not None:
        for expectation in ordered:
            expected_times, expected_correlation_time = fixed_layout[
                expectation.bag_id
            ]
            if not np.array_equal(expectation.times, expected_times):
                raise ValueError(
                    "E-step time grid changed for bag {!r}".format(
                        expectation.bag_id
                    )
                )
            if expectation.correlation_time != expected_correlation_time:
                raise ValueError(
                    "E-step correlation time changed for bag {!r}".format(
                        expectation.bag_id
                    )
                )
    return ordered


@dataclass(frozen=True)
class DiagonalQEmResult:
    """Completed diagonal-Q EM result with strict termination semantics."""

    config: DiagonalQEmConfig
    pilots: Tuple[DiagonalQInitialPilot, ...]
    initial_covariance: BodyWrenchDiagonalCovariance
    iterations: Tuple[DiagonalQEmIteration, ...]
    last_expectations: Tuple[DiagonalQBagExpectation, ...]
    covariance: BodyWrenchDiagonalCovariance
    converged: bool
    termination_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, DiagonalQEmConfig):
            raise TypeError("config must be DiagonalQEmConfig")
        pilots = _ordered_pilots(self.pilots)
        if not isinstance(
            self.initial_covariance, BodyWrenchDiagonalCovariance
        ):
            raise TypeError(
                "initial_covariance must be BodyWrenchDiagonalCovariance"
            )
        expected_initial = _initial_covariance_from_ordered_pilots(
            pilots, self.config.component_floor
        )
        if not np.array_equal(
            self.initial_covariance.stationary_variance,
            expected_initial.stationary_variance,
        ):
            raise ValueError("initial_covariance does not match the pilots")
        iterations = tuple(self.iterations)
        if not iterations or any(
            not isinstance(value, DiagonalQEmIteration)
            for value in iterations
        ):
            raise ValueError("result must contain completed EM iterations")
        if len(iterations) > self.config.maximum_iterations:
            raise ValueError("iteration trace exceeds maximum_iterations")
        expected_input = self.initial_covariance
        for index, value in enumerate(iterations, start=1):
            if value.iteration != index:
                raise ValueError("iteration trace must be contiguous and one-based")
            if not np.array_equal(
                value.input_covariance.stationary_variance,
                expected_input.stationary_variance,
            ):
                raise ValueError("iteration input covariance is not contiguous")
            expected_flag = (
                value.maximum_absolute_log_q_change
                <= self.config.log_q_tolerance
            )
            if value.converged != expected_flag:
                raise ValueError("iteration convergence flag is inconsistent")
            if value.converged and index != len(iterations):
                raise ValueError("EM must stop at the first converged iteration")
            expected_input = value.output_covariance
        expectations = _ordered_expectations(
            self.last_expectations, pilots, fixed_layout=None
        )
        last_likelihoods = tuple(
            (value.bag_id, value.approx_log_likelihood)
            for value in expectations
        )
        if last_likelihoods != iterations[-1].bag_approx_log_likelihoods:
            raise ValueError(
                "last expectations do not match the final iteration trace"
            )
        if not isinstance(self.covariance, BodyWrenchDiagonalCovariance):
            raise TypeError("covariance must be BodyWrenchDiagonalCovariance")
        if not np.array_equal(
            self.covariance.stationary_variance,
            iterations[-1].output_covariance.stationary_variance,
        ):
            raise ValueError("result covariance is not the final M-step output")
        if not isinstance(self.converged, (bool, np.bool_)):
            raise TypeError("converged must be boolean")
        selected_converged = bool(self.converged)
        if selected_converged:
            if (
                self.termination_reason != LOG_Q_TOLERANCE_TERMINATION
                or not iterations[-1].converged
            ):
                raise ValueError(
                    "converged EM must terminate by log_q_tolerance"
                )
        elif (
            self.termination_reason != MAXIMUM_ITERATIONS_TERMINATION
            or len(iterations) != self.config.maximum_iterations
            or iterations[-1].converged
        ):
            raise ValueError(
                "non-converged EM must exhaust maximum_iterations"
            )
        object.__setattr__(self, "pilots", pilots)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "last_expectations", expectations)
        object.__setattr__(self, "converged", selected_converged)

    @property
    def bag_ids(self) -> Tuple[str, ...]:
        return tuple(value.bag_id for value in self.pilots)


ExpectationStep = Callable[
    [BodyWrenchDiagonalCovariance, int],
    Iterable[DiagonalQBagExpectation],
]


def run_diagonal_q_em(
    pilots: Iterable[DiagonalQInitialPilot],
    expectation_step: ExpectationStep,
    config: DiagonalQEmConfig,
    *,
    progress_callback: Optional[ProgressCallback] = None,
    cancellation_token: Optional[CancellationToken] = None,
    run_id: str = "diagonal-q-em",
) -> DiagonalQEmResult:
    """Run one-based E/M iterations with an injected filtering implementation.

    The pilot bag IDs and boundary counts define the immutable selected bag
    set.  The first E-step additionally fixes each bag's exact time grid and
    correlation time; later E-steps must return the same problem layout.
    Cancellation is observed before and after every injected E-step and after
    every synchronous progress callback.
    """

    ordered_pilots = _ordered_pilots(pilots)
    if not callable(expectation_step):
        raise TypeError("expectation_step must be callable")
    if not isinstance(config, DiagonalQEmConfig):
        raise TypeError("config must be DiagonalQEmConfig")
    cancellation = (
        CancellationToken()
        if cancellation_token is None
        else cancellation_token
    )
    if not isinstance(cancellation, CancellationToken):
        raise TypeError("cancellation_token must be CancellationToken")
    tracker = ProgressTracker(
        run_id=run_id,
        total_units=config.maximum_iterations,
        callback=progress_callback,
        cancellation_token=cancellation,
    )
    initial_covariance = _initial_covariance_from_ordered_pilots(
        ordered_pilots, config.component_floor
    )
    covariance = initial_covariance
    fixed_layout = None
    trace = []
    last_expectations = None

    for iteration in range(1, config.maximum_iterations + 1):
        tracker.emit(
            completed_units=iteration - 1,
            stage_id="diagonal_q_expectation",
            stage_label="Diagonal Q expectation",
            iteration=iteration,
            maximum_iterations=config.maximum_iterations,
            message="starting diagonal-Q E-step {}/{}".format(
                iteration, config.maximum_iterations
            ),
        )
        tracker.checkpoint()
        input_covariance = BodyWrenchDiagonalCovariance(
            covariance.stationary_variance
        )
        callback_covariance = BodyWrenchDiagonalCovariance(
            input_covariance.stationary_variance
        )
        raw_expectations = expectation_step(callback_covariance, iteration)
        tracker.checkpoint()
        expectations = _ordered_expectations(
            raw_expectations, ordered_pilots, fixed_layout
        )
        if fixed_layout is None:
            fixed_layout = {
                value.bag_id: (
                    value.times.copy(), value.correlation_time
                )
                for value in expectations
            }
        statistics = tuple(
            value.sufficient_statistics for value in expectations
        )
        update = shared_diagonal_q_m_step(
            statistics, variance_floor=config.component_floor
        )
        log_change = _maximum_absolute_log_q_change(
            input_covariance, update.covariance
        )
        likelihoods = tuple(
            (value.bag_id, value.approx_log_likelihood)
            for value in expectations
        )
        total_likelihood = _finite_sum(
            (value[1] for value in likelihoods),
            "combined approx_log_likelihood",
        )
        converged = log_change <= config.log_q_tolerance
        record = DiagonalQEmIteration(
            iteration=iteration,
            input_covariance=input_covariance,
            update=update,
            bag_approx_log_likelihoods=likelihoods,
            approx_log_likelihood=total_likelihood,
            maximum_absolute_log_q_change=log_change,
            converged=converged,
        )
        trace.append(record)
        last_expectations = expectations
        covariance = update.covariance
        tracker.emit(
            completed_units=iteration,
            stage_id="diagonal_q_m_step",
            stage_label="Diagonal Q M-step",
            iteration=iteration,
            maximum_iterations=config.maximum_iterations,
            message=(
                "completed diagonal-Q iteration {}; "
                "max |delta log Q|={:.6g}"
            ).format(iteration, log_change),
        )
        tracker.checkpoint()
        if converged:
            return DiagonalQEmResult(
                config=config,
                pilots=ordered_pilots,
                initial_covariance=initial_covariance,
                iterations=tuple(trace),
                last_expectations=last_expectations,
                covariance=covariance,
                converged=True,
                termination_reason=LOG_Q_TOLERANCE_TERMINATION,
            )

    return DiagonalQEmResult(
        config=config,
        pilots=ordered_pilots,
        initial_covariance=initial_covariance,
        iterations=tuple(trace),
        last_expectations=last_expectations,
        covariance=covariance,
        converged=False,
        termination_reason=MAXIMUM_ITERATIONS_TERMINATION,
    )


__all__ = [
    "DiagonalQBagExpectation",
    "DiagonalQEmConfig",
    "DiagonalQEmIteration",
    "DiagonalQEmResult",
    "DiagonalQInitialPilot",
    "ExpectationStep",
    "LOG_Q_TOLERANCE_TERMINATION",
    "MAXIMUM_ITERATIONS_TERMINATION",
    "initial_diagonal_q_from_pilots",
    "run_diagonal_q_em",
]
