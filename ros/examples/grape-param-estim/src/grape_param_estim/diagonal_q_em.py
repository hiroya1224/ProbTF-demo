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
GENERALIZED_EM_UPDATE_REJECTED_TERMINATION = (
    "generalized_em_update_rejected"
)
BACKTRACKING_ACCEPTED = "accepted"
BACKTRACKING_LIKELIHOOD_DECREASE = "likelihood_decrease"
BACKTRACKING_NUMERICAL_FAILURE = "numerical_failure"
BACKTRACKING_OUTCOMES = (
    BACKTRACKING_ACCEPTED,
    BACKTRACKING_LIKELIHOOD_DECREASE,
    BACKTRACKING_NUMERICAL_FAILURE,
)
DEFAULT_BACKTRACKING_STEP_FRACTIONS = (
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
)


class DiagonalQExpectationNumericalError(RuntimeError):
    """A Q-dependent E-step failed numerically and may be backtracked."""


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


def _backtracking_step_fractions(value) -> Tuple[float, ...]:
    try:
        raw = tuple(value)
    except TypeError as error:
        raise ValueError(
            "backtracking_step_fractions must be an iterable"
        ) from error
    if any(isinstance(item, (bool, np.bool_)) for item in raw):
        raise ValueError(
            "backtracking_step_fractions must contain numeric fractions"
        )
    selected = tuple(
        _finite_scalar(item, "backtracking step fraction")
        for item in raw
    )
    if (
        not selected
        or selected[0] != 1.0
        or any(item <= 0.0 or item > 1.0 for item in selected)
        or any(
            following >= current
            for current, following in zip(selected, selected[1:])
        )
    ):
        raise ValueError(
            "backtracking_step_fractions must start at 1 and be strictly "
            "decreasing positive values"
        )
    return selected


def _likelihood_items(values, name: str) -> Tuple[Tuple[str, float], ...]:
    try:
        items = tuple(values)
    except TypeError as error:
        raise TypeError(
            "{} must be an iterable".format(name)
        ) from error
    normalised = []
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                "{} items must be (bag_id, value) tuples".format(name)
            )
        normalised.append(
            (
                _bag_id(item[0]),
                _finite_scalar(item[1], name),
            )
        )
    result = tuple(normalised)
    identifiers = tuple(item[0] for item in result)
    if (
        identifiers != tuple(sorted(identifiers))
        or len(set(identifiers)) != len(identifiers)
    ):
        raise ValueError(
            "{} bag IDs must be sorted and unique".format(name)
        )
    return result


def _log_interpolated_covariance(
    before: BodyWrenchDiagonalCovariance,
    target: BodyWrenchDiagonalCovariance,
    step_fraction: float,
) -> BodyWrenchDiagonalCovariance:
    alpha = _finite_scalar(step_fraction, "step_fraction")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("step_fraction must lie in [0, 1]")
    if alpha == 0.0:
        return BodyWrenchDiagonalCovariance(before.stationary_variance)
    if alpha == 1.0:
        return BodyWrenchDiagonalCovariance(target.stationary_variance)
    log_before = np.log(before.stationary_variance)
    log_target = np.log(target.stationary_variance)
    return BodyWrenchDiagonalCovariance(
        np.exp(log_before + alpha * (log_target - log_before))
    )


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
    backtracking_step_fractions: Tuple[float, ...] = (
        DEFAULT_BACKTRACKING_STEP_FRACTIONS
    )

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
        fractions = _backtracking_step_fractions(
            self.backtracking_step_fractions
        )
        object.__setattr__(self, "maximum_iterations", maximum)
        object.__setattr__(self, "log_q_tolerance", tolerance)
        object.__setattr__(self, "component_floor", floor)
        object.__setattr__(
            self, "backtracking_step_fractions", fractions
        )


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
class DiagonalQExpectationContext:
    """One physical E-step evaluation within an EM/backtracking search."""

    evaluation: int
    iteration: int
    trial: int
    step_fraction: float

    def __post_init__(self) -> None:
        evaluation = _positive_integer(self.evaluation, "evaluation")
        iteration = _positive_integer(self.iteration, "iteration")
        if isinstance(self.trial, (bool, np.bool_)):
            raise ValueError("trial must be a non-negative integer")
        try:
            trial = int(self.trial)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "trial must be a non-negative integer"
            ) from error
        if trial != self.trial or trial < 0:
            raise ValueError("trial must be a non-negative integer")
        fraction = _finite_scalar(self.step_fraction, "step_fraction")
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError("step_fraction must lie in [0, 1]")
        if (trial == 0) != (fraction == 0.0):
            raise ValueError(
                "only the initial input E-step may use trial and alpha zero"
            )
        object.__setattr__(self, "evaluation", evaluation)
        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "trial", trial)
        object.__setattr__(self, "step_fraction", fraction)


@dataclass(frozen=True)
class DiagonalQBacktrackingTrial:
    """Audited result of one positive log-Q line-search candidate."""

    trial: int
    step_fraction: float
    covariance: BodyWrenchDiagonalCovariance
    outcome: str
    bag_approx_log_likelihoods: Tuple[Tuple[str, float], ...] = ()
    approx_log_likelihood: Optional[float] = None

    def __post_init__(self) -> None:
        trial = _positive_integer(self.trial, "trial")
        fraction = _finite_scalar(self.step_fraction, "step_fraction")
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("step_fraction must lie in (0, 1]")
        if not isinstance(self.covariance, BodyWrenchDiagonalCovariance):
            raise TypeError(
                "covariance must be BodyWrenchDiagonalCovariance"
            )
        outcome = str(self.outcome)
        if outcome not in BACKTRACKING_OUTCOMES:
            raise ValueError("unknown backtracking outcome {!r}".format(outcome))
        likelihoods = _likelihood_items(
            self.bag_approx_log_likelihoods,
            "bag_approx_log_likelihoods",
        )
        if outcome == BACKTRACKING_NUMERICAL_FAILURE:
            if likelihoods or self.approx_log_likelihood is not None:
                raise ValueError(
                    "a numerical-failure trial cannot report likelihoods"
                )
            total = None
        else:
            if not likelihoods or self.approx_log_likelihood is None:
                raise ValueError(
                    "an evaluated trial must report likelihoods"
                )
            total = _finite_scalar(
                self.approx_log_likelihood, "approx_log_likelihood"
            )
            expected = _finite_sum(
                (item[1] for item in likelihoods),
                "combined approx_log_likelihood",
            )
            if total != expected:
                raise ValueError(
                    "approx_log_likelihood must equal the per-bag sum"
                )
        object.__setattr__(self, "trial", trial)
        object.__setattr__(self, "step_fraction", fraction)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "bag_approx_log_likelihoods", likelihoods)
        object.__setattr__(self, "approx_log_likelihood", total)


@dataclass(frozen=True)
class DiagonalQEmIteration:
    """Compact deterministic trace of one completed E/M iteration."""

    iteration: int
    input_covariance: BodyWrenchDiagonalCovariance
    update: DiagonalQEmUpdate
    accepted_covariance: BodyWrenchDiagonalCovariance
    accepted_step_fraction: float
    backtracking_trials: Tuple[DiagonalQBacktrackingTrial, ...]
    bag_approx_log_likelihoods: Tuple[Tuple[str, float], ...]
    approx_log_likelihood: float
    output_bag_approx_log_likelihoods: Tuple[Tuple[str, float], ...]
    output_approx_log_likelihood: float
    maximum_absolute_log_q_change: float
    accepted_maximum_absolute_log_q_change: float
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
        if not isinstance(
            self.accepted_covariance, BodyWrenchDiagonalCovariance
        ):
            raise TypeError(
                "accepted_covariance must be BodyWrenchDiagonalCovariance"
            )
        fraction = _finite_scalar(
            self.accepted_step_fraction, "accepted_step_fraction"
        )
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError("accepted_step_fraction must lie in [0, 1]")
        expected_accepted = _log_interpolated_covariance(
            self.input_covariance, self.update.covariance, fraction
        )
        if not np.array_equal(
            self.accepted_covariance.stationary_variance,
            expected_accepted.stationary_variance,
        ):
            raise ValueError(
                "accepted covariance is not the recorded log-Q step"
            )

        input_likelihoods = _likelihood_items(
            self.bag_approx_log_likelihoods,
            "bag_approx_log_likelihoods",
        )
        output_likelihoods = _likelihood_items(
            self.output_bag_approx_log_likelihoods,
            "output_bag_approx_log_likelihoods",
        )
        input_ids = tuple(item[0] for item in input_likelihoods)
        output_ids = tuple(item[0] for item in output_likelihoods)
        if (
            not input_likelihoods
            or input_ids != self.update.bag_ids
            or output_ids != self.update.bag_ids
        ):
            raise ValueError(
                "input/output likelihoods must match the M-step bag IDs"
            )
        input_total = _finite_scalar(
            self.approx_log_likelihood, "approx_log_likelihood"
        )
        expected_input_total = _finite_sum(
            (item[1] for item in input_likelihoods),
            "combined approx_log_likelihood",
        )
        output_total = _finite_scalar(
            self.output_approx_log_likelihood,
            "output_approx_log_likelihood",
        )
        expected_output_total = _finite_sum(
            (item[1] for item in output_likelihoods),
            "combined output_approx_log_likelihood",
        )
        if (
            input_total != expected_input_total
            or output_total != expected_output_total
        ):
            raise ValueError(
                "input/output likelihoods must equal their per-bag sums"
            )

        target_change = _finite_scalar(
            self.maximum_absolute_log_q_change,
            "maximum_absolute_log_q_change",
        )
        accepted_change = _finite_scalar(
            self.accepted_maximum_absolute_log_q_change,
            "accepted_maximum_absolute_log_q_change",
        )
        if target_change < 0.0 or accepted_change < 0.0:
            raise ValueError(
                "log-Q changes cannot be negative"
            )
        expected_target_change = _maximum_absolute_log_q_change(
            self.input_covariance, self.update.covariance
        )
        expected_accepted_change = _maximum_absolute_log_q_change(
            self.input_covariance, self.accepted_covariance
        )
        if (
            target_change != expected_target_change
            or accepted_change != expected_accepted_change
        ):
            raise ValueError(
                "log-Q changes do not match target and accepted updates"
            )

        try:
            trials = tuple(self.backtracking_trials)
        except TypeError as error:
            raise TypeError(
                "backtracking_trials must be an iterable"
            ) from error
        if not trials or any(
            not isinstance(value, DiagonalQBacktrackingTrial)
            for value in trials
        ):
            raise ValueError(
                "backtracking_trials must contain audited trials"
            )
        if tuple(value.trial for value in trials) != tuple(
            range(1, len(trials) + 1)
        ):
            raise ValueError("backtracking trials must be contiguous")
        if any(
            following.step_fraction >= current.step_fraction
            for current, following in zip(trials, trials[1:])
        ):
            raise ValueError(
                "backtracking trial fractions must be strictly decreasing"
            )
        for trial in trials:
            expected_covariance = _log_interpolated_covariance(
                self.input_covariance,
                self.update.covariance,
                trial.step_fraction,
            )
            if not np.array_equal(
                trial.covariance.stationary_variance,
                expected_covariance.stationary_variance,
            ):
                raise ValueError(
                    "backtracking trial covariance is inconsistent"
                )
            if trial.outcome != BACKTRACKING_NUMERICAL_FAILURE:
                trial_ids = tuple(
                    item[0] for item in trial.bag_approx_log_likelihoods
                )
                if trial_ids != self.update.bag_ids:
                    raise ValueError(
                        "trial likelihoods must match the M-step bag IDs"
                    )
                if trial.outcome == BACKTRACKING_LIKELIHOOD_DECREASE:
                    if not trial.approx_log_likelihood < input_total:
                        raise ValueError(
                            "a likelihood-decrease trial must decrease"
                        )
                elif not trial.approx_log_likelihood >= input_total:
                    raise ValueError(
                        "an accepted trial must not decrease likelihood"
                    )
        accepted_trials = tuple(
            value for value in trials
            if value.outcome == BACKTRACKING_ACCEPTED
        )
        if fraction == 0.0:
            if accepted_trials:
                raise ValueError("a rejected update cannot have an accepted trial")
            if output_likelihoods != input_likelihoods or output_total != input_total:
                raise ValueError(
                    "a rejected update must retain its input likelihood"
                )
        else:
            if accepted_trials != (trials[-1],):
                raise ValueError(
                    "the final positive trial must be the sole accepted trial"
                )
            accepted_trial = accepted_trials[0]
            if (
                fraction != accepted_trial.step_fraction
                or not np.array_equal(
                    self.accepted_covariance.stationary_variance,
                    accepted_trial.covariance.stationary_variance,
                )
                or output_likelihoods
                != accepted_trial.bag_approx_log_likelihoods
                or output_total != accepted_trial.approx_log_likelihood
            ):
                raise ValueError(
                    "accepted output must match the accepted trial"
                )
            if output_total < input_total:
                raise ValueError(
                    "accepted output likelihood cannot decrease"
                )
        if not isinstance(self.converged, (bool, np.bool_)):
            raise TypeError("converged must be boolean")
        if fraction == 0.0 and bool(self.converged):
            raise ValueError("a rejected update cannot be converged")
        object.__setattr__(self, "iteration", selected_iteration)
        object.__setattr__(self, "accepted_step_fraction", fraction)
        object.__setattr__(self, "backtracking_trials", trials)
        object.__setattr__(
            self, "bag_approx_log_likelihoods", input_likelihoods
        )
        object.__setattr__(
            self,
            "output_bag_approx_log_likelihoods",
            output_likelihoods,
        )
        object.__setattr__(self, "approx_log_likelihood", input_total)
        object.__setattr__(
            self, "output_approx_log_likelihood", output_total
        )
        object.__setattr__(
            self, "maximum_absolute_log_q_change", target_change
        )
        object.__setattr__(
            self,
            "accepted_maximum_absolute_log_q_change",
            accepted_change,
        )
        object.__setattr__(self, "converged", bool(self.converged))

    @property
    def output_covariance(self) -> BodyWrenchDiagonalCovariance:
        return self.accepted_covariance


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
    final_expectation_input_covariance: BodyWrenchDiagonalCovariance
    final_expectations: Tuple[DiagonalQBagExpectation, ...]
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
        previous_output_likelihoods = None
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
                and value.accepted_step_fraction > 0.0
            )
            if value.converged != expected_flag:
                raise ValueError("iteration convergence flag is inconsistent")
            if value.converged and index != len(iterations):
                raise ValueError("EM must stop at the first converged iteration")
            trial_fractions = tuple(
                trial.step_fraction for trial in value.backtracking_trials
            )
            if trial_fractions != self.config.backtracking_step_fractions[
                : len(trial_fractions)
            ]:
                raise ValueError(
                    "iteration trials must use the configured fraction prefix"
                )
            if (
                value.accepted_step_fraction == 0.0
                and trial_fractions
                != self.config.backtracking_step_fractions
            ):
                raise ValueError(
                    "a rejected generalized-EM update must exhaust the "
                    "configured backtracking fractions"
                )
            if (
                value.accepted_step_fraction == 0.0
                and index != len(iterations)
            ):
                raise ValueError(
                    "EM must stop at the first rejected generalized-EM update"
                )
            if (
                previous_output_likelihoods is not None
                and value.bag_approx_log_likelihoods
                != previous_output_likelihoods
            ):
                raise ValueError(
                    "cached E-step likelihoods are not contiguous"
                )
            expected_input = value.output_covariance
            previous_output_likelihoods = (
                value.output_bag_approx_log_likelihoods
            )
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
        if not isinstance(
            self.final_expectation_input_covariance,
            BodyWrenchDiagonalCovariance,
        ):
            raise TypeError(
                "final_expectation_input_covariance must be "
                "BodyWrenchDiagonalCovariance"
            )
        if not isinstance(self.covariance, BodyWrenchDiagonalCovariance):
            raise TypeError("covariance must be BodyWrenchDiagonalCovariance")
        if not np.array_equal(
            self.covariance.stationary_variance,
            iterations[-1].accepted_covariance.stationary_variance,
        ):
            raise ValueError(
                "result covariance is not the final accepted output"
            )
        if not np.array_equal(
            self.final_expectation_input_covariance.stationary_variance,
            self.covariance.stationary_variance,
        ):
            raise ValueError(
                "final expectation must be conditioned on result covariance"
            )
        fixed_layout = {
            value.bag_id: (value.times.copy(), value.correlation_time)
            for value in expectations
        }
        final_expectations = _ordered_expectations(
            self.final_expectations, pilots, fixed_layout=fixed_layout
        )
        for previous, final in zip(expectations, final_expectations):
            if previous.member_count != final.member_count:
                raise ValueError(
                    "final expectation member count changed for bag {!r}".format(
                        final.bag_id
                    )
                )
        final_likelihoods = tuple(
            (value.bag_id, value.approx_log_likelihood)
            for value in final_expectations
        )
        if (
            final_likelihoods
            != iterations[-1].output_bag_approx_log_likelihoods
        ):
            raise ValueError(
                "final expectations do not match the final accepted trial"
            )
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
        elif self.termination_reason == GENERALIZED_EM_UPDATE_REJECTED_TERMINATION:
            if iterations[-1].accepted_step_fraction != 0.0:
                raise ValueError(
                    "rejected generalized EM must retain its input Q"
                )
        elif (
            self.termination_reason != MAXIMUM_ITERATIONS_TERMINATION
            or len(iterations) != self.config.maximum_iterations
            or iterations[-1].converged
            or iterations[-1].accepted_step_fraction == 0.0
        ):
            raise ValueError(
                "non-converged EM must exhaust maximum_iterations or reject "
                "a generalized-EM update"
            )
        object.__setattr__(self, "pilots", pilots)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "last_expectations", expectations)
        object.__setattr__(
            self, "final_expectations", final_expectations
        )
        object.__setattr__(self, "converged", selected_converged)

    @property
    def bag_ids(self) -> Tuple[str, ...]:
        return tuple(value.bag_id for value in self.pilots)

    @property
    def final_approx_log_likelihood(self) -> float:
        return _finite_sum(
            (
                value.approx_log_likelihood
                for value in self.final_expectations
            ),
            "final approx_log_likelihood",
        )


ExpectationStep = Callable[
    [BodyWrenchDiagonalCovariance, DiagonalQExpectationContext],
    Iterable[DiagonalQBagExpectation],
]
ExpectationDiscardedCallback = Callable[
    [Tuple[DiagonalQBagExpectation, ...]], None
]


def run_diagonal_q_em(
    pilots: Iterable[DiagonalQInitialPilot],
    expectation_step: ExpectationStep,
    config: DiagonalQEmConfig,
    *,
    progress_callback: Optional[ProgressCallback] = None,
    cancellation_token: Optional[CancellationToken] = None,
    expectation_discarded_callback: Optional[
        ExpectationDiscardedCallback
    ] = None,
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
    if (
        expectation_discarded_callback is not None
        and not callable(expectation_discarded_callback)
    ):
        raise TypeError("expectation_discarded_callback must be callable")
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
    current_expectations = None
    evaluation_count = 0

    def evaluate_candidate(selected_covariance, iteration, trial, alpha):
        nonlocal evaluation_count
        evaluation_count += 1
        context = DiagonalQExpectationContext(
            evaluation=evaluation_count,
            iteration=iteration,
            trial=trial,
            step_fraction=alpha,
        )
        callback_covariance = BodyWrenchDiagonalCovariance(
            selected_covariance.stationary_variance
        )
        raw = expectation_step(callback_covariance, context)
        tracker.checkpoint()
        return _ordered_expectations(
            raw, ordered_pilots, fixed_layout
        )

    def finish_result(
        converged, termination_reason, completed_iteration, expectations
    ):
        tracker.checkpoint()
        final_input = BodyWrenchDiagonalCovariance(
            covariance.stationary_variance
        )
        tracker.emit(
            completed_units=completed_iteration,
            stage_id="diagonal_q_final_expectation",
            stage_label="Final diagonal Q expectation",
            iteration=completed_iteration,
            maximum_iterations=config.maximum_iterations,
            message=(
                "reused accepted candidate paths conditioned on output Q"
            ),
        )
        tracker.checkpoint()
        return DiagonalQEmResult(
            config=config,
            pilots=ordered_pilots,
            initial_covariance=initial_covariance,
            iterations=tuple(trace),
            last_expectations=last_expectations,
            final_expectation_input_covariance=final_input,
            final_expectations=expectations,
            covariance=covariance,
            converged=converged,
            termination_reason=termination_reason,
        )

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
        if current_expectations is None:
            try:
                expectations = evaluate_candidate(
                    input_covariance, iteration, 0, 0.0
                )
            except DiagonalQExpectationNumericalError as error:
                raise DiagonalQExpectationNumericalError(
                    "initial diagonal-Q E-step is not finite"
                ) from error
        else:
            expectations = current_expectations
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
        target_log_change = _maximum_absolute_log_q_change(
            input_covariance, update.covariance
        )
        input_likelihoods = tuple(
            (value.bag_id, value.approx_log_likelihood)
            for value in expectations
        )
        input_total_likelihood = _finite_sum(
            (value[1] for value in input_likelihoods),
            "combined approx_log_likelihood",
        )
        trials = []
        accepted_covariance = None
        accepted_fraction = 0.0
        accepted_expectations = None
        output_likelihoods = input_likelihoods
        output_total_likelihood = input_total_likelihood
        for trial_index, alpha in enumerate(
            config.backtracking_step_fractions, start=1
        ):
            candidate_covariance = _log_interpolated_covariance(
                input_covariance, update.covariance, alpha
            )
            tracker.emit(
                completed_units=iteration - 1,
                stage_id="diagonal_q_backtracking_trial",
                stage_label="Diagonal Q generalized-EM backtracking",
                iteration=iteration,
                maximum_iterations=config.maximum_iterations,
                message=(
                    "evaluating log-Q candidate alpha={:.8g} "
                    "(trial {}/{})"
                ).format(
                    alpha,
                    trial_index,
                    len(config.backtracking_step_fractions),
                ),
            )
            tracker.checkpoint()
            if np.array_equal(
                candidate_covariance.stationary_variance,
                input_covariance.stationary_variance,
            ):
                candidate_expectations = expectations
            else:
                try:
                    candidate_expectations = evaluate_candidate(
                        candidate_covariance,
                        iteration,
                        trial_index,
                        alpha,
                    )
                except DiagonalQExpectationNumericalError as error:
                    trials.append(
                        DiagonalQBacktrackingTrial(
                            trial=trial_index,
                            step_fraction=alpha,
                            covariance=candidate_covariance,
                            outcome=BACKTRACKING_NUMERICAL_FAILURE,
                        )
                    )
                    tracker.emit(
                        completed_units=iteration - 1,
                        stage_id="diagonal_q_backtracking_rejected",
                        stage_label="Diagonal Q candidate rejected",
                        iteration=iteration,
                        maximum_iterations=config.maximum_iterations,
                        message=(
                            "rejected alpha={:.8g}: {}"
                        ).format(alpha, error),
                    )
                    tracker.checkpoint()
                    continue
            candidate_likelihoods = tuple(
                (value.bag_id, value.approx_log_likelihood)
                for value in candidate_expectations
            )
            try:
                candidate_total = _finite_sum(
                    (value[1] for value in candidate_likelihoods),
                    "candidate combined approx_log_likelihood",
                )
            except ValueError as error:
                if (
                    candidate_expectations is not expectations
                    and expectation_discarded_callback is not None
                ):
                    expectation_discarded_callback(candidate_expectations)
                trials.append(
                    DiagonalQBacktrackingTrial(
                        trial=trial_index,
                        step_fraction=alpha,
                        covariance=candidate_covariance,
                        outcome=BACKTRACKING_NUMERICAL_FAILURE,
                    )
                )
                tracker.emit(
                    completed_units=iteration - 1,
                    stage_id="diagonal_q_backtracking_rejected",
                    stage_label="Diagonal Q candidate rejected",
                    iteration=iteration,
                    maximum_iterations=config.maximum_iterations,
                    message=(
                        "rejected alpha={:.8g}: {}"
                    ).format(alpha, error),
                )
                tracker.checkpoint()
                continue
            outcome = (
                BACKTRACKING_ACCEPTED
                if candidate_total >= input_total_likelihood
                else BACKTRACKING_LIKELIHOOD_DECREASE
            )
            trials.append(
                DiagonalQBacktrackingTrial(
                    trial=trial_index,
                    step_fraction=alpha,
                    covariance=candidate_covariance,
                    outcome=outcome,
                    bag_approx_log_likelihoods=candidate_likelihoods,
                    approx_log_likelihood=candidate_total,
                )
            )
            if outcome == BACKTRACKING_ACCEPTED:
                accepted_covariance = candidate_covariance
                accepted_fraction = alpha
                accepted_expectations = candidate_expectations
                output_likelihoods = candidate_likelihoods
                output_total_likelihood = candidate_total
                break
            if expectation_discarded_callback is not None:
                expectation_discarded_callback(candidate_expectations)
            tracker.emit(
                completed_units=iteration - 1,
                stage_id="diagonal_q_backtracking_rejected",
                stage_label="Diagonal Q candidate rejected",
                iteration=iteration,
                maximum_iterations=config.maximum_iterations,
                message=(
                    "rejected alpha={:.8g}: approximate log likelihood "
                    "decreased from {:.8g} to {:.8g}"
                ).format(
                    alpha, input_total_likelihood, candidate_total
                ),
            )
            tracker.checkpoint()

        update_rejected = accepted_covariance is None
        if update_rejected:
            accepted_covariance = BodyWrenchDiagonalCovariance(
                input_covariance.stationary_variance
            )
            accepted_expectations = expectations
        accepted_log_change = _maximum_absolute_log_q_change(
            input_covariance, accepted_covariance
        )
        converged = (
            not update_rejected
            and target_log_change <= config.log_q_tolerance
        )
        record = DiagonalQEmIteration(
            iteration=iteration,
            input_covariance=input_covariance,
            update=update,
            accepted_covariance=accepted_covariance,
            accepted_step_fraction=accepted_fraction,
            backtracking_trials=tuple(trials),
            bag_approx_log_likelihoods=input_likelihoods,
            approx_log_likelihood=input_total_likelihood,
            output_bag_approx_log_likelihoods=output_likelihoods,
            output_approx_log_likelihood=output_total_likelihood,
            maximum_absolute_log_q_change=target_log_change,
            accepted_maximum_absolute_log_q_change=accepted_log_change,
            converged=converged,
        )
        trace.append(record)
        last_expectations = expectations
        covariance = accepted_covariance
        current_expectations = accepted_expectations
        if (
            not update_rejected
            and accepted_expectations is not expectations
            and expectation_discarded_callback is not None
        ):
            expectation_discarded_callback(expectations)
        tracker.emit(
            completed_units=iteration,
            stage_id="diagonal_q_m_step",
            stage_label="Diagonal Q M-step",
            iteration=iteration,
            maximum_iterations=config.maximum_iterations,
            message=(
                "completed diagonal-Q iteration {}; "
                "target max |delta log Q|={:.6g}; accepted alpha={:.8g}; "
                "accepted max |delta log Q|={:.6g}"
            ).format(
                iteration,
                target_log_change,
                accepted_fraction,
                accepted_log_change,
            ),
        )
        tracker.checkpoint()
        if update_rejected:
            return finish_result(
                False,
                GENERALIZED_EM_UPDATE_REJECTED_TERMINATION,
                iteration,
                current_expectations,
            )
        if converged:
            return finish_result(
                True,
                LOG_Q_TOLERANCE_TERMINATION,
                iteration,
                current_expectations,
            )

    return finish_result(
        False,
        MAXIMUM_ITERATIONS_TERMINATION,
        config.maximum_iterations,
        current_expectations,
    )


__all__ = [
    "BACKTRACKING_ACCEPTED",
    "BACKTRACKING_LIKELIHOOD_DECREASE",
    "BACKTRACKING_NUMERICAL_FAILURE",
    "DEFAULT_BACKTRACKING_STEP_FRACTIONS",
    "DiagonalQBacktrackingTrial",
    "DiagonalQBagExpectation",
    "DiagonalQEmConfig",
    "DiagonalQEmIteration",
    "DiagonalQEmResult",
    "DiagonalQExpectationContext",
    "DiagonalQExpectationNumericalError",
    "DiagonalQInitialPilot",
    "ExpectationStep",
    "ExpectationDiscardedCallback",
    "GENERALIZED_EM_UPDATE_REJECTED_TERMINATION",
    "LOG_Q_TOLERANCE_TERMINATION",
    "MAXIMUM_ITERATIONS_TERMINATION",
    "initial_diagonal_q_from_pilots",
    "run_diagonal_q_em",
]
