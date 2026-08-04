"""Outer alternation of sparse E-steps, diagonal-Q updates, and lag refinement."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np

from grape_param_estim.batch.laplace_em import (
    DiagonalQDefinition,
    DiagonalQTarget,
    ExpectedResidualMoments,
    QInnerEvaluation,
    QUpdateResult,
    compute_diagonal_q_target,
    damped_diagonal_q_update,
)
from grape_param_estim.batch.state import BatchState


class EStepPhase(Enum):
    WIDE_LAG_PROFILE = "wide_lag_profile"
    FIXED_LAG = "fixed_lag"
    LOCAL_LAG_PROFILE = "local_lag_profile"


class QUpdatePolicy(Enum):
    FIXED = "fixed"
    LAPLACE_EM = "laplace_em"


class LaplaceEmTerminationReason(Enum):
    FIXED_BY_REQUEST = "fixed_by_request"
    CONVERGENCE_TOLERANCES = "convergence_tolerances"
    MAXIMUM_ITERATIONS = "maximum_iterations"
    REPEATED_Q_REJECTION = "repeated_q_rejection"
    REPEATED_LAG_PROFILE_FAILURE = "repeated_lag_profile_failure"


@dataclass(frozen=True)
class LaplaceEStepResult:
    """Successful trajectory MAP, covariance moments, and marginal evidence."""

    q: np.ndarray
    lag: float
    state: BatchState
    moments: ExpectedResidualMoments
    map_objective: float
    approximate_marginal_objective: float
    inner_iterations: int
    termination_reason: str

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)) or np.any(q <= 0.0):
            raise ValueError("q must contain six positive finite values")
        q = q.copy()
        q.setflags(write=False)
        lag = float(self.lag)
        objectives = np.asarray(
            (self.map_objective, self.approximate_marginal_objective),
            dtype=float,
        )
        if not np.isfinite(lag) or lag < 0.0:
            raise ValueError("lag must be finite and non-negative")
        if not np.all(np.isfinite(objectives)):
            raise ValueError("E-step objectives must be finite")
        if not isinstance(self.state, BatchState):
            raise TypeError("state must be a BatchState")
        if not isinstance(self.moments, ExpectedResidualMoments):
            raise TypeError("moments must be ExpectedResidualMoments")
        if (
            isinstance(self.inner_iterations, (bool, np.bool_))
            or not isinstance(self.inner_iterations, (int, np.integer))
            or self.inner_iterations < 0
        ):
            raise ValueError("inner_iterations must be a non-negative integer")
        if (
            not isinstance(self.termination_reason, str)
            or not self.termination_reason
            or self.termination_reason.strip() != self.termination_reason
        ):
            raise ValueError("termination_reason must be a canonical string")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "lag", lag)
        object.__setattr__(self, "map_objective", float(self.map_objective))
        object.__setattr__(
            self,
            "approximate_marginal_objective",
            float(self.approximate_marginal_objective),
        )
        object.__setattr__(self, "inner_iterations", int(self.inner_iterations))


class LaplaceEStepFailure(RuntimeError):
    """A fixed-Q trajectory solve or lag profile did not converge."""

    def __init__(
        self,
        reason: str,
        inner_iterations: int = 0,
        *,
        detail: str = "",
    ):
        if not isinstance(reason, str) or not reason:
            raise ValueError("failure reason must be a non-empty string")
        if (
            isinstance(inner_iterations, (bool, np.bool_))
            or not isinstance(inner_iterations, (int, np.integer))
            or inner_iterations < 0
        ):
            raise ValueError("inner_iterations must be a non-negative integer")
        if not isinstance(detail, str) or detail.strip() != detail:
            raise ValueError("failure detail must be a canonical string")
        self.reason = reason
        self.inner_iterations = int(inner_iterations)
        self.detail = detail
        super().__init__(
            reason if not detail else "{}: {}".format(reason, detail)
        )


LaplaceEStepSolver = Callable[
    [np.ndarray, EStepPhase, float, Optional[BatchState]],
    LaplaceEStepResult,
]


@dataclass(frozen=True)
class LaplaceEmSettings:
    maximum_iterations: int = 12
    minimum_iterations: int = 2
    log_q_tolerance: float = 1.0e-3
    lag_tolerance: float = 1.0e-5
    map_objective_tolerance: float = 1.0e-5
    marginal_objective_tolerance: float = 1.0e-5
    q_minimum_alpha: float = 1.0 / 64.0
    q_acceptance_objective_tolerance: float = 0.0
    maximum_repeated_q_rejections: int = 3
    maximum_repeated_lag_profile_failures: int = 3

    def __post_init__(self) -> None:
        for name in (
            "maximum_iterations",
            "minimum_iterations",
            "maximum_repeated_q_rejections",
            "maximum_repeated_lag_profile_failures",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError("{} must be a positive integer".format(name))
            object.__setattr__(self, name, int(value))
        if self.minimum_iterations > self.maximum_iterations:
            raise ValueError("minimum_iterations cannot exceed maximum_iterations")
        for name in (
            "log_q_tolerance",
            "lag_tolerance",
            "map_objective_tolerance",
            "marginal_objective_tolerance",
            "q_acceptance_objective_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("{} must be finite and non-negative".format(name))
            object.__setattr__(self, name, value)
        alpha = float(self.q_minimum_alpha)
        if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
            raise ValueError("q_minimum_alpha must be in (0, 1]")
        object.__setattr__(self, "q_minimum_alpha", alpha)


@dataclass(frozen=True)
class LaplaceEmIteration:
    iteration: int
    input_step: LaplaceEStepResult
    q_target: DiagonalQTarget
    q_update: QUpdateResult
    output_step: LaplaceEStepResult
    lag_refinement_failed: bool
    lag_refinement_failure_reason: str
    lag_change: float
    map_objective_change: float
    marginal_objective_change: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.iteration, (bool, np.bool_))
            or not isinstance(self.iteration, (int, np.integer))
            or self.iteration < 0
        ):
            raise ValueError("iteration must be a non-negative integer")
        for name, expected in (
            ("input_step", LaplaceEStepResult),
            ("q_target", DiagonalQTarget),
            ("q_update", QUpdateResult),
            ("output_step", LaplaceEStepResult),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError("{} has an invalid type".format(name))
        if not np.array_equal(
            self.input_step.q, self.q_update.input_evaluation.q
        ):
            raise ValueError("Q update input disagrees with the input E-step")
        if not np.array_equal(self.output_step.q, self.q_update.accepted_q):
            raise ValueError("output E-step disagrees with the retained Q")
        if not isinstance(self.lag_refinement_failed, (bool, np.bool_)):
            raise TypeError("lag_refinement_failed must be boolean")
        failed = bool(self.lag_refinement_failed)
        if failed and not self.lag_refinement_failure_reason:
            raise ValueError("a failed lag refinement needs a reason")
        if not failed and self.lag_refinement_failure_reason != "":
            raise ValueError("a successful lag refinement cannot have a reason")
        for name in (
            "lag_change",
            "map_objective_change",
            "marginal_objective_change",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("{} must be finite and non-negative".format(name))
            object.__setattr__(self, name, value)
        object.__setattr__(self, "iteration", int(self.iteration))
        object.__setattr__(self, "lag_refinement_failed", failed)


@dataclass(frozen=True)
class LaplaceEmResult:
    definition: DiagonalQDefinition
    iterations: Tuple[LaplaceEmIteration, ...]
    final_step: LaplaceEStepResult
    reason: LaplaceEmTerminationReason
    update_policy: QUpdatePolicy = QUpdatePolicy.LAPLACE_EM

    def __post_init__(self) -> None:
        if not isinstance(self.definition, DiagonalQDefinition):
            raise TypeError("definition must be DiagonalQDefinition")
        if (
            type(self.iterations) is not tuple
            or not self.iterations
            or any(
                not isinstance(value, LaplaceEmIteration)
                for value in self.iterations
            )
        ):
            raise TypeError("iterations must be a non-empty tuple")
        if tuple(value.iteration for value in self.iterations) != tuple(
            range(len(self.iterations))
        ):
            raise ValueError("EM iteration indices must be contiguous")
        if not isinstance(self.final_step, LaplaceEStepResult):
            raise TypeError("final_step must be LaplaceEStepResult")
        if self.final_step is not self.iterations[-1].output_step:
            raise ValueError("final_step must be the final iteration output")
        if not isinstance(self.reason, LaplaceEmTerminationReason):
            raise TypeError("reason must be LaplaceEmTerminationReason")
        if not isinstance(self.update_policy, QUpdatePolicy):
            raise TypeError("update_policy must be QUpdatePolicy")
        if self.update_policy is QUpdatePolicy.FIXED:
            if self.reason is not LaplaceEmTerminationReason.FIXED_BY_REQUEST:
                raise ValueError("fixed Q requires fixed_by_request termination")
            if len(self.iterations) != 1:
                raise ValueError("fixed Q requires one diagnostic solve record")
            record = self.iterations[0]
            if (
                record.input_step is not record.output_step
                or record.q_update.termination_reason != "fixed_by_request"
                or record.q_update.attempts
            ):
                raise ValueError("fixed Q record cannot contain an update")
        elif self.reason is LaplaceEmTerminationReason.FIXED_BY_REQUEST:
            raise ValueError("Laplace-EM cannot terminate as fixed Q")

    @property
    def converged(self) -> bool:
        return self.reason is LaplaceEmTerminationReason.CONVERGENCE_TOLERANCES

    @property
    def em_iteration_count(self) -> int:
        return (
            0
            if self.update_policy is QUpdatePolicy.FIXED
            else len(self.iterations)
        )


class LaplaceEmCancelled(RuntimeError):
    def __init__(
        self,
        iterations: Tuple[LaplaceEmIteration, ...],
        current_step: LaplaceEStepResult,
    ):
        self.iterations = iterations
        self.current_step = current_step
        super().__init__(
            "Laplace-EM cancelled after {} iterations".format(len(iterations))
        )


def _solve_checked(
    solver: LaplaceEStepSolver,
    q: np.ndarray,
    phase: EStepPhase,
    lag: float,
    warm_start: Optional[BatchState],
) -> LaplaceEStepResult:
    result = solver(q.copy(), phase, float(lag), warm_start)
    if not isinstance(result, LaplaceEStepResult):
        raise TypeError("E-step solver must return LaplaceEStepResult")
    if not np.array_equal(result.q, q):
        raise ValueError("E-step solver returned a result for a different Q")
    if phase is EStepPhase.FIXED_LAG and result.lag != float(lag):
        raise ValueError("fixed-lag E-step changed the requested lag")
    return result


def run_laplace_em(
    definition: DiagonalQDefinition,
    initial_q: np.ndarray,
    q_floor: np.ndarray,
    interval_time_steps: np.ndarray,
    initial_lag: float,
    solver: LaplaceEStepSolver,
    settings: LaplaceEmSettings = LaplaceEmSettings(),
    *,
    initial_warm_start: Optional[BatchState] = None,
    cancellation_requested: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[LaplaceEmIteration], None]] = None,
) -> LaplaceEmResult:
    """Alternate corrected Q updates with local continuous-lag profiles."""

    if not isinstance(definition, DiagonalQDefinition):
        raise TypeError("definition must be DiagonalQDefinition")
    q = np.asarray(initial_q, dtype=float)
    floor = np.asarray(q_floor, dtype=float)
    if (
        q.shape != (6,)
        or floor.shape != (6,)
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(floor))
        or np.any(q <= 0.0)
        or np.any(floor <= 0.0)
    ):
        raise ValueError("initial_q and q_floor must be positive finite six-vectors")
    time_steps = np.asarray(interval_time_steps, dtype=float)
    definition.interval_weights(time_steps)
    lag = float(initial_lag)
    if not np.isfinite(lag) or lag < 0.0:
        raise ValueError("initial_lag must be finite and non-negative")
    if not callable(solver):
        raise TypeError("solver must be callable")
    if not isinstance(settings, LaplaceEmSettings):
        raise TypeError("settings must be LaplaceEmSettings")
    if initial_warm_start is not None and not isinstance(
        initial_warm_start, BatchState
    ):
        raise TypeError("initial_warm_start must be BatchState or None")
    if cancellation_requested is not None and not callable(cancellation_requested):
        raise TypeError("cancellation_requested must be callable")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")

    try:
        current = _solve_checked(
            solver,
            q,
            EStepPhase.WIDE_LAG_PROFILE,
            lag,
            initial_warm_start,
        )
    except LaplaceEStepFailure as error:
        raise LaplaceEStepFailure(
            "initial_wide_profile:{}".format(error.reason),
            error.inner_iterations,
            detail=error.detail,
        ) from error
    if current.moments.interval_count != time_steps.size:
        raise ValueError("E-step residual count disagrees with interval_time_steps")

    records = []
    repeated_q_rejections = 0
    repeated_lag_failures = 0
    for iteration in range(settings.maximum_iterations):
        if cancellation_requested is not None and cancellation_requested():
            raise LaplaceEmCancelled(tuple(records), current)
        target = compute_diagonal_q_target(
            definition,
            current.moments,
            time_steps,
            floor,
        )
        input_evaluation = QInnerEvaluation(
            q=current.q,
            successful=True,
            map_objective=current.map_objective,
            approximate_marginal_objective=(
                current.approximate_marginal_objective
            ),
            lag=current.lag,
            failure_reason="",
            warm_start=current.state,
        )
        candidate_steps = {}

        def evaluate_candidate(candidate_q, warm_start):
            key = np.asarray(candidate_q, dtype="<f8").tobytes(order="C")
            try:
                candidate_step = _solve_checked(
                    solver,
                    candidate_q,
                    EStepPhase.FIXED_LAG,
                    current.lag,
                    warm_start,
                )
            except LaplaceEStepFailure as error:
                return QInnerEvaluation(
                    q=candidate_q,
                    successful=False,
                    map_objective=float("inf"),
                    approximate_marginal_objective=float("inf"),
                    lag=current.lag,
                    failure_reason=error.reason,
                    warm_start=None,
                )
            candidate_steps[key] = candidate_step
            return QInnerEvaluation(
                q=candidate_q,
                successful=True,
                map_objective=candidate_step.map_objective,
                approximate_marginal_objective=(
                    candidate_step.approximate_marginal_objective
                ),
                lag=candidate_step.lag,
                failure_reason="",
                warm_start=candidate_step.state,
            )

        q_update = damped_diagonal_q_update(
            input_evaluation,
            target,
            evaluate_candidate,
            minimum_alpha=settings.q_minimum_alpha,
            marginal_objective_tolerance=(
                settings.q_acceptance_objective_tolerance
            ),
        )
        lag_failed = False
        lag_failure_reason = ""
        if q_update.accepted:
            repeated_q_rejections = 0
            accepted_key = np.asarray(
                q_update.accepted_q, dtype="<f8"
            ).tobytes(order="C")
            fixed_step = candidate_steps[accepted_key]
            try:
                output = _solve_checked(
                    solver,
                    q_update.accepted_q,
                    EStepPhase.LOCAL_LAG_PROFILE,
                    fixed_step.lag,
                    fixed_step.state,
                )
                repeated_lag_failures = 0
            except LaplaceEStepFailure as error:
                output = fixed_step
                lag_failed = True
                lag_failure_reason = error.reason
                repeated_lag_failures += 1
        else:
            output = current
            repeated_q_rejections += 1
        if output.moments.interval_count != time_steps.size:
            raise ValueError("E-step residual count changed across EM iterations")
        record = LaplaceEmIteration(
            iteration=iteration,
            input_step=current,
            q_target=target,
            q_update=q_update,
            output_step=output,
            lag_refinement_failed=lag_failed,
            lag_refinement_failure_reason=lag_failure_reason,
            lag_change=abs(output.lag - current.lag),
            map_objective_change=abs(
                output.map_objective - current.map_objective
            ),
            marginal_objective_change=abs(
                output.approximate_marginal_objective
                - current.approximate_marginal_objective
            ),
        )
        records.append(record)
        if progress is not None:
            progress(record)
        current = output

        if repeated_q_rejections >= settings.maximum_repeated_q_rejections:
            reason = LaplaceEmTerminationReason.REPEATED_Q_REJECTION
            break
        if (
            repeated_lag_failures
            >= settings.maximum_repeated_lag_profile_failures
        ):
            reason = (
                LaplaceEmTerminationReason.REPEATED_LAG_PROFILE_FAILURE
            )
            break
        if (
            iteration + 1 >= settings.minimum_iterations
            and q_update.accepted
            and q_update.max_log_q_change <= settings.log_q_tolerance
            and record.lag_change <= settings.lag_tolerance
            and record.map_objective_change
            <= settings.map_objective_tolerance
            and record.marginal_objective_change
            <= settings.marginal_objective_tolerance
        ):
            reason = LaplaceEmTerminationReason.CONVERGENCE_TOLERANCES
            break
    else:
        reason = LaplaceEmTerminationReason.MAXIMUM_ITERATIONS
    return LaplaceEmResult(
        definition=definition,
        iterations=tuple(records),
        final_step=current,
        reason=reason,
    )


def run_fixed_q(
    definition: DiagonalQDefinition,
    fixed_q: np.ndarray,
    q_floor: np.ndarray,
    interval_time_steps: np.ndarray,
    initial_lag: float,
    solver: LaplaceEStepSolver,
    *,
    initial_warm_start: Optional[BatchState] = None,
    cancellation_requested: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[LaplaceEmIteration], None]] = None,
) -> LaplaceEmResult:
    """Solve one delay-profiled MAP/Laplace problem without updating Q."""

    if not isinstance(definition, DiagonalQDefinition):
        raise TypeError("definition must be DiagonalQDefinition")
    q = np.asarray(fixed_q, dtype=float)
    floor = np.asarray(q_floor, dtype=float)
    if (
        q.shape != (6,)
        or floor.shape != (6,)
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(floor))
        or np.any(q <= 0.0)
        or np.any(floor <= 0.0)
        or np.any(q < floor)
    ):
        raise ValueError("fixed_q must be finite, positive, and above q_floor")
    time_steps = np.asarray(interval_time_steps, dtype=float)
    definition.interval_weights(time_steps)
    lag = float(initial_lag)
    if not np.isfinite(lag) or lag < 0.0:
        raise ValueError("initial_lag must be finite and non-negative")
    if not callable(solver):
        raise TypeError("solver must be callable")
    if initial_warm_start is not None and not isinstance(
        initial_warm_start, BatchState
    ):
        raise TypeError("initial_warm_start must be BatchState or None")
    if cancellation_requested is not None and not callable(cancellation_requested):
        raise TypeError("cancellation_requested must be callable")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    try:
        step = _solve_checked(
            solver,
            q,
            EStepPhase.WIDE_LAG_PROFILE,
            lag,
            initial_warm_start,
        )
    except LaplaceEStepFailure as error:
        raise LaplaceEStepFailure(
            "fixed_q_wide_profile:{}".format(error.reason),
            error.inner_iterations,
            detail=error.detail,
        ) from error
    if step.moments.interval_count != time_steps.size:
        raise ValueError("E-step residual count disagrees with interval_time_steps")
    target = compute_diagonal_q_target(
        definition,
        step.moments,
        time_steps,
        floor,
    )
    input_evaluation = QInnerEvaluation(
        q=step.q,
        successful=True,
        map_objective=step.map_objective,
        approximate_marginal_objective=step.approximate_marginal_objective,
        lag=step.lag,
        failure_reason="",
        warm_start=step.state,
    )
    update = QUpdateResult(
        input_evaluation=input_evaluation,
        target=target,
        attempts=(),
        accepted=False,
        accepted_q=step.q,
        accepted_alpha=0.0,
        max_log_q_change=0.0,
        termination_reason="fixed_by_request",
    )
    record = LaplaceEmIteration(
        iteration=0,
        input_step=step,
        q_target=target,
        q_update=update,
        output_step=step,
        lag_refinement_failed=False,
        lag_refinement_failure_reason="",
        lag_change=0.0,
        map_objective_change=0.0,
        marginal_objective_change=0.0,
    )
    if progress is not None:
        progress(record)
    return LaplaceEmResult(
        definition=definition,
        iterations=(record,),
        final_step=step,
        reason=LaplaceEmTerminationReason.FIXED_BY_REQUEST,
        update_policy=QUpdatePolicy.FIXED,
    )


__all__ = [
    "EStepPhase",
    "LaplaceEStepFailure",
    "LaplaceEStepResult",
    "LaplaceEStepSolver",
    "LaplaceEmCancelled",
    "LaplaceEmIteration",
    "LaplaceEmResult",
    "LaplaceEmSettings",
    "LaplaceEmTerminationReason",
    "QUpdatePolicy",
    "run_fixed_q",
    "run_laplace_em",
]
