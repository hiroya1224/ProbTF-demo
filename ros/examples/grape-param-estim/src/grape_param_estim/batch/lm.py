"""Trust-ratio Levenberg--Marquardt iterations for sparse batch MAP."""

from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Optional, Tuple

import numpy as np

from grape_param_estim.batch.factor import FactorEvaluation
from grape_param_estim.batch.problem import (
    BatchProblem,
    ProblemLinearization,
    RecoverableModelEvaluationError,
)
from grape_param_estim.batch.sparse_solver import (
    BagFactorizationDiagnostics,
    solve_scaled_lm_step,
)
from grape_param_estim.batch.state import BatchState


class LMTerminationReason(Enum):
    """Machine-readable terminal condition of one inner MAP solve."""

    GRADIENT_TOLERANCE = "gradient_tolerance"
    SCALED_STEP_TOLERANCE = "scaled_step_tolerance"
    RELATIVE_OBJECTIVE_TOLERANCE = "relative_objective_tolerance"
    MAXIMUM_ITERATIONS = "maximum_iterations"
    NUMERICAL_FACTORIZATION_FAILURE = "numerical_factorization_failure"
    NONFINITE_MODEL_EVALUATION = "nonfinite_model_evaluation"
    ACTIVE_SET_OSCILLATION = "active_set_oscillation"


_CONVERGED_REASONS = {
    LMTerminationReason.GRADIENT_TOLERANCE,
    LMTerminationReason.SCALED_STEP_TOLERANCE,
    LMTerminationReason.RELATIVE_OBJECTIVE_TOLERANCE,
}


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("{} must be a finite non-negative scalar".format(name))
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("{} must be finite and non-negative".format(name))
    return result


@dataclass(frozen=True)
class LMSettings:
    """Explicit convergence and damping policy for the inner MAP solve."""

    maximum_iterations: int = 50
    initial_damping: float = 1.0e-3
    minimum_damping: float = 1.0e-12
    maximum_damping: float = 1.0e12
    acceptance_ratio: float = 1.0e-4
    gradient_tolerance: float = 1.0e-6
    scaled_step_tolerance: float = 1.0e-7
    relative_objective_tolerance: float = 1.0e-8
    maximum_factorization_retries: int = 4
    maximum_model_evaluation_retries: int = 4

    def __post_init__(self) -> None:
        for name in (
            "maximum_iterations",
            "maximum_factorization_retries",
            "maximum_model_evaluation_retries",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value <= 0
            ):
                raise ValueError("{} must be a positive integer".format(name))
            object.__setattr__(self, name, int(value))
        for name in (
            "initial_damping",
            "minimum_damping",
            "maximum_damping",
            "acceptance_ratio",
            "gradient_tolerance",
            "scaled_step_tolerance",
            "relative_objective_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        if not (
            self.minimum_damping
            <= self.initial_damping
            <= self.maximum_damping
        ):
            raise ValueError(
                "damping bounds must contain the initial damping"
            )
        if self.acceptance_ratio >= 1.0:
            raise ValueError("acceptance_ratio must be smaller than one")


@dataclass(frozen=True)
class LMIterationRecord:
    """One attempted step or failed factorization in deterministic order."""

    iteration: int
    objective_before: float
    trial_objective: Optional[float]
    damping_before: float
    damping_after: float
    gradient_inf_norm: float
    scaled_step_norm: Optional[float]
    predicted_reduction: Optional[float]
    actual_reduction: Optional[float]
    reduction_ratio: Optional[float]
    accepted: bool
    active_set_changed: bool
    model_evaluation_failed: bool
    factorization_failed: bool
    bag_diagnostics: Tuple[BagFactorizationDiagnostics, ...]


@dataclass(frozen=True)
class LMSolveResult:
    """Final state and complete convergence evidence for one inner solve."""

    state: BatchState
    objective: float
    reason: LMTerminationReason
    iterations: Tuple[LMIterationRecord, ...]
    final_gradient_inf_norm: float
    final_damping: float

    @property
    def converged(self) -> bool:
        return self.reason in _CONVERGED_REASONS


def _active_set_signature(factors: Tuple[FactorEvaluation, ...]) -> tuple:
    signature = []
    for factor_index, factor in enumerate(factors):
        for name in sorted(factor.active_set):
            mask = factor.active_set[name]
            signature.append(
                (factor_index, name, mask.shape, mask.tobytes())
            )
    return tuple(signature)


def _active_set_is_oscillating(history: list) -> bool:
    return (
        len(history) >= 4
        and history[-1] == history[-3]
        and history[-2] == history[-4]
        and history[-1] != history[-2]
    )


def _increase_damping(value: float, settings: LMSettings) -> float:
    positive_floor = max(
        settings.minimum_damping,
        np.finfo(float).eps,
    )
    return min(settings.maximum_damping, max(positive_floor, 10.0 * value))


def _accepted_damping(
    value: float,
    reduction_ratio: float,
    settings: LMSettings,
) -> float:
    if reduction_ratio > 0.75:
        return max(settings.minimum_damping, value / 3.0)
    if reduction_ratio < 0.25:
        return min(settings.maximum_damping, max(value * 2.0, settings.minimum_damping))
    return value


def solve_batch_map(
    problem: BatchProblem,
    initial_state: BatchState,
    settings: LMSettings = LMSettings(),
) -> LMSolveResult:
    """Run sparse LM while preserving every attempted-step diagnostic."""

    if not isinstance(problem, BatchProblem):
        raise TypeError("problem must be a BatchProblem")
    if not isinstance(initial_state, BatchState):
        raise TypeError("initial_state must be a BatchState")
    if not isinstance(settings, LMSettings):
        raise TypeError("settings must be LMSettings")
    current_state = initial_state
    current = problem.linearize(current_state)
    objective = float(current.sparse.objective)
    damping = settings.initial_damping
    scale = problem.coordinate_scale
    records = []
    active_history = [_active_set_signature(current.factors)]
    consecutive_factorization_failures = 0
    consecutive_model_evaluation_failures = 0

    for iteration in range(settings.maximum_iterations):
        scaled_gradient = scale * current.sparse.gradient
        gradient_inf_norm = float(np.linalg.norm(scaled_gradient, ord=np.inf))
        if gradient_inf_norm <= settings.gradient_tolerance:
            return _result(
                current_state,
                objective,
                LMTerminationReason.GRADIENT_TOLERANCE,
                records,
                gradient_inf_norm,
                damping,
            )
        damping_before = damping
        try:
            step = solve_scaled_lm_step(
                current.sparse,
                scale,
                damping,
            )
        except np.linalg.LinAlgError:
            consecutive_factorization_failures += 1
            next_damping = _increase_damping(damping, settings)
            records.append(
                LMIterationRecord(
                    iteration=iteration,
                    objective_before=objective,
                    trial_objective=None,
                    damping_before=damping_before,
                    damping_after=next_damping,
                    gradient_inf_norm=gradient_inf_norm,
                    scaled_step_norm=None,
                    predicted_reduction=None,
                    actual_reduction=None,
                    reduction_ratio=None,
                    accepted=False,
                    active_set_changed=False,
                    model_evaluation_failed=False,
                    factorization_failed=True,
                    bag_diagnostics=(),
                )
            )
            if (
                consecutive_factorization_failures
                >= settings.maximum_factorization_retries
                or next_damping <= damping
            ):
                return _result(
                    current_state,
                    objective,
                    LMTerminationReason.NUMERICAL_FACTORIZATION_FAILURE,
                    records,
                    gradient_inf_norm,
                    next_damping,
                )
            damping = next_damping
            continue
        consecutive_factorization_failures = 0
        if step.scaled_step_norm <= settings.scaled_step_tolerance:
            records.append(
                LMIterationRecord(
                    iteration=iteration,
                    objective_before=objective,
                    trial_objective=None,
                    damping_before=damping_before,
                    damping_after=damping,
                    gradient_inf_norm=gradient_inf_norm,
                    scaled_step_norm=step.scaled_step_norm,
                    predicted_reduction=step.predicted_reduction,
                    actual_reduction=None,
                    reduction_ratio=None,
                    accepted=False,
                    active_set_changed=False,
                    model_evaluation_failed=False,
                    factorization_failed=False,
                    bag_diagnostics=step.bag_diagnostics,
                )
            )
            return _result(
                current_state,
                objective,
                LMTerminationReason.SCALED_STEP_TOLERANCE,
                records,
                gradient_inf_norm,
                damping,
            )

        trial_state = current_state.retract(step.delta)
        trial = None
        model_evaluation_failed = False
        try:
            trial = problem.linearize(trial_state)
            trial_objective = float(trial.sparse.objective)
        except RecoverableModelEvaluationError:
            model_evaluation_failed = True
            trial_objective = None
        if model_evaluation_failed:
            consecutive_model_evaluation_failures += 1
        else:
            consecutive_model_evaluation_failures = 0
        if trial_objective is None or not np.isfinite(trial_objective):
            actual_reduction = None
            reduction_ratio = None
            accepted = False
            next_damping = _increase_damping(damping, settings)
            active_set_changed = False
        else:
            actual_reduction = objective - trial_objective
            if step.predicted_reduction > 0.0:
                reduction_ratio = actual_reduction / step.predicted_reduction
            else:
                reduction_ratio = -np.inf
            accepted = bool(
                actual_reduction > 0.0
                and np.isfinite(reduction_ratio)
                and reduction_ratio >= settings.acceptance_ratio
            )
            if accepted:
                next_damping = _accepted_damping(
                    damping,
                    reduction_ratio,
                    settings,
                )
                trial_signature = _active_set_signature(trial.factors)
                active_set_changed = trial_signature != active_history[-1]
            else:
                next_damping = _increase_damping(damping, settings)
                active_set_changed = False
        records.append(
            LMIterationRecord(
                iteration=iteration,
                objective_before=objective,
                trial_objective=trial_objective,
                damping_before=damping_before,
                damping_after=next_damping,
                gradient_inf_norm=gradient_inf_norm,
                scaled_step_norm=step.scaled_step_norm,
                predicted_reduction=step.predicted_reduction,
                actual_reduction=actual_reduction,
                reduction_ratio=reduction_ratio,
                accepted=accepted,
                active_set_changed=active_set_changed,
                model_evaluation_failed=model_evaluation_failed,
                factorization_failed=False,
                bag_diagnostics=step.bag_diagnostics,
            )
        )
        damping = next_damping
        if model_evaluation_failed and (
            consecutive_model_evaluation_failures
            >= settings.maximum_model_evaluation_retries
            or next_damping <= damping_before
        ):
            return _result(
                current_state,
                objective,
                LMTerminationReason.NONFINITE_MODEL_EVALUATION,
                records,
                gradient_inf_norm,
                damping,
            )
        if not accepted:
            continue

        previous_objective = objective
        current_state = trial_state
        current = trial
        objective = trial_objective
        active_history.append(_active_set_signature(current.factors))
        if _active_set_is_oscillating(active_history):
            final_gradient = float(
                np.linalg.norm(scale * current.sparse.gradient, ord=np.inf)
            )
            return _result(
                current_state,
                objective,
                LMTerminationReason.ACTIVE_SET_OSCILLATION,
                records,
                final_gradient,
                damping,
            )
        relative_decrease = (previous_objective - objective) / max(
            1.0,
            abs(previous_objective),
        )
        if relative_decrease <= settings.relative_objective_tolerance:
            final_gradient = float(
                np.linalg.norm(scale * current.sparse.gradient, ord=np.inf)
            )
            return _result(
                current_state,
                objective,
                LMTerminationReason.RELATIVE_OBJECTIVE_TOLERANCE,
                records,
                final_gradient,
                damping,
            )

    final_gradient = float(
        np.linalg.norm(scale * current.sparse.gradient, ord=np.inf)
    )
    return _result(
        current_state,
        objective,
        LMTerminationReason.MAXIMUM_ITERATIONS,
        records,
        final_gradient,
        damping,
    )


def _result(
    state: BatchState,
    objective: float,
    reason: LMTerminationReason,
    records: list,
    gradient_inf_norm: float,
    damping: float,
) -> LMSolveResult:
    return LMSolveResult(
        state=state,
        objective=float(objective),
        reason=reason,
        iterations=tuple(records),
        final_gradient_inf_norm=float(gradient_inf_norm),
        final_damping=float(damping),
    )


__all__ = [
    "LMIterationRecord",
    "LMSettings",
    "LMSolveResult",
    "LMTerminationReason",
    "solve_batch_map",
]
