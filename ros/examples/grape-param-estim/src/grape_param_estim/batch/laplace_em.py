"""Diagonal model-discrepancy covariance updates with Laplace correction."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Tuple

import numpy as np


Q_DIMENSION = 6
BODY_WRENCH_QUANTITY = "body_wrench"
BODY_WRENCH_COMPONENT_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
BODY_WRENCH_COMPONENT_UNITS = ("N", "N", "N", "N*m", "N*m", "N*m")


class QIntervalModel(Enum):
    """The sole production mapping from wrench spectral density to intervals."""

    CONTINUOUS_SPECTRAL_DENSITY = "continuous_spectral_density"


@dataclass(frozen=True)
class DiagonalQDefinition:
    """Strict body-wrench continuous spectral-density Q definition."""

    residual_quantity: str
    component_names: Tuple[str, ...]
    component_units: Tuple[str, ...]
    interval_model: QIntervalModel

    def __post_init__(self) -> None:
        if self.residual_quantity != BODY_WRENCH_QUANTITY:
            raise ValueError("residual_quantity must be 'body_wrench'")
        if self.component_names != BODY_WRENCH_COMPONENT_NAMES:
            raise ValueError(
                "component_names must use the canonical body-wrench axes"
            )
        if self.component_units != BODY_WRENCH_COMPONENT_UNITS:
            raise ValueError(
                "component_units must be N,N,N,N*m,N*m,N*m"
            )
        if not isinstance(self.interval_model, QIntervalModel):
            raise TypeError("interval_model must be a QIntervalModel")
        if self.interval_model is not QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY:
            raise ValueError(
                "interval_model must be continuous_spectral_density"
            )

    def interval_weights(self, time_step: np.ndarray) -> np.ndarray:
        values = np.asarray(time_step, dtype=float)
        if (
            values.ndim != 1
            or values.size == 0
            or not np.all(np.isfinite(values))
            or np.any(values <= 0.0)
        ):
            raise ValueError("time_step must contain positive finite values")
        result = values.copy()
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class ExpectedResidualMoments:
    """MAP residuals and diagonal ``J H^-1 J.T`` corrections."""

    map_residual: np.ndarray
    covariance_correction: np.ndarray

    def __post_init__(self) -> None:
        residual = np.asarray(self.map_residual, dtype=float)
        correction = np.asarray(self.covariance_correction, dtype=float)
        if (
            residual.ndim != 2
            or residual.shape[0] == 0
            or residual.shape[1] != Q_DIMENSION
            or not np.all(np.isfinite(residual))
        ):
            raise ValueError("map_residual must be a finite N by 6 matrix")
        if correction.shape != residual.shape or not np.all(
            np.isfinite(correction)
        ):
            raise ValueError(
                "covariance_correction must match map_residual and be finite"
            )
        tolerance = 1.0e-12 * max(
            1.0, float(np.max(np.abs(correction)))
        )
        if np.any(correction < -tolerance):
            raise ValueError("covariance_correction cannot be negative")
        correction = np.maximum(correction, 0.0)
        residual = residual.copy()
        correction = correction.copy()
        residual.setflags(write=False)
        correction.setflags(write=False)
        object.__setattr__(self, "map_residual", residual)
        object.__setattr__(self, "covariance_correction", correction)

    @property
    def interval_count(self) -> int:
        return int(self.map_residual.shape[0])

    @property
    def expected_squared_residual(self) -> np.ndarray:
        result = self.map_residual * self.map_residual
        result += self.covariance_correction
        result.setflags(write=False)
        return result


def _positive_six(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (Q_DIMENSION,)
        or not np.all(np.isfinite(result))
        or np.any(result <= 0.0)
    ):
        raise ValueError("{} must contain six positive finite values".format(name))
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DiagonalQTarget:
    """Laplace-EM target split into MAP and covariance contributions."""

    definition: DiagonalQDefinition
    map_second_moment: np.ndarray
    covariance_correction: np.ndarray
    raw_target: np.ndarray
    target: np.ndarray
    floor: np.ndarray
    floor_active: np.ndarray
    interval_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.definition, DiagonalQDefinition):
            raise TypeError("definition must be DiagonalQDefinition")
        for name in (
            "map_second_moment",
            "covariance_correction",
            "raw_target",
            "target",
            "floor",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (Q_DIMENSION,) or not np.all(np.isfinite(value)):
                raise ValueError("{} must contain six finite values".format(name))
            if np.any(value < 0.0):
                raise ValueError("{} cannot be negative".format(name))
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        floor_active = np.asarray(self.floor_active)
        if floor_active.shape != (Q_DIMENSION,) or floor_active.dtype != np.bool_:
            raise ValueError("floor_active must contain six boolean flags")
        floor_active = floor_active.copy()
        floor_active.setflags(write=False)
        object.__setattr__(self, "floor_active", floor_active)
        if not np.allclose(
            self.raw_target,
            self.map_second_moment + self.covariance_correction,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError("raw target must equal its two contributions")
        if not np.array_equal(self.target, np.maximum(self.raw_target, self.floor)):
            raise ValueError("target must apply only the component-wise floor")
        if not np.array_equal(self.floor_active, self.raw_target < self.floor):
            raise ValueError("floor_active disagrees with the raw target")
        if (
            isinstance(self.interval_count, (bool, np.bool_))
            or not isinstance(self.interval_count, (int, np.integer))
            or self.interval_count <= 0
        ):
            raise ValueError("interval_count must be a positive integer")
        object.__setattr__(self, "interval_count", int(self.interval_count))


def compute_diagonal_q_target(
    definition: DiagonalQDefinition,
    moments: ExpectedResidualMoments,
    time_step: np.ndarray,
    floor: np.ndarray,
) -> DiagonalQTarget:
    """Compute the six M-step targets including Laplace covariance volume."""

    if not isinstance(definition, DiagonalQDefinition):
        raise TypeError("definition must be DiagonalQDefinition")
    if not isinstance(moments, ExpectedResidualMoments):
        raise TypeError("moments must be ExpectedResidualMoments")
    weights = definition.interval_weights(time_step)
    if weights.shape != (moments.interval_count,):
        raise ValueError("time_step count must match residual interval count")
    selected_floor = _positive_six(floor, "floor")
    divisor = float(moments.interval_count)
    map_contribution = np.sum(
        weights[:, None] * moments.map_residual**2, axis=0
    ) / divisor
    covariance_contribution = np.sum(
        weights[:, None] * moments.covariance_correction, axis=0
    ) / divisor
    raw_target = map_contribution + covariance_contribution
    target = np.maximum(raw_target, selected_floor)
    return DiagonalQTarget(
        definition=definition,
        map_second_moment=map_contribution,
        covariance_correction=covariance_contribution,
        raw_target=raw_target,
        target=target,
        floor=selected_floor,
        floor_active=raw_target < selected_floor,
        interval_count=moments.interval_count,
    )


@dataclass(frozen=True)
class QInnerEvaluation:
    """Inner MAP/Laplace evidence evaluated at one explicit diagonal Q."""

    q: np.ndarray
    successful: bool
    map_objective: float
    approximate_marginal_objective: float
    lag: float
    failure_reason: str
    warm_start: Any = None

    def __post_init__(self) -> None:
        q = _positive_six(self.q, "q")
        if not isinstance(self.successful, (bool, np.bool_)):
            raise TypeError("successful must be boolean")
        successful = bool(self.successful)
        values = np.asarray(
            (
                self.map_objective,
                self.approximate_marginal_objective,
                self.lag,
            ),
            dtype=float,
        )
        if successful:
            if not np.all(np.isfinite(values)):
                raise ValueError("successful inner objectives and lag must be finite")
            if self.failure_reason != "":
                raise ValueError("successful evaluation cannot have a failure reason")
        else:
            if not isinstance(self.failure_reason, str) or not self.failure_reason:
                raise ValueError("failed evaluation needs a failure reason")
            if np.any(np.isnan(values)):
                raise ValueError("failed evaluation diagnostics cannot be NaN")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "successful", successful)
        object.__setattr__(self, "map_objective", float(self.map_objective))
        object.__setattr__(
            self,
            "approximate_marginal_objective",
            float(self.approximate_marginal_objective),
        )
        object.__setattr__(self, "lag", float(self.lag))


@dataclass(frozen=True)
class QUpdateAttempt:
    alpha: float
    candidate_q: np.ndarray
    evaluation: QInnerEvaluation
    accepted: bool
    rejection_reason: str

    def __post_init__(self) -> None:
        alpha = float(self.alpha)
        if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in (0, 1]")
        candidate = _positive_six(self.candidate_q, "candidate_q")
        if not isinstance(self.evaluation, QInnerEvaluation):
            raise TypeError("evaluation must be QInnerEvaluation")
        if not np.array_equal(candidate, self.evaluation.q):
            raise ValueError("candidate_q disagrees with its evaluation")
        if not isinstance(self.accepted, (bool, np.bool_)):
            raise TypeError("accepted must be boolean")
        accepted = bool(self.accepted)
        if accepted and self.rejection_reason != "":
            raise ValueError("accepted attempt cannot have a rejection reason")
        if not accepted and (
            not isinstance(self.rejection_reason, str)
            or not self.rejection_reason
        ):
            raise ValueError("rejected attempt needs a reason")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "candidate_q", candidate)
        object.__setattr__(self, "accepted", accepted)


@dataclass(frozen=True)
class QUpdateResult:
    input_evaluation: QInnerEvaluation
    target: DiagonalQTarget
    attempts: Tuple[QUpdateAttempt, ...]
    accepted: bool
    accepted_q: np.ndarray
    accepted_alpha: float
    max_log_q_change: float
    termination_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.input_evaluation, QInnerEvaluation):
            raise TypeError("input_evaluation must be QInnerEvaluation")
        if not self.input_evaluation.successful:
            raise ValueError("input_evaluation must be successful")
        if not isinstance(self.target, DiagonalQTarget):
            raise TypeError("target must be DiagonalQTarget")
        if (
            not isinstance(self.termination_reason, str)
            or not self.termination_reason
        ):
            raise ValueError("termination_reason must be a non-empty string")
        if type(self.attempts) is not tuple:
            raise TypeError("attempts must be a tuple")
        fixed_by_request = self.termination_reason == "fixed_by_request"
        if not self.attempts and not fixed_by_request:
            raise TypeError("attempts must be non-empty for a Q update")
        if self.attempts and fixed_by_request:
            raise ValueError("fixed Q cannot contain update attempts")
        if any(not isinstance(value, QUpdateAttempt) for value in self.attempts):
            raise TypeError("attempts contain an invalid value")
        if not isinstance(self.accepted, (bool, np.bool_)):
            raise TypeError("accepted must be boolean")
        accepted = bool(self.accepted)
        accepted_q = _positive_six(self.accepted_q, "accepted_q")
        accepted_attempts = tuple(value for value in self.attempts if value.accepted)
        if accepted:
            if len(accepted_attempts) != 1 or accepted_attempts[0] is not self.attempts[-1]:
                raise ValueError("only the final attempted Q may be accepted")
            if not np.array_equal(accepted_q, accepted_attempts[0].candidate_q):
                raise ValueError("accepted_q disagrees with the accepted attempt")
            if self.accepted_alpha != accepted_attempts[0].alpha:
                raise ValueError("accepted_alpha disagrees with the accepted attempt")
        else:
            if accepted_attempts:
                raise ValueError("a rejected result cannot contain an accepted attempt")
            if not np.array_equal(accepted_q, self.input_evaluation.q):
                raise ValueError("a rejected update must retain the input Q")
            if self.accepted_alpha != 0.0:
                raise ValueError("a rejected update must have zero accepted_alpha")
        expected_change = float(
            np.max(np.abs(np.log(accepted_q) - np.log(self.input_evaluation.q)))
        )
        if not np.isclose(
            float(self.max_log_q_change), expected_change, rtol=1.0e-12, atol=1.0e-15
        ):
            raise ValueError("max_log_q_change is inconsistent")
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "accepted_q", accepted_q)
        object.__setattr__(self, "accepted_alpha", float(self.accepted_alpha))
        object.__setattr__(self, "max_log_q_change", expected_change)


def damped_diagonal_q_update(
    input_evaluation: QInnerEvaluation,
    target: DiagonalQTarget,
    evaluator: Callable[[np.ndarray, Any], QInnerEvaluation],
    *,
    minimum_alpha: float = 1.0 / 64.0,
    marginal_objective_tolerance: float = 0.0,
) -> QUpdateResult:
    """Backtrack in log-Q and accept only finite non-worsening evidence."""

    if not isinstance(input_evaluation, QInnerEvaluation) or not input_evaluation.successful:
        raise ValueError("input_evaluation must be a successful QInnerEvaluation")
    if not isinstance(target, DiagonalQTarget):
        raise TypeError("target must be DiagonalQTarget")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    minimum = float(minimum_alpha)
    tolerance = float(marginal_objective_tolerance)
    if not np.isfinite(minimum) or minimum <= 0.0 or minimum > 1.0:
        raise ValueError("minimum_alpha must be in (0, 1]")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("marginal_objective_tolerance must be non-negative")

    input_log = np.log(input_evaluation.q)
    target_log = np.log(target.target)
    attempts = []
    alpha = 1.0
    accepted_evaluation = None
    while alpha + np.finfo(float).eps >= minimum:
        candidate_q = np.exp((1.0 - alpha) * input_log + alpha * target_log)
        evaluation = evaluator(candidate_q.copy(), input_evaluation.warm_start)
        if not isinstance(evaluation, QInnerEvaluation):
            raise TypeError("Q evaluator must return QInnerEvaluation")
        if not np.array_equal(evaluation.q, candidate_q):
            raise ValueError("Q evaluator returned a different candidate Q")
        if not evaluation.successful:
            accepted = False
            reason = "inner_failure:{}".format(evaluation.failure_reason)
        elif evaluation.approximate_marginal_objective > (
            input_evaluation.approximate_marginal_objective + tolerance
        ):
            accepted = False
            reason = "approximate_marginal_objective_worsened"
        else:
            accepted = True
            reason = ""
            accepted_evaluation = evaluation
        attempts.append(
            QUpdateAttempt(alpha, candidate_q, evaluation, accepted, reason)
        )
        if accepted:
            break
        alpha *= 0.5

    if accepted_evaluation is None:
        accepted_q = input_evaluation.q
        accepted_alpha = 0.0
        termination_reason = "all_damped_candidates_rejected"
        accepted = False
    else:
        accepted_q = accepted_evaluation.q
        accepted_alpha = attempts[-1].alpha
        termination_reason = "accepted_nonworsening_candidate"
        accepted = True
    change = float(
        np.max(np.abs(np.log(accepted_q) - np.log(input_evaluation.q)))
    )
    return QUpdateResult(
        input_evaluation=input_evaluation,
        target=target,
        attempts=tuple(attempts),
        accepted=accepted,
        accepted_q=accepted_q,
        accepted_alpha=accepted_alpha,
        max_log_q_change=change,
        termination_reason=termination_reason,
    )


__all__ = [
    "BODY_WRENCH_COMPONENT_NAMES",
    "BODY_WRENCH_COMPONENT_UNITS",
    "BODY_WRENCH_QUANTITY",
    "DiagonalQDefinition",
    "DiagonalQTarget",
    "ExpectedResidualMoments",
    "QInnerEvaluation",
    "QIntervalModel",
    "QUpdateAttempt",
    "QUpdateResult",
    "compute_diagonal_q_target",
    "damped_diagonal_q_update",
]
