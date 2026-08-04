"""Nonlinear batch problem boundary around analytic factor evaluations."""

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

from grape_param_estim.batch.factor import FactorEvaluation
from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.linearize import (
    SparseBatchLinearization,
    assemble_sparse_linearization,
)
from grape_param_estim.batch.state import BatchState, StateScaling


FactorEvaluator = Callable[[BatchState], Sequence[FactorEvaluation]]


class RecoverableModelEvaluationError(ValueError):
    """A trial point failed a declared model-domain or active-set gate."""


@dataclass(frozen=True)
class ProblemLinearization:
    """Factor evaluations and their assembled sparse linearization."""

    factors: Tuple[FactorEvaluation, ...]
    sparse: SparseBatchLinearization


@dataclass(frozen=True)
class BatchProblem:
    """Complete factor-evaluation boundary for one fixed Q and delay."""

    layout: VariableLayout
    scaling: StateScaling
    factor_evaluator: FactorEvaluator

    def __post_init__(self) -> None:
        if not isinstance(self.layout, VariableLayout):
            raise TypeError("layout must be a VariableLayout")
        if not isinstance(self.scaling, StateScaling):
            raise TypeError("scaling must be a StateScaling")
        if not callable(self.factor_evaluator):
            raise TypeError("factor_evaluator must be callable")

    def evaluate_factors(
        self,
        state: BatchState,
    ) -> Tuple[FactorEvaluation, ...]:
        """Evaluate every analytic factor at one complete nonlinear state."""

        if not isinstance(state, BatchState):
            raise TypeError("state must be a BatchState")
        if state.layout != self.layout:
            raise ValueError("state layout does not match this problem")
        try:
            factors = tuple(self.factor_evaluator(state))
        except RecoverableModelEvaluationError:
            raise
        if not factors:
            raise ValueError("factor_evaluator must return at least one factor")
        if any(not isinstance(factor, FactorEvaluation) for factor in factors):
            raise TypeError(
                "factor_evaluator must return only FactorEvaluation values"
            )
        return factors

    def linearize(self, state: BatchState) -> ProblemLinearization:
        """Evaluate factors and assemble their whitened sparse GN system."""

        factors = self.evaluate_factors(state)
        sparse = assemble_sparse_linearization(self.layout, factors)
        return ProblemLinearization(factors=factors, sparse=sparse)

    @property
    def coordinate_scale(self):
        """Return the immutable physical-per-scaled layout vector."""

        return self.scaling.vector_for(self.layout)


__all__ = [
    "BatchProblem",
    "FactorEvaluator",
    "ProblemLinearization",
    "RecoverableModelEvaluationError",
]
