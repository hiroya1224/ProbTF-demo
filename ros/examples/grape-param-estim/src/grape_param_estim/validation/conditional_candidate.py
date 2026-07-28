"""Compatibility boundary for evidence-gated optional model candidates."""

from grape_param_estim.alternative_backends import (
    ConditionalCandidateGate,
    evaluate_conditional_candidate,
)

__all__ = [
    "ConditionalCandidateGate",
    "evaluate_conditional_candidate",
]
