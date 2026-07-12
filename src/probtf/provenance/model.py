"""Structured provenance and approximation metadata."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import math


class ApproximationKind(Enum):
    EXACT = "exact"
    PRODUCER_SUPPLIED = "producer_supplied"
    LEGACY_ADAPTER = "legacy_adapter"
    TANGENT_SURROGATE = "tangent_surrogate"
    NUMERICAL_INTEGRATION = "numerical_integration"
    MONTE_CARLO = "monte_carlo"
    MOMENT_SUMMARY = "moment_summary"
    MIXTURE_REDUCTION = "mixture_reduction"
    BINGHAM_CLOSURE = "bingham_closure"
    REPRESENTATIVE_PROJECTION = "representative_projection"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ApproximationInfo:
    """Describe whether and how a value differs from its original joint law."""

    kind: ApproximationKind = ApproximationKind.EXACT
    lossy: bool = False
    detail: str = ""
    source: str = ""
    error_bound: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.kind, ApproximationKind):
            raise TypeError("kind must be an ApproximationKind.")
        object.__setattr__(self, "lossy", bool(self.lossy))
        object.__setattr__(self, "detail", str(self.detail).strip())
        object.__setattr__(self, "source", str(self.source).strip())
        if self.error_bound is not None:
            value = float(self.error_bound)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("error_bound must be finite and non-negative.")
            object.__setattr__(self, "error_bound", value)
        if self.kind is ApproximationKind.EXACT and self.lossy:
            raise ValueError("Exact metadata cannot be marked lossy.")


def _identifiers(values, name):
    result = tuple(str(value).strip() for value in values)
    if any(not value for value in result):
        raise ValueError("{} must not contain empty identifiers.".format(name))
    if len(set(result)) != len(result):
        raise ValueError("{} must contain unique identifiers.".format(name))
    return result


@dataclass(frozen=True)
class Provenance:
    source_ids: Tuple[str, ...] = ()
    derived_from_edge_ids: Tuple[str, ...] = ()
    method: str = ""
    detail: str = ""

    def __post_init__(self):
        object.__setattr__(self, "source_ids", _identifiers(self.source_ids, "source_ids"))
        object.__setattr__(
            self,
            "derived_from_edge_ids",
            _identifiers(self.derived_from_edge_ids, "derived_from_edge_ids"),
        )
        object.__setattr__(self, "method", str(self.method).strip())
        object.__setattr__(self, "detail", str(self.detail).strip())


@dataclass(frozen=True)
class ComponentProvenance(Provenance):
    pass


@dataclass(frozen=True)
class TransformProvenance(Provenance):
    pass
