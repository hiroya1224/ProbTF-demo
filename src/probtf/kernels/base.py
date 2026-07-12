"""Lazy kernel expression and evaluation result contracts."""

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from probtf.distributions import DistributionStatus
from probtf.distributions.validation import immutable_array, immutable_symmetric_matrix
from probtf.provenance import ApproximationInfo


class KernelRepresentation(Enum):
    EXPRESSION = "expression"
    NUMERICAL_LAW = "numerical_law"
    SAMPLES = "samples"
    MOMENTS = "moments"
    CLOSED_MIXTURE = "closed_mixture"


class KernelDiagnosticCode(Enum):
    OK = "ok"
    ZERO_MASS = "zero_mass"
    INVALID_DISTRIBUTION = "invalid_distribution"
    UNAVAILABLE_BACKEND = "unavailable_backend"
    UNSUPPORTED_REPRESENTATION = "unsupported_representation"
    DEPENDENCY_UNRESOLVED = "dependency_unresolved"
    APPROXIMATION_USED = "approximation_used"


@dataclass(frozen=True)
class KernelDiagnostics:
    codes: Tuple[KernelDiagnosticCode, ...] = ()
    messages: Tuple[str, ...] = ()
    repeated_dependency_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class KernelEvaluationOptions:
    representation: KernelRepresentation = KernelRepresentation.EXPRESSION
    sample_count: int = 0
    rng: Optional[object] = None
    isl_options: Optional[object] = None

    def __post_init__(self):
        if not isinstance(self.representation, KernelRepresentation):
            raise TypeError("representation must be KernelRepresentation.")
        count = int(self.sample_count)
        if count < 0 or count != self.sample_count:
            raise ValueError("sample_count must be a non-negative integer.")
        object.__setattr__(self, "sample_count", count)


@dataclass(frozen=True)
class KernelResult:
    status: DistributionStatus
    representation: KernelRepresentation
    value: object
    approximation: ApproximationInfo = field(default_factory=ApproximationInfo)
    diagnostics: KernelDiagnostics = field(default_factory=KernelDiagnostics)


class TransformKernelExpression(ABC):
    def latent_dependency_ids(self):
        return frozenset()


class PointLaw(ABC):
    pass


@dataclass(frozen=True)
class DiracPointLaw(PointLaw):
    point: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "point", immutable_array(self.point, (3,), "point"))


@dataclass(frozen=True)
class GaussianPointLaw(PointLaw):
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "mean", immutable_array(self.mean, (3,), "mean"))
        object.__setattr__(
            self,
            "covariance",
            immutable_symmetric_matrix(
                self.covariance,
                3,
                "covariance",
                positive_semidefinite=True,
            ),
        )


@dataclass(frozen=True)
class AppliedKernelExpression:
    kernel: TransformKernelExpression
    input_law: PointLaw

