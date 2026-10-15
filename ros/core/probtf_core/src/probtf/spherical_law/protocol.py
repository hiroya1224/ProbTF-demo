"""ISL backend protocol and law value types."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from probtf.distributions import BinghamOrientation
from probtf.distributions.validation import immutable_array, immutable_symmetric_matrix
from probtf.provenance import ApproximationInfo


class IslBackendUnavailableError(NotImplementedError):
    def __init__(self, message, code="UNAVAILABLE_BACKEND"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class IslEvaluationOptions:
    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 1e-6

    def __post_init__(self):
        for name in ("absolute_tolerance", "relative_tolerance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("{} must be finite and non-negative.".format(name))
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class InducedDirectionLaw:
    approximation: ApproximationInfo = field(default_factory=ApproximationInfo)


@dataclass(frozen=True)
class DiracDirectionLaw(InducedDirectionLaw):
    direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    def __post_init__(self):
        direction = immutable_array(self.direction, (3,), "direction")
        norm = float(np.linalg.norm(direction))
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-8):
            raise ValueError("direction must have unit norm.")
        direction = direction / norm
        direction.setflags(write=False)
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True)
class UniformDirectionLaw(InducedDirectionLaw):
    def density(self, direction):
        value = np.asarray(direction, dtype=float)
        if value.shape != (3,) or not np.isclose(np.linalg.norm(value), 1.0, atol=1e-8):
            raise ValueError("direction must be a unit 3-vector.")
        return 1.0 / (4.0 * np.pi)

    def log_density(self, direction):
        return float(np.log(self.density(direction)))


@dataclass(frozen=True)
class InducedVectorLaw:
    approximation: ApproximationInfo = field(default_factory=ApproximationInfo)


@dataclass(frozen=True)
class DiracVectorLaw(InducedVectorLaw):
    vector: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        object.__setattr__(self, "vector", immutable_array(self.vector, (3,), "vector"))


@dataclass(frozen=True)
class ScaledDirectionVectorLaw(InducedVectorLaw):
    direction_law: InducedDirectionLaw = field(default_factory=UniformDirectionLaw)
    radius: float = 1.0

    def __post_init__(self):
        if not isinstance(self.direction_law, InducedDirectionLaw):
            raise TypeError("direction_law must be an InducedDirectionLaw.")
        radius = float(self.radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive and finite.")
        object.__setattr__(self, "radius", radius)

    def surface_log_density(self, vector):
        """Density on the radius-r sphere's AREA, not 3D Lebesgue density."""
        value = np.asarray(vector, dtype=float)
        if (value.shape != (3,) or not np.all(np.isfinite(value)) or
                not np.isclose(np.linalg.norm(value), self.radius, rtol=1e-8, atol=0.)):
            raise ValueError("vector must lie on the law's fixed-radius sphere.")
        return self.direction_law.log_density(value / self.radius) - 2. * np.log(self.radius)

    def surface_density(self, vector):
        return float(np.exp(self.surface_log_density(vector)))


@dataclass(frozen=True)
class TangentInducedVectorLaw(InducedVectorLaw):
    mean: np.ndarray = field(default_factory=lambda: np.zeros(3))
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    mode: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    radius: float = 1.0

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
        object.__setattr__(self, "mode", immutable_array(self.mode, (3,), "mode"))
        object.__setattr__(self, "radius", float(self.radius))


@dataclass(frozen=True)
class TangentInducedDirectionLaw(InducedDirectionLaw):
    mean: np.ndarray = field(default_factory=lambda: np.zeros(3))
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    mode: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

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
        object.__setattr__(self, "mode", immutable_array(self.mode, (3,), "mode"))


class IslInducedLawBackend(Protocol):
    def rotate_direction(
        self,
        orientation: BinghamOrientation,
        direction: np.ndarray,
        options: IslEvaluationOptions,
    ) -> InducedDirectionLaw:
        ...

    def rotate_vector(
        self,
        orientation: BinghamOrientation,
        vector: np.ndarray,
        options: IslEvaluationOptions,
    ) -> InducedVectorLaw:
        ...
