"""Exact spherical image of one Bingham rotation acting on a fixed direction."""
from dataclasses import dataclass

import numpy as np
from scipy.special import i0e

from probtf.distributions import BinghamOrientation, OrientationKind
from probtf.distributions.validation import immutable_array
from probtf.geometry import quat_left_matrix, quat_right_matrix, quat_to_rotmat
from probtf.spherical_law.protocol import (
    DiracDirectionLaw,
    InducedDirectionLaw,
    IslBackendUnavailableError,
    UniformDirectionLaw,
)


_PURE_LEFT = np.array([quat_left_matrix(np.r_[0., axis]) for axis in np.eye(3)])


def _unit_directions(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError(name + " must contain finite 3-vectors.")
    lengths = np.linalg.norm(values, axis=-1)
    if not np.all(np.isclose(lengths, 1., rtol=0., atol=1e-8)):
        raise ValueError(name + " must contain unit 3-vectors.")
    return values / lengths[..., None]


@dataclass(frozen=True)
class FiniteBinghamInducedDirectionLaw(InducedDirectionLaw):
    """The exact image law, with density relative to sphere AREA (total 4*pi).

    ``orientation`` retains the native Bingham law. Only its normalizer is
    evaluated numerically; the image density uses the closed fiber integral,
    with no tangent approximation or rotation sampling.
    """

    orientation: object = None
    direction: object = None
    log_normalizer: float = 0.
    normalizer_integration_steps: int = 0

    def __post_init__(self):
        if not isinstance(self.orientation, BinghamOrientation) or self.orientation.kind is not OrientationKind.FINITE_BINGHAM:
            raise TypeError("FiniteBinghamInducedDirectionLaw requires a finite Bingham orientation.")
        direction = _unit_directions(self.direction, "direction")
        object.__setattr__(self, "direction", immutable_array(direction, (3,), "direction"))
        parameter = self.orientation.backend_parameter_matrix()
        parameter.setflags(write=False)
        object.__setattr__(self, "_parameter", parameter)
        if not np.isfinite(self.log_normalizer):
            raise ValueError("log_normalizer must be finite.")

    def log_density(self, direction):
        """Evaluate chapter-4 ``globaldensity`` at one or several directions.

        Restricting A to an orthonormal basis of the rank-two fiber projector
        gives G. Then T=trace(G)/2 and sqrt(S)=hypot((G00-G11)/2,G01).
        This is the theorem's S without subtracting large nearly equal traces.
        """
        values = _unit_directions(direction, "direction")
        right = quat_right_matrix(np.r_[0., self.direction])
        left = np.einsum("...i,ijk->...jk", values, _PURE_LEFT)
        projector = .5 * (np.eye(4) - left @ right)
        _eigenvalues, eigenvectors = np.linalg.eigh(projector)
        basis = eigenvectors[..., :, -2:]
        restricted = np.swapaxes(basis, -2, -1) @ self._parameter @ basis
        center = .5 * (restricted[..., 0, 0] + restricted[..., 1, 1])
        radius = np.hypot(.5 * (restricted[..., 0, 0] - restricted[..., 1, 1]), restricted[..., 0, 1])
        # i0e(s)=exp(-s) I0(s); the canonical max-eigenvalue-zero gauge
        # keeps center+radius nonpositive even for strongly concentrated laws.
        result = np.log(np.pi / 2.) - self.log_normalizer + center + radius + np.log(i0e(radius))
        return float(result) if result.ndim == 0 else result

    def density(self, direction):
        return np.exp(self.log_density(direction))


def special_case_direction_law(orientation, direction):
    value = np.asarray(direction, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("direction must be a finite vector with shape (3,).")
    norm = float(np.linalg.norm(value))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("direction must have unit norm.")
    value = value / norm
    if orientation.kind is OrientationKind.DIRAC:
        return DiracDirectionLaw(direction=quat_to_rotmat(orientation.mode_wxyz) @ value)
    if orientation.kind is OrientationKind.UNIFORM:
        return UniformDirectionLaw()
    raise IslBackendUnavailableError(
        "No exact finite-Bingham induced-direction evaluator is implemented.",
        "UNAVAILABLE_EXACT_ISL_BACKEND",
    )

