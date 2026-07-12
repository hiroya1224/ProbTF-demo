import numpy as np

from probtf.distributions import OrientationKind
from probtf.geometry import quat_to_rotmat
from probtf.spherical_law.protocol import (
    DiracDirectionLaw,
    IslBackendUnavailableError,
    UniformDirectionLaw,
)


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

