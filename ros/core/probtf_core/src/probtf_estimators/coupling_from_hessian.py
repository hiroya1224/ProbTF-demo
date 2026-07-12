"""Construct ``vec(R)`` translation coupling from a local pose Hessian."""

from dataclasses import dataclass

import numpy as np

from probtf.geometry import right_perturbation_vec_rotation_jacobian


def _matrix(values, shape, name):
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError("{} must be a finite matrix with shape {}.".format(name, shape))
    return matrix.copy()


@dataclass(frozen=True)
class HessianCouplingResult:
    rotation_coupling: np.ndarray
    local_translation_map: np.ndarray
    rotation_jacobian: np.ndarray

    def __post_init__(self):
        for name, shape in (
            ("rotation_coupling", (3, 9)),
            ("local_translation_map", (3, 3)),
            ("rotation_jacobian", (9, 3)),
        ):
            value = _matrix(getattr(self, name), shape, name)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def coupling_from_hessian(hessian_xx, hessian_xu, reference_quaternion_wxyz):
    """Return the minimum-norm coupling ``C = B D+``.

    The local derivation is the plan's right perturbation convention:
    ``B = -H_xx^-1 H_xu``, ``D = d vec(R exp([u]x))/du``, and
    ``D+ = D.T / 2``.  Singular ``H_xx`` is rejected; this helper never adds
    unreported regularization.
    """

    hxx = _matrix(hessian_xx, (3, 3), "hessian_xx")
    hxu = _matrix(hessian_xu, (3, 3), "hessian_xu")
    if not np.allclose(hxx, hxx.T, rtol=0.0, atol=1e-10):
        raise ValueError("hessian_xx must be symmetric.")
    try:
        local_map = -np.linalg.solve(hxx, hxu)
    except np.linalg.LinAlgError as exc:
        raise ValueError("hessian_xx must be nonsingular; no implicit regularization is applied.") from exc
    jacobian = right_perturbation_vec_rotation_jacobian(reference_quaternion_wxyz)
    coupling = local_map @ (0.5 * jacobian.T)
    return HessianCouplingResult(coupling, local_map, jacobian)

