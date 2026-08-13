#!/usr/bin/env python3
"""Local and overlap-aware covariance for geometric SG acceleration data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import null_space
from scipy.spatial.transform import Rotation

try:  # Package import and direct-script import are both supported.
    from .savgol_trajectory import (
        PoseSgEvaluation,
        left_jacobian_with_directional_derivative,
        skew,
    )
except ImportError:  # pragma: no cover - exercised by CLI smoke tests
    from savgol_trajectory import (  # type: ignore
        PoseSgEvaluation,
        left_jacobian_with_directional_derivative,
        skew,
    )


GRAVITY_WORLD = np.asarray((0.0, 0.0, -9.80665), dtype=float)
COVARIANCE_MODES = (
    "full",
    "identity",
    "diagonal",
    "block_s_alpha",
    "full_no_R_uncertainty_in_s",
    "full_no_position_rotation_cross",
    "global_full",
)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def _symmetric(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    return 0.5 * (array + array.T)


def machine_pseudoinverse_symmetric(value: np.ndarray) -> np.ndarray:
    """Moore--Penrose inverse using only a machine-precision rank tolerance."""

    matrix = _symmetric(value)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    tolerance = max(matrix.shape) * np.finfo(float).eps * scale
    inverse = np.zeros_like(eigenvalues)
    retained = eigenvalues > tolerance
    inverse[retained] = 1.0 / eigenvalues[retained]
    return _symmetric((eigenvectors * inverse) @ eigenvectors.T)


def whitening_matrix(value: np.ndarray) -> np.ndarray:
    """Symmetric square root of ``value``'s machine-rank pseudoinverse."""

    matrix = _symmetric(value)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    tolerance = max(matrix.shape) * np.finfo(float).eps * scale
    inverse_root = np.zeros_like(eigenvalues)
    retained = eigenvalues > tolerance
    inverse_root[retained] = 1.0 / np.sqrt(eigenvalues[retained])
    return _symmetric((eigenvectors * inverse_root) @ eigenvectors.T)


def generalized_acceleration_from_xi(
    xi: Sequence[float],
    rotation_reference: np.ndarray,
    *,
    geometric_correction: bool = True,
    gravity_world: np.ndarray = GRAVITY_WORLD,
) -> np.ndarray:
    """Map ``[a_S, rho0, rho1, rho2]`` to ``[specific force, alpha_B]``."""

    value = np.asarray(xi, dtype=float)
    reference = np.asarray(rotation_reference, dtype=float)
    gravity = np.asarray(gravity_world, dtype=float)
    if (
        value.shape != (12,)
        or reference.shape != (3, 3)
        or gravity.shape != (3,)
        or any(np.any(~np.isfinite(x)) for x in (value, reference, gravity))
    ):
        raise ValueError("generalized-acceleration propagation input is invalid")
    acceleration = value[:3]
    rho0, rho1, rho2 = value[3:6], value[6:9], value[9:12]
    rotation = Rotation.from_rotvec(rho0).as_matrix() @ reference
    if geometric_correction:
        jacobian, directional = left_jacobian_with_directional_derivative(
            rho0, rho1
        )
        spatial_alpha = directional @ rho1 + jacobian @ rho2
    else:
        spatial_alpha = rho2
    specific = rotation.T @ (acceleration - gravity)
    alpha_body = rotation.T @ spatial_alpha
    return np.concatenate((specific, alpha_body))


def propagation_jacobian(
    xi: Sequence[float],
    rotation_reference: np.ndarray,
    *,
    geometric_correction: bool = True,
    include_rotation_uncertainty_in_specific_force: bool = True,
    gravity_world: np.ndarray = GRAVITY_WORLD,
) -> np.ndarray:
    """Jacobian of :func:`generalized_acceleration_from_xi`.

    The specific-force block and the rho1/rho2 angular blocks are analytic.
    Only the derivative of the geometric angular-acceleration expression with
    respect to rho0 uses a centered local numerical derivative (it contains a
    second directional derivative of ``J_l``).  This derivative is local to
    covariance propagation and never participates in parameter optimization.
    """

    value = np.asarray(xi, dtype=float)
    reference = np.asarray(rotation_reference, dtype=float)
    gravity = np.asarray(gravity_world, dtype=float)
    if value.shape != (12,):
        raise ValueError("xi must be 12-D")
    acceleration = value[:3]
    rho0, rho1, rho2 = value[3:6], value[6:9], value[9:12]
    rotation = Rotation.from_rotvec(rho0).as_matrix() @ reference
    left = left_jacobian_with_directional_derivative(rho0)
    result = np.zeros((6, 12), dtype=float)
    result[:3, :3] = rotation.T
    if include_rotation_uncertainty_in_specific_force:
        result[:3, 3:6] = (
            rotation.T @ skew(acceleration - gravity) @ left
        )

    if geometric_correction:
        _left, directional_rho1 = left_jacobian_with_directional_derivative(
            rho0, rho1
        )
        for column in range(3):
            direction = np.eye(3)[column]
            _, directional_basis = left_jacobian_with_directional_derivative(
                rho0, direction
            )
            result[3:, 6 + column] = rotation.T @ (
                directional_basis @ rho1 + directional_rho1 @ direction
            )
        result[3:, 9:12] = rotation.T @ left
    else:
        result[3:, 9:12] = rotation.T

    # The complete rho0 derivative includes both Exp(rho0) and D^2 J_l.
    # A cube-root-epsilon step balances truncation and roundoff for this smooth
    # three-dimensional map without introducing a scientific cutoff.
    for column in range(3):
        step = np.cbrt(np.finfo(float).eps) * max(1.0, abs(float(rho0[column])))
        plus = value.copy()
        minus = value.copy()
        plus[3 + column] += step
        minus[3 + column] -= step
        derivative = (
            generalized_acceleration_from_xi(
                plus,
                reference,
                geometric_correction=geometric_correction,
                gravity_world=gravity,
            )
            - generalized_acceleration_from_xi(
                minus,
                reference,
                geometric_correction=geometric_correction,
                gravity_world=gravity,
            )
        ) / (2.0 * step)
        result[3:, 3 + column] = derivative[3:]
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("covariance propagation Jacobian is non-finite")
    return result


def raw_pose_covariance(
    translation_residual: np.ndarray,
    rotation_residual: np.ndarray,
    degree: int,
    *,
    include_position_rotation_cross: bool = True,
) -> np.ndarray:
    """Estimate one full 6-D raw-pose covariance from local residuals."""

    translation = np.asarray(translation_residual, dtype=float)
    rotation = np.asarray(rotation_residual, dtype=float)
    if translation.shape != rotation.shape or translation.ndim != 2 or translation.shape[1] != 3:
        raise ValueError("translation/rotation local residuals must be matching Nx3")
    dof = int(translation.shape[0] - (int(degree) + 1))
    if dof <= 0:
        raise ValueError("local pose covariance has no residual degrees of freedom")
    residual = np.hstack((translation, rotation))
    covariance = _symmetric((residual.T @ residual) / dof)
    if not include_position_rotation_cross:
        covariance[:3, 3:] = 0.0
        covariance[3:, :3] = 0.0
    return covariance


def xi_operator(local_window: object) -> np.ndarray:
    """Per-observation linear maps from 6-D raw pose noise to 12-D xi."""

    translation = local_window.translation
    rotation = local_window.rotation_vector
    count = translation.sample_indices.size
    operator = np.zeros((count, 12, 6), dtype=float)
    identity = np.eye(3)
    for index in range(count):
        operator[index, :3, :3] = translation.derivative_rows[2, index] * identity
        operator[index, 3:6, 3:] = rotation.derivative_rows[0, index] * identity
        operator[index, 6:9, 3:] = rotation.derivative_rows[1, index] * identity
        operator[index, 9:12, 3:] = rotation.derivative_rows[2, index] * identity
    return operator


@dataclass(frozen=True)
class SgCovarianceEvaluation:
    """All local SG covariance products required by fitting and reporting."""

    mode: str
    z: np.ndarray
    local_omega: np.ndarray
    local_sigma_xi: np.ndarray
    local_sigma_z: np.ndarray
    propagation_jacobian: np.ndarray
    whitening: np.ndarray
    xi_operators: tuple[np.ndarray, ...]
    raw_indices: tuple[np.ndarray, ...]
    geometric_correction: bool
    include_position_rotation_cross: bool
    include_rotation_uncertainty_in_specific_force: bool

    def __post_init__(self) -> None:
        count = np.asarray(self.z).shape[0]
        expected = {
            "z": (count, 6),
            "local_omega": (count, 6, 6),
            "local_sigma_xi": (count, 12, 12),
            "local_sigma_z": (count, 6, 6),
            "propagation_jacobian": (count, 6, 12),
            "whitening": (count, 6, 6),
        }
        if self.mode not in COVARIANCE_MODES:
            raise ValueError("unknown covariance mode: {}".format(self.mode))
        if len(self.xi_operators) != count or len(self.raw_indices) != count:
            raise ValueError("SG covariance local-operator count mismatch")
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or np.any(~np.isfinite(value)):
                raise ValueError("{} has invalid shape or values".format(name))
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(
            self,
            "xi_operators",
            tuple(_readonly(value) for value in self.xi_operators),
        )
        object.__setattr__(
            self,
            "raw_indices",
            tuple(_readonly(value) for value in self.raw_indices),
        )

    @property
    def weight(self) -> np.ndarray:
        return np.asarray(
            [matrix.T @ matrix for matrix in self.whitening], dtype=float
        )


def build_sg_covariance(
    evaluation: PoseSgEvaluation,
    *,
    degree: int,
    mode: str = "full",
    geometric_correction: bool = True,
) -> SgCovarianceEvaluation:
    """Build local full covariance and one requested weighting ablation."""

    if mode not in COVARIANCE_MODES:
        raise ValueError("unknown covariance mode: {}".format(mode))
    include_cross = mode != "full_no_position_rotation_cross"
    include_r_in_s = mode != "full_no_R_uncertainty_in_s"
    count = evaluation.time.size
    z = np.empty((count, 6), dtype=float)
    omega = np.empty((count, 6, 6), dtype=float)
    sigma_xi = np.empty((count, 12, 12), dtype=float)
    sigma_z_full = np.empty((count, 6, 6), dtype=float)
    propagation = np.empty((count, 6, 12), dtype=float)
    operators: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for center, local in enumerate(evaluation.local_windows):
        local_omega = raw_pose_covariance(
            local.translation.residual,
            local.rotation_vector.residual,
            degree,
            include_position_rotation_cross=include_cross,
        )
        operator = xi_operator(local)
        local_sigma_xi = np.zeros((12, 12), dtype=float)
        for per_observation in operator:
            local_sigma_xi += per_observation @ local_omega @ per_observation.T
        xi = np.concatenate(
            (
                local.translation.second,
                local.rotation_vector.value,
                local.rotation_vector.first,
                local.rotation_vector.second,
            )
        )
        local_z = generalized_acceleration_from_xi(
            xi,
            local.rotation_reference,
            geometric_correction=geometric_correction,
        )
        local_g = propagation_jacobian(
            xi,
            local.rotation_reference,
            geometric_correction=geometric_correction,
            include_rotation_uncertainty_in_specific_force=include_r_in_s,
        )
        z[center] = local_z
        omega[center] = local_omega
        sigma_xi[center] = _symmetric(local_sigma_xi)
        propagation[center] = local_g
        sigma_z_full[center] = _symmetric(
            local_g @ local_sigma_xi @ local_g.T
        )
        operators.append(operator)
        indices.append(local.translation.sample_indices)

    if mode == "identity":
        sigma_z = np.repeat(np.eye(6)[None, :, :], count, axis=0)
    elif mode == "diagonal":
        sigma_z = np.asarray(
            [np.diag(np.diag(value)) for value in sigma_z_full], dtype=float
        )
    elif mode == "block_s_alpha":
        sigma_z = sigma_z_full.copy()
        sigma_z[:, :3, 3:] = 0.0
        sigma_z[:, 3:, :3] = 0.0
    elif mode == "global_full":
        global_value = _symmetric(np.mean(sigma_z_full, axis=0))
        sigma_z = np.repeat(global_value[None, :, :], count, axis=0)
    else:
        sigma_z = sigma_z_full
    whitening = np.asarray([whitening_matrix(item) for item in sigma_z])
    return SgCovarianceEvaluation(
        mode=mode,
        z=z,
        local_omega=omega,
        local_sigma_xi=sigma_xi,
        local_sigma_z=sigma_z,
        propagation_jacobian=propagation,
        whitening=whitening,
        xi_operators=tuple(operators),
        raw_indices=tuple(indices),
        geometric_correction=bool(geometric_correction),
        include_position_rotation_cross=include_cross,
        include_rotation_uncertainty_in_specific_force=include_r_in_s,
    )


def cross_time_generalized_covariance(
    covariance: SgCovarianceEvaluation, first: int, second: int
) -> np.ndarray:
    """Return ``Cov(z_first, z_second)`` from shared raw observations.

    A shared raw observation uses the arithmetic mean of the two local raw
    covariance estimates.  This preserves each marginal exactly when
    ``first == second`` and applies the same working raw-noise model to every
    overlap pair without a hand-selected temporal kernel.
    """

    first_indices = covariance.raw_indices[first]
    second_indices = covariance.raw_indices[second]
    common, first_locations, second_locations = np.intersect1d(
        first_indices, second_indices, return_indices=True
    )
    if common.size == 0:
        return np.zeros((6, 6), dtype=float)
    raw_covariance = 0.5 * (
        covariance.local_omega[first] + covariance.local_omega[second]
    )
    cross_xi = np.zeros((12, 12), dtype=float)
    first_operator = covariance.xi_operators[first]
    second_operator = covariance.xi_operators[second]
    for left, right in zip(first_locations, second_locations):
        cross_xi += (
            first_operator[left] @ raw_covariance @ second_operator[right].T
        )
    return (
        covariance.propagation_jacobian[first]
        @ cross_xi
        @ covariance.propagation_jacobian[second].T
    )


@dataclass(frozen=True)
class ParameterCovarianceResult:
    curvature: np.ndarray
    sandwich_middle: np.ndarray
    naive: np.ndarray
    overlap_corrected: np.ndarray
    gauge_basis: np.ndarray


def parameter_covariances(
    raw_parameter_jacobian: np.ndarray,
    covariance: SgCovarianceEvaluation,
    gauge_direction: Sequence[float],
) -> ParameterCovarianceResult:
    """Compute naive and overlap-corrected covariance on the exact gauge section."""

    jacobian = np.asarray(raw_parameter_jacobian, dtype=float)
    direction = np.asarray(gauge_direction, dtype=float)
    count, _, dimension = jacobian.shape
    if (
        jacobian.shape[:2] != (covariance.z.shape[0], 6)
        or direction.shape != (dimension,)
        or np.any(~np.isfinite(jacobian))
        or np.any(~np.isfinite(direction))
        or np.linalg.norm(direction) == 0.0
    ):
        raise ValueError("parameter covariance inputs are invalid")
    weights = covariance.weight
    curvature = np.zeros((dimension, dimension), dtype=float)
    for index in range(count):
        curvature += jacobian[index].T @ weights[index] @ jacobian[index]
    middle = np.zeros_like(curvature)
    for first in range(count):
        # Centered windows have compact overlap.  Stop once raw index ranges
        # no longer intersect; raw indices increase with center time.
        for second in range(count):
            if (
                covariance.raw_indices[first][-1]
                < covariance.raw_indices[second][0]
                or covariance.raw_indices[second][-1]
                < covariance.raw_indices[first][0]
            ):
                continue
            cross = cross_time_generalized_covariance(
                covariance, first, second
            )
            middle += (
                jacobian[first].T
                @ weights[first]
                @ cross
                @ weights[second]
                @ jacobian[second]
            )
    basis = null_space(direction.reshape(1, -1))
    reduced_curvature = _symmetric(basis.T @ curvature @ basis)
    reduced_inverse = machine_pseudoinverse_symmetric(reduced_curvature)
    naive = basis @ reduced_inverse @ basis.T
    reduced_middle = _symmetric(basis.T @ middle @ basis)
    overlap = basis @ (
        reduced_inverse @ reduced_middle @ reduced_inverse
    ) @ basis.T
    return ParameterCovarianceResult(
        curvature=_readonly(_symmetric(curvature)),
        sandwich_middle=_readonly(_symmetric(middle)),
        naive=_readonly(_symmetric(naive)),
        overlap_corrected=_readonly(_symmetric(overlap)),
        gauge_basis=_readonly(basis),
    )


def sum_mean_invariance(
    residual: np.ndarray, jacobian: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Return the exact algebraic sum/mean objective and curvature relation."""

    r = np.asarray(residual, dtype=float)
    j = np.asarray(jacobian, dtype=float)
    if r.ndim != 2 or j.shape[:2] != r.shape:
        raise ValueError("sum/mean inputs must be NxD residual/Jacobian blocks")
    count = r.shape[0]
    flat_r = r.reshape(-1)
    flat_j = j.reshape(-1, j.shape[-1])
    loss_sum = 0.5 * float(flat_r @ flat_r)
    hessian_sum = flat_j.T @ flat_j
    return {
        "sample_count": count,
        "loss_sum": loss_sum,
        "loss_mean": loss_sum / count,
        "hessian_sum": hessian_sum,
        "hessian_mean": hessian_sum / count,
    }
