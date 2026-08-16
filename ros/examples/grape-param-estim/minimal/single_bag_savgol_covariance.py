#!/usr/bin/env python3
"""Local and overlap-aware covariance for geometric SG acceleration data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

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


def _covariance_std_correlation(
    value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return component standard deviations and a zero-safe correlation."""

    covariance = _symmetric(value)
    diagonal = np.diag(covariance)
    scale = float(np.max(np.abs(diagonal))) if diagonal.size else 0.0
    tolerance = max(covariance.shape) * np.finfo(float).eps * scale
    variance = np.where(diagonal >= -tolerance, np.maximum(diagonal, 0.0), diagonal)
    if np.any(variance < 0.0):
        raise ValueError("covariance has a materially negative diagonal")
    standard_deviation = np.sqrt(variance)
    denominator = np.outer(standard_deviation, standard_deviation)
    correlation = np.zeros_like(covariance)
    np.divide(
        covariance,
        denominator,
        out=correlation,
        where=denominator > 0.0,
    )
    nonzero = standard_deviation > 0.0
    correlation[np.diag_indices_from(correlation)] = nonzero.astype(float)
    return _readonly(standard_deviation), _readonly(_symmetric(correlation))


def wrench_acceleration_closure_maps(
    mass_kg: float,
    inertia_kg_m2: np.ndarray,
    lever_arm_m: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the exact acceleration-to-wrench map and its inverse."""

    mass = float(mass_kg)
    inertia = _symmetric(np.asarray(inertia_kg_m2, dtype=float))
    lever = np.asarray(lever_arm_m, dtype=float)
    if (
        not np.isfinite(mass)
        or mass <= 0.0
        or inertia.shape != (3, 3)
        or lever.shape != (3,)
        or np.any(~np.isfinite(inertia))
        or np.any(~np.isfinite(lever))
    ):
        raise ValueError("wrench closure-map inputs are invalid")
    try:
        inertia_inverse = np.linalg.inv(inertia)
    except np.linalg.LinAlgError as error:
        raise ValueError("wrench closure inertia must be invertible") from error
    cross = skew(lever)
    acceleration_to_wrench = np.zeros((6, 6), dtype=float)
    acceleration_to_wrench[:3, :3] = mass * np.eye(3)
    acceleration_to_wrench[:3, 3:] = mass * cross
    acceleration_to_wrench[3:, 3:] = inertia
    wrench_to_acceleration = np.zeros((6, 6), dtype=float)
    wrench_to_acceleration[:3, :3] = np.eye(3) / mass
    wrench_to_acceleration[:3, 3:] = -cross @ inertia_inverse
    wrench_to_acceleration[3:, 3:] = inertia_inverse
    closure = wrench_to_acceleration @ acceleration_to_wrench
    if not np.allclose(closure, np.eye(6), rtol=3.0e-14, atol=3.0e-14):
        raise RuntimeError("wrench closure maps are not machine-precision inverses")
    return _readonly(acceleration_to_wrench), _readonly(wrench_to_acceleration)


@dataclass(frozen=True)
class ResidualWrenchUncertainty:
    """Post-fit residual-wrench products in the nominal-mass gauge."""

    mass_gauge_scale: float
    fixed_mass_kg: float
    modeled_wrench: np.ndarray
    required_wrench: np.ndarray
    wrench: np.ndarray
    centered_wrench: np.ndarray
    mean: np.ndarray
    empirical_covariance: np.ndarray
    empirical_std: np.ndarray
    empirical_correlation: np.ndarray
    sg_covariance_per_time: np.ndarray
    sg_covariance_mean: np.ndarray
    sg_std: np.ndarray
    sg_correlation: np.ndarray
    excess_covariance_raw: np.ndarray
    excess_covariance_raw_eigenvalues: np.ndarray
    appreciably_negative_raw_eigenvalues: np.ndarray
    raw_negative_eigenvalue_tolerance: float
    model_discrepancy_covariance: np.ndarray
    model_discrepancy_eigenvalues: np.ndarray
    model_discrepancy_std: np.ndarray
    model_discrepancy_correlation: np.ndarray
    acceleration_model_discrepancy_covariance: np.ndarray
    acceleration_model_discrepancy_eigenvalues: np.ndarray
    acceleration_model_discrepancy_std: np.ndarray
    acceleration_model_discrepancy_correlation: np.ndarray
    acceleration_to_wrench: np.ndarray
    wrench_to_acceleration: np.ndarray


def residual_wrench_uncertainty(
    *,
    raw_residual_wrench: np.ndarray,
    modeled_wrench: np.ndarray,
    required_wrench: np.ndarray,
    estimated_mass_kg: float,
    estimated_inertia_kg_m2: np.ndarray,
    fixed_mass_kg: float,
    lever_arm_m: Sequence[float],
    reference_sigma_z: np.ndarray,
) -> ResidualWrenchUncertainty:
    """Estimate excess wrench fluctuation without feeding it into the fit.

    ``reference_sigma_z`` is deliberately explicit: callers must provide the
    full reference SG covariance, independently of the optimization metric.
    """

    raw = np.asarray(raw_residual_wrench, dtype=float)
    modeled = np.asarray(modeled_wrench, dtype=float)
    required = np.asarray(required_wrench, dtype=float)
    sigma_z = np.asarray(reference_sigma_z, dtype=float)
    estimated_mass = float(estimated_mass_kg)
    fixed_mass = float(fixed_mass_kg)
    inertia = np.asarray(estimated_inertia_kg_m2, dtype=float)
    if (
        raw.ndim != 2
        or raw.shape[1] != 6
        or raw.shape[0] < 2
        or modeled.shape != raw.shape
        or required.shape != raw.shape
        or sigma_z.shape != (raw.shape[0], 6, 6)
        or inertia.shape != (3, 3)
        or any(np.any(~np.isfinite(item)) for item in (raw, modeled, required, sigma_z, inertia))
        or not np.isfinite(estimated_mass)
        or estimated_mass <= 0.0
        or not np.isfinite(fixed_mass)
        or fixed_mass <= 0.0
    ):
        raise ValueError("residual-wrench uncertainty inputs are invalid")

    scale = fixed_mass / estimated_mass
    fixed_inertia = scale * inertia
    acceleration_to_wrench, wrench_to_acceleration = (
        wrench_acceleration_closure_maps(
            fixed_mass, fixed_inertia, lever_arm_m
        )
    )
    wrench = scale * raw
    modeled_fixed = scale * modeled
    required_fixed = scale * required
    mean = np.mean(wrench, axis=0)
    centered = wrench - mean
    empirical = _symmetric(centered.T @ centered / (wrench.shape[0] - 1))
    empirical_std, empirical_correlation = _covariance_std_correlation(empirical)

    sg_per_time = np.asarray(
        [
            acceleration_to_wrench @ _symmetric(item) @ acceleration_to_wrench.T
            for item in sigma_z
        ],
        dtype=float,
    )
    sg_per_time = np.asarray([_symmetric(item) for item in sg_per_time])
    sg_mean = _symmetric(np.mean(sg_per_time, axis=0))
    sg_std, sg_correlation = _covariance_std_correlation(sg_mean)
    excess_raw = _symmetric(empirical - sg_mean)
    raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(excess_raw)
    raw_scale = max(
        float(np.linalg.norm(empirical, ord=2)),
        float(np.linalg.norm(sg_mean, ord=2)),
        float(np.max(np.abs(raw_eigenvalues))),
    )
    raw_negative_tolerance = 6.0 * np.finfo(float).eps * raw_scale
    appreciably_negative = raw_eigenvalues[
        raw_eigenvalues < -raw_negative_tolerance
    ]
    projected_eigenvalues = np.maximum(raw_eigenvalues, 0.0)
    model_discrepancy = _symmetric(
        (raw_eigenvectors * projected_eigenvalues) @ raw_eigenvectors.T
    )
    model_std, model_correlation = _covariance_std_correlation(
        model_discrepancy
    )
    acceleration_discrepancy = _symmetric(
        wrench_to_acceleration
        @ model_discrepancy
        @ wrench_to_acceleration.T
    )
    acceleration_eigenvalues = np.linalg.eigvalsh(acceleration_discrepancy)
    acceleration_std, acceleration_correlation = _covariance_std_correlation(
        acceleration_discrepancy
    )
    return ResidualWrenchUncertainty(
        mass_gauge_scale=scale,
        fixed_mass_kg=fixed_mass,
        modeled_wrench=_readonly(modeled_fixed),
        required_wrench=_readonly(required_fixed),
        wrench=_readonly(wrench),
        centered_wrench=_readonly(centered),
        mean=_readonly(mean),
        empirical_covariance=_readonly(empirical),
        empirical_std=empirical_std,
        empirical_correlation=empirical_correlation,
        sg_covariance_per_time=_readonly(sg_per_time),
        sg_covariance_mean=_readonly(sg_mean),
        sg_std=sg_std,
        sg_correlation=sg_correlation,
        excess_covariance_raw=_readonly(excess_raw),
        excess_covariance_raw_eigenvalues=_readonly(raw_eigenvalues),
        appreciably_negative_raw_eigenvalues=_readonly(appreciably_negative),
        raw_negative_eigenvalue_tolerance=raw_negative_tolerance,
        model_discrepancy_covariance=_readonly(model_discrepancy),
        model_discrepancy_eigenvalues=_readonly(projected_eigenvalues),
        model_discrepancy_std=model_std,
        model_discrepancy_correlation=model_correlation,
        acceleration_model_discrepancy_covariance=_readonly(
            acceleration_discrepancy
        ),
        acceleration_model_discrepancy_eigenvalues=_readonly(
            acceleration_eigenvalues
        ),
        acceleration_model_discrepancy_std=acceleration_std,
        acceleration_model_discrepancy_correlation=acceleration_correlation,
        acceleration_to_wrench=acceleration_to_wrench,
        wrench_to_acceleration=wrench_to_acceleration,
    )


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
    """Fully analytic Jacobian of :func:`generalized_acceleration_from_xi`."""

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
        spatial_alpha = directional_rho1 @ rho1 + left @ rho2
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
        spatial_alpha = rho2
        result[3:, 9:12] = rotation.T

    for column in range(3):
        zeta = np.eye(3)[column]
        spatial_directional = np.zeros(3)
        if geometric_correction:
            _, directional_zeta = left_jacobian_with_directional_derivative(
                rho0, zeta
            )
            _, _, second = (
                left_jacobian_with_directional_derivative(
                    rho0, rho1, zeta
                )
            )
            spatial_directional = second @ rho1 + directional_zeta @ rho2
        result[3:, 3 + column] = rotation.T @ (
            skew(spatial_alpha) @ left @ zeta + spatial_directional
        )
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
    wrench_corrected: np.ndarray
    sandwich_middle_wrench: np.ndarray
    sandwich_middle_total: np.ndarray


def parameter_covariances(
    raw_parameter_jacobian: np.ndarray,
    covariance: SgCovarianceEvaluation,
    gauge_direction: Sequence[float],
    additional_residual_covariance: Optional[np.ndarray] = None,
) -> ParameterCovarianceResult:
    """Compute SG and optional excess-wrench covariance on the gauge section."""

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
    additional = (
        np.zeros((6, 6), dtype=float)
        if additional_residual_covariance is None
        else _symmetric(np.asarray(additional_residual_covariance, dtype=float))
    )
    if additional.shape != (6, 6) or np.any(~np.isfinite(additional)):
        raise ValueError("additional residual covariance must be finite 6x6")
    additional_eigenvalues = np.linalg.eigvalsh(additional)
    additional_scale = (
        float(np.max(np.abs(additional_eigenvalues)))
        if additional_eigenvalues.size
        else 0.0
    )
    additional_tolerance = 6.0 * np.finfo(float).eps * additional_scale
    if np.any(additional_eigenvalues < -additional_tolerance):
        raise ValueError("additional residual covariance must be PSD")
    curvature = np.zeros((dimension, dimension), dtype=float)
    middle_wrench = np.zeros((dimension, dimension), dtype=float)
    for index in range(count):
        curvature += jacobian[index].T @ weights[index] @ jacobian[index]
        middle_wrench += (
            jacobian[index].T
            @ weights[index]
            @ additional
            @ weights[index]
            @ jacobian[index]
        )
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
    middle_total = _symmetric(middle + middle_wrench)
    reduced_total = _symmetric(basis.T @ middle_total @ basis)
    wrench_corrected = basis @ (
        reduced_inverse @ reduced_total @ reduced_inverse
    ) @ basis.T
    return ParameterCovarianceResult(
        curvature=_readonly(_symmetric(curvature)),
        sandwich_middle=_readonly(_symmetric(middle)),
        naive=_readonly(_symmetric(naive)),
        overlap_corrected=_readonly(_symmetric(overlap)),
        gauge_basis=_readonly(basis),
        wrench_corrected=_readonly(_symmetric(wrench_corrected)),
        sandwich_middle_wrench=_readonly(_symmetric(middle_wrench)),
        sandwich_middle_total=_readonly(middle_total),
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
