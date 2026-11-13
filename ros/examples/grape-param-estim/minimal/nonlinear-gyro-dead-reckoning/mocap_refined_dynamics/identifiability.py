"""Read-only local geometry audit of the completed rotational least squares.

The log-inertia chart can approach a physical inertia boundary while the
inertia tensor itself stays positive definite.  A small chart derivative
there must not be mistaken for an additional unobservable physical mode.
These diagnostics never accept/reject a fit or modify the propagated law.
"""
from __future__ import annotations

import numpy as np


def _tracefree_basis():
    basis = [np.diag([1., -1., 0.]) / np.sqrt(2.),
             np.diag([1., 1., -2.]) / np.sqrt(6.)]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        matrix = np.zeros((3, 3))
        matrix[i, j] = matrix[j, i] = 1. / np.sqrt(2.)
        basis.append(matrix)
    return np.asarray(basis)


def _spectrum(matrix):
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    tolerance = np.finfo(float).eps * max(matrix.shape) * singular[0]
    return singular, right, tolerance, int(np.count_nonzero(singular > tolerance))


def audit_fit(data, kinematics, fit_result):
    """Return JSON metadata and numeric arrays, using the training prefix only.

    The direct physical Jacobian uses five trace-free symmetric inertia
    directions (kg m^2), three CoG coordinates (m), four log-effectiveness
    coordinates and log(tau).  Its singular spectrum has mixed units: it is
    useful for separating chart degeneration from local physical sensitivity,
    not for a unit-independent condition-number or global-identifiability
    claim.  No Gaussian covariance is extrapolated through the boundary.
    """
    # The runner adds this experiment's directory to sys.path.  Keep the
    # existing actuator implementation as the only wrench implementation.
    from fit import actuator_at

    arrays = fit_result['arrays']
    inertia = np.asarray(fit_result['point_parameters']['inertia'], float)
    physical_jacobian = np.asarray(arrays['physical_parameter_jacobian'], float)
    chart_jacobian = np.asarray(arrays['weighted_residual_jacobian'], float)
    weights = np.asarray(arrays['training_time_weights'], float)
    omega = np.asarray(kinematics['omega_body_rad_s'], float)
    alpha = np.asarray(kinematics['alpha_body_rad_s2'], float)
    time = np.asarray(kinematics['time'], float)
    second = .5 * np.trace(inertia) * np.eye(3) - inertia
    inertia_eigenvalues, inertia_axes = np.linalg.eigh(inertia)
    second_eigenvalues, second_axes = np.linalg.eigh(second)

    physical_residual_jacobian = chart_jacobian.copy()
    basis = _tracefree_basis()
    for column, direction in enumerate(basis):
        derivative = alpha @ direction.T + np.cross(omega, omega @ direction.T)
        physical_residual_jacobian[:, column] = (
            derivative * np.sqrt(weights)[:, None]).ravel()
    chart_singular, chart_right, chart_tolerance, chart_rank = _spectrum(chart_jacobian)
    physical_singular, physical_right, physical_tolerance, physical_rank = _spectrum(
        physical_residual_jacobian)
    decode_singular, decode_right, _, _ = _spectrum(physical_jacobian)
    inertia_decode_singular = np.linalg.svd(physical_jacobian[:6, :5], compute_uv=False)
    alignment = float(abs(chart_right[-1] @ decode_right[-1]))
    wrench, _ = actuator_at(data, fit_result, time)
    mean_force = weights @ wrench[:, :3]
    gyroscopic = np.cross(omega, omega @ inertia.T)
    inertial = alpha @ inertia.T
    rms_gyroscopic = np.sqrt(weights @ (gyroscopic * gyroscopic))
    rms_inertial = np.sqrt(weights @ (inertial * inertial))

    numeric = dict(
        inertia_eigenvalues_kg_m2=inertia_eigenvalues,
        inertia_eigenvectors=inertia_axes,
        physical_second_moment_kg_m2=second,
        physical_second_moment_eigenvalues_kg_m2=second_eigenvalues,
        physical_second_moment_eigenvectors=second_axes,
        inertia_chart_physical_jacobian_singular_values=inertia_decode_singular,
        physical_decode_jacobian_singular_values=decode_singular,
        physical_decode_weakest_coordinate_direction=decode_right[-1],
        residual_chart_weakest_coordinate_direction=chart_right[-1],
        direct_physical_tracefree_inertia_basis=basis,
        direct_physical_residual_jacobian=physical_residual_jacobian,
        direct_physical_residual_jacobian_singular_values=physical_singular,
        direct_physical_residual_jacobian_right_singular_vectors=physical_right,
        training_time=time.copy(), training_actuator_force_body_n=wrench[:, :3],
        training_mean_actuator_force_body_n=mean_force,
        training_gyroscopic_torque_rms_nm=rms_gyroscopic,
        training_inertial_torque_rms_nm=rms_inertial)
    metadata = dict(
        schema='grape/mocap-refined-rotational-identifiability-audit/v1',
        role='read-only training diagnostics; no fit or prediction acceptance gate',
        inertia_eigenvalues_kg_m2=inertia_eigenvalues.tolist(),
        physical_second_moment_eigenvalues_kg_m2=second_eigenvalues.tolist(),
        physical_second_moment_min_over_trace=float(second_eigenvalues[0] / np.trace(second)),
        inertia_triangle_margin_kg_m2=float(inertia_eigenvalues[:2].sum() - inertia_eigenvalues[2]),
        boundary_interpretation=(
            'C=tr(J)/2 I-J is the physical second moment. A small positive C eigenvalue '
            'means the unconstrained log chart approaches the planar-body boundary; it is '
            'not set to zero. Vanishing sensitivity of that chart is distinct from '
            'an additional structurally unobservable physical parameter.'),
        inertia_chart_physical_jacobian_singular_values=inertia_decode_singular.tolist(),
        full_physical_decode_jacobian_singular_values=decode_singular.tolist(),
        weakest_chart_residual_vs_decode_direction_abs_cosine=alignment,
        residual_chart_jacobian_rank=chart_rank,
        residual_chart_jacobian_singular_values=chart_singular.tolist(),
        residual_chart_numerical_rank_tolerance=float(chart_tolerance),
        direct_physical_residual_jacobian_rank=physical_rank,
        direct_physical_residual_jacobian_singular_values=physical_singular.tolist(),
        direct_physical_numerical_rank_tolerance=float(physical_tolerance),
        direct_physical_coordinates=(
            'five Frobenius-orthonormal trace-free symmetric inertia perturbations in kg m^2; '
            'CoG xyz in m; four log force effectiveness coordinates; log thrust tau'),
        direct_physical_spectrum_caution=(
            'Mixed coordinate units; not a unit-independent condition-number. Full local '
            'rank does not establish global identifiability or calibrated uncertainty.'),
        covariance_caution=(
            'Log-chart Gauss-Newton support covariance is only local curvature. Near a '
            'physical inertia boundary it is not a reliable Gaussian physical posterior; '
            'a discarded numerical chart direction must not be interpreted as certain. '
            'No parameter covariance is used for the angular-rate comparison bands.'),
        training_mean_actuator_force_body_n=mean_force.tolist(),
        nominal_mass_weight_n=float(fit_result['model'].parameters.mass * 9.80665),
        force_scope=(
            'Rotational equation only: translational force balance and mass are not fitted. '
            'Effectiveness is an effective rotational calibration in the fixed nominal '
            'trace(J) scale gauge, not independently recovered thrust calibration.'),
        training_gyroscopic_torque_rms_nm=rms_gyroscopic.tolist(),
        training_inertial_torque_rms_nm=rms_inertial.tolist(),
        heldout_or_future_inputs_used=False,
        state_or_covariance_modified=False)
    return dict(metadata=metadata, arrays=numeric)
