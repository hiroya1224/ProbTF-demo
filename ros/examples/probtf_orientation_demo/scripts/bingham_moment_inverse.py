"""Unbounded numerical inversion of the existing Bingham moments.

This example keeps the core probability law, normalizer, moments and gyro
product unchanged. Only the numerical inverse map differs from the core's
bounded log-gap optimizer. Finite iteration residuals are returned, never
used as a reason to reject the next filter update.
"""

import numpy as np

from probtf.bingham import (
    bingham_log_normalizer,
    bingham_second_moment,
    canonical_bingham_parameter,
    quaternion_product_second_moment,
    _moment_eigenvalues_and_jacobian,
)
from probtf_estimators.orientation_imu import delta_quaternion_second_moment


def _dual_value_gradient(coordinates, target_eigenvalues, scale, integration_steps):
    values = np.r_[np.asarray(coordinates) * scale, 0.0]
    if not np.isfinite(values).all():
        raise FloatingPointError('Bingham inverse produced non-finite eigenvalues.')
    parameter = np.diag(values)
    moments = np.diag(bingham_second_moment(parameter, integration_steps=integration_steps))
    value = bingham_log_normalizer(parameter, integration_steps=integration_steps)
    value -= float(values @ target_eigenvalues)
    gradient = scale * (moments[:3] - target_eigenvalues[:3])
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError('Bingham inverse dual or score is non-finite.')
    return float(value), gradient


def fit_bingham_second_moment(second_moment, integration_steps=120,
                              initial_eigenvalues=None, max_iterations=80):
    """Return a finite fitted parameter and diagnostics for the target moment.

    In the target eigenbasis solve ``moment(A) = eigenvalues(M)``. The last
    eigenvalue is fixed to zero to remove the scalar gauge. The other three
    eigenvalues are free real numbers, with no ordering, gap floor or upper
    concentration bound. ``lambda_i = scale_i * coordinate_i`` merely
    conditions this numerical problem; it does not change any target moment.

    The usual positive target eigenvalues give ``scale_i = 1/(2 m_i)``.
    For zero/nonpositive target eigenvalues, or an unrepresentable reciprocal,
    use unit coordinate scale. Such a boundary target can lack a finite
    inverse; the finite numerical result and its actual residual are still
    returned. The target eigenvalues themselves are never clipped or renormalized.
    """
    target = np.asarray(second_moment, dtype=float)
    if target.shape != (4, 4):
        raise ValueError('A quaternion second moment must have shape (4, 4).')
    if not np.isfinite(target).all():
        raise FloatingPointError('Bingham inverse requires a finite target moment.')
    iterations = int(max_iterations)
    if iterations < 1 or iterations != max_iterations:
        raise ValueError('max_iterations must be a positive integer.')
    # Only the symmetric part contributes to tr(A M) for symmetric A.
    symmetric_target = 0.5 * (target + target.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_target)
    scale = np.ones(3, dtype=float)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        reciprocal = 0.5 / eigenvalues[:3]
    scaled = (eigenvalues[:3] > 0.0) & np.isfinite(reciprocal)
    scale[scaled] = reciprocal[scaled]
    if initial_eigenvalues is None:
        # A Gaussian-concentration estimate is an initialization only.
        # Uniform targets start at exactly zero; no eigenvalue gap is added.
        reference_scale = 0.5 / eigenvalues[-1] if eigenvalues[-1] > 0.0 else 1.0
        initial = reference_scale - scale
    else:
        initial_values = np.asarray(initial_eigenvalues, dtype=float)
        if initial_values.shape != (4,) or not np.isfinite(initial_values).all():
            raise ValueError('initial_eigenvalues must contain four finite values.')
        initial_values = np.sort(initial_values)
        initial = (initial_values - initial_values[-1])[:3]
    coordinates = initial / scale
    denominator = np.where(eigenvalues[:3] > 0.0, eigenvalues[:3], 1.0)
    evaluations = 0

    def evaluate(point):
        nonlocal evaluations
        values = np.r_[point * scale, 0.0]
        if not np.isfinite(values).all():
            raise FloatingPointError('Bingham inverse produced non-finite eigenvalues.')
        moments, jacobian = _moment_eigenvalues_and_jacobian(values, integration_steps)
        evaluations += 1
        residual = (moments[:3] - eigenvalues[:3]) / denominator
        derivative = jacobian[:3, :3] * scale[None, :] / denominator[:, None]
        if not np.isfinite(moments).all() or not np.isfinite(residual).all():
            raise FloatingPointError('Bingham moment inverse returned non-finite moments.')
        return moments, residual, derivative

    moments, relative_residual, derivative = evaluate(coordinates)
    reason = 'maximum_iterations_finite_result_retained'
    updates = 0
    # This controls only the numerical inverse iteration. An unfinished finite
    # inverse remains usable and never rejects the following filter update.
    tolerance = 512.0 * np.finfo(float).eps
    for iteration in range(iterations):
        residual_norm = float(np.linalg.norm(relative_residual))
        if residual_norm <= tolerance:
            reason = 'moment_equations_at_floating_precision'
            break
        try:
            step = np.linalg.solve(derivative, -relative_residual)
        except np.linalg.LinAlgError:
            reason = 'singular_moment_jacobian_finite_result_retained'
            break
        if not np.isfinite(step).all():
            reason = 'nonfinite_newton_step_finite_result_retained'
            break
        accepted = False
        for exponent in range(24):
            trial = coordinates + (2.0 ** -exponent) * step
            if np.array_equal(trial, coordinates):
                continue
            try:
                trial_moments, trial_residual, trial_derivative = evaluate(trial)
            except (FloatingPointError, ValueError):
                continue
            # Compare the moment equations directly. The dual objective can
            # round to the same float even when a gyro noise increment matters.
            if np.linalg.norm(trial_residual) < residual_norm:
                coordinates, moments = trial, trial_moments
                relative_residual, derivative = trial_residual, trial_derivative
                updates += 1
                accepted = True
                break
        if not accepted:
            reason = 'moment_residual_no_longer_decreases_finite_result_retained'
            break
    if np.linalg.norm(relative_residual) <= tolerance:
        reason = 'moment_equations_at_floating_precision'
    fitted_values = np.r_[coordinates * scale, 0.0]
    assembled = eigenvectors @ np.diag(fitted_values) @ eigenvectors.T
    # A mathematically symmetric spectral product can have absolute roundoff
    # skew above the core input tolerance at million-scale concentration.
    parameter = canonical_bingham_parameter(0.5 * (assembled + assembled.T))
    fitted_moment = bingham_second_moment(parameter, integration_steps=integration_steps)
    if not np.isfinite(parameter).all() or not np.isfinite(fitted_moment).all():
        raise FloatingPointError('Bingham inverse returned a non-finite parameter or moment.')
    residual = float(np.linalg.norm(fitted_moment - target))
    fitted_diagonal = np.diag(eigenvectors.T @ fitted_moment @ eigenvectors)
    score = fitted_diagonal[:3] - eigenvalues[:3]
    positive = eigenvalues > 0.0
    log_ratios = [float(np.log(fitted_diagonal[i] / eigenvalues[i])) if positive[i] else None
                  for i in range(4)]
    diagnostics = dict(
        method='scaled_bingham_moment_equations_in_target_eigenbasis',
        optimizer='Newton with analytic moment Jacobian and residual backtracking',
        optimizer_success=reason == 'moment_equations_at_floating_precision',
        optimizer_status=0 if reason == 'moment_equations_at_floating_precision' else 1,
        optimizer_message=reason, iterations=updates,
        function_evaluations=evaluations, maximum_iterations=iterations,
        relative_moment_equation_residual=float(np.linalg.norm(relative_residual)),
        target_trace=float(np.trace(target)),
        target_skew_norm=float(np.linalg.norm(target - target.T)),
        target_moment_eigenvalues=eigenvalues.tolist(),
        fitted_moment_in_target_basis=fitted_diagonal.tolist(),
        initial_parameter_eigenvalues=np.r_[initial, 0.0].tolist(),
        fitted_parameter_eigenvalues=np.linalg.eigvalsh(parameter).tolist(),
        coordinate_scale=scale.tolist(),
        unit_coordinate_scale_indices=np.flatnonzero(~scaled).tolist(),
        moment_residual_frobenius=residual,
        log_moment_ratios=log_ratios,
        score_in_free_eigenvalues=score.tolist(),
        score_in_scaled_coordinates=(scale * score).tolist(),
        finite_iteration_result_used=True,
        target_moment_modified=False, concentration_bounds=None)
    return parameter, diagnostics


def propagate(parameter, gyro, dt, covariance, integration_steps=120):
    """Propagate the same core quaternion-product moment, then invert it."""
    if dt == 0.0 or (not np.any(gyro) and not np.any(covariance)):
        return canonical_bingham_parameter(parameter), 0.0
    current = bingham_second_moment(parameter, integration_steps=integration_steps)
    increment = delta_quaternion_second_moment(gyro, dt, covariance)
    target = quaternion_product_second_moment(current, increment)
    predicted, diagnostic = fit_bingham_second_moment(
        target, integration_steps=integration_steps,
        initial_eigenvalues=np.linalg.eigvalsh(parameter), max_iterations=80)
    return predicted, diagnostic['moment_residual_frobenius']
