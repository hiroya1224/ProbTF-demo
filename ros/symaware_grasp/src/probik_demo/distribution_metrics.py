import os
import sys

import numpy as np


DEFAULT_BINGHAM_SOURCE_DIR = os.environ.get("BINGHAM_SOURCE_DIR", "/home/leus/BinghamNLL/src")
if os.path.isdir(DEFAULT_BINGHAM_SOURCE_DIR) and DEFAULT_BINGHAM_SOURCE_DIR not in sys.path:
    sys.path.insert(0, DEFAULT_BINGHAM_SOURCE_DIR)

import bingham.math.normconst as bingham_normconst

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - runtime fallback
    minimize = None


def _symmetrize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _safe_log(value, floor=1e-300):
    return float(np.log(max(float(value), floor)))


def gaussian_bhattacharyya_distance(
    mean_a,
    covariance_a,
    mean_b,
    covariance_b,
    regularization=1e-6,
):
    mean_a = np.asarray(mean_a, dtype=float)
    mean_b = np.asarray(mean_b, dtype=float)
    covariance_a = _symmetrize(covariance_a) + regularization * np.eye(3, dtype=float)
    covariance_b = _symmetrize(covariance_b) + regularization * np.eye(3, dtype=float)
    covariance_mid = 0.5 * (covariance_a + covariance_b)

    delta = mean_a - mean_b
    quadratic_term = 0.125 * float(delta.T @ np.linalg.inv(covariance_mid) @ delta)

    sign_a, logdet_a = np.linalg.slogdet(covariance_a)
    sign_b, logdet_b = np.linalg.slogdet(covariance_b)
    sign_mid, logdet_mid = np.linalg.slogdet(covariance_mid)
    if sign_a <= 0 or sign_b <= 0 or sign_mid <= 0:
        raise ValueError("Gaussian Bhattacharyya distance requires positive definite covariances.")

    determinant_term = 0.5 * (logdet_mid - 0.5 * (logdet_a + logdet_b))
    return float(max(quadratic_term + determinant_term, 0.0))


def _log_normalizer_and_moments_from_eigenvalues(eigenvalues, integration_steps=120, with_jacobian=False):
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    shifted_eigenvalues = eigenvalues - float(np.max(eigenvalues))

    normalizing_constant = float(
        np.asarray(bingham_normconst.calc_constant(shifted_eigenvalues, N=integration_steps)).reshape(-1)[0]
    )
    log_normalizer = _safe_log(normalizing_constant) + float(np.max(eigenvalues))

    derivative = np.asarray(bingham_normconst.calc_Dconstant(shifted_eigenvalues, N=integration_steps)).reshape(-1, 4)[0]
    moments = derivative / normalizing_constant

    if not with_jacobian:
        return log_normalizer, moments

    second_derivative = np.asarray(
        bingham_normconst.calc_DDconstant(shifted_eigenvalues, N=integration_steps, Hesse=True)
    ).reshape(-1, 4, 4)[0]
    jacobian = second_derivative / normalizing_constant - np.outer(derivative, derivative) / (
        normalizing_constant**2
    )
    return log_normalizer, moments, jacobian


def bingham_log_normalizer_from_A(matrix_a, integration_steps=120):
    eigenvalues = np.linalg.eigvalsh(_symmetrize(matrix_a))
    log_normalizer, _ = _log_normalizer_and_moments_from_eigenvalues(
        eigenvalues,
        integration_steps=integration_steps,
        with_jacobian=False,
    )
    return float(log_normalizer)


def bingham_bhattacharyya_distance(
    matrix_a,
    matrix_b,
    integration_steps=120,
    log_normalizer_a=None,
    log_normalizer_b=None,
):
    matrix_a = _symmetrize(matrix_a)
    matrix_b = _symmetrize(matrix_b)
    if log_normalizer_a is None:
        log_normalizer_a = bingham_log_normalizer_from_A(matrix_a, integration_steps=integration_steps)
    if log_normalizer_b is None:
        log_normalizer_b = bingham_log_normalizer_from_A(matrix_b, integration_steps=integration_steps)
    log_normalizer_mid = bingham_log_normalizer_from_A(
        0.5 * (matrix_a + matrix_b),
        integration_steps=integration_steps,
    )
    return float(max(0.5 * (log_normalizer_a + log_normalizer_b) - log_normalizer_mid, 0.0))


def concentrations_to_bingham_eigenvalues(concentrations):
    concentrations = np.asarray(concentrations, dtype=float).reshape(-1)
    if concentrations.shape[0] != 3:
        raise ValueError("Expected exactly three concentration values.")
    return np.array([-abs(concentrations[0]), -abs(concentrations[1]), -abs(concentrations[2]), 0.0], dtype=float)


def _eigenvalues_to_parameter_vector(eigenvalues):
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    delta_1 = max(eigenvalues[1] - eigenvalues[0], 1e-6)
    delta_2 = max(eigenvalues[2] - eigenvalues[1], 1e-6)
    delta_3 = max(-eigenvalues[2], 1e-6)
    return np.log(np.array([delta_1, delta_2, delta_3], dtype=float))


def _parameter_vector_to_eigenvalues(parameters):
    deltas = np.exp(np.asarray(parameters, dtype=float))
    eigenvalue_3 = -deltas[2]
    eigenvalue_2 = eigenvalue_3 - deltas[1]
    eigenvalue_1 = eigenvalue_2 - deltas[0]
    eigenvalues = np.array([eigenvalue_1, eigenvalue_2, eigenvalue_3, 0.0], dtype=float)
    return eigenvalues, deltas


def fit_bingham_from_quaternion_samples(
    quaternion_samples_wxyz,
    weights=None,
    initial_eigenvalues=None,
    integration_steps=80,
    max_iterations=40,
    fallback_concentrations=(420.0, 320.0, 220.0),
    reference_mode=None,
):
    quaternion_samples = np.asarray(quaternion_samples_wxyz, dtype=float)
    quaternion_samples = quaternion_samples / np.linalg.norm(quaternion_samples, axis=1, keepdims=True)

    if weights is None:
        weights = np.full(quaternion_samples.shape[0], 1.0 / quaternion_samples.shape[0], dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        weights = weights / np.sum(weights)

    scatter_matrix = np.einsum("n,ni,nj->ij", weights, quaternion_samples, quaternion_samples)
    scatter_matrix = _symmetrize(scatter_matrix)
    sample_moments, sample_eigenvectors = np.linalg.eigh(scatter_matrix)
    sample_moments = np.clip(sample_moments, 1e-6, 1.0)
    sample_moments /= np.sum(sample_moments)

    if initial_eigenvalues is None:
        initial_eigenvalues = concentrations_to_bingham_eigenvalues(fallback_concentrations)

    if minimize is None:
        fitted_eigenvalues = np.asarray(initial_eigenvalues, dtype=float)
    else:
        initial_parameters = _eigenvalues_to_parameter_vector(initial_eigenvalues)

        def objective(parameters):
            candidate_eigenvalues, deltas = _parameter_vector_to_eigenvalues(parameters)
            _, candidate_moments, jacobian = _log_normalizer_and_moments_from_eigenvalues(
                candidate_eigenvalues,
                integration_steps=integration_steps,
                with_jacobian=True,
            )
            error = candidate_moments - sample_moments
            loss = 0.5 * float(error @ error)

            loss_gradient_wrt_eigenvalues = jacobian.T @ error
            dz_dp = np.array(
                [
                    [-deltas[0], -deltas[1], -deltas[2]],
                    [0.0, -deltas[1], -deltas[2]],
                    [0.0, 0.0, -deltas[2]],
                    [0.0, 0.0, 0.0],
                ],
                dtype=float,
            )
            gradient = dz_dp.T @ loss_gradient_wrt_eigenvalues
            return loss, gradient

        optimization = minimize(
            lambda parameters: objective(parameters)[0],
            initial_parameters,
            method="L-BFGS-B",
            jac=lambda parameters: objective(parameters)[1],
            options={"maxiter": int(max_iterations)},
        )
        if optimization.success and np.all(np.isfinite(optimization.x)):
            fitted_eigenvalues, _ = _parameter_vector_to_eigenvalues(optimization.x)
        else:
            fitted_eigenvalues = np.asarray(initial_eigenvalues, dtype=float)

    fitted_matrix = sample_eigenvectors @ np.diag(fitted_eigenvalues) @ sample_eigenvectors.T
    mode_quaternion = sample_eigenvectors[:, -1].copy()
    if reference_mode is not None and float(np.dot(mode_quaternion, reference_mode)) < 0.0:
        mode_quaternion *= -1.0

    log_normalizer = bingham_log_normalizer_from_A(fitted_matrix, integration_steps=integration_steps)
    return {
        "A": _symmetrize(fitted_matrix),
        "Z": fitted_eigenvalues,
        "M": sample_eigenvectors,
        "mode": mode_quaternion,
        "scatter": scatter_matrix,
        "log_normalizer": log_normalizer,
    }
