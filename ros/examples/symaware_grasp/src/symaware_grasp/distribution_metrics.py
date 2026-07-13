"""Distribution distances and native-core Bingham fitting for the IK demo."""

import numpy as np

from probtf.bingham import (
    bingham_log_normalizer,
    bingham_mode,
    match_bingham_to_second_moment,
)


def _symmetrize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


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
        log_normalizer_a = bingham_log_normalizer(matrix_a, integration_steps)
    if log_normalizer_b is None:
        log_normalizer_b = bingham_log_normalizer(matrix_b, integration_steps)
    log_normalizer_mid = bingham_log_normalizer(
        0.5 * (matrix_a + matrix_b),
        integration_steps,
    )
    return float(max(0.5 * (log_normalizer_a + log_normalizer_b) - log_normalizer_mid, 0.0))


def concentrations_to_bingham_eigenvalues(concentrations):
    concentrations = np.asarray(concentrations, dtype=float).reshape(-1)
    if concentrations.shape[0] != 3:
        raise ValueError("Expected exactly three concentration values.")
    return np.array(
        [-abs(concentrations[0]), -abs(concentrations[1]), -abs(concentrations[2]), 0.0],
        dtype=float,
    )


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
    if quaternion_samples.ndim != 2 or quaternion_samples.shape[1] != 4:
        raise ValueError("quaternion_samples_wxyz must have shape (N, 4).")
    norms = np.linalg.norm(quaternion_samples, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("Quaternion samples must be finite and non-zero.")
    quaternion_samples = quaternion_samples / norms
    if weights is None:
        weights = np.full(quaternion_samples.shape[0], 1.0 / quaternion_samples.shape[0])
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape != (quaternion_samples.shape[0],):
            raise ValueError("weights must match the number of quaternion samples.")
        total_weight = float(np.sum(weights))
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or total_weight <= 0.0:
            raise ValueError("weights must be finite, non-negative, and have positive mass.")
        weights = weights / total_weight
    scatter_matrix = np.einsum("n,ni,nj->ij", weights, quaternion_samples, quaternion_samples)
    scatter_matrix = _symmetrize(scatter_matrix)
    if initial_eigenvalues is None:
        initial_eigenvalues = concentrations_to_bingham_eigenvalues(fallback_concentrations)
    fitted_matrix = match_bingham_to_second_moment(
        scatter_matrix,
        integration_steps=integration_steps,
        initial_eigenvalues=initial_eigenvalues,
        max_iterations=max_iterations,
    )
    mode_quaternion = bingham_mode(fitted_matrix)
    if reference_mode is not None and float(np.dot(mode_quaternion, reference_mode)) < 0.0:
        mode_quaternion *= -1.0
    eigenvalues, eigenvectors = np.linalg.eigh(fitted_matrix)
    return {
        "A": fitted_matrix,
        "Z": eigenvalues,
        "M": eigenvectors,
        "mode": mode_quaternion,
        "scatter": scatter_matrix,
        "log_normalizer": bingham_log_normalizer(fitted_matrix, integration_steps),
    }
