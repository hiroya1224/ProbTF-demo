import numpy as np

from probik_demo.prob_tf.bingham_moments import ensure_trace_zero
from probik_demo.prob_tf.geometry import (
    exp_s2,
    normalize_vec,
    quat_left_matrix,
    quat_to_rotmat,
    quat_normalize,
    tangent_basis,
    tangent_projector,
)


class TangentSurrogateResult:
    def __init__(self, mean, cov, lambda_mat, sigma_mat, mode, radius):
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.lambda_mat = np.asarray(lambda_mat, dtype=float)
        self.sigma_mat = np.asarray(sigma_mat, dtype=float)
        self.mode = np.asarray(mode, dtype=float)
        self.radius = float(radius)

    @staticmethod
    def _regularized_local_covariance(covariance):
        covariance = 0.5 * (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def sample(self, rng):
        if self.radius <= 1e-12:
            return np.zeros(3, dtype=float)
        basis = tangent_basis(self.mode)
        tangent_coordinates = rng.multivariate_normal(
            np.zeros(2, dtype=float),
            self._regularized_local_covariance(self.sigma_mat),
        )
        tangent_vector = basis @ tangent_coordinates
        return self.radius * exp_s2(self.mode, tangent_vector)


def _zero_surrogate():
    zero = np.zeros(3, dtype=float)
    return TangentSurrogateResult(
        mean=zero,
        cov=np.zeros((3, 3), dtype=float),
        lambda_mat=np.zeros((3, 3), dtype=float),
        sigma_mat=np.zeros((2, 2), dtype=float),
        mode=np.array([1.0, 0.0, 0.0], dtype=float),
        radius=0.0,
    )


def _mode_rotation_and_diagonal_frame(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    mode_quaternion = quat_normalize(eigenvectors[:, 0])
    if mode_quaternion[0] < 0.0:
        mode_quaternion *= -1.0
        eigenvectors[:, 0] *= -1.0
    tangent_vectors = eigenvectors[:, 1:4]
    # Small-angle expansion around the mode gives a local SO(3) precision
    # diag((lambda_0 - lambda_i) / 2). The non-polar surrogate expects
    # mu = (mu_1, mu_2, mu_3) with mu_3 allowed to be negative.
    local_precisions = 0.5 * np.maximum(eigenvalues[0] - eigenvalues[1:4], 1e-10)
    order_tangent = np.argsort(local_precisions)[::-1]
    local_precisions = local_precisions[order_tangent]
    tangent_vectors = tangent_vectors[:, order_tangent]

    left_matrix = quat_left_matrix(mode_quaternion)
    axis_candidates = []
    for tangent_vector in tangent_vectors.T:
        local_pure = 2.0 * (left_matrix.T @ tangent_vector)[1:4]
        axis_candidates.append(normalize_vec(local_pure))
    frame_guess = np.column_stack(axis_candidates)
    u_matrix, _, vh_matrix = np.linalg.svd(frame_guess)
    diagonal_frame = u_matrix @ vh_matrix
    if np.linalg.det(diagonal_frame) < 0.0:
        diagonal_frame[:, -1] *= -1.0

    # Paper definition:
    #   mu_1 = lambda_1 + lambda_2
    #   mu_2 = lambda_1 + lambda_3
    #   mu_3 = lambda_2 + lambda_3
    # for lambda_1 >= lambda_2 >= lambda_3 >= lambda_4 and trace(A)=0.
    del local_precisions
    mu = np.array(
        [
            eigenvalues[0] + eigenvalues[1],
            eigenvalues[0] + eigenvalues[2],
            eigenvalues[1] + eigenvalues[2],
        ],
        dtype=float,
    )
    return quat_to_rotmat(mode_quaternion), diagonal_frame, mu


def induced_vector_moments_tangent(r, param_mat, use_jacobian_correction=True):
    vector = np.asarray(r, dtype=float)
    radius = float(np.linalg.norm(vector))
    if radius <= 1e-12:
        return _zero_surrogate()

    matrix = ensure_trace_zero(param_mat)
    unit_vector = normalize_vec(vector)
    mode_rotation, diagonal_frame, mu = _mode_rotation_and_diagonal_frame(matrix)
    v0 = diagonal_frame.T @ unit_vector
    d_matrix = np.diag([mu[0], mu[1], -mu[2]])
    d_mu = float(np.trace(d_matrix) - v0.T @ d_matrix @ v0)
    projector = tangent_projector(v0)

    polar_tol = 1e-10
    is_polar = (
        abs(d_mu) <= polar_tol
        and abs(mu[1] - mu[2]) <= polar_tol
        and abs(abs(v0[0]) - 1.0) <= 1e-8
        and abs(v0[1]) <= 1e-8
        and abs(v0[2]) <= 1e-8
    )
    if is_polar:
        lambda_tan = 0.5 * (mu[0] - mu[1]) * projector
    else:
        if not np.isfinite(d_mu) or d_mu <= 1e-12:
            raise ValueError("Invalid non-polar tangent surrogate configuration.")
        b_matrix = np.diag(
            [
                (mu[0] + mu[1]) * (mu[0] - mu[2]),
                (mu[0] + mu[1]) * (mu[1] - mu[2]),
                (mu[0] - mu[2]) * (mu[1] - mu[2]),
            ]
        )
        lambda_tan = (projector @ b_matrix @ projector) / (2.0 * d_mu)

    lambda_loc = lambda_tan + (1.0 / 3.0) * projector if use_jacobian_correction else lambda_tan
    lambda_loc = 0.5 * (lambda_loc + lambda_loc.T)
    sigma_ambient = np.linalg.pinv(lambda_loc, rcond=1e-10)

    sigma_trace = float(np.trace(sigma_ambient))
    sigma_square = sigma_ambient @ sigma_ambient
    mean0 = (1.0 - 0.5 * sigma_trace) * v0
    cov0 = (
        0.5 * float(np.trace(sigma_square)) * np.outer(v0, v0)
        + sigma_ambient
        - (1.0 / 3.0) * (sigma_trace * sigma_ambient + 2.0 * sigma_square)
    )
    transform = mode_rotation @ diagonal_frame
    mean_unit = transform @ mean0
    cov_unit = 0.5 * (transform @ cov0 @ transform.T + (transform @ cov0 @ transform.T).T)
    mode = normalize_vec(transform @ v0)
    basis = tangent_basis(mode)
    sigma_world = 0.5 * (transform @ sigma_ambient @ transform.T + (transform @ sigma_ambient @ transform.T).T)
    sigma_mat = 0.5 * (basis.T @ sigma_world @ basis + (basis.T @ sigma_world @ basis).T)
    lambda_world = 0.5 * (transform @ lambda_loc @ transform.T + (transform @ lambda_loc @ transform.T).T)
    return TangentSurrogateResult(
        mean=radius * mean_unit,
        cov=(radius**2) * cov_unit,
        lambda_mat=lambda_world,
        sigma_mat=sigma_mat,
        mode=mode,
        radius=radius,
    )
