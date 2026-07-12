import numpy as np

import bingham.math.normconst as bingham_normconst


def ensure_trace_zero(param_mat):
    matrix = 0.5 * (np.asarray(param_mat, dtype=float) + np.asarray(param_mat, dtype=float).T)
    return matrix - np.trace(matrix) / 4.0 * np.eye(4, dtype=float)


def _eigendecompose_bingham(param_mat):
    eigenvalues, eigenvectors = np.linalg.eigh(ensure_trace_zero(param_mat))
    return eigenvalues, eigenvectors


def _shifted_eigenvalues(eigenvalues):
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return eigenvalues - float(np.max(eigenvalues))


def bingham_log_normalizer(lambda_vec, integration_steps=120):
    eigenvalues = np.asarray(lambda_vec, dtype=float)
    shifted = _shifted_eigenvalues(eigenvalues)
    constant = float(np.asarray(bingham_normconst.calc_constant(shifted, N=integration_steps)).reshape(-1)[0])
    return float(np.log(max(constant, 1e-300)) + np.max(eigenvalues))


def bingham_second_moment(param_mat, integration_steps=120):
    eigenvalues, eigenvectors = _eigendecompose_bingham(param_mat)
    shifted = _shifted_eigenvalues(eigenvalues)
    constant = float(np.asarray(bingham_normconst.calc_constant(shifted, N=integration_steps)).reshape(-1)[0])
    derivative = np.asarray(bingham_normconst.calc_Dconstant(shifted, N=integration_steps)).reshape(-1, 4)[0]
    moments_diag = derivative / max(constant, 1e-300)
    return 0.5 * (
        eigenvectors @ np.diag(moments_diag) @ eigenvectors.T
        + (eigenvectors @ np.diag(moments_diag) @ eigenvectors.T).T
    )


def bingham_fourth_moment(param_mat, integration_steps=120):
    eigenvalues, eigenvectors = _eigendecompose_bingham(param_mat)
    shifted = _shifted_eigenvalues(eigenvalues)
    constant = float(np.asarray(bingham_normconst.calc_constant(shifted, N=integration_steps)).reshape(-1)[0])
    second_derivative = np.asarray(
        bingham_normconst.calc_DDconstant(shifted, N=integration_steps, Hesse=True)
    ).reshape(-1, 4, 4)[0]
    diagonalized_tensor = np.zeros((4, 4, 4, 4), dtype=float)
    normalized_moments = second_derivative / max(constant, 1e-300)

    for index in range(4):
        diagonalized_tensor[index, index, index, index] = normalized_moments[index, index]

    for row in range(4):
        for col in range(row + 1, 4):
            value = normalized_moments[row, col]
            for indices in {
                (row, row, col, col),
                (row, col, row, col),
                (row, col, col, row),
                (col, row, row, col),
                (col, row, col, row),
                (col, col, row, row),
            }:
                diagonalized_tensor[indices] = value

    return np.einsum(
        "ai,bj,ck,dl,ijkl->abcd",
        eigenvectors,
        eigenvectors,
        eigenvectors,
        eigenvectors,
        diagonalized_tensor,
    )
