"""Compatibility exports for shared ProbTF quaternion moments."""

import numpy as np

import bingham.math.normconst as bingham_normconst

from probtf.bingham import bingham_fourth_moment, bingham_second_moment


def ensure_trace_zero(parameter_matrix):
    matrix = np.asarray(parameter_matrix, dtype=float).reshape(4, 4)
    matrix = 0.5 * (matrix + matrix.T)
    return matrix - np.trace(matrix) * 0.25 * np.eye(4, dtype=float)


def bingham_log_normalizer(eigenvalues, integration_steps=120):
    eigenvalues = np.asarray(eigenvalues, dtype=float).reshape(4)
    maximum = float(np.max(eigenvalues))
    shifted = eigenvalues - maximum
    constant = float(
        np.asarray(
            bingham_normconst.calc_constant(shifted, N=int(integration_steps)),
            dtype=float,
        ).reshape(-1)[0]
    )
    return float(np.log(max(constant, 1e-300)) + maximum)


__all__ = [
    "bingham_fourth_moment",
    "bingham_log_normalizer",
    "bingham_second_moment",
    "ensure_trace_zero",
]
