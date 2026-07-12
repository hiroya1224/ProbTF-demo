import os
import sys

import numpy as np

from probik_demo.prob_tf.bingham_moments import ensure_trace_zero


DEFAULT_BINGHAM_SOURCE_DIR = os.environ.get("BINGHAM_SOURCE_DIR", "/home/leus/BinghamNLL/src")
if os.path.isdir(DEFAULT_BINGHAM_SOURCE_DIR) and DEFAULT_BINGHAM_SOURCE_DIR not in sys.path:
    sys.path.insert(0, DEFAULT_BINGHAM_SOURCE_DIR)

import bingham.math.normconst as bingham_normconst

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - runtime fallback
    minimize = None


def _parameter_vector_to_eigenvalues(parameters):
    deltas = np.exp(np.asarray(parameters, dtype=float))
    eigenvalue_3 = -deltas[2]
    eigenvalue_2 = eigenvalue_3 - deltas[1]
    eigenvalue_1 = eigenvalue_2 - deltas[0]
    return np.array([eigenvalue_1, eigenvalue_2, eigenvalue_3, 0.0], dtype=float), deltas


def _eigenvalues_to_parameter_vector(eigenvalues):
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    delta_1 = max(eigenvalues[1] - eigenvalues[0], 1e-6)
    delta_2 = max(eigenvalues[2] - eigenvalues[1], 1e-6)
    delta_3 = max(-eigenvalues[2], 1e-6)
    return np.log(np.array([delta_1, delta_2, delta_3], dtype=float))


def _log_moments_and_jacobian(eigenvalues, integration_steps=120):
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    shifted = eigenvalues - float(np.max(eigenvalues))
    constant = float(np.asarray(bingham_normconst.calc_constant(shifted, N=integration_steps)).reshape(-1)[0])
    derivative = np.asarray(bingham_normconst.calc_Dconstant(shifted, N=integration_steps)).reshape(-1, 4)[0]
    second_derivative = np.asarray(
        bingham_normconst.calc_DDconstant(shifted, N=integration_steps, Hesse=True)
    ).reshape(-1, 4, 4)[0]
    moments = derivative / max(constant, 1e-300)
    jacobian = second_derivative / max(constant, 1e-300) - np.outer(derivative, derivative) / max(
        constant * constant,
        1e-300,
    )
    return moments, jacobian


def quaternion_product_second_moment(second_moment_a, second_moment_b):
    moment_a = np.asarray(second_moment_a, dtype=float).reshape(4, 4)
    moment_b = np.asarray(second_moment_b, dtype=float).reshape(4, 4)

    bilinear_forms = [
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
            ],
            dtype=float,
        ),
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=float,
        ),
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    ]

    output = np.zeros((4, 4), dtype=float)
    for row in range(4):
        for col in range(4):
            output[row, col] = float(
                np.einsum(
                    "ik,jl,ij,kl->",
                    moment_a,
                    moment_b,
                    bilinear_forms[row],
                    bilinear_forms[col],
                )
            )
    return 0.5 * (output + output.T)


def match_bingham_to_second_moment(
    second_moment,
    integration_steps=120,
    initial_eigenvalues=None,
    max_iterations=40,
):
    covariance = 0.5 * (np.asarray(second_moment, dtype=float) + np.asarray(second_moment, dtype=float).T)
    moment_eigenvalues, moment_eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(moment_eigenvalues)
    moment_eigenvalues = moment_eigenvalues[order]
    moment_eigenvectors = moment_eigenvectors[:, order]

    if initial_eigenvalues is None:
        initial_eigenvalues = np.array([-30.0, -15.0, -5.0, 0.0], dtype=float)

    if minimize is None:
        fitted_eigenvalues = np.asarray(initial_eigenvalues, dtype=float)
    else:
        initial_parameters = _eigenvalues_to_parameter_vector(initial_eigenvalues)

        def objective(parameters):
            candidate_eigenvalues, deltas = _parameter_vector_to_eigenvalues(parameters)
            candidate_moments, jacobian = _log_moments_and_jacobian(
                candidate_eigenvalues,
                integration_steps=integration_steps,
            )
            error = candidate_moments - moment_eigenvalues
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
            jac=lambda parameters: objective(parameters)[1],
            method="L-BFGS-B",
            options={"maxiter": int(max_iterations)},
        )
        if optimization.success and np.all(np.isfinite(optimization.x)):
            fitted_eigenvalues, _ = _parameter_vector_to_eigenvalues(optimization.x)
        else:
            fitted_eigenvalues = np.asarray(initial_eigenvalues, dtype=float)

    parameter_matrix = moment_eigenvectors @ np.diag(fitted_eigenvalues) @ moment_eigenvectors.T
    return ensure_trace_zero(parameter_matrix)
