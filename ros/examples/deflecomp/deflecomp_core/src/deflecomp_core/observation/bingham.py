"""Utilities for IMU Bingham orientation likelihoods.

The Bingham matrix returned here is not a direct distribution over stiffness K.
For a predicted frame quaternion z_f(theta_eq; K), it defines the observation
log-likelihood term

    ell_f = z_f.T @ A_f @ z_f.

The stiffness estimator combines this likelihood with a Gaussian prior over
x = log K and forms a local Laplace approximation of the posterior.
"""

import numpy as np

from deflecomp_core.utils.linalg import normalize
from probtf.geometry import quat_left_matrix, quat_right_matrix


class BinghamUtils:
    @staticmethod
    def qmat_from_quat_wxyz(z: np.ndarray) -> np.ndarray:
        return quat_left_matrix(z, normalize_input=False)[:, 1:]

    @staticmethod
    def _lmat(q: np.ndarray) -> np.ndarray:
        return quat_left_matrix(q, normalize_input=False)

    @staticmethod
    def _rmat(q: np.ndarray) -> np.ndarray:
        return quat_right_matrix(q, normalize_input=False)

    @staticmethod
    def simple_bingham_unit(before_vec3: np.ndarray, after_vec3: np.ndarray, parameter: float = 100.0) -> np.ndarray:
        """Build A for the quaternion likelihood z.T @ A @ z between two unit vectors."""
        b = normalize(np.asarray(before_vec3, dtype=float))
        a = normalize(np.asarray(after_vec3, dtype=float))
        vq = np.array([0.0, b[0], b[1], b[2]], dtype=float)
        xq = np.array([0.0, a[0], a[1], a[2]], dtype=float)
        P = BinghamUtils._lmat(xq) - BinghamUtils._rmat(vq)
        A0 = -0.25 * (P.T @ P)
        return float(parameter) * A0


def simple_bingham_unit(before_vec3: np.ndarray, after_vec3: np.ndarray, parameter: float = 100.0) -> np.ndarray:
    return BinghamUtils.simple_bingham_unit(before_vec3, after_vec3, parameter=parameter)
