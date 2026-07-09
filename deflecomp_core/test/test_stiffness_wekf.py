import unittest

import numpy as np

from deflecomp_core.estimator.stiffness_wekf import MultiFrameStiffnessWEKF
from deflecomp_core.observation.bingham import BinghamUtils


class FakeRobot:
    def __init__(self, frame_quaternions, frame_jacobians):
        self.frame_quaternions = frame_quaternions
        self.frame_jacobians = frame_jacobians

    def frame_quaternion_wxyz_base(self, theta, fid):
        del theta
        return self.frame_quaternions[fid].copy()

    def frame_angular_jacobian_world(self, theta, fid):
        del theta
        return self.frame_jacobians[fid].copy()


class FakeSensitivity:
    def __init__(self, robot, J_q, J_x):
        self.robot = robot
        self.J_q = J_q
        self.J_x = J_x

    def equilibrium_jacobians(self, theta_eq, theta_cmd, kp_vec):
        del theta_eq, theta_cmd, kp_vec
        return self.J_q.copy(), self.J_x.copy()


class FakeSolver:
    def __init__(self, theta_eq):
        self.theta_eq = theta_eq
        self.solve_calls = 0

    def solve(self, theta_cmd, kp_vec, theta_init=None):
        del theta_cmd, kp_vec, theta_init
        self.solve_calls += 1
        return self.theta_eq.copy()


def make_estimator(
    z=None,
    J_w=None,
    J_q=None,
    J_x=None,
    x0=None,
    P0=None,
    Q=None,
    **kwargs,
):
    n = 3 if x0 is None else x0.size
    z = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if z is None else np.asarray(z, dtype=float)
    J_w = np.eye(n) if J_w is None else np.asarray(J_w, dtype=float)
    J_q = np.eye(n) if J_q is None else np.asarray(J_q, dtype=float)
    J_x = np.eye(n) if J_x is None else np.asarray(J_x, dtype=float)
    x0 = np.zeros(n, dtype=float) if x0 is None else np.asarray(x0, dtype=float)
    P0 = np.eye(n, dtype=float) if P0 is None else np.asarray(P0, dtype=float)
    Q = np.zeros((n, n), dtype=float) if Q is None else np.asarray(Q, dtype=float)
    theta_eq = np.zeros(J_q.shape[0], dtype=float)
    robot = FakeRobot({0: z}, {0: J_w})
    solver = FakeSolver(theta_eq)
    sensitivity = FakeSensitivity(robot, J_q, J_x)
    estimator = MultiFrameStiffnessWEKF(
        x0=x0,
        P0=P0,
        Q=Q,
        solver=solver,
        sensitivity=sensitivity,
        **kwargs,
    )
    return estimator, solver


class StiffnessWEKFLaplaceTests(unittest.TestCase):
    def test_local_laplace_terms_match_linear_model_finite_difference(self):
        z = np.array([0.91, 0.22, -0.28, 0.21], dtype=float)
        z = z / np.linalg.norm(z)
        J_w = np.array(
            [
                [0.7, -0.2, 0.1],
                [0.1, 0.5, -0.3],
                [-0.4, 0.2, 0.6],
            ],
            dtype=float,
        )
        J_q = np.array(
            [
                [1.8, 0.2, -0.1],
                [0.1, 1.5, 0.3],
                [-0.2, 0.1, 1.7],
            ],
            dtype=float,
        )
        J_x = np.diag([0.4, -0.7, 0.5])
        A = -np.array(
            [
                [0.4, -0.1, 0.2, 0.0],
                [-0.1, 1.2, 0.1, -0.2],
                [0.2, 0.1, 0.8, 0.3],
                [0.0, -0.2, 0.3, 1.5],
            ],
            dtype=float,
        )
        A = 0.5 * (A + A.T)
        estimator, _ = make_estimator(z=z, J_w=J_w, J_q=J_q, J_x=J_x)

        terms = estimator._compute_local_laplace_terms(
            theta_eq=np.zeros(3),
            J_q=J_q,
            J_x=J_x,
            A_map={0: A},
        )
        grad = terms["gradient"]
        info = terms["information"]

        theta_x = np.linalg.pinv(J_q, rcond=1e-12) @ J_x
        M = BinghamUtils.qmat_from_quat_wxyz(z) @ (J_w @ theta_x)

        def ell(delta_x):
            z_local = z - 0.5 * (M @ delta_x)
            return float(z_local.T @ (A @ z_local))

        direction = np.array([0.3, -0.4, 0.5], dtype=float)
        eps_grad = 1e-6
        eps_hess = 1e-3
        first_diff = (ell(eps_grad * direction) - ell(-eps_grad * direction)) / (2.0 * eps_grad)
        second_diff = (
            ell(eps_hess * direction) - 2.0 * ell(np.zeros(3)) + ell(-eps_hess * direction)
        ) / (eps_hess * eps_hess)

        self.assertTrue(np.allclose(info, info.T, atol=1e-12))
        self.assertTrue(np.isclose(first_diff, float(grad.T @ direction), rtol=1e-6, atol=1e-8))
        self.assertTrue(np.isclose(second_diff, -float(direction.T @ info @ direction), rtol=2e-4, atol=1e-8))

    def test_observable_subspace_clips_small_negative_information_eigenvalues(self):
        estimator, _ = make_estimator(observability_abs=1e-12)
        info = np.diag([-1e-12, 0.0, 2.0])

        eigvals, _, keep, raw_eigvals, min_raw_eig = estimator._observable_subspace(info)

        self.assertLess(min_raw_eig, 0.0)
        self.assertTrue(np.all(eigvals >= 0.0))
        self.assertLess(raw_eigvals[0], 0.0)
        self.assertEqual(np.count_nonzero(keep), 1)

    def test_zero_information_skips_mean_update_and_keeps_process_noise(self):
        Q = np.eye(3) * 0.25
        estimator, solver = make_estimator(Q=Q)
        x_before = estimator.x.copy()
        P_before = estimator.P.copy()
        theta_cmd = np.zeros(3)

        estimator.update_with_multi(theta_cmd, {0: np.zeros((4, 4))}, theta_init_eq_pred=None)

        self.assertEqual(solver.solve_calls, 1)
        self.assertTrue(np.allclose(estimator.x, x_before))
        self.assertTrue(np.allclose(estimator.P, P_before + Q))
        self.assertEqual(estimator.last_debug["laplace_update_skipped_reason"], "no_observable_information")

    def test_nonzero_gradient_with_zero_information_is_safely_skipped(self):
        A = np.zeros((4, 4), dtype=float)
        A[0, 1] = 1.0
        A[1, 0] = 1.0
        estimator, _ = make_estimator()
        theta_cmd = np.zeros(3)

        estimator.update_with_multi(theta_cmd, {0: A}, theta_init_eq_pred=None)

        self.assertGreater(estimator.last_debug["laplace_grad_norm"], 0.0)
        self.assertEqual(estimator.last_debug["laplace_update_skipped_reason"], "no_observable_information")
        self.assertTrue(np.allclose(estimator.x, np.zeros(3)))

    def test_zero_gradient_nonzero_information_only_shrinks_covariance(self):
        A = np.diag([0.0, -2.0, -3.0, -4.0])
        estimator, _ = make_estimator(P0=np.eye(3) * 2.0, laplace_jitter=0.0)
        theta_cmd = np.zeros(3)
        P_before = estimator.P.copy()

        estimator.update_with_multi(theta_cmd, {0: A}, theta_init_eq_pred=None)

        self.assertTrue(np.allclose(estimator.last_update_step, np.zeros(3), atol=1e-12))
        self.assertTrue(np.allclose(estimator.x, np.zeros(3), atol=1e-12))
        self.assertTrue(np.all(np.diag(estimator.P) < np.diag(P_before)))
        self.assertNotIn("laplace_update_skipped_reason", estimator.last_debug)

    def test_estimator_ignores_old_step_cap_but_keeps_kp_bounds(self):
        A = np.zeros((4, 4), dtype=float)
        A[0, 1] = -20.0
        A[1, 0] = -20.0
        A[1, 1] = -0.1
        estimator, _ = make_estimator(
            P0=np.eye(3) * 10.0,
            max_log_kp_step=1e-6,
            laplace_jitter=0.0,
        )
        theta_cmd = np.zeros(3)

        result = estimator.update_with_multi(theta_cmd, {0: A}, theta_init_eq_pred=None, kp_lim=(0.1, 500.0))

        self.assertTrue(result.update_applied)
        self.assertGreater(np.max(np.abs(estimator.last_update_step)), 1e-6)
        self.assertTrue(np.all(estimator.kp_hat <= 500.0 + 1e-12))
        self.assertTrue(np.all(estimator.kp_hat >= 0.1 - 1e-12))


if __name__ == "__main__":
    unittest.main()
