import types
import unittest

import numpy as np

from deflecomp_core.estimator.stiffness_particle_supervisor import (
    StiffnessParticleScanConfig,
    StiffnessParticleScanSupervisor,
)
from deflecomp_core.estimator.stiffness_wekf import MultiFrameStiffnessWEKF


class FakeRobot:
    def frame_quaternion_wxyz_base(self, theta, fid):
        del theta, fid
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def frame_angular_jacobian_world(self, theta, fid):
        del theta, fid
        return np.eye(3, dtype=float)


class FakeSensitivity:
    def __init__(self, robot):
        self.robot = robot

    def equilibrium_jacobians(self, theta_eq, theta_cmd, kp_vec):
        del theta_eq, theta_cmd, kp_vec
        return np.eye(3, dtype=float), np.eye(3, dtype=float)


class FakeSolver:
    def solve(self, theta_cmd, kp_vec, theta_init=None):
        del kp_vec, theta_init
        return np.asarray(theta_cmd, dtype=float).copy()


def make_wekf():
    robot = FakeRobot()
    return MultiFrameStiffnessWEKF(
        x0=np.log(np.array([10.0, 20.0, 30.0], dtype=float)),
        P0=np.eye(3, dtype=float) * 0.2,
        Q=np.zeros((3, 3), dtype=float),
        solver=FakeSolver(),
        sensitivity=FakeSensitivity(robot),
    )


class FakeScanEstimator:
    def __init__(self, x_current, x_target, score_scale=10.0):
        self.x_est = np.asarray(x_current, dtype=float).copy()
        self.P_est = np.diag([0.01, 0.01, 1.0])
        self.x_target = np.asarray(x_target, dtype=float).copy()
        self.score_scale = float(score_scale)

    def evaluate_log_likelihood_at_x(
        self,
        x_eval,
        theta_cmd_sent,
        A_map,
        theta_init_eq_pred,
        kp_lim=None,
    ):
        del theta_cmd_sent, A_map, theta_init_eq_pred
        x = np.asarray(x_eval, dtype=float).copy()
        if kp_lim is not None:
            x = np.clip(x, np.log(float(kp_lim[0])), np.log(float(kp_lim[1])))
        score = -self.score_scale * float(np.sum((x - self.x_target) ** 2))
        return types.SimpleNamespace(
            valid=True,
            log_likelihood=score,
            theta_eq=np.array([score], dtype=float),
            error=None,
        )


class StiffnessParticleSupervisorTests(unittest.TestCase):
    def test_likelihood_evaluation_does_not_change_estimator_state(self):
        estimator = make_wekf()
        x_before = estimator.x_est.copy()
        P_before = estimator.P_est.copy()
        estimator.last_theta_eq = np.array([0.2, 0.3, 0.4], dtype=float)
        last_theta_before = estimator.last_theta_eq.copy()

        evaluation = estimator.evaluate_log_likelihood_at_x(
            x_eval=np.log(np.array([15.0, 25.0, 35.0], dtype=float)),
            theta_cmd_sent=np.array([0.1, -0.2, 0.3], dtype=float),
            A_map={0: -np.eye(4, dtype=float)},
            theta_init_eq_pred=np.zeros(3, dtype=float),
            kp_lim=(1.0, 100.0),
        )

        self.assertTrue(evaluation.valid)
        self.assertTrue(np.isfinite(evaluation.log_likelihood))
        self.assertTrue(np.allclose(estimator.x_est, x_before))
        self.assertTrue(np.allclose(estimator.P_est, P_before))
        self.assertTrue(np.allclose(estimator.last_theta_eq, last_theta_before))

    def test_axis_candidates_include_current_and_stay_inside_bounds(self):
        supervisor = StiffnessParticleScanSupervisor(
            StiffnessParticleScanConfig(grid_size=5)
        )
        x_current = np.log(np.array([10.0, 20.0, 30.0], dtype=float))

        candidates = supervisor._make_axis_candidates(
            x_current=x_current,
            active_indices=np.array([0, 2]),
            kp_lim=(1.0, 100.0),
        )

        self.assertLessEqual(len(candidates), 1 + 2 * 5)
        self.assertTrue(any(np.allclose(candidate, x_current) for candidate in candidates))
        for candidate in candidates:
            self.assertTrue(np.all(candidate >= np.log(1.0) - 1e-12))
            self.assertTrue(np.all(candidate <= np.log(100.0) + 1e-12))

    def test_scan_accepts_clear_map_improvement(self):
        x_current = np.log(np.array([10.0, 10.0, 10.0], dtype=float))
        x_target = np.log(np.array([100.0, 10.0, 10.0], dtype=float))
        estimator = FakeScanEstimator(x_current=x_current, x_target=x_target)
        supervisor = StiffnessParticleScanSupervisor(
            StiffnessParticleScanConfig(
                enabled=True,
                window_size=3,
                grid_size=5,
            )
        )
        for stamp in range(3):
            supervisor.add_record(np.zeros(3), {0: np.eye(4)}, None, float(stamp))

        result = supervisor.maybe_scan(
            estimator=estimator,
            kp_lim=(1.0, 100.0),
        )

        self.assertTrue(result.attempted)
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "accepted")
        self.assertTrue(np.allclose(result.x_best, x_target))
        self.assertGreater(result.gain_per_obs, 0.1)
        self.assertIsNotNone(result.theta_eq_best)

    def test_scan_rejects_when_score_does_not_improve(self):
        x_current = np.log(np.array([10.0, 10.0, 10.0], dtype=float))
        x_target = x_current.copy()
        estimator = FakeScanEstimator(x_current=x_current, x_target=x_target)
        supervisor = StiffnessParticleScanSupervisor(
            StiffnessParticleScanConfig(
                enabled=True,
                window_size=2,
                grid_size=5,
            )
        )
        for stamp in range(2):
            supervisor.add_record(np.zeros(3), {0: np.eye(4)}, None, float(stamp))

        result = supervisor.maybe_scan(
            estimator=estimator,
            kp_lim=(1.0, 100.0),
        )

        self.assertTrue(result.attempted)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "no_score_improvement")

    def test_scan_uses_all_dimensions_without_covariance_or_information_gates(self):
        x_current = np.log(np.array([10.0, 10.0, 10.0], dtype=float))
        x_target = np.log(np.array([100.0, 10.0, 10.0], dtype=float))
        estimator = FakeScanEstimator(x_current=x_current, x_target=x_target)
        estimator.P_est = np.eye(3, dtype=float) * 100.0
        supervisor = StiffnessParticleScanSupervisor(
            StiffnessParticleScanConfig(
                enabled=True,
                window_size=20,
                grid_size=5,
            )
        )
        supervisor.add_record(np.zeros(3), {0: np.eye(4)}, None, 0.0)

        result = supervisor.maybe_scan(
            estimator=estimator,
            kp_lim=(1.0, 100.0),
        )

        self.assertTrue(result.attempted)
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "accepted")
        self.assertTrue(np.array_equal(result.active_indices, np.array([0, 1, 2])))
        self.assertTrue(np.allclose(result.x_best, x_target))


if __name__ == "__main__":
    unittest.main()
