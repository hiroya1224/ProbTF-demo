import types
import unittest

import numpy as np

from deflecomp_core.observation.imu_observation import FrameImuObservation
from deflecomp_core.pipeline.compensator import DeflectionCompensator


class FakeRobot:
    def tau_gravity(self, theta):
        return np.ones_like(theta, dtype=float) * 10.0


class FakeSpring:
    def theta_cmd_from_theta_ref(self, tau_gravity, theta_ref, kp_vec):
        return np.asarray(theta_ref, dtype=float) + np.asarray(tau_gravity, dtype=float) / np.asarray(kp_vec, dtype=float)


class FakeEquilibriumSolver:
    def __init__(self):
        self.calls = []

    def solve(self, theta_cmd, kp_vec, theta_init=None):
        self.calls.append(
            {
                "theta_cmd": np.asarray(theta_cmd, dtype=float).copy(),
                "kp_vec": np.asarray(kp_vec, dtype=float).copy(),
                "theta_init": None if theta_init is None else np.asarray(theta_init, dtype=float).copy(),
            }
        )
        return np.asarray(theta_cmd, dtype=float).copy()


class FakeObservationBuilder:
    def build_A_map(self, observations):
        del observations
        return {0: np.eye(4)}


class FakeStiffnessEstimator:
    def __init__(self):
        self.x_est = np.log(np.array([10.0, 10.0], dtype=float))
        self.P_est = np.eye(2)
        self.last_theta_eq = None
        self.update_calls = []

    @property
    def kp_est(self):
        return np.exp(self.x_est)

    @property
    def kp_hat(self):
        return self.kp_est

    def update_with_multi(self, theta_cmd_sent, A_map, theta_init_eq_pred, kp_lim=None):
        self.update_calls.append(
            {
                "theta_cmd_sent": np.asarray(theta_cmd_sent, dtype=float).copy(),
                "A_map": A_map,
                "theta_init_eq_pred": None
                if theta_init_eq_pred is None
                else np.asarray(theta_init_eq_pred, dtype=float).copy(),
                "kp_lim": kp_lim,
            }
        )
        self.x_est = np.log(np.array([100.0, 100.0], dtype=float))
        self.last_theta_eq = np.asarray(theta_cmd_sent, dtype=float).copy()
        debug = {
            "est_update_applied": True,
            "est_update_skipped_reason": None,
            "est_obs_rank": 2,
        }
        return types.SimpleNamespace(
            theta_eq=self.last_theta_eq.copy(),
            x_est=self.x_est.copy(),
            kp_est=self.kp_est.copy(),
            P_est=self.P_est.copy(),
            gradient=np.ones(2),
            information=np.eye(2),
            obs_rank=2,
            update_applied=True,
            update_skipped_reason=None,
            debug=debug,
        )


class FakeParticleStiffnessEstimator:
    def __init__(self):
        self.x_est = np.log(np.array([10.0, 10.0], dtype=float))
        self.P_est = np.eye(2) * 100.0
        self.last_theta_eq = None
        self.update_calls = []
        self.correction_calls = []
        self.target_x = np.log(np.array([200.0, 100.0], dtype=float))

    @property
    def kp_est(self):
        return np.exp(self.x_est)

    @property
    def kp_hat(self):
        return self.kp_est

    def update_with_multi(self, theta_cmd_sent, A_map, theta_init_eq_pred, kp_lim=None):
        self.update_calls.append(
            {
                "theta_cmd_sent": np.asarray(theta_cmd_sent, dtype=float).copy(),
                "A_map": A_map,
                "theta_init_eq_pred": None
                if theta_init_eq_pred is None
                else np.asarray(theta_init_eq_pred, dtype=float).copy(),
                "kp_lim": kp_lim,
            }
        )
        self.x_est = np.log(np.array([100.0, 100.0], dtype=float))
        self.last_theta_eq = np.asarray(theta_cmd_sent, dtype=float).copy()
        return types.SimpleNamespace(
            theta_eq=self.last_theta_eq.copy(),
            x_est=self.x_est.copy(),
            kp_est=self.kp_est.copy(),
            P_est=self.P_est.copy(),
            gradient=np.ones(2),
            information=np.zeros((2, 2), dtype=float),
            obs_rank=0,
            update_applied=True,
            update_skipped_reason=None,
            debug={
                "est_update_applied": True,
                "est_update_skipped_reason": None,
                "est_obs_rank": 0,
            },
        )

    def evaluate_log_likelihood_at_x(
        self,
        x_eval,
        theta_cmd_sent,
        A_map,
        theta_init_eq_pred,
        kp_lim=None,
    ):
        del A_map, theta_init_eq_pred
        x = np.asarray(x_eval, dtype=float).copy()
        if kp_lim is not None:
            x = np.clip(x, np.log(float(kp_lim[0])), np.log(float(kp_lim[1])))
        score = -10.0 * float(np.sum((x - self.target_x) ** 2))
        return types.SimpleNamespace(
            valid=True,
            log_likelihood=score,
            theta_eq=np.asarray(theta_cmd_sent, dtype=float).copy(),
            error=None,
        )

    def apply_particle_correction(
        self,
        x_new,
        active_indices,
        reset_std,
        theta_eq=None,
        kp_lim=None,
    ):
        x = np.asarray(x_new, dtype=float).copy()
        if kp_lim is not None:
            x = np.clip(x, np.log(float(kp_lim[0])), np.log(float(kp_lim[1])))
        self.x_est = x
        for j in np.asarray(active_indices, dtype=int):
            self.P_est[j, j] = max(float(self.P_est[j, j]), float(reset_std) ** 2)
        if theta_eq is not None:
            self.last_theta_eq = np.asarray(theta_eq, dtype=float).copy()
        self.correction_calls.append(
            {
                "x_new": x.copy(),
                "active_indices": np.asarray(active_indices, dtype=int).copy(),
                "reset_std": float(reset_std),
            }
        )


def make_compensator():
    estimator = FakeStiffnessEstimator()
    solver = FakeEquilibriumSolver()
    comp = DeflectionCompensator(
        robot=FakeRobot(),
        spring_model=FakeSpring(),
        stiffness_estimator=estimator,
        equilibrium_solver=solver,
        observation_builder=FakeObservationBuilder(),
        config={
            "kp_lim": (1.0, 500.0),
            "kp_exec_tau": 1.0,
            "max_log_kp_exec_step": 0.0,
            "theta_cmd_tau": 0.0,
            "project_unobservable_feedforward": False,
        },
    )
    return comp, estimator, solver


def make_particle_compensator():
    estimator = FakeParticleStiffnessEstimator()
    solver = FakeEquilibriumSolver()
    comp = DeflectionCompensator(
        robot=FakeRobot(),
        spring_model=FakeSpring(),
        stiffness_estimator=estimator,
        equilibrium_solver=solver,
        observation_builder=FakeObservationBuilder(),
        config={
            "kp_lim": (1.0, 200.0),
            "kp_exec_tau": 1.0,
            "max_log_kp_exec_step": 0.0,
            "theta_cmd_tau": 0.0,
            "project_unobservable_feedforward": False,
            "particle_scan_enabled": True,
            "particle_scan_plain": True,
            "particle_scan_window_size": 1,
            "particle_scan_period": 1,
            "particle_scan_grid_size": 2,
            "particle_scan_max_active_dims": 1,
            "particle_scan_std_trigger": 0.2,
            "particle_scan_min_gain_per_obs": 1000.0,
            "particle_scan_min_log_jump": 1000.0,
            "particle_scan_reset_std": 0.10,
            "particle_scan_cooldown": 20,
        },
    )
    return comp, estimator, solver


class KExecSeparationTests(unittest.TestCase):
    def test_est_update_sets_exec_target_but_exec_moves_smoothly(self):
        comp, estimator, _ = make_compensator()
        theta_ref = np.array([0.0, 0.0], dtype=float)

        first = comp.step(theta_ref=theta_ref, imu_observations=None, dt=0.1, stamp=0.0)
        sent_cmd = first.theta_cmd.copy()

        obs = [FrameImuObservation(frame_name="f", gravity_dir=np.array([0.0, 0.0, -1.0]), stamp=1.0)]
        second = comp.step(theta_ref=theta_ref, imu_observations=obs, dt=0.1, stamp=0.1)

        self.assertEqual(len(estimator.update_calls), 1)
        self.assertTrue(np.allclose(estimator.update_calls[0]["theta_cmd_sent"], sent_cmd))
        self.assertTrue(second.debug["used_theta_cmd_sent_for_update"])
        self.assertTrue(np.allclose(second.kp_est, np.array([100.0, 100.0])))
        self.assertTrue(np.allclose(second.kp_exec_target, np.array([100.0, 100.0])))
        self.assertTrue(np.all(second.kp_exec > np.array([10.0, 10.0])))
        self.assertTrue(np.all(second.kp_exec < second.kp_est))
        self.assertFalse(np.allclose(second.kp_exec, second.kp_est))

    def test_same_observation_stamp_is_not_used_twice(self):
        comp, estimator, _ = make_compensator()
        theta_ref = np.array([0.0, 0.0], dtype=float)
        obs = [FrameImuObservation(frame_name="f", gravity_dir=np.array([0.0, 0.0, -1.0]), stamp=1.0)]

        comp.step(theta_ref=theta_ref, imu_observations=None, dt=0.1, stamp=0.0)
        comp.step(theta_ref=theta_ref, imu_observations=obs, dt=0.1, stamp=0.1)
        third = comp.step(theta_ref=theta_ref, imu_observations=obs, dt=0.1, stamp=0.2)

        self.assertEqual(len(estimator.update_calls), 1)
        self.assertEqual(third.debug["est_update_skipped_reason"], "duplicate_observation")
        self.assertFalse(third.debug["used_theta_cmd_sent_for_update"])

    def test_particle_scan_correction_updates_est_target_without_exec_jump(self):
        comp, estimator, _ = make_particle_compensator()
        theta_ref = np.array([0.0, 0.0], dtype=float)

        comp.step(theta_ref=theta_ref, imu_observations=None, dt=0.1, stamp=0.0)
        obs = [FrameImuObservation(frame_name="f", gravity_dir=np.array([0.0, 0.0, -1.0]), stamp=1.0)]
        second = comp.step(theta_ref=theta_ref, imu_observations=obs, dt=0.1, stamp=0.1)

        expected_x = np.log(np.array([200.0, 100.0], dtype=float))
        self.assertEqual(len(estimator.update_calls), 1)
        self.assertEqual(len(estimator.correction_calls), 1)
        self.assertTrue(second.debug["particle_scan_attempted"])
        self.assertTrue(second.debug["particle_scan_accepted"])
        self.assertTrue(np.allclose(estimator.x_est, expected_x))
        self.assertTrue(np.allclose(second.kp_est, np.exp(expected_x)))
        self.assertTrue(np.allclose(second.kp_exec_target, second.kp_est))
        self.assertFalse(np.allclose(second.kp_exec, second.kp_est))
        self.assertTrue(np.all(second.kp_exec > np.array([10.0, 10.0])))
        self.assertTrue(np.all(second.kp_exec < second.kp_exec_target))


if __name__ == "__main__":
    unittest.main()
