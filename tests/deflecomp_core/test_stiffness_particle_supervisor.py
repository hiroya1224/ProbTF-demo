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

    def frame_angular_jacobian_base(self, theta, fid):
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
        self.P_est = np.eye(self.x_est.size, dtype=float)
        self.x_target = np.asarray(x_target, dtype=float).copy()
        self.score_scale = float(score_scale)

    def target_for_record(self, theta_cmd_sent):
        del theta_cmd_sent
        return self.x_target

    def equilibrium_for_candidate(self, x, theta_cmd_sent):
        del x, theta_cmd_sent
        return np.zeros_like(self.x_est)

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
        target = np.asarray(self.target_for_record(theta_cmd_sent), dtype=float)
        score = -self.score_scale * float(np.sum((x - target) ** 2))
        return types.SimpleNamespace(
            valid=True,
            log_likelihood=score,
            theta_eq=np.asarray(
                self.equilibrium_for_candidate(x, theta_cmd_sent), dtype=float
            ),
            error=None,
        )


class HoldoutConflictEstimator(FakeScanEstimator):
    def __init__(self, x_current, training_target):
        super().__init__(x_current=x_current, x_target=training_target)
        self.x_current_reference = np.asarray(x_current, dtype=float).copy()

    def target_for_record(self, theta_cmd_sent):
        if float(np.asarray(theta_cmd_sent)[0]) > 0.5:
            return self.x_current_reference
        return self.x_target


class BranchJumpEstimator(FakeScanEstimator):
    def __init__(self, x_current, x_target):
        super().__init__(x_current=x_current, x_target=x_target)
        self.x_current_reference = np.asarray(x_current, dtype=float).copy()

    def equilibrium_for_candidate(self, x, theta_cmd_sent):
        del theta_cmd_sent
        if np.max(np.abs(np.asarray(x) - self.x_current_reference)) > 0.1:
            return np.ones_like(self.x_est)
        return np.zeros_like(self.x_est)


def safe_config(**overrides):
    values = {
        "enabled": True,
        "window_size": 4,
        "period": 1,
        "grid_size": 5,
        "max_active_dims": 3,
        "min_validation_records": 1,
        "validation_fraction": 0.25,
        "min_gain_per_obs": 0.1,
        "min_log_jump": 0.01,
        "max_log_jump": float(np.log(10.0)),
        "cooldown": 3,
    }
    values.update(overrides)
    return StiffnessParticleScanConfig(**values)


def add_records(supervisor, count, information=None, validation_marker=False):
    if information is None:
        information = np.diag([3.0, 2.0, 1.0])
    for index in range(count):
        theta_cmd = np.zeros(3, dtype=float)
        if validation_marker and index == count - 1:
            theta_cmd[0] = 1.0
        supervisor.add_record(
            theta_cmd_sent=theta_cmd,
            A_map={0: np.eye(4)},
            theta_init_eq_pred=np.zeros(3),
            stamp=float(index),
            information=information,
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

    def test_axis_candidates_include_current_stay_bounded_and_limit_jump(self):
        supervisor = StiffnessParticleScanSupervisor(
            StiffnessParticleScanConfig(grid_size=5, max_log_jump=0.4)
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
            self.assertLessEqual(np.max(np.abs(candidate - x_current)), 0.4 + 1e-12)

    def test_scan_waits_for_a_complete_window(self):
        x_current = np.log(np.array([10.0, 10.0, 10.0]))
        estimator = FakeScanEstimator(x_current, np.log(np.array([100.0, 10.0, 10.0])))
        supervisor = StiffnessParticleScanSupervisor(safe_config(window_size=4))
        add_records(supervisor, 3)

        result = supervisor.maybe_scan(estimator, kp_lim=(1.0, 100.0))

        self.assertFalse(result.attempted)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "window_not_full")

    def test_scan_accepts_improvement_on_discovery_and_holdout_records(self):
        x_current = np.log(np.array([10.0, 10.0, 10.0], dtype=float))
        x_target = np.log(np.array([100.0, 10.0, 10.0], dtype=float))
        estimator = FakeScanEstimator(x_current=x_current, x_target=x_target)
        supervisor = StiffnessParticleScanSupervisor(safe_config())
        add_records(supervisor, 4)

        result = supervisor.maybe_scan(estimator=estimator, kp_lim=(1.0, 100.0))

        self.assertTrue(result.attempted)
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "accepted")
        self.assertTrue(np.allclose(result.x_best, x_target))
        self.assertGreater(result.debug["training_gain_per_obs"], 0.1)
        self.assertGreater(result.debug["validation_gain_per_obs"], 0.1)
        self.assertIsNotNone(result.theta_eq_best)

    def test_scan_rejects_when_score_does_not_improve(self):
        x_current = np.log(np.array([10.0, 10.0, 10.0], dtype=float))
        estimator = FakeScanEstimator(x_current=x_current, x_target=x_current)
        supervisor = StiffnessParticleScanSupervisor(safe_config())
        add_records(supervisor, 4)

        result = supervisor.maybe_scan(estimator=estimator, kp_lim=(1.0, 100.0))

        self.assertTrue(result.attempted)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "no_training_improvement")

    def test_information_gate_does_not_scan_unobservable_target_dimension(self):
        x_current = np.log(np.array([10.0, 10.0, 10.0], dtype=float))
        x_target = np.log(np.array([100.0, 10.0, 10.0], dtype=float))
        estimator = FakeScanEstimator(x_current=x_current, x_target=x_target)
        supervisor = StiffnessParticleScanSupervisor(
            safe_config(max_active_dims=1)
        )
        add_records(supervisor, 4, information=np.diag([0.0, 5.0, 0.0]))

        result = supervisor.maybe_scan(estimator=estimator, kp_lim=(1.0, 100.0))

        self.assertTrue(result.attempted)
        self.assertFalse(result.accepted)
        self.assertTrue(np.array_equal(result.active_indices, np.array([1])))
        self.assertEqual(result.debug["active_direction_count"], 1)

    def test_missing_information_is_fail_closed(self):
        x_current = np.log(np.array([10.0, 10.0, 10.0], dtype=float))
        estimator = FakeScanEstimator(x_current=x_current, x_target=x_current + 1.0)
        supervisor = StiffnessParticleScanSupervisor(safe_config())
        for index in range(4):
            supervisor.add_record(np.zeros(3), {0: np.eye(4)}, None, float(index))

        result = supervisor.maybe_scan(estimator=estimator, kp_lim=(1.0, 100.0))

        self.assertTrue(result.attempted)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "no_active_directions")
        self.assertEqual(result.debug["information_reason"], "missing_or_invalid_information")

    def test_holdout_rejects_candidate_that_only_fits_discovery_records(self):
        x_current = np.log(np.array([10.0, 10.0, 10.0], dtype=float))
        x_target = np.log(np.array([100.0, 10.0, 10.0], dtype=float))
        estimator = HoldoutConflictEstimator(x_current=x_current, training_target=x_target)
        supervisor = StiffnessParticleScanSupervisor(safe_config())
        add_records(supervisor, 4, validation_marker=True)

        result = supervisor.maybe_scan(estimator=estimator, kp_lim=(1.0, 100.0))

        self.assertTrue(result.attempted)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "validation_gain_too_small")
        self.assertGreater(result.debug["training_gain_per_obs"], 0.0)
        self.assertLess(result.debug["validation_gain_per_obs"], 0.0)

    def test_equilibrium_branch_jump_is_rejected(self):
        x_current = np.log(np.array([10.0, 10.0, 10.0], dtype=float))
        x_target = np.log(np.array([100.0, 10.0, 10.0], dtype=float))
        estimator = BranchJumpEstimator(x_current=x_current, x_target=x_target)
        supervisor = StiffnessParticleScanSupervisor(
            safe_config(max_equilibrium_jump=0.25)
        )
        add_records(supervisor, 4)

        result = supervisor.maybe_scan(estimator=estimator, kp_lim=(1.0, 100.0))

        self.assertTrue(result.attempted)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "equilibrium_branch_discontinuity")
        self.assertGreater(result.debug["branch_rejection_count"], 0)

    def test_proposal_is_limited_to_configured_log_jump(self):
        x_current = np.log(np.array([10.0, 10.0, 10.0], dtype=float))
        x_target = np.log(np.array([100.0, 10.0, 10.0], dtype=float))
        estimator = FakeScanEstimator(x_current=x_current, x_target=x_target)
        limit = float(np.log(2.0))
        supervisor = StiffnessParticleScanSupervisor(
            safe_config(max_log_jump=limit)
        )
        add_records(supervisor, 4)

        result = supervisor.maybe_scan(estimator=estimator, kp_lim=(1.0, 100.0))

        self.assertTrue(result.accepted)
        self.assertLessEqual(np.max(np.abs(result.x_best - x_current)), limit + 1e-12)
        self.assertFalse(np.allclose(result.x_best, x_target))

    def test_period_and_cooldown_count_new_records_not_idle_calls(self):
        x_current = np.log(np.array([10.0, 10.0, 10.0], dtype=float))
        x_target = np.log(np.array([100.0, 10.0, 10.0], dtype=float))
        estimator = FakeScanEstimator(x_current=x_current, x_target=x_target)
        supervisor = StiffnessParticleScanSupervisor(
            safe_config(period=2, cooldown=3)
        )
        add_records(supervisor, 4)
        accepted = supervisor.maybe_scan(estimator=estimator, kp_lim=(1.0, 100.0))
        self.assertTrue(accepted.accepted)
        supervisor.register_result(accepted, applied=True)

        idle_result = supervisor.maybe_scan(estimator=estimator, kp_lim=(1.0, 100.0))
        self.assertEqual(idle_result.reason, "cooldown")
        add_records(supervisor, 2)
        still_cooling = supervisor.maybe_scan(estimator=estimator, kp_lim=(1.0, 100.0))
        self.assertEqual(still_cooling.reason, "cooldown")
        add_records(supervisor, 1)
        next_result = supervisor.maybe_scan(estimator=estimator, kp_lim=(1.0, 100.0))
        self.assertTrue(next_result.attempted)

    def test_snapshot_copies_information_and_async_bookkeeping(self):
        supervisor = StiffnessParticleScanSupervisor(safe_config())
        add_records(supervisor, 4)
        supervisor.last_scan_record_count = 3
        supervisor.last_accept_record_count = 2

        snapshot = supervisor.snapshot()
        snapshot.records[0].information[0, 0] = 999.0

        self.assertEqual(snapshot.record_count, supervisor.record_count)
        self.assertEqual(snapshot.last_scan_record_count, 3)
        self.assertEqual(snapshot.last_accept_record_count, 2)
        self.assertNotEqual(snapshot.records[0].information[0, 0], supervisor.records[0].information[0, 0])

    def test_async_result_freshness_rejects_old_or_drifted_result(self):
        x_current = np.log(np.array([10.0, 10.0, 10.0], dtype=float))
        x_target = x_current.copy()
        x_target[0] += 0.5
        estimator = FakeScanEstimator(x_current=x_current, x_target=x_target)
        supervisor = StiffnessParticleScanSupervisor(
            safe_config(
                grid_size=3,
                max_log_jump=0.5,
                max_result_age_records=1,
                max_estimator_drift=0.1,
            )
        )
        add_records(supervisor, 4)
        result = supervisor.maybe_scan(estimator=estimator, kp_lim=(1.0, 100.0))
        self.assertTrue(result.accepted)

        add_records(supervisor, 2)
        fresh, reason, debug = supervisor.result_freshness(result, x_current)
        self.assertFalse(fresh)
        self.assertEqual(reason, "stale_result_records")
        self.assertEqual(debug["result_age_records"], 2)

        supervisor.config.max_result_age_records = 10
        fresh, reason, _ = supervisor.result_freshness(result, x_current + 0.2)
        self.assertFalse(fresh)
        self.assertEqual(reason, "stale_result_estimator_drift")


if __name__ == "__main__":
    unittest.main()
