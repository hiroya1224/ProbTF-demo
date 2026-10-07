import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

PACKAGE = Path(__file__).resolve().parents[2] / "ros/examples/deflecomp/deflecomp_debug"
sys.path.insert(0, str(PACKAGE / "src"))
from deflecomp_debug.convergence_demo import (
    PredictionConvergenceMonitor,
    evaluate_snapshot, interpolate_reference, phase_summary, prediction_stable_window, stable_window,
    validate_scenario,
)


class Pose:
    def __init__(self, theta):
        self.translation = np.array([theta[0], 0.0, 0.0])
        self.rotation = np.eye(3)


class Robot:
    nv = 1

    def fk_pose(self, theta):
        return Pose(theta)


class ConvergenceDemoTest(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((PACKAGE / "config/convergence_demo.yaml").read_text())
        self.criteria = self.config["criteria"]

    def window(self):
        return [{"time_s": t, "prediction_joint_rms_deg": 0.5,
                 "execution_log_k_gap": 0.01, "kp_est_0": 5.0,
                 "gate_ready": 1, "update_applied": 0}
                for t in np.arange(0.0, 3.01, 0.1)]

    def test_load_change_cannot_also_change_reference(self):
        config = copy.deepcopy(self.config)
        config["phases"][-2]["target_deg"][0] += 5
        with self.assertRaisesRegex(ValueError, "keep the previous reference"):
            validate_scenario(config)

    def test_reject_nonfinite_load_and_short_hold(self):
        for key, value in (("mass_kg", np.nan), ("hold_s", 0.5)):
            config = copy.deepcopy(self.config)
            config["phases"][0][key] = value
            with self.assertRaises(ValueError):
                validate_scenario(config)

    def test_short_requested_holds_are_allowed_when_not_checked_for_convergence(self):
        config = yaml.safe_load((PACKAGE / "config/tip_load_step.yaml").read_text())
        validate_scenario(config)
        np.testing.assert_allclose(np.radians(config["phases"][2]["target_deg"]),
                                   [0, -np.pi / 3, np.pi / 5, 0, 0, 0])
        config["phases"][0]["check_convergence"] = True
        with self.assertRaisesRegex(ValueError, "hold too short"):
            validate_scenario(config)

    def test_invalid_convergence_wait_is_rejected(self):
        config = yaml.safe_load((PACKAGE / "config/tip_load_step.yaml").read_text())
        config["phases"][-2]["min_hold_s"] = 0
        with self.assertRaisesRegex(ValueError, "min_hold_s"):
            validate_scenario(config)

    def test_optional_tip_error_cannot_be_missing_or_large(self):
        criteria = dict(self.criteria, prediction_tip_mm=5)
        rows = self.window()
        self.assertFalse(prediction_stable_window(rows, criteria))
        for row in rows:
            row["prediction_tip_mm"] = 1.0
        self.assertTrue(prediction_stable_window(rows, criteria))
        rows[-1]["prediction_tip_mm"] = 10.0
        self.assertFalse(prediction_stable_window(rows, criteria))

    def test_wait_uses_only_post_load_samples_and_rejects_missing_data(self):
        monitor = PredictionConvergenceMonitor(self.criteria, start_time_s=10, min_hold_s=5)
        for row in self.window():
            monitor.observe(row)
        self.assertFalse(monitor.converged(15))
        for t in np.arange(10, 13.01, 0.1):
            row = dict(self.window()[0], time_s=float(t))
            monitor.observe(row)
        self.assertFalse(monitor.converged(13))
        self.assertFalse(monitor.converged(15))
        for t in np.arange(13.1, 15.11, 0.1):
            monitor.observe(dict(row, time_s=float(t)))
        self.assertTrue(monitor.converged(15.11))
        self.assertFalse(monitor.converged(16))
        monitor.observe(dict(row, time_s=16))
        self.assertFalse(monitor.converged(16))

    def test_live_wait_covers_the_offline_sampling_confirmation_margin(self):
        monitor = PredictionConvergenceMonitor(self.criteria, start_time_s=0,
                                               min_hold_s=3, confirmation_s=0.5)
        for row in self.window():
            monitor.observe(row)
        self.assertFalse(monitor.converged(3))
        for t in (3.1, 3.2, 3.3, 3.4, 3.5):
            monitor.observe(dict(self.window()[0], time_s=t))
        self.assertTrue(monitor.converged(3.5))

    def test_smooth_move_finishes_at_exact_hold_and_never_overshoots(self):
        start, target = np.array([-0.5, 0.2]), np.array([0.5, -0.2])
        values = np.array([interpolate_reference(start, target, t, 2) for t in np.linspace(-1, 3, 81)])
        self.assertTrue(np.all(values >= np.minimum(start, target)))
        self.assertTrue(np.all(values <= np.maximum(start, target)))
        np.testing.assert_array_equal(interpolate_reference(start, target, 2, 2), target)
        np.testing.assert_array_equal(interpolate_reference(start, target, 0, 0), target)

    def test_flat_stiffness_does_not_hide_inaccurate_prediction(self):
        rows = self.window()
        self.assertTrue(stable_window(rows, self.criteria, 1))
        rows[-1]["prediction_joint_rms_deg"] = 8.0
        self.assertFalse(stable_window(rows, self.criteria, 1))

    def test_stale_gap_closed_gate_and_execution_lag_fail_stability(self):
        rows = self.window()
        self.assertFalse(stable_window(rows[:5] + rows[20:], self.criteria, 1))
        for row in rows:
            row["gate_ready"] = 0
        self.assertFalse(stable_window(rows, self.criteria, 1))
        rows = self.window()
        rows[-1]["execution_log_k_gap"] = 0.2
        self.assertFalse(stable_window(rows, self.criteria, 1))

    def test_stiffness_drift_fails_even_when_prediction_is_accurate(self):
        rows = self.window()
        rows[-1]["kp_est_0"] = 6.0
        self.assertFalse(stable_window(rows, self.criteria, 1))
        self.assertTrue(prediction_stable_window(rows, self.criteria))

    def test_snapshot_rejects_old_data_and_reports_metres_as_mm(self):
        latest = {key: (10.0, [value]) for key, value in
                  {"reference": 0.0, "command": 0.0, "actual": 0.001,
                   "prediction": 0.0, "kp_est": 5.0, "kp_exec": 5.0,
                   "kp_cov": 0.01}.items()}
        row = evaluate_snapshot(latest, Robot(), 10.1, 10.0)
        self.assertAlmostEqual(row["prediction_tip_mm"], 1.0)
        latest["actual"] = (9.0, [0.001])
        self.assertIsNone(evaluate_snapshot(latest, Robot(), 10.1, 10.0))
        latest["actual"] = (11.0, [0.001])
        self.assertIsNone(evaluate_snapshot(latest, Robot(), 10.1, 10.0))

    def test_missing_tail_is_not_reported_as_converged(self):
        rows = self.window()
        for row in rows:
            for key in ("prediction_tip_mm", "prediction_tip_deg", "tracking_tip_mm", "tracking_tip_deg"):
                row[key] = 0.0
        events = [{"kind": "hold", "phase": "test", "time_s": 0.0, "mass_kg": 0.0},
                  {"kind": "finished", "phase": "test", "time_s": 5.0, "mass_kg": 0.0}]
        summary = phase_summary(rows, events, self.criteria, 1)[0]
        self.assertFalse(summary["stable_at_end"])
        self.assertIsNotNone(summary["first_stable_window_s"])


if __name__ == "__main__":
    unittest.main()
