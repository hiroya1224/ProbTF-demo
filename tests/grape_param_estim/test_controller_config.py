import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from grape_param_estim.controller import ControllerConfig
from grape_param_estim.controller_config import (
    BaselineSelectionRequired,
    PID_GROUPS,
    PidGainComparison,
    PidGainConfiguration,
    apply_pid_gain_configuration,
    configuration_from_controller_snapshot,
    render_pid_diff_yaml,
    render_proposed_pid_yaml,
    select_baseline_pid_configuration,
    validate_controller_yaml_key_contract,
)


class ControllerConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.recorded = np.asarray(
            (
                (3.0, 0.1, 1.0),
                (5.0, 1.0, 2.5),
                (20.0, 1.0, 8.0),
                (4.0, 1.0, 2.0),
            )
        )
        self.snapshot = SimpleNamespace(
            groups=PID_GROUPS,
            gains=self.recorded,
            record_times=np.asarray((10.0, 10.1, 10.2, 10.3)),
            source_kinds=("recorded_startup",) * 4,
        )

    def test_recorded_snapshot_is_exact_current_configuration(self):
        configuration = configuration_from_controller_snapshot(
            self.snapshot, "failed-flight"
        )
        np.testing.assert_array_equal(configuration.values, self.recorded)
        self.assertEqual(configuration.provenance.bag_id, "failed-flight")
        self.assertIn("/xy/parameter_updates", configuration.provenance.topics[0])

    def test_differing_snapshots_require_explicit_baseline_without_average(self):
        first = PidGainConfiguration(self.recorded)
        second_values = self.recorded.copy()
        second_values[0, 0] = 4.0
        second = PidGainConfiguration(second_values)
        with self.assertRaises(BaselineSelectionRequired):
            select_baseline_pid_configuration(
                {"failed": first, "success": second}
            )
        selected = select_baseline_pid_configuration(
            {"failed": first, "success": second}, "failed"
        )
        np.testing.assert_array_equal(selected.values, self.recorded)

    def test_applying_gains_preserves_limits_and_controller_model_flags(self):
        original = ControllerConfig.grape()
        gains = PidGainConfiguration(self.recorded)
        applied = apply_pid_gain_configuration(original, gains)
        self.assertEqual(applied.xy_control_mode, original.xy_control_mode)
        self.assertEqual(
            applied.source_compatible_gyro_term,
            original.source_compatible_gyro_term,
        )
        for before, after in zip(original.pid, applied.pid):
            self.assertEqual(before.limit_sum, after.limit_sum)
            self.assertEqual(before.limit_error_i, after.limit_error_i)
        np.testing.assert_allclose(
            (applied.pid[0].p_gain, applied.pid[2].p_gain,
             applied.pid[3].p_gain, applied.pid[5].p_gain),
            self.recorded[:, 0],
        )

    def test_proposed_yaml_contains_only_twelve_gain_keys(self):
        yaml_text = render_proposed_pid_yaml(
            PidGainConfiguration(self.recorded)
        )
        self.assertNotIn("controller:", yaml_text)
        self.assertNotIn("limit", yaml_text)
        self.assertEqual(yaml_text.count("_gain:"), 12)
        for group in PID_GROUPS:
            self.assertIn("{}:\n".format(group), yaml_text)

    def test_diff_uses_recorded_current_for_difference_and_ratio(self):
        proposed = self.recorded * np.asarray((1.1, 1.0, 0.9, 1.0))[:, None]
        comparison = PidGainComparison.from_configurations(
            PidGainConfiguration(self.recorded),
            PidGainConfiguration(proposed),
        )
        text = render_pid_diff_yaml(comparison)
        self.assertIn("current: 3", text)
        self.assertIn("proposed: 3.3000000000000003", text)
        self.assertIn("ratio: 1.1000000000000001", text)
        self.assertNotIn("z:", text)

    def test_reference_yaml_is_key_contract_only_and_never_modified(self):
        source = Path(
            "/home/leus/catkin_ws/src/jsk_aerial_robot/robots/"
            "gimbalrotor/config/grape/GimbalrotorControl.yaml"
        )
        before = source.read_bytes()
        paths = validate_controller_yaml_key_contract(str(source))
        self.assertEqual(len(paths), 12)
        self.assertEqual(source.read_bytes(), before)

    def test_missing_key_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "controller.yaml"
            source.write_text("controller:\n  xy:\n    p_gain: 1\n")
            with self.assertRaises(ValueError):
                validate_controller_yaml_key_contract(str(source))


if __name__ == "__main__":
    unittest.main()
