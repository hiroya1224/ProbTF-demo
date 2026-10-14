"""Verify the distinction between displayed and recursively retained evidence."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'ros/examples/probtf_orientation_demo/scripts/compare_feedback_orientation.py'
spec = importlib.util.spec_from_file_location('orientation_feedback_comparison', SCRIPT)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class FeedbackComparisonTest(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(5, dtype=float)*0.04
        self.gyro = np.tile([0.1, -0.2, 0.3], (5, 1))
        self.acc = np.tile([0.6, -0.4, 9.8], (5, 1))
        self.mag = np.tile([0.2, 0.8, -0.5], (5, 1))
        self.q = np.array([1., 0., 0., 0.])
        self.reference = np.array([1., 0., 0.])
        self.options = dict(initial_concentration=20., gyro_variance=0.,
                            acceleration_concentration=5., magnetic_concentration=5.)

    def run_six(self, acceleration=None, magnetic=None):
        return comparison.compare_six(self.gyro, self.acc if acceleration is None else acceleration,
            self.mag if magnetic is None else magnetic, self.times, self.q, self.reference, **self.options)

    def test_output_only_evidence_is_not_accumulated(self):
        self.gyro[:] = 0
        output, _, retained, _ = self.run_six()
        initial = 20.*(np.outer(self.q, self.q)-np.eye(4))
        gravity, field = comparison.direction_factors(self.acc, self.mag, self.reference, 5., 5.)
        canonical = comparison.original.canonical_bingham_parameter
        for k in range(5):
            np.testing.assert_allclose(retained[k, 0], initial, atol=2e-12)
            np.testing.assert_allclose(output[k, 1], canonical(initial+gravity[k]), atol=2e-12)
            np.testing.assert_allclose(output[k, 3], canonical(initial+gravity[k]+field[k]), atol=2e-12)
            np.testing.assert_allclose(output[k, 4], canonical(initial+(k+1)*gravity[k]+field[k]), atol=2e-12)
            np.testing.assert_allclose(output[k, 5], canonical(initial+(k+1)*(gravity[k]+field[k])), atol=2e-12)

    def test_past_magnetic_data_cannot_change_partial_feedback_output(self):
        other_mag = self.mag.copy()
        other_mag[:-1] = [0.8, -0.2, 0.5]
        first, _, state_a, _ = self.run_six()
        second, _, state_b, _ = self.run_six(magnetic=other_mag)
        np.testing.assert_allclose(state_a[:, :2], state_b[:, :2], atol=1e-12)
        np.testing.assert_allclose(first[-1, :5], second[-1, :5], atol=1e-12)
        self.assertGreater(np.linalg.norm(first[-1, 5]-second[-1, 5]), 1.)

    def test_past_acceleration_changes_only_feedback_cases_at_final_sample(self):
        other_acc = self.acc.copy()
        other_acc[:-1] = [9.8, -0.4, 0.6]
        first, _, _, _ = self.run_six()
        second, _, _, _ = self.run_six(acceleration=other_acc)
        np.testing.assert_allclose(first[-1, [0, 1, 3]], second[-1, [0, 1, 3]], atol=1e-12)
        for case in (2, 4, 5):
            self.assertGreater(np.linalg.norm(first[-1, case]-second[-1, case]), 1.)

    def test_cache_conversion_matches_independent_fresh_replay(self):
        old, old_modes, old_residuals = comparison.original.compare_conditions(
            self.gyro, self.acc, self.mag, self.times, self.q, self.reference, **self.options)
        metadata = dict(case_labels=comparison.original.CASE_IDS,
                        reference_magnetic_world_unit=self.reference,
                        initial_quaternion_wxyz=self.q,
                        acceleration_direction_concentration=5., magnetic_direction_concentration=5.)
        base = dict(acceleration_m_s2=self.acc, magnetic_raw=self.mag,
                    display_parameter=old, mode_wxyz=old_modes, moment_fit_residual=old_residuals)
        converted = comparison.from_five(base, metadata)
        fresh = self.run_six()
        for a, b in zip(converted, fresh):
            np.testing.assert_allclose(a, b, atol=2e-12)

    def test_prediction_uses_right_endpoint_and_rejects_duplicate_time(self):
        self.gyro[0] = [100., 100., 100.]
        first, _, _, _ = self.run_six()
        self.gyro[0] = [-100., -100., -100.]
        second, _, _, _ = self.run_six()
        np.testing.assert_array_equal(first, second)
        self.gyro[1] *= -1
        third, _, _, _ = self.run_six()
        self.assertGreater(np.linalg.norm(first[1:, 0]-third[1:, 0]), 0.1)
        self.times[1] = self.times[0]
        with self.assertRaisesRegex(ValueError, 'strictly increase'):
            self.run_six()


if __name__ == '__main__':
    unittest.main()
