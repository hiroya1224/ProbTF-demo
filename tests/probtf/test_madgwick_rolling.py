"""Checks of the Madgwick control and its measurement-frame conventions."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

root = Path.cwd() if 'PROBTF_MADGWICK_SCRIPT' in os.environ else Path(__file__).resolve().parents[2]
script = Path(os.environ.get('PROBTF_MADGWICK_SCRIPT',
    root/'ros/examples/probtf_orientation_demo/scripts/compare_madgwick_rolling.py'))
spec = importlib.util.spec_from_file_location('madgwick_comparison', script)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def wxyz(rotation):
    return rotation.as_quat()[..., [3, 0, 1, 2]]


class MadgwickComparisonTest(unittest.TestCase):
    def test_residual_is_the_physical_body_direction(self):
        attitude = Rotation.from_euler('xyz', [31, -17, 53], degrees=True)
        direction = np.array([0.3, 0.4, -0.8660254037844386])
        expected = attitude.inv().apply(direction)
        residual, _ = analysis.direction_residual_jacobian(wxyz(attitude), direction, expected)
        np.testing.assert_allclose(residual, 0, atol=4e-16)

    def test_jacobian_matches_independent_numerical_differentiation(self):
        q = wxyz(Rotation.from_euler('xyz', [31, -17, 53], degrees=True))
        direction = np.array([0.3, 0.4, -0.8660254037844386])
        observed = np.array([-0.2, 0.7, 0.5])
        _, analytical = analysis.direction_residual_jacobian(q, direction, observed)
        numerical = np.empty((3, 4))
        for component in range(4):
            offset = np.zeros(4); offset[component] = 1e-6
            plus, _ = analysis.direction_residual_jacobian(q+offset, direction, observed)
            minus, _ = analysis.direction_residual_jacobian(q-offset, direction, observed)
            numerical[:, component] = (plus-minus)/(2e-6)
        np.testing.assert_allclose(analytical, numerical, atol=2e-10, rtol=0)

    def test_exact_stationary_alignment_remains_finite_and_fixed(self):
        attitude = Rotation.from_euler('xyz', [31, -17, 53], degrees=True)
        q = wxyz(attitude)
        accel = attitude.inv().apply([0, 0, 9.81])
        magnetic = attitude.inv().apply([0.6, 0, -0.8])
        next_q = analysis.madgwick_update(q, [0, 0, 0], accel, 0.005, magnetic)
        self.assertTrue(np.isfinite(next_q).all())
        np.testing.assert_allclose(next_q, q, atol=2e-16)

    def test_static_two_direction_observations_recover_attitude(self):
        truth = Rotation.from_euler('xyz', [26, -17, 48], degrees=True)
        initial = Rotation.from_euler('xyz', [15, 6, -10], degrees=True)*truth
        accel = truth.inv().apply([0, 0, 9.81])
        magnetic = truth.inv().apply([0.6, 0, -0.8])
        q = wxyz(initial)
        for _ in range(3000):
            q = analysis.madgwick_update(q, [0, 0, 0], accel, 0.005, magnetic)
        error = (truth.inv()*analysis.original.rotations_from_wxyz(q)).magnitude()
        self.assertLess(np.rad2deg(error), 0.15)

    def test_six_axis_update_is_invariant_to_world_vertical_heading(self):
        first = Rotation.from_euler('xyz', [20, -14, 30], degrees=True)
        yaw = Rotation.from_euler('z', 79, degrees=True)
        accel = np.array([0.2, -0.3, 9.6])
        gyro = np.array([0.1, -0.2, 0.3])
        a = analysis.madgwick_update(wxyz(first), gyro, accel, 0.005)
        b = analysis.madgwick_update(wxyz(yaw*first), gyro, accel, 0.005)
        expected = wxyz(yaw*analysis.original.rotations_from_wxyz(a))
        np.testing.assert_allclose(b, expected, atol=4e-16)

    def test_normalization_removes_sensor_units(self):
        q = wxyz(Rotation.from_euler('xyz', [12, 7, 21], degrees=True))
        a = np.array([0.3, 0.1, 9.7]); m = np.array([13.0, -4.0, -20.0])
        first = analysis.madgwick_update(q, [0.1, 0.2, 0.3], a, 0.005, m)
        second = analysis.madgwick_update(q, [0.1, 0.2, 0.3], a*0.1, 0.005, m*1000)
        np.testing.assert_allclose(first, second, atol=2e-16)

    def test_prefix_heading_frame_has_zero_east_and_preserves_initial_pose(self):
        reference = np.array([-0.3, 0.4, -0.8660254037844386])
        transform, _ = analysis.heading_frame(reference)
        np.testing.assert_allclose(transform.inv().apply(reference), [0.5, 0, reference[2]], atol=2e-16)
        truth = Rotation.from_euler('xyz', [12, -19, 44], degrees=True)
        count = 4
        arrays = {'measurement_stamps': np.arange(count)*0.005,
                  'mocap_wxyz': np.tile(wxyz(truth), (count, 1)),
                  'gyro_rad_s': np.zeros((count, 3)),
                  'acceleration_m_s2': np.tile(truth.inv().apply([0, 0, 9.81]), (count, 1)),
                  'magnetic_raw': np.tile(truth.inv().apply(reference), (count, 1))}
        modes = analysis.run_madgwick(arrays, reference)
        np.testing.assert_allclose(modes, np.tile(wxyz(truth), (count, 2, 1)), atol=4e-16)

    def test_no_later_mocap_orientation_enters_either_filter(self):
        count = 8
        arrays = {'measurement_stamps': np.arange(count)*0.005,
                  'mocap_wxyz': np.tile([1, 0, 0, 0], (count, 1)).astype(float),
                  'gyro_rad_s': np.tile([0.1, 0.2, 0.3], (count, 1)),
                  'acceleration_m_s2': np.tile([0, 0, 9.81], (count, 1)),
                  'magnetic_raw': np.tile([0.6, 0, -0.8], (count, 1))}
        first = analysis.run_madgwick(arrays, [0.6, 0, -0.8])
        arrays['mocap_wxyz'][1:] = wxyz(Rotation.from_euler('z', 90, degrees=True))
        second = analysis.run_madgwick(arrays, [0.6, 0, -0.8])
        np.testing.assert_array_equal(first, second)

    def test_right_end_samples_and_measured_intervals(self):
        stamps = np.array([0, 0.005, 0.013, 0.016, 0.035])
        gyro = np.column_stack((np.zeros((5, 2)), np.array([999, 0.3, 0.5, 0.7, 0.9])))
        arrays = {'measurement_stamps': stamps,
                  'mocap_wxyz': np.tile([1, 0, 0, 0], (5, 1)).astype(float),
                  'gyro_rad_s': gyro, 'acceleration_m_s2': np.tile([0, 0, 9.81], (5, 1)),
                  'magnetic_raw': np.tile([0.6, 0, -0.8], (5, 1))}
        modes = analysis.run_madgwick(arrays, [0.6, 0, -0.8], beta=0)
        expected_yaw = 2*np.cumsum(np.arctan(gyro[1:, 2]*np.diff(stamps)/2))
        measured_yaw = analysis.original.rotations_from_wxyz(modes[1:, 0]).as_rotvec()[:, 2]
        np.testing.assert_allclose(measured_yaw, expected_yaw, atol=1e-16)
        np.testing.assert_array_equal(modes[:, 0], modes[:, 1])

    def test_compass_control_uses_reference_tilt_and_reveals_field_rotation(self):
        truth = Rotation.from_euler('xyz', [[10, -12, 25], [20, -18, 45], [-30, 22, 67]], degrees=True)
        reference = np.array([-0.3, 0.4, -0.8660254037844386])
        expected_yaw = truth.as_euler('xyz', degrees=True)[:, 2]
        arrays = {'mocap_wxyz': wxyz(truth), 'mocap_rpy_deg': truth.as_euler('xyz', degrees=True),
                  'magnetic_raw': truth.inv().apply(reference)}
        _, yaw, error = analysis.magnetic_heading_control(arrays, reference)
        np.testing.assert_allclose(yaw, expected_yaw, atol=6e-14)
        np.testing.assert_allclose(error, 0, atol=6e-14)
        rotated_field = Rotation.from_euler('z', 15, degrees=True).apply(reference)
        arrays['magnetic_raw'] = truth.inv().apply(rotated_field)
        _, yaw, error = analysis.magnetic_heading_control(arrays, reference)
        np.testing.assert_allclose(yaw, expected_yaw-15, atol=6e-14)
        np.testing.assert_allclose(error, 15, atol=6e-14)

    def test_plots_compare_only_the_two_nine_axis_results_in_four_rows(self):
        count = 3
        error = np.tile(np.array([2.0, 50.0]), (count, 1))
        arrays = {'time_seconds': np.arange(count),
                  'rpy_error_deg': np.repeat(error[:, :, None], 3, axis=2),
                  'error_deg': error}
        checked = []
        def inspect(figure, output, name):
            self.assertEqual(len(figure.axes), 4)
            for axis in figure.axes:
                self.assertEqual([line.get_label() for line in axis.lines],
                                 ['9軸の全情報の融合', 'Madgwick フィルタ（9軸）'])
                self.assertEqual(len(axis.lines), 2)
                self.assertEqual(axis.get_yscale(), 'log' if name.endswith('semilog') else 'linear')
                lower, upper = axis.get_ylim()
                for line in axis.lines:
                    values = np.ma.array(line.get_ydata()).compressed()
                    self.assertTrue(np.all(values >= lower))
                    self.assertTrue(np.all(values <= upper))
            checked.append(name)
        with mock.patch.object(analysis.original, 'save_figure', side_effect=inspect):
            analysis.plot_comparison(arrays, Path('/tmp'))
        self.assertEqual(checked, ['method_comparison', 'method_comparison_semilog'])


if __name__ == '__main__':
    unittest.main()
