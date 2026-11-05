"""Frame, sample-clock and independence checks for the synthetic IMU control."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ros/examples/probtf_orientation_demo/scripts'))
from translational_madgwick import MadgwickBaseline, DEFAULT_BETA
from compare_madgwick_rolling import madgwick_update


def wxyz(rotation):
    return rotation.as_quat()[[3, 0, 1, 2]]


def rotation(quaternion):
    return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]])


class TranslationalMadgwickTests(unittest.TestCase):
    def test_known_world_field_with_nonzero_east_keeps_static_tilt(self):
        truth = Rotation.from_euler('xyz', [24., -19., 57.], degrees=True)
        reference = np.array([.3, -.4, -.7])
        state = MadgwickBaseline(wxyz(truth), reference)
        accel = truth.inv().apply([0., 0., 9.80665])
        magnetic = truth.inv().apply(reference)
        for _ in range(20):
            output = state.update(np.zeros(3), accel, .005, magnetic)
        self.assertLess((truth.inv()*rotation(output)).magnitude(), 2e-15)

    def test_six_axis_recovers_pitch_from_stationary_specific_force(self):
        truth = Rotation.from_euler('xyz', [20., -17., 0.], degrees=True)
        state = MadgwickBaseline([1., 0., 0., 0.])
        accel = truth.inv().apply([0., 0., 9.80665])
        for _ in range(1600):
            output = state.update(np.zeros(3), accel, .005)
        pitch = rotation(output).as_euler('xyz', degrees=True)[1]
        self.assertLess(abs(pitch+17.), .12)

    def test_absent_accel_is_gyro_only_without_truth_input(self):
        state = MadgwickBaseline([1., 0., 0., 0.])
        omega = np.array([0., .4, 0.])
        intervals = [.005, .008, .004, .011]
        for dt in intervals:
            output = state.update(omega, None, dt)
        expected_angle = sum(2*np.arctan(.4*dt/2) for dt in intervals)
        np.testing.assert_allclose(rotation(output).as_rotvec(),
                                   [0., expected_angle, 0.], atol=2e-16)

    def test_wrapper_reuses_the_original_imu_update(self):
        q = wxyz(Rotation.from_euler('xyz', [8., -14., 22.], degrees=True))
        gyro = [.13, -.2, .31]
        accel = [.2, -.4, 9.7]
        state = MadgwickBaseline(q)
        expected = madgwick_update(q, gyro, accel, .006, beta=DEFAULT_BETA)
        np.testing.assert_allclose(state.update(gyro, accel, .006), expected, atol=3e-16)

    def test_magnetic_missing_tick_is_the_standard_imu_branch(self):
        q = wxyz(Rotation.from_euler('xyz', [8., -14., 22.], degrees=True))
        ordinary = MadgwickBaseline(q)
        heading = MadgwickBaseline(q, [.3, -.4, -.7])
        gyro, accel = [.13, -.2, .31], [.2, -.4, 9.7]
        a = ordinary.update(gyro, accel, .006)
        b = heading.update(gyro, accel, .006, magnetic=None)
        np.testing.assert_allclose(a, b, atol=4e-16)

    def test_sensor_magnitude_does_not_change_normalized_vector_control(self):
        q = wxyz(Rotation.from_euler('xyz', [18., -4., 32.], degrees=True))
        a = MadgwickBaseline(q, [.3, -.4, -.7])
        b = MadgwickBaseline(q, [.3, -.4, -.7])
        gyro = [.1, -.2, .3]
        accel, magnetic = np.array([.2, -.3, 9.7]), np.array([.4, -.3, -.8])
        np.testing.assert_allclose(a.update(gyro, accel, .005, magnetic),
                                   b.update(gyro, 4*accel, .005, 3*magnetic), atol=3e-16)

    def test_inputs_need_no_future_or_true_attitude_sequence(self):
        initial = np.array([1., 0., 0., 0.])
        state = MadgwickBaseline(initial)
        initial[:] = [0., 1., 0., 0.]
        output = state.update([0., 0., 0.], [0., 0., 9.80665], .005)
        np.testing.assert_array_equal(output, [1., 0., 0., 0.])
        output[:] = 0.
        np.testing.assert_array_equal(state.quaternion_wxyz, [1., 0., 0., 0.])

    def test_marg_requires_explicit_world_reference(self):
        state = MadgwickBaseline([1., 0., 0., 0.])
        with self.assertRaises(ValueError):
            state.update([0., 0., 0.], [0., 0., 9.8], .005, [.5, 0., -.8])


if __name__ == '__main__':
    unittest.main()
