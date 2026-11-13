"""Exact physical rate scaling with consistent quaternion reconstruction."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from trajectory import trajectory, generate_dataset, INERTIA
from scaled_motion import scaled_trajectory, generate_scaled_dataset


class ScaledMotionTests(unittest.TestCase):
    def test_exact_body_rate_and_acceleration_scale(self):
        times = np.linspace(0., 5., 113)
        original = trajectory(times, 'three_frequency')
        scaled = scaled_trajectory(times)
        # Multiplication by binary 1/75 and division by 75 differ by at most
        # an ulp. This checks physical scaling, not floating operation order.
        np.testing.assert_allclose(scaled['omega_body_rad_s'], original['omega_body_rad_s']/75.,
                                   atol=1e-17, rtol=4*np.finfo(float).eps)
        np.testing.assert_allclose(scaled['alpha_body_rad_s2'], original['alpha_body_rad_s2']/75.,
                                   atol=1e-17, rtol=4*np.finfo(float).eps)
        np.testing.assert_allclose(scaled['gyroscopic_torque_nm'], original['gyroscopic_torque_nm']/75.**2,
                                   atol=1e-19, rtol=1e-14)
        np.testing.assert_allclose(scaled['torque_body_nm'],
            scaled['alpha_body_rad_s2']@INERTIA.T+scaled['gyroscopic_torque_nm'], atol=1e-17)

    def test_integrated_rotation_and_euler_derivatives_match_scaled_rates(self):
        times = np.linspace(.05, 4.95, 73)
        delta = 1e-5
        center = scaled_trajectory(times)
        before, after = scaled_trajectory(times-delta), scaled_trajectory(times+delta)
        rotations = Rotation.from_quat(center['quaternion_xyzw']).as_matrix()
        rotation_dot = (Rotation.from_quat(after['quaternion_xyzw']).as_matrix()
                      -Rotation.from_quat(before['quaternion_xyzw']).as_matrix())/(2*delta)
        skew = np.einsum('tji,tjk->tik', rotations, rotation_dot)
        numerical = .5*np.column_stack((skew[:, 2, 1]-skew[:, 1, 2],
                                       skew[:, 0, 2]-skew[:, 2, 0],
                                       skew[:, 1, 0]-skew[:, 0, 1]))
        np.testing.assert_allclose(center['omega_body_rad_s'], numerical, atol=2e-9, rtol=2e-7)
        np.testing.assert_allclose(center['rpy_velocity_rad_s'],
            (after['rpy_rad']-before['rpy_rad'])/(2*delta), atol=2e-9, rtol=2e-7)
        np.testing.assert_allclose(center['rpy_acceleration_rad_s2'],
            (after['rpy_velocity_rad_s']-before['rpy_velocity_rad_s'])/(2*delta), atol=2e-9, rtol=2e-7)

    def test_sensor_noise_bias_clocks_and_initial_pose_unchanged(self):
        original, _ = generate_dataset(motion_profile='three_frequency')
        scaled, metadata = generate_scaled_dataset()
        for key in ('time', 'imu_time', 'imu_stamp_ns', 'imu_record_ns', 'imu_source_index',
                    'gyro_realized_noise_rad_s', 'gyro_bias_rad_s',
                    'generator_gyro_covariance_rad2_s2', 'inertia_body_kg_m2',
                    'initial_quaternion_wxyz', 'torque_time'):
            np.testing.assert_array_equal(scaled[key], original[key])
        np.testing.assert_allclose(scaled['gyro_true_body_rad_s'], original['gyro_true_body_rad_s']/75.,
                                   atol=1e-17, rtol=4*np.finfo(float).eps)
        np.testing.assert_array_equal(scaled['gyro'], scaled['gyro_true_body_rad_s']
                                     +scaled['gyro_bias_rad_s']+scaled['gyro_realized_noise_rad_s'])
        self.assertEqual(metadata['angular_rate_scale'], 1./75.)
        self.assertFalse(metadata['source_coefficients_are_new_euler_amplitudes'])
        self.assertNotIn('motion_amplitude_deg', scaled)
        self.assertEqual(scaled['source_motion_amplitude_deg'].shape, (3, 3))

    def test_initial_pose_and_scalar_queries(self):
        np.testing.assert_array_equal(scaled_trajectory(0.)['quaternion_xyzw'],
                                      trajectory(0., 'three_frequency')['quaternion_xyzw'])
        scalar, vector = scaled_trajectory(1.25), scaled_trajectory(np.array([1.25]))
        for key in scalar:
            np.testing.assert_array_equal(scalar[key], vector[key][0])
        # A unit scaling recovers the source motion, not a time-rescaled one.
        times = np.linspace(0., 5., 37)
        recovered = scaled_trajectory(times, angular_rate_scale=1.)
        source = trajectory(times, 'three_frequency')
        error = (Rotation.from_quat(source['quaternion_xyzw']).inv()
                 *Rotation.from_quat(recovered['quaternion_xyzw'])).magnitude()
        self.assertLess(error.max(), 1e-9)


if __name__ == '__main__':
    unittest.main()
