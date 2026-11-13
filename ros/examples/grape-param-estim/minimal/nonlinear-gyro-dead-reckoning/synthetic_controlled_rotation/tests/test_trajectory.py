"""Analytic kinematics and generation-only noise provenance checks."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

PATH = Path(__file__).resolve().parents[1]/'trajectory.py'
SPEC = importlib.util.spec_from_file_location('synthetic_trajectory_tested', PATH)
TRAJECTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAJECTORY)


class SyntheticTrajectoryTests(unittest.TestCase):
    def test_body_rate_matches_rotation_matrix_derivative(self):
        times = np.linspace(.1, 4.9, 47)
        step = 1e-6
        value = TRAJECTORY.trajectory(times)
        minus = Rotation.from_quat(TRAJECTORY.trajectory(times-step)['quaternion_xyzw']).as_matrix()
        plus = Rotation.from_quat(TRAJECTORY.trajectory(times+step)['quaternion_xyzw']).as_matrix()
        rotation = Rotation.from_quat(value['quaternion_xyzw']).as_matrix()
        skew = np.einsum('tji,tjk->tik', rotation, (plus-minus)/(2*step))
        numerical = np.column_stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]))
        np.testing.assert_allclose(value['omega_body_rad_s'], numerical, atol=1e-9, rtol=2e-8)

    def test_body_acceleration_matches_rate_derivative(self):
        times = np.linspace(.1, 4.9, 49)
        step = 1e-6
        numerical = (TRAJECTORY.trajectory(times+step)['omega_body_rad_s']
                     -TRAJECTORY.trajectory(times-step)['omega_body_rad_s'])/(2*step)
        np.testing.assert_allclose(TRAJECTORY.trajectory(times)['alpha_body_rad_s2'], numerical,
                                   atol=2e-9, rtol=2e-8)

    def test_exact_torque_balance_with_three_distinct_physical_inertias(self):
        value = TRAJECTORY.trajectory(np.linspace(0., 5., 201))
        inertia = TRAJECTORY.INERTIA
        eigenvalues = np.linalg.eigvalsh(inertia)
        self.assertEqual(len(np.unique(eigenvalues)), 3)
        self.assertGreater(eigenvalues[:2].sum(), eigenvalues[-1])
        required = value['alpha_body_rad_s2']@inertia.T+np.cross(
            value['omega_body_rad_s'], value['omega_body_rad_s']@inertia.T)
        np.testing.assert_array_equal(value['torque_body_nm'], required)
        self.assertGreater(np.sqrt(np.mean(value['gyroscopic_torque_nm']**2)), .01)

    def test_generated_sensor_values_preserve_one_realization(self):
        first, metadata = TRAJECTORY.generate_dataset()
        second, _ = TRAJECTORY.generate_dataset()
        for key in first:
            if key == 'torque_body_nm':
                # BLAS evaluation can differ by one ulp with allocation
                # alignment. Keep the realized random sensor sequence exact;
                # this bound applies only to derived deterministic torques.
                np.testing.assert_allclose(first[key], second[key],
                    rtol=8*np.finfo(float).eps, atol=8*np.finfo(float).eps)
            else:
                np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first['gyro'], first['gyro_true_body_rad_s']
                                      +first['gyro_bias_rad_s']+first['gyro_realized_noise_rad_s'])
        np.testing.assert_array_equal(first['generator_gyro_covariance_rad2_s2'],
            np.eye(3)*np.deg2rad(.2)**2)
        np.testing.assert_array_equal(first['imu_source_index'], np.arange(1001))
        np.testing.assert_array_equal(first['imu_stamp_ns'], np.arange(1001)*5_000_000)
        self.assertEqual(len(first['torque_time']), 5001)
        self.assertTrue(metadata['random_sensor_noise_generation'])
        self.assertFalse(metadata['monte_carlo_state_inference'])

    def test_scalar_and_array_queries_share_the_same_motion(self):
        scalar = TRAJECTORY.trajectory(1.2)
        array = TRAJECTORY.trajectory(np.array([1.2]))
        for key in scalar:
            np.testing.assert_array_equal(scalar[key], array[key][0])
        q0 = TRAJECTORY.trajectory(0.)['quaternion_xyzw']
        data, _ = TRAJECTORY.generate_dataset()
        np.testing.assert_array_equal(data['initial_quaternion_wxyz'], q0[[3, 0, 1, 2]])

    def test_nonuniform_duration_preserves_exact_reference_and_torque_endpoint(self):
        data, _ = TRAJECTORY.generate_dataset(duration=1.2345)
        self.assertEqual(data['reference_time'][-1], 1.2345)
        self.assertEqual(data['torque_time'][-1], 1.2345)
        self.assertLessEqual(data['imu_time'][-1], 1.2345)


if __name__ == '__main__':
    unittest.main()
