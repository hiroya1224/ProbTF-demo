import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from grape_param_estim.kinematics import (
    KinematicsConfig,
    derivative_noise_estimate,
    estimate_kinematics,
)


class KinematicsTest(unittest.TestCase):
    def test_sinusoidal_translation_and_rotation_derivatives_are_accurate(self):
        timestamps = np.linspace(0.0, 8.0, 801)
        frequency = 0.8
        position_amplitude = np.array([0.4, 0.25, 0.15])
        positions = np.column_stack(
            [
                position_amplitude[0] * np.sin(frequency * timestamps),
                position_amplitude[1] * np.cos(frequency * timestamps),
                position_amplitude[2] * np.sin(frequency * timestamps + 0.3),
            ]
        )
        yaw_amplitude = 0.35
        yaw = yaw_amplitude * np.sin(frequency * timestamps)
        quaternions_xyzw = Rotation.from_euler("z", yaw).as_quat()
        config = KinematicsConfig(
            window_length=21,
            polynomial_order=4,
            position_sigma=0.01,
            orientation_sigma=np.deg2rad(1.0),
        )

        result = estimate_kinematics(
            timestamps,
            positions,
            quaternions_xyzw,
            config,
        )
        valid = result.valid_mask
        expected_acceleration = np.column_stack(
            [
                -position_amplitude[0]
                * frequency**2
                * np.sin(frequency * timestamps),
                -position_amplitude[1]
                * frequency**2
                * np.cos(frequency * timestamps),
                -position_amplitude[2]
                * frequency**2
                * np.sin(frequency * timestamps + 0.3),
            ]
        )
        expected_omega = np.zeros((timestamps.size, 3))
        expected_omega[:, 2] = (
            yaw_amplitude * frequency * np.cos(frequency * timestamps)
        )
        expected_alpha = np.zeros((timestamps.size, 3))
        expected_alpha[:, 2] = (
            -yaw_amplitude * frequency**2 * np.sin(frequency * timestamps)
        )

        np.testing.assert_allclose(
            result.linear_acceleration_world[valid],
            expected_acceleration[valid],
            rtol=0.0,
            atol=2.5e-4,
        )
        np.testing.assert_allclose(
            result.angular_velocity_body[valid],
            expected_omega[valid],
            rtol=0.0,
            atol=2.0e-4,
        )
        np.testing.assert_allclose(
            result.angular_acceleration_body[valid],
            expected_alpha[valid],
            rtol=0.0,
            atol=4.0e-4,
        )

        rotations = Rotation.from_quat(
            result.quaternion_xyzw_smoothed[valid]
        ).as_matrix()
        expected_specific = np.einsum(
            "nji,nj->ni",
            rotations,
            expected_acceleration[valid] - np.asarray(config.gravity_world),
        )
        np.testing.assert_allclose(
            result.specific_acceleration_body[valid],
            expected_specific,
            rtol=0.0,
            atol=2.5e-4,
        )

    def test_irregular_timestamps_and_quaternion_sign_flips_are_supported(self):
        rng = np.random.default_rng(14)
        increments = 0.01 + rng.uniform(-0.001, 0.001, size=500)
        timestamps = np.cumsum(increments)
        positions = np.column_stack(
            [timestamps**2, np.zeros_like(timestamps), np.zeros_like(timestamps)]
        )
        yaw_rate = 0.2
        quaternions = Rotation.from_euler("z", yaw_rate * timestamps).as_quat()
        quaternions[::7] *= -1.0

        result = estimate_kinematics(
            timestamps,
            positions,
            quaternions,
            KinematicsConfig(window_length=17, polynomial_order=3),
        )
        valid = result.valid_mask
        np.testing.assert_allclose(
            result.linear_acceleration_world[valid, 0],
            2.0,
            rtol=0.0,
            atol=2.0e-3,
        )
        np.testing.assert_allclose(
            result.angular_velocity_body[valid, 2],
            yaw_rate,
            rtol=0.0,
            atol=3.0e-4,
        )
        np.testing.assert_allclose(
            result.angular_acceleration_body[valid],
            0.0,
            rtol=0.0,
            atol=3.0e-3,
        )

    def test_edge_margin_is_explicit_and_invalid_values_are_nan(self):
        timestamps = np.linspace(0.0, 2.0, 101)
        positions = np.zeros((timestamps.size, 3))
        quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (timestamps.size, 1))
        config = KinematicsConfig(window_length=15, polynomial_order=3)
        result = estimate_kinematics(timestamps, positions, quaternions, config)

        self.assertEqual(result.edge_margin, config.window_length // 2 + 2)
        self.assertFalse(np.any(result.valid_mask[: result.edge_margin]))
        self.assertFalse(np.any(result.valid_mask[-result.edge_margin :]))
        self.assertTrue(
            np.all(result.valid_mask[result.edge_margin : -result.edge_margin])
        )
        self.assertTrue(
            np.all(np.isnan(result.specific_acceleration_body[~result.valid_mask]))
        )
        np.testing.assert_allclose(
            result.specific_acceleration_body[result.valid_mask],
            np.tile(
                [0.0, 0.0, 9.80665],
                (np.count_nonzero(result.valid_mask), 1),
            ),
            rtol=0.0,
            atol=1e-12,
        )

    def test_derivative_noise_scales_linearly_with_mocap_sigma(self):
        first = derivative_noise_estimate(
            0.01,
            KinematicsConfig(
                window_length=15,
                polynomial_order=3,
                position_sigma=0.01,
                orientation_sigma=0.02,
            ),
        )
        second = derivative_noise_estimate(
            0.01,
            KinematicsConfig(
                window_length=15,
                polynomial_order=3,
                position_sigma=0.02,
                orientation_sigma=0.04,
            ),
        )
        self.assertAlmostEqual(
            second.linear_acceleration_std,
            2.0 * first.linear_acceleration_std,
        )
        self.assertAlmostEqual(
            second.angular_velocity_std,
            2.0 * first.angular_velocity_std,
        )
        self.assertAlmostEqual(
            second.angular_acceleration_std,
            2.0 * first.angular_acceleration_std,
        )
        self.assertEqual(first.specific_acceleration_covariance.shape, (3, 3))

    def test_one_degree_tangent_noise_does_not_explode_angular_acceleration(self):
        timestamps = np.arange(1001, dtype=float) / 50.0
        euler = np.column_stack(
            (
                0.3 * np.sin(1.2 * timestamps),
                0.25 * np.sin(1.6 * timestamps + 0.4),
                0.35 * np.sin(2.0 * timestamps + 0.8),
            )
        )
        exact_quaternion = Rotation.from_euler("xyz", euler).as_quat()
        rng = np.random.default_rng(7)
        noisy_quaternion = (
            Rotation.from_quat(exact_quaternion)
            * Rotation.from_rotvec(
                rng.normal(0.0, np.deg2rad(1.0), size=(timestamps.size, 3))
            )
        ).as_quat()
        positions = np.zeros((timestamps.size, 3))
        config = KinematicsConfig(window_length=51, polynomial_order=3)
        exact = estimate_kinematics(timestamps, positions, exact_quaternion, config)
        noisy = estimate_kinematics(timestamps, positions, noisy_quaternion, config)
        valid = exact.valid_mask & noisy.valid_mask
        acceleration_error = (
            noisy.angular_acceleration_body[valid]
            - exact.angular_acceleration_body[valid]
        )
        self.assertLess(float(np.max(np.std(acceleration_error, axis=0))), 0.08)

    def test_invalid_pose_inputs_are_rejected(self):
        cases = (
            (np.arange(20, dtype=float), "zero quaternions"),
            (np.r_[np.arange(19, dtype=float), 18.0], "strictly increasing"),
        )
        for timestamps, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                positions = np.zeros((20, 3))
                quaternions = np.tile(
                    [0.0, 0.0, 0.0, 1.0],
                    (20, 1),
                )
                if len(np.unique(timestamps)) == len(timestamps):
                    quaternions[3] = 0.0
                with self.assertRaisesRegex(ValueError, expected_message):
                    estimate_kinematics(timestamps, positions, quaternions)


if __name__ == "__main__":
    unittest.main()
