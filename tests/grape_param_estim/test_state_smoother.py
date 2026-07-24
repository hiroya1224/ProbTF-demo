import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from grape_param_estim.state_smoother import (
    SmootherConfig,
    TrajectoryObservations,
    smooth_trajectory,
)


def synthetic_observations(seed=5, dropout=(3.0, 4.5), future_shift=False):
    rng = np.random.default_rng(seed)
    imu_times = np.arange(0.0, 8.0 + 1.0e-9, 0.02)
    positions = np.zeros((imu_times.size, 3))
    velocities = np.zeros_like(positions)
    quaternions = np.empty((imu_times.size, 4))
    quaternions[0] = [0.0, 0.0, 0.0, 1.0]
    accelerometer = np.empty_like(positions)
    gyro = np.empty_like(positions)
    gravity = np.array([0.0, 0.0, -9.80665])
    accel_bias = np.array([0.04, -0.03, 0.06])
    gyro_bias = np.array([0.006, -0.004, 0.008])

    for index, stamp in enumerate(imu_times):
        acceleration_world = np.array(
            [
                0.35 * np.sin(0.9 * stamp),
                -0.25 * np.cos(0.7 * stamp),
                0.12 * np.sin(1.1 * stamp),
            ]
        )
        angular_velocity = np.array(
            [
                0.20 * np.sin(0.55 * stamp),
                0.16 * np.cos(0.45 * stamp),
                0.65 + 0.10 * np.sin(0.8 * stamp),
            ]
        )
        rotation = Rotation.from_quat(quaternions[index])
        accelerometer[index] = (
            rotation.inv().apply(acceleration_world - gravity)
            + accel_bias
            + rng.normal(0.0, 0.025, 3)
        )
        gyro[index] = angular_velocity + gyro_bias + rng.normal(0.0, 0.002, 3)
        if index + 1 < imu_times.size:
            delta = imu_times[index + 1] - stamp
            positions[index + 1] = (
                positions[index]
                + velocities[index] * delta
                + 0.5 * acceleration_world * delta * delta
            )
            velocities[index + 1] = velocities[index] + acceleration_world * delta
            quaternions[index + 1] = (
                rotation * Rotation.from_rotvec(angular_velocity * delta)
            ).as_quat()

    mocap_indices = np.arange(0, imu_times.size, 5)
    mocap_times = imu_times[mocap_indices]
    mocap_positions = positions[mocap_indices] + rng.normal(
        0.0, 0.008, (mocap_indices.size, 3)
    )
    mocap_quaternions = (
        Rotation.from_quat(quaternions[mocap_indices])
        * Rotation.from_rotvec(
            rng.normal(0.0, np.deg2rad(0.5), (mocap_indices.size, 3))
        )
    ).as_quat()
    # Quaternion sign is representational and must not cause a wrap.
    mocap_quaternions[::9] *= -1.0
    mocap_valid = ~(
        (mocap_times > float(dropout[0])) & (mocap_times < float(dropout[1]))
    )
    if future_shift:
        mocap_positions[mocap_times > 5.0] += 100.0
    return (
        TrajectoryObservations(
            mocap_times=mocap_times,
            mocap_positions_world=mocap_positions,
            mocap_quaternions_xyzw=mocap_quaternions,
            imu_times=imu_times,
            accelerometer_body=accelerometer,
            gyro_body=gyro,
            mocap_valid_mask=mocap_valid,
        ),
        imu_times,
        positions,
        quaternions,
    )


class StateSmootherTests(unittest.TestCase):
    def config(self, samples=12):
        return SmootherConfig(
            mocap_position_sigma=0.01,
            mocap_orientation_sigma=np.deg2rad(1.0),
            accelerometer_noise_sigma=0.08,
            gyro_noise_sigma=np.deg2rad(0.3),
            accelerometer_bias_random_walk_sigma=0.01,
            gyro_bias_random_walk_sigma=np.deg2rad(0.02),
            mocap_nis_gate=100.0,
            trajectory_sample_count=samples,
            seed=11,
        )

    def test_offline_rts_tracks_motion_through_mocap_dropout(self):
        observations, truth_times, truth_positions, _ = synthetic_observations()
        posterior = smooth_trajectory(observations, self.config())
        expected = np.column_stack(
            [
                np.interp(
                    posterior.timestamps, truth_times, truth_positions[:, column]
                )
                for column in range(3)
            ]
        )
        rmse = np.sqrt(np.mean((posterior.position_world - expected) ** 2))
        self.assertLess(rmse, 0.04)
        self.assertTrue(posterior.is_smoothed)
        self.assertEqual(posterior.sample_position_world.shape[0], 12)
        self.assertEqual(posterior.sample_ids.tolist(), list(range(12)))
        np.testing.assert_allclose(np.sum(posterior.sample_weights), 1.0)

        position_variance = np.trace(posterior.covariance[:, :3, :3], axis1=1, axis2=2)
        dropout = (posterior.timestamps > 3.2) & (posterior.timestamps < 4.3)
        observed = (posterior.timestamps > 1.0) & (posterior.timestamps < 2.5)
        self.assertGreater(
            float(np.max(position_variance[dropout])),
            float(np.median(position_variance[observed])),
        )

    def test_prefix_filter_does_not_use_future_measurements(self):
        baseline, _, _, _ = synthetic_observations()
        changed_future, _, _, _ = synthetic_observations(future_shift=True)
        first = smooth_trajectory(
            baseline, self.config(samples=4), online_prefix=True, cutoff=4.0
        )
        second = smooth_trajectory(
            changed_future,
            self.config(samples=4),
            online_prefix=True,
            cutoff=4.0,
        )
        self.assertFalse(first.is_smoothed)
        self.assertEqual(
            first.sampling_approximation, "shared_whitened_filter_marginals"
        )
        np.testing.assert_array_equal(first.timestamps, second.timestamps)
        np.testing.assert_allclose(first.position_world, second.position_world)
        np.testing.assert_allclose(first.covariance, second.covariance)
        np.testing.assert_allclose(
            first.sample_position_world, second.sample_position_world
        )

    def test_orientation_sign_wrap_and_aggressive_rotation_stay_finite(self):
        observations, _, _, truth_quaternions = synthetic_observations()
        posterior = smooth_trajectory(observations, self.config(samples=6))
        self.assertTrue(np.all(np.isfinite(posterior.quaternion_xyzw)))
        self.assertTrue(np.all(np.isfinite(posterior.angular_velocity_body)))
        self.assertTrue(np.all(np.linalg.norm(posterior.quaternion_xyzw, axis=1) > 0.999999))
        truth_at_output = Rotation.from_quat(truth_quaternions)(
            # This branch is intentionally unreachable; Rotation is not
            # callable.  Interpolation below uses Slerp in a version-portable
            # way and exercises more than quaternion component comparison.
        ) if False else None
        del truth_at_output
        from scipy.spatial.transform import Slerp

        expected = Slerp(
            observations.imu_times, Rotation.from_quat(truth_quaternions)
        )(posterior.timestamps)
        error = (
            expected.inv() * Rotation.from_quat(posterior.quaternion_xyzw)
        ).magnitude()
        self.assertLess(float(np.sqrt(np.mean(error * error))), np.deg2rad(4.0))

    def test_invalid_saturated_imu_and_mocap_outlier_are_masked(self):
        observations, _, _, _ = synthetic_observations()
        acceleration = np.array(observations.accelerometer_body, copy=True)
        acceleration[100] = [1000.0, 0.0, 0.0]
        positions = np.array(observations.mocap_positions_world, copy=True)
        positions[30] += 30.0
        modified = TrajectoryObservations(
            mocap_times=observations.mocap_times,
            mocap_positions_world=positions,
            mocap_quaternions_xyzw=observations.mocap_quaternions_xyzw,
            imu_times=observations.imu_times,
            accelerometer_body=acceleration,
            gyro_body=observations.gyro_body,
            mocap_valid_mask=observations.mocap_valid_mask,
        )
        posterior = smooth_trajectory(modified, self.config(samples=0))
        imu_event = np.flatnonzero(np.isclose(posterior.timestamps, observations.imu_times[100]))
        self.assertEqual(imu_event.size, 1)
        self.assertFalse(posterior.imu_used[imu_event[0]])
        mocap_event = np.flatnonzero(
            np.isclose(posterior.timestamps, observations.mocap_times[30])
        )
        self.assertEqual(mocap_event.size, 1)
        self.assertTrue(posterior.mocap_rejected[mocap_event[0]])


if __name__ == "__main__":
    unittest.main()
