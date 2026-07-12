import numpy as np
import pytest

from probtf_estimators.imu_preprocessing import ImuKinematicsPreprocessor
from probtf_estimators.imu_kinematics import ImuKinematics


def test_polynomial_preprocessor_recovers_value_and_derivative_at_latest_time():
    processor = ImuKinematicsPreprocessor(
        "imu",
        window_size=7,
        polynomial_order=2,
        minimum_samples=5,
    )
    covariance = np.eye(3) * 1e-5
    result = None
    for stamp in np.linspace(1.0, 1.6, 7):
        angular_velocity = np.array(
            [
                0.5 + 2.0 * stamp + 0.3 * stamp ** 2,
                -0.2 + 0.4 * stamp,
                1.5,
            ]
        )
        force = np.array([stamp, stamp ** 2, 9.81])
        result = processor.update(
            stamp,
            angular_velocity,
            force,
            covariance,
            covariance,
        )

    expected_velocity = np.array([0.5 + 2.0 * 1.6 + 0.3 * 1.6 ** 2, -0.2 + 0.4 * 1.6, 1.5])
    expected_acceleration = np.array([2.0 + 0.6 * 1.6, 0.4, 0.0])
    np.testing.assert_allclose(result.angular_velocity, expected_velocity, atol=1e-10)
    np.testing.assert_allclose(result.angular_acceleration, expected_acceleration, atol=1e-10)
    np.testing.assert_allclose(result.specific_force, [1.6, 1.6 ** 2, 9.81], atol=1e-10)


def test_preprocessor_waits_for_minimum_samples_and_rejects_time_reversal():
    processor = ImuKinematicsPreprocessor("imu", window_size=5, minimum_samples=4)
    covariance = np.eye(3)
    for stamp in (0.0, 0.1, 0.2):
        assert processor.update(stamp, np.ones(3), np.ones(3), covariance, covariance) is None

    with pytest.raises(ValueError, match="strictly increasing"):
        processor.update(0.2, np.ones(3), np.ones(3), covariance, covariance)


def test_preprocessor_rejects_invalid_covariance():
    processor = ImuKinematicsPreprocessor("imu")
    with pytest.raises(ValueError, match="positive semidefinite"):
        processor.update(
            0.0,
            np.ones(3),
            np.ones(3),
            np.diag([1.0, 1.0, -1.0]),
            np.eye(3),
        )


def test_imu_kinematics_rejects_non_psd_covariance():
    with pytest.raises(ValueError, match="positive semidefinite"):
        ImuKinematics(
            "imu",
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.diag([1.0, 1.0, -0.1]),
            np.eye(3),
            np.eye(3),
        )
