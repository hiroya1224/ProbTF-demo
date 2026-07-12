import math

import numpy as np
import pytest

from probtf.geometry import quat_to_rotmat
from probtf.imu_relative_pose import (
    ImuRelativePoseEstimator,
    RecursiveGaussianLeastSquares,
    rigid_point_acceleration_operator,
    vector_alignment_bingham,
)
from probtf.models import ImuKinematics


def make_observation(frame_id, omega, alpha, force, stamp=None, variance=1e-4):
    covariance = np.eye(3, dtype=float) * variance
    return ImuKinematics(
        frame_id=frame_id,
        angular_velocity=omega,
        angular_acceleration=alpha,
        specific_force=force,
        angular_velocity_covariance=covariance,
        angular_acceleration_covariance=covariance,
        specific_force_covariance=covariance,
        stamp=stamp,
    )


def test_vector_alignment_likelihood_is_antipodally_symmetric():
    parameter = vector_alignment_bingham(
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        np.eye(3) * 0.01,
        np.eye(3) * 0.01,
    )
    quaternion = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])

    assert quaternion @ parameter @ quaternion == pytest.approx(0.0, abs=1e-10)
    assert (-quaternion) @ parameter @ (-quaternion) == pytest.approx(0.0, abs=1e-10)
    assert np.max(np.linalg.eigvalsh(parameter)) == pytest.approx(0.0, abs=1e-10)


def test_recursive_least_squares_recovers_centripetal_offset():
    estimate = RecursiveGaussianLeastSquares(prior_variance=1e5)
    expected_position = np.array([0.2, -0.1, 0.05])
    covariance = np.eye(3) * 1e-5

    for omega in np.eye(3):
        coefficient = rigid_point_acceleration_operator(omega, np.zeros(3))
        estimate.update(coefficient, coefficient @ expected_position, covariance)

    np.testing.assert_allclose(estimate.mean, expected_position, atol=1e-7)
    assert np.max(np.linalg.eigvalsh(estimate.covariance)) < 1e-4


def test_imu_estimator_recovers_synthetic_identity_relative_pose():
    estimator = ImuRelativePoseEstimator(
        "imu_parent",
        "imu_child",
        position_forgetting_factor=0.9,
        integration_steps=60,
    )
    expected_position = np.array([0.18, -0.07, 0.04])
    gravity = np.array([0.0, 0.0, 9.81])
    stamp = 0.0

    excitation = (
        np.array([1.2, 0.0, 0.0]),
        np.array([0.0, 1.1, 0.0]),
        np.array([0.0, 0.0, 1.3]),
        np.array([0.8, -0.6, 0.4]),
    )
    result = None
    for _ in range(12):
        for omega in excitation:
            operator = rigid_point_acceleration_operator(omega, np.zeros(3))
            child_force = gravity + operator @ expected_position
            parent = make_observation(
                "imu_parent",
                omega,
                np.zeros(3),
                gravity,
                stamp=stamp,
                variance=2e-4,
            )
            child = make_observation(
                "imu_child",
                omega,
                np.zeros(3),
                child_force,
                stamp=stamp,
                variance=2e-4,
            )
            result = estimator.update(parent, child)
            stamp += 0.01

    np.testing.assert_allclose(result.position_mean, expected_position, atol=7e-3)
    mode_rotation = quat_to_rotmat(result.orientation_mode_wxyz)
    np.testing.assert_allclose(mode_rotation, np.eye(3), atol=2e-2)
    assert result.orientation.fourth_moment.shape == (4, 4, 4, 4)
    assert result.orientation.second_moment.shape == (4, 4)


def test_registered_joint_offsets_must_be_paired():
    estimator = ImuRelativePoseEstimator("parent", "child", integration_steps=40)
    with pytest.raises(ValueError, match="together"):
        estimator.register_joint_geometry([0.1, 0.0, 0.0], None)


def test_estimator_rejects_unsynchronized_samples():
    estimator = ImuRelativePoseEstimator("parent", "child", integration_steps=40)
    parent = make_observation("parent", [1, 0, 0], [0, 0, 0], [0, 0, 9.81], stamp=1.0)
    child = make_observation("child", [1, 0, 0], [0, 0, 0], [0, 0, 9.81], stamp=1.2)

    with pytest.raises(ValueError, match="synchronized"):
        estimator.update(parent, child)
