"""Regression against the rigid-pair equations, rather than a point-pose fit."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from probtf.bingham import bingham_second_moment, canonical_bingham_parameter, rotation_first_moment
from probtf.distributions import OrientationKind, RepresentativeKind
from probtf.geometry import quat_to_rotmat, rpy_to_quat
from probtf_estimators.imu_extended_leastsq import (
    ExtendedLeastSquaresImuRelativePoseEstimator,
    expected_acceleration_operator,
    gyro_residual_covariance,
)
from probtf_estimators.imu_kinematics import ImuKinematics
from probtf_estimators.imu_relative_pose import skew


def observation(frame, stamp, omega, alpha, force, covariance=None):
    covariance = np.diag([0.004, 0.006, 0.009]) if covariance is None else covariance
    return ImuKinematics(frame, omega, alpha, force, covariance, np.eye(3) * 0.02,
                         np.eye(3) * 0.03, stamp)


def source_noise_matrix():
    # Nonzero entries of NoiseCovarianceHelper.get_Nmat at source 291f66f.
    matrix = np.zeros((4, 24))
    entries = (
        ((1, -1), (6, -1), (11, -1), (13, 1), (18, 1), (23, 1)),
        ((0, 1), (7, -1), (10, 1), (12, -1), (19, -1), (22, 1)),
        ((3, 1), (4, 1), (9, -1), (15, 1), (16, -1), (21, -1)),
        ((2, -1), (5, 1), (8, 1), (14, -1), (17, 1), (20, -1)),
    )
    for row, values in enumerate(entries):
        for column, value in values:
            matrix[row, column] = value
    return matrix


@pytest.mark.parametrize("use_child", [False, True])
def test_matches_source_bingham_and_gaussian_recurrence(use_child):
    estimator = ExtendedLeastSquaresImuRelativePoseEstimator(
        "p", "c", use_child_gyro=use_child, rotation_forgetting_factor=0.98,
        position_observation_weight=0.8, prior_position_variance=1e6,
    )
    parameter = np.zeros((4, 4))
    dimension = 6 if use_child else 3
    mean = np.zeros(dimension)
    covariance = np.eye(dimension) * 1e6
    precision = None
    rotation = quat_to_rotmat(rpy_to_quat(0.3, -0.5, 0.7))
    position = np.array([0.19, -0.08, 0.035])
    for index in range(12):
        omega = np.array([0.5 + index * 0.05, np.sin(index), np.cos(index)])
        alpha = np.array([0.05, np.cos(index), -np.sin(index)])
        force = np.array([0.3 * index, -0.5, 9.81])
        true_operator = skew(omega) @ skew(omega) + skew(alpha)
        gyro_cov = np.array([[0.004, 0.001, -0.0003], [0.001, 0.006, 0.0002], [-0.0003, 0.0002, 0.009]])
        parent = observation("p", index * 0.01, omega, alpha, force, gyro_cov * (index + 1))
        child = observation("c", index * 0.01, rotation.T @ omega, rotation.T @ alpha,
                            rotation.T @ (force + true_operator @ position), gyro_cov * 0.7)
        if precision is None:
            joint_cov = np.zeros((6, 6))
            joint_cov[:3, :3] = parent.angular_velocity_covariance
            joint_cov[3:, 3:] = child.angular_velocity_covariance
            noise = source_noise_matrix()
            precision = np.linalg.inv(noise @ np.kron(joint_cov, np.eye(4) * 0.25) @ noise.T)
        difference = parent.angular_velocity - child.angular_velocity
        total = parent.angular_velocity + child.angular_velocity
        h_matrix = np.zeros((4, 4))
        h_matrix[0, 1:] = -difference
        h_matrix[1:, 0] = difference
        h_matrix[1:, 1:] = skew(total)
        parameter = canonical_bingham_parameter(0.98 * parameter - 0.5 * h_matrix.T @ precision @ h_matrix)
        e_rotation = rotation_first_moment(bingham_second_moment(parameter, integration_steps=50))
        operator_p = expected_acceleration_operator(parent)
        operator_c = e_rotation @ expected_acceleration_operator(child) @ e_rotation.T
        delta = e_rotation @ child.specific_force - parent.specific_force
        if use_child:
            coefficient = np.zeros((6, 6))
            coefficient[:3, :3] = operator_p
            coefficient[3:, 3:] = operator_c
            data = np.tile(delta, 2)
        else:
            coefficient, data = 0.5 * (operator_p + operator_c), delta
        # Source SequentialLeastSquare.update_ordinal, with the declared finite prior.
        covariance = np.linalg.inv(np.linalg.inv(covariance) + 0.8 * coefficient.T @ coefficient)
        mean += covariance @ coefficient.T @ (0.8 * (data - coefficient @ mean))
        if use_child:
            cp, cc = covariance[:3, :3], covariance[3:, 3:]
            sum_inv = np.linalg.inv(cp + cc)
            expected_mean = cc @ sum_inv @ mean[:3] + cp @ sum_inv @ mean[3:]
            expected_covariance = cp @ sum_inv @ cc
        else:
            expected_mean, expected_covariance = mean, covariance
        record = estimator.update(parent, child)
        component = record.distribution.components[0]
        np.testing.assert_allclose(estimator.rotation_parameter, parameter, atol=1e-9)
        np.testing.assert_allclose(component.translation.mean_at_reference, expected_mean, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(component.translation.residual_covariance, expected_covariance, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(estimator.rotation_residual_precision, precision)
        assert record.stamp == parent.stamp
        assert record.representative is None
        assert record.representative_kind is RepresentativeKind.NONE
        assert component.orientation.kind is OrientationKind.FINITE_BINGHAM
        assert component.approximation.lossy
        assert "E[R]" in component.approximation.detail


def test_source_noise_covariance_matches_uniform_quaternion_approximation():
    parent = np.array([[0.1, 0.01, -0.02], [0.01, 0.2, 0.03], [-0.02, 0.03, 0.3]])
    child = parent * 0.6
    block = np.zeros((6, 6))
    block[:3, :3], block[3:, 3:] = parent, child
    n_matrix = source_noise_matrix()
    expected = n_matrix @ np.kron(block, np.eye(4) / 4) @ n_matrix.T
    np.testing.assert_allclose(gyro_residual_covariance(parent, child), expected)
    np.testing.assert_allclose(expected, np.eye(4) * np.trace(parent + child) / 4)


def test_nonidentity_rigid_transform_and_distribution_keep_updating():
    estimator = ExtendedLeastSquaresImuRelativePoseEstimator("p", "c")
    rotation = quat_to_rotmat(rpy_to_quat(0.5, -0.25, 0.8))
    position = np.array([0.18, -0.09, 0.04])
    saved = None
    # The source begins translation updates before rotation is identifiable.
    # Keep that transient (including gravity rotated by a broad Bingham),
    # then check recovery with continued multi-axis excitation.
    for index in range(3000):
        time = index * 0.03
        omega = np.array([np.sin(time) + 0.2, 1.1 * np.cos(1.3 * time), 0.7 * np.sin(0.8 * time)])
        alpha = np.array([np.cos(time), -1.43 * np.sin(1.3 * time), 0.56 * np.cos(0.8 * time)])
        force = np.array([0.3 * np.cos(time), 0.2, 9.81])
        operator = skew(omega) @ skew(omega) + skew(alpha)
        parent = observation("p", time, omega, alpha, force, np.eye(3) * 1e-4)
        child = observation("c", time, rotation.T @ omega, rotation.T @ alpha,
                            rotation.T @ (force + operator @ position), np.eye(3) * 1e-4)
        result = estimator.update(parent, child)
        if index == 1499:
            saved = result
    component = result.distribution.components[0]
    np.testing.assert_allclose(quat_to_rotmat(component.orientation.mode_wxyz), rotation, atol=1e-7)
    np.testing.assert_allclose(component.translation.mean_at_reference, position, atol=0.006)
    assert estimator.update_count == 3000
    assert component.orientation.kind is OrientationKind.FINITE_BINGHAM
    assert component.orientation.inverse_concentration > 0
    assert np.linalg.eigvalsh(component.translation.residual_covariance).min() > 0
    assert result.stamp > saved.stamp
    assert component.orientation.inverse_concentration < saved.distribution.components[0].orientation.inverse_concentration
    assert np.trace(component.translation.residual_covariance) < np.trace(saved.distribution.components[0].translation.residual_covariance)


def test_invalid_update_is_transactional_and_reset_clears_frozen_noise():
    estimator = ExtendedLeastSquaresImuRelativePoseEstimator("p", "c")
    parent = observation("p", 1.0, [1, 0, 0], [0, 1, 0], [0, 0, 9.81])
    child = observation("c", 1.0, [1, 0, 0], [0, 1, 0], [0, 0, 9.81])
    result = estimator.update(parent, child)
    with pytest.raises(ValueError, match="strictly increasing"):
        estimator.update(parent, child)
    assert estimator.result() is result
    assert estimator.update_count == 1
    estimator.reset()
    assert estimator.rotation_residual_precision is None
    with pytest.raises(ValueError, match="No timestamped"):
        estimator.result()
    parent.stamp, child.stamp = 2.0, 2.1
    with pytest.raises(ValueError, match="same acquisition time"):
        estimator.update(parent, child)
    assert estimator.update_count == 0
    parent.stamp = child.stamp = 2.0
    parent.angular_velocity_covariance = child.angular_velocity_covariance = np.zeros((3, 3))
    with pytest.raises(np.linalg.LinAlgError):
        estimator.update(parent, child)
    assert estimator.rotation_residual_precision is None


def test_core_import_does_not_import_ros_or_source_repository():
    source = Path(__file__).resolve().parents[2] / "ros/core/probtf_core/src"
    environment = dict(os.environ, PYTHONPATH=str(source))
    code = (
        "from probtf_estimators import ImuRelativePosePipeline; import sys; "
        "assert 'rospy' not in sys.modules; assert 'imu_relpose_estim' not in sys.modules"
    )
    subprocess.run([sys.executable, "-B", "-c", code], env=environment, check=True, cwd="/tmp")
