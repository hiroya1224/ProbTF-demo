import math

import numpy as np
import pytest

from probtf.distributions import TransformDistributionStamped
from probtf.geometry import quat_to_rotmat, rotation_action_matrix
from probtf.provenance import ApproximationKind
from probtf_estimators.imu_relative_pose import (
    ImuRelativePoseEstimator,
    RecursiveGaussianLeastSquares,
    rigid_point_acceleration_operator,
    vector_alignment_bingham,
)
from probtf_estimators.imu_kinematics import ImuKinematics


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

    assert isinstance(result, TransformDistributionStamped)
    assert result.parent_frame_id == "imu_parent"
    assert result.child_frame_id == "imu_child"
    assert result.edge_id == "imu_parent__to__imu_child"
    assert result.authority == "imu_relative_pose"
    assert result.stamp == pytest.approx(0.47)
    assert len(result.distribution.components) == 1

    component = result.distribution.components[0]
    np.testing.assert_allclose(
        component.translation.mean_at_reference,
        expected_position,
        atol=7e-3,
    )
    np.testing.assert_allclose(component.translation.rotation_coupling, np.zeros((3, 9)))
    mode_rotation = quat_to_rotmat(component.orientation.mode_wxyz)
    np.testing.assert_allclose(mode_rotation, np.eye(3), atol=2e-2)
    assert component.approximation.kind is ApproximationKind.PRODUCER_SUPPLIED
    assert component.approximation.lossy is True
    assert component.provenance.method == "plugin_orientation_rls"
    assert result.approximation == component.approximation
    assert result.provenance.source_ids == ("imu_relative_pose",)


def test_registered_joint_preserves_rotation_translation_coupling():
    estimator = ImuRelativePoseEstimator(
        "parent",
        "child",
        integration_steps=40,
        source_id="two_imu_calibrator",
        edge_id="registered_joint",
        authority="calibration_pipeline",
    )
    parent_to_joint = np.array([0.3, -0.1, 0.2])
    child_to_joint = np.array([0.08, 0.04, -0.03])
    estimator.register_joint_geometry(
        parent_to_joint,
        child_to_joint,
        parent_covariance=np.zeros((3, 3)),
        child_covariance=np.zeros((3, 3)),
    )
    parent = make_observation(
        "parent",
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 9.81],
        stamp=2.0,
    )
    child = make_observation(
        "child",
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 9.81],
        stamp=2.01,
    )

    result = estimator.update(parent, child)
    component = result.distribution.components[0]

    assert result.stamp == pytest.approx(2.01)
    assert result.edge_id == "registered_joint"
    assert result.authority == "calibration_pipeline"
    assert result.provenance.source_ids == ("two_imu_calibrator",)
    assert component.provenance.method == "registered_joint_geometry"
    assert component.approximation.lossy is False
    np.testing.assert_allclose(
        component.translation.rotation_coupling,
        -rotation_action_matrix(child_to_joint),
    )
    for quaternion in (
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
    ):
        np.testing.assert_allclose(
            component.conditional_translation_mean(quaternion),
            parent_to_joint - quat_to_rotmat(quaternion) @ child_to_joint,
            atol=1e-12,
        )


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
