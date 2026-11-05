"""Example-owned ROS field adapters, tested without importing ROS."""

from types import SimpleNamespace

import numpy as np
import pytest

from probtf_ros.v2_conversions import (
    V2MessageTypes, transform_distribution_from_msg, transform_distribution_to_msg,
)
from relative_coords_from_paired_imu.ros_interface import imu_kinematics_from_msg


class Stamp(float):
    def to_sec(self):
        return float(self)


def Vector3():
    return SimpleNamespace(x=0.0, y=0.0, z=0.0)


def Quaternion():
    return SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0)


def Header():
    return SimpleNamespace(frame_id="", stamp=Stamp(0.0))


MESSAGE_TYPES = V2MessageTypes(
    lambda: SimpleNamespace(reference_quaternion=Quaternion()),
    lambda: SimpleNamespace(mean_at_reference=Vector3()),
    SimpleNamespace,
    lambda: SimpleNamespace(header=Header(), representative=SimpleNamespace(
        translation=Vector3(), rotation=Quaternion())),
    SimpleNamespace,
    SimpleNamespace,
    SimpleNamespace,
)


def _assert_record_equal(actual, expected):
    assert actual.parent_frame_id == expected.parent_frame_id
    assert actual.child_frame_id == expected.child_frame_id
    assert actual.edge_id == expected.edge_id
    assert actual.authority == expected.authority
    assert actual.stamp == expected.stamp
    assert actual.is_static == expected.is_static
    assert actual.representative_kind is expected.representative_kind
    assert actual.approximation == expected.approximation
    assert actual.provenance == expected.provenance
    assert len(actual.distribution.components) == len(expected.distribution.components)
    for left, right in zip(actual.distribution.components, expected.distribution.components):
        assert left.component_id == right.component_id
        assert left.raw_weight == right.raw_weight
        assert left.orientation.kind is right.orientation.kind
        assert left.orientation.inverse_concentration == right.orientation.inverse_concentration
        np.testing.assert_allclose(left.orientation.shape_matrix, right.orientation.shape_matrix)
        np.testing.assert_allclose(
            left.orientation.reference_quaternion_wxyz,
            right.orientation.reference_quaternion_wxyz,
        )
        np.testing.assert_allclose(left.translation.mean_at_reference, right.translation.mean_at_reference)
        np.testing.assert_allclose(
            left.translation.residual_covariance,
            right.translation.residual_covariance,
        )
        np.testing.assert_allclose(left.translation.rotation_coupling, right.translation.rotation_coupling)
        assert left.approximation == right.approximation
        assert left.provenance == right.provenance


def test_raw_imu_pipeline_wire_preserves_the_estimated_law_and_acquisition_time():
    from relative_coords_from_paired_imu.imu_pair_pipeline import ImuRelativePosePipeline

    pipeline = ImuRelativePosePipeline("parent", "child", window_size=5, polynomial_order=2)
    covariance = np.array([[0.004, 0.001, 0.0], [0.001, 0.006, -0.001], [0.0, -0.001, 0.008]])
    records = []
    for index in range(16):
        for frame, offset in (("parent__aabbcc", 0.0), ("child__ddeeff", 0.0004)):
            stamp = 1800000000.0 + index * 0.002 + offset
            time = stamp - 1800000000.0
            record = pipeline.update(
                frame, stamp, [1 + time, 0.3 + 2 * time, 0.4 - time],
                [0.2 * time, -0.1, 9.81], covariance, covariance * 3,
            )
            if record is None:
                continue
            message = transform_distribution_to_msg(record, MESSAGE_TYPES, Stamp)
            restored = transform_distribution_from_msg(message)
            _assert_record_equal(restored, record)
            assert restored.representative is None
            assert restored.stamp == pipeline.last_pair[0].stamp == pipeline.last_pair[1].stamp
            assert restored.stamp < stamp
            np.testing.assert_allclose(
                restored.distribution.components[0].orientation.backend_parameter_matrix(),
                pipeline.estimator.rotation_parameter, rtol=1e-12, atol=1e-9,
            )
            assert restored.provenance.source_ids == ("imu_extended_leastsq", "parent__aabbcc", "child__ddeeff")
            assert restored.distribution.components[0].raw_weight == 1.0
            assert restored.distribution.components[0].approximation.lossy
            records.append(record)
    assert len(records) > 10


def test_imu_kinematics_conversion_uses_row_major_covariances():
    message = type("ImuKinematicsMessage", (), {})()
    message.header = Header()
    message.header.frame_id = "imu"
    message.header.stamp = Stamp(1.5)
    message.angular_velocity = Vector3()
    message.angular_velocity.x = 1.0
    message.angular_acceleration = Vector3()
    message.angular_acceleration.y = 2.0
    message.specific_force = Vector3()
    message.specific_force.z = 3.0
    gyro_covariance = np.array(
        [[2.0, 0.1, 0.2], [0.1, 3.0, 0.3], [0.2, 0.3, 4.0]],
        dtype=float,
    )
    message.angular_velocity_covariance = gyro_covariance.reshape(-1).tolist()
    message.angular_acceleration_covariance = np.eye(3).reshape(-1).tolist()
    message.specific_force_covariance = (np.eye(3) * 2.0).reshape(-1).tolist()
    converted = imu_kinematics_from_msg(message)

    assert converted.frame_id == "imu"
    assert converted.stamp == 1.5
    np.testing.assert_allclose(converted.angular_velocity, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(converted.angular_acceleration, [0.0, 2.0, 0.0])
    np.testing.assert_allclose(converted.specific_force, [0.0, 0.0, 3.0])
    np.testing.assert_allclose(converted.angular_velocity_covariance, gyro_covariance)
