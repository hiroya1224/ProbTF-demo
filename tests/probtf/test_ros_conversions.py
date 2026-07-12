from pathlib import Path
import sys

import numpy as np

from probtf.fusion import TransformEvidence
from probtf.models import ProbabilisticTransform


ROS_PYTHON = Path(__file__).resolve().parents[2] / "ros" / "core" / "probtf_core" / "src"
sys.path.insert(0, str(ROS_PYTHON))

from probtf_ros.conversions import (  # noqa: E402
    imu_kinematics_from_msg,
    probabilistic_transform_to_msg,
    transform_evidence_from_msg,
    transform_evidence_to_msg,
)


class Vector3:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class Quaternion(Vector3):
    def __init__(self):
        super().__init__()
        self.w = 0.0


class Stamp:
    def __init__(self, seconds=0.0):
        self.seconds = float(seconds)

    def to_sec(self):
        return self.seconds


class Header:
    def __init__(self):
        self.frame_id = ""
        self.stamp = Stamp()


class BinghamMessage:
    def __init__(self):
        self.matrix = [0.0] * 16


class ProbabilisticTransformMessage:
    def __init__(self):
        self.header = Header()
        self.parent_frame_id = ""
        self.child_frame_id = ""
        self.edge_id = ""
        self.source_id = ""
        self.evidence_source_ids = []
        self.position_mean = Vector3()
        self.position_covariance = [0.0] * 9
        self.orientation_bingham = BinghamMessage()
        self.orientation_mode = Quaternion()
        self.approximation_type = ""
        self.closure_approximation = False


class TransformEvidenceMessage:
    def __init__(self):
        self.header = Header()
        self.parent_frame_id = ""
        self.child_frame_id = ""
        self.source_id = ""
        self.has_sequence = False
        self.sequence = 0
        self.has_orientation = False
        self.orientation_bingham = BinghamMessage()
        self.has_position = False
        self.position_information = [0.0] * 9
        self.position_information_vector = Vector3()


def test_probabilistic_transform_conversion_preserves_wxyz_and_metadata():
    transform = ProbabilisticTransform.from_arrays(
        "base",
        "imu",
        [1.0, 2.0, 3.0],
        np.diag([0.1, 0.2, 0.3]),
        np.diag([-3.0, -2.0, -1.0, 0.0]),
        [0.5, 0.5, -0.5, 0.5],
        stamp=12.25,
        source_id="relative_pose",
        evidence_source_ids=("gyro", "force"),
        closure_approximation=True,
    )

    message = probabilistic_transform_to_msg(
        transform,
        message_type=ProbabilisticTransformMessage,
        time_factory=Stamp,
    )

    assert message.header.frame_id == "base"
    assert message.header.stamp.to_sec() == 12.25
    assert message.edge_id == "base__to__imu"
    assert message.source_id == "relative_pose"
    assert message.evidence_source_ids == ["gyro", "force"]
    assert [
        message.orientation_mode.w,
        message.orientation_mode.x,
        message.orientation_mode.y,
        message.orientation_mode.z,
    ] == [0.5, 0.5, -0.5, 0.5]
    assert message.closure_approximation is True


def test_transform_evidence_round_trip_preserves_optional_payloads():
    evidence = TransformEvidence.from_gaussian_position(
        "camera",
        "world",
        "tool",
        [0.1, 0.2, 0.3],
        np.diag([0.2, 0.3, 0.4]),
        orientation_bingham=np.diag([-4.0, -3.0, -2.0, 0.0]),
        timestamp=3.5,
        sequence=9,
    )
    message = transform_evidence_to_msg(
        evidence,
        message_type=TransformEvidenceMessage,
        time_factory=Stamp,
    )
    restored = transform_evidence_from_msg(message)

    assert restored.source_id == "camera"
    assert restored.sequence == 9
    assert restored.timestamp == 3.5
    np.testing.assert_allclose(restored.orientation_bingham, evidence.orientation_bingham)
    np.testing.assert_allclose(restored.position_information, evidence.position_information)
    np.testing.assert_allclose(
        restored.position_information_vector,
        evidence.position_information_vector,
    )


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
