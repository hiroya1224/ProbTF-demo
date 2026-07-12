import numpy as np

from probtf.distributions import BinghamOrientation, trace_zero_matrix
from probtf.models import ProbabilisticTransform
from probtf.provenance import ApproximationInfo, ApproximationKind, Provenance
from probtf_estimators.evidence_fusion import TransformEvidence
from probtf_estimators.ros_conversions import (
    imu_kinematics_from_msg,
    orientation_distribution_to_msg,
    transform_evidence_from_msg,
    transform_evidence_to_msg,
)

from probtf_ros.conversions import (
    probabilistic_transform_to_msg,
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
        self.seq = 0
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
        self.evidence_kind = ""
        self.evidence_source_ids = []
        self.has_position = False
        self.position_mean = Vector3()
        self.position_covariance = [0.0] * 9
        self.has_orientation = False
        self.orientation_bingham = BinghamMessage()
        self.orientation_mode = Quaternion()
        self.approximation_type = ""
        self.closure_approximation = False


class TransformEvidenceMessage:
    def __init__(self):
        self.header = Header()
        self.child_frame_id = ""
        self.source_id = ""
        self.evidence_kind = ""
        self.has_sequence = False
        self.sequence = 0
        self.has_orientation = False
        self.orientation_natural_parameter_upper_wxyz = [0.0] * 10
        self.has_translation = False
        self.translation_information_upper = [0.0] * 6
        self.translation_information_vector = Vector3()
        self.approximation = ApproximationMessage()
        self.provenance = ProvenanceMessage()


class ApproximationMessage:
    def __init__(self):
        self.kind = 0
        self.lossy = False
        self.detail = ""
        self.source = ""
        self.has_error_bound = False
        self.error_bound = 0.0


class ProvenanceMessage:
    def __init__(self):
        self.source_ids = []
        self.derived_from_edge_ids = []
        self.method = ""
        self.detail = ""


class BinghamOrientationMessage:
    def __init__(self):
        self.kind = 0
        self.inverse_concentration = 0.0
        self.shape_upper_wxyz = [0.0] * 10
        self.reference_quaternion = Quaternion()


class OrientationDistributionMessage:
    def __init__(self):
        self.header = Header()
        self.child_frame_id = ""
        self.edge_id = ""
        self.authority = ""
        self.orientation = BinghamOrientationMessage()
        self.approximation = ApproximationMessage()
        self.provenance = ProvenanceMessage()


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
    assert message.has_position is True
    assert message.has_orientation is True
    assert [
        message.orientation_mode.w,
        message.orientation_mode.x,
        message.orientation_mode.y,
        message.orientation_mode.z,
    ] == [0.5, 0.5, -0.5, 0.5]
    assert message.closure_approximation is True


def test_transform_evidence_round_trip_preserves_optional_payloads():
    approximation = ApproximationInfo(
        kind=ApproximationKind.BINGHAM_CLOSURE,
        lossy=True,
        detail="Moment closure.",
        source="camera_filter",
        error_bound=0.25,
    )
    provenance = Provenance(
        source_ids=("camera",),
        derived_from_edge_ids=("raw_camera_pose",),
        method="likelihood_conversion",
        detail="Calibrated camera evidence.",
    )
    evidence = TransformEvidence.from_gaussian_position(
        "camera",
        "world",
        "tool",
        [0.1, 0.2, 0.3],
        np.diag([0.2, 0.3, 0.4]),
        orientation_bingham=np.diag([-4.0, -3.0, -2.0, 0.0]),
        timestamp=3.5,
        sequence=9,
        provenance=provenance,
        approximation=approximation,
    )
    message = transform_evidence_to_msg(
        evidence,
        message_type=TransformEvidenceMessage,
        time_factory=Stamp,
    )
    restored = transform_evidence_from_msg(message)

    assert restored.source_id == "camera"
    assert restored.evidence_kind == "likelihood"
    assert restored.parent_frame_id == "world"
    assert restored.sequence == 9
    assert restored.timestamp == 3.5
    assert message.header.frame_id == "world"
    assert not hasattr(message, "parent_frame_id")
    np.testing.assert_allclose(
        restored.orientation_bingham,
        trace_zero_matrix(evidence.orientation_bingham),
    )
    np.testing.assert_allclose(restored.position_information, evidence.position_information)
    np.testing.assert_allclose(
        restored.position_information_vector,
        evidence.position_information_vector,
    )
    assert restored.approximation == approximation
    assert restored.provenance == provenance


def test_transform_evidence_round_trip_accepts_singular_translation_information():
    evidence = TransformEvidence(
        source_id="planar_sensor",
        parent_frame_id="world",
        child_frame_id="tool",
        position_information=np.diag([2.0, 3.0, 0.0]),
        position_information_vector=[4.0, 9.0, 0.0],
        timestamp=2.0,
    )

    message = transform_evidence_to_msg(
        evidence,
        message_type=TransformEvidenceMessage,
        time_factory=Stamp,
    )
    restored = transform_evidence_from_msg(message)

    assert message.has_orientation is False
    assert message.has_translation is True
    assert message.translation_information_upper == [2.0, 0.0, 0.0, 3.0, 0.0, 0.0]
    np.testing.assert_allclose(restored.position_information, evidence.position_information)
    np.testing.assert_allclose(
        restored.position_information_vector,
        evidence.position_information_vector,
    )


def test_orientation_distribution_conversion_has_no_translation_payload():
    orientation = BinghamOrientation.from_parameter_matrix(
        np.diag([0.0, -2.0, -4.0, -6.0])
    )
    approximation = ApproximationInfo(
        kind=ApproximationKind.BINGHAM_CLOSURE,
        lossy=True,
        detail="One-component posterior closure.",
        source="orientation_filter",
    )
    provenance = Provenance(
        source_ids=("gyro", "gravity"),
        method="orientation_filter_update",
    )

    message = orientation_distribution_to_msg(
        orientation,
        parent_frame_id="world",
        child_frame_id="imu",
        stamp=8.25,
        edge_id="world__to__imu",
        authority="orientation_filter",
        approximation=approximation,
        provenance=provenance,
        message_type=OrientationDistributionMessage,
        time_factory=Stamp,
        sequence=17,
    )

    assert message.header.frame_id == "world"
    assert message.header.stamp.to_sec() == 8.25
    assert message.header.seq == 17
    assert message.child_frame_id == "imu"
    assert message.edge_id == "world__to__imu"
    assert message.authority == "orientation_filter"
    assert message.orientation.kind == 0
    assert message.orientation.inverse_concentration == orientation.inverse_concentration
    np.testing.assert_allclose(
        message.orientation.shape_upper_wxyz,
        [
            orientation.shape_matrix[row, column]
            for row, column in (
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 2),
                (2, 3),
                (3, 3),
            )
        ],
    )
    assert message.approximation.kind == 8
    assert message.provenance.source_ids == ["gyro", "gravity"]
    assert not hasattr(message, "translation")


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
