"""Duck-typed ROS conversions for estimator-specific messages."""

import numpy as np

from probtf.models import ImuKinematics
from probtf_estimators.evidence_fusion import TransformEvidence


def _vector3_to_array(vector):
    return np.array([vector.x, vector.y, vector.z], dtype=float)


def _assign_vector3(vector, values):
    vector.x, vector.y, vector.z = (float(value) for value in values)


def _stamp_to_seconds(stamp):
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return float(stamp.secs) + float(stamp.nsecs) * 1e-9
    return float(stamp)


def _time_from_seconds(seconds, time_factory=None):
    if time_factory is None:
        raise ValueError("time_factory is required at the ROS-independent estimator boundary.")
    return time_factory(float(seconds))


def imu_kinematics_from_msg(message):
    return ImuKinematics(
        frame_id=message.header.frame_id,
        angular_velocity=_vector3_to_array(message.angular_velocity),
        angular_acceleration=_vector3_to_array(message.angular_acceleration),
        specific_force=_vector3_to_array(message.specific_force),
        angular_velocity_covariance=np.asarray(
            message.angular_velocity_covariance,
            dtype=float,
        ).reshape(3, 3),
        angular_acceleration_covariance=np.asarray(
            message.angular_acceleration_covariance,
            dtype=float,
        ).reshape(3, 3),
        specific_force_covariance=np.asarray(
            message.specific_force_covariance,
            dtype=float,
        ).reshape(3, 3),
        stamp=_stamp_to_seconds(message.header.stamp),
    )


def transform_evidence_from_msg(message):
    orientation = None
    if message.has_orientation:
        orientation = np.asarray(message.orientation_bingham.matrix, dtype=float).reshape(4, 4)
    information = None
    information_vector = None
    if message.has_position:
        information = np.asarray(message.position_information, dtype=float).reshape(3, 3)
        information_vector = _vector3_to_array(message.position_information_vector)
    sequence = int(message.sequence) if message.has_sequence else None
    return TransformEvidence(
        source_id=message.source_id,
        evidence_kind=message.evidence_kind or "likelihood",
        parent_frame_id=message.parent_frame_id,
        child_frame_id=message.child_frame_id,
        orientation_bingham=orientation,
        position_information=information,
        position_information_vector=information_vector,
        timestamp=_stamp_to_seconds(message.header.stamp),
        sequence=sequence,
    )


def transform_evidence_to_msg(evidence, message_type=None, time_factory=None):
    if not isinstance(evidence, TransformEvidence):
        raise TypeError("evidence must be a TransformEvidence.")
    if message_type is None:
        raise ValueError("message_type is required at the ROS-independent estimator boundary.")
    message = message_type()
    message.header.frame_id = evidence.parent_frame_id
    if evidence.timestamp is not None:
        message.header.stamp = _time_from_seconds(evidence.timestamp, time_factory)
    message.parent_frame_id = evidence.parent_frame_id
    message.child_frame_id = evidence.child_frame_id
    message.source_id = evidence.source_id
    message.evidence_kind = evidence.evidence_kind
    message.has_sequence = evidence.sequence is not None
    message.sequence = 0 if evidence.sequence is None else evidence.sequence
    message.has_orientation = evidence.orientation_bingham is not None
    if message.has_orientation:
        message.orientation_bingham.matrix = evidence.orientation_bingham.reshape(-1).tolist()
    message.has_position = evidence.position_information is not None
    if message.has_position:
        message.position_information = evidence.position_information.reshape(-1).tolist()
        _assign_vector3(
            message.position_information_vector,
            evidence.position_information_vector,
        )
    return message
