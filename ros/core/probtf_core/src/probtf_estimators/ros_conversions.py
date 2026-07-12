"""Duck-typed ROS conversions for estimator-specific messages."""

from numbers import Integral

import numpy as np

from probtf.distributions import BinghamOrientation, OrientationKind, trace_zero_matrix
from probtf.geometry import pack_symmetric_upper, unpack_symmetric_upper
from probtf.provenance import ApproximationInfo, ApproximationKind, Provenance
from probtf_estimators.evidence_fusion import TransformEvidence
from probtf_estimators.imu_kinematics import ImuKinematics


_ORIENTATION_TO_WIRE = {
    OrientationKind.FINITE_BINGHAM: 0,
    OrientationKind.DIRAC: 1,
    OrientationKind.UNIFORM: 2,
}
_APPROXIMATION_ORDER = (
    ApproximationKind.EXACT,
    ApproximationKind.PRODUCER_SUPPLIED,
    None,
    ApproximationKind.TANGENT_SURROGATE,
    ApproximationKind.NUMERICAL_INTEGRATION,
    ApproximationKind.MONTE_CARLO,
    ApproximationKind.MOMENT_SUMMARY,
    ApproximationKind.MIXTURE_REDUCTION,
    ApproximationKind.BINGHAM_CLOSURE,
    ApproximationKind.REPRESENTATIVE_PROJECTION,
    ApproximationKind.UNAVAILABLE,
)
_APPROXIMATION_TO_WIRE = {
    kind: index for index, kind in enumerate(_APPROXIMATION_ORDER) if kind is not None
}
_UINT32_MAX = (1 << 32) - 1


def _vector3_to_array(vector):
    return np.array([vector.x, vector.y, vector.z], dtype=float)


def _assign_vector3(vector, values):
    vector.x, vector.y, vector.z = (float(value) for value in values)


def _assign_quaternion_xyzw(message, quat_wxyz):
    message.w = float(quat_wxyz[0])
    message.x = float(quat_wxyz[1])
    message.y = float(quat_wxyz[2])
    message.z = float(quat_wxyz[3])


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


def _assign_approximation(message, value):
    if not isinstance(value, ApproximationInfo):
        raise TypeError("approximation must be ApproximationInfo.")
    message.kind = _APPROXIMATION_TO_WIRE[value.kind]
    message.lossy = value.lossy
    message.detail = value.detail
    message.source = value.source
    message.has_error_bound = value.error_bound is not None
    message.error_bound = 0.0 if value.error_bound is None else value.error_bound


def _approximation_from_msg(message):
    index = int(message.kind)
    if (
        index < 0
        or index >= len(_APPROXIMATION_ORDER)
        or _APPROXIMATION_ORDER[index] is None
    ):
        raise ValueError("Unknown approximation kind {}.".format(message.kind))
    return ApproximationInfo(
        kind=_APPROXIMATION_ORDER[index],
        lossy=bool(message.lossy),
        detail=message.detail,
        source=message.source,
        error_bound=float(message.error_bound) if message.has_error_bound else None,
    )


def _assign_provenance(message, value):
    if not isinstance(value, Provenance):
        raise TypeError("provenance must be Provenance.")
    message.source_ids = list(value.source_ids)
    message.derived_from_edge_ids = list(value.derived_from_edge_ids)
    message.method = value.method
    message.detail = value.detail


def _provenance_from_msg(message):
    return Provenance(
        source_ids=tuple(message.source_ids),
        derived_from_edge_ids=tuple(message.derived_from_edge_ids),
        method=message.method,
        detail=message.detail,
    )


def _assign_orientation(message, orientation):
    if not isinstance(orientation, BinghamOrientation):
        raise TypeError("orientation must be BinghamOrientation.")
    message.kind = _ORIENTATION_TO_WIRE[orientation.kind]
    message.inverse_concentration = orientation.inverse_concentration
    message.shape_upper_wxyz = pack_symmetric_upper(
        orientation.shape_matrix
    ).tolist()
    _assign_quaternion_xyzw(
        message.reference_quaternion,
        orientation.reference_quaternion_wxyz,
    )


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
        orientation = unpack_symmetric_upper(
            message.orientation_natural_parameter_upper_wxyz,
            4,
        )
    information = None
    information_vector = None
    if message.has_translation:
        information = unpack_symmetric_upper(message.translation_information_upper, 3)
        information_vector = _vector3_to_array(
            message.translation_information_vector
        )
    sequence = int(message.sequence) if message.has_sequence else None
    return TransformEvidence(
        source_id=message.source_id,
        evidence_kind=message.evidence_kind or "likelihood",
        parent_frame_id=message.header.frame_id,
        child_frame_id=message.child_frame_id,
        orientation_bingham=orientation,
        position_information=information,
        position_information_vector=information_vector,
        timestamp=_stamp_to_seconds(message.header.stamp),
        sequence=sequence,
        provenance=_provenance_from_msg(message.provenance),
        approximation=_approximation_from_msg(message.approximation),
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
    message.child_frame_id = evidence.child_frame_id
    message.source_id = evidence.source_id
    message.evidence_kind = evidence.evidence_kind
    message.has_sequence = evidence.sequence is not None
    message.sequence = 0 if evidence.sequence is None else evidence.sequence
    message.has_orientation = evidence.orientation_bingham is not None
    if message.has_orientation:
        message.orientation_natural_parameter_upper_wxyz = pack_symmetric_upper(
            trace_zero_matrix(evidence.orientation_bingham)
        ).tolist()
    message.has_translation = evidence.position_information is not None
    if message.has_translation:
        message.translation_information_upper = pack_symmetric_upper(
            evidence.position_information
        ).tolist()
        _assign_vector3(
            message.translation_information_vector,
            evidence.position_information_vector,
        )
    _assign_approximation(message.approximation, evidence.approximation)
    _assign_provenance(message.provenance, evidence.provenance)
    return message


def orientation_distribution_to_msg(
    orientation,
    parent_frame_id,
    child_frame_id,
    stamp,
    edge_id,
    authority,
    approximation,
    provenance,
    message_type=None,
    time_factory=None,
    sequence=None,
):
    """Serialize an orientation-only posterior without synthetic translation."""

    if not isinstance(orientation, BinghamOrientation):
        raise TypeError("orientation must be BinghamOrientation.")
    if message_type is None:
        raise ValueError("message_type is required at the ROS-independent boundary.")
    parent = str(parent_frame_id).strip()
    child = str(child_frame_id).strip()
    if not parent or not child or parent == child:
        raise ValueError("parent and child frame IDs must be non-empty and distinct.")

    message = message_type()
    message.header.frame_id = parent
    message.header.stamp = _time_from_seconds(stamp, time_factory)
    if sequence is not None:
        if (
            isinstance(sequence, (bool, np.bool_))
            or not isinstance(sequence, Integral)
            or int(sequence) < 0
            or int(sequence) > _UINT32_MAX
        ):
            raise ValueError("sequence must be an unsigned 32-bit integer.")
        message.header.seq = int(sequence)
    message.child_frame_id = child
    message.edge_id = str(edge_id).strip()
    message.authority = str(authority).strip()
    if not message.edge_id or not message.authority:
        raise ValueError("edge_id and authority must be non-empty.")
    _assign_orientation(message.orientation, orientation)
    _assign_approximation(message.approximation, approximation)
    _assign_provenance(message.provenance, provenance)
    return message
