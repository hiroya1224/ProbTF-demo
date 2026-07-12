"""ROS-independent numerical and domain primitives for ProbTF."""

from probtf.bingham import (
    RotationMoment,
    bingham_fourth_moment,
    bingham_mode,
    bingham_second_moment,
    canonical_bingham_parameter,
    match_bingham_to_second_moment,
    quaternion_product_second_moment,
    rotation_first_moment,
    rotation_kronecker_moment,
    rotation_moment_from_bingham,
    validate_bingham_parameter,
)
from probtf.fusion import (
    EvidenceProvenance,
    FusedTransformEvidence,
    TransformEvidence,
    fuse_evidence,
    fuse_transform_evidence,
)

from probtf.geometry import (
    axis_angle_to_quat,
    quat_conj,
    quat_left_matrix,
    quat_mul,
    quat_normalize,
    quat_right_matrix,
    quat_to_rotmat,
    rpy_to_quat,
)
from probtf.imu_preprocessing import ImuKinematicsPreprocessor
from probtf.imu_relative_pose import (
    ImuRelativePoseEstimator,
    RecursiveGaussianLeastSquares,
    rigid_point_acceleration_operator,
    vector_alignment_bingham,
)
from probtf.models import (
    BinghamRotation,
    GaussianPosition,
    ImuKinematics,
    ProbabilisticTransform,
    SensorMount,
)
from probtf.sensor_config import load_sensor_mounts, parse_sensor_mounts
from probtf.symbolic_urdf import (
    SymbolicUrdfTemplate,
    find_symbolic_urdf_placeholders,
    materialize_symbolic_urdf,
    parse_symbolic_urdf,
)

__all__ = [
    "BinghamRotation",
    "EvidenceProvenance",
    "FusedTransformEvidence",
    "GaussianPosition",
    "ImuKinematics",
    "ImuKinematicsPreprocessor",
    "ImuRelativePoseEstimator",
    "ProbabilisticTransform",
    "RecursiveGaussianLeastSquares",
    "RotationMoment",
    "SensorMount",
    "SymbolicUrdfTemplate",
    "TransformEvidence",
    "axis_angle_to_quat",
    "bingham_fourth_moment",
    "bingham_mode",
    "bingham_second_moment",
    "canonical_bingham_parameter",
    "fuse_evidence",
    "fuse_transform_evidence",
    "find_symbolic_urdf_placeholders",
    "match_bingham_to_second_moment",
    "materialize_symbolic_urdf",
    "load_sensor_mounts",
    "parse_symbolic_urdf",
    "parse_sensor_mounts",
    "quat_conj",
    "quat_left_matrix",
    "quat_mul",
    "quat_normalize",
    "quat_right_matrix",
    "quat_to_rotmat",
    "quaternion_product_second_moment",
    "rotation_first_moment",
    "rotation_kronecker_moment",
    "rotation_moment_from_bingham",
    "rigid_point_acceleration_operator",
    "rpy_to_quat",
    "validate_bingham_parameter",
    "vector_alignment_bingham",
]
