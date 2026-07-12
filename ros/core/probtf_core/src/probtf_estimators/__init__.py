"""Prob-TF producers and evidence estimators.

This package depends on :mod:`probtf`; the foundation package never imports
back from this package.
"""

from probtf_estimators.coupling_from_hessian import (
    HessianCouplingResult,
    coupling_from_hessian,
)
from probtf_estimators.evidence_fusion import (
    EvidenceProvenance,
    FusedTransformEvidence,
    TransformEvidence,
    fuse_evidence,
    fuse_transform_evidence,
)
from probtf_estimators.imu_kinematics import ImuKinematics
from probtf_estimators.imu_preprocessing import ImuKinematicsPreprocessor
from probtf_estimators.imu_relative_pose import (
    ImuRelativePoseEstimator,
    RecursiveGaussianLeastSquares,
    rigid_point_acceleration_operator,
    vector_alignment_bingham,
)
from probtf_estimators.materialization import (
    TransformMaterializationSummary,
    summarize_transform_for_materialization,
)
from probtf_estimators.orientation_imu import (
    OrientationBinghamFilter,
    OrientationEvidence,
    OrientationFilterUpdate,
    delta_quaternion_second_moment,
    gravity_bingham_evidence,
    magnetic_bingham_evidence,
    predict_orientation_bingham,
    vector_alignment_bingham_evidence,
)

__all__ = [
    "EvidenceProvenance",
    "FusedTransformEvidence",
    "HessianCouplingResult",
    "ImuKinematics",
    "ImuKinematicsPreprocessor",
    "ImuRelativePoseEstimator",
    "OrientationBinghamFilter",
    "OrientationEvidence",
    "OrientationFilterUpdate",
    "RecursiveGaussianLeastSquares",
    "TransformEvidence",
    "TransformMaterializationSummary",
    "coupling_from_hessian",
    "delta_quaternion_second_moment",
    "fuse_evidence",
    "fuse_transform_evidence",
    "gravity_bingham_evidence",
    "magnetic_bingham_evidence",
    "predict_orientation_bingham",
    "rigid_point_acceleration_operator",
    "summarize_transform_for_materialization",
    "vector_alignment_bingham",
    "vector_alignment_bingham_evidence",
]
