"""Probabilistic single-view AprilTag pose producer."""

from prob_artag_detector.camera import (
    analytic_pinhole_pose_jacobian,
    finite_difference_pose_jacobian,
    ippe_square_object_points,
    pose_jacobian,
    project_points,
    transform_points,
)
from prob_artag_detector.calibration import (
    CameraCalibration,
    approximate_camera_model,
    load_camera_calibration,
)
from prob_artag_detector.detector import ArucoCornerDetector, isotropic_image_covariance
from prob_artag_detector.estimator import (
    PoseEstimationError,
    PoseMixtureEstimator,
    PoseSeed,
    bingham_parameter_from_tangent_precision,
    image_precision,
    local_gauss_newton_hessian,
    normalize_log_weights,
    reconstruct_pose_hessian,
    solve_ippe_square_candidates,
)
from prob_artag_detector.models import (
    CameraModel,
    CandidateDiagnostic,
    EstimationDiagnostics,
    MarkerObservation,
    PoseMixtureResult,
)
from prob_artag_detector.visualization import draw_debug_image

__all__ = [
    "ArucoCornerDetector",
    "CameraModel",
    "CameraCalibration",
    "CandidateDiagnostic",
    "EstimationDiagnostics",
    "MarkerObservation",
    "PoseMixtureResult",
    "PoseEstimationError",
    "PoseMixtureEstimator",
    "PoseSeed",
    "analytic_pinhole_pose_jacobian",
    "approximate_camera_model",
    "finite_difference_pose_jacobian",
    "bingham_parameter_from_tangent_precision",
    "draw_debug_image",
    "image_precision",
    "ippe_square_object_points",
    "isotropic_image_covariance",
    "local_gauss_newton_hessian",
    "load_camera_calibration",
    "normalize_log_weights",
    "pose_jacobian",
    "project_points",
    "reconstruct_pose_hessian",
    "solve_ippe_square_candidates",
    "transform_points",
]
