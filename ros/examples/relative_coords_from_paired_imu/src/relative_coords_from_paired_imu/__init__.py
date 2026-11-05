"""Example-owned numerical API, independent of ROS.

Load algorithms on demand so ROS tools can import the generated ``.msg``
namespace without first importing the numerical stack or ProbTF.
"""

from importlib import import_module


_EXPORT_MODULES = {
    "ExtendedLeastSquaresImuRelativePoseEstimator": "imu_extended_leastsq",
    "ImuKinematics": "imu_kinematics",
    "ImuKinematicsPreprocessor": "imu_preprocessing",
    "ImuRelativePoseEstimator": "imu_relative_pose",
    "ImuRelativePosePipeline": "imu_pair_pipeline",
    "LatestImuRelativePosePipeline": "latest_imu_pair",
    "LatestImuPairBuffer": "latest_imu_pair",
    "RecursiveGaussianLeastSquares": "imu_relative_pose",
    "SynchronizedImuPairPreprocessor": "imu_pair_pipeline",
    "imu_measurement_covariance": "imu_pair_pipeline",
    "rigid_point_acceleration_operator": "imu_relative_pose",
    "vector_alignment_bingham": "imu_relative_pose",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name):
    if name not in _EXPORT_MODULES:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    value = getattr(import_module("." + _EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value
