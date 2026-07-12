import numpy as np
from geometry_msgs.msg import Quaternion, Vector3
from probik_msgs.msg import BinghamDistribution, ProbabilisticTF

from symaware_grasp.models import ProbabilisticTransform
from symaware_grasp.ptf_utils import make_bingham_distribution, normalize_wxyz, quaternion_array_to_wxyz


def vector3_from_msg(message):
    return np.array([message.x, message.y, message.z], dtype=float)


def vector3_msg_from_array(values_xyz):
    values = np.asarray(values_xyz, dtype=float).reshape(3)
    return Vector3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def quaternion_wxyz_from_msg(message):
    return normalize_wxyz([message.w, message.x, message.y, message.z])


def quaternion_msg_from_wxyz(quaternion_wxyz):
    quaternion = normalize_wxyz(quaternion_wxyz)
    return Quaternion(
        w=float(quaternion[0]),
        x=float(quaternion[1]),
        y=float(quaternion[2]),
        z=float(quaternion[3]),
    )


def probabilistic_transform_from_msg(message):
    mode = np.array(
        [
            message.orientation_mode.w,
            message.orientation_mode.x,
            message.orientation_mode.y,
            message.orientation_mode.z,
        ],
        dtype=float,
    )
    if np.linalg.norm(mode) <= 1e-8:
        distribution = make_bingham_distribution(message.orientation_bingham.matrix)
        mode = quaternion_array_to_wxyz(distribution.mode())

    return ProbabilisticTransform(
        parent_frame_id=message.parent_frame_id or message.header.frame_id,
        child_frame_id=message.child_frame_id,
        position_mean=vector3_from_msg(message.position_mean),
        position_covariance=message.position_covariance,
        orientation_bingham=message.orientation_bingham.matrix,
        orientation_mode_wxyz=mode,
        approximation_type=message.approximation_type,
    )


def position_covariance_from_msg(message):
    return probabilistic_transform_from_msg(message).position_covariance


def ptf_mode_quaternion_wxyz(message):
    return probabilistic_transform_from_msg(message).orientation_mode_wxyz


def probabilistic_transform_to_msg(transform, stamp=None):
    message = ProbabilisticTF()
    if stamp is not None:
        message.header.stamp = stamp
    message.header.frame_id = transform.parent_frame_id
    message.parent_frame_id = transform.parent_frame_id
    message.child_frame_id = transform.child_frame_id
    message.position_mean = vector3_msg_from_array(transform.position_mean)
    message.position_covariance = transform.position_covariance.reshape(-1).tolist()
    message.orientation_bingham = BinghamDistribution()
    message.orientation_bingham.matrix = transform.orientation_bingham.reshape(-1).tolist()
    message.orientation_mode = quaternion_msg_from_wxyz(transform.orientation_mode_wxyz)
    message.approximation_type = transform.approximation_type
    return message


def make_probabilistic_tf_message(
    parent_frame_id,
    child_frame_id,
    position_mean_xyz,
    position_covariance,
    orientation_bingham_matrix,
    orientation_mode_wxyz,
    stamp=None,
    approximation_type="gaussian_position_bingham_orientation",
):
    transform = ProbabilisticTransform(
        parent_frame_id=parent_frame_id,
        child_frame_id=child_frame_id,
        position_mean=position_mean_xyz,
        position_covariance=position_covariance,
        orientation_bingham=orientation_bingham_matrix,
        orientation_mode_wxyz=orientation_mode_wxyz,
        approximation_type=approximation_type,
    )
    return probabilistic_transform_to_msg(transform, stamp=stamp)
