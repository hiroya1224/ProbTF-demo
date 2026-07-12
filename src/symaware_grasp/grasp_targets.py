import numpy as np

from symaware_grasp.models import ProbabilisticTransform
from symaware_grasp.ptf_utils import (
    make_bingham_distribution,
    pushforward_bingham_right,
    quaternion_multiply_wxyz,
    rotation_matrix_from_quaternion,
)


def compose_grasp_targets(
    object_transform,
    candidates,
    rotation_covariance_samples=80,
    covariance_floor=1e-4,
):
    distribution = make_bingham_distribution(object_transform.orientation_bingham)
    object_mode = object_transform.orientation_mode_wxyz
    object_rotation_mode = rotation_matrix_from_quaternion(object_mode)

    targets = []
    for candidate in candidates:
        grasp_offset = candidate.object_to_grasp_position
        grasp_orientation = candidate.object_to_grasp_orientation_wxyz
        target_mean = object_transform.position_mean + object_rotation_mode @ grasp_offset
        target_covariance = object_transform.position_covariance + float(covariance_floor) * np.eye(3)

        if int(rotation_covariance_samples) > 1 and np.linalg.norm(grasp_offset) > 1e-8:
            sampled_quaternions = distribution.update_sample(N_sample=int(rotation_covariance_samples))
            rotated_offsets = np.asarray(
                [
                    rotation_matrix_from_quaternion(sampled_quaternion) @ grasp_offset
                    for sampled_quaternion in sampled_quaternions
                ],
                dtype=float,
            )
            target_covariance += np.cov(rotated_offsets.T)

        targets.append(
            ProbabilisticTransform(
                parent_frame_id=object_transform.parent_frame_id,
                child_frame_id=candidate.grasp_id,
                position_mean=target_mean,
                position_covariance=target_covariance,
                orientation_bingham=pushforward_bingham_right(
                    object_transform.orientation_bingham,
                    grasp_orientation,
                ),
                orientation_mode_wxyz=quaternion_multiply_wxyz(
                    object_mode,
                    grasp_orientation,
                ),
            )
        )
    return targets

