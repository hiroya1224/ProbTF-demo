import math
import os

import yaml

from symaware_grasp.models import GraspCandidate
from symaware_grasp.ptf_utils import (
    quaternion_from_approach_and_finger_axes,
    quaternion_from_rpy,
)


def _candidate_from_dict(candidate_dict):
    approach_axis = candidate_dict.get("approach_axis")
    finger_axis = candidate_dict.get("finger_axis")

    if approach_axis is not None and finger_axis is not None:
        quat_wxyz = quaternion_from_approach_and_finger_axes(approach_axis, finger_axis)
    elif "object_to_grasp_orientation_wxyz" in candidate_dict:
        quat_wxyz = candidate_dict["object_to_grasp_orientation_wxyz"]
    elif "object_to_grasp_orientation_rpy_deg" in candidate_dict:
        roll_deg, pitch_deg, yaw_deg = candidate_dict["object_to_grasp_orientation_rpy_deg"]
        quat_wxyz = quaternion_from_rpy(
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(yaw_deg),
        )
    else:
        raise KeyError("Grasp candidate requires an orientation.")

    return GraspCandidate(
        grasp_id=candidate_dict["grasp_id"],
        object_to_grasp_position=candidate_dict["object_to_grasp_position"],
        object_to_grasp_orientation_wxyz=quat_wxyz,
        approach_axis=approach_axis or [1.0, 0.0, 0.0],
        finger_axis=finger_axis or [0.0, 0.0, 1.0],
        weight=candidate_dict.get("weight", 1.0),
    )


def load_grasp_library(path, object_id):
    with open(os.path.expanduser(path), "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    objects = data.get("objects", {})
    if object_id not in objects:
        raise KeyError(f"Object id '{object_id}' is not defined in the grasp library.")
    return [_candidate_from_dict(candidate_dict) for candidate_dict in objects[object_id].get("candidates", [])]
