#!/usr/bin/env python3

import math

import numpy as np
import rospy

from symaware_grasp.msg import ProbabilisticTF
from symaware_grasp.ptf_utils import (
    axially_symmetric_bingham_matrix,
    make_probabilistic_tf_message,
    quaternion_from_rpy,
)


class ObjectPoseNode:
    def __init__(self):
        self.publisher = rospy.Publisher("object_prob_tf", ProbabilisticTF, queue_size=1, latch=True)
        self.parent_frame_id = rospy.get_param("~parent_frame_id", "base_link")
        self.child_frame_id = rospy.get_param("~child_frame_id", "demo_cylinder")
        self.publish_rate = float(rospy.get_param("~publish_rate", 1.0))

        self.position_mean = np.asarray(
            rospy.get_param("~position_mean", [0.58, 0.12, 0.46]),
            dtype=float,
        )
        self.position_covariance = np.asarray(
            rospy.get_param(
                "~position_covariance",
                [0.0015, 0.0002, 0.0, 0.0002, 0.0010, 0.0, 0.0, 0.0, 0.0008],
            ),
            dtype=float,
        ).reshape(3, 3)
        mode_rpy_deg = rospy.get_param("~mode_rpy_deg", [0.0, 0.0, 40.0])
        self.mode_quaternion = quaternion_from_rpy(
            math.radians(mode_rpy_deg[0]),
            math.radians(mode_rpy_deg[1]),
            math.radians(mode_rpy_deg[2]),
        )
        symmetry_axis_body = rospy.get_param("~symmetry_axis_body", [0.0, 0.0, 1.0])
        concentrations = rospy.get_param("~bingham_concentrations", [220.0, 220.0, 4.0])
        self.bingham_matrix = axially_symmetric_bingham_matrix(
            self.mode_quaternion,
            symmetry_axis_body,
            concentrations,
        )
        rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate, 1e-3)), self.publish_message)

    def publish_message(self, _event):
        message = make_probabilistic_tf_message(
            parent_frame_id=self.parent_frame_id,
            child_frame_id=self.child_frame_id,
            position_mean_xyz=self.position_mean,
            position_covariance=self.position_covariance,
            orientation_bingham_matrix=self.bingham_matrix,
            orientation_mode_wxyz=self.mode_quaternion,
            stamp=rospy.Time.now(),
        )
        self.publisher.publish(message)


def main():
    rospy.init_node("object_pose_node")
    ObjectPoseNode()
    rospy.spin()


if __name__ == "__main__":
    main()
