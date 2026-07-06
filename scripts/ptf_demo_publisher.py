#!/usr/bin/env python3

import math

import numpy as np
import rospy

from probik_demo.msg import ProbabilisticTF
from probik_demo.ptf_utils import demo_bingham_matrix, quaternion_from_rpy


class ProbabilisticTFDemoPublisher:
    def __init__(self):
        self.publisher = rospy.Publisher("input_ptf", ProbabilisticTF, queue_size=1, latch=True)

        self.frame_id = rospy.get_param("~frame_id", "base_link")
        self.child_frame_id = rospy.get_param("~child_frame_id", "probabilistic_target")
        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))

        self.mean_translation = np.asarray(
            rospy.get_param("~mean_translation", [0.55, 0.10, 0.55]),
            dtype=float,
        )
        covariance_values = rospy.get_param(
            "~translation_covariance",
            [0.0016, 0.0, 0.0, 0.0, 0.0009, 0.0, 0.0, 0.0, 0.0012],
        )
        self.translation_covariance = np.asarray(covariance_values, dtype=float).reshape(3, 3)

        mode_rpy_deg = rospy.get_param("~mode_rpy_deg", [15.0, 30.0, -20.0])
        mode_rpy = [math.radians(value) for value in mode_rpy_deg]
        mode_quaternion = quaternion_from_rpy(*mode_rpy)

        concentrations = rospy.get_param("~bingham_concentrations", [120.0, 65.0, 30.0])
        self.rotation_matrix = demo_bingham_matrix(mode_quaternion, concentrations)

        rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate, 1e-3)), self.publish_once)

    def publish_once(self, _event):
        message = ProbabilisticTF()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.frame_id
        message.child_frame_id = self.child_frame_id

        message.translation.mean.x = float(self.mean_translation[0])
        message.translation.mean.y = float(self.mean_translation[1])
        message.translation.mean.z = float(self.mean_translation[2])
        message.translation.covariance = self.translation_covariance.reshape(-1).tolist()

        message.rotation.matrix = self.rotation_matrix.reshape(-1).tolist()

        self.publisher.publish(message)


def main():
    rospy.init_node("ptf_demo_publisher")
    ProbabilisticTFDemoPublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
