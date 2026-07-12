#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from symaware_grasp.grasp_library import load_grasp_library
from symaware_grasp.grasp_targets import compose_grasp_targets
from probtf_msgs.msg import ProbabilisticTF, ProbabilisticTFArray
from symaware_grasp.ptf_utils import (
    rotation_matrix_from_quaternion,
)
from symaware_grasp_ros.messages import (
    probabilistic_transform_from_msg,
    probabilistic_transform_to_msg,
)


class ProbTFGraspTargetNode:
    def __init__(self):
        grasp_library_path = rospy.get_param("~grasp_library_path")
        self.object_id = rospy.get_param("~object_id", "demo_cylinder")
        self.candidates = load_grasp_library(grasp_library_path, self.object_id)
        self.rotation_covariance_samples = int(rospy.get_param("~rotation_covariance_samples", 80))
        self.covariance_floor = float(rospy.get_param("~covariance_floor", 1e-4))
        self.axis_length = float(rospy.get_param("~marker_axis_length", 0.12))
        self.publish_markers = bool(rospy.get_param("~publish_markers", False))

        self.targets_publisher = rospy.Publisher("grasp_target_ptfs", ProbabilisticTFArray, queue_size=1, latch=True)
        self.marker_publisher = None
        if self.publish_markers:
            self.marker_publisher = rospy.Publisher("grasp_target_mode_axes", Marker, queue_size=1, latch=True)
        input_topic = rospy.get_param("~object_ptf_topic", "object_prob_tf")
        self.subscriber = rospy.Subscriber(input_topic, ProbabilisticTF, self.handle_object_ptf, queue_size=1)

    def handle_object_ptf(self, object_message):
        targets = compose_grasp_targets(
            probabilistic_transform_from_msg(object_message),
            self.candidates,
            rotation_covariance_samples=self.rotation_covariance_samples,
            covariance_floor=self.covariance_floor,
        )
        target_messages = [
            probabilistic_transform_to_msg(target, stamp=object_message.header.stamp)
            for target in targets
        ]

        output = ProbabilisticTFArray()
        output.header.stamp = object_message.header.stamp
        output.header.frame_id = object_message.parent_frame_id or object_message.header.frame_id
        output.object_id = self.object_id
        output.transforms = target_messages
        self.targets_publisher.publish(output)
        if self.marker_publisher is not None:
            self.marker_publisher.publish(self.build_marker(output))

    def build_marker(self, target_array):
        marker = Marker()
        marker.header = target_array.header
        marker.ns = "grasp_target_modes"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.008
        marker.pose.orientation.w = 1.0

        colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
        ]
        for target_message in target_array.transforms:
            target = probabilistic_transform_from_msg(target_message)
            origin = target.position_mean
            rotation = rotation_matrix_from_quaternion(target.orientation_mode_wxyz)
            for axis_index, color in enumerate(colors):
                endpoint = origin + self.axis_length * rotation[:, axis_index]
                marker.points.extend(
                    [
                        Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2])),
                        Point(x=float(endpoint[0]), y=float(endpoint[1]), z=float(endpoint[2])),
                    ]
                )
                marker.colors.extend([color, color])
        return marker


def main():
    rospy.init_node("prob_tf_grasp_target_node")
    ProbTFGraspTargetNode()
    rospy.spin()


if __name__ == "__main__":
    main()
