#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from probik_demo.grasp_library import load_grasp_library
from probik_demo.msg import ProbabilisticTF, ProbabilisticTFArray
from probik_demo.ptf_utils import (
    make_bingham_distribution,
    make_probabilistic_tf_message,
    position_covariance_from_msg,
    ptf_mode_quaternion_wxyz,
    pushforward_bingham_right,
    quaternion_multiply_wxyz,
    quaternion_wxyz_from_msg,
    rotation_matrix_from_quaternion,
    vector3_from_msg,
)


class ProbTFGraspTargetNode:
    def __init__(self):
        grasp_library_path = rospy.get_param("~grasp_library_path")
        self.object_id = rospy.get_param("~object_id", "demo_cylinder")
        self.candidates = load_grasp_library(grasp_library_path, self.object_id)
        self.rotation_covariance_samples = int(rospy.get_param("~rotation_covariance_samples", 80))
        self.covariance_floor = float(rospy.get_param("~covariance_floor", 1e-4))
        self.axis_length = float(rospy.get_param("~marker_axis_length", 0.12))

        self.targets_publisher = rospy.Publisher("grasp_target_ptfs", ProbabilisticTFArray, queue_size=1, latch=True)
        self.marker_publisher = rospy.Publisher("grasp_target_mode_axes", Marker, queue_size=1, latch=True)
        input_topic = rospy.get_param("~object_ptf_topic", "object_prob_tf")
        self.subscriber = rospy.Subscriber(input_topic, ProbabilisticTF, self.handle_object_ptf, queue_size=1)

    def handle_object_ptf(self, object_message):
        distribution = make_bingham_distribution(object_message.orientation_bingham)
        object_mode = ptf_mode_quaternion_wxyz(object_message)
        object_rotation_mode = rotation_matrix_from_quaternion(object_mode)
        object_covariance = position_covariance_from_msg(object_message)
        object_mean = vector3_from_msg(object_message.position_mean)

        target_messages = []
        for candidate in self.candidates:
            grasp_offset = vector3_from_msg(candidate.object_to_grasp_position)
            grasp_orientation = quaternion_wxyz_from_msg(candidate.object_to_grasp_orientation)
            target_mean = object_mean + object_rotation_mode @ grasp_offset
            target_covariance = object_covariance + self.covariance_floor * np.eye(3, dtype=float)

            if self.rotation_covariance_samples > 1 and np.linalg.norm(grasp_offset) > 1e-8:
                sampled_quaternions = distribution.update_sample(N_sample=self.rotation_covariance_samples)
                rotated_offsets = np.asarray(
                    [rotation_matrix_from_quaternion(sampled_quaternion) @ grasp_offset for sampled_quaternion in sampled_quaternions],
                    dtype=float,
                )
                target_covariance += np.cov(rotated_offsets.T)

            target_mode = quaternion_multiply_wxyz(object_mode, grasp_orientation)
            target_bingham = pushforward_bingham_right(object_message.orientation_bingham.matrix, grasp_orientation)
            target_messages.append(
                make_probabilistic_tf_message(
                    parent_frame_id=object_message.parent_frame_id or object_message.header.frame_id,
                    child_frame_id=candidate.grasp_id,
                    position_mean_xyz=target_mean,
                    position_covariance=target_covariance,
                    orientation_bingham_matrix=target_bingham,
                    orientation_mode_wxyz=target_mode,
                    stamp=object_message.header.stamp,
                )
            )

        output = ProbabilisticTFArray()
        output.header.stamp = object_message.header.stamp
        output.header.frame_id = object_message.parent_frame_id or object_message.header.frame_id
        output.object_id = self.object_id
        output.transforms = target_messages
        self.targets_publisher.publish(output)
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
            origin = vector3_from_msg(target_message.position_mean)
            rotation = rotation_matrix_from_quaternion(ptf_mode_quaternion_wxyz(target_message))
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
