#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker

from probik_demo.msg import ProbabilisticTF
from probik_demo.ptf_utils import (
    make_bingham_distribution,
    pack_rgb,
    position_covariance_from_msg,
    ptf_mode_quaternion_wxyz,
    rotation_matrix_from_quaternion,
    vector3_from_msg,
)


class ProbabilisticTFVisualizer:
    def __init__(self):
        self.axis_length = float(rospy.get_param("~axis_length", 0.18))
        self.sample_count = int(rospy.get_param("~sample_count", 300))
        seed = int(rospy.get_param("~seed", 7))
        self.rng = np.random.default_rng(seed if seed >= 0 else None)

        input_topic = rospy.get_param("~input_topic", "input_ptf")
        cloud_topic = rospy.get_param("~cloud_topic", "ptf_axes_cloud")
        pose_topic = rospy.get_param("~pose_topic", "ptf_mode_pose")
        marker_topic = rospy.get_param("~marker_topic", "ptf_mode_axes")

        self.point_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        self.axis_colors = [
            pack_rgb(255, 0, 0),
            pack_rgb(0, 255, 0),
            pack_rgb(0, 0, 255),
        ]

        self.cloud_publisher = rospy.Publisher(cloud_topic, PointCloud2, queue_size=1)
        self.pose_publisher = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)
        self.marker_publisher = rospy.Publisher(marker_topic, Marker, queue_size=1)
        self.subscriber = rospy.Subscriber(input_topic, ProbabilisticTF, self.handle_ptf, queue_size=1)

    def handle_ptf(self, message):
        if self.sample_count <= 0:
            rospy.logwarn_throttle(5.0, "sample_count must be positive.")
            return

        header = Header()
        header.frame_id = message.parent_frame_id or message.header.frame_id
        header.stamp = message.header.stamp if message.header.stamp != rospy.Time() else rospy.Time.now()

        mean_translation = vector3_from_msg(message.position_mean)
        covariance = position_covariance_from_msg(message) + 1e-6 * np.eye(3, dtype=float)
        translation_samples = self.rng.multivariate_normal(mean_translation, covariance, size=self.sample_count)

        distribution = make_bingham_distribution(message.orientation_bingham)
        orientation_samples = distribution.update_sample(N_sample=self.sample_count)

        cloud_points = []
        for sample_index, orientation in enumerate(orientation_samples):
            rotation = rotation_matrix_from_quaternion(orientation)
            translation = translation_samples[sample_index]
            for axis_index, color in enumerate(self.axis_colors):
                endpoint = translation + self.axis_length * rotation[:, axis_index]
                cloud_points.append([float(endpoint[0]), float(endpoint[1]), float(endpoint[2]), color])

        self.cloud_publisher.publish(point_cloud2.create_cloud(header, self.point_fields, cloud_points))
        self.pose_publisher.publish(self.build_mode_pose(header, mean_translation, message))
        self.marker_publisher.publish(self.build_mode_marker(header, mean_translation, message))

    def build_mode_pose(self, header, translation, message):
        mode_quaternion = ptf_mode_quaternion_wxyz(message)
        pose = PoseStamped()
        pose.header = header
        pose.pose.position.x = float(translation[0])
        pose.pose.position.y = float(translation[1])
        pose.pose.position.z = float(translation[2])
        pose.pose.orientation.w = float(mode_quaternion[0])
        pose.pose.orientation.x = float(mode_quaternion[1])
        pose.pose.orientation.y = float(mode_quaternion[2])
        pose.pose.orientation.z = float(mode_quaternion[3])
        return pose

    def build_mode_marker(self, header, translation, message):
        mode_quaternion = ptf_mode_quaternion_wxyz(message)
        rotation = rotation_matrix_from_quaternion(mode_quaternion)
        marker = Marker()
        marker.header = header
        marker.ns = f"{message.child_frame_id}_mode_axes"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.pose.orientation.w = 1.0

        origin = np.asarray(translation, dtype=float)
        colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
        ]
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
    rospy.init_node("ptf_visualizer")
    ProbabilisticTFVisualizer()
    rospy.spin()


if __name__ == "__main__":
    main()
