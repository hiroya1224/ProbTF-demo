#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker

from probtf.distributions import DistributionStatus
from probtf.geometry import quat_to_rotmat
from probtf.probability import apply_transform_samples, sample_transform_distribution
from probtf_ros import RosProbTfListener
from probtf_ros.bridge import PROBTF_STATIC_TOPIC, PROBTF_TOPIC
from symaware_grasp.beliefs import representative_component
from symaware_grasp.msg import ObjectBelief, SelectedGraspTarget
from symaware_grasp.runtime import lookup_message_record
from symaware_grasp.visualization import pack_rgb


_INPUT_TYPES = {
    "object": ObjectBelief,
    "selected_target": SelectedGraspTarget,
}


class ProbTfV2Visualizer:
    def __init__(self):
        self.axis_length = float(rospy.get_param("~axis_length", 0.18))
        self.sample_count = int(rospy.get_param("~sample_count", 300))
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive.")
        seed = int(rospy.get_param("~seed", 7))
        self.rng = np.random.default_rng(seed if seed >= 0 else None)
        self.lookup_timeout = float(rospy.get_param("~lookup_timeout", 2.0))
        input_kind = str(rospy.get_param("~input_kind", "object")).strip().lower()
        if input_kind not in _INPUT_TYPES:
            raise ValueError(
                "input_kind must be one of: {}.".format(
                    ", ".join(sorted(_INPUT_TYPES))
                )
            )

        self.listener = RosProbTfListener(
            dynamic_topic=rospy.get_param("~probtf_topic", PROBTF_TOPIC),
            static_topic=rospy.get_param("~probtf_static_topic", PROBTF_STATIC_TOPIC),
        )
        self.point_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        self.axis_colors = (
            pack_rgb(255, 0, 0),
            pack_rgb(0, 255, 0),
            pack_rgb(0, 0, 255),
        )
        self.cloud_publisher = rospy.Publisher(
            rospy.get_param("~cloud_topic", "ptf_axes_cloud"),
            PointCloud2,
            queue_size=1,
        )
        self.pose_publisher = rospy.Publisher(
            rospy.get_param("~pose_topic", "ptf_mode_pose"),
            PoseStamped,
            queue_size=1,
        )
        self.marker_publisher = rospy.Publisher(
            rospy.get_param("~marker_topic", "ptf_mode_axes"),
            Marker,
            queue_size=1,
        )
        self.geometry_marker_publisher = None
        geometry_marker_topic = str(
            rospy.get_param("~geometry_marker_topic", "")
        ).strip()
        if geometry_marker_topic:
            self.geometry_type = (
                str(rospy.get_param("~geometry_type", "cylinder")).strip().lower()
            )
            if self.geometry_type != "cylinder":
                raise ValueError("Only cylinder geometry is supported by this demo visualizer.")
            self.geometry_scale = np.asarray(
                rospy.get_param("~geometry_scale", [0.10, 0.10, 0.18]),
                dtype=float,
            ).reshape(3)
            self.geometry_color = np.asarray(
                rospy.get_param("~geometry_color", [0.18, 0.52, 0.82, 0.82]),
                dtype=float,
            ).reshape(4)
            if not np.all(np.isfinite(self.geometry_scale)) or np.any(
                self.geometry_scale <= 0.0
            ):
                raise ValueError("geometry_scale must contain three finite positive values.")
            if not np.all(np.isfinite(self.geometry_color)) or np.any(
                (self.geometry_color < 0.0) | (self.geometry_color > 1.0)
            ):
                raise ValueError("geometry_color must contain four values in [0, 1].")
            self.geometry_marker_publisher = rospy.Publisher(
                geometry_marker_topic,
                Marker,
                queue_size=1,
                latch=True,
            )
        self.subscriber = rospy.Subscriber(
            rospy.get_param("~input_topic", "/symaware_grasp/object_belief"),
            _INPUT_TYPES[input_kind],
            self.handle_belief,
            queue_size=1,
        )

    def handle_belief(self, message):
        try:
            record = lookup_message_record(
                self.listener,
                message.transform,
                timeout=self.lookup_timeout,
            )
        except (RuntimeError, ValueError) as exc:
            rospy.logwarn("Cannot resolve visualization belief in ProbTF graph: %s", exc)
            return
        samples = sample_transform_distribution(record.distribution, self.sample_count, self.rng)
        cloud_points = []
        for axis_index, color in enumerate(self.axis_colors):
            local_points = np.zeros((self.sample_count, 3), dtype=float)
            local_points[:, axis_index] = self.axis_length
            endpoints = apply_transform_samples(samples, local_points)
            cloud_points.extend(
                [float(point[0]), float(point[1]), float(point[2]), color]
                for point in endpoints
            )

        header = Header()
        header.frame_id = record.parent_frame_id
        header.stamp = message.transform.header.stamp
        self.cloud_publisher.publish(
            point_cloud2.create_cloud(header, self.point_fields, cloud_points)
        )
        _, translation, quaternion = representative_component(record)
        self.pose_publisher.publish(self.build_mode_pose(header, translation, quaternion))
        self.marker_publisher.publish(self.build_component_marker(header, record))
        if self.geometry_marker_publisher is not None:
            self.geometry_marker_publisher.publish(
                self.build_geometry_marker(
                    header,
                    translation,
                    quaternion,
                    record.child_frame_id,
                )
            )

    @staticmethod
    def build_mode_pose(header, translation, quaternion):
        pose = PoseStamped()
        pose.header = header
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (
            float(value) for value in translation
        )
        pose.pose.orientation.w = float(quaternion[0])
        pose.pose.orientation.x = float(quaternion[1])
        pose.pose.orientation.y = float(quaternion[2])
        pose.pose.orientation.z = float(quaternion[3])
        return pose

    def build_component_marker(self, header, record):
        marker = Marker()
        marker.header = header
        marker.ns = "{}_component_modes".format(record.child_frame_id)
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.pose.orientation.w = 1.0
        colors = (
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
        )
        normalized = record.distribution.normalize_weights()
        if normalized.status is not DistributionStatus.OK:
            return marker
        for weighted in normalized.components:
            component = weighted.component
            quaternion = component.orientation.mode_wxyz
            origin = component.conditional_translation_mean(quaternion)
            rotation = quat_to_rotmat(quaternion)
            for axis_index, base_color in enumerate(colors):
                endpoint = origin + self.axis_length * rotation[:, axis_index]
                color = ColorRGBA(
                    r=base_color.r,
                    g=base_color.g,
                    b=base_color.b,
                    a=max(0.2, min(1.0, weighted.weight)),
                )
                marker.points.extend(
                    (
                        Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2])),
                        Point(x=float(endpoint[0]), y=float(endpoint[1]), z=float(endpoint[2])),
                    )
                )
                marker.colors.extend((color, color))
        return marker

    def build_geometry_marker(self, header, translation, quaternion, child_frame_id):
        marker = Marker()
        marker.header = header
        marker.ns = "{}_geometry".format(child_frame_id)
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
            float(value) for value in translation
        )
        marker.pose.orientation.w = float(quaternion[0])
        marker.pose.orientation.x = float(quaternion[1])
        marker.pose.orientation.y = float(quaternion[2])
        marker.pose.orientation.z = float(quaternion[3])
        marker.scale.x, marker.scale.y, marker.scale.z = (
            float(value) for value in self.geometry_scale
        )
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (
            float(value) for value in self.geometry_color
        )
        return marker


def main():
    rospy.init_node("ptf_visualizer")
    ProbTfV2Visualizer()
    rospy.spin()


if __name__ == "__main__":
    main()
