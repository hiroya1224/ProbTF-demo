#!/usr/bin/env python3
"""Publish a complete D2 box law and an independently fixed true box marker."""
import math

import numpy as np
import rospy
from visualization_msgs.msg import Marker

from probtf.geometry import rpy_to_quat
from probtf_ros import ProbTfBroadcaster
from probtf_ros.bridge import PROBTF_STATIC_TOPIC, PROBTF_TOPIC
from probtf_msgs.msg import ProbabilisticTransformArray, ProbabilisticTransformStamped
from symaware_grasp.box_geometry import BOX_CONCENTRATION, BOX_DIMENSIONS, BOX_POSITION, BOX_YAW_DEG, make_box_record
from symaware_grasp.msg import ObjectBelief
from symaware_grasp_ros import object_belief_to_msg


class BoxPoseNode:
    def __init__(self):
        self.object_id = str(rospy.get_param("~object_id", "demo_box"))
        self.parent_frame_id = str(rospy.get_param("~parent_frame_id", "base_link"))
        self.child_frame_id = str(rospy.get_param("~child_frame_id", self.object_id))
        self.edge_id = str(rospy.get_param("~edge_id", "symaware_box_pose"))
        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))
        if not math.isfinite(self.publish_rate) or self.publish_rate <= 0.0:
            raise ValueError("publish_rate must be finite and positive")
        self.distribution = str(rospy.get_param("~distribution", "mixture"))
        self.position = np.asarray(rospy.get_param("~position_mean", list(BOX_POSITION)), dtype=float).reshape(3)
        self.position_covariance = np.asarray(rospy.get_param("~position_covariance", [0.0] * 9), dtype=float).reshape(3, 3)
        self.dimensions = np.asarray(rospy.get_param("~dimensions", list(BOX_DIMENSIONS)), dtype=float).reshape(3)
        self.mode_rpy_deg = np.asarray(rospy.get_param("~mode_rpy_deg", [0.0, 0.0, BOX_YAW_DEG]), dtype=float).reshape(3)
        self.concentration = float(rospy.get_param("~concentration", BOX_CONCENTRATION))
        self.true_quaternion = rpy_to_quat(*(math.radians(value) for value in self.mode_rpy_deg))
        self.geometry_color = np.asarray(rospy.get_param("~geometry_color", [0.18, 0.52, 0.82, 0.82]), dtype=float).reshape(4)
        if not np.all(np.isfinite(self.geometry_color)) or np.any((self.geometry_color < 0.0) | (self.geometry_color > 1.0)):
            raise ValueError("geometry_color must contain four finite values in [0, 1]")
        self.prototype = make_box_record(
            distribution=self.distribution, position=self.position, position_covariance=self.position_covariance,
            mode_rpy_deg=self.mode_rpy_deg, dimensions=self.dimensions, concentration=self.concentration,
            parent_frame_id=self.parent_frame_id, child_frame_id=self.child_frame_id,
            edge_id=self.edge_id, authority=rospy.get_name(),
        )
        self.parameters = dict(distribution=self.distribution, position=self.position,
                               position_covariance=self.position_covariance, mode_rpy_deg=self.mode_rpy_deg,
                               dimensions=self.dimensions, concentration=self.concentration,
                               parent_frame_id=self.parent_frame_id, child_frame_id=self.child_frame_id,
                               edge_id=self.edge_id, authority=rospy.get_name())
        components = []
        for item in self.prototype.distribution.normalized_components():
            orientation = item.component.orientation
            components.append({"component_id": item.component.component_id, "weight": float(item.weight),
                               "mode_wxyz": orientation.reference_quaternion_wxyz.tolist(),
                               "parameter_matrix": orientation.backend_parameter_matrix().tolist()})
        rospy.set_param("~components", components)
        rospy.set_param("~component_count", len(components))
        rospy.set_param("~true_position", self.position.tolist())
        rospy.set_param("~true_quaternion_wxyz", self.true_quaternion.tolist())
        self.broadcaster = ProbTfBroadcaster(
            rospy.Publisher(rospy.get_param("~probtf_topic", PROBTF_TOPIC), ProbabilisticTransformStamped, queue_size=10),
            rospy.Publisher(rospy.get_param("~probtf_static_topic", PROBTF_STATIC_TOPIC), ProbabilisticTransformArray, queue_size=1, latch=True),
        )
        self.publisher = rospy.Publisher(rospy.get_param("~object_belief_topic", "/symaware_grasp/object_belief"), ObjectBelief, queue_size=1, latch=True)
        self.geometry_publisher = rospy.Publisher(rospy.get_param("~geometry_marker_topic", "/symaware_grasp/object_geometry"), Marker, queue_size=1, latch=True)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish_message)

    def build_geometry_marker(self, now):
        marker = Marker()
        marker.header.frame_id, marker.header.stamp = self.parent_frame_id, now
        marker.ns, marker.id = self.child_frame_id + "_geometry", 0
        marker.type, marker.action = Marker.CUBE, Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, self.position)
        marker.pose.orientation.w, marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z = map(float, self.true_quaternion)
        marker.scale.x, marker.scale.y, marker.scale.z = map(float, self.dimensions)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = map(float, self.geometry_color)
        return marker

    def publish_message(self, _event):
        now = rospy.Time.now()
        record = make_box_record(stamp=now.to_sec(), **self.parameters)
        self.broadcaster.send_transform(record)
        self.publisher.publish(object_belief_to_msg(record, self.object_id))
        self.geometry_publisher.publish(self.build_geometry_marker(now))


def main():
    rospy.init_node("box_pose_node")
    BoxPoseNode()
    rospy.spin()


if __name__ == "__main__":
    main()
