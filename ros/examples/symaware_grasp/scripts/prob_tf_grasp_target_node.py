#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from probtf.distributions import DistributionStatus
from probtf.geometry import quat_to_rotmat
from probtf_ros import ProbTfBroadcaster, RosProbTfListener
from probtf_ros.bridge import PROBTF_STATIC_TOPIC, PROBTF_TOPIC
from probtf_msgs.msg import ProbabilisticTransformArray, ProbabilisticTransformStamped
from symaware_grasp.grasp_library import load_grasp_library
from symaware_grasp.grasp_targets import compose_grasp_targets
from symaware_grasp.msg import GraspTargetArray, ObjectBelief
from symaware_grasp.runtime import lookup_message_record
from symaware_grasp_ros import grasp_targets_to_msg


class ProbTFGraspTargetNode:
    def __init__(self):
        self.object_id = str(rospy.get_param("~object_id", "demo_cylinder"))
        self.candidates = load_grasp_library(
            rospy.get_param("~grasp_library_path"),
            self.object_id,
        )
        self.axis_length = float(rospy.get_param("~marker_axis_length", 0.12))
        self.lookup_timeout = float(rospy.get_param("~lookup_timeout", 2.0))
        self.publish_markers = bool(rospy.get_param("~publish_markers", False))
        dynamic_topic = rospy.get_param("~probtf_topic", PROBTF_TOPIC)
        static_topic = rospy.get_param("~probtf_static_topic", PROBTF_STATIC_TOPIC)

        self.listener = RosProbTfListener(
            dynamic_topic=dynamic_topic,
            static_topic=static_topic,
        )
        dynamic_publisher = rospy.Publisher(
            dynamic_topic,
            ProbabilisticTransformStamped,
            queue_size=20,
        )
        static_publisher = rospy.Publisher(
            static_topic,
            ProbabilisticTransformArray,
            queue_size=1,
            latch=True,
        )
        self.broadcaster = ProbTfBroadcaster(dynamic_publisher, static_publisher)
        self.targets_publisher = rospy.Publisher(
            rospy.get_param("~grasp_targets_topic", "/symaware_grasp/grasp_targets"),
            GraspTargetArray,
            queue_size=1,
            latch=True,
        )
        self.marker_publisher = None
        if self.publish_markers:
            self.marker_publisher = rospy.Publisher(
                rospy.get_param("~marker_topic", "/symaware_grasp/grasp_target_mode_axes"),
                Marker,
                queue_size=1,
                latch=True,
            )
        self.subscriber = rospy.Subscriber(
            rospy.get_param("~object_belief_topic", "/symaware_grasp/object_belief"),
            ObjectBelief,
            self.handle_object_belief,
            queue_size=1,
        )

    def handle_object_belief(self, object_message):
        if object_message.object_id != self.object_id:
            rospy.logwarn_throttle(5.0, "Ignoring object belief for '%s'.", object_message.object_id)
            return
        try:
            object_record = lookup_message_record(
                self.listener,
                object_message.transform,
                timeout=self.lookup_timeout,
            )
        except (RuntimeError, ValueError) as exc:
            rospy.logwarn("Cannot resolve object belief in ProbTF graph: %s", exc)
            return
        targets = compose_grasp_targets(object_record, self.candidates)
        self.broadcaster.send_transforms(targets)
        output = grasp_targets_to_msg(targets, self.candidates, self.object_id)
        self.targets_publisher.publish(output)
        if self.marker_publisher is not None:
            self.marker_publisher.publish(self.build_marker(output.header, targets))

    def build_marker(self, header, targets):
        marker = Marker()
        marker.header = header
        marker.ns = "grasp_target_component_modes"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.008
        marker.pose.orientation.w = 1.0
        colors = (
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
        )
        for target in targets:
            normalized = target.distribution.normalize_weights()
            if normalized.status is not DistributionStatus.OK:
                continue
            for weighted in normalized.components:
                component = weighted.component
                quaternion = component.orientation.mode_wxyz
                origin = component.conditional_translation_mean(quaternion)
                rotation = quat_to_rotmat(quaternion)
                alpha = max(0.2, min(1.0, weighted.weight))
                for axis_index, base_color in enumerate(colors):
                    endpoint = origin + self.axis_length * rotation[:, axis_index]
                    color = ColorRGBA(
                        r=base_color.r,
                        g=base_color.g,
                        b=base_color.b,
                        a=alpha,
                    )
                    marker.points.extend(
                        (
                            Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2])),
                            Point(x=float(endpoint[0]), y=float(endpoint[1]), z=float(endpoint[2])),
                        )
                    )
                    marker.colors.extend((color, color))
        return marker


def main():
    rospy.init_node("prob_tf_grasp_target_node")
    ProbTFGraspTargetNode()
    rospy.spin()


if __name__ == "__main__":
    main()
