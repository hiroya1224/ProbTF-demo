#!/usr/bin/env python3

import math

import numpy as np
import rospy

from probtf.geometry import rpy_to_quat
from probtf.provenance import ApproximationInfo, ApproximationKind
from probtf_ros import ProbTfBroadcaster
from probtf_ros.bridge import PROBTF_STATIC_TOPIC, PROBTF_TOPIC
from probtf_msgs.msg import ProbabilisticTransformArray, ProbabilisticTransformStamped
from symaware_grasp.beliefs import make_transform_record
from symaware_grasp.geometry_utils import axially_symmetric_bingham_parameter
from symaware_grasp.msg import ObjectBelief
from symaware_grasp_ros import object_belief_to_msg


class ObjectPoseNode:
    def __init__(self):
        self.object_id = str(rospy.get_param("~object_id", "demo_cylinder"))
        self.parent_frame_id = str(rospy.get_param("~parent_frame_id", "base_link"))
        self.child_frame_id = str(rospy.get_param("~child_frame_id", self.object_id))
        self.edge_id = str(rospy.get_param("~edge_id", "symaware_object_pose"))
        self.publish_rate = float(rospy.get_param("~publish_rate", 1.0))

        self.position_mean = np.asarray(
            rospy.get_param("~position_mean", [0.58, 0.12, 0.46]),
            dtype=float,
        ).reshape(3)
        self.position_covariance = np.asarray(
            rospy.get_param(
                "~position_covariance",
                [0.0015, 0.0002, 0.0, 0.0002, 0.0010, 0.0, 0.0, 0.0, 0.0008],
            ),
            dtype=float,
        ).reshape(3, 3)
        mode_rpy_deg = rospy.get_param("~mode_rpy_deg", [0.0, 0.0, 40.0])
        self.mode_quaternion = rpy_to_quat(
            math.radians(mode_rpy_deg[0]),
            math.radians(mode_rpy_deg[1]),
            math.radians(mode_rpy_deg[2]),
        )
        self.bingham_parameter = axially_symmetric_bingham_parameter(
            self.mode_quaternion,
            rospy.get_param("~symmetry_axis_body", [0.0, 0.0, 1.0]),
            rospy.get_param("~bingham_concentrations", [220.0, 220.0, 4.0]),
        )

        dynamic_topic = rospy.get_param("~probtf_topic", PROBTF_TOPIC)
        static_topic = rospy.get_param("~probtf_static_topic", PROBTF_STATIC_TOPIC)
        dynamic_publisher = rospy.Publisher(
            dynamic_topic,
            ProbabilisticTransformStamped,
            queue_size=10,
        )
        static_publisher = rospy.Publisher(
            static_topic,
            ProbabilisticTransformArray,
            queue_size=1,
            latch=True,
        )
        self.broadcaster = ProbTfBroadcaster(dynamic_publisher, static_publisher)
        self.publisher = rospy.Publisher(
            rospy.get_param("~object_belief_topic", "/symaware_grasp/object_belief"),
            ObjectBelief,
            queue_size=1,
            latch=True,
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.publish_rate, 1e-3)),
            self.publish_message,
        )

    def publish_message(self, _event):
        now = rospy.Time.now()
        record = make_transform_record(
            parent_frame_id=self.parent_frame_id,
            child_frame_id=self.child_frame_id,
            stamp=now.to_sec(),
            edge_id=self.edge_id,
            authority=rospy.get_name(),
            position_mean=self.position_mean,
            position_covariance=self.position_covariance,
            orientation_parameter=self.bingham_parameter,
            orientation_reference_wxyz=self.mode_quaternion,
            source_id="object_pose_parameters",
            component_id="{}:pose_belief".format(self.object_id),
            approximation=ApproximationInfo(
                kind=ApproximationKind.PRODUCER_SUPPLIED,
                detail="Object belief parameters were supplied by the demo configuration.",
                source="object_pose_node",
            ),
        )
        self.broadcaster.send_transform(record)
        self.publisher.publish(object_belief_to_msg(record, self.object_id))


def main():
    rospy.init_node("object_pose_node")
    ObjectPoseNode()
    rospy.spin()


if __name__ == "__main__":
    main()
