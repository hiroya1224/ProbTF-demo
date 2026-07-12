#!/usr/bin/env python3

import numpy as np
import rospy

from probtf.bingham import bingham_mode, canonical_bingham_parameter
from probtf_estimators.evidence_fusion import fuse_transform_evidence
from probtf_estimators.ros_conversions import transform_evidence_from_msg
from probtf_msgs.msg import ProbabilisticTF, TransformEvidence


class ProbTfFusionNode:
    def __init__(self):
        topics = rospy.get_param("~evidence_topics")
        if not isinstance(topics, list) or not topics:
            raise ValueError("~evidence_topics must be a non-empty list")
        if len(set(topics)) != len(topics):
            raise ValueError("~evidence_topics must not contain duplicates")
        self.source_id = rospy.get_param("~source_id", "probtf_fusion")
        self.maximum_stamp_skew = float(rospy.get_param("~maximum_stamp_skew", 0.1))
        self.latest = [None] * len(topics)
        self.publisher = rospy.Publisher("~fused", ProbabilisticTF, queue_size=10)
        self.subscribers = [
            rospy.Subscriber(
                topic,
                TransformEvidence,
                self._update,
                callback_args=index,
                queue_size=20,
            )
            for index, topic in enumerate(topics)
        ]

    def _update(self, message, index):
        if message.header.frame_id and message.header.frame_id != message.parent_frame_id:
            rospy.logwarn_throttle(2.0, "Ignoring transform evidence with conflicting parent fields")
            return
        try:
            self.latest[index] = transform_evidence_from_msg(message)
            if any(evidence is None for evidence in self.latest):
                return
            timestamps = [
                evidence.timestamp
                for evidence in self.latest
                if evidence.timestamp is not None
            ]
            if timestamps and max(timestamps) - min(timestamps) > self.maximum_stamp_skew:
                rospy.logwarn_throttle(2.0, "ProbTF evidence timestamps exceed maximum skew")
                return
            fused = fuse_transform_evidence(self.latest)
            self.publisher.publish(self._to_message(fused, timestamps))
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            rospy.logwarn_throttle(2.0, "ProbTF evidence fusion rejected an update: %s", error)

    def _to_message(self, fused, timestamps):
        message = ProbabilisticTF()
        message.header.frame_id = fused.parent_frame_id
        if timestamps:
            message.header.stamp = rospy.Time.from_sec(max(timestamps))
        message.parent_frame_id = fused.parent_frame_id
        message.child_frame_id = fused.child_frame_id
        message.edge_id = "{}__to__{}".format(
            fused.parent_frame_id,
            fused.child_frame_id,
        )
        message.source_id = self.source_id
        message.evidence_source_ids = list(fused.source_ids)
        message.has_position = fused.position_information is not None
        if message.has_position:
            try:
                mean, covariance = fused.gaussian_position()
            except ValueError:
                message.has_position = False
            else:
                message.position_mean.x = float(mean[0])
                message.position_mean.y = float(mean[1])
                message.position_mean.z = float(mean[2])
                message.position_covariance = covariance.reshape(-1).tolist()
        message.has_orientation = fused.orientation_bingham is not None
        if message.has_orientation:
            parameter = canonical_bingham_parameter(fused.orientation_bingham)
            message.orientation_bingham.matrix = parameter.reshape(-1).tolist()
            mode = bingham_mode(parameter)
            message.orientation_mode.w = float(mode[0])
            message.orientation_mode.x = float(mode[1])
            message.orientation_mode.y = float(mode[2])
            message.orientation_mode.z = float(mode[3])
        message.approximation_type = "independent_likelihood_product"
        message.closure_approximation = False
        return message


if __name__ == "__main__":
    rospy.init_node("probtf_fusion")
    ProbTfFusionNode()
    rospy.spin()
