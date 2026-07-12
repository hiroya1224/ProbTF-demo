#!/usr/bin/env python3

import numpy as np
import rospy

from probtf.distributions import BinghamOrientation
from probtf_estimators.evidence_fusion import fuse_transform_evidence
from probtf_estimators.ros_conversions import (
    orientation_distribution_to_msg,
    transform_evidence_from_msg,
)
from probtf_msgs.msg import (
    OrientationDistributionStamped,
    TransformEvidenceStamped,
)


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
        self.publisher = rospy.Publisher(
            "~fused",
            OrientationDistributionStamped,
            queue_size=10,
        )
        self.subscribers = [
            rospy.Subscriber(
                topic,
                TransformEvidenceStamped,
                self._update,
                callback_args=index,
                queue_size=20,
            )
            for index, topic in enumerate(topics)
        ]

    def _update(self, message, index):
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
            if fused.orientation_bingham is None:
                raise ValueError("orientation fusion requires orientation evidence")
            self.publisher.publish(self._to_message(fused, timestamps))
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            rospy.logwarn_throttle(2.0, "ProbTF evidence fusion rejected an update: %s", error)

    def _to_message(self, fused, timestamps):
        sequences = tuple(
            item.sequence
            for item in fused.evidence_provenance
            if item.sequence is not None
        )
        return orientation_distribution_to_msg(
            BinghamOrientation.from_parameter_matrix(fused.orientation_bingham),
            parent_frame_id=fused.parent_frame_id,
            child_frame_id=fused.child_frame_id,
            stamp=max(timestamps) if timestamps else 0.0,
            edge_id="{}__to__{}".format(
                fused.parent_frame_id,
                fused.child_frame_id,
            ),
            authority=self.source_id,
            approximation=fused.approximation,
            provenance=fused.provenance,
            message_type=OrientationDistributionStamped,
            time_factory=rospy.Time.from_sec,
            sequence=max(sequences) if sequences else None,
        )


if __name__ == "__main__":
    rospy.init_node("probtf_fusion")
    ProbTfFusionNode()
    rospy.spin()
