#!/usr/bin/env python3

from pathlib import Path

import rospy
import rospkg

from probtf_ros import ProbTfBroadcaster
from probtf_ros.bridge import PROBTF_STATIC_TOPIC, PROBTF_TOPIC
from probtf_msgs.msg import ProbabilisticTransformArray, ProbabilisticTransformStamped
from symaware_grasp.probtf_config import load_prob_tf_config


def _default_config_path():
    return Path(rospkg.RosPack().get_path("symaware_grasp")) / "configs" / "simple_six_dof_prob_tf.yaml"


class ProbTfStaticBroadcasterNode:
    def __init__(self):
        config_path = Path(rospy.get_param("~config_path", str(_default_config_path())))
        config = load_prob_tf_config(
            config_path,
            stamp=0.0,
            authority=rospy.get_name(),
        )
        dynamic_publisher = rospy.Publisher(
            rospy.get_param("~probtf_topic", PROBTF_TOPIC),
            ProbabilisticTransformStamped,
            queue_size=1,
        )
        static_publisher = rospy.Publisher(
            rospy.get_param("~probtf_static_topic", PROBTF_STATIC_TOPIC),
            ProbabilisticTransformArray,
            queue_size=1,
            latch=True,
        )
        self.broadcaster = ProbTfBroadcaster(dynamic_publisher, static_publisher)
        self.broadcaster.send_transforms(config.records)
        rospy.loginfo("Published %d native v2 static ProbTF records from %s", len(config.records), config_path)


def main():
    rospy.init_node("probtf_static_broadcaster")
    ProbTfStaticBroadcasterNode()
    rospy.spin()


if __name__ == "__main__":
    main()
