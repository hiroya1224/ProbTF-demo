#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import rospy
import rospkg
from sensor_msgs.msg import JointState

from probtf_ros import ProbTfBroadcaster
from probtf_ros.bridge import PROBTF_STATIC_TOPIC, PROBTF_TOPIC
from probtf_msgs.msg import ProbabilisticTransformArray, ProbabilisticTransformStamped
from symaware_grasp.probtf_config import (
    config_from_mapping,
    load_prob_tf_config,
    load_prob_tf_mapping,
    revolute_joint_names,
)


def _default_config_path():
    return Path(rospkg.RosPack().get_path("symaware_grasp")) / "configs" / "simple_six_dof_prob_tf.yaml"


class ProbTfConfiguredBroadcasterNode:
    def __init__(self):
        self.config_path = Path(rospy.get_param("~config_path", str(_default_config_path())))
        self.authority = rospy.get_name()
        self.source_id = "config:{}".format(self.config_path.name)
        self.follow_joint_states = bool(rospy.get_param("~follow_joint_states", False))
        dynamic_publisher = rospy.Publisher(
            rospy.get_param("~probtf_topic", PROBTF_TOPIC),
            ProbabilisticTransformStamped,
            queue_size=20,
        )
        static_publisher = rospy.Publisher(
            rospy.get_param("~probtf_static_topic", PROBTF_STATIC_TOPIC),
            ProbabilisticTransformArray,
            queue_size=1,
            latch=True,
        )
        self.broadcaster = ProbTfBroadcaster(dynamic_publisher, static_publisher)

        if not self.follow_joint_states:
            config = load_prob_tf_config(
                self.config_path,
                stamp=0.0,
                authority=self.authority,
            )
            self.broadcaster.send_transforms(config.records)
            rospy.loginfo(
                "Published %d native v2 static ProbTF records from %s",
                len(config.records),
                self.config_path,
            )
            return

        self.mapping = load_prob_tf_mapping(self.config_path)
        self.joint_names = revolute_joint_names(self.mapping)
        self.joint_positions = self._initial_joint_positions()

        baseline = load_prob_tf_config(
            self.config_path,
            stamp=0.0,
            authority=self.authority,
        )
        dynamic_edge_ids = set(self.joint_names)
        fixed_records = tuple(
            record for record in baseline.records if record.edge_id not in dynamic_edge_ids
        )
        self.broadcaster.send_transforms(fixed_records)
        self._publish_joint_records(rospy.Time.now().to_sec())
        self.subscriber = rospy.Subscriber(
            rospy.get_param("~joint_states_topic", "joint_states"),
            JointState,
            self.handle_joint_state,
            queue_size=1,
        )
        rospy.loginfo(
            "Publishing %d joint-driven native v2 ProbTF records and %d fixed static records from %s",
            len(self.joint_names),
            len(fixed_records),
            self.config_path,
        )

    def _initial_joint_positions(self):
        configured = rospy.get_param("~initial_positions", [0.0] * len(self.joint_names))
        if isinstance(configured, dict):
            unknown = set(configured) - set(self.joint_names)
            if unknown:
                raise ValueError("Unknown initial joint positions: {}.".format(sorted(unknown)))
            positions = {
                joint_name: float(configured.get(joint_name, 0.0))
                for joint_name in self.joint_names
            }
        else:
            values = np.asarray(configured, dtype=float).reshape(-1)
            if values.shape != (len(self.joint_names),):
                raise ValueError(
                    "initial_positions must contain {} values.".format(len(self.joint_names))
                )
            positions = dict(zip(self.joint_names, values.tolist()))
        if not all(np.isfinite(value) for value in positions.values()):
            raise ValueError("initial_positions must be finite.")
        return positions

    def _publish_joint_records(self, stamp):
        config = config_from_mapping(
            self.mapping,
            stamp=float(stamp),
            authority=self.authority,
            source_id=self.source_id,
            joint_positions=self.joint_positions,
            dynamic_joints=True,
        )
        dynamic_records = tuple(record for record in config.records if not record.is_static)
        self.broadcaster.send_transforms(dynamic_records)

    def handle_joint_state(self, message):
        if not message.position:
            return
        updated_positions = dict(self.joint_positions)
        if message.name:
            name_to_position = dict(zip(message.name, message.position))
            for joint_name in self.joint_names:
                if joint_name in name_to_position:
                    updated_positions[joint_name] = float(name_to_position[joint_name])
        elif len(message.position) == len(self.joint_names):
            updated_positions.update(zip(self.joint_names, map(float, message.position)))
        else:
            rospy.logwarn_throttle(
                5.0,
                "Ignoring unnamed JointState with %d positions; expected %d.",
                len(message.position),
                len(self.joint_names),
            )
            return
        if not all(np.isfinite(value) for value in updated_positions.values()):
            rospy.logwarn_throttle(5.0, "Ignoring JointState containing non-finite positions.")
            return
        self.joint_positions = updated_positions
        stamp = message.header.stamp.to_sec()
        self._publish_joint_records(stamp if stamp > 0.0 else rospy.Time.now().to_sec())


def main():
    rospy.init_node("probtf_static_broadcaster")
    ProbTfConfiguredBroadcasterNode()
    rospy.spin()


if __name__ == "__main__":
    main()
