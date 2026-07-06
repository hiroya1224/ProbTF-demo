#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import JointState

from probik_demo.arm_kinematics import ToyArm6DOF
from probik_demo.msg import ProbabilisticTF
from probik_demo.ptf_utils import demo_bingham_matrix, make_probabilistic_tf_message


class HandProbTFPublisher:
    def __init__(self):
        self.robot_model = ToyArm6DOF()
        self.parent_frame_id = rospy.get_param("~parent_frame_id", "base_link")
        self.child_frame_id = rospy.get_param("~child_frame_id", "tool0_prob")
        self.publish_rate = float(rospy.get_param("~publish_rate", 8.0))
        self.sample_count = int(rospy.get_param("~sample_count", 60))
        seed = int(rospy.get_param("~seed", 23))
        self.rng = np.random.default_rng(seed if seed >= 0 else None)

        joint_noise_stddev = rospy.get_param("~joint_noise_stddev", [0.03, 0.03, 0.03, 0.05, 0.05, 0.05])
        if isinstance(joint_noise_stddev, list):
            self.joint_noise_stddev = np.asarray(joint_noise_stddev, dtype=float)
        else:
            self.joint_noise_stddev = np.full(self.robot_model.dof, float(joint_noise_stddev), dtype=float)

        self.orientation_concentrations = rospy.get_param("~orientation_concentrations", [420.0, 320.0, 220.0])
        self.position_covariance_floor = float(rospy.get_param("~position_covariance_floor", 5e-5))

        self.latest_joint_positions = None
        self.publisher = rospy.Publisher("hand_prob_tf", ProbabilisticTF, queue_size=1)
        self.subscriber = rospy.Subscriber("joint_states", JointState, self.handle_joint_state, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate, 1e-3)), self.publish_message)

    def handle_joint_state(self, message):
        if not message.position:
            return
        positions = np.zeros(self.robot_model.dof, dtype=float)
        name_to_index = {name: index for index, name in enumerate(message.name)}
        for joint_index, joint_name in enumerate(self.robot_model.joint_names):
            if joint_name in name_to_index:
                positions[joint_index] = message.position[name_to_index[joint_name]]
        self.latest_joint_positions = self.robot_model.clip_to_limits(positions)

    def publish_message(self, _event):
        if self.latest_joint_positions is None:
            return

        position_mode, quaternion_mode, _ = self.robot_model.forward_kinematics(self.latest_joint_positions)
        sampled_joint_positions = self.rng.normal(
            loc=self.latest_joint_positions,
            scale=self.joint_noise_stddev,
            size=(max(self.sample_count, 2), self.robot_model.dof),
        )
        sampled_joint_positions = np.asarray(
            [self.robot_model.clip_to_limits(sample) for sample in sampled_joint_positions],
            dtype=float,
        )
        sampled_positions = np.asarray(
            [self.robot_model.forward_kinematics(sample)[0] for sample in sampled_joint_positions],
            dtype=float,
        )
        position_covariance = np.cov(sampled_positions.T) + self.position_covariance_floor * np.eye(3, dtype=float)
        orientation_bingham = demo_bingham_matrix(quaternion_mode, self.orientation_concentrations)
        message = make_probabilistic_tf_message(
            parent_frame_id=self.parent_frame_id,
            child_frame_id=self.child_frame_id,
            position_mean_xyz=position_mode,
            position_covariance=position_covariance,
            orientation_bingham_matrix=orientation_bingham,
            orientation_mode_wxyz=quaternion_mode,
            stamp=rospy.Time.now(),
        )
        self.publisher.publish(message)


def main():
    rospy.init_node("hand_prob_tf_publisher")
    HandProbTFPublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
