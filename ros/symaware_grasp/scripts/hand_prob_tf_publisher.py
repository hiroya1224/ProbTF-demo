#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import JointState

from symaware_grasp.arm_kinematics import ToyArm6DOF
from symaware_grasp.ee_belief import EndEffectorBeliefModel
from probik_msgs.msg import ProbabilisticTF
from symaware_grasp.ptf_utils import make_probabilistic_tf_message


def _get_belief_param(name, default):
    return rospy.get_param("~" + name, rospy.get_param("/symaware_grasp/hand_belief/" + name, default))


class HandProbTFPublisher:
    def __init__(self):
        self.robot_model = ToyArm6DOF()
        self.parent_frame_id = rospy.get_param("~parent_frame_id", "base_link")
        self.child_frame_id = rospy.get_param("~child_frame_id", "tool0_prob")
        self.publish_rate = float(rospy.get_param("~publish_rate", 8.0))

        joint_noise_stddev = _get_belief_param("joint_noise_stddev", [0.03, 0.03, 0.03, 0.05, 0.05, 0.05])
        if isinstance(joint_noise_stddev, list):
            joint_noise_stddev = np.asarray(joint_noise_stddev, dtype=float)
        else:
            joint_noise_stddev = np.full(self.robot_model.dof, float(joint_noise_stddev), dtype=float)
        self.belief_model = EndEffectorBeliefModel(
            robot_model=self.robot_model,
            joint_noise_stddev=joint_noise_stddev,
            position_covariance_floor=float(_get_belief_param("position_covariance_floor", 5e-5)),
            sample_count=int(_get_belief_param("sample_count", 36)),
            sample_seed=int(_get_belief_param("seed", 23)),
            bingham_integration_steps=int(_get_belief_param("bingham_integration_steps", 80)),
            bingham_fit_max_iterations=int(_get_belief_param("bingham_fit_max_iterations", 40)),
            orientation_initial_concentrations=_get_belief_param(
                "orientation_concentrations",
                [420.0, 320.0, 220.0],
            ),
        )

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

        estimate = self.belief_model.estimate_distribution(self.latest_joint_positions)
        message = make_probabilistic_tf_message(
            parent_frame_id=self.parent_frame_id,
            child_frame_id=self.child_frame_id,
            position_mean_xyz=estimate["position_mean"],
            position_covariance=estimate["position_covariance"],
            orientation_bingham_matrix=estimate["orientation_bingham"],
            orientation_mode_wxyz=estimate["orientation_mode"],
            stamp=rospy.Time.now(),
        )
        self.publisher.publish(message)


def main():
    rospy.init_node("hand_prob_tf_publisher")
    HandProbTFPublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
