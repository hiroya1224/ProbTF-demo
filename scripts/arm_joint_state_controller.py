#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import JointState

from probik_demo.arm_kinematics import ToyArm6DOF


class ArmJointStateController:
    def __init__(self):
        self.robot_model = ToyArm6DOF()
        initial_positions = rospy.get_param("~initial_positions", [0.6, -1.0, 1.2, 0.2, -0.5, 0.8])
        self.current_positions = self.robot_model.clip_to_limits(initial_positions)
        self.target_positions = self.current_positions.copy()
        self.publish_rate = float(rospy.get_param("~publish_rate", 30.0))

        max_joint_speed = rospy.get_param("~max_joint_speed", 0.8)
        if isinstance(max_joint_speed, list):
            self.max_joint_speed = np.asarray(max_joint_speed, dtype=float)
        else:
            self.max_joint_speed = np.full(self.robot_model.dof, float(max_joint_speed), dtype=float)

        self.publisher = rospy.Publisher("joint_states", JointState, queue_size=1)
        self.subscriber = rospy.Subscriber("target_joint_states", JointState, self.handle_target_state, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate, 1e-3)), self.publish_state)

    def handle_target_state(self, message):
        if message.position and message.name:
            name_to_index = {name: index for index, name in enumerate(message.name)}
            for joint_index, joint_name in enumerate(self.robot_model.joint_names):
                if joint_name in name_to_index:
                    self.target_positions[joint_index] = message.position[name_to_index[joint_name]]
        elif message.position:
            positions = np.asarray(message.position, dtype=float)
            if positions.shape[0] == self.robot_model.dof:
                self.target_positions = positions
        self.target_positions = self.robot_model.clip_to_limits(self.target_positions)

    def publish_state(self, _event):
        dt = 1.0 / max(self.publish_rate, 1e-3)
        delta = self.target_positions - self.current_positions
        max_step = self.max_joint_speed * dt
        clipped_delta = np.clip(delta, -max_step, max_step)
        self.current_positions = self.robot_model.clip_to_limits(self.current_positions + clipped_delta)

        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = list(self.robot_model.joint_names)
        message.position = self.current_positions.tolist()
        self.publisher.publish(message)


def main():
    rospy.init_node("robot_controller")
    ArmJointStateController()
    rospy.spin()


if __name__ == "__main__":
    main()
