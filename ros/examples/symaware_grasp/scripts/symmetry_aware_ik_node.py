#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import JointState

from probtf_ros import RosProbTfListener
from probtf_ros.bridge import PROBTF_STATIC_TOPIC, PROBTF_TOPIC
from symaware_grasp.arm_kinematics import ToyArm6DOF
from symaware_grasp.msg import GraspTargetArray, IKResult
from symaware_grasp.runtime import lookup_message_record
from symaware_grasp.symmetry_aware_ik import SymmetryAwareIKSolver


def _get_ik_param(name, default):
    return rospy.get_param(
        "~" + name,
        rospy.get_param("/symaware_grasp/ik/" + name, default),
    )


class SymmetryAwareIKNode:
    def __init__(self):
        self.robot_model = ToyArm6DOF()
        self.grasp_targets_topic = _get_ik_param(
            "grasp_targets_topic",
            "/symaware_grasp/grasp_targets",
        )
        self.joint_states_topic = _get_ik_param("joint_states_topic", "joint_states")
        self.lookup_timeout = float(_get_ik_param("lookup_timeout", 2.0))
        self.fresh_target_timeout = float(_get_ik_param("fresh_target_timeout", 5.0))
        self.listener = RosProbTfListener(
            dynamic_topic=_get_ik_param("probtf_topic", PROBTF_TOPIC),
            static_topic=_get_ik_param("probtf_static_topic", PROBTF_STATIC_TOPIC),
        )
        self.target_publisher = rospy.Publisher(
            _get_ik_param("target_joint_states_topic", "target_joint_states"),
            JointState,
            queue_size=1,
            latch=True,
        )
        self.result_publisher = rospy.Publisher(
            _get_ik_param("ik_result_topic", "/symaware_grasp/symmetry_aware_ik_result"),
            IKResult,
            queue_size=1,
            latch=True,
        )
    def run_once(self):
        listener_ready_time = rospy.Time.now().to_sec()
        target_array = self.wait_for_fresh_target_array(listener_ready_time)
        joint_state = rospy.wait_for_message(self.joint_states_topic, JointState)
        targets = []
        for target_message in target_array.targets:
            try:
                targets.append(
                    lookup_message_record(
                        self.listener,
                        target_message.transform,
                        timeout=self.lookup_timeout,
                    )
                )
            except (RuntimeError, ValueError) as exc:
                rospy.logwarn(
                    "Skipping unresolved grasp target '%s': %s",
                    target_message.grasp_id,
                    exc,
                )
        if not targets:
            rospy.logerr("No grasp targets were available in the ProbTF graph.")
            return

        theta_now = self.joint_state_to_array(joint_state)
        solver = SymmetryAwareIKSolver(
            robot_model=self.robot_model,
            w_position=float(_get_ik_param("w_position", 8.0)),
            w_orientation=float(_get_ik_param("w_orientation", 1.0)),
            w_motion=float(_get_ik_param("w_motion", 0.4)),
            w_joint_limit=float(_get_ik_param("w_joint_limit", 1.0)),
            max_iterations=int(_get_ik_param("max_iterations", 90)),
            restarts=int(_get_ik_param("restarts", 6)),
            random_seed=int(_get_ik_param("seed", 31)),
            bingham_integration_steps=int(_get_ik_param("bingham_integration_steps", 80)),
        )
        best_result, _ = solver.solve(targets, theta_now)
        if best_result is None:
            rospy.logerr("No feasible pointwise symmetry-aware IK solution was found.")
            return

        self.result_publisher.publish(
            self.build_result_message(best_result, SymmetryAwareIKSolver.METHOD_POINTWISE)
        )
        command = JointState()
        command.header.stamp = rospy.Time.now()
        command.name = list(self.robot_model.joint_names)
        command.position = best_result["theta_solution"].tolist()
        self.target_publisher.publish(command)
        rospy.loginfo(
            "IK method=%s grasp=%s cost=%.4f motion=%.4f",
            SymmetryAwareIKSolver.METHOD_POINTWISE,
            best_result["grasp_id"],
            best_result["total_cost"],
            float(np.linalg.norm(best_result["theta_solution"] - theta_now)),
        )
        rospy.sleep(0.25)

    def wait_for_fresh_target_array(self, minimum_stamp):
        """Ignore a latched target whose dynamic graph record predates this listener."""

        deadline = rospy.get_time() + self.fresh_target_timeout
        while not rospy.is_shutdown():
            remaining = deadline - rospy.get_time()
            if remaining <= 0.0:
                raise rospy.ROSException(
                    "Timed out waiting for a fresh grasp target after IK node startup."
                )
            message = rospy.wait_for_message(
                self.grasp_targets_topic,
                GraspTargetArray,
                timeout=remaining,
            )
            if message.header.stamp.to_sec() + 1.0e-9 >= minimum_stamp:
                return message
            rospy.sleep(0.02)
        raise rospy.ROSInterruptException(
            "ROS shutdown while waiting for a fresh grasp target."
        )

    def joint_state_to_array(self, message):
        name_to_index = {name: index for index, name in enumerate(message.name)}
        theta_now = np.zeros(self.robot_model.dof, dtype=float)
        for joint_index, joint_name in enumerate(self.robot_model.joint_names):
            if joint_name in name_to_index:
                theta_now[joint_index] = message.position[name_to_index[joint_name]]
        return self.robot_model.clip_to_limits(theta_now)

    @staticmethod
    def build_result_message(result, solver_name):
        message = IKResult()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = result["target"].parent_frame_id
        message.solver_name = solver_name
        message.grasp_id = result["grasp_id"]
        message.theta_solution = result["theta_solution"].tolist()
        message.total_cost = float(result["total_cost"])
        message.position_cost = float(result["position_cost"])
        message.orientation_cost = float(result["orientation_cost"])
        message.motion_cost = float(result["motion_cost"])
        message.joint_limit_cost = float(result["joint_limit_cost"])
        message.success = bool(result["success"])
        return message


def main():
    rospy.init_node("symmetry_aware_ik_node")
    SymmetryAwareIKNode().run_once()


if __name__ == "__main__":
    main()
