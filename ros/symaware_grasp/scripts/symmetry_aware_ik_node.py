#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import JointState

from symaware_grasp.arm_kinematics import ToyArm6DOF
from symaware_grasp.ee_belief import EndEffectorBeliefModel
from probik_msgs.msg import IKResult, ProbabilisticTF, ProbabilisticTFArray
from symaware_grasp.symmetry_aware_ik import SymmetryAwareIKSolver


def _get_hand_belief_param(name, default):
    return rospy.get_param("~" + name, rospy.get_param("/symaware_grasp/hand_belief/" + name, default))


def _get_ik_param(name, default):
    return rospy.get_param("~" + name, rospy.get_param("/symaware_grasp/ik/" + name, default))


class SymmetryAwareIKNode:
    def __init__(self):
        self.robot_model = ToyArm6DOF()
        self.target_publisher = rospy.Publisher("target_joint_states", JointState, queue_size=1, latch=True)
        self.result_publisher = rospy.Publisher("symmetry_aware_ik_result", IKResult, queue_size=1, latch=True)
        self.baseline_publisher = rospy.Publisher("deterministic_ik_result", IKResult, queue_size=1, latch=True)
        self.selected_target_publisher = rospy.Publisher(
            "selected_grasp_target_prob_tf",
            ProbabilisticTF,
            queue_size=1,
            latch=True,
        )

        self.grasp_targets_topic = rospy.get_param("~grasp_targets_topic", "grasp_target_ptfs")
        self.joint_states_topic = rospy.get_param("~joint_states_topic", "joint_states")
        self.ik_method = str(_get_ik_param("method", SymmetryAwareIKSolver.METHOD_BHATTACHARYYA)).strip().lower()

        joint_noise_stddev = _get_hand_belief_param("joint_noise_stddev", [0.03, 0.03, 0.03, 0.05, 0.05, 0.05])
        if isinstance(joint_noise_stddev, list):
            joint_noise_stddev = np.asarray(joint_noise_stddev, dtype=float)
        else:
            joint_noise_stddev = np.full(self.robot_model.dof, float(joint_noise_stddev), dtype=float)
        self.hand_belief_model = EndEffectorBeliefModel(
            robot_model=self.robot_model,
            joint_noise_stddev=joint_noise_stddev,
            position_covariance_floor=float(_get_hand_belief_param("position_covariance_floor", 5e-5)),
            sample_count=int(_get_hand_belief_param("sample_count", 36)),
            sample_seed=int(_get_hand_belief_param("seed", 23)),
            bingham_integration_steps=int(_get_hand_belief_param("bingham_integration_steps", 80)),
            bingham_fit_max_iterations=int(_get_hand_belief_param("bingham_fit_max_iterations", 40)),
            orientation_initial_concentrations=_get_hand_belief_param(
                "orientation_concentrations",
                [420.0, 320.0, 220.0],
            ),
        )

    def run_once(self):
        target_array = rospy.wait_for_message(self.grasp_targets_topic, ProbabilisticTFArray)
        joint_state = rospy.wait_for_message(self.joint_states_topic, JointState)
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
            hand_belief_model=self.hand_belief_model,
            bingham_integration_steps=int(_get_ik_param("bingham_integration_steps", 80)),
        )

        best_result, _ = solver.solve(target_array.transforms, theta_now, method=self.ik_method)
        baseline_result, _ = solver.solve(target_array.transforms, theta_now, method=SymmetryAwareIKSolver.METHOD_DETERMINISTIC)

        if best_result is None:
            rospy.logerr("No feasible IK solution was found for method '%s'.", self.ik_method)
            return

        self.result_publisher.publish(self.build_result_message(best_result, self.ik_method))
        if baseline_result is not None:
            self.baseline_publisher.publish(self.build_result_message(baseline_result, "deterministic_mode"))
            rospy.loginfo(
                "IK method=%s grasp=%s cost=%.4f motion=%.4f | baseline grasp=%s cost=%.4f motion=%.4f",
                self.ik_method,
                best_result["grasp_id"],
                best_result["total_cost"],
                float(np.linalg.norm(best_result["theta_solution"] - theta_now)),
                baseline_result["grasp_id"],
                baseline_result["total_cost"],
                float(np.linalg.norm(baseline_result["theta_solution"] - theta_now)),
            )
        else:
            rospy.loginfo(
                "IK method=%s grasp=%s cost=%.4f motion=%.4f",
                self.ik_method,
                best_result["grasp_id"],
                best_result["total_cost"],
                float(np.linalg.norm(best_result["theta_solution"] - theta_now)),
            )

        command = JointState()
        command.header.stamp = rospy.Time.now()
        command.name = list(self.robot_model.joint_names)
        command.position = best_result["theta_solution"].tolist()
        self.target_publisher.publish(command)
        self.selected_target_publisher.publish(best_result["target_message"])
        rospy.sleep(0.25)

    def joint_state_to_array(self, message):
        name_to_index = {name: index for index, name in enumerate(message.name)}
        theta_now = np.zeros(self.robot_model.dof, dtype=float)
        for joint_index, joint_name in enumerate(self.robot_model.joint_names):
            if joint_name in name_to_index:
                theta_now[joint_index] = message.position[name_to_index[joint_name]]
        return self.robot_model.clip_to_limits(theta_now)

    def build_result_message(self, result, solver_name):
        message = IKResult()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = "base_link"
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
    node = SymmetryAwareIKNode()
    node.run_once()


if __name__ == "__main__":
    main()
