import math

import numpy as np

from probik_demo.arm_kinematics import ToyArm6DOF
from probik_demo.distribution_metrics import (
    bingham_bhattacharyya_distance,
    bingham_log_normalizer_from_A,
    gaussian_bhattacharyya_distance,
)
from probik_demo.ptf_utils import (
    make_bingham_distribution,
    position_covariance_from_msg,
    ptf_mode_quaternion_wxyz,
    regularized_inverse_covariance,
)

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - runtime fallback
    minimize = None


def _sign_invariant_quaternion_distance(target_wxyz, current_wxyz):
    dot_value = abs(float(np.dot(target_wxyz, current_wxyz)))
    dot_value = min(max(dot_value, 0.0), 1.0)
    return 1.0 - dot_value**2


class SymmetryAwareIKSolver:
    METHOD_BHATTACHARYYA = "bhattacharyya"
    METHOD_POINTWISE = "symmetry_aware_pointwise"
    METHOD_DETERMINISTIC = "deterministic_mode"

    def __init__(
        self,
        robot_model=None,
        w_position=1.0,
        w_orientation=1.0,
        w_motion=0.1,
        w_joint_limit=1.0,
        numerical_diff_eps=1e-4,
        max_iterations=120,
        restarts=6,
        random_seed=13,
        hand_belief_model=None,
        bingham_integration_steps=None,
    ):
        self.robot_model = robot_model if robot_model is not None else ToyArm6DOF()
        self.w_position = float(w_position)
        self.w_orientation = float(w_orientation)
        self.w_motion = float(w_motion)
        self.w_joint_limit = float(w_joint_limit)
        self.numerical_diff_eps = float(numerical_diff_eps)
        self.max_iterations = int(max_iterations)
        self.restarts = int(restarts)
        self.rng = np.random.default_rng(random_seed)
        self.hand_belief_model = hand_belief_model
        if bingham_integration_steps is None and hand_belief_model is not None:
            bingham_integration_steps = hand_belief_model.bingham_integration_steps
        self.bingham_integration_steps = int(bingham_integration_steps if bingham_integration_steps is not None else 80)

    def _normalize_method(self, method, use_bingham_orientation):
        if method is None:
            if use_bingham_orientation is None:
                return self.METHOD_BHATTACHARYYA
            return self.METHOD_POINTWISE if bool(use_bingham_orientation) else self.METHOD_DETERMINISTIC

        method = str(method).strip().lower()
        aliases = {
            "bhattacharyya": self.METHOD_BHATTACHARYYA,
            "bhat": self.METHOD_BHATTACHARYYA,
            "symmetry_aware_pointwise": self.METHOD_POINTWISE,
            "pointwise": self.METHOD_POINTWISE,
            "old": self.METHOD_POINTWISE,
            "deterministic_mode": self.METHOD_DETERMINISTIC,
            "deterministic": self.METHOD_DETERMINISTIC,
        }
        if method not in aliases:
            raise ValueError(
                "Unknown IK method '%s'. Expected one of: %s"
                % (
                    method,
                    ", ".join(
                        [
                            self.METHOD_BHATTACHARYYA,
                            self.METHOD_POINTWISE,
                            self.METHOD_DETERMINISTIC,
                        ]
                    ),
                )
            )
        return aliases[method]

    def solve(self, target_messages, theta_now, method=None, use_bingham_orientation=None):
        method = self._normalize_method(method, use_bingham_orientation)
        if method == self.METHOD_BHATTACHARYYA and self.hand_belief_model is None:
            raise RuntimeError("Symmetry-aware IK requires an EndEffectorBeliefModel to evaluate EE uncertainty.")
        theta_now = self.robot_model.clip_to_limits(theta_now)
        if self.hand_belief_model is not None:
            self.hand_belief_model.clear_cache()
        results = []
        for target_message in target_messages:
            result = self.solve_single_target(target_message, theta_now, method)
            results.append(result)

        feasible_results = [result for result in results if result["success"]]
        if not feasible_results:
            return None, results
        best_result = min(feasible_results, key=lambda result: result["total_cost"])
        return best_result, results

    def solve_single_target(self, target_message, theta_now, method):
        covariance = position_covariance_from_msg(target_message)
        inverse_covariance = regularized_inverse_covariance(covariance)
        target_position = np.array(
            [
                target_message.position_mean.x,
                target_message.position_mean.y,
                target_message.position_mean.z,
            ],
            dtype=float,
        )
        distribution = make_bingham_distribution(target_message.orientation_bingham.matrix)
        target_mode = ptf_mode_quaternion_wxyz(target_message)
        target_A = distribution.A.copy()
        target_log_normalizer = bingham_log_normalizer_from_A(
            target_A,
            integration_steps=self.bingham_integration_steps,
        )
        objective_cache = {}

        def objective(theta_vector):
            theta_vector = self.robot_model.clip_to_limits(theta_vector)
            cache_key = tuple(np.round(theta_vector, 8).tolist())
            if cache_key in objective_cache:
                return objective_cache[cache_key]
            cost_parts = self.evaluate_cost(
                theta_vector,
                theta_now,
                target_position,
                covariance,
                inverse_covariance,
                target_A,
                target_log_normalizer,
                target_mode,
                method,
            )
            objective_cache[cache_key] = cost_parts
            return cost_parts

        seeds = [theta_now.copy(), self.robot_model.heuristic_seed(target_position)]
        for _ in range(self.restarts):
            noise = self.rng.normal(scale=np.array([0.35, 0.45, 0.45, 0.6, 0.6, 0.6]), size=theta_now.shape)
            seeds.append(self.robot_model.clip_to_limits(theta_now + noise))

        best_theta = None
        best_parts = None
        for seed in seeds:
            theta_candidate, cost_parts = self._optimize(objective, seed)
            if not math.isfinite(cost_parts["total_cost"]):
                continue
            if best_parts is None or cost_parts["total_cost"] < best_parts["total_cost"]:
                best_theta = theta_candidate
                best_parts = cost_parts

        if best_theta is None:
            return {
                "grasp_id": target_message.child_frame_id,
                "theta_solution": theta_now.copy(),
                "total_cost": float("inf"),
                "position_cost": float("inf"),
                "orientation_cost": float("inf"),
                "motion_cost": float("inf"),
                "joint_limit_cost": float("inf"),
                "success": False,
                "target_message": target_message,
            }

        return {
            "grasp_id": target_message.child_frame_id,
            "theta_solution": best_theta,
            "total_cost": best_parts["total_cost"],
            "position_cost": best_parts["position_cost"],
            "orientation_cost": best_parts["orientation_cost"],
            "motion_cost": best_parts["motion_cost"],
            "joint_limit_cost": best_parts["joint_limit_cost"],
            "success": True,
            "target_message": target_message,
        }

    def evaluate_cost(
        self,
        theta_vector,
        theta_now,
        target_position,
        target_covariance,
        inverse_covariance,
        target_A,
        target_log_normalizer,
        target_mode,
        method,
    ):
        theta_vector = self.robot_model.clip_to_limits(theta_vector)
        if method == self.METHOD_BHATTACHARYYA:
            hand_estimate = self.hand_belief_model.estimate_distribution(theta_vector)
            position_cost = gaussian_bhattacharyya_distance(
                target_position,
                target_covariance,
                hand_estimate["position_mean"],
                hand_estimate["position_covariance"],
            )
            orientation_cost = bingham_bhattacharyya_distance(
                target_A,
                hand_estimate["orientation_bingham"],
                integration_steps=self.bingham_integration_steps,
                log_normalizer_a=target_log_normalizer,
                log_normalizer_b=hand_estimate["orientation_log_normalizer"],
            )
        else:
            hand_position, hand_quaternion, _ = self.robot_model.forward_kinematics(theta_vector)
            if method == self.METHOD_POINTWISE:
                position_error = hand_position - target_position
                position_cost = 0.5 * float(position_error.T @ inverse_covariance @ position_error)
                orientation_cost = -float(hand_quaternion.T @ target_A @ hand_quaternion)
            elif method == self.METHOD_DETERMINISTIC:
                position_error = hand_position - target_position
                position_cost = 0.5 * float(position_error.T @ inverse_covariance @ position_error)
                orientation_cost = _sign_invariant_quaternion_distance(target_mode, hand_quaternion)
            else:
                raise ValueError("Unsupported IK method '%s'." % method)

        delta_theta = theta_vector - theta_now
        motion_cost = 0.5 * float(delta_theta.T @ delta_theta)
        joint_limit_cost = float(self.robot_model.joint_limit_cost(theta_vector))

        total_cost = (
            self.w_position * position_cost
            + self.w_orientation * orientation_cost
            + self.w_motion * motion_cost
            + self.w_joint_limit * joint_limit_cost
        )
        return {
            "total_cost": float(total_cost),
            "position_cost": float(position_cost),
            "orientation_cost": float(orientation_cost),
            "motion_cost": float(motion_cost),
            "joint_limit_cost": float(joint_limit_cost),
        }

    def _optimize(self, objective, theta_seed):
        theta_seed = self.robot_model.clip_to_limits(theta_seed)
        if minimize is not None:
            result = minimize(
                lambda theta: objective(theta)["total_cost"],
                theta_seed,
                method="L-BFGS-B",
                bounds=list(zip(self.robot_model.lower_limits, self.robot_model.upper_limits)),
                options={"maxiter": self.max_iterations},
            )
            theta_value = self.robot_model.clip_to_limits(result.x)
            return theta_value, objective(theta_value)
        return self._gradient_descent(objective, theta_seed)

    def _gradient_descent(self, objective, theta_seed):
        theta_value = self.robot_model.clip_to_limits(theta_seed)
        cost_parts = objective(theta_value)
        for _ in range(self.max_iterations):
            gradient = self._finite_difference_gradient(objective, theta_value)
            gradient_norm = float(np.linalg.norm(gradient))
            if not np.isfinite(gradient_norm) or gradient_norm < 1e-5:
                break
            direction = gradient / max(gradient_norm, 1.0)
            step_size = 0.18
            improved = False
            for _ in range(16):
                candidate = self.robot_model.clip_to_limits(theta_value - step_size * direction)
                candidate_parts = objective(candidate)
                if candidate_parts["total_cost"] < cost_parts["total_cost"]:
                    theta_value = candidate
                    cost_parts = candidate_parts
                    improved = True
                    break
                step_size *= 0.5
            if not improved:
                break
        return theta_value, cost_parts

    def _finite_difference_gradient(self, objective, theta_value):
        gradient = np.zeros_like(theta_value, dtype=float)
        baseline = objective(theta_value)["total_cost"]
        for index in range(theta_value.shape[0]):
            theta_plus = theta_value.copy()
            theta_plus[index] += self.numerical_diff_eps
            theta_plus = self.robot_model.clip_to_limits(theta_plus)
            plus_cost = objective(theta_plus)["total_cost"]

            theta_minus = theta_value.copy()
            theta_minus[index] -= self.numerical_diff_eps
            theta_minus = self.robot_model.clip_to_limits(theta_minus)
            minus_cost = objective(theta_minus)["total_cost"]

            denominator = max(theta_plus[index] - theta_minus[index], self.numerical_diff_eps)
            gradient[index] = (plus_cost - minus_cost) / denominator
        if not np.all(np.isfinite(gradient)):
            gradient.fill(baseline)
        return gradient
