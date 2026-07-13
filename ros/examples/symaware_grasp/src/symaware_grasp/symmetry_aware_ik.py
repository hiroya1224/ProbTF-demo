import math
from dataclasses import dataclass

import numpy as np

from probtf.bingham import bingham_log_normalizer
from probtf.distributions import DistributionStatus, OrientationKind, TransformDistributionStamped
from probtf.probability import PointMomentSummary, forward_component_point_moments
from symaware_grasp.arm_kinematics import ToyArm6DOF
from symaware_grasp.beliefs import distribution_point_moments
from symaware_grasp.distribution_metrics import (
    bingham_bhattacharyya_distance,
    gaussian_bhattacharyya_distance,
)

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - runtime fallback
    minimize = None


def _sign_invariant_quaternion_distance(target_wxyz, current_wxyz):
    dot_value = abs(float(np.dot(target_wxyz, current_wxyz)))
    return 1.0 - min(max(dot_value, 0.0), 1.0) ** 2


def _regularized_inverse(covariance, epsilon=1e-6):
    matrix = np.asarray(covariance, dtype=float).reshape(3, 3)
    matrix = 0.5 * (matrix + matrix.T)
    return np.linalg.inv(matrix + epsilon * np.eye(3, dtype=float))


@dataclass(frozen=True)
class _ComponentCostModel:
    weight: float
    mean: np.ndarray
    covariance: np.ndarray
    inverse_covariance: np.ndarray
    orientation_kind: OrientationKind
    orientation_parameter: object
    orientation_log_normalizer: object
    orientation_mode: np.ndarray


def _component_cost_models(record, integration_steps):
    if not isinstance(record, TransformDistributionStamped):
        raise TypeError("IK targets must be native TransformDistributionStamped records.")
    normalized = record.distribution.normalize_weights()
    if normalized.status is not DistributionStatus.OK:
        raise ValueError("IK cannot consume a {} transform distribution.".format(normalized.status.value))
    origin = PointMomentSummary(np.zeros(3), np.zeros((3, 3)))
    models = []
    for item in normalized.components:
        component = item.component
        moments = forward_component_point_moments(component, origin, integration_steps)
        orientation = component.orientation
        parameter = None
        log_normalizer = None
        if orientation.kind is OrientationKind.FINITE_BINGHAM:
            parameter = orientation.backend_parameter_matrix()
            log_normalizer = bingham_log_normalizer(parameter, integration_steps)
        models.append(
            _ComponentCostModel(
                weight=item.weight,
                mean=moments.mean,
                covariance=moments.covariance,
                inverse_covariance=_regularized_inverse(moments.covariance),
                orientation_kind=orientation.kind,
                orientation_parameter=parameter,
                orientation_log_normalizer=log_normalizer,
                orientation_mode=orientation.mode_wxyz,
            )
        )
    return tuple(models)


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
        self.bingham_integration_steps = int(
            bingham_integration_steps if bingham_integration_steps is not None else 80
        )

    def _normalize_method(self, method, use_bingham_orientation):
        if method is None:
            if use_bingham_orientation is None:
                return self.METHOD_BHATTACHARYYA
            return self.METHOD_POINTWISE if bool(use_bingham_orientation) else self.METHOD_DETERMINISTIC
        aliases = {
            "bhattacharyya": self.METHOD_BHATTACHARYYA,
            "bhat": self.METHOD_BHATTACHARYYA,
            "symmetry_aware_pointwise": self.METHOD_POINTWISE,
            "pointwise": self.METHOD_POINTWISE,
            "old": self.METHOD_POINTWISE,
            "deterministic_mode": self.METHOD_DETERMINISTIC,
            "deterministic": self.METHOD_DETERMINISTIC,
        }
        normalized = str(method).strip().lower()
        if normalized not in aliases:
            raise ValueError("Unknown IK method '{}'.".format(method))
        return aliases[normalized]

    def solve(self, targets, theta_now, method=None, use_bingham_orientation=None):
        method = self._normalize_method(method, use_bingham_orientation)
        if method == self.METHOD_BHATTACHARYYA and self.hand_belief_model is None:
            raise RuntimeError("Bhattacharyya IK requires an EndEffectorBeliefModel.")
        theta_now = self.robot_model.clip_to_limits(theta_now)
        if self.hand_belief_model is not None:
            self.hand_belief_model.clear_cache()
        results = [self.solve_single_target(target, theta_now, method) for target in targets]
        feasible = [result for result in results if result["success"]]
        return (min(feasible, key=lambda result: result["total_cost"]) if feasible else None), results

    def solve_single_target(self, target, theta_now, method):
        target_models = _component_cost_models(target, self.bingham_integration_steps)
        if method == self.METHOD_BHATTACHARYYA and any(
            model.orientation_kind is not OrientationKind.FINITE_BINGHAM
            for model in target_models
        ):
            raise ValueError("Bhattacharyya IK supports finite Bingham target components only.")
        target_position = distribution_point_moments(
            target,
            integration_steps=self.bingham_integration_steps,
        ).mean
        objective_cache = {}

        def objective(theta_vector):
            theta_vector = self.robot_model.clip_to_limits(theta_vector)
            cache_key = tuple(np.round(theta_vector, 8).tolist())
            if cache_key not in objective_cache:
                objective_cache[cache_key] = self.evaluate_cost(
                    theta_vector,
                    theta_now,
                    target_models,
                    method,
                )
            return objective_cache[cache_key]

        seeds = [theta_now.copy(), self.robot_model.heuristic_seed(target_position)]
        for _ in range(self.restarts):
            noise = self.rng.normal(
                scale=np.array([0.35, 0.45, 0.45, 0.6, 0.6, 0.6]),
                size=theta_now.shape,
            )
            seeds.append(self.robot_model.clip_to_limits(theta_now + noise))

        best_theta = None
        best_parts = None
        for seed in seeds:
            theta_candidate, cost_parts = self._optimize(objective, seed)
            if math.isfinite(cost_parts["total_cost"]) and (
                best_parts is None or cost_parts["total_cost"] < best_parts["total_cost"]
            ):
                best_theta = theta_candidate
                best_parts = cost_parts
        if best_theta is None:
            return self._result(target, theta_now, None)
        return self._result(target, best_theta, best_parts)

    @staticmethod
    def _result(target, theta, parts):
        failed = parts is None
        return {
            "grasp_id": target.child_frame_id,
            "theta_solution": theta.copy(),
            "total_cost": float("inf") if failed else parts["total_cost"],
            "position_cost": float("inf") if failed else parts["position_cost"],
            "orientation_cost": float("inf") if failed else parts["orientation_cost"],
            "motion_cost": float("inf") if failed else parts["motion_cost"],
            "joint_limit_cost": float("inf") if failed else parts["joint_limit_cost"],
            "success": not failed,
            "target": target,
        }

    def evaluate_cost(self, theta_vector, theta_now, target_models, method):
        theta_vector = self.robot_model.clip_to_limits(theta_vector)
        if method == self.METHOD_BHATTACHARYYA:
            hand_record = self.hand_belief_model.estimate_record(theta_vector)
            hand_models = _component_cost_models(hand_record, self.bingham_integration_steps)
            if any(
                model.orientation_kind is not OrientationKind.FINITE_BINGHAM
                for model in hand_models
            ):
                raise ValueError("Bhattacharyya IK supports finite Bingham hand components only.")
            position_cost = 0.0
            orientation_cost = 0.0
            for target_model in target_models:
                for hand_model in hand_models:
                    weight = target_model.weight * hand_model.weight
                    position_cost += weight * gaussian_bhattacharyya_distance(
                        target_model.mean,
                        target_model.covariance,
                        hand_model.mean,
                        hand_model.covariance,
                    )
                    orientation_cost += weight * bingham_bhattacharyya_distance(
                        target_model.orientation_parameter,
                        hand_model.orientation_parameter,
                        integration_steps=self.bingham_integration_steps,
                        log_normalizer_a=target_model.orientation_log_normalizer,
                        log_normalizer_b=hand_model.orientation_log_normalizer,
                    )
        else:
            hand_position, hand_quaternion, _ = self.robot_model.forward_kinematics(theta_vector)
            position_cost = 0.0
            orientation_cost = 0.0
            for model in target_models:
                position_error = hand_position - model.mean
                position_cost += model.weight * 0.5 * float(
                    position_error.T @ model.inverse_covariance @ position_error
                )
                if method == self.METHOD_DETERMINISTIC:
                    component_orientation_cost = _sign_invariant_quaternion_distance(
                        model.orientation_mode,
                        hand_quaternion,
                    )
                elif model.orientation_kind is OrientationKind.FINITE_BINGHAM:
                    component_orientation_cost = -float(
                        hand_quaternion.T @ model.orientation_parameter @ hand_quaternion
                    )
                elif model.orientation_kind is OrientationKind.DIRAC:
                    component_orientation_cost = _sign_invariant_quaternion_distance(
                        model.orientation_mode,
                        hand_quaternion,
                    )
                elif model.orientation_kind is OrientationKind.UNIFORM:
                    component_orientation_cost = 0.0
                else:  # pragma: no cover - enum exhaustiveness
                    raise ValueError("Unsupported target orientation kind.")
                orientation_cost += model.weight * component_orientation_cost

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
            theta_minus = theta_value.copy()
            theta_minus[index] -= self.numerical_diff_eps
            theta_minus = self.robot_model.clip_to_limits(theta_minus)
            denominator = max(theta_plus[index] - theta_minus[index], self.numerical_diff_eps)
            gradient[index] = (
                objective(theta_plus)["total_cost"] - objective(theta_minus)["total_cost"]
            ) / denominator
        if not np.all(np.isfinite(gradient)):
            gradient.fill(baseline)
        return gradient
