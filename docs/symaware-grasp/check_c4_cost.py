"""Save C4 mixture and weighted-quadratic counterexample measurements."""

import json
import math
from pathlib import Path

import numpy as np

import symaware_grasp.symmetry_aware_ik as ik
from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, TransformComponent,
    TransformDistribution, TransformDistributionStamped,
)


def _yaw_quaternion(angle):
    return np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)])


def _c4_modes(reference_yaw):
    return np.array([_yaw_quaternion(reference_yaw + index * math.pi / 2.0) for index in range(4)])


def _c4_parameters(concentration, reference_yaw):
    return [concentration * (np.outer(mode, mode) - np.eye(4)) for mode in _c4_modes(reference_yaw)]


def _record(parameters):
    components = tuple(
        TransformComponent(
            "C4:{}".format(index), 0.25, BinghamOrientation.from_parameter_matrix(parameter),
            ConditionalGaussianTranslation(np.array([0.7, 0.1, 0.4]), 0.02 * np.eye(3), np.zeros((3, 9))),
        )
        for index, parameter in enumerate(parameters)
    )
    return TransformDistributionStamped("base_link", "target", 0.0, "test_target", "check_c4_cost", TransformDistribution(components))


class _QuaternionProbe:
    def __init__(self, quaternion):
        self.quaternion = quaternion

    @staticmethod
    def clip_to_limits(theta):
        return np.asarray(theta, dtype=float)

    def forward_kinematics(self, theta):
        return np.array([0.7, 0.1, 0.4]), self.quaternion, np.eye(4)

    @staticmethod
    def joint_limit_cost(theta):
        return 0.0


def _orientation_cost(models, quaternion):
    solver = ik.SymmetryAwareIKSolver(robot_model=_QuaternionProbe(quaternion))
    return solver.evaluate_cost(np.zeros(6), np.zeros(6), models)["orientation_cost"]


results = {}
for concentration in (30.0, 5000.0):
    reference = math.radians(67.5)
    modes = _c4_modes(reference)
    models = ik._component_cost_models(_record(_c4_parameters(concentration, reference)), 80)
    average = sum(model.weight * model.orientation_parameter for model in models)
    costs = [_orientation_cost(models, quaternion) for quaternion in modes]
    between = [_orientation_cost(models, _yaw_quaternion(reference + math.pi / 4.0 + index * math.pi / 2.0)) for index in range(4)]
    yaw_costs = [
        -float((quaternion := _yaw_quaternion(yaw)) @ average @ quaternion)
        for yaw in np.linspace(-math.pi, math.pi, 33)
    ]
    asymptotic_gap = concentration * math.sin(math.pi / 8.0)**2 - math.log(2.0)
    results[str(concentration)] = {
        "reference_yaw_degrees": 67.5,
        "mode_yaw_degrees": [67.5 + 90.0 * index for index in range(4)],
        "intermediate_yaw_degrees": [112.5 + 90.0 * index for index in range(4)],
        "mode_costs": costs, "intermediate_costs": between,
        "log_normalizer": math.log(models[0].weight) - models[0].orientation_log_weight_minus_normalizer,
        "intermediate_minus_mode_cost": between[0] - costs[0],
        "large_concentration_gap_formula": asymptotic_gap,
        "average_parameter": average.tolist(),
        "weighted_quadratic_upright_yaw_minimum": min(yaw_costs),
        "weighted_quadratic_upright_yaw_maximum": max(yaw_costs),
        "weighted_quadratic_upright_yaw_expected": concentration / 2.0,
        "maximum_average_parameter_error": float(np.max(np.abs(average - concentration * np.diag([-0.5, -1.0, -1.0, -0.5])))),
        "quaternion_gram_matrix": (modes @ modes.T).tolist(),
    }
path = Path(__file__).resolve().parent / "c4-check.json"
path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
print(json.dumps({key: {name: value for name, value in result.items() if name not in ("average_parameter", "quaternion_gram_matrix")} for key, result in results.items()}, indent=2))
