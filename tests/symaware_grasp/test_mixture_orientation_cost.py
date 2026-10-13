"""Regression tests for normalized mixture Bingham pointwise IK costs."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import hyp1f1

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
import symaware_grasp.symmetry_aware_ik as ik


def _record(parameters, weights=None):
    weights = np.ones(len(parameters)) if weights is None else weights
    components = tuple(
        TransformComponent(
            "component_{}".format(index),
            float(weight),
            BinghamOrientation.from_parameter_matrix(parameter),
            ConditionalGaussianTranslation(
                np.array([0.7, 0.1, 0.4]),
                0.02 * np.eye(3),
                np.zeros((3, 9)),
            ),
        )
        for index, (parameter, weight) in enumerate(zip(parameters, weights))
    )
    return TransformDistributionStamped(
        "base_link", "target", 2.0, "base_target", "test", TransformDistribution(components)
    )


def _d2_parameters(concentration):
    return tuple(
        concentration * (np.outer(mode, mode) - np.eye(4)) for mode in np.eye(4)
    )


class _FixedPoseRobot:
    def __init__(self):
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def clip_to_limits(theta):
        return np.clip(np.asarray(theta, dtype=float), -2.0, 2.0)

    def forward_kinematics(self, theta):
        return np.array([0.3, -0.1, 0.2]), self.quaternion.copy(), np.eye(4)

    @staticmethod
    def joint_limit_cost(theta):
        return 0.02 * float(theta @ theta)


def _orientation_cost(models, quaternion):
    robot = _FixedPoseRobot()
    robot.quaternion = np.asarray(quaternion, dtype=float)
    solver = ik.SymmetryAwareIKSolver(robot_model=robot)
    return solver.evaluate_cost(np.zeros(6), np.zeros(6), models)["orientation_cost"]


def _axisymmetric_log_normalizer(concentration):
    # The squared mode coordinate on S^3 is Beta(1/2, 3/2).
    return math.log(2.0 * math.pi**2) - concentration + math.log(
        float(hyp1f1(0.5, 2.0, concentration))
    )


def test_d2_mixture_has_four_modes_and_rejects_between_mode_orientation():
    concentration = 30.0
    models = ik._component_cost_models(_record(_d2_parameters(concentration)), 80)
    mode_costs = np.array([_orientation_cost(models, mode) for mode in np.eye(4)])
    between = np.array([1.0, 1.0, 0.0, 0.0]) / math.sqrt(2.0)
    between_cost = _orientation_cost(models, between)

    np.testing.assert_allclose(mode_costs, mode_costs[0], atol=1e-12)
    expected_gap = math.log1p(3.0 * math.exp(-concentration)) - math.log(
        2.0 * math.exp(-concentration / 2.0) + 2.0 * math.exp(-concentration)
    )
    assert between_cost - mode_costs[0] == pytest.approx(expected_gap, abs=1e-11)
    assert between_cost - mode_costs[0] > 14.0

    # The former weighted quadratic is constant on this equal D2 mixture.
    averaged_parameter = sum(model.weight * model.orientation_parameter for model in models)
    np.testing.assert_allclose(averaged_parameter, -0.75 * concentration * np.eye(4))


@pytest.mark.parametrize("quaternion", [np.array([0.6, 0.8, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0])])
def test_unequal_weights_and_normalizers_match_independent_sphere_integral(quaternion):
    concentrations = [2.0, 9.0]
    weights = np.array([0.2, 0.8])
    parameters = [_d2_parameters(concentrations[0])[0], _d2_parameters(concentrations[1])[1]]
    models = ik._component_cost_models(_record(parameters, weights), 80)
    quadratics = np.array([quaternion @ parameter @ quaternion for parameter in parameters])
    normalizers = np.array([_axisymmetric_log_normalizer(value) for value in concentrations])
    expected = -math.log(float(np.sum(weights * np.exp(quadratics - normalizers))))
    without_normalizers = -math.log(float(np.sum(weights * np.exp(quadratics))))

    assert _orientation_cost(models, quaternion) == pytest.approx(expected, abs=3e-5)
    assert abs(expected - without_normalizers) > 0.05
    swapped_models = ik._component_cost_models(_record(parameters, weights[::-1]), 80)
    assert abs(_orientation_cost(models, quaternion) - _orientation_cost(swapped_models, quaternion)) > 0.2


def test_orientation_cost_is_invariant_to_quaternion_sign_and_parameter_gauge():
    parameters = [np.diag([0.0, -3.0, -4.0, -5.0]), np.diag([-8.0, 0.0, -1.0, -2.0])]
    models = ik._component_cost_models(_record(parameters, [1.0, 3.0]), 80)
    shifted = ik._component_cost_models(
        _record([parameters[0] + 7.5 * np.eye(4), parameters[1] - 100.0 * np.eye(4)], [1.0, 3.0]),
        80,
    )
    quaternion = np.array([1.0, -2.0, 3.0, -4.0]) / math.sqrt(30.0)
    cost = _orientation_cost(models, quaternion)

    assert _orientation_cost(models, -quaternion) == pytest.approx(cost, abs=1e-13)
    assert _orientation_cost(shifted, quaternion) == pytest.approx(cost, abs=1e-12)


def _legacy_cost(solver, theta, theta_now, models):
    theta = solver.robot_model.clip_to_limits(theta)
    position, quaternion, _ = solver.robot_model.forward_kinematics(theta)
    position_cost = sum(
        model.weight * 0.5 * float((position - model.mean) @ model.inverse_covariance @ (position - model.mean))
        for model in models
    )
    orientation_cost = sum(-model.weight * float(quaternion @ model.orientation_parameter @ quaternion) for model in models)
    motion_cost = 0.5 * float((theta - theta_now) @ (theta - theta_now))
    joint_limit_cost = float(solver.robot_model.joint_limit_cost(theta))
    return {
        "position_cost": position_cost,
        "orientation_cost": orientation_cost,
        "motion_cost": motion_cost,
        "joint_limit_cost": joint_limit_cost,
        "total_cost": solver.w_position * position_cost
        + solver.w_orientation * orientation_cost
        + solver.w_motion * motion_cost
        + solver.w_joint_limit * joint_limit_cost,
    }


def test_single_component_preserves_legacy_gradient_and_optimized_solution():
    models = ik._component_cost_models(_record([np.diag([0.0, -7.0, -5.0, -2.0])]), 80)
    solver = ik.SymmetryAwareIKSolver(
        w_position=0.7, w_orientation=1.3, w_motion=0.2, w_joint_limit=0.4,
        restarts=0, max_iterations=60,
    )
    theta_now = np.array([0.1, -0.1, 0.2, -0.1, 0.1, -0.2])
    theta = np.array([0.17, -0.24, 0.39, 0.13, -0.22, 0.33])
    new_objective = lambda value: solver.evaluate_cost(value, theta_now, models)
    old_objective = lambda value: _legacy_cost(solver, value, theta_now, models)
    expected_offset = -solver.w_orientation * models[0].orientation_log_weight_minus_normalizer

    for value in (theta, theta_now, theta + 0.05):
        new_parts = new_objective(value)
        old_parts = old_objective(value)
        for name in ("position_cost", "motion_cost", "joint_limit_cost"):
            assert new_parts[name] == pytest.approx(old_parts[name], abs=1e-12)
        assert new_parts["total_cost"] - old_parts["total_cost"] == pytest.approx(expected_offset, abs=1e-12)
    np.testing.assert_allclose(
        solver._finite_difference_gradient(new_objective, theta),
        solver._finite_difference_gradient(old_objective, theta),
        rtol=1e-8, atol=1e-8,
    )
    new_theta, _ = solver._optimize(new_objective, theta)
    old_theta, _ = solver._optimize(old_objective, theta)
    np.testing.assert_allclose(new_theta, old_theta, rtol=0.0, atol=2e-4)


def test_high_concentration_cost_and_gradient_remain_finite():
    concentration = 5000.0
    models = ik._component_cost_models(_record(_d2_parameters(concentration)), 80)
    mode_costs = [_orientation_cost(models, mode) for mode in np.eye(4)]
    between = np.ones(4) / 2.0
    between_cost = _orientation_cost(models, between)
    assert np.all(np.isfinite(mode_costs + [between_cost]))
    assert between_cost - mode_costs[0] > 3700.0

    # Scaled radial sphere integral is independent of the production backend.
    radial_integral, _ = quad(
        lambda value: math.exp(-value**2) * value**2 / math.sqrt(1.0 - value**2 / concentration),
        0.0, 20.0, epsabs=1e-12, epsrel=1e-12,
    )
    reference_log_normalizer = math.log(8.0 * math.pi * radial_integral / concentration**1.5)
    expected_mode_cost = reference_log_normalizer + math.log(4.0)
    assert mode_costs[0] == pytest.approx(expected_mode_cost, abs=3e-5)

    solver = ik.SymmetryAwareIKSolver(restarts=0)
    theta = np.array([0.17, -0.24, 0.39, 0.13, -0.22, 0.33])
    gradient = solver._finite_difference_gradient(
        lambda value: solver.evaluate_cost(value, np.zeros(6), models), theta
    )
    assert np.all(np.isfinite(gradient))


def test_zero_weight_components_are_excluded_before_normalizer_evaluation():
    parameter = _d2_parameters(9.0)[0]
    baseline = ik._component_cost_models(_record([parameter]), 80)
    # A zero-weight uniform component must not trigger finite-Bingham rejection.
    with_zero = ik._component_cost_models(_record([parameter, np.zeros((4, 4))], [2.0, 0.0]), 80)
    assert len(with_zero) == 1
    quaternion = np.array([0.6, 0.8, 0.0, 0.0])
    assert _orientation_cost(with_zero, quaternion) == pytest.approx(_orientation_cost(baseline, quaternion), abs=1e-12)


def test_normalizers_are_cached_by_spectrum_and_not_recomputed_by_objective(monkeypatch):
    ik._cached_orientation_log_normalizer.cache_clear()
    calls = []
    original = ik.bingham_log_normalizer

    def tracked(parameter, integration_steps):
        calls.append((parameter.copy(), integration_steps))
        return original(parameter, integration_steps)

    monkeypatch.setattr(ik, "bingham_log_normalizer", tracked)
    try:
        models = ik._component_cost_models(_record(_d2_parameters(40.0)), 80)
        assert len(calls) == 1
        for quaternion in np.eye(4):
            _orientation_cost(models, quaternion)
        ik._component_cost_models(_record(_d2_parameters(40.0)), 80)
        assert len(calls) == 1
        ik._component_cost_models(_record(_d2_parameters(40.0)), 60)
        assert len(calls) == 2
        assert calls[0][1] == 80 and calls[1][1] == 60
    finally:
        ik._cached_orientation_log_normalizer.cache_clear()
