from __future__ import annotations

import unittest

import numpy as np

from _support import synthetic_problem_parts
from grape_param_estim.system import ActuatorParameters
from single_bag_savgol_core import (
    COMMON_SCALE_DIRECTION,
    LmSettings,
    ObjectiveEvaluation,
    PHYSICAL_DIMENSION,
    SiParameterChart,
    SingleBagDynamicsProblem,
    adaptive_kkt_lm,
    generate_actuator_history,
    physical_parameter_jacobian,
    physical_parameter_vector,
    solve_kkt_lm_step,
)
from smooth_command import QuinticSmoothZoh


class CoreTests(unittest.TestCase):
    def test_parameter_chart_physicality_jacobian_and_exact_scale(self):
        _dataset, model, _actuator = synthetic_problem_parts()
        chart = SiParameterChart(model.parameters)
        rng = np.random.default_rng(7)
        for _ in range(10):
            coordinate = 0.1 * rng.standard_normal(PHYSICAL_DIMENSION)
            parameters, jacobian = chart.decode_with_jacobian(coordinate)
            self.assertGreater(parameters.mass, 0.0)
            self.assertTrue(np.allclose(parameters.inertia, parameters.inertia.T))
            self.assertTrue(np.all(np.linalg.eigvalsh(parameters.inertia) > 0.0))
            direction = rng.standard_normal(PHYSICAL_DIMENSION)
            direction /= np.linalg.norm(direction)
            step = 1e-6
            finite = (
                physical_parameter_vector(
                    chart.decode(coordinate + step * direction)
                )
                - physical_parameter_vector(
                    chart.decode(coordinate - step * direction)
                )
            ) / (2 * step)
            analytic = physical_parameter_jacobian(jacobian) @ direction
            self.assertTrue(np.allclose(analytic, finite, rtol=2e-6, atol=2e-8))
        coordinate = 0.05 * rng.standard_normal(PHYSICAL_DIMENSION)
        shift = 0.37
        first = chart.decode(coordinate)
        second = chart.decode(coordinate + shift * COMMON_SCALE_DIRECTION)
        factor = np.exp(shift)
        self.assertTrue(np.isclose(second.mass, factor * first.mass))
        self.assertTrue(np.allclose(second.inertia, factor * first.inertia))
        self.assertTrue(
            np.allclose(
                second.force_effectiveness, factor * first.force_effectiveness
            )
        )
        self.assertTrue(np.allclose(second.cog_offset, first.cog_offset))

    def test_kkt_step_and_solver_enforce_step_section(self):
        jacobian = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        residual = np.asarray((1.0, -2.0))
        gauge = np.asarray((0.0, 0.0, 1.0))
        step, _predicted, _multiplier, violation = solve_kkt_lm_step(
            jacobian, residual, 1e-3, gauge
        )
        self.assertLess(abs(gauge @ step), 1e-13)
        self.assertLess(violation, 1e-13)

        def evaluator(value):
            return ObjectiveEvaluation(
                jacobian @ value - np.asarray((2.0, 3.0)), jacobian
            )

        result = adaptive_kkt_lm(
            evaluator,
            np.asarray((0.0, 0.0, 4.0)),
            settings=LmSettings(),
            max_nfev=30,
            gauge_direction=gauge,
        )
        self.assertLess(abs(gauge @ result.coordinate), 1e-13)

    def test_newton_euler_analytic_jacobian(self):
        dataset, model, actuator = synthetic_problem_parts()
        problem = SingleBagDynamicsProblem(dataset, model, actuator)
        coordinate = np.linspace(-0.02, 0.02, 14)
        evaluation = problem.evaluate_physical(coordinate, 0.01, 0.015)
        finite = np.empty_like(evaluation.acceleration_jacobian)
        step = 1e-6
        for column in range(14):
            plus, minus = coordinate.copy(), coordinate.copy()
            plus[column] += step
            minus[column] -= step
            finite[:, :, column] = (
                problem.evaluate_physical(
                    plus, 0.01, 0.015
                ).acceleration_residual
                - problem.evaluate_physical(
                    minus, 0.01, 0.015
                ).acceleration_residual
            ) / (2 * step)
        self.assertTrue(
            np.allclose(
                evaluation.acceleration_jacobian,
                finite,
                rtol=3e-5,
                atol=2e-6,
            ),
            msg=str(np.max(np.abs(evaluation.acceleration_jacobian - finite))),
        )
        self.assertTrue(
            np.allclose(
                evaluation.raw_residual_wrench,
                evaluation.required_wrench - evaluation.modeled_wrench,
            )
        )

    def test_actuator_history_consistency_active_set_and_strict_final(self):
        times = np.asarray((0.0, 0.25, 0.5, 0.75))
        command_times = np.asarray((-0.1, 0.3, 0.6))
        rotor = QuinticSmoothZoh(
            command_times,
            np.asarray(((-2.0,) * 4, (100.0,) * 4, (5.0,) * 4)),
        )
        gimbal = QuinticSmoothZoh(
            command_times,
            np.asarray(((2.0,) * 4, (-2.0,) * 4, (0.0,) * 4)),
        )
        parameters = ActuatorParameters(
            minimum_thrust=0.0,
            maximum_thrust=10.0,
            maximum_gimbal_angle=0.5,
            maximum_gimbal_rate=0.1,
        )
        kwargs = dict(
            time_axis=times,
            rotor_history=rotor,
            gimbal_history=gimbal,
            initial_gimbal=np.zeros(4),
            rotor_lag_seconds=0.0,
            gimbal_lag_seconds=0.0,
            actuator_parameters=parameters,
            propagation_mode="stateful",
            command_mode="strict",
        )
        first = generate_actuator_history(**kwargs)
        second = generate_actuator_history(**kwargs)
        self.assertTrue(np.array_equal(first.actual_thrust, second.actual_thrust))
        self.assertTrue(np.array_equal(first.actual_gimbal, second.actual_gimbal))
        self.assertGreater(first.active_set_counts.get("gimbal_rate_lower", 0), 0)
        self.assertGreater(first.active_set_counts.get("thrust_command_upper", 0), 0)
        smooth = generate_actuator_history(
            **{**kwargs, "command_mode": "smooth", "smooth_width_fraction": 1.0}
        )
        self.assertFalse(smooth.strict_final)
        strict = generate_actuator_history(**kwargs)
        self.assertTrue(strict.strict_final)


if __name__ == "__main__":
    unittest.main()
