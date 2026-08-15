from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from _support import synthetic_problem_parts
from grape_param_estim.system import ActuatorParameters
from single_bag_savgol_core import (
    COMMON_SCALE_DIRECTION,
    EstimatorConfig,
    LmSettings,
    ObjectiveEvaluation,
    PHYSICAL_DIMENSION,
    SolverResult,
    SYMMETRIC_BASIS,
    SiParameterChart,
    SingleBagDynamicsProblem,
    adaptive_kkt_lm,
    estimate_single_bag,
    generate_actuator_history,
    solve_kkt_lm_step,
)
from smooth_command import QuinticSmoothZoh


class CoreTests(unittest.TestCase):
    def test_parameter_chart_closed_form_derivatives_and_exact_scale(self):
        _dataset, model, _actuator = synthetic_problem_parts()
        chart = SiParameterChart(model.parameters)
        parameters, jacobian = chart.decode_with_jacobian(np.zeros(14))
        self.assertEqual(jacobian.mass[0], parameters.mass)
        self.assertTrue(
            np.array_equal(
                jacobian.force_effectiveness[:, 10:14],
                np.diag(parameters.force_effectiveness),
            )
        )
        base = chart.reference_second_moment_sqrt
        for local_index, basis in enumerate(SYMMETRIC_BASIS):
            second_moment_derivative = base @ basis @ base
            expected = (
                np.trace(second_moment_derivative) * np.eye(3)
                - second_moment_derivative
            )
            self.assertTrue(
                np.allclose(
                    jacobian.inertia[:, :, 1 + local_index], expected
                )
            )
        rng = np.random.default_rng(7)
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

    def test_newton_euler_exact_gauge_and_closure(self):
        dataset, model, actuator = synthetic_problem_parts()
        problem = SingleBagDynamicsProblem(dataset, model, actuator)
        coordinate = np.linspace(-0.02, 0.02, 14)
        evaluation = problem.evaluate_physical(coordinate, 0.01, 0.015)
        self.assertTrue(
            np.allclose(
                np.einsum(
                    "nij,j->ni",
                    evaluation.acceleration_jacobian,
                    COMMON_SCALE_DIRECTION,
                ),
                0.0,
                atol=2e-12,
            )
        )
        self.assertTrue(
            np.allclose(
                evaluation.raw_residual_wrench,
                evaluation.required_wrench - evaluation.modeled_wrench,
            )
        )

    def test_lag_estimated_scientific_ridge_is_physical_14d(self):
        dataset, model, actuator = synthetic_problem_parts()
        problem = SingleBagDynamicsProblem(dataset, model, actuator)
        result = estimate_single_bag(
            problem,
            EstimatorConfig(
                lag_mode="split_strict_only",
                lag_bounds=(0.0, 0.2),
                strict_max_nfev=20,
                strict_alternations=1,
            ),
        )
        self.assertEqual(result.ridge["dimension"], 14)
        self.assertEqual(result.ridge["right_singular_vectors"].shape, (14, 14))
        self.assertEqual(result.evaluation.whitened_jacobian.shape[-1], 14)

    def test_first_optimizer_failure_is_not_overwritten(self):
        dataset, model, actuator = synthetic_problem_parts()
        problem = SingleBagDynamicsProblem(dataset, model, actuator)

        def staged_solver(_problem, initial, evaluator, **_kwargs):
            objective = evaluator(np.asarray(initial))
            is_smooth = np.asarray(initial).size > 14
            return SolverResult(
                coordinate=np.asarray(initial),
                evaluation=objective,
                success=not is_smooth,
                status="deliberate_failure" if is_smooth else "ftol",
                message=(
                    "first stage failed" if is_smooth else "later stage succeeded"
                ),
                nfev=1,
                elapsed_seconds=0.0,
            )

        with patch("single_bag_savgol_core._solve_stage", side_effect=staged_solver):
            result = estimate_single_bag(
                problem,
                EstimatorConfig(
                    lag_mode="split_estimated",
                    lag_bounds=(0.0, 0.2),
                    smooth_width_schedule=(1.0,),
                    strict_alternations=1,
                ),
            )
        self.assertFalse(result.success)
        self.assertEqual(result.first_failure_stage, "smooth_continuation")
        self.assertEqual(result.status, "deliberate_failure")
        self.assertEqual(result.message, "first stage failed")

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
