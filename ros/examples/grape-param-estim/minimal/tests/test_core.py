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
    _standard_gauge_least_squares,
    adaptive_kkt_lm,
    estimate_single_bag,
    generate_gimbal_command_replay,
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

    def test_kkt_solver_bounds_non_evaluating_lm_rejections(self):
        def evaluator(value):
            return ObjectiveEvaluation(
                np.asarray((value[0] - 1.0,)), np.asarray(((1.0, 0.0),))
            )

        invalid_step = (np.asarray((1.0, 0.0)), -1.0, 0.0, 0.0)
        with patch(
            "single_bag_savgol_core.solve_kkt_lm_step",
            return_value=invalid_step,
        ) as mocked_step:
            result = adaptive_kkt_lm(
                evaluator,
                np.zeros(2),
                settings=LmSettings(),
                max_nfev=30,
                gauge_direction=np.asarray((0.0, 1.0)),
            )

        self.assertFalse(result.success)
        self.assertEqual(result.status, "numerical_stagnation")
        self.assertEqual(result.nfev, 1)
        self.assertEqual(mocked_step.call_count, 48)

    def test_kkt_solver_accepts_tiny_step_as_xtol_without_trial(self):
        def evaluator(value):
            return ObjectiveEvaluation(
                np.asarray((value[0] - 1.0,)), np.asarray(((1.0, 0.0),))
            )

        with patch(
            "single_bag_savgol_core.solve_kkt_lm_step",
            return_value=(np.zeros(2), 0.0, 0.0, 0.0),
        ):
            result = adaptive_kkt_lm(
                evaluator,
                np.zeros(2),
                settings=LmSettings(),
                max_nfev=30,
                gauge_direction=np.asarray((0.0, 1.0)),
                allow_pretrial_xtol=True,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.status, "xtol")
        self.assertEqual(result.nfev, 1)

    def test_standard_solver_rejects_numerically_invalid_trials(self):
        invalid_trials = []

        def evaluator(value):
            if abs(value[0]) > 0.25:
                invalid_trials.append(value.copy())
                raise ValueError("synthetic chart overflow")
            return ObjectiveEvaluation(
                np.asarray((value[0] - 1.0, value[0] - 1.0)),
                np.asarray(((1.0, 0.0), (1.0, 0.0))),
            )

        result = _standard_gauge_least_squares(
            evaluator,
            np.zeros(2),
            np.asarray((0.0, 1.0)),
            max_nfev=200,
            settings=LmSettings(),
            lower=np.full(2, -np.inf),
            upper=np.full(2, np.inf),
        )
        self.assertTrue(invalid_trials)
        self.assertTrue(np.isfinite(result.evaluation.residual).all())
        self.assertLessEqual(abs(result.coordinate[0]), 0.25)
        self.assertGreater(result.diagnostics[0]["invalid_trial_count"], 0)
        self.assertEqual(
            result.diagnostics[0]["invalid_trial_exception_types"],
            ["ValueError"],
        )

    def test_newton_euler_exact_gauge_and_closure(self):
        dataset, model, actuator = synthetic_problem_parts()
        problem = SingleBagDynamicsProblem(dataset, model, actuator)
        coordinate = np.linspace(-0.02, 0.02, 14)
        evaluation = problem.evaluate_physical(coordinate, 0.01)
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
                lag_mode="estimated",
                lag_continuation_enabled=False,
                strict_max_nfev=20,
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
                    lag_mode="estimated",
                    lag_continuation_schedule=(1.0,),
                ),
            )
        self.assertFalse(result.success)
        self.assertEqual(result.first_failure_stage, "smooth_continuation")
        self.assertEqual(result.status, "deliberate_failure")
        self.assertEqual(result.message, "first stage failed")

    def test_gimbal_replay_consistency_and_rate_limit_active_set(self):
        times = np.asarray((0.0, 0.25, 0.5, 0.75))
        command_times = np.asarray((-0.1, 0.3, 0.6))
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
            gimbal_history=gimbal,
            initial_gimbal=np.zeros(4),
            actuator_parameters=parameters,
        )
        first = generate_gimbal_command_replay(**kwargs)
        second = generate_gimbal_command_replay(**kwargs)
        self.assertTrue(np.array_equal(first.angle, second.angle))
        self.assertGreater(first.active_set_counts.get("gimbal_rate_lower", 0), 0)


if __name__ == "__main__":
    unittest.main()
