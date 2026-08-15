from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from _support import synthetic_problem_parts
from gimbal_savgol import IrregularSavitzkyGolayGimbal
from rotor_lag import StrictZohCellGrid, power_of_two_epsilon
from single_bag_savgol_core import (
    EstimatorConfig,
    SolverResult,
    SingleBagDynamicsProblem,
    estimate_single_bag,
)
from single_bag_savgol_estimator import build_argument_parser
from smooth_command import QuinticSmoothZoh


class GimbalRotorLagTests(unittest.TestCase):
    def test_irregular_gimbal_sg_exact_polynomial_recovery(self):
        rng = np.random.default_rng(11)
        time = np.cumsum(0.015 + 0.01 * rng.random(180))
        coefficients = np.asarray(
            (
                (0.2, -0.1, 0.03, 0.01),
                (-0.3, 0.2, 0.01, -0.02),
                (0.1, 0.05, -0.04, 0.03),
                (0.0, -0.2, 0.02, 0.01),
            )
        )
        angle = np.column_stack(
            [
                sum(coefficients[joint, power] * time**power for power in range(4))
                for joint in range(4)
            ]
        )
        smoother = IrregularSavitzkyGolayGimbal(
            time, angle, window_seconds=0.8, degree=5
        )
        query = np.linspace(
            smoother.valid_start_time, smoother.valid_end_time, 23
        )
        expected = np.column_stack(
            [
                sum(coefficients[joint, power] * query**power for power in range(4))
                for joint in range(4)
            ]
        )
        self.assertTrue(
            np.allclose(smoother.evaluate(query).angle, expected, atol=2e-12)
        )

    def test_default_objective_uses_measured_gimbal_sg_and_is_15d(self):
        dataset, model, actuator = synthetic_problem_parts()
        problem = SingleBagDynamicsProblem(dataset, model, actuator)
        history = problem.actuator_history(0.1, command_mode="strict")
        self.assertTrue(np.array_equal(history.actual_gimbal, dataset.gimbal_sg_angle))
        residual, jacobian, _evaluation = problem.global_residual_jacobian(
            np.concatenate((np.zeros(14), np.asarray((0.1,)))),
            command_mode="smooth",
            epsilon=1.0,
        )
        self.assertEqual(jacobian.shape, (residual.size, 15))

    def test_smooth_command_converges_to_strict_away_from_switch(self):
        history = QuinticSmoothZoh(
            np.asarray((0.0, 1.0, 2.0)),
            np.asarray(((0.0,), (2.0,), (-1.0,))),
        )
        strict = history.exact_zoh(1.3, 0.1)
        errors = [
            np.linalg.norm(history.evaluate(1.3, 0.1, epsilon).value - strict)
            for epsilon in power_of_two_epsilon(9)
        ]
        self.assertLessEqual(errors[-1], errors[0])
        self.assertLess(errors[-1], 1e-14)
        self.assertTrue(
            np.allclose(history.transition_half_widths(1.0), 0.5)
        )

    def test_exact_cell_history_and_residual_invariance_and_boundary_change(self):
        dataset, model, actuator = synthetic_problem_parts()
        problem = SingleBagDynamicsProblem(dataset, model, actuator)
        grid = problem.strict_lag_grid
        cell = grid.cell(min(2, grid.cell_count - 2))
        first_lag = cell.lower + 0.25 * cell.width
        second_lag = cell.lower + 0.75 * cell.width
        self.assertTrue(
            np.array_equal(
                grid.command_indices(first_lag), grid.command_indices(second_lag)
            )
        )
        first = problem.evaluate_physical(np.zeros(14), first_lag)
        second = problem.evaluate_physical(np.zeros(14), second_lag)
        self.assertTrue(
            np.array_equal(
                first.actuator_history.actual_thrust,
                second.actuator_history.actual_thrust,
            )
        )
        self.assertTrue(np.array_equal(first.acceleration_residual, second.acceleration_residual))
        neighbor = grid.neighbor(cell, 1)
        self.assertIsNotNone(neighbor)
        self.assertFalse(
            np.array_equal(
                grid.command_indices(cell.representative),
                grid.command_indices(neighbor.representative),  # type: ignore[union-attr]
            )
        )

    def test_depth_6_9_12_orchestration_reaches_same_strict_cell(self):
        dataset, model, actuator = synthetic_problem_parts()
        problem = SingleBagDynamicsProblem(dataset, model, actuator)

        def identity_solver(_problem, initial, evaluator, **_kwargs):
            objective = evaluator(np.asarray(initial))
            return SolverResult(
                coordinate=np.asarray(initial),
                evaluation=objective,
                success=True,
                status="test_identity",
                message="test identity solver",
                nfev=1,
                elapsed_seconds=0.0,
            )

        selected = []
        with patch("single_bag_savgol_core._solve_stage", side_effect=identity_solver):
            for depth in (6, 9, 12):
                result = estimate_single_bag(
                    problem, EstimatorConfig(lag_continuation_depth=depth)
                )
                selected.append(
                    (
                        result.diagnostics["lag"]["strict_lag_cell_lower_seconds"],
                        result.diagnostics["lag"]["strict_lag_cell_upper_seconds"],
                    )
                )
        self.assertEqual(selected[0], selected[1])
        self.assertEqual(selected[1], selected[2])

    def test_manual_lag_bound_and_gimbal_lag_are_absent_from_cli(self):
        destinations = {action.dest for action in build_argument_parser()._actions}
        self.assertNotIn("lag_bounds", destinations)
        self.assertNotIn("initial_gimbal_lag", destinations)
        self.assertNotIn("fixed_gimbal_lag", destinations)
        self.assertNotIn("actuator_propagation", destinations)


if __name__ == "__main__":
    unittest.main()
