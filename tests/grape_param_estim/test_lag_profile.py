import unittest

import numpy as np

from grape_param_estim.batch.lag_profile import (
    LagObjectiveResult,
    LagProfileFailure,
    LagProfileSettings,
    optimize_lag_profile,
)
from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.state import BatchState
from grape_param_estim.batch.variables import VariableKey, VariableKind


KNOT_KINDS = (
    VariableKind.POSITION,
    VariableKind.ORIENTATION_TANGENT,
    VariableKind.LINEAR_VELOCITY,
    VariableKind.ANGULAR_VELOCITY,
    VariableKind.CONTROLLER_INTEGRAL,
    VariableKind.ACTUATOR_THRUST,
    VariableKind.GIMBAL_ANGLE,
)


def _state_with_marker(marker):
    keys = [VariableKey(VariableKind.STATIC_PARAMETERS)]
    keys.extend(
        VariableKey(kind, bag_id="bag-a", knot_index=0)
        for kind in KNOT_KINDS
    )
    layout = VariableLayout(tuple(keys))
    values = {}
    for key in layout.variable_keys:
        if key.kind is VariableKind.ORIENTATION_TANGENT:
            values[key] = np.eye(3)
        else:
            values[key] = np.zeros(key.dimension)
    values[VariableKey(VariableKind.STATIC_PARAMETERS)][0] = marker
    return BatchState(layout, values)


class LagProfileTests(unittest.TestCase):
    def test_continuous_optimum_is_not_quantized_to_coarse_grid(self):
        optimum = 0.0137
        calls = []

        def evaluator(lag, warm_start):
            calls.append((lag, warm_start))
            return LagObjectiveResult(
                objective=(lag - optimum) ** 2 + 2.0,
                converged=True,
                state=_state_with_marker(lag),
                inner_iterations=4,
                termination_reason="gradient_tolerance",
            )

        result = optimize_lag_profile(
            evaluator,
            LagProfileSettings(
                minimum_lag=0.0,
                maximum_lag=0.04,
                coarse_grid_points=5,
                refinement_tolerance=1.0e-7,
                maximum_refinement_evaluations=32,
            ),
        )
        self.assertAlmostEqual(result.best_lag, optimum, delta=2.0e-6)
        self.assertNotIn(result.best_lag, np.linspace(0.0, 0.04, 5))
        self.assertEqual(len(calls), len(result.points))
        self.assertTrue(any(point.phase == "refinement" for point in result.points))
        self.assertTrue(
            any(point.warm_start_lag is not None for point in result.points[1:])
        )
        self.assertLess(
            result.final_refinement_bracket[1]
            - result.final_refinement_bracket[0],
            result.initial_refinement_bracket[1]
            - result.initial_refinement_bracket[0],
        )

    def test_failed_inner_points_are_retained_but_not_selected(self):
        def evaluator(lag, _warm_start):
            converged = not (0.009 < lag < 0.012)
            return LagObjectiveResult(
                objective=(lag - 0.02) ** 2 if converged else None,
                converged=converged,
                state=_state_with_marker(lag) if converged else None,
                inner_iterations=3,
                termination_reason=(
                    "gradient_tolerance" if converged else "maximum_iterations"
                ),
            )

        result = optimize_lag_profile(
            evaluator,
            LagProfileSettings(0.0, 0.03, coarse_grid_points=7),
        )
        self.assertAlmostEqual(result.best_lag, 0.02, delta=2.0e-5)
        failed = [point for point in result.points if not point.converged]
        self.assertTrue(failed)
        self.assertTrue(all(point.objective is None for point in failed))

    def test_no_converged_coarse_point_is_an_explicit_failure(self):
        def evaluator(_lag, _warm_start):
            return LagObjectiveResult(
                objective=None,
                converged=False,
                state=None,
                inner_iterations=2,
                termination_reason="numerical_factorization_failure",
            )

        with self.assertRaisesRegex(LagProfileFailure, "no coarse-grid"):
            optimize_lag_profile(
                evaluator,
                LagProfileSettings(0.0, 0.03, coarse_grid_points=5),
            )

    def test_settings_and_result_contracts_are_strict(self):
        with self.assertRaises(ValueError):
            LagProfileSettings(-0.1, 0.1)
        with self.assertRaises(ValueError):
            LagProfileSettings(0.1, 0.1)
        with self.assertRaises(ValueError):
            LagProfileSettings(0.0, 0.1, coarse_grid_points=2)
        with self.assertRaises(ValueError):
            LagObjectiveResult(None, True, None, 0, "failed")


if __name__ == "__main__":
    unittest.main()
