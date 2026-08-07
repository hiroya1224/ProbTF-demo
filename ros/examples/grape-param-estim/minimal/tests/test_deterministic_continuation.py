from pathlib import Path
import sys
import unittest

import numpy as np


_MINIMAL = Path(__file__).resolve().parents[1]
_PACKAGE = _MINIMAL.parent / "src"
sys.path.insert(0, str(_MINIMAL))
sys.path.insert(0, str(_PACKAGE))

from legacies import deterministic_continuation_estimator as estimator  # noqa: E402
from grape_param_estim.system import VehicleParameters  # noqa: E402


class ContinuationScheduleTests(unittest.TestCase):
    def test_schedule_is_increasing_and_ends_at_exact_full_duration(self):
        durations = estimator.continuation_durations(
            5.0,
            (2.0, 0.5, 1.0, 5.0, 7.0, 1.0),
        )
        np.testing.assert_array_equal(durations, (0.5, 1.0, 2.0, 5.0))

    def test_coarse_grid_includes_endpoint_and_nominal_delay(self):
        grid = estimator.inclusive_delay_grid(
            0.0,
            0.08,
            0.03,
            required=(0.01,),
        )
        np.testing.assert_allclose(grid, (0.0, 0.01, 0.03, 0.06, 0.08))

    def test_branches_walk_outward_from_anchor(self):
        branches = estimator.branch_order((0.0, 0.01, 0.02, 0.04), 0.01)
        self.assertEqual(branches[0], (0.01, 0.02, 0.04))
        self.assertEqual(branches[1], (0.01, 0.0))


class FourteenCoordinateTests(unittest.TestCase):
    def test_expansion_fixes_both_time_constant_coordinates(self):
        physical = np.linspace(-0.1, 0.1, estimator.PHYSICAL_DIMENSION)
        expanded = estimator._expand_coordinate(physical, 0.0275)
        np.testing.assert_array_equal(expanded[:13], physical)
        np.testing.assert_array_equal(expanded[13:15], (0.0, 0.0))
        self.assertEqual(expanded[15], 0.0275)
        coordinate_14d = np.concatenate((expanded[:13], expanded[15:]))
        self.assertEqual(coordinate_14d.shape, (estimator.TOTAL_DIMENSION,))

    def test_nominal_decode_retains_fixed_actuator_time_constants(self):
        parameterization = estimator.analytic.PhysicalSearchParameterization(
            VehicleParameters.nominal()
        )
        decoded = parameterization.decode(
            estimator._expand_coordinate(np.zeros(13), 0.01)
        )
        self.assertEqual(
            decoded.actuator_parameters.thrust_time_constant,
            estimator.FIXED_THRUST_TIME_CONSTANT,
        )
        self.assertEqual(
            decoded.actuator_parameters.gimbal_time_constant,
            estimator.FIXED_GIMBAL_TIME_CONSTANT,
        )
        self.assertEqual(decoded.delay, 0.01)

    def test_cached_objective_exposes_only_thirteen_analytic_columns(self):
        class FakeEvaluator:
            def optimization_residual_and_jacobian(
                self, problem, smooth, fixed_delay
            ):
                del problem, fixed_delay
                residual = np.asarray((smooth[0], smooth[12]), dtype=float)
                jacobian = np.arange(30, dtype=float).reshape(2, 15)
                return residual, jacobian

        objective = estimator._CachedPhysicalObjective(
            FakeEvaluator(), object(), 0.01
        )
        coordinate = np.linspace(0.0, 0.12, 13)
        residual = objective.residual(coordinate)
        jacobian = objective.jacobian(coordinate)
        np.testing.assert_allclose(residual, (coordinate[0], coordinate[12]))
        np.testing.assert_array_equal(
            jacobian,
            np.arange(30, dtype=float).reshape(2, 15)[:, :13],
        )
        self.assertEqual(objective.linearization_count, 1)

    def test_parser_has_no_sobol_tempering_or_time_constant_search(self):
        arguments = estimator.create_argument_parser().parse_args([])
        self.assertFalse(hasattr(arguments, "sobol_power"))
        self.assertFalse(hasattr(arguments, "replica_count"))
        self.assertFalse(hasattr(arguments, "thrust_time_constant_scale_bounds"))
        self.assertGreater(arguments.full_max_nfev, arguments.max_nfev)
        self.assertIsNone(arguments.pid_gain_min_scale)
        self.assertIsNone(arguments.pid_gain_max_scale)


if __name__ == "__main__":
    unittest.main()
