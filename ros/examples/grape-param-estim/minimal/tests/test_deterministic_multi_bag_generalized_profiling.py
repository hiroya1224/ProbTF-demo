from pathlib import Path
import json
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


_MINIMAL = Path(__file__).resolve().parents[1]
_PACKAGE = _MINIMAL.parent / "src"
sys.path.insert(0, str(_MINIMAL))
sys.path.insert(0, str(_PACKAGE))

import deterministic_estimator as baseline  # noqa: E402
import deterministic_multi_bag_generalized_profiling_estimator as estimator  # noqa: E402
import deterministic_multi_bag_multiple_shooting_estimator as multi  # noqa: E402
import deterministic_multiple_shooting_estimator as strict  # noqa: E402
import estimate_recorded_control as entrypoint  # noqa: E402
from grape_param_estim.real_rosbag import load_flight_data  # noqa: E402
from grape_param_estim.system import VehicleParameters  # noqa: E402


class SplineBasisTests(unittest.TestCase):
    def test_open_uniform_basis_is_partition_of_unity_with_derivatives(self):
        time = np.linspace(2.0, 7.0, 101)
        basis = estimator.open_uniform_spline_basis(time, 12, 3)
        np.testing.assert_allclose(np.sum(basis.value, axis=1), 1.0, atol=2.0e-15)
        np.testing.assert_allclose(np.sum(basis.first, axis=1), 0.0, atol=3.0e-14)
        np.testing.assert_allclose(np.sum(basis.second, axis=1), 0.0, atol=3.0e-13)
        self.assertEqual(basis.value.shape, (101, 12))

    def test_invalid_coefficient_count_is_rejected(self):
        with self.assertRaises(ValueError):
            estimator.open_uniform_spline_basis(np.linspace(0.0, 1.0, 5), 3, 3)


class ParameterSeedTests(unittest.TestCase):
    def test_missing_result_falls_back_to_nominal_and_applies_mass_override(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            seed = estimator.load_parameter_seed(missing, 0.04, 3.1)
        self.assertEqual(seed.source_kind, "nominal_fallback")
        self.assertIsNone(seed.source_path)
        self.assertAlmostEqual(seed.delay_seconds, 0.04)
        self.assertAlmostEqual(seed.source_mass_kg, VehicleParameters.nominal().mass)
        self.assertAlmostEqual(seed.corrected_mass_kg, 3.1)
        expected = math.log(3.1 / VehicleParameters.nominal().mass)
        self.assertAlmostEqual(seed.physical_coordinate[0], expected)

    def test_multi_bag_selection_is_loaded(self):
        coordinate = np.linspace(-0.1, 0.1, strict.PHYSICAL_DIMENSION)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": multi.SCHEMA,
                        "selection": {
                            "physical_coordinate": coordinate.tolist(),
                            "delay_seconds": 0.12,
                        },
                    }
                ),
                encoding="utf-8",
            )
            seed = estimator.load_parameter_seed(path, 0.01, None)
        self.assertEqual(seed.source_kind, "estimator_result")
        self.assertAlmostEqual(seed.delay_seconds, 0.12)
        np.testing.assert_allclose(seed.physical_coordinate, coordinate)

    def test_existing_malformed_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                estimator.load_parameter_seed(path, 0.01, None)


class EntryPointTests(unittest.TestCase):
    def test_generalized_profiling_method_routes_to_new_estimator(self):
        with patch.object(estimator, "main", return_value=0) as selected_main:
            status = entrypoint.main(
                (
                    "--method",
                    "generalized_profiling_multi",
                    "--config",
                    "config.json",
                )
            )
        self.assertEqual(status, 0)
        selected_main.assert_called_once_with(["--config", "config.json"])


@unittest.skipUnless(baseline.DEFAULT_BAG.is_file(), "sample rosbag unavailable")
class RecordedFlightProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flight = load_flight_data(
            str(baseline.DEFAULT_BAG),
            start_local=19.0,
            end_local=19.2,
            include_fc_specific_force=True,
            compute_sha256=False,
        )

    def test_zero_correction_evaluates_finite_continuous_dynamics(self):
        arguments = estimator.create_argument_parser().parse_args(
            (
                "--config",
                "unused.json",
                "--spline-knot-count",
                "4",
            )
        )
        seed = estimator.load_parameter_seed(None, 0.01, 3.0)
        specification = multi.BagSpecification(
            "short",
            baseline.DEFAULT_BAG,
            19.0,
            19.2,
            1.0,
        )
        problem = estimator._make_bag_problem(
            specification,
            1.0,
            self.flight,
            seed,
            arguments,
        )
        parameters = estimator._decode_parameters(
            seed.physical_coordinate, seed.delay_seconds
        )
        coefficients = np.zeros(problem.coefficient_shape)
        evaluation = problem.evaluate(coefficients, parameters)
        self.assertEqual(evaluation.pose_residual.shape, (problem.time.size, 6))
        self.assertEqual(evaluation.wrench_residual.shape, (problem.time.size, 6))
        self.assertTrue(np.all(np.isfinite(evaluation.body_acceleration_world)))
        self.assertTrue(np.all(np.isfinite(evaluation.angular_acceleration_body)))
        self.assertTrue(np.all(np.isfinite(problem.residual(coefficients, parameters))))
        self.assertAlmostEqual(parameters.mass, 3.0)


if __name__ == "__main__":
    unittest.main()
