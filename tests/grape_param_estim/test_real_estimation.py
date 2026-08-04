from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.batch.lag_profile import (
    LagProfilePoint,
    LagProfileResult,
)
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.batch_artifact import file_sha256
from grape_param_estim.batch_request import validate_batch_estimation_request
from grape_param_estim.real_estimation import (
    estimate_delay_uncertainty,
    prepare_real_estimation_inputs,
    production_state_scaling,
)
from tests.grape_param_estim.test_batch_preparation import (
    _flight_data,
    _request_payload,
)


def _profile(points):
    values = tuple(
        LagProfilePoint(
            lag=float(lag),
            phase="coarse",
            objective=float(objective),
            converged=True,
            inner_iterations=1,
            termination_reason="synthetic",
            warm_start_lag=None,
        )
        for lag, objective in points
    )
    best = min(values, key=lambda value: (value.objective, value.lag))
    return LagProfileResult(
        best_lag=best.lag,
        best_objective=best.objective,
        best_state=None,
        initial_refinement_bracket=(0.0, 0.08),
        final_refinement_bracket=(0.0, 0.08),
        points=values,
    )


class RealEstimationTests(unittest.TestCase):
    def test_delay_uncertainty_uses_positive_local_profile_curvature(self):
        center = 0.035
        standard_deviation = 0.004
        profile = _profile(
            (
                (lag, 9.0 + 0.5 * ((lag - center) / standard_deviation) ** 2)
                for lag in (0.025, 0.03, 0.035, 0.04, 0.045)
            )
        )
        result = estimate_delay_uncertainty((profile,), (0.0, 0.08))
        self.assertEqual(
            result.source, "positive local quadratic profile curvature"
        )
        self.assertAlmostEqual(
            result.standard_deviation_seconds, standard_deviation, places=10
        )

    def test_delay_uncertainty_reports_uniform_prior_fallback(self):
        result = estimate_delay_uncertainty((), (0.0, 0.12))
        self.assertIsNone(result.curvature)
        self.assertEqual(
            result.source,
            "uniform delay prior because local profile curvature is unavailable",
        )
        self.assertAlmostEqual(
            result.standard_deviation_seconds, 0.12 / np.sqrt(12.0)
        )

    def test_real_input_preparation_authenticates_bag_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bag_path = root / "flight-a.bag"
            bag_path.write_bytes(b"synthetic real-estimation flight")
            digest = file_sha256(bag_path)
            flight = _flight_data(bag_path, digest, "flight-a")
            request = validate_batch_estimation_request(
                _request_payload(
                    root, (("flight-a", bag_path, digest),)
                )
            )
            progress = []

            def loader(bag, include_accelerometer, checkpoint):
                self.assertEqual(bag["bag_id"], "flight-a")
                self.assertFalse(include_accelerometer)
                checkpoint()
                return flight

            result = prepare_real_estimation_inputs(
                request,
                flight_loader=loader,
                progress=lambda *value: progress.append(value),
            )
            self.assertEqual(result.request, request)
            self.assertEqual(result.flight_data, (flight,))
            self.assertEqual(result.initializations[0].bag_id, "flight-a")
            self.assertEqual(progress[0][0], "preparing_trajectory")
            self.assertEqual(result.actuator_parameters.delay, 0.0)

    def test_production_scaling_covers_every_batch_variable_kind(self):
        scaling = production_state_scaling()
        self.assertEqual(
            set(scaling.kind_scales), set(VariableKind)
        )
        self.assertTrue(
            all(value > 0.0 for value in scaling.kind_scales.values())
        )


if __name__ == "__main__":
    unittest.main()
