from dataclasses import replace
import os
import unittest
from unittest.mock import patch

import numpy as np

from grape_param_estim.controller_config import PidGainConfiguration
from grape_param_estim.pid.metrics import ForecastMetrics
from grape_param_estim.pid.particle_search import (
    BODY_WRENCH_MODEL_DISCREPANCY,
    CONTINUOUS_SPECTRAL_DENSITY,
    SAMPLE_MODEL_DISCREPANCY,
    ModelDiscrepancyConfiguration,
    evaluate_pid_candidates,
    resolve_forecast_worker_count,
)
from grape_param_estim.pid.proposal import (
    PhysicalPlantPosterior,
    user_pid_candidate,
)
from grape_param_estim.system import VehicleParameters


def _metrics(value):
    return ForecastMetrics(
        position_rmse=value,
        orientation_rmse=2.0 * value,
        maximum_position_error=3.0 * value,
        maximum_orientation_error=4.0 * value,
        forecast_completion=1.0,
        numerical_failure_count=0,
        actuator_saturation_duration=0.5 * value,
        actuator_saturation_rate=min(0.1 * value, 1.0),
    )


def _deterministic_evaluator(candidate, sample, bag_id, realization):
    disturbance = realization.interval_average_residual((0.1, 0.2))
    value = (
        float(candidate.configuration.values[0, 0])
        + 0.01 * sample.parameters.mass
        + 0.001 * sum(ord(character) for character in bag_id)
        + 1.0e-6 * float(np.sum(disturbance))
    )
    return _metrics(value)


def _worker_failure_evaluator(_candidate, sample, _bag_id, _realization):
    if sample.sample_id == "sample-b":
        raise RuntimeError("intentional worker failure")
    return _metrics(1.0)


def _single_blas_thread_evaluator(_candidate, _sample, _bag_id, _realization):
    variables = (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    if any(os.environ.get(name) != "1" for name in variables):
        raise RuntimeError("forecast worker BLAS environment is not single-threaded")
    return _metrics(1.0)


class PidParallelEvaluationTests(unittest.TestCase):
    def setUp(self):
        nominal = VehicleParameters.nominal()
        self.posterior = PhysicalPlantPosterior.from_aligned_values(
            ("sample-a", "sample-b", "sample-c"),
            tuple(
                replace(nominal, mass=nominal.mass * scale)
                for scale in (0.9, 1.0, 1.1)
            ),
            (0.01, 0.02, 0.03),
            ("mode-map",) * 3,
        )
        self.current = PidGainConfiguration(np.ones((4, 3)))
        self.user = user_pid_candidate(
            "user-a", PidGainConfiguration(np.full((4, 3), 1.25))
        )
        self.discrepancy = ModelDiscrepancyConfiguration(
            SAMPLE_MODEL_DISCREPANCY,
            np.arange(1.0, 7.0),
            base_seed=20260804,
            residual_quantity=BODY_WRENCH_MODEL_DISCREPANCY,
            interval_model=CONTINUOUS_SPECTRAL_DENSITY,
            replicates=2,
        )

    def _evaluate(self, workers):
        return evaluate_pid_candidates(
            (self.user,),
            self.posterior,
            ("bag-a", "bag-b"),
            _deterministic_evaluator,
            self.current,
            self.discrepancy,
            worker_count=workers,
        )

    def test_parallel_and_sequential_records_are_bit_identical_and_ordered(self):
        sequential = self._evaluate(1)
        parallel = self._evaluate(2)
        self.assertEqual(parallel.records, sequential.records)
        self.assertEqual(
            tuple(
                (
                    value.candidate_id,
                    value.sample_id,
                    value.bag_id,
                    value.replicate_index,
                )
                for value in parallel.records
            ),
            tuple(
                (candidate, sample, bag, replicate)
                for candidate in ("current", "user-a")
                for sample in ("sample-a", "sample-b", "sample-c")
                for bag in ("bag-a", "bag-b")
                for replicate in range(2)
            ),
        )
        self.assertEqual(parallel.decision, sequential.decision)
        for left, right in zip(parallel.summaries, sequential.summaries):
            np.testing.assert_array_equal(left.mean, right.mean)
            np.testing.assert_array_equal(left.quantile, right.quantile)
            np.testing.assert_array_equal(left.upper_cvar, right.upper_cvar)

    def test_auto_policy_and_worker_blas_environment_are_bounded(self):
        with patch(
            "grape_param_estim.pid.particle_search.available_forecast_cpu_count",
            return_value=15,
        ):
            self.assertEqual(resolve_forecast_worker_count("auto", 100), 7)
            self.assertEqual(resolve_forecast_worker_count("auto", 3), 3)
        result = evaluate_pid_candidates(
            tuple(),
            self.posterior,
            ("bag-a",),
            _single_blas_thread_evaluator,
            self.current,
            self.discrepancy,
            sample_ids=("sample-a", "sample-b"),
            worker_count=2,
        )
        self.assertEqual(len(result.records), 4)

    def test_worker_failure_is_propagated_and_pool_stops(self):
        with self.assertRaisesRegex(RuntimeError, "intentional worker failure"):
            evaluate_pid_candidates(
                tuple(),
                self.posterior,
                ("bag-a",),
                _worker_failure_evaluator,
                self.current,
                self.discrepancy,
                worker_count=2,
            )


if __name__ == "__main__":
    unittest.main()
