import unittest

import numpy as np

from grape_param_estim.controller_config import PidGainConfiguration
from grape_param_estim.pid.metrics import (
    FORECAST_COST_METRICS,
    ForecastMetricRecord,
    ForecastMetrics,
    decide_recommendation,
    empirical_lower_cvar,
    empirical_upper_cvar,
    pareto_nondominated_candidate_ids,
    summarize_forecast_records,
)


class PidMetricCoreTests(unittest.TestCase):
    def setUp(self):
        self.current = PidGainConfiguration(np.ones((4, 3)))
        self.changed = PidGainConfiguration(np.full((4, 3), 1.1))

    @staticmethod
    def _records(candidate_id, scale, completion=1.0, failures=0):
        records = []
        for index, multiplier in enumerate((0.8, 1.0, 1.2, 1.4)):
            value = scale * multiplier
            records.append(
                ForecastMetricRecord(
                    candidate_id=candidate_id,
                    sample_id="chain-a:{:08d}".format(index),
                    bag_id="bag-a" if index % 2 == 0 else "bag-b",
                    replicate_index=0,
                    discrepancy_seed=100 + index,
                    metrics=ForecastMetrics(
                        position_rmse=value,
                        orientation_rmse=2.0 * value,
                        maximum_position_error=3.0 * value,
                        maximum_orientation_error=4.0 * value,
                        forecast_completion=completion,
                        numerical_failure_count=failures,
                        actuator_saturation_duration=5.0 * value,
                        actuator_saturation_rate=min(1.0, value / 10.0),
                    ),
                )
            )
        return tuple(records)

    def test_fractional_empirical_cvar_is_equal_weight(self):
        values = (1.0, 2.0, 3.0, 4.0)
        self.assertAlmostEqual(
            empirical_upper_cvar(values, 0.625), (4.0 + 0.5 * 3.0) / 1.5
        )
        self.assertAlmostEqual(
            empirical_lower_cvar(values, 0.625), (1.0 + 0.5 * 2.0) / 1.5
        )

    def test_physical_failure_saturation_and_completion_metrics_stay_separate(self):
        records = self._records("candidate", 0.2, completion=0.75, failures=1)
        summary = summarize_forecast_records(
            records,
            self.current,
            self.changed,
            quantile_level=0.75,
            cvar_level=0.75,
        )
        self.assertEqual(summary.metric_names, FORECAST_COST_METRICS)
        self.assertEqual(summary.mean.shape, (7,))
        self.assertEqual(
            summary.mean[summary.metric_index("numerical_failure_count")],
            1.0,
        )
        self.assertEqual(summary.forecast_completion_mean, 0.75)
        self.assertGreater(
            summary.upper_cvar[
                summary.metric_index("maximum_position_error")
            ],
            summary.upper_cvar[summary.metric_index("position_rmse")],
        )
        self.assertGreater(summary.gain_change_magnitude, 0.0)

    def test_improved_candidate_is_returned_as_pareto_recommendation(self):
        current = summarize_forecast_records(
            self._records("current", 1.0),
            self.current,
            self.current,
            quantile_level=0.75,
            cvar_level=0.75,
        )
        improved = summarize_forecast_records(
            self._records("improved", 0.5),
            self.current,
            self.changed,
            quantile_level=0.75,
            cvar_level=0.75,
        )
        decision = decide_recommendation((current, improved))
        self.assertTrue(decision.recommendation_available)
        self.assertEqual(decision.recommended_candidate_ids, ("improved",))
        self.assertEqual(
            set(pareto_nondominated_candidate_ids((current, improved))),
            {"current", "improved"},
        )
        self.assertEqual(decision.rejection_reason, "")

    def test_tradeoff_or_worse_candidate_produces_no_recommendation(self):
        current = summarize_forecast_records(
            self._records("current", 1.0),
            self.current,
            self.current,
            quantile_level=0.75,
            cvar_level=0.75,
        )
        tradeoff_records = list(self._records("tradeoff", 0.5))
        tradeoff_records[0] = ForecastMetricRecord(
            candidate_id="tradeoff",
            sample_id=tradeoff_records[0].sample_id,
            bag_id=tradeoff_records[0].bag_id,
            replicate_index=0,
            discrepancy_seed=tradeoff_records[0].discrepancy_seed,
            metrics=ForecastMetrics(
                position_rmse=0.4,
                orientation_rmse=0.8,
                maximum_position_error=1.2,
                maximum_orientation_error=1.6,
                forecast_completion=0.5,
                numerical_failure_count=1,
                actuator_saturation_duration=1.0,
                actuator_saturation_rate=0.1,
            ),
        )
        tradeoff = summarize_forecast_records(
            tradeoff_records,
            self.current,
            self.changed,
            quantile_level=0.75,
            cvar_level=0.75,
        )
        decision = decide_recommendation((current, tradeoff))
        self.assertFalse(decision.recommendation_available)
        self.assertEqual(decision.recommended_candidate_ids, tuple())
        self.assertIn("recommendation unavailable", decision.rejection_reason)


if __name__ == "__main__":
    unittest.main()
