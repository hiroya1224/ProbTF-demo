from dataclasses import replace
import unittest

import numpy as np

from grape_param_estim.controller_config import PidGainConfiguration
from grape_param_estim.pid.metrics import ForecastMetrics
from grape_param_estim.pid.particle_search import (
    BODY_WRENCH_MODEL_DISCREPANCY,
    CONTINUOUS_SPECTRAL_DENSITY,
    SAMPLE_MODEL_DISCREPANCY,
    ZERO_MODEL_DISCREPANCY,
    ModelDiscrepancyConfiguration,
    ParticleRefinementSettings,
    evaluate_pid_candidates,
    refine_pid_candidate_particles,
)
from grape_param_estim.pid.proposal import (
    PhysicalPlantPosterior,
    derive_pid_proposals,
    sample_pid_candidate,
    user_pid_candidate,
)
from grape_param_estim.system import GrapeGeometry, VehicleParameters


class PidParticleSearchCoreTests(unittest.TestCase):
    def setUp(self):
        self.nominal = VehicleParameters.nominal()
        self.posterior = PhysicalPlantPosterior.from_aligned_values(
            ("chain-a:00000001", "chain-a:00000002", "chain-b:00000001"),
            tuple(
                replace(self.nominal, mass=self.nominal.mass * scale)
                for scale in (0.9, 1.0, 1.1)
            ),
            (0.0, 0.015, 0.03),
            ("mode-map",) * 3,
        )
        self.current = PidGainConfiguration(np.ones((4, 3)))
        self.proposals = derive_pid_proposals(
            self.posterior,
            self.nominal,
            GrapeGeometry.grape(),
            self.current,
        )

    @staticmethod
    def _metrics(error):
        return ForecastMetrics(
            position_rmse=error,
            orientation_rmse=2.0 * error,
            maximum_position_error=3.0 * error,
            maximum_orientation_error=4.0 * error,
            forecast_completion=1.0,
            numerical_failure_count=0,
            actuator_saturation_duration=0.5 * error,
            actuator_saturation_rate=min(error / 10.0, 1.0),
        )

    def test_candidate_is_cross_evaluated_not_only_on_its_source_sample(self):
        candidate = sample_pid_candidate(
            self.proposals, "chain-a:00000001"
        )
        discrepancy = ModelDiscrepancyConfiguration(
            SAMPLE_MODEL_DISCREPANCY,
            np.asarray((1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
            base_seed=1234,
            residual_quantity=BODY_WRENCH_MODEL_DISCREPANCY,
            interval_model=CONTINUOUS_SPECTRAL_DENSITY,
            replicates=2,
        )
        calls = []

        def evaluator(candidate_value, sample, bag_id, realization):
            disturbance = realization.interval_average_residual((0.1, 0.2))
            calls.append(
                (
                    candidate_value.candidate_id,
                    sample.sample_id,
                    sample.delay,
                    bag_id,
                    realization.replicate_index,
                    realization.seed,
                    disturbance,
                )
            )
            return self._metrics(0.2)

        result = evaluate_pid_candidates(
            (candidate,),
            self.posterior,
            ("bag-failure-a", "bag-failure-b"),
            evaluator,
            self.current,
            discrepancy,
            quantile_level=0.75,
            cvar_level=0.75,
        )
        self.assertEqual(
            tuple(value.candidate_id for value in result.candidates),
            ("current", candidate.candidate_id),
        )
        self.assertEqual(len(result.records), 2 * 3 * 2 * 2)
        derived_samples = {
            value.sample_id
            for value in result.records
            if value.candidate_id == candidate.candidate_id
        }
        self.assertEqual(derived_samples, set(self.posterior.sample_id.tolist()))
        self.assertGreater(len(derived_samples), 1)
        self.assertFalse(result.recommendation_available)

        by_identity = {}
        for call in calls:
            key = (call[1], call[3], call[4])
            by_identity.setdefault(key, []).append((call[5], call[6]))
        self.assertEqual(len(by_identity), 3 * 2 * 2)
        for values in by_identity.values():
            self.assertEqual(len(values), 2)
            self.assertEqual(values[0][0], values[1][0])
            np.testing.assert_array_equal(values[0][1], values[1][1])
        observed_delays = {
            sample_id: delay for _candidate, sample_id, delay, *_rest in calls
        }
        self.assertEqual(observed_delays["chain-b:00000001"], 0.03)

    def test_zero_and_sampled_q_policies_are_distinct_and_repeatable(self):
        q = np.arange(1.0, 7.0)
        zero = ModelDiscrepancyConfiguration(
            ZERO_MODEL_DISCREPANCY,
            q,
            base_seed=91,
            residual_quantity=BODY_WRENCH_MODEL_DISCREPANCY,
            interval_model=CONTINUOUS_SPECTRAL_DENSITY,
        ).realization("chain-a:00000001", "bag-a", 0)
        sampled_configuration = ModelDiscrepancyConfiguration(
            SAMPLE_MODEL_DISCREPANCY,
            q,
            base_seed=91,
            residual_quantity=BODY_WRENCH_MODEL_DISCREPANCY,
            interval_model=CONTINUOUS_SPECTRAL_DENSITY,
        )
        sampled = sampled_configuration.realization(
            "chain-a:00000001", "bag-a", 0
        )
        np.testing.assert_array_equal(
            zero.interval_average_residual((0.1, 0.2)), np.zeros((2, 6))
        )
        first = sampled.interval_average_residual((0.1, 0.2))
        second = sampled.interval_average_residual((0.1, 0.2))
        np.testing.assert_array_equal(first, second)
        self.assertGreater(np.linalg.norm(first), 0.0)
        other = sampled_configuration.realization(
            "chain-a:00000002", "bag-a", 0
        )
        self.assertNotEqual(sampled.seed, other.seed)
        self.assertGreater(
            np.linalg.norm(
                first - other.interval_average_residual((0.1, 0.2))
            ),
            0.0,
        )

    def test_refinement_uses_coarse_subset_then_all_samples_for_finalists(self):
        user = user_pid_candidate(
            "user-high", PidGainConfiguration(np.full((4, 3), 1.8))
        )
        discrepancy = ModelDiscrepancyConfiguration(
            ZERO_MODEL_DISCREPANCY,
            np.ones(6),
            base_seed=17,
            residual_quantity=BODY_WRENCH_MODEL_DISCREPANCY,
            interval_model=CONTINUOUS_SPECTRAL_DENSITY,
        )

        def evaluator(candidate, sample, _bag_id, _realization):
            gain = float(candidate.configuration.values[0, 0])
            plant_offset = abs(sample.parameters.mass / self.nominal.mass - 1.0)
            return self._metrics(abs(gain - 1.5) + 0.1 * plant_offset)

        result = refine_pid_candidate_particles(
            (user,),
            self.posterior,
            ("bag-a", "bag-b"),
            evaluator,
            self.current,
            discrepancy,
            ParticleRefinementSettings(
                maximum_generations=1,
                survivor_count=2,
                mutations_per_survivor=1,
                log_gain_standard_deviation=0.1,
                maximum_log_gain_step=0.2,
                random_seed=5,
                stagnation_generations=2,
                maximum_finalists=3,
            ),
            coarse_sample_ids=(
                "chain-a:00000001",
                "chain-a:00000002",
            ),
            quantile_level=0.75,
            cvar_level=0.75,
        )
        self.assertEqual(
            result.initial_evaluation.plant_sample_ids,
            ("chain-a:00000001", "chain-a:00000002"),
        )
        self.assertEqual(len(result.generation_evaluations), 1)
        self.assertTrue(
            any(
                candidate.source == "mutation"
                for candidate in result.generation_evaluations[0].candidates
            )
        )
        self.assertEqual(
            result.final_evaluation.plant_sample_ids,
            tuple(self.posterior.sample_id.tolist()),
        )
        self.assertEqual(
            result.final_evaluation.plant_sample_subset_method,
            "final_all_equal_weight_mcmc_samples",
        )
        expected_per_candidate = 3 * 2 * discrepancy.replicates
        self.assertEqual(
            len(result.final_evaluation.records),
            len(result.final_evaluation.candidates) * expected_per_candidate,
        )


if __name__ == "__main__":
    unittest.main()
