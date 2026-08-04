from dataclasses import replace
import unittest

import numpy as np

from grape_param_estim.controller_config import PidGainConfiguration
from grape_param_estim.pid.particle_search import (
    build_initial_candidate_population,
    select_proposal_medoids,
)
from grape_param_estim.pid.proposal import (
    PhysicalPlantPosterior,
    derive_pid_proposals,
    sample_pid_candidate,
    user_pid_candidate,
)
from grape_param_estim.system import GrapeGeometry, VehicleParameters


class PidSampleProposalTests(unittest.TestCase):
    def setUp(self):
        self.nominal = VehicleParameters.nominal()
        self.geometry = GrapeGeometry.grape()
        self.current = PidGainConfiguration(
            np.asarray(
                (
                    (3.0, 0.1, 1.0),
                    (5.0, 1.0, 2.5),
                    (20.0, 1.0, 8.0),
                    (4.0, 1.0, 2.0),
                )
            )
        )

    def _posterior(self):
        parameters = tuple(
            replace(self.nominal, mass=self.nominal.mass * scale)
            for scale in (0.85, 1.0, 1.2, 1.35)
        )
        return PhysicalPlantPosterior.from_aligned_values(
            (
                "chain-a:00000001",
                "chain-a:00000002",
                "chain-b:00000001",
                "chain-b:00000002",
            ),
            parameters,
            (0.0, 0.01, 0.02, 0.03),
            ("mode-map",) * 4,
        )

    def test_equal_weight_samples_keep_physics_delay_and_id_aligned(self):
        posterior = self._posterior()
        self.assertEqual(posterior.equal_weight, 0.25)
        self.assertEqual(
            posterior.sample("chain-b:00000001").delay, 0.02
        )
        np.testing.assert_array_equal(
            posterior.sample_id,
            (
                "chain-a:00000001",
                "chain-a:00000002",
                "chain-b:00000001",
                "chain-b:00000002",
            ),
        )
        self.assertFalse(posterior.sample_id.flags.writeable)

    def test_every_mcmc_sample_derives_one_exact_correlated_gain(self):
        posterior = self._posterior()
        proposals = derive_pid_proposals(
            posterior,
            self.nominal,
            self.geometry,
            self.current,
        )
        np.testing.assert_array_equal(
            proposals.source_sample_id, posterior.sample_id
        )
        np.testing.assert_array_equal(proposals.source_delay, posterior.delay)
        self.assertEqual(proposals.exact_gain_values.shape, (4, 4, 3))
        for index in range(4):
            np.testing.assert_allclose(
                proposals.exact_gain_values[index],
                self.current.values
                * proposals.group_scales[index, :, None],
            )
        selected = sample_pid_candidate(proposals, "chain-b:00000001")
        self.assertEqual(selected.source_sample_id, "chain-b:00000001")
        self.assertEqual(selected.source_mode_id, "mode-map")
        np.testing.assert_array_equal(
            selected.configuration.values,
            proposals.exact_gain_values[2],
        )

    def test_delay_is_not_absorbed_into_static_gain_transform(self):
        posterior = PhysicalPlantPosterior.from_aligned_values(
            ("chain-a:1", "chain-a:2"),
            (self.nominal, self.nominal),
            (0.0, 0.08),
            ("mode-map", "mode-map"),
        )
        proposals = derive_pid_proposals(
            posterior,
            self.nominal,
            self.geometry,
            self.current,
        )
        np.testing.assert_array_equal(
            proposals.exact_gain_values[0], proposals.exact_gain_values[1]
        )
        np.testing.assert_array_equal(proposals.source_delay, (0.0, 0.08))

    def test_k_medoids_returns_raw_samples_and_baseline_is_always_first(self):
        proposals = derive_pid_proposals(
            self._posterior(),
            self.nominal,
            self.geometry,
            self.current,
        )
        selected_ids = select_proposal_medoids(proposals, 2)
        self.assertEqual(len(selected_ids), 2)
        self.assertTrue(
            set(selected_ids).issubset(set(proposals.source_sample_id.tolist()))
        )
        user = user_pid_candidate(
            "user-soft", PidGainConfiguration(self.current.values * 0.9)
        )
        candidates = build_initial_candidate_population(
            proposals,
            maximum_derived_candidates=2,
            user_candidates=(user,),
        )
        self.assertEqual(candidates[0].candidate_id, "current")
        self.assertEqual(sum(value.source == "sample-derived" for value in candidates), 2)
        self.assertEqual(candidates[-1].candidate_id, "user-soft")
        for candidate in candidates[1:-1]:
            index = proposals.sample_index(candidate.source_sample_id)
            np.testing.assert_array_equal(
                candidate.configuration.values,
                proposals.exact_gain_values[index],
            )

    def test_required_source_replaces_a_medoid_without_averaging(self):
        proposals = derive_pid_proposals(
            self._posterior(),
            self.nominal,
            self.geometry,
            self.current,
        )
        required = "chain-b:00000002"
        candidates = build_initial_candidate_population(
            proposals,
            maximum_derived_candidates=1,
            required_source_sample_ids=(required,),
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[1].source_sample_id, required)
        np.testing.assert_array_equal(
            candidates[1].configuration.values,
            proposals.configuration_for_sample(required).values,
        )


if __name__ == "__main__":
    unittest.main()
