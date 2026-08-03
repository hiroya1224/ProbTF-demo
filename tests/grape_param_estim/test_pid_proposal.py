import unittest
from dataclasses import replace

import numpy as np

from grape_param_estim.controller_config import PidGainConfiguration
from grape_param_estim.pid_proposal import (
    derive_pid_proposal_ensemble,
    member_pid_candidate,
)
from grape_param_estim.system import GrapeGeometry, VehicleParameters


class PidProposalTest(unittest.TestCase):
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

    def test_nominal_member_proposes_current_exact_gains(self):
        proposal = derive_pid_proposal_ensemble(
            (7,),
            (self.nominal,),
            (0.013,),
            ("nominal",),
            self.nominal,
            self.geometry,
            self.current,
        )
        np.testing.assert_allclose(proposal.group_scales, 1.0, atol=2.0e-12)
        np.testing.assert_allclose(
            proposal.exact_gain_values[0], self.current.values, atol=2.0e-12
        )
        self.assertEqual(proposal.constant_delay[0], 0.013)

    def test_mass_compensation_scales_xy_and_z_without_changing_pid_ratio(self):
        heavier = replace(self.nominal, mass=1.2 * self.nominal.mass)
        proposal = derive_pid_proposal_ensemble(
            (4,),
            (heavier,),
            (0.02,),
            ("nominal",),
            self.nominal,
            self.geometry,
            self.current,
        )
        np.testing.assert_allclose(
            proposal.group_scales[0, :2], (1.2, 1.2), rtol=2.0e-3
        )
        for group in range(4):
            np.testing.assert_allclose(
                proposal.exact_gain_values[0, group]
                / self.current.values[group],
                proposal.group_scales[0, group],
            )

    def test_raw_members_modes_and_source_identity_are_not_averaged(self):
        less_effective = replace(
            self.nominal, force_effectiveness=np.full(4, 0.8)
        )
        proposal = derive_pid_proposal_ensemble(
            (11, 29),
            (self.nominal, less_effective),
            (0.0, 0.031),
            ("nominal", "alternate"),
            self.nominal,
            self.geometry,
            self.current,
        )
        np.testing.assert_array_equal(proposal.source_member_id, (11, 29))
        self.assertEqual(proposal.source_mode_id, ("nominal", "alternate"))
        self.assertGreater(proposal.group_scales[1, 0], 1.2)
        candidate = member_pid_candidate(proposal, 29)
        self.assertEqual(candidate.source_member_id, 29)
        self.assertEqual(candidate.source_mode_id, "alternate")
        np.testing.assert_array_equal(
            candidate.configuration.values, proposal.exact_gain_values[1]
        )

    def test_percentiles_are_ranges_not_an_automatic_candidate(self):
        members = tuple(
            replace(self.nominal, mass=self.nominal.mass * scale)
            for scale in (0.9, 1.0, 1.1, 1.2)
        )
        proposal = derive_pid_proposal_ensemble(
            (0, 1, 2, 3),
            members,
            (0.01, 0.02, 0.03, 0.04),
            ("nominal",) * 4,
            self.nominal,
            self.geometry,
            self.current,
        )
        interval_50, interval_95 = proposal.percentile_ranges()
        self.assertEqual(interval_50.shape, (2, 4, 3))
        self.assertEqual(interval_95.shape, (2, 4, 3))
        self.assertTrue(np.all(interval_95[0] <= interval_50[0]))
        self.assertTrue(np.all(interval_95[1] >= interval_50[1]))


if __name__ == "__main__":
    unittest.main()
