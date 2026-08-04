import unittest

import numpy as np

from grape_param_estim.posterior.delayed_acceptance import (
    DelayedAcceptanceSampler,
    GaussianSubspaceKernel,
    PosteriorPoint,
    ProposalMixture,
    QuadraticSurrogate,
    TargetEvaluation,
)
from grape_param_estim.posterior.run import (
    McmcRunSettings,
    initialize_mcmc_chains,
    run_mcmc_chains,
)


class McmcRunTests(unittest.TestCase):
    def setUp(self):
        self.map_point = PosteriorPoint(np.zeros(18), 0.02)
        self.covariance = np.diag(np.linspace(0.02, 0.05, 18))
        self.ridge = np.eye(18)[0]
        self.settings = McmcRunSettings(
            mode_id="recorded-mode",
            chain_count=2,
            warmup_steps=1,
            retained_draws=4,
            thinning=1,
            random_seed=19,
            rhat_threshold=1.2,
            minimum_effective_sample_size=1.0,
        )
        self.initializations = initialize_mcmc_chains(
            self.map_point,
            self.covariance,
            delay_standard_deviation=0.004,
            exact_ridge_direction=self.ridge,
            delay_bounds=(0.0, 0.05),
            chain_count=2,
            random_seed=23,
        )

    @staticmethod
    def _sampler(_chain_id):
        center = PosteriorPoint(np.zeros(18), 0.02)
        information = np.eye(19)
        surrogate = QuadraticSurrogate(center, 0.0, information)
        basis = np.eye(19)
        proposal = ProposalMixture(
            (
                GaussianSubspaceKernel(
                    "local", basis, np.full(19, 0.02)
                ),
            ),
            np.ones(1),
        )

        def target(point, warm_start):
            density = -0.5 * float(
                (point.vector - center.vector)
                @ information
                @ (point.vector - center.vector)
            )
            return TargetEvaluation(
                point=point,
                log_density=density,
                successful=True,
                failure_reason="",
                inner_iterations=1,
                warm_start=point.vector.copy(),
            )

        return DelayedAcceptanceSampler(
            surrogate, proposal, (0.0, 0.05), target
        )

    def test_initializers_include_map_and_dispersed_points_within_bounds(self):
        values = initialize_mcmc_chains(
            self.map_point,
            self.covariance,
            delay_standard_deviation=0.02,
            exact_ridge_direction=self.ridge,
            delay_bounds=(0.0, 0.05),
            chain_count=4,
            random_seed=5,
        )
        self.assertEqual(
            tuple(value.source for value in values),
            (
                "map",
                "laplace_dispersion",
                "exact_ridge_dispersion",
                "laplace_dispersion",
            ),
        )
        np.testing.assert_array_equal(
            values[0].point.vector, self.map_point.vector
        )
        self.assertTrue(
            all(0.0 <= value.point.delay <= 0.05 for value in values)
        )

    def test_runs_multiple_chains_and_can_resume_completed_chain(self):
        checkpoints = []
        progress = []
        first = run_mcmc_chains(
            self._sampler,
            self.settings,
            self.initializations,
            self.ridge,
            progress=lambda chain, chain_count, completed, total, step: (
                progress.append((chain, chain_count, completed, total))
            ),
            checkpoint_completed_chain=checkpoints.append,
        )
        self.assertEqual(len(first.chains), 2)
        self.assertEqual(first.diagnostics.draws_per_chain, 4)
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual(
            len(progress),
            self.settings.chain_count
            * (
                self.settings.warmup_steps
                + self.settings.retained_draws * self.settings.thinning
            ),
        )

        created = []

        def factory(chain_id):
            created.append(chain_id)
            return self._sampler(chain_id)

        resumed = run_mcmc_chains(
            factory,
            self.settings,
            self.initializations,
            self.ridge,
            completed_chains={first.chains[0].chain_id: first.chains[0]},
        )
        self.assertEqual(created, ["chain-001"])
        np.testing.assert_array_equal(
            resumed.chains[0].static_coordinate,
            first.chains[0].static_coordinate,
        )
        np.testing.assert_allclose(
            resumed.chains[1].static_coordinate,
            first.chains[1].static_coordinate,
        )


if __name__ == "__main__":
    unittest.main()
