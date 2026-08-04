from dataclasses import replace
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
from grape_param_estim.posterior.mcmc import McmcCancelled


class McmcRunTests(unittest.TestCase):
    def setUp(self):
        self.map_point = PosteriorPoint(np.zeros(18), 0.02)
        self.covariance = np.diag(np.linspace(0.02, 0.05, 18))
        self.joint_covariance = np.zeros((19, 19))
        self.joint_covariance[:-1, :-1] = self.covariance
        self.joint_covariance[-1, -1] = 0.004**2
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
            self.joint_covariance,
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
            np.block(
                [
                    [self.covariance, np.zeros((18, 1))],
                    [np.zeros((1, 18)), np.asarray(((0.02**2,),))],
                ]
            ),
            exact_ridge_direction=self.ridge,
            delay_bounds=(0.0, 0.05),
            chain_count=4,
            random_seed=5,
        )
        self.assertEqual(
            tuple(value.source for value in values),
            (
                "map",
                "joint_laplace_dispersion",
                "exact_ridge_plus_joint_laplace_dispersion",
                "joint_laplace_dispersion",
            ),
        )
        np.testing.assert_array_equal(
            values[0].point.vector, self.map_point.vector
        )
        self.assertTrue(
            all(0.0 <= value.point.delay <= 0.05 for value in values)
        )

    def test_joint_laplace_initializer_uses_parameter_delay_cross_covariance(self):
        joint = self.joint_covariance.copy()
        joint[0, -1] = 0.0002
        joint[-1, 0] = 0.0002
        seed = 31
        values = initialize_mcmc_chains(
            self.map_point,
            joint,
            exact_ridge_direction=self.ridge,
            delay_bounds=(0.0, 0.05),
            chain_count=2,
            random_seed=seed,
        )
        random = np.random.RandomState(seed)
        expected = 0.5 * np.linalg.cholesky(joint) @ random.normal(size=19)
        np.testing.assert_allclose(
            values[1].point.static_coordinate,
            self.map_point.static_coordinate + expected[:-1],
        )
        self.assertAlmostEqual(
            values[1].point.delay,
            self.map_point.delay + expected[-1],
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

    def _partial_first_chain(self, split_transition=3):
        proposal_checkpoints = []

        def checkpoint(chain_id, value):
            proposal_checkpoints.append((chain_id, value))

        with self.assertRaises(McmcCancelled) as context:
            run_mcmc_chains(
                self._sampler,
                self.settings,
                self.initializations,
                self.ridge,
                cancellation_requested=lambda: (
                    len(proposal_checkpoints) >= split_transition
                ),
                checkpoint_chain_proposal=checkpoint,
            )
        self.assertEqual(context.exception.completed_transitions, split_transition)
        self.assertIsNotNone(context.exception.checkpoint)
        self.assertEqual(
            tuple(value[0] for value in proposal_checkpoints),
            ("chain-000",) * split_transition,
        )
        self.assertEqual(
            context.exception.checkpoint.completed_transition,
            split_transition,
        )
        return context.exception.checkpoint

    def test_resumes_incomplete_chain_and_forwards_chain_aware_checkpoints(self):
        checkpoint = self._partial_first_chain()
        uninterrupted = run_mcmc_chains(
            self._sampler,
            self.settings,
            self.initializations,
            self.ridge,
        )
        events = []
        completed = []
        resumed = run_mcmc_chains(
            self._sampler,
            self.settings,
            self.initializations,
            self.ridge,
            chain_checkpoints={"chain-000": checkpoint},
            checkpoint_chain_proposal=lambda chain_id, value: events.append(
                (chain_id, value.completed_transition)
            ),
            checkpoint_completed_chain=completed.append,
        )
        self.assertEqual(events[0], ("chain-000", 4))
        self.assertIn(("chain-001", 1), events)
        self.assertEqual(
            tuple(value.chain_id for value in completed),
            ("chain-000", "chain-001"),
        )
        for expected, actual in zip(uninterrupted.chains, resumed.chains):
            for name in (
                "sample_id",
                "draw_index",
                "static_coordinate",
                "delay",
                "log_density",
                "attempted_kernel",
                "accepted_kernel",
                "accepted",
                "stage_one_accepted",
                "stage_two_attempted",
                "full_target_cache_hit",
                "inner_solve_failed",
                "inner_iterations",
            ):
                np.testing.assert_array_equal(
                    getattr(actual, name), getattr(expected, name), err_msg=name
                )
            self.assertEqual(
                actual.kernel_summaries, expected.kernel_summaries
            )

    def test_rejects_checkpoint_overlap_unknown_id_and_settings_mismatch(self):
        checkpoint = self._partial_first_chain()
        completed = run_mcmc_chains(
            self._sampler,
            self.settings,
            self.initializations,
            self.ridge,
        ).chains[0]
        with self.assertRaises(ValueError):
            run_mcmc_chains(
                self._sampler,
                self.settings,
                self.initializations,
                self.ridge,
                completed_chains={"chain-000": completed},
                chain_checkpoints={"chain-000": checkpoint},
            )
        with self.assertRaises(ValueError):
            run_mcmc_chains(
                self._sampler,
                self.settings,
                self.initializations,
                self.ridge,
                chain_checkpoints={"chain-999": checkpoint},
            )
        with self.assertRaises(ValueError):
            run_mcmc_chains(
                self._sampler,
                self.settings,
                self.initializations,
                self.ridge,
                chain_checkpoints={
                    "chain-000": replace(checkpoint, mode_id="other-mode")
                },
            )
        with self.assertRaises(ValueError):
            run_mcmc_chains(
                self._sampler,
                self.settings,
                self.initializations,
                self.ridge,
                chain_checkpoints={"chain-001": checkpoint},
            )

        def changed_kernel_factory(chain_id):
            sampler = self._sampler(chain_id)
            original = sampler.proposal.kernels[0]
            sampler.proposal = ProposalMixture(
                (
                    GaussianSubspaceKernel(
                        "changed-kernel",
                        original.basis,
                        original.scales,
                    ),
                ),
                np.ones(1),
            )
            return sampler

        with self.assertRaises(ValueError):
            run_mcmc_chains(
                changed_kernel_factory,
                self.settings,
                self.initializations,
                self.ridge,
                chain_checkpoints={"chain-000": checkpoint},
            )

    def test_pre_chain_cancellation_preserves_supplied_checkpoint(self):
        checkpoint = self._partial_first_chain()
        with self.assertRaises(McmcCancelled) as context:
            run_mcmc_chains(
                self._sampler,
                self.settings,
                self.initializations,
                self.ridge,
                chain_checkpoints={"chain-000": checkpoint},
                cancellation_requested=lambda: True,
            )
        self.assertIs(context.exception.checkpoint, checkpoint)
        self.assertEqual(
            context.exception.completed_transitions,
            checkpoint.completed_transition,
        )


if __name__ == "__main__":
    unittest.main()
