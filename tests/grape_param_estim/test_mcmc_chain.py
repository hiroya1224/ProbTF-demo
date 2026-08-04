import unittest

import numpy as np

from grape_param_estim.posterior.delayed_acceptance import (
    DelayedAcceptanceSampler,
    PosteriorPoint,
    ProposalMixture,
    QuadraticSurrogate,
    SymmetricProposalKernel,
    TargetEvaluation,
)
from grape_param_estim.posterior.mcmc import (
    McmcCancelled,
    McmcChainSettings,
    run_mcmc_chain,
)


class IncrementKernel(SymmetricProposalKernel):
    @property
    def name(self):
        return "increment"

    def propose(self, point, random_state):
        vector = point.vector.copy()
        vector[0] += random_state.normal(size=1)[0]
        return PosteriorPoint.from_vector(vector)


class DeterministicRandom:
    def __init__(self, increments, uniforms):
        self.increments = iter(increments)
        self.uniforms = iter(uniforms)

    def choice(self, count, p):
        return 0

    def normal(self, size):
        return np.asarray((next(self.increments),))

    def uniform(self):
        return next(self.uniforms)


def _evaluation(point, log_density=None, iterations=1):
    density = (
        -0.5 * float(point.static_coordinate @ point.static_coordinate)
        if log_density is None
        else float(log_density)
    )
    return TargetEvaluation(
        point=point,
        log_density=density,
        successful=True,
        failure_reason="",
        inner_iterations=iterations,
        warm_start=point.static_coordinate.copy(),
    )


class McmcChainTests(unittest.TestCase):
    def _sampler(self):
        center = PosteriorPoint(np.zeros(18), 0.02)
        surrogate = QuadraticSurrogate(center, 0.0, np.eye(19))
        proposal = ProposalMixture((IncrementKernel(),), np.ones(1))

        def target(point, warm_start):
            return _evaluation(point, iterations=3)

        return (
            DelayedAcceptanceSampler(
                surrogate,
                proposal,
                (0.0, 0.05),
                target,
            ),
            _evaluation(center, 0.0),
        )

    def test_runs_warmup_thinning_and_keeps_equal_weight_draws(self):
        sampler, initial = self._sampler()
        settings = McmcChainSettings(
            chain_id="chain-a",
            mode_id="mode-map",
            warmup_steps=2,
            retained_draws=3,
            thinning=2,
        )
        progress = []
        random = DeterministicRandom(
            increments=(0.1,) * settings.total_transitions,
            uniforms=(0.1, 0.9) * settings.total_transitions,
        )
        result = run_mcmc_chain(
            sampler,
            initial,
            settings,
            random,
            progress=lambda completed, total, step: progress.append(
                (completed, total, step.kernel_name)
            ),
        )
        self.assertEqual(result.static_coordinate.shape, (3, 18))
        np.testing.assert_array_equal(result.draw_index, (1, 2, 3))
        self.assertEqual(
            result.sample_id.tolist(),
            ["chain-a:00000001", "chain-a:00000002", "chain-a:00000003"],
        )
        self.assertEqual(result.mode_id, "mode-map")
        self.assertEqual(result.total_transitions, 8)
        self.assertEqual(len(progress), 8)
        summary = result.kernel_summaries["increment"]
        self.assertEqual(summary.attempts, 8)
        self.assertEqual(summary.stage_one_accepted, 8)
        self.assertEqual(summary.stage_two_attempted, 8)
        self.assertEqual(summary.stage_two_accepted, 8)
        self.assertEqual(summary.inner_iterations, 24)
        self.assertAlmostEqual(summary.overall_acceptance_rate, 1.0)
        self.assertFalse(result.static_coordinate.flags.writeable)
        self.assertFalse(result.sample_id.flags.writeable)

    def test_cancellation_is_checked_before_each_proposal(self):
        sampler, initial = self._sampler()
        settings = McmcChainSettings("chain-a", "mode-map", 0, 4)
        checks = []

        def cancelled():
            checks.append(True)
            return len(checks) == 3

        random = DeterministicRandom(
            increments=(0.1, 0.1),
            uniforms=(0.1, 0.9, 0.1, 0.9),
        )
        with self.assertRaises(McmcCancelled) as context:
            run_mcmc_chain(
                sampler,
                initial,
                settings,
                random,
                cancellation_requested=cancelled,
            )
        self.assertEqual(context.exception.completed_transitions, 2)

    def test_rejected_move_records_empty_accepted_kernel(self):
        sampler, initial = self._sampler()
        settings = McmcChainSettings("chain-a", "mode-map", 0, 1)
        random = DeterministicRandom(
            increments=(2.0,),
            uniforms=(0.9,),
        )
        result = run_mcmc_chain(
            sampler, initial, settings, random
        )
        self.assertEqual(result.attempted_kernel.tolist(), ["increment"])
        self.assertEqual(result.accepted_kernel.tolist(), [""])
        self.assertFalse(result.accepted[0])
        summary = result.kernel_summaries["increment"]
        self.assertEqual(summary.stage_one_accepted, 0)
        self.assertEqual(summary.stage_two_attempted, 0)

    def test_preserves_auditable_laplace_target_component_traces(self):
        center = PosteriorPoint(np.zeros(18), 0.02)
        surrogate = QuadraticSurrogate(
            center, -0.1, np.zeros((19, 19))
        )
        proposal = ProposalMixture((IncrementKernel(),), np.ones(1))

        def target(point, warm_start):
            objective = 0.5 * float(
                point.static_coordinate @ point.static_coordinate
            )
            log_determinant = 0.2
            delay_prior = -0.5 * point.delay**2
            return TargetEvaluation(
                point=point,
                log_density=(
                    delay_prior - objective - 0.5 * log_determinant
                ),
                successful=True,
                failure_reason="",
                inner_iterations=2,
                warm_start=point.static_coordinate.copy(),
                graph_objective=objective,
                local_log_determinant=log_determinant,
                delay_log_prior=delay_prior,
            )

        sampler = DelayedAcceptanceSampler(
            surrogate,
            proposal,
            (0.0, 0.05),
            target,
        )
        settings = McmcChainSettings("chain-components", "mode-map", 0, 2)
        result = run_mcmc_chain(
            sampler,
            target(center, None),
            settings,
            DeterministicRandom(
                increments=(0.1, 0.1),
                uniforms=(0.0, 0.0, 0.0, 0.0),
            ),
        )
        np.testing.assert_allclose(result.graph_objective, (0.005, 0.02))
        np.testing.assert_allclose(result.local_log_determinant, (0.2, 0.2))
        np.testing.assert_allclose(
            result.log_density,
            result.delay_log_prior
            - result.graph_objective
            - 0.5 * result.local_log_determinant,
        )
        self.assertFalse(result.graph_objective.flags.writeable)

    def test_settings_reject_invalid_counts(self):
        invalid = (
            {"warmup_steps": -1, "retained_draws": 2, "thinning": 1},
            {"warmup_steps": 0, "retained_draws": 0, "thinning": 1},
            {"warmup_steps": 0, "retained_draws": 2, "thinning": 0},
        )
        for values in invalid:
            with self.assertRaises(ValueError):
                McmcChainSettings("chain", "mode", **values)


if __name__ == "__main__":
    unittest.main()
