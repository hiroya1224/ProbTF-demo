import unittest

import numpy as np

from grape_param_estim.posterior.diagnostics import (
    diagnose_mcmc_chains,
    effective_sample_size,
    split_rhat,
)
from grape_param_estim.posterior.mcmc import (
    KernelAcceptanceSummary,
    McmcChainResult,
)


def _chain(chain_id, values, mode_id="mode-a"):
    points = np.asarray(values, dtype=float)
    draw_count = points.shape[0]
    summary = KernelAcceptanceSummary(
        attempts=draw_count,
        stage_one_accepted=draw_count,
        stage_two_attempted=draw_count,
        stage_two_accepted=draw_count,
        full_target_cache_hits=0,
        inner_solve_failures=0,
        inner_iterations=2 * draw_count,
    )
    return McmcChainResult(
        chain_id=chain_id,
        mode_id=mode_id,
        sample_id=np.asarray(
            ["{}:{:04d}".format(chain_id, index) for index in range(draw_count)]
        ),
        draw_index=np.arange(1, draw_count + 1, dtype=np.int64),
        static_coordinate=points[:, :18],
        delay=points[:, 18],
        log_density=-0.5 * np.sum(points * points, axis=1),
        attempted_kernel=np.asarray(("local",) * draw_count),
        accepted_kernel=np.asarray(("local",) * draw_count),
        accepted=np.ones(draw_count, dtype=bool),
        stage_one_accepted=np.ones(draw_count, dtype=bool),
        stage_two_attempted=np.ones(draw_count, dtype=bool),
        full_target_cache_hit=np.zeros(draw_count, dtype=bool),
        inner_solve_failed=np.zeros(draw_count, dtype=bool),
        inner_iterations=np.full(draw_count, 2, dtype=np.int64),
        warmup_steps=0,
        thinning=1,
        total_transitions=draw_count,
        kernel_summaries={"local": summary},
    )


class McmcDiagnosticTests(unittest.TestCase):
    def test_split_rhat_detects_a_shifted_chain(self):
        generator = np.random.RandomState(12)
        chains = generator.normal(size=(4, 600, 2))
        baseline = split_rhat(chains)
        self.assertTrue(np.all(baseline < 1.02))
        chains[0, :, 0] += 2.0
        shifted = split_rhat(chains)
        self.assertGreater(shifted[0], 1.1)
        self.assertLess(shifted[1], 1.02)

    def test_ess_distinguishes_iid_and_autocorrelated_draws(self):
        generator = np.random.RandomState(8)
        iid = generator.normal(size=(4, 800, 1))
        correlated = np.empty_like(iid)
        noise = generator.normal(size=correlated.shape)
        correlated[:, 0] = noise[:, 0]
        for index in range(1, correlated.shape[1]):
            correlated[:, index] = (
                0.95 * correlated[:, index - 1] + noise[:, index]
            )
        iid_ess = effective_sample_size(iid)[0]
        correlated_ess = effective_sample_size(correlated)[0]
        self.assertGreater(iid_ess, 2000.0)
        self.assertLess(correlated_ess, 0.2 * iid_ess)

    def test_constant_identical_chains_have_unit_rhat_and_full_ess(self):
        chains = np.ones((3, 20, 2))
        np.testing.assert_array_equal(split_rhat(chains), np.ones(2))
        np.testing.assert_array_equal(
            effective_sample_size(chains), np.full(2, 60.0)
        )

    def test_mode_local_summary_separates_completed_and_converged(self):
        generator = np.random.RandomState(91)
        values = generator.normal(scale=0.2, size=(4, 500, 19))
        chains = tuple(
            _chain("chain-{}".format(index), values[index])
            for index in range(4)
        )
        ridge = np.zeros(18)
        ridge[0] = 2.0
        diagnostics = diagnose_mcmc_chains(
            chains,
            ridge,
            rhat_threshold=1.05,
            minimum_effective_sample_size=200.0,
        )
        self.assertTrue(diagnostics.completed)
        self.assertTrue(diagnostics.converged)
        np.testing.assert_allclose(
            diagnostics.ridge_coordinate_trace,
            values[:, :, 0],
        )
        np.testing.assert_allclose(diagnostics.delay_trace, values[:, :, 18])
        self.assertEqual(
            diagnostics.kernel_summaries["local"].attempts,
            4 * 500,
        )
        self.assertEqual(diagnostics.inner_solve_failure_count, 0)
        self.assertFalse(diagnostics.split_rhat.flags.writeable)

    def test_diagnostics_reject_mixed_modes(self):
        values = np.zeros((8, 19))
        ridge = np.ones(18)
        with self.assertRaisesRegex(ValueError, "mode IDs"):
            diagnose_mcmc_chains(
                (_chain("a", values, "mode-a"), _chain("b", values, "mode-b")),
                ridge,
            )


if __name__ == "__main__":
    unittest.main()
