import unittest

import numpy as np

from grape_param_estim.inference import (
    BoundedLogitTransform,
    BoxUniformPrior,
    TemperedResampleMoveSmc,
    TemperedSmcConfig,
    chain_diagnostics,
    marginalize_trajectory_log_likelihood,
    predictive_interval_coverage,
)


class InferenceTests(unittest.TestCase):
    def test_bounded_transform_round_trip_and_jacobian_are_finite(self):
        transform = BoundedLogitTransform([-2.0, 0.1], [3.0, 4.0])
        values = np.array([[-1.5, 0.2], [0.0, 2.0], [2.5, 3.8]])
        unconstrained = transform.to_unconstrained(values)
        reconstructed = transform.from_unconstrained(unconstrained)
        np.testing.assert_allclose(reconstructed, values, atol=1.0e-12)
        self.assertTrue(
            np.all(np.isfinite(transform.log_abs_det_jacobian(unconstrained)))
        )

    def test_box_prior_never_returns_a_logit_boundary(self):
        class BoundaryGenerator:
            def uniform(self, lower, upper, size):
                return np.broadcast_to(lower, size).copy()

        prior = BoxUniformPrior([-1.0, 0.0], [1.0, 2.0])
        values = prior.sample(3, BoundaryGenerator())
        self.assertTrue(np.all(values > prior.lower))
        self.assertTrue(np.all(values < prior.upper))
        transformed = BoundedLogitTransform(
            prior.lower, prior.upper
        ).to_unconstrained(values)
        self.assertTrue(np.all(np.isfinite(transformed)))

    def test_tempered_smc_recovers_known_gaussian_target_reproducibly(self):
        lower = np.array([-4.0, -4.0])
        upper = np.array([4.0, 4.0])
        truth = np.array([1.2, -0.7])
        sigma = np.array([0.25, 0.35])

        def log_likelihood(values):
            standardized = (values - truth) / sigma
            return -0.5 * np.sum(standardized * standardized, axis=1)

        config = TemperedSmcConfig(
            particle_count=768,
            target_ess_fraction=0.75,
            resample_ess_fraction=0.45,
            mcmc_steps=2,
            proposal_scale=0.8,
            seed=19,
        )
        prior = BoxUniformPrior(lower, upper)
        transform = BoundedLogitTransform(lower, upper)
        first = TemperedResampleMoveSmc(prior, transform, config).run(
            log_likelihood
        )
        second = TemperedResampleMoveSmc(prior, transform, config).run(
            log_likelihood
        )
        np.testing.assert_allclose(first.mean(), truth, atol=0.06)
        np.testing.assert_array_equal(first.particles, second.particles)
        np.testing.assert_array_equal(first.weights, second.weights)
        self.assertGreater(len(first.stages), 1)
        self.assertTrue(any(item.resampled for item in first.stages))
        self.assertAlmostEqual(first.stages[-1].inverse_temperature, 1.0)

    def test_seed_chains_have_rhat_like_mixing_diagnostic(self):
        lower = np.array([-3.0])
        upper = np.array([3.0])
        prior = BoxUniformPrior(lower, upper)
        transform = BoundedLogitTransform(lower, upper)

        def likelihood(values):
            return -0.5 * ((values[:, 0] - 0.4) / 0.3) ** 2

        posteriors = []
        for seed in (1, 2, 3):
            posteriors.append(
                TemperedResampleMoveSmc(
                    prior,
                    transform,
                    TemperedSmcConfig(
                        particle_count=512,
                        mcmc_steps=2,
                        seed=seed,
                    ),
                ).run(likelihood)
            )
        diagnostic = chain_diagnostics(posteriors, draw_count=1000, seed=10)
        self.assertEqual(diagnostic.r_hat.shape, (1,))
        self.assertLess(diagnostic.maximum_r_hat, 1.1)
        self.assertTrue(diagnostic.converged)

    def test_trajectory_likelihood_is_marginalized_not_averaged_in_log_space(self):
        conditional = np.log(
            np.array(
                [
                    [0.9, 0.1],
                    [0.2, 0.8],
                    [0.5, 0.5],
                ]
            )
        )
        weights = np.array([0.75, 0.25])
        marginalized = marginalize_trajectory_log_likelihood(
            conditional, weights
        )
        expected = np.log(
            np.sum(np.exp(conditional) * weights[None, :], axis=1)
        )
        np.testing.assert_allclose(marginalized, expected)
        zero_weight = marginalize_trajectory_log_likelihood(
            conditional, np.array([1.0, 0.0])
        )
        np.testing.assert_allclose(zero_weight, conditional[:, 0])

        impossible = marginalize_trajectory_log_likelihood(
            np.full((2, 2), -np.inf), np.array([0.5, 0.5])
        )
        self.assertTrue(np.all(np.isneginf(impossible)))

    def test_predictive_interval_coverage_is_reported_at_all_levels(self):
        rng = np.random.default_rng(4)
        observations = rng.normal(size=200)
        samples = observations[None, :] + rng.normal(
            0.0, 0.2, size=(1000, observations.size)
        )
        coverage = predictive_interval_coverage(observations, samples)
        self.assertEqual(set(coverage), {0.5, 0.8, 0.95})
        self.assertLessEqual(coverage[0.5], coverage[0.8])
        self.assertLessEqual(coverage[0.8], coverage[0.95])


if __name__ == "__main__":
    unittest.main()
