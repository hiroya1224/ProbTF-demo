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
    ChainInitialization,
    McmcRunSettings,
    run_mcmc_chains,
)


def _embedded_basis(active_indices, basis):
    selected = np.asarray(basis, dtype=float)
    result = np.zeros((19, selected.shape[1]), dtype=float)
    result[np.asarray(active_indices), :] = selected
    return result


def _point(active_indices, active_value):
    vector = np.zeros(19, dtype=float)
    vector[np.asarray(active_indices)] = active_value
    return PosteriorPoint.from_vector(vector)


class McmcStatisticalRecoveryTests(unittest.TestCase):
    def test_linear_gaussian_mean_covariance_and_proper_prior_ridge(self):
        active = (0, 1, 18)
        likelihood_direction = np.asarray((1.0, -1.0))
        likelihood_sigma = 0.30
        observed_difference = 0.52
        prior_mean = np.asarray((0.10, -0.20))
        prior_covariance = np.asarray(((0.64, 0.12), (0.12, 1.00)))
        likelihood_information = np.outer(
            likelihood_direction, likelihood_direction
        ) / likelihood_sigma**2
        prior_information = np.linalg.inv(prior_covariance)
        static_information = likelihood_information + prior_information
        static_linear = (
            likelihood_direction * observed_difference / likelihood_sigma**2
            + prior_information @ prior_mean
        )
        static_mean = np.linalg.solve(static_information, static_linear)
        delay_mean = 0.45
        delay_sigma = 0.08
        mean = np.concatenate((static_mean, (delay_mean,)))
        information = np.zeros((3, 3), dtype=float)
        information[:2, :2] = static_information
        information[2, 2] = 1.0 / delay_sigma**2
        covariance = np.linalg.inv(information)

        center = _point(active, mean)
        surrogate_information = np.zeros((19, 19), dtype=float)
        surrogate_information[np.ix_(active, active)] = 0.72 * information
        surrogate = QuadraticSurrogate(
            center,
            0.0,
            surrogate_information,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        proposal = ProposalMixture(
            (
                GaussianSubspaceKernel(
                    "active_gaussian",
                    _embedded_basis(active, eigenvectors),
                    1.10 * np.sqrt(eigenvalues),
                ),
            ),
            np.ones(1),
        )

        def target(point, _warm_start):
            value = point.vector[np.asarray(active)]
            difference = value - mean
            return TargetEvaluation(
                point=point,
                log_density=-0.5 * float(
                    difference @ information @ difference
                ),
                successful=True,
                failure_reason="",
                inner_iterations=1,
                warm_start=value.copy(),
            )

        def sampler_factory(_chain_id):
            return DelayedAcceptanceSampler(
                surrogate,
                proposal,
                (0.0, 1.0),
                target,
            )

        random = np.random.RandomState(913)
        cholesky = np.linalg.cholesky(covariance)
        initializations = tuple(
            ChainInitialization(
                "chain-{:03d}".format(index),
                "analytic_gaussian_dispersion",
                _point(active, mean + cholesky @ random.normal(size=3)),
            )
            for index in range(4)
        )
        ridge = np.zeros(18)
        ridge[:2] = 1.0 / np.sqrt(2.0)
        result = run_mcmc_chains(
            sampler_factory,
            McmcRunSettings(
                mode_id="linear-gaussian",
                chain_count=4,
                warmup_steps=1000,
                retained_draws=4000,
                thinning=1,
                random_seed=613,
                rhat_threshold=1.08,
                minimum_effective_sample_size=80.0,
            ),
            initializations,
            ridge,
        )
        draws = np.concatenate(
            tuple(
                np.column_stack(
                    (
                        chain.static_coordinate[:, :2],
                        chain.delay,
                    )
                )
                for chain in result.chains
            ),
            axis=0,
        )
        recovered_mean = np.mean(draws, axis=0)
        recovered_covariance = np.cov(draws, rowvar=False)
        standard_error = np.sqrt(np.diag(covariance) / draws.shape[0])
        for coordinate in range(3):
            self.assertAlmostEqual(
                recovered_mean[coordinate],
                mean[coordinate],
                delta=7.0 * standard_error[coordinate] + 0.012,
            )
        np.testing.assert_allclose(
            recovered_covariance,
            covariance,
            rtol=0.14,
            atol=0.008,
        )
        self.assertLess(
            np.max(result.diagnostics.split_rhat[np.asarray(active)]),
            1.06,
        )
        self.assertGreater(
            np.min(result.diagnostics.effective_sample_size[np.asarray(active)]),
            80.0,
        )

        static_ridge = np.asarray((1.0, 1.0)) / np.sqrt(2.0)
        self.assertAlmostEqual(
            float(static_ridge @ likelihood_information @ static_ridge),
            0.0,
            delta=2.0e-15,
        )
        self.assertGreater(
            float(static_ridge @ prior_information @ static_ridge),
            0.5,
        )
        analytic_ridge_variance = float(
            static_ridge @ covariance[:2, :2] @ static_ridge
        )
        sampled_ridge_variance = float(
            np.var(draws[:, :2] @ static_ridge, ddof=1)
        )
        self.assertAlmostEqual(
            sampled_ridge_variance,
            analytic_ridge_variance,
            delta=0.12 * analytic_ridge_variance,
        )
        summary = result.diagnostics.kernel_summaries["active_gaussian"]
        self.assertGreater(summary.stage_two_attempted, 5000)
        self.assertGreater(
            summary.stage_two_attempted - summary.stage_two_accepted,
            100,
        )

    def test_nonlinear_curved_ridge_traversal_differs_from_local_laplace(self):
        active = (0, 1)
        x_sigma = 1.0
        conditional_sigma = 0.20
        curvature = 0.40
        center = _point((0, 1, 18), (0.0, 0.0, 0.5))
        surrogate_information = np.zeros((19, 19))
        surrogate_information[0, 0] = 1.0 / x_sigma**2
        surrogate_information[1, 1] = 1.0 / conditional_sigma**2
        surrogate = QuadraticSurrogate(
            center,
            0.0,
            surrogate_information,
        )
        joint_basis = _embedded_basis(active, np.eye(2))
        ridge_basis = _embedded_basis((0,), np.ones((1, 1)))
        proposal = ProposalMixture(
            (
                GaussianSubspaceKernel(
                    "curved_local",
                    joint_basis,
                    np.asarray((0.18, 0.18)),
                ),
                GaussianSubspaceKernel(
                    "tangent_ridge",
                    ridge_basis,
                    np.asarray((0.24,)),
                ),
            ),
            np.asarray((0.82, 0.18)),
        )

        def target(point, _warm_start):
            x_value = point.static_coordinate[0]
            y_value = point.static_coordinate[1]
            transverse = y_value - curvature * x_value**2
            density = -0.5 * (
                (x_value / x_sigma) ** 2
                + (transverse / conditional_sigma) ** 2
            )
            return TargetEvaluation(
                point=point,
                log_density=density,
                successful=True,
                failure_reason="",
                inner_iterations=1,
                warm_start=np.asarray((x_value, y_value)),
            )

        def sampler_factory(_chain_id):
            return DelayedAcceptanceSampler(
                surrogate,
                proposal,
                (0.0, 1.0),
                target,
            )

        initial_x = (-0.9, -0.35, 0.35, 0.9)
        initializations = tuple(
            ChainInitialization(
                "chain-{:03d}".format(index),
                "curved_ridge_dispersion",
                _point(
                    (0, 1, 18),
                    (value, curvature * value**2, 0.5),
                ),
            )
            for index, value in enumerate(initial_x)
        )
        ridge = np.zeros(18)
        ridge[0] = 1.0
        result = run_mcmc_chains(
            sampler_factory,
            McmcRunSettings(
                mode_id="curved-ridge",
                chain_count=4,
                warmup_steps=2500,
                retained_draws=6500,
                thinning=1,
                random_seed=811,
                rhat_threshold=1.12,
                minimum_effective_sample_size=45.0,
            ),
            initializations,
            ridge,
        )
        draws = np.concatenate(
            tuple(chain.static_coordinate[:, :2] for chain in result.chains),
            axis=0,
        )
        x_value = draws[:, 0]
        y_value = draws[:, 1]
        transverse = y_value - curvature * x_value**2
        self.assertAlmostEqual(np.mean(x_value), 0.0, delta=0.22)
        self.assertAlmostEqual(np.var(x_value), x_sigma**2, delta=0.22)
        self.assertAlmostEqual(
            np.mean(y_value),
            curvature * x_sigma**2,
            delta=0.10,
        )
        self.assertAlmostEqual(
            np.std(transverse), conditional_sigma, delta=0.025
        )
        self.assertGreater(np.corrcoef(y_value, x_value**2)[0, 1], 0.82)
        self.assertLess(np.quantile(x_value, 0.03), -1.3)
        self.assertGreater(np.quantile(x_value, 0.97), 1.3)
        self.assertGreater(
            np.mean(y_value),
            1.5 * conditional_sigma,
        )
        self.assertLess(
            np.max(result.diagnostics.split_rhat[:2]),
            1.10,
        )
        self.assertGreater(
            np.min(result.diagnostics.effective_sample_size[:2]),
            45.0,
        )
        self.assertGreater(
            result.diagnostics.kernel_summaries["tangent_ridge"].attempts,
            1500,
        )


if __name__ == "__main__":
    unittest.main()
