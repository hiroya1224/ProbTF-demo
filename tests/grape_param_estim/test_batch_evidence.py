import unittest

import numpy as np

from grape_param_estim.batch.evidence import (
    MarginalObjectiveBreakdown,
    compute_static_laplace_geometry,
    dynamics_q_log_normalization,
)
from grape_param_estim.batch.laplace_em import (
    DiagonalQDefinition,
    QIntervalModel,
)
from grape_param_estim.parameterization import PARAMETER_DIMENSION


def _definition(interval_model):
    return DiagonalQDefinition(
        residual_quantity="test_residual",
        component_names=tuple("x{}".format(index) for index in range(6)),
        component_units=("u",) * 6,
        interval_model=interval_model,
    )


class BatchEvidenceTests(unittest.TestCase):
    def test_continuous_q_normalization_includes_variable_dt(self):
        q = np.exp(np.arange(6, dtype=float) * 0.2)
        dt = np.asarray((0.01, 0.025, 0.04))
        actual = dynamics_q_log_normalization(
            _definition(QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY),
            q,
            dt,
        )
        expected = 0.5 * sum(
            np.sum(np.log(q / time_step)) for time_step in dt
        )
        self.assertAlmostEqual(actual, expected)

    def test_fixed_interval_q_normalization_does_not_use_dt(self):
        q = np.exp(np.arange(6, dtype=float) * 0.1)
        first = dynamics_q_log_normalization(
            _definition(QIntervalModel.FIXED_INTERVAL_COVARIANCE),
            q,
            np.asarray((0.01, 0.02)),
        )
        second = dynamics_q_log_normalization(
            _definition(QIntervalModel.FIXED_INTERVAL_COVARIANCE),
            q,
            np.asarray((1.0, 3.0)),
        )
        self.assertAlmostEqual(first, second)
        self.assertAlmostEqual(first, np.sum(np.log(q)))

    def test_breakdown_requires_exact_sum(self):
        with self.assertRaisesRegex(ValueError, "sum"):
            MarginalObjectiveBreakdown(1.0, 2.0, 3.0, 7.0)

    def test_static_geometry_separates_prior_and_reports_alignment(self):
        likelihood = np.diag(np.arange(PARAMETER_DIMENSION, dtype=float))
        prior_whitening = np.diag(
            np.linspace(1.0, 2.0, PARAMETER_DIMENSION)
        )
        posterior = likelihood + prior_whitening.T @ prior_whitening

        class _Diagnostics:
            log_determinant = 0.0

        class _Factorization:
            reduced_hessian = posterior
            diagnostics = _Diagnostics()

        # The public function deliberately type-checks the production
        # factorization.  Exercise its numerical contract with a lightweight
        # subclass instance that bypasses sparse construction.
        from grape_param_estim.batch.covariance import (
            ArrowheadLaplaceFactorization,
        )

        fake = object.__new__(ArrowheadLaplaceFactorization)
        fake._reduced_hessian = posterior
        fake._diagnostics = _Diagnostics()
        result = compute_static_laplace_geometry(
            fake,
            prior_whitening,
            np.eye(PARAMETER_DIMENSION)[0],
        )
        np.testing.assert_allclose(
            result.information.likelihood.hessian, likelihood
        )
        np.testing.assert_allclose(
            result.information.posterior.hessian, posterior
        )
        np.testing.assert_allclose(
            result.covariance, np.linalg.inv(posterior)
        )
        self.assertEqual(result.information.likelihood.effective_rank, 17)
        self.assertAlmostEqual(result.ridge_alignment, 1.0)


if __name__ == "__main__":
    unittest.main()
