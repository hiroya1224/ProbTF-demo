import unittest

import numpy as np

from grape_param_estim.model_error import GaussMarkovWrenchProcess


class GaussMarkovWrenchProcessTests(unittest.TestCase):
    def setUp(self):
        self.times = np.asarray((0.0, 0.03, 0.11, 0.24, 0.29))
        self.sigma = np.asarray((0.8, 0.5, 0.3, 0.12, 0.09, 0.07))
        self.tau = 0.18
        self.process = GaussMarkovWrenchProcess(
            self.times, self.sigma, self.tau
        )

    def test_dimension_shape_and_zero_innovations(self):
        self.assertEqual(self.process.innovation_dimension, 6 * self.times.size)
        decoded = self.process.decode(
            np.zeros(self.process.innovation_dimension)
        )
        self.assertEqual(decoded.shape, (self.times.size, 6))
        np.testing.assert_array_equal(decoded, 0.0)

    def test_irregular_time_recurrence_is_exact(self):
        innovations = np.arange(
            1.0, self.process.innovation_dimension + 1.0
        ).reshape(self.times.size, 6) / 17.0
        decoded = self.process.decode(innovations.reshape(-1))
        expected = np.empty_like(decoded)
        expected[0] = self.sigma * innovations[0]
        for index in range(1, self.times.size):
            time_step = self.times[index] - self.times[index - 1]
            rho = np.exp(-time_step / self.tau)
            expected[index] = (
                rho * expected[index - 1]
                + self.sigma
                * np.sqrt(1.0 - rho**2)
                * innovations[index]
            )
        np.testing.assert_allclose(decoded, expected, atol=2.0e-16)

        # The unequal time steps must produce unequal transition factors.
        transition = np.exp(-np.diff(self.times) / self.tau)
        self.assertGreater(np.ptp(transition), 0.1)

    def test_sample_innovations_is_reproducible_and_recentered(self):
        first = self.process.sample_innovations(64, seed=31)
        repeated = self.process.sample_innovations(64, seed=31)
        different = self.process.sample_innovations(64, seed=32)
        self.assertEqual(first.shape, (64, self.process.innovation_dimension))
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, different))
        np.testing.assert_allclose(
            np.mean(first, axis=0), 0.0, atol=2.0e-16
        )

    def test_large_ensemble_has_stationary_covariance_and_ou_correlation(self):
        member_count = 30000
        standard = self.process.sample_innovations(member_count, seed=9)
        decoded = np.asarray(
            [self.process.decode(value) for value in standard]
        )

        empirical_variance = np.var(decoded, axis=0, ddof=1)
        expected_variance = np.broadcast_to(
            self.sigma**2, empirical_variance.shape
        )
        np.testing.assert_allclose(
            empirical_variance, expected_variance, rtol=0.025, atol=0.0
        )

        for index, time_step in enumerate(np.diff(self.times), start=1):
            empirical_cross_covariance = np.sum(
                decoded[:, index - 1] * decoded[:, index], axis=0
            ) / (member_count - 1.0)
            empirical_correlation = (
                empirical_cross_covariance / (self.sigma**2)
            )
            expected_correlation = np.exp(-time_step / self.tau)
            np.testing.assert_allclose(
                empirical_correlation,
                expected_correlation,
                rtol=0.025,
                atol=0.01,
            )

        # Independent wrench components retain negligible cross-covariance.
        covariance = np.cov(decoded[:, -1], rowvar=False)
        normalized = covariance / np.outer(self.sigma, self.sigma)
        off_diagonal = normalized - np.diag(np.diag(normalized))
        self.assertLess(np.max(np.abs(off_diagonal)), 0.025)

    def test_invalid_process_decode_and_sampling_inputs_are_rejected(self):
        invalid_times = (
            np.asarray(()),
            np.zeros((2, 2)),
            np.asarray((0.0, np.nan)),
            np.asarray((0.0, 0.0)),
            np.asarray((0.1, 0.0)),
        )
        for times in invalid_times:
            with self.assertRaisesRegex(ValueError, "times"):
                GaussMarkovWrenchProcess(times, self.sigma, self.tau)

        invalid_sigma = (
            np.ones(5),
            np.ones(7),
            np.asarray((1.0, 1.0, 1.0, 1.0, 1.0, 0.0)),
            np.asarray((1.0, 1.0, 1.0, 1.0, 1.0, -0.1)),
            np.full(6, np.nan),
        )
        for sigma in invalid_sigma:
            with self.assertRaisesRegex(ValueError, "standard_deviation"):
                GaussMarkovWrenchProcess(self.times, sigma, self.tau)
        for correlation_time in (0.0, -0.1, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "correlation_time"):
                GaussMarkovWrenchProcess(
                    self.times, self.sigma, correlation_time
                )

        for invalid in (
            np.zeros(self.process.innovation_dimension - 1),
            np.zeros(self.process.innovation_dimension + 1),
            np.zeros((self.times.size, 6)),
            np.full(self.process.innovation_dimension, np.nan),
        ):
            with self.assertRaisesRegex(ValueError, "finite values"):
                self.process.decode(invalid)
        for member_count in (0, -1, 2.5, True):
            with self.assertRaisesRegex(ValueError, "member_count"):
                self.process.sample_innovations(member_count, seed=1)


if __name__ == "__main__":
    unittest.main()
