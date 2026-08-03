import unittest

import numpy as np

from grape_param_estim.ensemble_state_smoother import (
    deterministic_square_root_update,
    ensemble_rts_smoother,
    ensemble_rts_smoothing_step,
    exact_gaussian_ensemble,
)


def _sample_covariance(values):
    selected = np.asarray(values, dtype=float)
    anomalies = selected - np.mean(selected, axis=0, keepdims=True)
    return anomalies.T @ anomalies / (selected.shape[0] - 1.0)


class ExactGaussianEnsembleTest(unittest.TestCase):
    def test_mean_covariance_and_cross_covariance_are_exact(self):
        mean = np.asarray((1.2, -0.7, 0.4))
        covariance = np.asarray(
            ((0.5, 0.1, -0.04), (0.1, 0.3, 0.02), (-0.04, 0.02, 0.2))
        )
        reference = exact_gaussian_ensemble(
            np.zeros(2), np.asarray(((1.0, 0.2), (0.2, 0.8))), 12, 3
        )
        values = exact_gaussian_ensemble(
            mean, covariance, 12, 7, orthogonal_to=reference
        )
        repeated = exact_gaussian_ensemble(
            mean, covariance, 12, 7, orthogonal_to=reference
        )

        np.testing.assert_allclose(np.mean(values, axis=0), mean, atol=3.0e-16)
        np.testing.assert_allclose(
            _sample_covariance(values), covariance, atol=3.0e-15
        )
        cross = (
            (reference - np.mean(reference, axis=0)).T
            @ (values - np.mean(values, axis=0))
            / 11.0
        )
        np.testing.assert_allclose(cross, 0.0, atol=3.0e-16)
        np.testing.assert_array_equal(values, repeated)

    def test_orthogonal_dimension_and_input_validation_are_strict(self):
        reference = exact_gaussian_ensemble(
            np.zeros(3), np.eye(3), member_count=5, seed=1
        )
        with self.assertRaisesRegex(ValueError, "orthogonal anomaly"):
            exact_gaussian_ensemble(
                np.zeros(2),
                np.eye(2),
                member_count=5,
                seed=2,
                orthogonal_to=reference,
            )
        for covariance in (
            np.eye(3),
            np.asarray(((1.0, 2.0), (0.0, 1.0))),
            np.asarray(((1.0, 0.0), (0.0, 0.0))),
        ):
            with self.subTest(covariance=covariance), self.assertRaises(
                ValueError
            ):
                exact_gaussian_ensemble(
                    np.zeros(2), covariance, member_count=6, seed=2
                )


class DeterministicEnsembleUpdateTest(unittest.TestCase):
    def test_linear_update_matches_analytic_kalman_mean_and_covariance(self):
        mean = np.asarray((0.6, -0.2))
        covariance = np.asarray(((0.8, 0.25), (0.25, 0.5)))
        observation_matrix = np.asarray(((1.2, -0.4),))
        observation_covariance = np.asarray(((0.3,),))
        observation = np.asarray((1.1,))
        forecast = exact_gaussian_ensemble(
            mean, covariance, member_count=8, seed=11
        )
        predicted = forecast @ observation_matrix.T

        result = deterministic_square_root_update(
            forecast,
            predicted,
            observation,
            observation_covariance,
        )
        innovation_covariance = (
            observation_matrix @ covariance @ observation_matrix.T
            + observation_covariance
        )
        gain = (
            covariance
            @ observation_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )
        expected_mean = mean + gain @ (observation - observation_matrix @ mean)
        expected_covariance = covariance - (
            gain @ observation_matrix @ covariance
        )

        np.testing.assert_allclose(
            np.mean(result.analysis_ensemble, axis=0),
            expected_mean,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            _sample_covariance(result.analysis_ensemble),
            expected_covariance,
            atol=4.0e-15,
        )
        np.testing.assert_allclose(result.kalman_gain, gain, atol=2.0e-15)
        np.testing.assert_allclose(
            result.innovation_covariance,
            innovation_covariance,
            atol=2.0e-15,
        )
        expected_log_likelihood = -0.5 * (
            np.log(2.0 * np.pi * innovation_covariance[0, 0])
            + (observation - observation_matrix @ mean)[0] ** 2
            / innovation_covariance[0, 0]
        )
        self.assertAlmostEqual(
            result.approximate_log_likelihood,
            expected_log_likelihood,
            places=14,
        )

    def test_update_rejects_member_and_covariance_mismatch(self):
        forecast = np.zeros((4, 2))
        predicted = np.zeros((4, 1))
        with self.assertRaises(ValueError):
            deterministic_square_root_update(
                forecast, predicted[:3], np.zeros(1), np.eye(1)
            )
        with self.assertRaises(ValueError):
            deterministic_square_root_update(
                forecast,
                predicted,
                np.zeros(1),
                np.asarray(((0.0,),)),
            )


class EnsembleRtsSmootherTest(unittest.TestCase):
    def test_scalar_filter_and_smoother_match_analytic_kalman_rts(self):
        member_count = 8
        transition = 0.85
        process_variance = 0.18
        observation_variance = 0.25
        observations = (0.4, 1.1, 0.7)

        forecast_ensembles = []
        analysis_ensembles = []
        forecast_means = []
        forecast_variances = []
        analysis_means = []
        analysis_variances = []

        forecast = exact_gaussian_ensemble(
            np.asarray((0.2,)),
            np.asarray(((0.9,),)),
            member_count,
            seed=21,
        )
        for index, observation in enumerate(observations):
            forecast_ensembles.append(forecast)
            forecast_mean = float(np.mean(forecast))
            forecast_variance = float(_sample_covariance(forecast)[0, 0])
            forecast_means.append(forecast_mean)
            forecast_variances.append(forecast_variance)
            update = deterministic_square_root_update(
                forecast,
                forecast.copy(),
                np.asarray((observation,)),
                np.asarray(((observation_variance,),)),
            )
            analysis = update.analysis_ensemble
            analysis_ensembles.append(analysis)
            analytic_gain = forecast_variance / (
                forecast_variance + observation_variance
            )
            analysis_means.append(
                forecast_mean
                + analytic_gain * (observation - forecast_mean)
            )
            analysis_variances.append(
                (1.0 - analytic_gain) * forecast_variance
            )
            if index + 1 < len(observations):
                noise = exact_gaussian_ensemble(
                    np.zeros(1),
                    np.asarray(((process_variance,),)),
                    member_count,
                    seed=30 + index,
                    orthogonal_to=analysis,
                )
                forecast = transition * analysis + noise

        result = ensemble_rts_smoother(
            analysis_ensembles, forecast_ensembles
        )
        expected_means = list(analysis_means)
        expected_variances = list(analysis_variances)
        for index in range(len(observations) - 2, -1, -1):
            smoothing_gain = (
                analysis_variances[index]
                * transition
                / forecast_variances[index + 1]
            )
            expected_means[index] = analysis_means[index] + smoothing_gain * (
                expected_means[index + 1] - forecast_means[index + 1]
            )
            expected_variances[index] = analysis_variances[index] + (
                smoothing_gain**2
                * (
                    expected_variances[index + 1]
                    - forecast_variances[index + 1]
                )
            )

        for index, ensemble in enumerate(result.smoothed_ensembles):
            self.assertAlmostEqual(
                float(np.mean(ensemble)), expected_means[index], places=13
            )
            # Member-wise EnRTS is a finite-ensemble approximation: the
            # deterministic filter transform preserves the Kalman moments,
            # but its future member coupling need not reproduce the exact RTS
            # covariance.  It should remain close on this linear audit case.
            self.assertTrue(
                np.isclose(
                    float(_sample_covariance(ensemble)[0, 0]),
                    expected_variances[index],
                    rtol=0.02,
                    atol=2.0e-4,
                )
            )
        self.assertEqual(len(result.smoothing_gains), 2)

    def test_single_step_supports_different_adjacent_state_dimensions(self):
        analysis = exact_gaussian_ensemble(
            np.zeros(2), np.eye(2), member_count=6, seed=2
        )
        next_forecast = analysis[:, :1] * 0.7
        next_smoothed = next_forecast + 0.2
        smoothed, gain = ensemble_rts_smoothing_step(
            analysis, next_forecast, next_smoothed
        )
        self.assertEqual(smoothed.shape, (6, 2))
        self.assertEqual(gain.shape, (2, 1))
        self.assertTrue(np.all(np.isfinite(smoothed)))


if __name__ == "__main__":
    unittest.main()
