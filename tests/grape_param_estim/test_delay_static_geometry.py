import unittest

import numpy as np

from grape_param_estim.batch.evidence import (
    FALLBACK_SURROGATE_METHOD,
    JOINT_SURROGATE_METHOD,
    compute_delay_static_laplace_geometry,
    mcmc_parameter_delay_initialization_covariance,
    mcmc_quadratic_surrogate_information,
)
from grape_param_estim.batch.lag_profile import LagProfilePoint, LagProfileResult


def _quadratic_profile(
    support,
    center,
    curvature,
    sensitivity,
    *,
    gradient=0.0,
    best_lag=None,
):
    coordinate_center = np.linspace(-0.3, 0.4, 18)
    points = []
    for lag in support:
        offset = lag - center
        points.append(
            LagProfilePoint(
                lag=lag,
                phase="refinement",
                objective=4.0 + gradient * offset + 0.5 * curvature * offset**2,
                converged=True,
                inner_iterations=2,
                termination_reason="synthetic",
                warm_start_lag=None,
                approximate_marginal_objective=(
                    8.0 + 0.25 * offset + 0.2 * offset**2
                ),
                static_coordinate=coordinate_center + sensitivity * offset,
            )
        )
    selected_best = center if best_lag is None else best_lag
    center_point = next(point for point in points if point.lag == selected_best)
    profile = LagProfileResult(
        best_lag=selected_best,
        best_objective=center_point.objective,
        best_state=None,
        initial_refinement_bracket=(min(support), max(support)),
        final_refinement_bracket=(min(support), max(support)),
        points=tuple(points),
    )
    return profile, coordinate_center


class DelayStaticGeometryTests(unittest.TestCase):
    def setUp(self):
        self.center = 0.031
        self.bounds = (0.0, 0.08)
        self.support = (0.022, self.center, 0.044)
        self.curvature = 2.5e5
        self.sensitivity = np.linspace(-1.2, 0.8, 18)
        diagonal = np.linspace(2.0, 7.0, 18)
        self.information = np.diag(diagonal)

    def _compute(self, profile, coordinate):
        return compute_delay_static_laplace_geometry(
            (profile,),
            self.bounds,
            self.information,
            self.center,
            coordinate,
            1.0e-5,
        )

    def test_nonuniform_quadratic_support_recovers_exact_joint_algebra(self):
        profile, coordinate = _quadratic_profile(
            self.support,
            self.center,
            self.curvature,
            self.sensitivity,
        )
        result = self._compute(profile, coordinate)
        self.assertTrue(result.valid)
        self.assertEqual(
            result.mcmc_quadratic_surrogate_method, JOINT_SURROGATE_METHOD
        )
        self.assertAlmostEqual(result.profile_gradient, 0.0, places=10)
        self.assertAlmostEqual(result.curvature, self.curvature, places=7)
        np.testing.assert_allclose(result.static_sensitivity, self.sensitivity)
        expected_cross = self.sensitivity / self.curvature
        expected_static_marginal = np.linalg.inv(self.information) + np.outer(
            self.sensitivity, self.sensitivity
        ) / self.curvature
        np.testing.assert_allclose(
            result.parameter_delay_cross_covariance, expected_cross
        )
        np.testing.assert_allclose(
            result.joint_covariance[:-1, :-1], expected_static_marginal
        )
        np.testing.assert_allclose(
            result.joint_information @ result.joint_covariance,
            np.eye(19),
            atol=1.0e-12,
        )
        self.assertGreater(
            np.linalg.norm(result.joint_information[:-1, -1]), 0.0
        )
        np.testing.assert_allclose(
            mcmc_quadratic_surrogate_information(self.information, result),
            result.joint_information,
        )
        np.testing.assert_allclose(
            mcmc_parameter_delay_initialization_covariance(
                np.linalg.inv(self.information), result
            ),
            result.joint_covariance,
        )

    def test_boundary_missing_support_nonpositive_and_stationarity_are_invalid(self):
        cases = []
        boundary, coordinate = _quadratic_profile(
            (0.0, 0.01, 0.02),
            0.0,
            self.curvature,
            self.sensitivity,
            best_lag=0.0,
        )
        cases.append((boundary, coordinate, 0.0, "profile_optimum_at_boundary"))
        missing, coordinate = _quadratic_profile(
            (self.center - 0.01, self.center),
            self.center,
            self.curvature,
            self.sensitivity,
        )
        cases.append(
            (missing, coordinate, self.center, "missing_bilateral_profile_support")
        )
        concave, coordinate = _quadratic_profile(
            self.support,
            self.center,
            -self.curvature,
            self.sensitivity,
        )
        cases.append(
            (concave, coordinate, self.center, "nonpositive_profile_curvature")
        )
        nonstationary, coordinate = _quadratic_profile(
            self.support,
            self.center,
            self.curvature,
            self.sensitivity,
            gradient=0.7 * self.curvature * min(
                self.center - self.support[0],
                self.support[2] - self.center,
            ),
        )
        cases.append(
            (nonstationary, coordinate, self.center, "unstable_profile_stationarity")
        )
        for profile, expected_coordinate, delay, reason in cases:
            with self.subTest(reason=reason):
                result = compute_delay_static_laplace_geometry(
                    (profile,),
                    self.bounds,
                    self.information,
                    delay,
                    expected_coordinate,
                    1.0e-5,
                )
                self.assertFalse(result.valid)
                self.assertEqual(result.reason, reason)
                self.assertEqual(
                    result.mcmc_quadratic_surrogate_method,
                    FALLBACK_SURROGATE_METHOD,
                )
                self.assertEqual(result.joint_information.shape, (0, 0))
                self.assertEqual(result.joint_covariance.shape, (0, 0))
                self.assertEqual(
                    result.parameter_delay_cross_covariance.shape, (0,)
                )
                self.assertAlmostEqual(
                    result.standard_deviation_seconds,
                    (self.bounds[1] - self.bounds[0]) / np.sqrt(12.0),
                )
                fallback = mcmc_quadratic_surrogate_information(
                    self.information, result, 0.003
                )
                np.testing.assert_allclose(
                    fallback[:-1, -1], np.zeros(18)
                )
                np.testing.assert_allclose(
                    fallback[-1, :-1], np.zeros(18)
                )
                np.testing.assert_allclose(
                    fallback[:-1, :-1], self.information
                )
                self.assertAlmostEqual(fallback[-1, -1], 1.0 / 0.003**2)
                initialization_covariance = (
                    mcmc_parameter_delay_initialization_covariance(
                        np.linalg.inv(self.information), result, 0.003
                    )
                )
                np.testing.assert_allclose(
                    initialization_covariance[:-1, -1], np.zeros(18)
                )
                np.testing.assert_allclose(
                    initialization_covariance[:-1, :-1],
                    np.linalg.inv(self.information),
                )
                self.assertAlmostEqual(
                    initialization_covariance[-1, -1], 0.003**2
                )
                self.assertNotAlmostEqual(
                    result.standard_deviation_seconds, 0.003
                )

    def test_map_center_mismatch_and_non_spd_information_raise(self):
        profile, coordinate = _quadratic_profile(
            self.support,
            self.center,
            self.curvature,
            self.sensitivity,
        )
        with self.assertRaisesRegex(ValueError, "center delay disagrees"):
            compute_delay_static_laplace_geometry(
                (profile,),
                self.bounds,
                self.information,
                self.center + 0.001,
                coordinate,
                1.0e-5,
            )
        with self.assertRaisesRegex(ValueError, "center static coordinate"):
            compute_delay_static_laplace_geometry(
                (profile,),
                self.bounds,
                self.information,
                self.center,
                coordinate + 0.01,
                1.0e-5,
            )
        non_spd = self.information.copy()
        non_spd[0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "positive definite"):
            compute_delay_static_laplace_geometry(
                (profile,),
                self.bounds,
                non_spd,
                self.center,
                coordinate,
                1.0e-5,
            )


if __name__ == "__main__":
    unittest.main()
