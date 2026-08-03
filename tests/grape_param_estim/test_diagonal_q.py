import unittest

import numpy as np

from grape_param_estim.diagonal_q import (
    BODY_WRENCH_COMPONENT_ORDER,
    BODY_WRENCH_COMPONENT_UNITS,
    BODY_WRENCH_FRAME,
    BODY_WRENCH_VARIANCE_UNITS,
    BodyWrenchDiagonalCovariance,
    DiagonalQEmSufficientStatistics,
    diagonal_q_em_sufficient_statistics,
    ou_transition_factors,
    shared_diagonal_q_m_step,
)


class BodyWrenchDiagonalCovarianceTest(unittest.TestCase):
    def test_metadata_and_matrix_are_exactly_diagonal(self):
        variance = np.asarray((1.0, 2.0, 3.0, 0.4, 0.5, 0.6))
        covariance = BodyWrenchDiagonalCovariance(variance)

        self.assertEqual(covariance.frame, BODY_WRENCH_FRAME)
        self.assertEqual(covariance.component_order, BODY_WRENCH_COMPONENT_ORDER)
        self.assertEqual(covariance.component_units, BODY_WRENCH_COMPONENT_UNITS)
        self.assertEqual(covariance.variance_units, BODY_WRENCH_VARIANCE_UNITS)
        np.testing.assert_array_equal(
            covariance.stationary_standard_deviation, np.sqrt(variance)
        )
        np.testing.assert_array_equal(covariance.matrix, np.diag(variance))
        off_diagonal = covariance.matrix.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        np.testing.assert_array_equal(off_diagonal, 0.0)

        variance[:] = 99.0
        np.testing.assert_array_equal(
            covariance.stationary_variance,
            np.asarray((1.0, 2.0, 3.0, 0.4, 0.5, 0.6)),
        )

    def test_covariance_rejects_invalid_variances(self):
        for value in (
            np.ones(5),
            np.ones(7),
            np.asarray((1.0, 1.0, 1.0, 1.0, 1.0, 0.0)),
            np.asarray((1.0, 1.0, 1.0, 1.0, 1.0, -0.1)),
            np.full(6, np.nan),
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BodyWrenchDiagonalCovariance(value)


class OuTransitionFactorsTest(unittest.TestCase):
    def test_irregular_rho_and_innovation_variance_are_exact(self):
        times = np.asarray((0.0, 0.05, 0.21, 0.50))
        correlation_time = 0.18
        factors = ou_transition_factors(times, correlation_time)
        expected_ratio = np.diff(times) / correlation_time
        expected_rho = np.exp(-expected_ratio)
        expected_fraction = -np.expm1(-2.0 * expected_ratio)

        np.testing.assert_array_equal(factors.time_step, np.diff(times))
        np.testing.assert_allclose(factors.rho, expected_rho, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            factors.innovation_variance_fraction,
            expected_fraction,
            atol=0.0,
            rtol=0.0,
        )
        self.assertGreater(np.ptp(factors.rho), 0.4)

        covariance = BodyWrenchDiagonalCovariance(
            np.asarray((1.0, 2.0, 3.0, 0.1, 0.2, 0.3))
        )
        expected_variance = (
            expected_fraction[:, None]
            * covariance.stationary_variance[None, :]
        )
        np.testing.assert_array_equal(
            factors.innovation_variance(covariance), expected_variance
        )
        matrices = factors.innovation_covariance(covariance)
        for index in range(times.size - 1):
            np.testing.assert_array_equal(
                matrices[index], np.diag(expected_variance[index])
            )

    def test_transition_validation_is_strict(self):
        for times in (
            np.asarray(()),
            np.zeros((2, 2)),
            np.asarray((0.0, np.nan)),
            np.asarray((0.0, 0.0)),
            np.asarray((1.0, 0.0)),
        ):
            with self.subTest(times=times), self.assertRaisesRegex(
                ValueError, "times"
            ):
                ou_transition_factors(times, 0.2)
        for correlation_time in (0.0, -0.1, np.nan, np.inf, None):
            with self.subTest(
                correlation_time=correlation_time
            ), self.assertRaisesRegex(ValueError, "correlation_time"):
                ou_transition_factors((0.0, 0.1), correlation_time)


class DiagonalQEmTest(unittest.TestCase):
    def test_member_first_statistics_and_initial_term_match_hand_calculation(self):
        times = np.asarray((0.0, 0.2, 0.55))
        tau = 0.4
        wrench = np.asarray(
            (
                (
                    (1.0, 2.0, 3.0, 0.1, 0.2, 0.3),
                    (1.3, 1.7, 2.8, 0.2, 0.1, 0.4),
                    (0.9, 1.8, 3.2, 0.0, 0.3, 0.2),
                ),
                (
                    (-0.5, 1.0, 2.0, -0.2, 0.4, 0.1),
                    (-0.1, 0.8, 2.4, -0.1, 0.5, 0.0),
                    (0.2, 0.6, 2.1, -0.3, 0.2, 0.2),
                ),
            )
        )
        statistics = diagonal_q_em_sufficient_statistics(
            "bag-a", times, tau, wrench
        )
        rho = np.exp(-np.diff(times) / tau)
        fraction = 1.0 - rho**2
        expected_initial = np.mean(wrench[:, 0, :] ** 2, axis=0)
        residual = wrench[:, 1:, :] - rho[None, :, None] * wrench[:, :-1, :]
        expected_transition = np.mean(residual**2, axis=0)
        expected_numerator = expected_initial + np.sum(
            expected_transition / fraction[:, None], axis=0
        )

        self.assertEqual(statistics.member_count, 2)
        self.assertEqual(statistics.boundary_count, 3)
        self.assertEqual(statistics.transition_count, 2)
        np.testing.assert_allclose(
            statistics.initial_second_moment, expected_initial, atol=1.0e-15
        )
        np.testing.assert_allclose(
            statistics.transition_second_moment,
            expected_transition,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            statistics.m_step_numerator, expected_numerator, atol=1.0e-15
        )

        update = shared_diagonal_q_m_step(
            statistics=(statistics,), variance_floor=1.0e-12
        )
        np.testing.assert_allclose(
            update.raw_stationary_variance,
            expected_numerator / times.size,
            atol=1.0e-15,
        )

        # The stationary initial term is a real likelihood term, not a
        # transition: removing it must change every non-zero component.
        without_initial = (
            expected_numerator - expected_initial
        ) / times.size
        self.assertTrue(
            np.all(update.raw_stationary_variance > without_initial)
        )

    def test_shared_m_step_weights_unequal_multi_bag_windows_by_boundaries(self):
        first_times = np.asarray((0.0, 0.1, 0.4, 0.9))
        second_times = np.asarray((0.0, 0.25))
        first_tau = 0.3
        second_tau = 0.8
        first = np.asarray(
            (
                np.arange(24, dtype=float).reshape(4, 6) / 10.0,
                -np.arange(24, dtype=float).reshape(4, 6) / 20.0,
                np.ones((4, 6)) * np.asarray((0.2, 0.3, 0.4, 0.05, 0.06, 0.07)),
            )
        )
        second = np.asarray(
            (
                np.asarray(((0.2,) * 6, (0.5,) * 6)),
                np.asarray(((-0.3,) * 6, (0.1,) * 6)),
            )
        )
        first_stats = diagonal_q_em_sufficient_statistics(
            "bag-z", first_times, first_tau, first
        )
        second_stats = diagonal_q_em_sufficient_statistics(
            "bag-a", second_times, second_tau, second
        )
        expected = (
            first_stats.m_step_numerator + second_stats.m_step_numerator
        ) / float(first_times.size + second_times.size)

        update = shared_diagonal_q_m_step(
            (first_stats, second_stats), variance_floor=1.0e-14
        )
        reversed_update = shared_diagonal_q_m_step(
            (second_stats, first_stats), variance_floor=1.0e-14
        )
        self.assertEqual(update.bag_ids, ("bag-a", "bag-z"))
        self.assertEqual(update.total_boundary_count, 6)
        self.assertEqual(update.total_transition_count, 4)
        np.testing.assert_allclose(
            update.raw_stationary_variance, expected, atol=2.0e-16
        )
        np.testing.assert_array_equal(
            update.raw_stationary_variance,
            reversed_update.raw_stationary_variance,
        )
        np.testing.assert_array_equal(
            update.covariance.stationary_variance,
            reversed_update.covariance.stationary_variance,
        )

    def test_floor_provenance_is_componentwise_and_deterministic(self):
        times = np.asarray((0.0, 0.2))
        wrench = np.zeros((2, 2, 6))
        wrench[:, :, 1] = np.asarray(((2.0, 2.0), (-2.0, -2.0)))
        statistics = diagonal_q_em_sufficient_statistics(
            "bag-a", times, 0.5, wrench
        )
        floor = np.asarray((0.1, 0.2, 0.3, 0.04, 0.05, 0.06))

        first = shared_diagonal_q_m_step((statistics,), floor)
        repeated = shared_diagonal_q_m_step((statistics,), floor.copy())
        expected_applied = first.raw_stationary_variance < floor
        np.testing.assert_array_equal(first.floor_applied, expected_applied)
        np.testing.assert_array_equal(
            first.covariance.stationary_variance,
            np.maximum(first.raw_stationary_variance, floor),
        )
        self.assertFalse(first.floor_applied[1])
        self.assertTrue(np.all(first.floor_applied[np.arange(6) != 1]))
        np.testing.assert_array_equal(
            first.raw_stationary_variance,
            repeated.raw_stationary_variance,
        )
        np.testing.assert_array_equal(
            first.floor_applied, repeated.floor_applied
        )

    def test_statistics_and_m_step_reject_invalid_inputs(self):
        valid = np.zeros((2, 3, 6))
        for wrench in (
            np.zeros((3, 6)),
            np.zeros((2, 2, 6)),
            np.zeros((2, 3, 5)),
            np.full((2, 3, 6), np.nan),
        ):
            with self.subTest(shape=wrench.shape), self.assertRaisesRegex(
                ValueError, "member-first"
            ):
                diagonal_q_em_sufficient_statistics(
                    "bag-a", (0.0, 0.1, 0.2), 0.3, wrench
                )
        with self.assertRaisesRegex(ValueError, "bag_id"):
            diagonal_q_em_sufficient_statistics(
                "", (0.0, 0.1, 0.2), 0.3, valid
            )
        with self.assertRaisesRegex(ValueError, "bag_id"):
            diagonal_q_em_sufficient_statistics(
                3, (0.0, 0.1, 0.2), 0.3, valid
            )

        statistic = diagonal_q_em_sufficient_statistics(
            "bag-a", (0.0, 0.1, 0.2), 0.3, valid
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            shared_diagonal_q_m_step((), 1.0e-6)
        with self.assertRaisesRegex(ValueError, "unique"):
            shared_diagonal_q_m_step((statistic, statistic), 1.0e-6)
        for floor in (0.0, -1.0, np.nan, np.ones(5), np.zeros(6)):
            with self.subTest(floor=floor), self.assertRaisesRegex(
                ValueError, "variance_floor"
            ):
                shared_diagonal_q_m_step((statistic,), floor)

        with self.assertRaises(ValueError):
            DiagonalQEmSufficientStatistics(
                bag_id="bag-a",
                member_count=2,
                times=np.asarray((0.0, 0.1)),
                correlation_time=0.3,
                initial_second_moment=np.zeros(6),
                transition_second_moment=np.zeros((2, 6)),
            )


if __name__ == "__main__":
    unittest.main()
