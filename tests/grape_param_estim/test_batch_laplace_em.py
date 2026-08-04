import unittest

import numpy as np

from grape_param_estim.batch.laplace_em import (
    BODY_WRENCH_COMPONENT_NAMES,
    BODY_WRENCH_COMPONENT_UNITS,
    BODY_WRENCH_QUANTITY,
    DiagonalQDefinition,
    ExpectedResidualMoments,
    QInnerEvaluation,
    QIntervalModel,
    compute_diagonal_q_target,
    damped_diagonal_q_update,
)


def _definition():
    return DiagonalQDefinition(
        residual_quantity=BODY_WRENCH_QUANTITY,
        component_names=BODY_WRENCH_COMPONENT_NAMES,
        component_units=BODY_WRENCH_COMPONENT_UNITS,
        interval_model=QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY,
    )


def _inner(q, marginal=10.0, successful=True, reason="", warm="state"):
    return QInnerEvaluation(
        q=np.asarray(q, dtype=float),
        successful=successful,
        map_objective=8.0 if successful else float("inf"),
        approximate_marginal_objective=(
            marginal if successful else float("inf")
        ),
        lag=0.012 if successful else float("inf"),
        failure_reason=reason,
        warm_start=warm,
    )


class BatchLaplaceEmTests(unittest.TestCase):
    def setUp(self):
        self.residual = np.asarray(
            (
                (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                (2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
            )
        )
        self.correction = np.asarray(
            (
                (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                (0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
            )
        )
        self.moments = ExpectedResidualMoments(
            self.residual, self.correction
        )
        self.time_step = np.asarray((0.01, 0.03))
        self.floor = np.full(6, 1.0e-12)

    def test_spectral_density_target_uses_variable_interval_duration(self):
        result = compute_diagonal_q_target(
            _definition(),
            self.moments,
            self.time_step,
            self.floor,
        )
        expected_map = np.sum(
            self.time_step[:, None] * self.residual**2, axis=0
        ) / 2.0
        expected_correction = np.sum(
            self.time_step[:, None] * self.correction, axis=0
        ) / 2.0
        np.testing.assert_allclose(result.map_second_moment, expected_map)
        np.testing.assert_allclose(
            result.covariance_correction, expected_correction
        )
        np.testing.assert_allclose(
            result.raw_target, expected_map + expected_correction
        )

    def test_covariance_correction_prevents_hard_em_update(self):
        zero_residual = ExpectedResidualMoments(
            np.zeros((3, 6)), np.full((3, 6), 0.4)
        )
        result = compute_diagonal_q_target(
            _definition(),
            zero_residual,
            np.ones(3),
            np.full(6, 1.0e-6),
        )
        np.testing.assert_allclose(result.map_second_moment, 0.0)
        np.testing.assert_allclose(result.target, 0.4)
        self.assertFalse(np.any(result.floor_active))

    def test_component_floor_is_explicit_and_no_upper_cap_is_applied(self):
        moments = ExpectedResidualMoments(
            np.asarray(((0.0, 0.0, 0.0, 1.0e9, 2.0, 3.0),)),
            np.zeros((1, 6)),
        )
        floor = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        result = compute_diagonal_q_target(
            _definition(),
            moments,
            np.ones(1),
            floor,
        )
        np.testing.assert_array_equal(result.floor_active[:3], True)
        self.assertEqual(result.target[3], 1.0e18)
        self.assertEqual(result.target[4], 4.0)

    def test_log_q_backtracking_accepts_first_nonworsening_candidate(self):
        target = compute_diagonal_q_target(
            _definition(),
            ExpectedResidualMoments(np.full((1, 6), 4.0), np.zeros((1, 6))),
            np.ones(1),
            self.floor,
        )
        current = _inner(np.ones(6), marginal=10.0, warm="map-state")
        seen = []

        def evaluator(q, warm_start):
            seen.append((q.copy(), warm_start))
            alpha = np.log(q[0]) / np.log(16.0)
            marginal = 11.0 if alpha > 0.75 else 9.5
            return _inner(q, marginal=marginal, warm="candidate")

        result = damped_diagonal_q_update(current, target, evaluator)
        self.assertTrue(result.accepted)
        self.assertEqual(result.accepted_alpha, 0.5)
        np.testing.assert_allclose(result.accepted_q, 4.0)
        self.assertEqual(len(result.attempts), 2)
        self.assertFalse(result.attempts[0].accepted)
        self.assertTrue(result.attempts[1].accepted)
        self.assertEqual([item[1] for item in seen], ["map-state", "map-state"])
        self.assertAlmostEqual(result.max_log_q_change, np.log(4.0))

    def test_failed_inner_solves_are_rejected_without_changing_q(self):
        target = compute_diagonal_q_target(
            _definition(),
            self.moments,
            self.time_step,
            self.floor,
        )
        current = _inner(np.ones(6), marginal=10.0)

        def evaluator(q, warm_start):
            return _inner(
                q,
                successful=False,
                reason="lm_nonconverged",
                warm=None,
            )

        result = damped_diagonal_q_update(
            current,
            target,
            evaluator,
            minimum_alpha=0.25,
        )
        self.assertFalse(result.accepted)
        np.testing.assert_array_equal(result.accepted_q, current.q)
        self.assertEqual(result.accepted_alpha, 0.0)
        self.assertEqual(len(result.attempts), 3)
        self.assertEqual(result.max_log_q_change, 0.0)
        self.assertEqual(
            result.termination_reason,
            "all_damped_candidates_rejected",
        )

    def test_q_definition_is_strict_body_wrench_spectral_density(self):
        with self.assertRaises(TypeError):
            DiagonalQDefinition()
        with self.assertRaises(ValueError):
            DiagonalQDefinition(
                residual_quantity="specific_acceleration",
                component_names=BODY_WRENCH_COMPONENT_NAMES,
                component_units=BODY_WRENCH_COMPONENT_UNITS,
                interval_model=QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY,
            )
        with self.assertRaises(TypeError):
            DiagonalQDefinition(
                residual_quantity=BODY_WRENCH_QUANTITY,
                component_names=BODY_WRENCH_COMPONENT_NAMES,
                component_units=BODY_WRENCH_COMPONENT_UNITS,
                interval_model="fixed_interval_covariance",
            )


if __name__ == "__main__":
    unittest.main()
