from pathlib import Path
import sys
import unittest

import numpy as np


_MINIMAL = (
    Path(__file__).resolve().parents[2]
    / "ros"
    / "examples"
    / "grape-param-estim"
    / "minimal"
)
sys.path.insert(0, str(_MINIMAL))

import deterministic_q_estimator as estimator  # noqa: E402


class DeterministicQTests(unittest.TestCase):
    def test_closed_form_target_minimizes_diagonal_gaussian_nll(self):
        residual = np.asarray(
            ((1.0, -2.0, 3.0, 0.4, -0.5, 0.6),
             (2.0, 1.0, -1.0, -0.2, 0.3, -0.4)),
            dtype=float,
        )
        time_step = np.asarray((0.02, 0.05), dtype=float)
        floor = np.full(6, 1.0e-12)
        target = estimator.q_target(residual, time_step, floor)
        expected = np.mean(
            time_step[:, None] * residual * residual, axis=0
        )
        np.testing.assert_allclose(target, expected)
        target_nll = estimator.q_negative_log_likelihood(
            residual, time_step, target
        )
        self.assertLess(
            target_nll,
            estimator.q_negative_log_likelihood(
                residual, time_step, 0.5 * target
            ),
        )
        self.assertLess(
            target_nll,
            estimator.q_negative_log_likelihood(
                residual, time_step, 2.0 * target
            ),
        )

    def test_floor_is_applied_componentwise(self):
        residual = np.zeros((3, 6), dtype=float)
        floor = np.asarray((1, 2, 3, 4, 5, 6), dtype=float) * 1.0e-8
        np.testing.assert_array_equal(
            estimator.q_target(residual, np.ones(3), floor), floor
        )


if __name__ == "__main__":
    unittest.main()
