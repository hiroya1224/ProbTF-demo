from pathlib import Path
import sys
import unittest

import numpy as np


_MINIMAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_MINIMAL))

from smooth_command import QuinticSmoothZoh  # noqa: E402


class QuinticSmoothZohTests(unittest.TestCase):
    def test_outside_transition_is_exact_zoh(self):
        history = QuinticSmoothZoh(
            (0.0, 1.0, 2.0),
            np.asarray(((0.0,), (2.0,), (-1.0,))),
        )
        for time in (0.2, 0.6, 1.4, 1.8, 2.4):
            evaluation = history.evaluate(time, 0.1, 0.1)
            np.testing.assert_array_equal(
                evaluation.value,
                history.exact_zoh(time, 0.1),
            )
            np.testing.assert_array_equal(
                evaluation.delay_derivative,
                np.zeros(1),
            )

    def test_transition_edges_are_c2(self):
        history = QuinticSmoothZoh(
            (0.0, 1.0, 2.0),
            np.asarray(((0.0,), (1.0,), (3.0,))),
        )
        epsilon = history.transition_half_widths(0.2)[0]
        step = 1.0e-5
        for edge in (1.0 - epsilon, 1.0 + epsilon):
            center = history.evaluate(edge, 0.0, 0.2).value
            left = history.evaluate(edge - step, 0.0, 0.2).value
            right = history.evaluate(edge + step, 0.0, 0.2).value
            first = (right - left) / (2.0 * step)
            second = (right - 2.0 * center + left) / step**2
            np.testing.assert_allclose(first, 0.0, atol=2.0e-8)
            np.testing.assert_allclose(second, 0.0, atol=5.0e-3)

    def test_constant_command_has_zero_delay_derivative(self):
        history = QuinticSmoothZoh(
            (0.0, 0.7, 1.5),
            np.tile(np.asarray((1.0, -2.0)), (3, 1)),
        )
        for time in np.linspace(-0.2, 2.0, 21):
            np.testing.assert_array_equal(
                history.evaluate(time, 0.1, 0.5).delay_derivative,
                np.zeros(2),
            )

    def test_delay_derivative_matches_central_difference(self):
        history = QuinticSmoothZoh(
            (0.0, 1.0, 2.0),
            np.asarray(((0.0, 2.0), (3.0, -1.0), (4.0, 5.0))),
        )
        time = 1.04
        delay = 0.03
        step = 1.0e-7
        analytic = history.evaluate(time, delay, 0.2).delay_derivative
        numerical = (
            history.evaluate(time, delay + step, 0.2).value
            - history.evaluate(time, delay - step, 0.2).value
        ) / (2.0 * step)
        np.testing.assert_allclose(analytic, numerical, rtol=1.0e-8, atol=1.0e-9)

    def test_small_width_converges_to_zoh_away_from_switches(self):
        history = QuinticSmoothZoh(
            (0.0, 1.0, 2.0),
            np.asarray(((0.0,), (2.0,), (-1.0,))),
        )
        for time in (0.25, 0.75, 1.25, 1.75, 2.25):
            np.testing.assert_array_equal(
                history.evaluate(time, 0.0, 1.0e-6).value,
                history.exact_zoh(time, 0.0),
            )

    def test_uneven_transitions_do_not_overlap_and_duplicates_keep_last(self):
        history = QuinticSmoothZoh(
            (0.0, 1.0, 1.0, 1.2, 3.0),
            np.asarray(((0.0,), (1.0,), (2.0,), (3.0,), (4.0,))),
        )
        np.testing.assert_array_equal(history.times, (0.0, 1.0, 1.2, 3.0))
        self.assertEqual(history.exact_zoh(1.0, 0.0)[0], 2.0)
        widths = history.transition_half_widths(0.5)
        gaps = np.diff(history.times)
        self.assertTrue(np.all(widths[:-1] + widths[1:] < gaps[1:]))


if __name__ == "__main__":
    unittest.main()
