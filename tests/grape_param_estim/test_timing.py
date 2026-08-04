import unittest

import numpy as np

from grape_param_estim.timing import (
    ZeroOrderHoldCommandHistory,
    validate_constant_delay,
    zero_order_hold_values,
)


class TimingTest(unittest.TestCase):
    def test_sub_sample_delay_is_not_rounded_to_publish_ticks(self):
        history = ZeroOrderHoldCommandHistory(0.025)
        history.append(0.0, "first")
        history.append(0.1, "second")
        self.assertEqual(history.value_at(0.124), "first")
        self.assertEqual(history.value_at(0.125), "second")
        np.testing.assert_allclose(history.switch_times(0.1, 0.2), (0.125,))

    def test_delay_can_cross_multiple_publish_periods(self):
        values = zero_order_hold_values(
            (0.0, 0.1, 0.2),
            np.asarray(((1.0,), (2.0,), (3.0,))),
            (0.0, 0.21, 0.29, 0.41),
            0.11,
        )
        np.testing.assert_array_equal(values[:, 0], (1.0, 1.0, 2.0, 3.0))

    def test_zero_delay_and_low_publish_rate(self):
        values = zero_order_hold_values(
            (0.0, 0.5), np.asarray((10.0, 20.0)), (0.49, 0.5), 0.0
        )
        np.testing.assert_array_equal(values, (10.0, 20.0))

    def test_invalid_delay_and_history_are_rejected(self):
        for value in (-0.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_constant_delay(value)
        history = ZeroOrderHoldCommandHistory(0.0)
        with self.assertRaises(ValueError):
            history.value_at(0.0)
        history.append(0.0, "first")
        with self.assertRaises(ValueError):
            history.append(0.0, "duplicate")

if __name__ == "__main__":
    unittest.main()
