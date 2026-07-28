import unittest

import numpy as np

from grape_param_estim.controller_sample import (
    SamplePidAxis,
    command_to_wrench,
)


class ControllerSampleTests(unittest.TestCase):
    def test_equal_hover_commands_produce_finite_vertical_wrench(self):
        wrench = command_to_wrench(
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(wrench.shape, (6,))
        self.assertTrue(np.all(np.isfinite(wrench)))
        self.assertAlmostEqual(wrench[0], 0.0, places=12)
        self.assertAlmostEqual(wrench[1], 0.0, places=12)
        self.assertAlmostEqual(wrench[2], 4.0, places=12)

    def test_pid_anti_windup_reverts_saturating_integral(self):
        controller = SamplePidAxis(
            proportional_gain=2.0,
            integral_gain=1.0,
            derivative_gain=0.0,
            integral_limit=10.0,
            output_limit=1.0,
        )
        output, terms = controller.step(1.0, 0.1)
        self.assertEqual(output, 1.0)
        self.assertEqual(controller.integral, 0.0)
        self.assertEqual(terms[1], 0.0)

    def test_pid_reset_clears_history(self):
        controller = SamplePidAxis(1.0, 0.1, 0.2, 1.0, 10.0)
        controller.step(0.5, 0.1)
        controller.reset()
        self.assertEqual(controller.integral, 0.0)
        self.assertIsNone(controller.previous_error)


if __name__ == "__main__":
    unittest.main()
