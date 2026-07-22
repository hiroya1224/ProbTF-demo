import unittest

import numpy as np

from grape_param_estim.grape_geometry import allocate_wrench, reconstruct_actuator_wrench


class GrapeGeometryTests(unittest.TestCase):
    def test_symmetric_vertical_thrust_has_no_horizontal_force(self):
        wrench = reconstruct_actuator_wrench(np.full(4, 5.0), np.zeros(4))
        np.testing.assert_allclose(wrench[:2], np.zeros(2), atol=1.0e-12)
        self.assertAlmostEqual(wrench[2], 20.0, places=12)
        np.testing.assert_allclose(
            wrench[3:],
            [0.0220000168588264, -0.345999937364882, 0.0],
            atol=1.0e-10,
        )

    def test_synthetic_allocator_round_trip(self):
        target = np.array([1.0, -0.7, 22.0, 0.15, -0.12, 0.08])
        thrust, angle, normalized_residual = allocate_wrench(target)
        predicted = reconstruct_actuator_wrench(thrust, angle)
        self.assertLess(normalized_residual, 1.0e-5)
        np.testing.assert_allclose(predicted, target, rtol=0.0, atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
