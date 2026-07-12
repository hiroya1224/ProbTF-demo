import unittest

import numpy as np

from deflecomp_core.estimator.initialization import initial_log_kp_state, initial_log_kp_std
from deflecomp_core.model.spring import (
    JointTypeAwareSpringModel,
    LinearSpringModel,
    PeriodicSpringModel,
    spring_model_from_name,
)
from deflecomp_core.observation.imu_buffer import ImuBuffer


class BridgeExtractionTest(unittest.TestCase):
    def test_imu_buffer_interpolates_out_of_order_samples(self):
        buffer = ImuBuffer(maxlen=3)
        buffer.push(2.0, [0.0, 1.0, 0.0])
        buffer.push(0.0, [1.0, 0.0, 0.0])

        expected = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        np.testing.assert_allclose(buffer.interpolate(1.0), expected, atol=1e-10)

    def test_log_stiffness_initialization_uses_configured_range(self):
        limits = (1.0, 100.0)
        state = initial_log_kp_state(3, limits)

        np.testing.assert_allclose(np.exp(state), [10.0, 10.0, 10.0])
        self.assertAlmostEqual(initial_log_kp_std(limits), np.log(100.0) / 4.0)

    def test_spring_factory_returns_requested_model(self):
        self.assertIsInstance(spring_model_from_name("linear"), LinearSpringModel)
        self.assertIsInstance(spring_model_from_name("periodic"), PeriodicSpringModel)
        self.assertIsInstance(
            spring_model_from_name("auto", ["revolute", "prismatic"]),
            JointTypeAwareSpringModel,
        )


if __name__ == "__main__":
    unittest.main()
