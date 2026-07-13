import unittest

import numpy as np

from deflecomp_core.observation.imu_buffer import (
    ImuBuffer,
    TimedVectorHistory,
    imu_sample_is_quasi_static,
)


class ImuBufferTest(unittest.TestCase):
    def test_interpolation_rejects_stale_endpoint(self):
        buffer = ImuBuffer()
        buffer.push(1.0, np.array([0.0, 0.0, -1.0]))

        self.assertIsNotNone(buffer.interpolate(1.05, max_age=0.1))
        self.assertIsNone(buffer.interpolate(1.2, max_age=0.1))

    def test_endpoint_hold_preserves_actual_source_stamp(self):
        buffer = ImuBuffer()
        buffer.push(1.0, np.array([0.0, 0.0, -1.0]))

        first = buffer.interpolate_with_support_stamp(1.02, max_age=0.1)
        second = buffer.interpolate_with_support_stamp(1.08, max_age=0.1)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first[1], 1.0)
        self.assertEqual(second[1], 1.0)

    def test_interpolation_reports_newest_support_stamp(self):
        buffer = ImuBuffer()
        buffer.push(1.0, np.array([0.0, 0.0, -1.0]))
        buffer.push(1.1, np.array([0.0, 1.0, 0.0]))

        sample = buffer.interpolate_with_support_stamp(1.05, max_age=0.1)

        self.assertIsNotNone(sample)
        self.assertEqual(sample[1], 1.1)

    def test_causal_interpolation_rejects_first_future_sample(self):
        buffer = ImuBuffer()
        buffer.push(1.0, np.array([0.0, 0.0, -1.0]))

        self.assertIsNone(buffer.interpolate(0.999, max_age=0.1))
        self.assertIsNotNone(buffer.interpolate(1.0, max_age=0.1))

    def test_interpolation_does_not_bridge_stale_gap(self):
        buffer = ImuBuffer()
        buffer.push(1.0, np.array([0.0, 0.0, -1.0]))
        buffer.push(2.0, np.array([0.0, 1.0, 0.0]))

        self.assertIsNone(buffer.interpolate(1.5, max_age=0.1))

    def test_clear_removes_samples(self):
        buffer = ImuBuffer()
        buffer.push(1.0, np.array([0.0, 0.0, -1.0]))
        buffer.clear()
        self.assertIsNone(buffer.interpolate(1.0))

    def test_latest_timestamp_reports_newest_sample(self):
        buffer = ImuBuffer()
        self.assertIsNone(buffer.latest_timestamp())
        buffer.push(2.0, np.array([0.0, 0.0, -1.0]))
        buffer.push(1.0, np.array([0.0, 1.0, 0.0]))
        self.assertEqual(buffer.latest_timestamp(), 2.0)

    def test_quasi_static_gate_rejects_motion(self):
        self.assertTrue(
            imu_sample_is_quasi_static(
                np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.05])
            )
        )
        self.assertFalse(
            imu_sample_is_quasi_static(
                np.array([0.0, 0.0, 12.0]), np.array([0.0, 0.0, 0.0])
            )
        )
        self.assertFalse(
            imu_sample_is_quasi_static(
                np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.5])
            )
        )


class TimedVectorHistoryTest(unittest.TestCase):
    def test_apply_delay_prevents_future_command_assignment(self):
        history = TimedVectorHistory()
        history.push(1.0, np.array([1.0]))
        history.push(2.0, np.array([2.0]))

        before_application = history.value_at(2.01, apply_delay=0.02)
        after_application = history.value_at(2.02, apply_delay=0.02)

        self.assertIsNotNone(before_application)
        self.assertIsNotNone(after_application)
        self.assertTrue(np.allclose(before_application[0], np.array([1.0])))
        self.assertEqual(before_application[1], 1.02)
        self.assertTrue(np.allclose(after_application[0], np.array([2.0])))
        self.assertEqual(after_application[1], 2.02)

    def test_settle_gate_requires_history_covering_full_window(self):
        history = TimedVectorHistory()
        history.push(1.0, np.array([1.0]))

        self.assertIsNone(
            history.settled_value_at(1.4, dwell_time=0.5, tolerance=0.0)
        )
        self.assertIsNotNone(
            history.settled_value_at(1.5, dwell_time=0.5, tolerance=0.0)
        )

    def test_settle_gate_detects_slow_accumulated_ramp(self):
        history = TimedVectorHistory()
        history.push(0.0, np.array([0.0]))
        history.push(0.2, np.array([0.0006]))
        history.push(0.4, np.array([0.0012]))

        self.assertIsNone(
            history.settled_value_at(0.4, dwell_time=0.4, tolerance=0.001)
        )

    def test_settle_gate_accepts_constant_value_after_change(self):
        history = TimedVectorHistory()
        history.push(0.0, np.array([0.0, 0.0]))
        history.push(1.0, np.array([1.0, -1.0]))
        history.push(1.25, np.array([1.0, -1.0]))
        history.push(1.5, np.array([1.0, -1.0]))

        settled = history.settled_value_at(
            1.5,
            dwell_time=0.5,
            tolerance=1.0e-12,
        )

        self.assertIsNotNone(settled)
        self.assertTrue(np.allclose(settled[0], np.array([1.0, -1.0])))


if __name__ == "__main__":
    unittest.main()
