import unittest

import numpy as np

from grape_param_estim.episode import (
    EVENT_TIME_HEADER,
    EVENT_TIME_RECORD,
    EpisodeDataset,
    EpisodeMetadata,
    EventSeries,
    QualityEvent,
    clock_diagnostics,
    message_header_time,
    stable_hash,
)


class _Stamp:
    def __init__(self, seconds):
        self._seconds = seconds

    def to_sec(self):
        return self._seconds


class _Header:
    def __init__(self, seconds):
        self.stamp = _Stamp(seconds)


class _Message:
    def __init__(self, seconds):
        self.header = _Header(seconds)


class EpisodeTests(unittest.TestCase):
    def _series(self):
        return EventSeries(
            role="mocap",
            topic="/mocap",
            message_type="geometry_msgs/PoseStamped",
            event_times=np.array([1.0, 1.1, 1.2]),
            record_times=np.array([1.01, 1.11, 1.21]),
            event_time_sources=(
                EVENT_TIME_HEADER,
                EVENT_TIME_HEADER,
                EVENT_TIME_RECORD,
            ),
            values=("a", "b", "c"),
            valid_mask=np.array([True, False, True]),
            unit="m",
        )

    def test_nearest_and_causal_queries_make_age_and_validity_explicit(self):
        series = self._series()
        nearest = series.nearest(1.19, max_age=0.02)
        self.assertEqual(nearest.value, "c")
        self.assertAlmostEqual(nearest.age, 0.01)
        self.assertIsNone(series.nearest(1.11, max_age=0.02))
        invalid = series.nearest(1.11, max_age=0.02, require_valid=False)
        self.assertEqual(invalid.value, "b")
        self.assertFalse(invalid.valid)
        causal = series.nearest(1.19, max_age=0.10, causal=True)
        self.assertIsNone(causal)
        causal = series.nearest(
            1.19, max_age=0.10, causal=True, require_valid=False
        )
        self.assertEqual(causal.value, "b")
        self.assertGreaterEqual(causal.age, 0.0)

    def test_episode_hash_and_mode_segments_are_reproducible(self):
        state = EventSeries(
            role="flight_state",
            topic="/flight_state",
            message_type="std_msgs/UInt8",
            event_times=[2.0, 3.0, 4.0],
            record_times=[2.0, 3.0, 4.0],
            event_time_sources=(EVENT_TIME_RECORD,) * 3,
            values=(3, 5, 5),
            valid_mask=[True, True, True],
        )
        metadata = EpisodeMetadata(
            episode_id="bag-4",
            source_bag="/data/bag-4.bag",
            source_bag_sha256="a" * 64,
            start_time=1.0,
            end_time=5.0,
            labels=("attitude_failure",),
            split="test",
            assumptions={"frame": "world"},
            unknowns=("propeller_id",),
        )
        first = EpisodeDataset(
            metadata=metadata,
            series={"flight_state": state, "mocap": self._series()},
            quality_events=(
                QualityEvent(3.2, "gap", "/mocap", "dropout"),
            ),
            config_hash="b" * 64,
        )
        second = EpisodeDataset(
            metadata=metadata,
            series={"mocap": self._series(), "flight_state": state},
            quality_events=(
                QualityEvent(3.2, "gap", "/mocap", "dropout"),
            ),
            config_hash="b" * 64,
        )
        self.assertEqual(first.dataset_hash, second.dataset_hash)
        segments = first.mode_segments()
        self.assertEqual([(item.start, item.end, item.mode) for item in segments], [
            (2.0, 3.0, 3),
            (3.0, 5.0, 5),
        ])
        np.testing.assert_allclose(
            first.common_timeline(("mocap", "flight_state")),
            [1.0, 1.1, 1.2, 2.0, 3.0, 4.0],
        )

    def test_event_arrays_and_metadata_are_immutable(self):
        series = self._series()
        with self.assertRaises(ValueError):
            series.event_times[0] = 9.0
        with self.assertRaises(TypeError):
            series.metadata["new"] = True

    def test_header_time_and_stable_hash_validation(self):
        self.assertAlmostEqual(message_header_time(_Message(12.5)), 12.5)
        self.assertIsNone(message_header_time(_Message(0.0)))
        self.assertEqual(stable_hash({"b": 2, "a": 1}), stable_hash({"a": 1, "b": 2}))
        with self.assertRaisesRegex(ValueError, "finite and JSON-compatible"):
            stable_hash({"bad": float("nan")})

    def test_clock_offset_drift_and_jump_are_auditable(self):
        event_times = np.arange(6, dtype=float)
        # 100 ppm drift plus one explicit 30 ms step at sample four.
        offsets = 0.25 + 100.0e-6 * event_times
        offsets[4:] += 0.03
        diagnostic = clock_diagnostics(
            event_times,
            event_times + offsets,
            (EVENT_TIME_HEADER,) * 6,
            jump_threshold=0.02,
        )
        self.assertEqual(diagnostic.header_sample_count, 6)
        self.assertEqual(diagnostic.record_fallback_count, 0)
        self.assertEqual(diagnostic.offset_jump_count, 1)
        self.assertGreater(diagnostic.maximum_offset_step_s, 0.02)
        self.assertIsNotNone(diagnostic.offset_median_s)
        self.assertIsNotNone(diagnostic.offset_drift_ppm)

        series = EventSeries(
            role="imu",
            topic="/imu",
            message_type="sensor_msgs/Imu",
            event_times=event_times,
            record_times=event_times + offsets,
            event_time_sources=(EVENT_TIME_HEADER,) * 6,
            values=tuple(range(6)),
            valid_mask=np.ones(6, dtype=bool),
        )
        self.assertEqual(series.clock_diagnostics().offset_jump_count, 1)


if __name__ == "__main__":
    unittest.main()
