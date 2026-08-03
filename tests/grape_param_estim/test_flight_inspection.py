from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import numpy as np

from grape_param_estim.artifact_io import (
    IncompleteArtifactError,
    load_inspection_bundle,
    read_manifest,
)
from grape_param_estim.inspection import (
    CONFIGURATION_FINGERPRINT_FIELDS,
    FLIGHT_INSPECTION_SCHEMA,
    INSPECTION_BUNDLE_SCHEMA,
    INSPECTION_PREVIEW_SCHEMA,
    INSPECTION_REQUEST_SCHEMA,
    InspectionBagRequest,
    InspectionRequest,
    inspect_flight_arrays,
    inspect_flights,
    load_inspection_request,
)
from grape_param_estim.progress import (
    CancellationToken,
    ProgressCancelled,
    ProgressEvent,
)
from grape_param_estim.real_rosbag import (
    ControllerGainEvents,
    FlightStateSeries,
    PidReferenceSeries,
    RosbagArrayData,
    TOPIC_TYPE_CONTRACT,
    TimedVectorSeries,
    _select_controller_snapshot,
    build_real_flight_episode,
    list_flight_episode_candidates,
    recommend_smoothing_interval,
    select_continuous_flight_window,
)


def _target_like_flight_states():
    return FlightStateSeries(
        np.asarray((100.0, 101.0, 102.0, 103.0, 110.0, 112.0, 113.0, 114.0)),
        np.asarray((0, 1, 2, 3, 4, 17, 6, 0)),
    )


def _fake_arrays(path="/tmp/fake-inspection.bag", sha256="a" * 64):
    sample_times = np.arange(100.0, 120.01, 0.1)
    count = sample_times.size
    position = np.column_stack(
        (
            0.02 * (sample_times - 100.0),
            np.zeros(count),
            np.full(count, 0.8),
        )
    )
    orientation = np.zeros((count, 4))
    orientation[:, 3] = 1.0
    target_position = position.copy()
    target_velocity = np.zeros((count, 3))
    target_rpy = np.zeros((count, 3))
    target_omega = np.zeros((count, 3))
    pid_term = np.zeros((count, 6))
    gain_times = np.asarray((100.01, 100.02, 100.03, 100.04))
    gains = np.asarray(
        (
            (3.0, 0.1, 1.0),
            (5.0, 1.0, 2.5),
            (20.0, 1.0, 8.0),
            (4.0, 1.0, 2.0),
        )
    )
    topic_names = tuple(value[0] for value in TOPIC_TYPE_CONTRACT)
    topic_types = tuple(value[1] for value in TOPIC_TYPE_CONTRACT)
    return RosbagArrayData(
        bag_path=path,
        bag_sha256=sha256,
        bag_size_bytes=4096,
        bag_record_start=100.0,
        bag_record_end=120.0,
        topic_names=topic_names,
        topic_types=topic_types,
        cog_position=TimedVectorSeries(sample_times, position),
        baselink_orientation=TimedVectorSeries(sample_times, orientation),
        pid=PidReferenceSeries(
            sample_times,
            target_position,
            target_velocity,
            target_rpy,
            target_omega,
            pid_term,
            pid_term,
            pid_term,
            pid_term,
        ),
        flight_state=_target_like_flight_states(),
        controller_gain_events=ControllerGainEvents(
            gain_times,
            ("xy", "z", "roll_pitch", "yaw"),
            gains,
            np.zeros(4, dtype=bool),
        ),
        joint_position=TimedVectorSeries(
            sample_times, np.zeros((count, 4))
        ),
        joint_names=("gimbal1", "gimbal2", "gimbal3", "gimbal4"),
        commanded_thrust=TimedVectorSeries(
            sample_times, np.full((count, 4), 5.0)
        ),
    )


class FlightIntervalCandidateTests(unittest.TestCase):
    def test_force_landing_is_inside_complete_episode_and_hover_falls_back(self):
        candidates = list_flight_episode_candidates(
            _target_like_flight_states(), 100.0
        )
        self.assertEqual(len(candidates), 1)
        episode = candidates[0]
        self.assertEqual(
            tuple(value.state for value in episode.state_intervals),
            (3, 4, 17),
        )
        recommendation = recommend_smoothing_interval(episode)
        self.assertEqual(recommendation.interval.state, 3)
        self.assertEqual(
            recommendation.reason, "longest_control_active_fallback"
        )
        self.assertIn("no state=5", recommendation.warnings[0])
        with self.assertRaises(FrozenInstanceError):
            episode.episode_index = 9

    def test_preferred_state_uses_longest_contiguous_candidate(self):
        series = FlightStateSeries(
            np.asarray((100.0, 101.0, 103.0, 104.0, 107.0, 109.0, 110.0)),
            np.asarray((0, 3, 5, 4, 5, 4, 6)),
        )
        episode = list_flight_episode_candidates(series, 100.0)[0]
        recommendation = recommend_smoothing_interval(episode)
        self.assertEqual(recommendation.interval.state, 5)
        self.assertEqual(recommendation.interval.start_record_time, 107.0)
        self.assertEqual(recommendation.interval.end_record_time, 109.0)
        self.assertIn("multiple state=5", recommendation.warnings[0])

    def test_manual_window_cannot_concatenate_state_intervals(self):
        series = _target_like_flight_states()
        automatic = select_continuous_flight_window(series, 100.0)
        self.assertEqual(automatic[:2], (103.0, 110.0))
        manual = select_continuous_flight_window(
            series, 100.0, start_local=4.0, end_local=9.0
        )
        self.assertEqual(manual[:2], (104.0, 109.0))
        with self.assertRaisesRegex(ValueError, "cannot be concatenated"):
            select_continuous_flight_window(
                series, 100.0, start_local=4.0, end_local=11.0
            )
        complete_manual = select_continuous_flight_window(
            series,
            100.0,
            start_local=4.0,
            end_local=12.5,
            window_state=None,
        )
        self.assertEqual(complete_manual[:2], (104.0, 112.5))

    def test_topic_gap_inside_recommended_interval_is_rejected(self):
        arrays = _fake_arrays()
        times = arrays.cog_position.record_times
        keep = ~((times > 104.0) & (times < 104.5))
        with_gap = replace(
            arrays,
            cog_position=TimedVectorSeries(
                times[keep], arrays.cog_position.values[keep]
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            "CoG odometry contains a gap inside the flight window",
        ):
            build_real_flight_episode(with_gap, sample_period=0.04)

    def test_inspection_blocks_bag_with_no_valid_flight_interval(self):
        arrays = replace(
            _fake_arrays(),
            flight_state=FlightStateSeries(
                np.asarray((100.0, 101.0, 102.0, 110.0, 114.0)),
                np.asarray((0, 1, 3, 17, 0)),
            ),
        )
        result = inspect_flight_arrays(
            InspectionBagRequest(
                bag_id="no_valid_interval",
                path=arrays.bag_path,
            ),
            arrays,
        )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.episodes, ())
        self.assertIsNone(result.recommendation)
        self.assertIsNone(result.controller_snapshot)
        self.assertIn(
            "no complete TAKEOFF-to-STOP flight episode found",
            result.warnings,
        )
        self.assertEqual(
            result.estimated_work_units["member_bag_forecast_units"], 0
        )


class ControllerSnapshotTests(unittest.TestCase):
    def test_recorded_startup_is_exact_and_later_gate_controls_updates(self):
        arrays = _fake_arrays()
        events = arrays.controller_gain_events
        extended = ControllerGainEvents(
            np.append(events.record_times, (101.0, 102.0)),
            events.groups + ("xy", "xy"),
            np.vstack((events.gains, (99.0, 9.0, 9.0), (7.0, 0.2, 3.0))),
            np.append(events.pid_control_flags, (False, True)),
        )
        snapshot = _select_controller_snapshot(extended, 103.0, 110.0)
        np.testing.assert_array_equal(snapshot.gains[0], (7.0, 0.2, 3.0))
        self.assertEqual(
            snapshot.source_kinds[0], "dynamic_reconfigure_applied"
        )
        np.testing.assert_array_equal(snapshot.gains[1], (5.0, 1.0, 2.5))
        self.assertEqual(
            snapshot.source_kinds[1],
            "recorded_startup_parameter_update",
        )

    def test_applied_gain_change_inside_window_is_rejected(self):
        events = _fake_arrays().controller_gain_events
        changed_inside = ControllerGainEvents(
            np.append(events.record_times, 105.0),
            events.groups + ("yaw",),
            np.vstack((events.gains, (8.0, 0.4, 3.0))),
            np.append(events.pid_control_flags, True),
        )
        with self.assertRaisesRegex(ValueError, "change inside"):
            _select_controller_snapshot(changed_inside, 103.0, 110.0)


class InspectionBundleTests(unittest.TestCase):
    def test_request_loader_and_pickle_free_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag_path = root / "failed-flight.bag"
            bag_bytes = b"inspection fixture only"
            bag_path.write_bytes(bag_bytes)
            bag_sha256 = hashlib.sha256(bag_bytes).hexdigest()
            request_path = root / "request.json"
            request_path.write_text(
                json.dumps(
                    {
                        "schema": INSPECTION_REQUEST_SCHEMA,
                        "request_id": "inspect_failed_flight",
                        "preview_max_samples": 24,
                        "bags": [
                            {
                                "bag_id": "failed_flight",
                                "path": str(bag_path),
                                "configuration_provenance": {
                                    "payload": "unknown-from-bag"
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            request = load_inspection_request(str(request_path))
            self.assertIsInstance(request, InspectionRequest)
            arrays = _fake_arrays(str(bag_path), bag_sha256)
            output = inspect_flights(
                request,
                str(root / "inspection"),
                arrays_loader=lambda _path: arrays,
            )
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema"], INSPECTION_BUNDLE_SCHEMA)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["bag_ids"], ["failed_flight"])
            self.assertEqual(
                manifest["artifacts"]["bags"]["failed_flight"],
                {
                    "inspection": (
                        "bags/failed_flight.inspection.json"
                    ),
                    "preview": "bags/failed_flight.preview.npz",
                },
            )
            inspection = json.loads(
                (output / "bags/failed_flight.inspection.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(inspection["schema"], FLIGHT_INSPECTION_SCHEMA)
            self.assertEqual(
                inspection["status"],
                "needs_configuration_confirmation",
            )
            self.assertEqual(inspection["bag_sha256"], bag_sha256)
            self.assertEqual(
                inspection["recommended_interval"]["interval"]["state"], 3
            )
            np.testing.assert_array_equal(
                inspection["controller_snapshot"]["gains"][0],
                (3.0, 0.1, 1.0),
            )
            fingerprint = inspection["configuration_fingerprint"]
            self.assertFalse(fingerprint["complete"])
            self.assertEqual(
                set(fingerprint["missing_components"]),
                set(CONFIGURATION_FINGERPRINT_FIELDS) - {"payload"},
            )
            self.assertGreater(
                inspection["estimated_work_units"]["integration_step_units"],
                0,
            )
            with np.load(
                str(output / "bags/failed_flight.preview.npz"),
                allow_pickle=False,
            ) as preview:
                self.assertEqual(
                    str(preview["schema"][0]), INSPECTION_PREVIEW_SCHEMA
                )
                self.assertLessEqual(preview["time"].size, 24)
                for key in preview.files:
                    self.assertFalse(preview[key].dtype.hasobject)
            bundle = load_inspection_bundle(output)
            self.assertEqual(
                bundle.inspections["failed_flight"]["bag_path"],
                str(bag_path),
            )
            self.assertEqual(
                bundle.previews["failed_flight"]["time"].shape,
                bundle.previews["failed_flight"]["flight_state"].shape,
            )

    def test_progress_is_jsonl_compatible_and_cancel_is_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag_path = root / "cancel-flight.bag"
            bag_path.write_bytes(b"inspection cancellation fixture")
            request = InspectionRequest(
                request_id="cancel-inspection",
                bags=(
                    InspectionBagRequest(
                        bag_id="cancel_flight",
                        path=str(bag_path),
                    ),
                ),
            )
            arrays = _fake_arrays(str(bag_path), "b" * 64)
            cancellation = CancellationToken()
            events = []

            def cancel_after_read(_path):
                cancellation.cancel("test_requested")
                return arrays

            with self.assertRaises(ProgressCancelled):
                inspect_flights(
                    request,
                    str(root / "cancelled"),
                    arrays_loader=cancel_after_read,
                    progress_callback=events.append,
                    cancellation_token=cancellation,
                )
            self.assertTrue(events)
            self.assertTrue(all(isinstance(value, ProgressEvent)
                                for value in events))
            manifest = read_manifest(root / "cancelled")
            self.assertEqual(manifest["status"], "cancelled")
            self.assertEqual(
                manifest["cancellation_reason"], "test_requested"
            )
            with self.assertRaises(IncompleteArtifactError):
                load_inspection_bundle(root / "cancelled")

    def test_unknown_request_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(
                json.dumps({"schema": "unknown", "bags": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported"):
                load_inspection_request(str(path))


if __name__ == "__main__":
    unittest.main()
