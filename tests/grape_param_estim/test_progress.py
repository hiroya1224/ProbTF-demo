import io
import json
import unittest

from grape_param_estim.progress import (
    CancellationToken,
    JsonlProgressWriter,
    PROGRESS_EVENT_SCHEMA,
    ProgressCancelled,
    ProgressEvent,
    ProgressTracker,
    ProgressValidationError,
    STAGE_COMPUTING_LOCAL_POSTERIOR_GEOMETRY,
    STAGE_LABELS,
    STAGE_OPTIMIZING_FULL_TRAJECTORY,
    STAGE_PREPARING_TRAJECTORY,
    STAGE_REFINING_CONSTANT_DELAY,
    STAGE_SAMPLING_PARAMETER_POSTERIOR,
    STAGE_UPDATING_MODEL_ERROR_COVARIANCE,
    STAGE_WRITING_ARTIFACTS,
)


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


def _event(**updates):
    values = {
        "run_id": "run-a",
        "stage_id": STAGE_OPTIMIZING_FULL_TRAJECTORY,
        "stage_label": "Optimizing full trajectory",
        "stage_completed_units": 1,
        "stage_total_units": 2,
        "stage_fraction": 0.5,
        "completed_units": 3,
        "total_units": 6,
        "fraction": 0.5,
        "stage_elapsed_seconds": 1.0,
        "stage_eta_seconds": 1.0,
        "elapsed_seconds": 3.0,
        "eta_seconds": 3.0,
        "sample_id": "chain-01/draw-0000042",
    }
    values.update(updates)
    return ProgressEvent(**values)


class ProgressTests(unittest.TestCase):
    def test_stage_ids_have_exact_plan_labels(self):
        self.assertEqual(
            dict(STAGE_LABELS),
            {
                STAGE_PREPARING_TRAJECTORY: "Preparing trajectory",
                STAGE_OPTIMIZING_FULL_TRAJECTORY: (
                    "Optimizing full trajectory"
                ),
                STAGE_REFINING_CONSTANT_DELAY: "Refining constant delay",
                STAGE_UPDATING_MODEL_ERROR_COVARIANCE: (
                    "Updating model-error covariance"
                ),
                STAGE_COMPUTING_LOCAL_POSTERIOR_GEOMETRY: (
                    "Computing local posterior geometry"
                ),
                STAGE_SAMPLING_PARAMETER_POSTERIOR: (
                    "Sampling parameter posterior"
                ),
                STAGE_WRITING_ARTIFACTS: "Writing artifacts",
            },
        )

    def test_jsonl_has_stage_local_and_monotonic_overall_progress(self):
        clock = FakeClock()
        stream = io.StringIO()
        tracker = ProgressTracker(
            "run-a",
            overall_total_units=6,
            callback=JsonlProgressWriter(stream),
            clock=clock,
            eta_calibration_units=2,
            measurement_window=4,
        )

        preparing = tracker.begin_stage(STAGE_PREPARING_TRAJECTORY, 2)
        first = preparing.emit(0, bag_id="bag-a")
        self.assertIsNone(first.eta_seconds)
        self.assertIsNone(first.stage_eta_seconds)
        clock.advance(2.0)
        prepared = preparing.complete(bag_id="bag-a")
        self.assertEqual(prepared.stage_fraction, 1.0)
        self.assertAlmostEqual(prepared.eta_seconds, 4.0)

        optimizing = tracker.begin_stage(
            STAGE_OPTIMIZING_FULL_TRAJECTORY, 3
        )
        restarted = optimizing.emit(
            0,
            iteration=1,
            maximum_iterations=3,
            sample_id="map",
        )
        self.assertEqual(restarted.stage_fraction, 0.0)
        self.assertEqual(restarted.completed_units, 2)
        clock.advance(3.0)
        optimized = optimizing.complete(
            iteration=3,
            maximum_iterations=3,
            sample_id="map",
        )
        self.assertAlmostEqual(optimized.eta_seconds, 1.0)

        writing = tracker.begin_stage(STAGE_WRITING_ARTIFACTS, 1)
        clock.advance(0.5)
        final = writing.complete()
        self.assertEqual(final.fraction, 1.0)
        self.assertEqual(final.eta_seconds, 0.0)
        self.assertEqual(final.stage_eta_seconds, 0.0)

        events = [
            ProgressEvent.from_json(line)
            for line in stream.getvalue().splitlines()
        ]
        self.assertEqual(
            [event.fraction for event in events],
            [0.0, 2.0 / 6.0, 2.0 / 6.0, 5.0 / 6.0, 1.0],
        )
        self.assertEqual(
            [event.stage_fraction for event in events],
            [0.0, 1.0, 0.0, 1.0, 1.0],
        )
        self.assertEqual(
            json.loads(stream.getvalue().splitlines()[0])["schema"],
            PROGRESS_EVENT_SCHEMA,
        )

    def test_initial_eta_applies_to_stage_and_overall_until_measured(self):
        clock = FakeClock()
        tracker = ProgressTracker(
            "run-a",
            overall_total_units=10,
            clock=clock,
            eta_calibration_units=3,
            initial_seconds_per_unit=4.0,
        )
        stage = tracker.begin_stage(STAGE_REFINING_CONSTANT_DELAY, 4)
        initial = stage.emit(0)
        self.assertEqual(initial.eta_seconds, 40.0)
        self.assertEqual(initial.stage_eta_seconds, 16.0)
        clock.advance(3.0)
        measured = stage.emit(3)
        self.assertAlmostEqual(measured.eta_seconds, 7.0)
        self.assertAlmostEqual(measured.stage_eta_seconds, 1.0)

    def test_stage_boundaries_and_counters_are_enforced(self):
        clock = FakeClock()
        tracker = ProgressTracker("run-a", 5, clock=clock)
        first = tracker.begin_stage(STAGE_PREPARING_TRAJECTORY, 2)
        first.emit(1)
        with self.assertRaisesRegex(
            ProgressValidationError, "active stage must be complete"
        ):
            tracker.begin_stage(STAGE_REFINING_CONSTANT_DELAY, 1)
        with self.assertRaisesRegex(
            ProgressValidationError, "must be monotonic"
        ):
            first.emit(0)
        first.complete()
        second = tracker.begin_stage(STAGE_REFINING_CONSTANT_DELAY, 3)
        with self.assertRaisesRegex(
            ProgressValidationError, "inactive progress stage"
        ):
            first.complete()
        second.complete()
        with self.assertRaisesRegex(
            ProgressValidationError, "exceeds overall_total_units"
        ):
            tracker.begin_stage(STAGE_WRITING_ARTIFACTS, 1)

    def test_cancellation_is_first_reason_wins_and_checked_at_boundary(self):
        token = CancellationToken()
        tracker = ProgressTracker(
            "run-a", 5, cancellation_token=token, clock=FakeClock()
        )
        stage = tracker.begin_stage(STAGE_OPTIMIZING_FULL_TRAJECTORY, 5)
        self.assertTrue(tracker.cancel("user_requested"))
        self.assertFalse(tracker.cancel("timeout"))
        self.assertEqual(token.reason, "user_requested")
        with self.assertRaises(ProgressCancelled) as context:
            stage.emit(1)
        self.assertEqual(context.exception.reason, "user_requested")
        self.assertIsNone(tracker.last_event)

    def test_event_parser_rejects_legacy_member_key_and_schema_drift(self):
        event = _event()
        legacy = event.to_dict()
        legacy["member_id"] = 7
        legacy.pop("sample_id")
        with self.assertRaisesRegex(
            ProgressValidationError, "missing=.*sample_id.*extra=.*member_id"
        ):
            ProgressEvent.from_dict(legacy)

        extra = event.to_dict()
        extra["unexpected"] = 1
        with self.assertRaisesRegex(ProgressValidationError, "extra"):
            ProgressEvent.from_dict(extra)
        wrong_schema = event.to_dict()
        wrong_schema["schema"] = "grape-param-estim/progress-event/v1"
        with self.assertRaisesRegex(
            ProgressValidationError, "unsupported progress schema"
        ):
            ProgressEvent.from_dict(wrong_schema)

    def test_event_validation_rejects_invalid_label_fraction_and_sample(self):
        with self.assertRaisesRegex(
            ProgressValidationError, "stage_label"
        ):
            _event(stage_label="Arbitrary worker text")
        with self.assertRaisesRegex(
            ProgressValidationError, "stage_fraction must equal"
        ):
            _event(stage_fraction=0.6)
        with self.assertRaisesRegex(
            ProgressValidationError, "fraction must equal"
        ):
            _event(fraction=0.6)
        with self.assertRaisesRegex(ProgressValidationError, "sample_id"):
            _event(sample_id=7)

    def test_nonfinite_json_is_rejected(self):
        mapping = _event().to_dict()
        line = json.dumps(mapping).replace("3.0", "NaN", 1)
        with self.assertRaises(ProgressValidationError):
            ProgressEvent.from_json(line)

    def test_callback_is_synchronous_and_exceptions_propagate(self):
        received = []

        def callback(event):
            received.append(event)
            raise RuntimeError("writer failed")

        tracker = ProgressTracker(
            "run-a", 2, callback=callback, clock=FakeClock()
        )
        stage = tracker.begin_stage(STAGE_PREPARING_TRAJECTORY, 2)
        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            stage.emit(0)
        self.assertEqual(len(received), 1)
        self.assertIs(received[0], tracker.last_event)

    def test_jsonl_writer_accepts_events_only(self):
        writer = JsonlProgressWriter(io.StringIO())
        with self.assertRaises(TypeError):
            writer({"fraction": 0.5})


if __name__ == "__main__":
    unittest.main()
