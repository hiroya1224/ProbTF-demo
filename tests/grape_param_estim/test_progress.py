import io
import json
import unittest

from grape_param_estim.progress import (
    CancellationToken,
    JsonlProgressWriter,
    ProgressCancelled,
    ProgressEvent,
    ProgressTracker,
    ProgressValidationError,
)


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


class ProgressTests(unittest.TestCase):
    def test_jsonl_fraction_is_monotonic_and_eta_becomes_measured(self):
        clock = FakeClock()
        stream = io.StringIO()
        tracker = ProgressTracker(
            "run-a",
            total_units=10,
            callback=JsonlProgressWriter(stream),
            clock=clock,
            eta_calibration_units=2,
            measurement_window=4,
        )
        first = tracker.emit(0, "prepare", "Preparing")
        self.assertIsNone(first.eta_seconds)
        clock.advance(2.0)
        second = tracker.emit(
            2,
            "ensemble_forecast",
            "Iteration 1/2",
            iteration=1,
            maximum_iterations=2,
            bag_id="bag-a",
            member_id=7,
            message="member 2/10",
        )
        self.assertAlmostEqual(second.eta_seconds, 8.0)
        clock.advance(3.0)
        final = tracker.emit(10, "artifact_writing", "Writing artifacts")
        self.assertEqual(final.eta_seconds, 0.0)

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        events = [ProgressEvent.from_json(line) for line in lines]
        fractions = [event.fraction for event in events]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(fractions, [0.0, 0.2, 1.0])
        self.assertEqual(json.loads(lines[1])["schema"], events[1].schema)

    def test_initial_eta_is_used_only_until_measured_rate_is_available(self):
        clock = FakeClock()
        tracker = ProgressTracker(
            "run-a",
            total_units=10,
            clock=clock,
            eta_calibration_units=3,
            initial_seconds_per_unit=4.0,
        )
        self.assertEqual(
            tracker.emit(0, "validation", "Validating").eta_seconds, 40.0
        )
        clock.advance(3.0)
        measured = tracker.emit(3, "forecast", "Forecasting")
        self.assertAlmostEqual(measured.eta_seconds, 7.0)

    def test_decreasing_work_or_backwards_clock_is_rejected(self):
        clock = FakeClock()
        tracker = ProgressTracker("run-a", 5, clock=clock)
        tracker.emit(2, "forecast", "Forecasting")
        with self.assertRaisesRegex(
            ProgressValidationError, "must be monotonic"
        ):
            tracker.emit(1, "forecast", "Forecasting")
        clock.value = 99.0
        with self.assertRaisesRegex(
            ProgressValidationError, "clock moved backwards"
        ):
            tracker.emit(2, "forecast", "Forecasting")

    def test_cancellation_is_first_reason_wins_and_checked_at_boundary(self):
        token = CancellationToken()
        tracker = ProgressTracker(
            "run-a", 5, cancellation_token=token, clock=FakeClock()
        )
        self.assertTrue(tracker.cancel("user_requested"))
        self.assertFalse(tracker.cancel("timeout"))
        self.assertEqual(token.reason, "user_requested")
        with self.assertRaises(ProgressCancelled) as context:
            tracker.emit(1, "forecast", "Forecasting")
        self.assertEqual(context.exception.reason, "user_requested")
        self.assertIsNone(tracker.last_event)

    def test_event_parser_rejects_schema_drift_and_nonfinite_values(self):
        event = ProgressEvent(
            run_id="run-a",
            stage_id="forecast",
            stage_label="Forecasting",
            completed_units=1,
            total_units=2,
            fraction=0.5,
            elapsed_seconds=1.0,
            eta_seconds=1.0,
        )
        mapping = event.to_dict()
        mapping["unexpected"] = 1
        with self.assertRaisesRegex(ProgressValidationError, "extra"):
            ProgressEvent.from_dict(mapping)
        with self.assertRaises(ProgressValidationError):
            ProgressEvent.from_json(event.to_json().replace("1.0", "NaN", 1))
        with self.assertRaisesRegex(
            ProgressValidationError, "fraction must equal"
        ):
            ProgressEvent(
                run_id="run-a",
                stage_id="forecast",
                stage_label="Forecasting",
                completed_units=1,
                total_units=2,
                fraction=0.6,
                elapsed_seconds=1.0,
                eta_seconds=1.0,
            )

    def test_callback_is_synchronous_and_exceptions_propagate(self):
        received = []

        def callback(event):
            received.append(event)
            raise RuntimeError("writer failed")

        tracker = ProgressTracker(
            "run-a", 2, callback=callback, clock=FakeClock()
        )
        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            tracker.emit(0, "prepare", "Preparing")
        self.assertEqual(len(received), 1)
        self.assertIs(received[0], tracker.last_event)

    def test_jsonl_writer_accepts_events_only(self):
        writer = JsonlProgressWriter(io.StringIO())
        with self.assertRaises(TypeError):
            writer({"fraction": 0.5})


if __name__ == "__main__":
    unittest.main()
