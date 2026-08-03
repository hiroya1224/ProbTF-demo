import io
from pathlib import Path
import tempfile
import unittest

from grape_param_estim.artifact_io import (
    IncompleteArtifactError,
    load_pid_proposal_evaluation,
    read_manifest,
)
from grape_param_estim.pid_proposal_cli import run_request
from grape_param_estim.progress import (
    CancellationToken,
    JsonlProgressWriter,
    ProgressCancelled,
    ProgressEvent,
)
try:
    from .test_pid_evaluation_input import (
        prepare_completed_run,
        write_pid_request,
    )
except ImportError:  # nosetests imports this directory as top-level modules.
    from test_pid_evaluation_input import (
        prepare_completed_run,
        write_pid_request,
    )


class PidProposalCliTests(unittest.TestCase):
    def test_small_completed_bundle_runs_with_jsonl_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = prepare_completed_run(root / "run")
            request = write_pid_request(root / "request.json", run)
            stream = io.StringIO()
            output = run_request(
                str(request),
                str(root / "evaluation"),
                progress_callback=JsonlProgressWriter(stream),
            )
            events = [
                ProgressEvent.from_json(line)
                for line in stream.getvalue().splitlines()
            ]
            self.assertTrue(events)
            self.assertEqual(events[0].stage_id, "request_validation")
            self.assertEqual(events[-1].stage_id, "complete")
            self.assertEqual(events[-1].fraction, 1.0)
            self.assertIn(
                "candidate_member_bag_forecast",
                {value.stage_id for value in events},
            )
            bundle = load_pid_proposal_evaluation(output)
            self.assertEqual(bundle.manifest["status"], "complete")
            self.assertEqual(
                tuple(bundle.summary["candidate_id"]),
                ("current", "member-pick", "user-exact"),
            )
            self.assertEqual(
                str(bundle.summary["current_pid_baseline_bag_id"][0]),
                "bag-b",
            )
            self.assertEqual(
                tuple(bundle.summary["candidate_source"]),
                ("current", "member-derived", "user"),
            )

    def test_cancellation_is_authoritative_over_partial_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = prepare_completed_run(root / "run")
            request = write_pid_request(root / "request.json", run)
            cancellation = CancellationToken()
            events = []

            def cancel_after_input(event):
                events.append(event)
                if event.stage_id == "evaluation_input_restoration":
                    cancellation.cancel("test_requested")

            output = root / "cancelled"
            with self.assertRaises(ProgressCancelled):
                run_request(
                    str(request),
                    str(output),
                    progress_callback=cancel_after_input,
                    cancellation_token=cancellation,
                )
            manifest = read_manifest(output)
            self.assertEqual(manifest["status"], "cancelled")
            self.assertEqual(
                manifest["cancellation_reason"], "test_requested"
            )
            with self.assertRaises(IncompleteArtifactError):
                load_pid_proposal_evaluation(output)
            self.assertTrue(
                all(isinstance(value, ProgressEvent) for value in events)
            )


if __name__ == "__main__":
    unittest.main()
