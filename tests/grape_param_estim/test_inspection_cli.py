from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from grape_param_estim.artifact_io import load_inspection_bundle
from grape_param_estim.inspection import INSPECTION_REQUEST_SCHEMA
from grape_param_estim.inspection_cli import main, run_request
from grape_param_estim.progress import (
    JsonlProgressWriter,
    ProgressCancelled,
    ProgressEvent,
)
try:
    from .test_flight_inspection import _fake_arrays
except ImportError:  # nosetests imports this directory as top-level modules.
    from test_flight_inspection import _fake_arrays


class InspectionCliTests(unittest.TestCase):
    def test_run_request_emits_typed_progress_and_completes_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag_path = root / "flight.bag"
            bag_path.write_bytes(b"inspection CLI fixture")
            request_path = root / "request.json"
            request_path.write_text(
                json.dumps(
                    {
                        "schema": INSPECTION_REQUEST_SCHEMA,
                        "request_id": "cli-inspection",
                        "bags": [
                            {
                                "bag_id": "flight",
                                "path": str(bag_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            stream = io.StringIO()
            output = run_request(
                str(request_path),
                str(root / "output"),
                progress_callback=JsonlProgressWriter(stream),
                arrays_loader=lambda path: _fake_arrays(path),
            )
            events = [
                ProgressEvent.from_json(line)
                for line in stream.getvalue().splitlines()
            ]
            self.assertTrue(events)
            self.assertEqual(events[0].stage_id, "request_validation")
            self.assertEqual(events[-1].stage_id, "complete")
            self.assertEqual(events[-1].fraction, 1.0)
            self.assertIn("sha256", {value.stage_id for value in events})
            self.assertEqual(
                load_inspection_bundle(output).manifest["status"],
                "complete",
            )

    def test_main_reserves_stdout_for_progress_jsonl(self):
        def fake_run(_request, output, progress_callback, **_kwargs):
            progress_callback(
                ProgressEvent(
                    run_id="cli-inspection",
                    stage_id="complete",
                    stage_label="Inspection bundle complete",
                    completed_units=1,
                    total_units=1,
                    fraction=1.0,
                    elapsed_seconds=0.1,
                    eta_seconds=0.0,
                )
            )
            return Path(output)

        standard_output = io.StringIO()
        standard_error = io.StringIO()
        with patch(
            "grape_param_estim.inspection_cli.run_request",
            side_effect=fake_run,
        ), redirect_stdout(standard_output), redirect_stderr(standard_error):
            exit_code = main(
                ("--request", "/tmp/request.json", "--output", "/tmp/out")
            )
        self.assertEqual(exit_code, 0)
        lines = standard_output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        ProgressEvent.from_json(lines[0])
        self.assertIn("inspection bundle complete", standard_error.getvalue())

    def test_main_reports_cancellation_on_stderr_only(self):
        standard_output = io.StringIO()
        standard_error = io.StringIO()
        with patch(
            "grape_param_estim.inspection_cli.run_request",
            side_effect=ProgressCancelled("signal_SIGTERM"),
        ), redirect_stdout(standard_output), redirect_stderr(standard_error):
            exit_code = main(
                ("--request", "/tmp/request.json", "--output", "/tmp/out")
            )
        self.assertEqual(exit_code, 2)
        self.assertEqual(standard_output.getvalue(), "")
        self.assertIn("signal_SIGTERM", standard_error.getvalue())


if __name__ == "__main__":
    unittest.main()
