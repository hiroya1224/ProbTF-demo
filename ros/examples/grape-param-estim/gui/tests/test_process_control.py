import json
from pathlib import Path
import signal
import tempfile
import unittest

from grape_param_estim_gui.process_control import (
    finalize_cancelled_bundle,
    send_cooperative_interrupt,
)


BATCH_ESTIMATION_CHECKPOINT_SCHEMA = (
    "grape-param-estim/batch-estimation-checkpoint/v1"
)


def batch_checkpoint_path(output: Path) -> Path:
    return output.parent / ".{}-batch-checkpoint".format(output.name)


class ProcessControlTests(unittest.TestCase):
    def test_posix_worker_receives_sigint_immediately(self):
        calls = []
        self.assertTrue(
            send_cooperative_interrupt(
                4321,
                platform_name="posix",
                signal_sender=lambda process_id, signum: calls.append(
                    (process_id, signum)
                ),
            )
        )
        self.assertEqual(calls, [(4321, signal.SIGINT)])

    def test_non_posix_or_missing_process_uses_qprocess_fallback(self):
        def unexpected(_process_id, _signum):
            raise AssertionError("a signal must not be sent")

        self.assertFalse(
            send_cooperative_interrupt(
                4321, platform_name="nt", signal_sender=unexpected
            )
        )
        self.assertFalse(
            send_cooperative_interrupt(
                0, platform_name="posix", signal_sender=unexpected
            )
        )

    def test_signal_delivery_error_is_not_hidden(self):
        def failed(_process_id, _signum):
            raise ProcessLookupError("worker is gone")

        with self.assertRaises(ProcessLookupError):
            send_cooperative_interrupt(
                4321, platform_name="posix", signal_sender=failed
            )

    def test_generic_writing_bundle_is_marked_cancelled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            root.mkdir()
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            calls = []
            self.assertTrue(
                finalize_cancelled_bundle(
                    root,
                    "user_requested",
                    manifest_reader=lambda _root: {"status": "writing"},
                    cancellation_marker=lambda target, reason: calls.append(
                        (Path(target), reason)
                    ),
                )
            )
            self.assertEqual(calls, [(root, "user_requested")])

    def test_existing_cancelled_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            root.mkdir()
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            self.assertTrue(
                finalize_cancelled_bundle(
                    root,
                    "different_reason",
                    manifest_reader=lambda _root: {"status": "cancelled"},
                    cancellation_marker=lambda _target, _reason: self.fail(
                        "an existing cancellation must not be overwritten"
                    ),
                )
            )

    def test_batch_checkpoint_is_marked_cancelled_without_a_partial_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "estimation_run"
            checkpoint = batch_checkpoint_path(output)
            checkpoint.mkdir()
            manifest_path = checkpoint / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": BATCH_ESTIMATION_CHECKPOINT_SCHEMA,
                        "status": "sampling",
                        "cancellation_reason": "",
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(finalize_cancelled_bundle(output, "user requested"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "cancelled")
            self.assertEqual(manifest["cancellation_reason"], "user requested")
            self.assertTrue(finalize_cancelled_bundle(output, "user requested"))

    def test_missing_output_and_checkpoint_are_not_finalised(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "estimation_run"
            self.assertFalse(finalize_cancelled_bundle(output, "cancelled"))


if __name__ == "__main__":
    unittest.main()
