import signal
from pathlib import Path
import tempfile
import unittest

from grape_param_estim_gui.process_control import (
    finalize_cancelled_bundle,
    send_cooperative_interrupt,
)


class ProcessControlTest(unittest.TestCase):
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
                4321,
                platform_name="nt",
                signal_sender=unexpected,
            )
        )
        self.assertFalse(
            send_cooperative_interrupt(
                0,
                platform_name="posix",
                signal_sender=unexpected,
            )
        )

    def test_signal_delivery_error_is_not_hidden(self):
        def failed(_process_id, _signum):
            raise ProcessLookupError("worker is gone")

        with self.assertRaises(ProcessLookupError):
            send_cooperative_interrupt(
                4321,
                platform_name="posix",
                signal_sender=failed,
            )

    def test_force_stopped_writing_bundle_is_marked_cancelled(self):
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
                    manifest_reader=lambda _root: {
                        "status": "cancelled",
                        "cancellation_reason": "signal_SIGINT",
                    },
                    cancellation_marker=lambda _target, _reason: self.fail(
                        "an existing cancellation must not be overwritten"
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
