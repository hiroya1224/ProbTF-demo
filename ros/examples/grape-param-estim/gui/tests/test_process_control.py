import signal
from pathlib import Path
import tempfile
import unittest
from unittest import mock

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

    def test_diagonal_q_force_stop_uses_its_typed_cancellation_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "q"
            root.mkdir()
            (root / "manifest.json").write_text(
                '{"schema":"grape-param-estim/diagonal-wrench-q-estimate/v1"}',
                encoding="utf-8",
            )
            with mock.patch(
                "grape_param_estim.diagonal_q_artifact."
                "read_diagonal_q_manifest",
                return_value={"status": "writing"},
            ), mock.patch(
                "grape_param_estim.diagonal_q_artifact."
                "mark_diagonal_q_artifact_cancelled"
            ) as marker:
                self.assertTrue(finalize_cancelled_bundle(root, "forced"))
            marker.assert_called_once_with(root.resolve(), "forced")

    def test_parameter_force_stop_uses_its_typed_cancellation_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "parameters"
            root.mkdir()
            (root / "manifest.json").write_text(
                '{"schema":"grape-param-estim/fixed-q-augmented-parameter-estimate/v1"}',
                encoding="utf-8",
            )
            with mock.patch(
                "grape_param_estim.augmented_parameter_artifact."
                "read_augmented_parameter_manifest",
                return_value={"status": "writing"},
            ), mock.patch(
                "grape_param_estim.augmented_parameter_artifact."
                "mark_augmented_parameter_artifact_cancelled"
            ) as marker:
                self.assertTrue(finalize_cancelled_bundle(root, "forced"))
            marker.assert_called_once_with(root.resolve(), "forced")


if __name__ == "__main__":
    unittest.main()
