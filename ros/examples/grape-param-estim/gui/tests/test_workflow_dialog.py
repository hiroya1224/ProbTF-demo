import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication, QDialog
except ImportError as error:
    raise unittest.SkipTest("PySide6 is unavailable: {}".format(error))

from grape_param_estim_gui.widgets.workflow_dialog import (
    WorkflowLaunchDialog,
    WorkflowLaunchSelection,
)
from grape_param_estim_gui.workflow import StageStatus, WorkflowMode


class WorkflowLaunchDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def _dialog(
        self,
        diagonal_q: StageStatus = StageStatus.READY,
        static_parameters: StageStatus = StageStatus.BLOCKED,
        **kwargs,
    ) -> WorkflowLaunchDialog:
        dialog = WorkflowLaunchDialog(
            {
                "diagonal_q": diagonal_q,
                "static_parameters": static_parameters,
            },
            **kwargs,
        )
        self.addCleanup(dialog.close)
        return dialog

    def test_default_is_recommended_staged_mode_and_start_returns_typed_value(self):
        dialog = self._dialog()
        emitted = []
        dialog.launchRequested.connect(emitted.append)

        self.assertTrue(dialog.isModal())
        self.assertEqual(dialog.selected_mode, WorkflowMode.STEP)
        self.assertTrue(dialog.staged_mode_radio.isChecked())
        self.assertIn("recommended", dialog.staged_mode_radio.text())

        dialog.all_mode_radio.click()
        dialog.start_button.click()

        expected = WorkflowLaunchSelection(mode=WorkflowMode.ALL)
        self.assertEqual(dialog.result(), QDialog.Accepted)
        self.assertEqual(dialog.launch_selection, expected)
        self.assertEqual(emitted, [expected])

    def test_all_stage_statuses_and_artifact_reuse_copy_are_displayed(self):
        dialog = self._dialog(
            diagonal_q=StageStatus.COMPLETE,
            reusable_artifacts={"diagonal_q": True},
        )
        self.assertEqual(dialog.status_text("diagonal_q"), "COMPLETE")
        self.assertEqual(
            dialog.artifact_text("diagonal_q"),
            "Completed artifact will be reused.",
        )

        for status in StageStatus:
            with self.subTest(status=status):
                dialog.set_stage_status("static_parameters", status)
                self.assertEqual(
                    dialog.status_text("static_parameters"), status.value
                )

        dialog.set_stage_status("static_parameters", StageStatus.STALE)
        self.assertEqual(
            dialog.artifact_text("static_parameters"),
            "Completed artifact is stale and will not be reused.",
        )

    def test_reusable_q_detail_is_visible(self):
        dialog = WorkflowLaunchDialog(
            {
                "diagonal_q": StageStatus.COMPLETE,
                "static_parameters": StageStatus.READY,
            },
            reusable_artifacts={"diagonal_q": True},
            artifact_details={
                "diagonal_q": (
                    "Q diag [Fx, Fy, Fz, tau_x, tau_y, tau_z] = "
                    "[1, 2, 3, 4, 5, 6]."
                )
            },
        )
        text = dialog.artifact_text("diagonal_q")
        self.assertIn("will be reused", text)
        self.assertIn("[1, 2, 3, 4, 5, 6]", text)

    def test_running_attempt_locks_mode_and_start_but_keeps_cancel_available(self):
        dialog = self._dialog(running=True, selected_mode=WorkflowMode.ALL)
        self.assertTrue(dialog.running)
        self.assertFalse(dialog.staged_mode_radio.isEnabled())
        self.assertFalse(dialog.all_mode_radio.isEnabled())
        self.assertFalse(dialog.start_button.isEnabled())
        self.assertTrue(dialog.cancel_button.isEnabled())
        self.assertEqual(dialog.selected_mode, WorkflowMode.ALL)
        self.assertTrue(dialog.all_mode_radio.isChecked())

        dialog.staged_mode_radio.click()
        dialog.set_selected_mode(WorkflowMode.STEP)
        self.assertEqual(dialog.selected_mode, WorkflowMode.ALL)

        dialog.set_running(False)
        self.assertFalse(dialog.running)
        self.assertTrue(dialog.staged_mode_radio.isEnabled())
        self.assertTrue(dialog.all_mode_radio.isEnabled())
        self.assertTrue(dialog.start_button.isEnabled())

    def test_running_stage_also_locks_mode(self):
        dialog = self._dialog(diagonal_q=StageStatus.RUNNING)
        self.assertTrue(dialog.running)
        self.assertFalse(dialog.all_mode_radio.isEnabled())
        self.assertFalse(dialog.start_button.isEnabled())

        dialog.set_stage_status("diagonal_q", StageStatus.RETRY)
        self.assertFalse(dialog.running)
        self.assertTrue(dialog.all_mode_radio.isEnabled())
        self.assertTrue(dialog.start_button.isEnabled())

    def test_start_is_disabled_when_no_stage_can_start(self):
        for statuses in (
            (StageStatus.COMPLETE, StageStatus.COMPLETE),
            (StageStatus.BLOCKED, StageStatus.BLOCKED),
        ):
            with self.subTest(statuses=statuses):
                dialog = self._dialog(*statuses)
                self.assertFalse(dialog.start_button.isEnabled())

    def test_cancel_rejects_without_a_launch_selection(self):
        dialog = self._dialog()
        emitted = []
        dialog.launchRequested.connect(emitted.append)

        dialog.cancel_button.click()

        self.assertEqual(dialog.result(), QDialog.Rejected)
        self.assertIsNone(dialog.launch_selection)
        self.assertEqual(emitted, [])

    def test_constructor_requires_exact_two_stage_contract(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            WorkflowLaunchDialog({"diagonal_q": StageStatus.READY})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self._dialog(reusable_artifacts={"another_stage": True})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self._dialog(artifact_details={"another_stage": "detail"})


if __name__ == "__main__":
    unittest.main()
