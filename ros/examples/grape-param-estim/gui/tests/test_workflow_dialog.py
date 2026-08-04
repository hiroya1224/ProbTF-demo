import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from grape_param_estim_gui.workflow import WorkflowMode
from grape_param_estim_gui.widgets.workflow_dialog import WorkflowLaunchDialog


class WorkflowLaunchDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_step_maps_to_estimate_only_choice(self):
        dialog = WorkflowLaunchDialog(selected_mode=WorkflowMode.STEP)
        dialog.start_button.click()
        self.assertEqual(dialog.launch_selection.mode, WorkflowMode.STEP)
        self.assertIn("Estimate only", dialog.staged_mode_radio.text())

    def test_all_maps_to_estimate_and_sample_choice(self):
        dialog = WorkflowLaunchDialog(selected_mode=WorkflowMode.ALL)
        dialog.start_button.click()
        self.assertEqual(dialog.launch_selection.mode, WorkflowMode.ALL)
        self.assertIn("sample posterior", dialog.all_mode_radio.text())

    def test_running_locks_launch(self):
        dialog = WorkflowLaunchDialog(running=True)
        self.assertFalse(dialog.start_button.isEnabled())
        self.assertFalse(dialog.staged_mode_radio.isEnabled())


if __name__ == "__main__":
    unittest.main()
