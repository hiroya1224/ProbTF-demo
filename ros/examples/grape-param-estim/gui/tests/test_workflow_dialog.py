import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

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
        self.assertEqual(dialog.launch_selection.q_update_policy, "fixed")
        self.assertEqual(dialog.launch_selection.solver_method, "sparse_lm")
        self.assertEqual(dialog.launch_selection.maximum_iterations, 30)
        self.assertIn("Estimate only", dialog.staged_mode_radio.text())

    def test_all_maps_to_estimate_and_sample_choice(self):
        dialog = WorkflowLaunchDialog(selected_mode=WorkflowMode.ALL)
        dialog.start_button.click()
        self.assertEqual(dialog.launch_selection.mode, WorkflowMode.ALL)
        self.assertIn("sample posterior", dialog.all_mode_radio.text())

    def test_laplace_em_q_policy_is_explicit(self):
        dialog = WorkflowLaunchDialog(
            selected_q_update_policy="laplace_em"
        )
        dialog.start_button.click()
        self.assertEqual(
            dialog.launch_selection.q_update_policy, "laplace_em"
        )
        self.assertTrue(dialog.estimate_q_radio.isChecked())

    def test_ieks_and_relinearization_limit_are_explicit(self):
        dialog = WorkflowLaunchDialog(
            selected_solver_method="ieks",
            selected_maximum_iterations=12,
        )
        dialog.start_button.click()
        self.assertEqual(dialog.launch_selection.solver_method, "ieks")
        self.assertEqual(dialog.launch_selection.maximum_iterations, 12)
        self.assertTrue(dialog.ieks_radio.isChecked())
        self.assertTrue(
            any(
                "lag" in label.text().lower()
                for label in dialog.findChildren(QLabel)
            )
        )
        self.assertTrue(
            any(
                "each delay-profile candidate" in label.text()
                for label in dialog.findChildren(QLabel)
            )
        )

    def test_running_locks_launch(self):
        dialog = WorkflowLaunchDialog(running=True)
        self.assertFalse(dialog.start_button.isEnabled())
        self.assertFalse(dialog.staged_mode_radio.isEnabled())
        self.assertFalse(dialog.fixed_q_radio.isEnabled())
        self.assertFalse(dialog.ieks_radio.isEnabled())
        self.assertFalse(dialog.maximum_iterations_spin.isEnabled())


if __name__ == "__main__":
    unittest.main()
