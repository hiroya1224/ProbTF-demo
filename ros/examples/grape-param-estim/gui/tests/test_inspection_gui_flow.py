import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication, QInputDialog

from grape_param_estim_gui.artifact_loader import FlightResult, InspectionArtifact
from grape_param_estim_gui.main_window import MainWindow
from grape_param_estim_gui.project_io import (
    ProjectIoError,
    copy_bag_into_project,
    create_project_directory,
    load_project_archive,
    new_project_manifest,
    read_project_manifest,
    save_project_archive,
    validate_project_manifest,
)
from grape_param_estim_gui.state import BagRecord, ProjectStore
from grape_param_estim_gui.workflow import WorkflowMode


def _preview(bag_id: str) -> FlightResult:
    time = np.asarray((0.0, 1.0, 2.0))
    position = np.asarray(
        ((0.0, 0.0, 0.0), (0.1, -0.1, 0.4), (0.2, -0.2, 0.8))
    )
    quaternion = np.tile(np.asarray((0.0, 0.0, 0.0, 1.0)), (3, 1))
    return FlightResult(
        bag_id=bag_id,
        time=time,
        record_time=time + 100.0,
        reference_position=position + 0.01,
        reference_rpy=np.zeros((3, 3)),
        observed_position=position,
        observed_orientation_xyzw=quaternion,
        nominal_position=None,
        nominal_orientation_xyzw=None,
        member_position=None,
        member_orientation_xyzw=None,
        correction_translation=None,
        correction_rotation_vector=None,
        observed_correction_translation=None,
        observed_correction_rotation_vector=None,
        residual_wrench=None,
        flight_state=np.asarray((3.0, 3.0, 17.0)),
        q_resolution_sufficient=None,
        provenance={},
        calibration={},
        coverage={},
        objective_contribution=None,
    )


def _inspection(
    root: Path,
    bag_id: str,
    digest: str,
    *,
    status: str = "needs_configuration_confirmation",
    fingerprint_marker: str = "1",
) -> InspectionArtifact:
    inspection = {
        "bag_id": bag_id,
        "bag_sha256": digest,
        "status": status,
        "recommended_interval": {
            "episode_index": 0,
            "interval": {"start_local_time": 0.0, "end_local_time": 2.0},
        },
        "configuration_fingerprint": {
            "value": "incomplete:" + fingerprint_marker * 64,
            "complete": False,
            "components": {},
            "missing_components": [
                "payload",
                "rotor_propeller",
                "geometry",
                "robot_model_revision",
                "actuator_wiring",
                "hardware_revision",
            ],
        },
        "controller_snapshot": {"gains": [[1.0, 0.1, 0.5]] * 4},
        "topic_contract": [],
        "warnings": ["configuration fingerprint is incomplete"],
    }
    return InspectionArtifact(
        root=root,
        manifest={"status": "complete"},
        inspections={bag_id: inspection},
        previews={bag_id: _preview(bag_id)},
    )


class InspectionGuiFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = new_project_manifest("inspection-gui-flow")
        self.manifest["estimator_settings"] = {
            "sample_period": 0.1,
            "maximum_knots": 2,
            "ensemble_size": 58,
            "maximum_iterations": 1,
            "convergence_tolerance": 1.0e-3,
            "minimum_line_search_step": 1.0 / 64.0,
            "seed": 7,
            "delay_prior_mean": 0.02,
            "delay_prior_standard_deviation": 0.01,
            "allow_configuration_mismatch": False,
        }
        self.project = create_project_directory(
            self.root / "projects", self.manifest
        )
        source = self.root / "source.bag"
        source.write_bytes(b"recorded flight")
        entry = copy_bag_into_project(self.project, source, "bag-a")
        self.manifest["bags"].append(entry)
        self.store = ProjectStore(self.project, self.manifest)
        self.record = BagRecord(
            bag_id="bag-a",
            path=self.project / entry["relative_path"],
            source_path=source,
            sha256=entry["sha256"],
        )
        self.store.add(self.record)

    def tearDown(self):
        self.temporary.cleanup()

    def _window(self) -> MainWindow:
        patches = (
            mock.patch("grape_param_estim_gui.widgets.scene_3d.pv", None),
            mock.patch(
                "grape_param_estim_gui.widgets.scene_3d.QtInteractor", None
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        return MainWindow(self.store, self.root / "package")

    def test_async_inspection_rebinds_current_preview_to_all_preview_plots(self):
        window = self._window()
        view = window.bag_browser
        self.assertIsNone(view.trajectory_panel.session)

        artifact = _inspection(self.root / "inspection", "bag-a", self.record.sha256)
        self.store.apply_inspection(artifact)

        self.assertIs(view.trajectory_panel.session, artifact.previews["bag-a"])
        self.assertIs(view.scene.session, artifact.previews["bag-a"])
        self.assertEqual(len(view.trajectory_panel.plots[0].listDataItems()), 2)
        self.assertEqual(len(view.flight_state_plot.listDataItems()), 1)
        self.assertIn("3 samples / 2.0 s", view.samples_label.text())
        self.assertTrue(view.confirm_group_button.isEnabled())
        self.assertFalse(view.include_checkbox.isEnabled())
        window.close()

    def test_manual_group_confirmation_enables_and_launches_smoothing(self):
        window = self._window()
        artifact = _inspection(self.root / "inspection", "bag-a", self.record.sha256)
        self.store.apply_inspection(artifact)
        with mock.patch.object(
            QInputDialog,
            "getText",
            return_value=("single-bag-test", True),
        ):
            window.confirm_configuration_group("bag-a")

        self.assertEqual(self.record.status, "ready")
        self.assertTrue(self.record.included)
        self.assertTrue(
            self.record.configuration_fingerprint.startswith(
                "manual-group:sha256:"
            )
        )
        self.assertEqual(
            self.record.configuration_confirmation["group_id"],
            "single-bag-test",
        )
        persisted = read_project_manifest(self.project)
        self.assertEqual(
            persisted["configuration_confirmations"]["bag-a"]["group_id"],
            "single-bag-test",
        )
        archive = save_project_archive(
            self.project, self.root / "confirmed-project.zip"
        )
        restored_project = load_project_archive(
            archive, self.root / "restored-projects"
        )
        persisted = read_project_manifest(restored_project)
        tampered = copy.deepcopy(persisted)
        tampered["configuration_confirmations"]["bag-a"][
            "group_id"
        ] = "different-group"
        with self.assertRaisesRegex(
            ProjectIoError, "does not match its group ID"
        ):
            validate_project_manifest(tampered)
        restored = ProjectStore(restored_project, persisted)
        restored.replace_project(
            restored_project,
            persisted,
            MainWindow._records_from_manifest(restored_project, persisted),
        )
        restored.apply_inspection(artifact)
        self.assertEqual(restored.get("bag-a").status, "ready")
        self.assertEqual(
            restored.get("bag-a").configuration_confirmation["group_id"],
            "single-bag-test",
        )
        self.assertTrue(window.run_action.isEnabled())

        started = []
        window._start_worker = lambda *arguments: started.append(arguments) or True
        with mock.patch.object(
            window,
            "_choose_workflow_mode",
            return_value=WorkflowMode.STEP,
        ):
            window.start_assimilation()
        self.assertEqual(len(started), 1)
        request_path = Path(started[0][2])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(request["bags"][0]["bag_id"], "bag-a")
        self.assertEqual(
            request["bags"][0]["configuration_fingerprint"],
            self.record.configuration_fingerprint,
        )
        window.close()

    def test_manual_group_cannot_override_a_blocked_inspection(self):
        window = self._window()
        artifact = _inspection(
            self.root / "inspection",
            "bag-a",
            self.record.sha256,
            status="blocked",
        )
        self.store.apply_inspection(artifact)

        with self.assertRaisesRegex(ValueError, "blocked inspection"):
            self.store.confirm_configuration_group("bag-a", "unsafe-override")
        self.assertEqual(self.record.status, "blocked")
        self.assertFalse(self.record.included)
        self.assertFalse(window.bag_browser.confirm_group_button.isEnabled())
        self.assertFalse(window.run_action.isEnabled())
        window.close()

    def test_reinspection_preserves_only_a_matching_valid_confirmation(self):
        window = self._window()
        original = _inspection(
            self.root / "inspection-a", "bag-a", self.record.sha256
        )
        self.store.apply_inspection(original)
        self.store.confirm_configuration_group("bag-a", "single-bag-test")

        same_source = _inspection(
            self.root / "inspection-b", "bag-a", self.record.sha256
        )
        self.store.apply_inspection(same_source)
        self.assertEqual(self.record.status, "ready")
        self.assertTrue(self.record.included)
        self.assertTrue(self.record.configuration_confirmation)

        changed_source = _inspection(
            self.root / "inspection-c",
            "bag-a",
            self.record.sha256,
            fingerprint_marker="2",
        )
        self.store.apply_inspection(changed_source)
        self.assertEqual(
            self.record.status, "needs_configuration_confirmation"
        )
        self.assertFalse(self.record.included)
        self.assertFalse(self.record.configuration_confirmation)

        self.store.confirm_configuration_group("bag-a", "single-bag-test")
        blocked = _inspection(
            self.root / "inspection-d",
            "bag-a",
            self.record.sha256,
            status="blocked",
            fingerprint_marker="2",
        )
        self.store.apply_inspection(blocked)
        self.assertEqual(self.record.status, "blocked")
        self.assertFalse(self.record.included)
        self.assertFalse(self.record.configuration_confirmation)
        window.close()

    def test_configuration_prompt_timer_is_bound_to_window_lifetime(self):
        window = self._window()
        artifact = _inspection(
            self.root / "inspection", "bag-a", self.record.sha256
        )
        self.store.apply_inspection(artifact)
        window.show()
        with mock.patch.object(
            window, "confirm_configuration_group"
        ) as confirm:
            window._schedule_configuration_prompts(("bag-a",))
            self.application.processEvents()
            confirm.assert_called_once_with("bag-a")

            confirm.reset_mock()
            window._schedule_configuration_prompts(("bag-a",))
            window._cancel_configuration_prompts()
            self.application.processEvents()
            confirm.assert_not_called()

            window._close_after_worker = True
            window._schedule_configuration_prompts(("bag-a",))
            self.application.processEvents()
            confirm.assert_not_called()
            window._close_after_worker = False
        window.close()

    def test_cancelled_smoothing_restores_ready_input_and_visible_details(self):
        window = self._window()
        artifact = _inspection(
            self.root / "inspection", "bag-a", self.record.sha256
        )
        self.store.apply_inspection(artifact)
        self.store.confirm_configuration_group("bag-a", "single-bag-test")
        self.record.status = "running"
        self.store.recordChanged.emit("bag-a")
        self.assertIn("status: running", window.bag_browser.inspection_details.text())

        window._operation = "assimilation"
        window._worker_cancelled()

        self.assertEqual(self.record.status, "ready")
        self.assertTrue(self.record.included)
        self.assertIn("status: ready", window.bag_browser.inspection_details.text())
        self.assertTrue(window.run_action.isEnabled())
        window.close()

    def test_worker_start_failure_restores_inspection_and_smoothing_inputs(self):
        window = self._window()
        artifact = _inspection(
            self.root / "inspection", "bag-a", self.record.sha256
        )
        self.store.apply_inspection(artifact)
        self.store.confirm_configuration_group("bag-a", "single-bag-test")
        window._start_worker = mock.Mock(return_value=False)

        window.inspect_bags(("bag-a",))
        self.assertEqual(self.record.status, "ready")
        self.assertTrue(self.record.included)
        self.assertTrue(window.run_action.isEnabled())

        with mock.patch.object(
            window,
            "_choose_workflow_mode",
            return_value=WorkflowMode.STEP,
        ):
            window.start_assimilation()
        self.assertEqual(self.record.status, "ready")
        self.assertTrue(self.record.included)
        self.assertTrue(window.run_action.isEnabled())
        window.close()

    def test_invalid_worker_output_restores_ready_input(self):
        window = self._window()
        artifact = _inspection(
            self.root / "inspection", "bag-a", self.record.sha256
        )
        self.store.apply_inspection(artifact)
        self.store.confirm_configuration_group("bag-a", "single-bag-test")
        self.record.status = "writing"
        self.store.recordChanged.emit("bag-a")
        window._operation = "assimilation"

        with mock.patch(
            "grape_param_estim_gui.main_window.load_assimilation",
            side_effect=ProjectIoError("invalid test artifact"),
        ), mock.patch.object(window, "_show_error") as show_error:
            window._worker_finished(str(self.root / "invalid-run"))

        self.assertEqual(self.record.status, "ready")
        self.assertTrue(self.record.included)
        self.assertTrue(window.run_action.isEnabled())
        show_error.assert_called_once()
        window.close()

    def test_adding_a_bag_opens_the_bag_browser(self):
        empty_manifest = new_project_manifest("add-bag-tab")
        empty_project = create_project_directory(
            self.root / "empty-projects", empty_manifest
        )
        empty_store = ProjectStore(empty_project, empty_manifest)
        with mock.patch("grape_param_estim_gui.widgets.scene_3d.pv", None), mock.patch(
            "grape_param_estim_gui.widgets.scene_3d.QtInteractor", None
        ):
            window = MainWindow(empty_store, self.root / "package")
            window.inspect_bags = mock.Mock()
            window.add_bag_files((self.root / "source.bag",))
            self.assertIs(window.tabs.currentWidget(), window.bag_browser)
            window.inspect_bags.assert_called_once()
            window.close()


if __name__ == "__main__":
    unittest.main()
