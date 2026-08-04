import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GRAPE_PARAM_ESTIM_DISABLE_3D", "1")

from PySide6.QtWidgets import QApplication

from grape_param_estim.batch_request import validate_batch_estimation_request
from grape_param_estim_gui.app import default_batch_estimator_settings
from grape_param_estim_gui.main_window import MainWindow
from grape_param_estim_gui.project_io import (
    copy_bag_into_project,
    create_project_directory,
    new_project_manifest,
)
from grape_param_estim_gui.state import (
    BagRecord,
    ProjectStore,
    bag_estimation_settings_from_inspection,
)
from grape_param_estim_gui.stage_requests import build_batch_estimation_request
from grape_param_estim_gui.workflow import WorkflowMode


def _inspection() -> dict[str, object]:
    topics = (
        "/gimbalrotor/mocap/pose",
        "/gimbalrotor/uav/baselink/odom",
        "/gimbalrotor/sensor_plugin/imu1/ros_converted",
        "/gimbalrotor/four_axes/command",
        "/gimbalrotor/gimbals_ctrl",
        "/gimbalrotor/joint_states",
        "/gimbalrotor/debug/pose/pid",
    )
    return {
        "topic_contract": [
            {"topic": topic, "present": True, "type_matches": True}
            for topic in topics
        ]
    }


class BatchRunRequestGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_default_settings_use_audited_actuator_contract_and_bounded_workload(self):
        settings = default_batch_estimator_settings()
        actuator = settings["actuator_model"]
        self.assertEqual(actuator["thrust_time_constant_seconds"], 0.01)
        self.assertEqual(actuator["gimbal_time_constant_seconds"], 0.02)
        self.assertIn("gimbal time constant provisional", actuator["source"])
        self.assertEqual(settings["knot_policy"]["period_seconds"], 0.05)
        self.assertEqual(settings["delay"]["coarse_grid_points"], 5)
        self.assertEqual(settings["delay"]["maximum_refinement_evaluations"], 8)
        self.assertEqual(settings["solver_settings"]["maximum_iterations"], 30)
        self.assertEqual(settings["em_settings"]["maximum_iterations"], 5)

    def test_run_builds_backend_valid_estimate_only_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = new_project_manifest("batch-run-gui")
            manifest["estimator_settings"] = default_batch_estimator_settings()
            project = create_project_directory(root / "projects", manifest)
            source = root / "flight.bag"
            source.write_bytes(b"flight")
            entry = copy_bag_into_project(project, source, "bag-a")
            manifest["bags"].append(entry)
            inspection = _inspection()
            manifest["bag_estimation_settings"]["bag-a"] = (
                bag_estimation_settings_from_inspection(inspection)
            )
            store = ProjectStore(project, manifest)
            record = BagRecord(
                bag_id="bag-a",
                path=project / entry["relative_path"],
                source_path=source,
                sha256=entry["sha256"],
                inspection=None,
                included=True,
                auto_interval=(18.0, 24.0),
                selected_interval=(18.0, 24.0),
                status="ready",
                configuration_fingerprint="manual-group:sha256:" + "a" * 64,
            )
            store.add(record)
            package = Path(__file__).resolve().parents[2]
            window = MainWindow(store, package)
            record.inspection = inspection
            sampled_inputs = window._derive_workflow_inputs(
                [record], WorkflowMode.ALL
            )
            sampled_request = build_batch_estimation_request(
                run_id="sampled-run",
                run_mode=sampled_inputs["run_mode"],
                resume=False,
                output_directory=root / "sampled-run",
                bags=sampled_inputs["bags"],
                settings=sampled_inputs["settings"],
            )
            sampled = validate_batch_estimation_request(sampled_request)
            self.assertEqual(sampled.payload["run_mode"], "estimate_and_sample")
            self.assertTrue(sampled.payload["mcmc_settings"]["enabled"])
            captured = {}

            def fake_start(operation, run_id, request_path, output, script):
                captured.update(
                    operation=operation,
                    run_id=run_id,
                    request_path=request_path,
                    output=output,
                    script=script,
                )
                window._operation = operation
                window._operation_context = {}
                return True

            window._start_worker = fake_start
            with mock.patch.object(
                window, "_choose_workflow_mode", return_value=WorkflowMode.STEP
            ):
                window.start_estimation()
            payload = json.loads(
                captured["request_path"].read_text(encoding="utf-8")
            )
            parsed = validate_batch_estimation_request(payload)
            self.assertEqual(parsed.payload["run_mode"], "estimate_only")
            self.assertFalse(parsed.payload["mcmc_settings"]["enabled"])
            self.assertEqual(tuple(parsed.payload["bags"][0]["interval_seconds"]), (18.0, 24.0))
            self.assertEqual(
                parsed.payload["bags"][0]["observation_factors"]["accelerometer"]["disabled_reason"],
                "accelerometer disabled: sensor frame and lever arm are not confirmed by inspection",
            )
            self.assertEqual(captured["script"].name, "grape_estimate_flights.py")
            window.close()


if __name__ == "__main__":
    unittest.main()
