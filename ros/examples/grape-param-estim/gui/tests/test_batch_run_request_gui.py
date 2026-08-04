import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

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
from grape_param_estim_gui.workflow import (
    ArtifactRef,
    WorkflowMode,
    artifact_content_fingerprint,
    canonical_fingerprint,
    completion_fingerprint,
)
from grape_param_estim_gui.widgets.workflow_dialog import (
    WorkflowLaunchSelection,
)


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
        self.assertEqual(settings["q"]["update_policy"], "fixed")

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
                window,
                "_choose_workflow_mode",
                return_value=WorkflowLaunchSelection(
                    WorkflowMode.STEP, "fixed"
                ),
            ):
                window.start_estimation()
            payload = json.loads(
                captured["request_path"].read_text(encoding="utf-8")
            )
            parsed = validate_batch_estimation_request(payload)
            self.assertEqual(parsed.payload["run_mode"], "estimate_only")
            self.assertFalse(parsed.payload["mcmc_settings"]["enabled"])
            self.assertEqual(parsed.payload["q"]["update_policy"], "fixed")
            self.assertEqual(tuple(parsed.payload["bags"][0]["interval_seconds"]), (18.0, 24.0))
            self.assertEqual(
                parsed.payload["bags"][0]["observation_factors"]["accelerometer"]["disabled_reason"],
                "accelerometer disabled: sensor frame and lever arm are not confirmed by inspection",
            )
            self.assertEqual(captured["script"].name, "grape_estimate_flights.py")
            window.close()

    def test_all_appends_and_resumes_sampling_in_same_completed_run(self):
        from grape_param_estim.posterior_sampling_request import (
            validate_posterior_sampling_request,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = new_project_manifest("posterior-append-gui")
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
                inspection=inspection,
                included=True,
                auto_interval=(18.0, 24.0),
                selected_interval=(18.0, 24.0),
                status="complete",
                configuration_fingerprint=(
                    "manual-group:sha256:" + "a" * 64
                ),
            )
            store.add(record)
            package = Path(__file__).resolve().parents[2]
            window = MainWindow(store, package)
            inputs = window._derive_workflow_inputs(
                (record,), WorkflowMode.STEP
            )
            attempt_id = "batch-estimate-only"
            parent = project / "runs" / attempt_id
            output = parent / "estimation_run"
            output.mkdir(parents=True)
            request_path = parent / "request.json"
            request = build_batch_estimation_request(
                run_id=attempt_id,
                run_mode="estimate_only",
                resume=False,
                output_directory=output,
                bags=inputs["bags"],
                settings=inputs["settings"],
            )
            window._write_request(request_path, request)
            request_fingerprint = canonical_fingerprint(request)
            state = window._workflow_state.begin_attempt(
                stage_id="batch_estimation",
                attempt_id=attempt_id,
                request_path=request_path.relative_to(project).as_posix(),
                output_path=output.relative_to(project).as_posix(),
                root_input_fingerprint=inputs["root_fingerprint"],
                stage_input=inputs["stage_input"],
                request_fingerprint=request_fingerprint,
                created_at="2026-08-04T00:00:00+00:00",
            )
            state = state.mark_running(
                attempt_id, started_at="2026-08-04T00:00:01+00:00"
            )
            attempt = state.attempt(attempt_id)
            content = artifact_content_fingerprint(
                {"schema": "grape-param-estim/batch-estimation-run/v1"},
                {"map_static.npz": "1" * 64},
            )
            completion = completion_fingerprint(
                stage_input=attempt.stage_input_fingerprint,
                request_fingerprint=attempt.request_fingerprint,
                artifact_schema=(
                    "grape-param-estim/batch-estimation-run/v1"
                ),
                artifact_content=content,
            )
            state = state.mark_complete(
                attempt_id,
                ArtifactRef(
                    schema="grape-param-estim/batch-estimation-run/v1",
                    artifact_id=attempt_id,
                    relative_path=attempt.output_path,
                    content_fingerprint=content,
                    completion_fingerprint=completion,
                ),
                finished_at="2026-08-04T00:00:02+00:00",
            )
            window._workflow_state = state
            window._save_workflow_state()
            run_manifest = {
                "schema": "grape-param-estim/batch-estimation-run/v1",
                "status": "complete",
                "run_id": attempt_id,
                "request_fingerprint": request_fingerprint,
                "configuration_fingerprint": "sha256:" + "2" * 64,
                "controller_snapshot_fingerprint": "sha256:" + "3" * 64,
                "estimator_revision": "revision-a",
                "selected_bag_ids": ["bag-a"],
                "selected_intervals": {"bag-a": [18.0, 24.0]},
                "selected_bag_sha256": {
                    "bag-a": "sha256:" + entry["sha256"]
                },
                "mcmc_settings": {"enabled": False},
            }
            run = SimpleNamespace(
                root=output,
                manifest=run_manifest,
                mcmc=None,
                request_fingerprint=request_fingerprint,
                static_map=SimpleNamespace(bag_objective={}),
                bags={},
            )
            store.estimation_run = run
            checkpoint = output.parent / (
                ".{}-batch-checkpoint".format(output.name)
            )
            checkpoint.mkdir()
            checkpoint_manifest = {
                "schema": "grape-param-estim/batch-estimation-checkpoint/v1",
                "status": "published",
                "run_id": attempt_id,
                "request_fingerprint": request_fingerprint,
                "configuration_fingerprint": "sha256:" + "2" * 64,
                "controller_snapshot_fingerprint": "sha256:" + "3" * 64,
                "estimator_revision": "revision-a",
                "output_directory": str(output),
                "chain_checkpoints": {},
                "sampling_context": None,
            }
            (checkpoint / "manifest.json").write_text(
                json.dumps(checkpoint_manifest), encoding="utf-8"
            )
            captured = []

            def fake_start(operation, run_id, selected_request, selected_output, script):
                captured.append(
                    (operation, run_id, selected_request, selected_output, script)
                )
                window._operation = operation
                window._operation_context = {}
                return True

            window._start_worker = fake_start
            with mock.patch.object(
                window,
                "_choose_workflow_mode",
                return_value=WorkflowLaunchSelection(
                    WorkflowMode.ALL, "fixed"
                ),
            ):
                window.start_estimation()
            first_payload = json.loads(
                captured[-1][2].read_text(encoding="utf-8")
            )
            first = validate_posterior_sampling_request(first_payload)
            self.assertFalse(first.payload["resume"])
            self.assertEqual(captured[-1][0], "posterior_sampling")
            self.assertEqual(captured[-1][3], output)
            self.assertEqual(
                captured[-1][4].name,
                "grape_sample_parameter_posterior.py",
            )

            checkpoint_manifest["status"] = "cancelled"
            checkpoint_manifest["sampling_context"] = {
                "sampling_request_fingerprint": first.fingerprint,
                "mcmc_settings": {
                    "enabled": True,
                    **first_payload["mcmc_settings"],
                },
                "sampler_revision": "revision-a",
            }
            (checkpoint / "manifest.json").write_text(
                json.dumps(checkpoint_manifest), encoding="utf-8"
            )
            record.status = "complete"
            window._operation = None
            window._operation_context = {}
            with mock.patch.object(
                window,
                "_choose_workflow_mode",
                return_value=WorkflowLaunchSelection(
                    WorkflowMode.ALL, "fixed"
                ),
            ):
                window.start_estimation()
            resumed_payload = json.loads(
                captured[-1][2].read_text(encoding="utf-8")
            )
            resumed = validate_posterior_sampling_request(resumed_payload)
            self.assertTrue(resumed.payload["resume"])
            self.assertEqual(first.fingerprint, resumed.fingerprint)
            self.assertEqual(
                len(
                    window._workflow_state.stage(
                        "batch_estimation"
                    ).attempts
                ),
                1,
            )
            completed_attempt = window._workflow_state.attempt(attempt_id)
            upgraded_content = artifact_content_fingerprint(
                {
                    "schema": "grape-param-estim/batch-estimation-run/v1",
                    "mcmc": "complete",
                },
                {"mcmc_samples.npz": "5" * 64},
            )
            upgraded_reference = ArtifactRef(
                schema="grape-param-estim/batch-estimation-run/v1",
                artifact_id=attempt_id,
                relative_path=completed_attempt.output_path,
                content_fingerprint=upgraded_content,
                completion_fingerprint=completion_fingerprint(
                    stage_input=completed_attempt.stage_input_fingerprint,
                    request_fingerprint=completed_attempt.request_fingerprint,
                    artifact_schema=(
                        "grape-param-estim/batch-estimation-run/v1"
                    ),
                    artifact_content=upgraded_content,
                ),
            )
            run.mcmc = SimpleNamespace()
            run_manifest["mcmc_settings"] = {
                "enabled": True,
                "sampling_request_fingerprint": resumed.fingerprint,
            }
            sampling_inputs = window._derive_posterior_sampling_inputs(
                (record,)
            )
            self.assertTrue(sampling_inputs["already_complete"])
            with mock.patch(
                "grape_param_estim_gui.main_window."
                "artifact_ref_from_validated_bundle",
                return_value=upgraded_reference,
            ):
                window._adopt_completed_posterior_sampling(
                    sampling_inputs
                )
            self.assertEqual(
                window._workflow_state.attempt(attempt_id).artifact,
                upgraded_reference,
            )
            window.close()


if __name__ == "__main__":
    unittest.main()
