import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from grape_param_estim_gui.artifact_loader import (
    AssimilationRun,
    FlightResult,
    SharedPosterior,
)
from grape_param_estim_gui.main_window import MainWindow
from grape_param_estim_gui.project_io import (
    copy_bag_into_project,
    create_project_directory,
    new_project_manifest,
    write_project_manifest,
)
from grape_param_estim_gui.stage_requests import (
    DIAGONAL_Q_STAGE_ID,
    STATIC_PARAMETERS_STAGE_ID,
)
from grape_param_estim_gui.state import BagRecord, ProjectStore
from grape_param_estim_gui.workflow import AttemptStatus, WorkflowMode
from grape_param_estim_gui.workflow_io import load_workflow
from grape_param_estim_gui.widgets.timeline import _residual_time_axis


class StagedMainWindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        manifest = new_project_manifest("staged-main-window")
        manifest["estimator_settings"] = {
            "sample_period": 0.1,
            "ensemble_size": 58,
            "maximum_iterations": 1,
            "convergence_tolerance": 1.0e-3,
            "seed": 17,
            "delay_prior_mean": 0.02,
            "delay_prior_standard_deviation": 0.01,
            "forecast_workers": 1,
        }
        self.project = create_project_directory(
            self.root / "projects", manifest
        )
        source = self.root / "source.bag"
        source.write_bytes(b"staged workflow flight")
        entry = copy_bag_into_project(self.project, source, "bag-a")
        manifest["bags"].append(entry)
        self.store = ProjectStore(self.project, manifest)
        self.record = BagRecord(
            bag_id="bag-a",
            path=self.project / entry["relative_path"],
            source_path=source,
            sha256=entry["sha256"],
        )
        self.record.inspection = {
            "status": "ready",
            "recommended_interval": {
                "episode_index": 0,
                "interval": {
                    "start_local_time": 0.0,
                    "end_local_time": 1.0,
                },
            },
            "configuration_fingerprint": {
                "value": "complete:" + "a" * 64,
                "complete": True,
            },
            "controller_snapshot": {"gains": [[1.0, 0.1, 0.5]] * 4},
        }
        self.record.auto_interval = (0.0, 1.0)
        self.record.selected_interval = (0.0, 1.0)
        self.record.status = "ready"
        self.record.included = True
        self.record.configuration_fingerprint = "complete:" + "a" * 64
        self.record.controller_snapshot = {
            "gains": [[1.0, 0.1, 0.5]] * 4
        }
        self.store.add(self.record)
        self.store._sync_manifest_inputs()
        write_project_manifest(self.project, self.store.manifest)
        self.starts = []

    def tearDown(self):
        self.temporary.cleanup()

    def _window(self):
        patches = (
            mock.patch("grape_param_estim_gui.widgets.scene_3d.pv", None),
            mock.patch(
                "grape_param_estim_gui.widgets.scene_3d.QtInteractor", None
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        window = MainWindow(self.store, self.root / "package")

        def start_worker(operation, run_id, request_path, output, script):
            self.starts.append(
                {
                    "operation": operation,
                    "run_id": run_id,
                    "request_path": Path(request_path),
                    "output": Path(output),
                    "script": Path(script),
                }
            )
            window._operation = operation
            window._operation_context = {
                "request": Path(request_path),
                "output": Path(output),
            }
            return True

        window._start_worker = start_worker
        return window

    def _launch(self, window, mode):
        with mock.patch.object(
            window, "_choose_workflow_mode", return_value=mode
        ):
            window.start_assimilation()
        self.assertEqual(self.starts[-1]["operation"], DIAGONAL_Q_STAGE_ID)
        self.assertEqual(
            window._workflow_state.active_attempt.status,
            AttemptStatus.RUNNING,
        )

    def _manifest(self, window, output, schema):
        attempt = window._workflow_state.active_attempt
        self.assertIsNotNone(attempt)
        output.mkdir(parents=True, exist_ok=True)
        return {
            "schema": schema,
            "status": "complete",
            "run_id": attempt.attempt_id,
            "stage_id": window._operation,
            "stage_input_fingerprint": attempt.stage_input_fingerprint,
            "request_fingerprint": attempt.request_fingerprint,
            "project_fingerprint": attempt.root_input_fingerprint,
            "artifacts": {
                "payload": {
                    "path": "payload.npz",
                    "sha256": "sha256:" + "b" * 64,
                    "size_bytes": 1,
                }
            },
        }

    def _q_bundle(self, window):
        output = self.starts[0]["output"]
        manifest = self._manifest(
            window,
            output,
            "grape-param-estim/diagonal-wrench-q-estimate/v1",
        )
        return SimpleNamespace(
            root=output,
            manifest=manifest,
            covariance=SimpleNamespace(
                stationary_variance=np.asarray(
                    (0.1, 0.2, 0.3, 0.01, 0.02, 0.03)
                )
            ),
        )

    def _stage2_run(self, window):
        output = self.starts[-1]["output"]
        manifest = self._manifest(
            window,
            output,
            "grape-param-estim/fixed-q-augmented-parameter-estimate/v1",
        )
        manifest["project_request_fingerprint"] = manifest[
            "project_fingerprint"
        ]
        shared = SharedPosterior(
            member_id=np.asarray((3, 9), dtype=np.int64),
            parameter_coordinate=np.zeros((2, 19)),
            mass=np.asarray((2.2, 2.3)),
            inertia=np.asarray((np.eye(3), 1.1 * np.eye(3))),
            cog=np.zeros((2, 3)),
            force_effectiveness=np.ones((2, 4)),
            torque_effectiveness=np.ones((2, 4)),
            constant_delay=np.asarray((0.015, 0.025)),
            ridge={},
            mode={},
            iteration_diagnostics={},
        )
        times = np.asarray((0.0, 1.0))
        quaternion = np.tile(
            np.asarray((0.0, 0.0, 0.0, 1.0)), (2, 1)
        )
        flight = FlightResult(
            bag_id="bag-a",
            time=times,
            record_time=times + 100.0,
            reference_position=np.zeros((2, 3)),
            reference_rpy=np.zeros((2, 3)),
            observed_position=np.zeros((2, 3)),
            observed_orientation_xyzw=quaternion,
            nominal_position=np.zeros((2, 3)),
            nominal_orientation_xyzw=quaternion,
            member_position=np.zeros((2, 2, 3)),
            member_orientation_xyzw=np.tile(quaternion, (2, 1, 1)),
            correction_translation=np.zeros((2, 2, 3)),
            correction_rotation_vector=np.zeros((2, 2, 3)),
            observed_correction_translation=np.zeros((2, 3)),
            observed_correction_rotation_vector=np.zeros((2, 3)),
            residual_wrench=np.zeros((2, 2, 6)),
            flight_state=None,
            q_resolution_sufficient=None,
            provenance={},
            calibration={},
            coverage={},
            objective_contribution=-4.0,
        )
        return AssimilationRun(
            root=output,
            manifest=manifest,
            shared_posterior=shared,
            bag_results={"bag-a": flight},
            diagnostics={},
            warnings=("sequential EnRTS marginal",),
        )

    def _persisted(self):
        return load_workflow(self.project, self.store.project_id)

    def test_step_stops_after_successful_q_boundary(self):
        window = self._window()
        self._launch(window, WorkflowMode.STEP)
        q_bundle = self._q_bundle(window)
        with mock.patch(
            "grape_param_estim_gui.main_window.load_diagonal_q_stage",
            return_value=q_bundle,
        ), mock.patch(
            "grape_param_estim_gui.main_window.QTimer.singleShot"
        ) as single_shot:
            window._worker_finished(str(q_bundle.root))
        self.assertEqual(len(self.starts), 1)
        single_shot.assert_not_called()
        q_attempt = window._workflow_state.stage(
            DIAGONAL_Q_STAGE_ID
        ).attempts[-1]
        self.assertEqual(q_attempt.status, AttemptStatus.COMPLETE)
        self.assertEqual(
            window._workflow_state.stage(STATIC_PARAMETERS_STAGE_ID).attempts,
            (),
        )
        self.assertEqual(self._persisted(), window._workflow_state)
        window.close()

    def test_all_continues_after_q_and_stage2_completion_applies_new_run(self):
        window = self._window()
        self._launch(window, WorkflowMode.ALL)
        q_bundle = self._q_bundle(window)
        scheduled = []
        q_fingerprint = "sha256:" + "c" * 64
        with mock.patch(
            "grape_param_estim_gui.main_window.load_diagonal_q_stage",
            return_value=q_bundle,
        ), mock.patch(
            "grape_param_estim_gui.main_window.diagonal_q_stage_fingerprint",
            return_value=q_fingerprint,
        ), mock.patch(
            "grape_param_estim_gui.main_window.QTimer.singleShot",
            side_effect=lambda _delay, callback: scheduled.append(callback),
        ):
            window._worker_finished(str(q_bundle.root))
            self.assertEqual(len(scheduled), 1)
            scheduled[0]()
        self.assertEqual(
            [value["operation"] for value in self.starts],
            [DIAGONAL_Q_STAGE_ID, STATIC_PARAMETERS_STAGE_ID],
        )
        self.assertEqual(
            window._workflow_state.active_attempt.status,
            AttemptStatus.RUNNING,
        )
        request_text = self.starts[-1]["request_path"].read_text(
            encoding="utf-8"
        )
        self.assertIn(q_fingerprint, request_text)

        run = self._stage2_run(window)
        with mock.patch(
            "grape_param_estim_gui.main_window.load_augmented_parameter_assimilation",
            return_value=run,
        ) as loader:
            window._worker_finished(str(run.root))
        loader.assert_called_once_with(run.root)
        self.assertIs(self.store.assimilation_run, run)
        self.assertIs(self.record.result, run.bag_results["bag-a"])
        self.assertEqual(self.record.status, "complete")
        self.assertEqual(
            self.store.manifest["run_request_fingerprint"],
            run.manifest["project_fingerprint"],
        )
        self.assertEqual(
            window._workflow_state.stage(STATIC_PARAMETERS_STAGE_ID)
            .attempts[-1]
            .status,
            AttemptStatus.COMPLETE,
        )
        self.assertEqual(self._persisted(), window._workflow_state)
        window.close()

    def test_worker_failure_is_persisted_as_retryable_attempt(self):
        window = self._window()
        self._launch(window, WorkflowMode.STEP)
        with mock.patch.object(window, "_show_error") as show_error:
            window._worker_failed("forecast exploded")
        attempt = window._workflow_state.stage(
            DIAGONAL_Q_STAGE_ID
        ).attempts[-1]
        self.assertEqual(attempt.status, AttemptStatus.FAILED)
        self.assertEqual(attempt.failure, "worker_failed: forecast exploded")
        self.assertEqual(self._persisted(), window._workflow_state)
        show_error.assert_called_once()
        window.close()

    def test_worker_cancellation_is_persisted_as_retryable_attempt(self):
        window = self._window()
        self._launch(window, WorkflowMode.STEP)
        window._worker_cancelled()
        attempt = window._workflow_state.stage(
            DIAGONAL_Q_STAGE_ID
        ).attempts[-1]
        self.assertEqual(attempt.status, AttemptStatus.CANCELLED)
        self.assertEqual(attempt.failure, "user_requested")
        self.assertEqual(self._persisted(), window._workflow_state)
        window.close()

    def test_residual_time_axis_accepts_staged_boundaries_and_legacy_intervals(self):
        times = np.asarray((0.0, 0.1, 0.2))
        np.testing.assert_array_equal(
            _residual_time_axis(times, np.zeros((2, 3, 6))), times
        )
        np.testing.assert_array_equal(
            _residual_time_axis(times, np.zeros((2, 2, 6))), times[:-1]
        )
        with self.assertRaisesRegex(ValueError, "N or N-1"):
            _residual_time_axis(times, np.zeros((2, 1, 6)))
        with self.assertRaisesRegex(ValueError, "member, time, component"):
            _residual_time_axis(times, np.zeros((2, 6)))


if __name__ == "__main__":
    unittest.main()
