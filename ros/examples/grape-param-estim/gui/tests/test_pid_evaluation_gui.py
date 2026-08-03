import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
except ImportError as error:  # exactly the supported optional-test guard
    raise unittest.SkipTest("PySide6 is unavailable: {}".format(error))

import numpy as np

from grape_param_estim_gui.artifact_loader import (
    AssimilationRun,
    PidProposalEvaluation,
    SharedPosterior,
)
from grape_param_estim_gui.main_window import MainWindow
from grape_param_estim_gui.pid_request import PidEvaluationLaunchOptions
from grape_param_estim_gui.project_io import new_project_manifest
from grape_param_estim_gui.state import BagRecord, ProjectStore
from grape_param_estim_gui.widgets.next_experiment import NextExperimentView


def _shared() -> SharedPosterior:
    return SharedPosterior(
        member_id=np.asarray((11, 23), dtype=np.int64),
        parameter_coordinate=np.zeros((2, 1)),
        mass=np.asarray((1.1, 1.2)),
        inertia=np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        cog=np.zeros((2, 3)),
        force_effectiveness=np.ones((2, 2)),
        torque_effectiveness=np.ones((2, 2)),
        constant_delay=np.asarray((0.01, 0.02)),
        ridge={},
        mode={"selected_mode_id": np.asarray(("nominal",))},
        iteration_diagnostics={},
    )


def _run(root: Path) -> AssimilationRun:
    return AssimilationRun(
        root=root,
        manifest={
            "schema": "grape-param-estim/assimilation-run/v1",
            "status": "complete",
            "run_id": "source-run",
            "project_request_fingerprint": "sha256:" + "a" * 64,
            "selected_bag_ids": ["bag-a", "bag-b"],
        },
        shared_posterior=_shared(),
        bag_results={},
        diagnostics={},
        warnings=(),
    )


def _evaluation(root: Path) -> PidProposalEvaluation:
    candidate_ids = np.asarray(("current", "member-23-exact"))
    members = np.asarray((11, 23), dtype=np.int64)
    times = np.asarray((0.0, 1.0, 2.0))
    zeros = np.zeros((2, 2, 3, 3))
    prediction = zeros.copy()
    prediction[0, :, :, 0] = times
    prediction[1, :, :, 1] = times
    summary = {
        "candidate_id": candidate_ids,
        "candidate_source": np.asarray(("current", "member-derived")),
        "forecast_completion": np.ones(2),
        "numerical_failure_count": np.zeros(2, dtype=int),
        "log_gain_change": np.asarray((0.0, 0.2)),
        "pareto_dominated": np.asarray((False, False)),
        "improves_current": np.asarray((False, True)),
        "recommendation_available": np.asarray((False,)),
        "recommended_candidate_id": np.asarray(("",)),
        "rejection_reason": np.asarray(("not explicitly selected",)),
        "scenario_assumption": np.asarray(
            (
                "same recorded reference; residual policy by bag: "
                "bag-b=posterior_replay; this is not a forecast of a new "
                "disturbance realization",
            )
        ),
        "current_pid": np.ones((4, 3)),
        "proposed_pid": np.asarray((np.ones((4, 3)), 1.1 * np.ones((4, 3)))),
        "difference": np.asarray((np.zeros((4, 3)), 0.1 * np.ones((4, 3)))),
        "ratio": np.asarray((np.ones((4, 3)), 1.1 * np.ones((4, 3)))),
        "cvar_level": np.asarray((0.9,)),
        "position_threshold": np.asarray((np.nan,)),
        "orientation_threshold": np.asarray((np.nan,)),
        "bag_id": np.asarray(("bag-b",)),
        "member_bag_forecast_completion": np.ones((2, 1, 2), dtype=bool),
    }
    for metric in (
        "position_rmse",
        "orientation_rmse",
        "maximum_position_error",
        "maximum_orientation_error",
    ):
        summary["aggregate_{}_mean".format(metric)] = np.zeros(2)
        summary["aggregate_{}_upper_cvar".format(metric)] = np.zeros(2)
        summary["per_bag_{}_mean".format(metric)] = np.zeros((2, 1))
        summary["per_bag_{}_upper_cvar".format(metric)] = np.zeros((2, 1))
        summary["member_bag_{}".format(metric)] = np.zeros((2, 1, 2))
    bags = {
        "bag-b": {
            "candidate_id": candidate_ids,
            "member_id": members,
            "times": times,
            "reference_position": np.zeros((3, 3)),
            "prediction_position": prediction,
            "correction_translation": zeros.copy(),
            "correction_rotation_vector": zeros.copy(),
            "forecast_success": np.ones((2, 2), dtype=bool),
            "forecast_failure_reason": np.full((2, 2), ""),
        }
    }
    proposal = {
        "source_member_id": members,
        "proposed_pid": np.asarray((1.05 * np.ones((4, 3)), 1.1 * np.ones((4, 3)))),
    }
    return PidProposalEvaluation(
        root=root,
        manifest={
            "schema": "grape-param-estim/pid-proposal-evaluation/v1",
            "status": "complete",
            "evaluation_id": "evaluation-a",
            "source_run_id": "source-run",
            "selected_bag_ids": ["bag-b"],
        },
        proposal_ensemble=proposal,
        summary=summary,
        bags=bags,
        proposed_yaml="controller: exact\n",
        proposed_diff_yaml="difference: exact\n",
    )


class PidEvaluationGuiQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "assimilation_run").mkdir()
        self.store = ProjectStore(self.root, new_project_manifest("pid-gui-test"))
        for bag_id in ("bag-a", "bag-b"):
            bag = self.root / "bags" / (bag_id + ".bag")
            bag.parent.mkdir(exist_ok=True)
            bag.write_bytes(bag_id.encode("ascii"))
            digest = ("a" if bag_id == "bag-a" else "b") * 64
            self.store.manifest["bags"].append(
                {
                    "bag_id": bag_id,
                    "source_path": str(bag),
                    "relative_path": "bags/{}.bag".format(bag_id),
                    "sha256": digest,
                }
            )
            self.store.add(
                BagRecord(
                    bag_id=bag_id,
                    path=bag,
                    source_path=bag,
                    sha256=digest,
                    controller_snapshot={
                        "gains": (
                            np.arange(12, dtype=float).reshape(4, 3) + 1.0
                            + (10.0 if bag_id == "bag-b" else 0.0)
                        ).tolist()
                    },
                )
            )
        self.store.set_current("bag-b")
        self.store.assimilation_run = _run(self.root / "assimilation_run")
        self.store.set_selected_member(23)
        self.scene_patch_pv = mock.patch(
            "grape_param_estim_gui.widgets.scene_3d.pv", None
        )
        self.scene_patch_qt = mock.patch(
            "grape_param_estim_gui.widgets.scene_3d.QtInteractor", None
        )
        self.scene_patch_pv.start()
        self.scene_patch_qt.start()

    def tearDown(self):
        self.scene_patch_qt.stop()
        self.scene_patch_pv.stop()
        self.temporary.cleanup()

    def test_launch_controls_emit_exact_member_and_user_options(self):
        view = NextExperimentView(self.store)
        self.assertEqual(view.source_member_label.text(), "Selected member: 23")
        view.baseline_combo.setCurrentText("bag-b")
        view.residual_combo.setCurrentText("zero")
        view.cvar_spin.setValue(0.85)
        expected = np.arange(12, dtype=float).reshape(4, 3) + 11.0
        actual = np.asarray(
            [
                [editor.value() for editor in row]
                for row in view.user_gain_inputs
            ]
        )
        np.testing.assert_array_equal(actual, expected)
        view.user_candidate_group.setChecked(True)
        view.user_gain_inputs[0][0].setValue(44.0)
        view.selection_target_combo.setCurrentIndex(
            view.selection_target_combo.findData("user")
        )
        received = []
        view.evaluationRequested.connect(received.append)
        view.evaluate_button.click()
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].source_member_id, 23)
        self.assertEqual(received[0].baseline_bag_id, "bag-b")
        self.assertEqual(received[0].residual_policy, "zero")
        self.assertEqual(received[0].selected_candidate_source, "user")
        self.assertEqual(received[0].selected_candidate_id, "user-exact")
        self.assertEqual(received[0].user_candidate_values[0][0], 44.0)
        view.close()

    def test_3d_comparison_uses_same_exact_store_selection(self):
        view = NextExperimentView(self.store)
        evaluation = _evaluation(self.root / "evaluation")
        self.store.apply_pid_evaluation(evaluation)
        self.assertEqual(
            view.scenario_label.text().count("residual policy"), 1
        )
        self.assertIn(
            "does not apply gains automatically",
            view.yaml_safety_label.text(),
        )
        self.store.set_selected_pid_proposal("member-23-exact")
        self.assertEqual(view.comparison_scene.selected_bag_id, "bag-b")
        self.assertEqual(view.comparison_scene.selected_member_id, 23)
        self.assertEqual(
            view.comparison_scene.selected_candidate_id, "member-23-exact"
        )
        selection = view.comparison_scene._selection_indices()
        self.assertIsNotNone(selection)
        self.assertIn("x = red", view.correction_key.text())
        self.assertIn("Dashed lines", view.correction_key.text())
        self.assertIn("5–95% raw-member interval", view.correction_key.text())
        self.assertIn("zero desired correction", view.correction_key.text())
        view.close()

    def test_evaluation_clear_removes_all_old_gain_and_recommendation_evidence(self):
        view = NextExperimentView(self.store)
        evaluation = _evaluation(self.root / "evaluation-clear")
        evaluation.summary["recommendation_available"] = np.asarray((True,))
        evaluation.summary["recommended_candidate_id"] = np.asarray(
            ("member-23-exact",)
        )
        view.set_evaluation(evaluation)
        view.candidate_table.selectRow(1)
        self.assertIsNotNone(view.gain_table.item(0, 3))
        self.assertIn("member-23-exact", view.recommendation_label.text())

        view.set_evaluation(None)
        self.assertTrue(
            all(
                view.gain_table.item(row, column) is None
                for row in range(view.gain_table.rowCount())
                for column in range(view.gain_table.columnCount())
            )
        )
        self.assertEqual(view.recommendation_label.text(), "Recommendation: —")
        self.assertEqual(view.threshold_label.text(), "Thresholds: Not configured")
        self.assertEqual(view.candidate_table.rowCount(), 0)
        view.close()

    def test_threshold_and_candidate_comparison_are_artifact_backed(self):
        view = NextExperimentView(self.store)
        evaluation = _evaluation(self.root / "evaluation-threshold")
        evaluation.summary.update(
            {
                "position_threshold": np.asarray((0.35,)),
                "orientation_threshold": np.asarray((0.2,)),
                "position_threshold_configured": np.asarray((True,)),
                "orientation_threshold_configured": np.asarray((True,)),
                "position_threshold_metric": np.asarray(("position_rmse",)),
                "orientation_threshold_metric": np.asarray(
                    ("maximum_orientation_error",)
                ),
            }
        )
        view.set_evaluation(evaluation)
        self.assertIn("position RMSE limit = 0.35 m", view.threshold_label.text())
        self.assertIn(
            "maximum orientation error limit = 0.2 rad",
            view.threshold_label.text(),
        )
        self.assertTrue(view.threshold_label.wordWrap())
        self.assertEqual(view.candidate_table.columnCount(), 20)
        self.assertEqual(
            view.candidate_table.item(1, 2).text(), "1.1 / 1.1 / 1.1"
        )
        self.assertIn(
            "Position RMSE mean",
            view.candidate_table.horizontalHeaderItem(8).text(),
        )
        self.assertIn(
            "completion",
            view.gain_table.item(0, 8).text(),
        )
        view.close()

    def test_main_window_starts_worker_and_auto_loads_completed_artifact(self):
        window = MainWindow(self.store, self.root)
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
            window._operation_context = {
                "request": request_path,
                "output": output,
            }
            return True

        window._start_worker = fake_start
        window.start_pid_evaluation(
            PidEvaluationLaunchOptions(
                23,
                "bag-b",
                "posterior_replay",
                0.9,
                selected_candidate_source="member-derived",
            )
        )
        self.assertEqual(captured["operation"], "pid_evaluation")
        request = json.loads(captured["request_path"].read_text(encoding="utf-8"))
        self.assertEqual(request["candidates"][1]["source_member_id"], 23)
        self.assertEqual(request["selected_candidate_id"], "member-23-exact")

        evaluation = _evaluation(captured["output"])
        with mock.patch(
            "grape_param_estim_gui.main_window.load_pid_evaluation",
            return_value=evaluation,
        ):
            window._worker_finished(str(captured["output"]))
        self.assertIs(self.store.pid_evaluation, evaluation)
        self.assertEqual(
            self.store.manifest["current_pid_proposal_evaluation_id"],
            "evaluation-a",
        )
        window.close()

    def test_configuration_mismatch_is_rejected_before_stage_dialog(self):
        settings = {
            "sample_period": 0.1,
            "maximum_knots": 2,
            "ensemble_size": 58,
            "maximum_iterations": 1,
            "seed": 7,
            "delay_prior_mean": 0.02,
            "delay_prior_standard_deviation": 0.01,
            "allow_configuration_mismatch": False,
        }
        self.store.set_estimator_settings(settings)
        shared_snapshot = {"gains": np.ones((4, 3)).tolist()}
        for index, record in enumerate(self.store.records()):
            record.included = True
            record.inspection = {
                "recommended_interval": {"episode_index": 0}
            }
            record.auto_interval = (0.0, 1.0)
            record.selected_interval = (0.0, 1.0)
            record.status = "ready"
            record.configuration_fingerprint = "complete:" + (
                "a" if index == 0 else "b"
            ) * 64
            record.controller_snapshot = shared_snapshot
        self.store._sync_manifest_inputs()

        window = MainWindow(self.store, self.root)
        starts = []
        window._start_worker = lambda *arguments: starts.append(arguments) or True
        with mock.patch.object(window, "_show_error") as show_error, mock.patch.object(
            window, "_choose_workflow_mode"
        ) as choose_mode:
            window.start_assimilation()
        self.assertEqual(starts, [])
        choose_mode.assert_not_called()
        show_error.assert_called_once()
        self.assertIn(
            "share one confirmed configuration fingerprint",
            str(show_error.call_args.args[1]),
        )
        self.assertFalse(
            self.store.manifest["estimator_settings"]
            ["allow_configuration_mismatch"]
        )
        window.close()


if __name__ == "__main__":
    unittest.main()
