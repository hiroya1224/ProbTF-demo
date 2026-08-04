import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GRAPE_PARAM_ESTIM_DISABLE_3D", "1")

try:
    from PySide6.QtWidgets import QApplication
except ImportError as error:
    raise unittest.SkipTest("PySide6 is unavailable: {}".format(error))

import numpy as np

from grape_param_estim_gui.artifact_loader import PidProposalEvaluation
from grape_param_estim_gui.project_io import new_project_manifest
from grape_param_estim_gui.state import BagRecord, ProjectStore
from grape_param_estim_gui.widgets.next_experiment import NextExperimentView

from test_pid_request import _run


class PidEvaluationGuiQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        run_root = self.root / "estimation"
        run_root.mkdir()
        self.store = ProjectStore(self.root, new_project_manifest("pid-gui"))
        for bag_id, digest in (("bag-a", "1" * 64), ("bag-b", "2" * 64)):
            path = self.root / (bag_id + ".bag")
            path.write_bytes(bag_id.encode("ascii"))
            self.store.add(
                BagRecord(
                    bag_id=bag_id,
                    path=path,
                    source_path=path,
                    sha256=digest,
                    included=True,
                    controller_snapshot={
                        "gains": np.ones((4, 3)).tolist(),
                        "linear_drag": [0.1, 0.2, 0.3],
                        "angular_drag": [0.01, 0.02, 0.03],
                    },
                )
            )
        self.store.estimation_run = _run(run_root)
        self.store._selected_sample_id = "chain-b:0001"
        self.store._selected_mode_id = "mode-a"

    def tearDown(self):
        self.temporary.cleanup()

    def test_launch_uses_sample_and_all_source_bags(self):
        view = NextExperimentView(self.store)
        view.baseline_combo.setCurrentText("bag-b")
        view.discrepancy_combo.setCurrentIndex(
            view.discrepancy_combo.findData("sample_model_discrepancy")
        )
        received = []
        view.evaluationRequested.connect(received.append)
        view.evaluate_button.click()
        self.assertEqual(len(received), 1)
        options = received[0]
        self.assertEqual(options.source_sample_id, "chain-b:0001")
        self.assertEqual(options.selected_mode_id, "mode-a")
        self.assertEqual({item[0] for item in options.bags}, {"bag-a", "bag-b"})
        self.assertEqual(options.model_discrepancy_policy, "sample_model_discrepancy")
        view.close()

    def test_new_artifact_fields_populate_metrics_and_sources(self):
        view = NextExperimentView(self.store)
        metrics = np.asarray(("position_rmse", "orientation_rmse"))
        evaluation = PidProposalEvaluation(
            root=self.root / "pid",
            manifest={
                "model_discrepancy_policy": "zero_model_discrepancy",
                "model_discrepancy_residual_quantity": "body_wrench",
                "model_discrepancy_replicates": 1,
                "plant_sample_ids": ["chain-a:0001", "chain-b:0001"],
                "bag_ids": ["bag-a", "bag-b"],
                "recommended_candidate_ids": [],
                "rejection_reason": "no candidate improves current",
            },
            source_samples={
                "sample_id": np.asarray(("chain-a:0001", "chain-b:0001")),
                "source_mode_id": np.asarray(("mode-a", "mode-a")),
                "delay": np.asarray((0.01, 0.02)),
                "mass": np.asarray((1.0, 1.1)),
                "cog": np.zeros((2, 3)),
            },
            candidate_particles={
                "candidate_id": np.asarray(("current", "sample_test")),
                "source": np.asarray(("current", "sample-derived")),
                "source_sample_id": np.asarray(("", "chain-b:0001")),
                "generation": np.asarray((0, 0)),
                "gain_values": np.ones((2, 4, 3)),
            },
            summary={
                "metric_names": metrics,
                "candidate_id": np.asarray(("current", "sample_test")),
                "mean": np.zeros((2, 2)),
                "upper_cvar": np.ones((2, 2)),
                "forecast_completion_mean": np.ones(2),
                "forecast_completion_lower_cvar": np.ones(2),
                "gain_change_magnitude": np.asarray((0.0, 0.1)),
                "nondominated_candidate_id": np.asarray(("current",)),
                "recommended_candidate_id": np.asarray((), dtype=str),
            },
            bags={},
            proposed_yaml=None,
            proposed_diff_yaml=None,
        )
        view.set_evaluation(evaluation)
        self.assertEqual(view.candidate_table.rowCount(), 2)
        self.assertEqual(view.source_table.rowCount(), 2)
        self.assertIn("Recommendation unavailable", view.recommendation_label.text())
        self.assertIn("never applies", view.yaml_safety_label.text())
        view.close()


if __name__ == "__main__":
    unittest.main()
