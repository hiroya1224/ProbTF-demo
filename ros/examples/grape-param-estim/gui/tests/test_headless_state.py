from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim_gui.artifact_loader import (
    AssimilationRun,
    PidProposalEvaluation,
    SharedPosterior,
)
from grape_param_estim_gui.presentation import (
    member_parameter_text,
    scenario_assumption_text,
)
from grape_param_estim_gui.project_io import freshness_fingerprint, new_project_manifest
from grape_param_estim_gui.state import BagRecord, ProjectStore


def _posterior():
    second_inertia = 1.2 * np.eye(3)
    second_inertia[0, 1] = second_inertia[1, 0] = 0.031
    return SharedPosterior(
        member_id=np.array([11, 29]),
        parameter_coordinate=np.zeros((2, 19)),
        mass=np.array([1.9, 2.2]),
        inertia=np.stack((np.eye(3), second_inertia)),
        cog=np.zeros((2, 3)),
        force_effectiveness=np.ones((2, 4)),
        torque_effectiveness=np.ones((2, 4)),
        constant_delay=np.array([0.014, 0.037]),
        ridge={},
        mode={"selected_mode_id": np.array(["nominal"])},
        iteration_diagnostics={},
    )


class HeadlessProjectStateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        manifest = new_project_manifest("headless-state")
        manifest["bags"] = [
            {
                "bag_id": bag_id,
                "source_path": "/source/{}.bag".format(bag_id),
                "relative_path": "bags/{}.bag".format(bag_id),
                "sha256": digest * 64,
            }
            for bag_id, digest in (("bag-a", "a"), ("bag-b", "b"))
        ]
        self.store = ProjectStore(self.root, manifest)
        for bag_id, digest in (("bag-a", "a"), ("bag-b", "b")):
            self.store.add(
                BagRecord(
                    bag_id=bag_id,
                    path=self.root / "bags" / (bag_id + ".bag"),
                    source_path=Path("/source/{}.bag".format(bag_id)),
                    sha256=digest * 64,
                    included=True,
                    auto_interval=(1.0, 5.0),
                    selected_interval=(1.0, 5.0),
                    configuration_fingerprint="complete:shared",
                    controller_snapshot={"gains": [1.0]},
                )
            )

    def tearDown(self):
        self.temporary.cleanup()

    def test_bag_switching_and_interval_update(self):
        self.assertEqual(self.store.current_bag_id, "bag-a")
        self.store.set_current("bag-b")
        self.assertEqual(self.store.current_record().bag_id, "bag-b")
        self.store.update_interval("bag-b", (1.3, 4.6), "MODIFIED")
        self.assertEqual(self.store.get("bag-b").selected_range, (1.3, 4.6))
        self.assertEqual(self.store.get("bag-b").interval_state, "MODIFIED")
        self.store.update_interval("bag-b", (1.3, 4.6), "LOCKED")
        self.assertEqual(self.store.get("bag-b").interval_state, "LOCKED")
        self.store.restore_auto_interval("bag-b")
        self.assertEqual(self.store.get("bag-b").selected_range, (1.0, 5.0))
        self.assertEqual(self.store.get("bag-b").interval_state, "AUTO")

    def test_selected_member_and_pid_selection_are_shared_state(self):
        posterior = _posterior()
        project_fingerprint = freshness_fingerprint(self.store.manifest)
        run = AssimilationRun(
            root=self.root,
            manifest={
                "run_id": "run-a",
                "project_request_fingerprint": project_fingerprint,
            },
            shared_posterior=posterior,
            bag_results={},
            diagnostics={},
            warnings=(),
        )
        member_events = []
        pid_events = []
        self.store.selectedMemberChanged.connect(member_events.append)
        self.store.selectedPidProposalChanged.connect(pid_events.append)
        self.store.apply_assimilation(run)
        self.store.set_selected_member(29)
        self.assertEqual(self.store.selected_member_id, 29)
        self.assertEqual(member_events[-1], 29)

        evaluation = PidProposalEvaluation(
            root=self.root,
            manifest={"evaluation_id": "pid-a", "source_run_id": "run-a"},
            proposal_ensemble={},
            summary={"candidate_id": np.array(["current", "member-29"])},
            bags={},
            proposed_yaml="xy: {}\n",
            proposed_diff_yaml="{}\n",
        )
        self.store.apply_pid_evaluation(evaluation)
        self.store.set_selected_pid_proposal("member-29")
        self.assertEqual(self.store.selected_pid_proposal_id, "member-29")
        self.assertEqual(pid_events[-1], "member-29")

        replacement = AssimilationRun(
            root=self.root,
            manifest={
                "run_id": "run-b",
                "project_request_fingerprint": project_fingerprint,
            },
            shared_posterior=posterior,
            bag_results={},
            diagnostics={},
            warnings=(),
        )
        self.store.apply_assimilation(replacement)
        self.assertIsNone(self.store.pid_evaluation)
        self.assertIsNone(self.store.selected_pid_proposal_id)
        self.assertIsNone(
            self.store.manifest["current_pid_proposal_evaluation_id"]
        )

    def test_loaded_run_and_pid_evaluation_must_match_current_project(self):
        posterior = _posterior()
        mismatched_run = AssimilationRun(
            root=self.root,
            manifest={
                "run_id": "run-wrong",
                "project_request_fingerprint": "sha256:" + "0" * 64,
            },
            shared_posterior=posterior,
            bag_results={},
            diagnostics={},
            warnings=(),
        )
        with self.assertRaisesRegex(ValueError, "current project inputs"):
            self.store.apply_assimilation(mismatched_run)
        self.assertIsNone(self.store.assimilation_run)

        valid_run = AssimilationRun(
            root=self.root,
            manifest={
                "run_id": "run-current",
                "project_request_fingerprint": freshness_fingerprint(
                    self.store.manifest
                ),
            },
            shared_posterior=posterior,
            bag_results={},
            diagnostics={},
            warnings=(),
        )
        self.store.apply_assimilation(valid_run)
        wrong_evaluation = PidProposalEvaluation(
            root=self.root,
            manifest={
                "evaluation_id": "pid-wrong",
                "source_run_id": "run-other",
            },
            proposal_ensemble={},
            summary={"candidate_id": np.array(["current"])},
            bags={},
            proposed_yaml="",
            proposed_diff_yaml="",
        )
        with self.assertRaisesRegex(ValueError, "source_run_id"):
            self.store.apply_pid_evaluation(wrong_evaluation)
        self.assertIsNone(self.store.pid_evaluation)

    def test_selection_interval_fingerprint_and_settings_make_results_stale(self):
        self.store.manifest["run_request_fingerprint"] = freshness_fingerprint(
            self.store.manifest
        )
        self.store._refresh_stale()
        self.assertFalse(self.store.results_stale)
        self.store.set_included("bag-b", False)
        self.assertTrue(self.store.results_stale)

        self.store.set_included("bag-b", True)
        self.store.manifest["run_request_fingerprint"] = freshness_fingerprint(
            self.store.manifest
        )
        self.store._refresh_stale()
        self.store.update_interval("bag-a", (1.1, 4.9), "MODIFIED")
        self.assertTrue(self.store.results_stale)

        self.store.manifest["run_request_fingerprint"] = freshness_fingerprint(
            self.store.manifest
        )
        self.store._refresh_stale()
        self.store.set_estimator_settings({"ensemble_size": 64})
        self.assertTrue(self.store.results_stale)

    def test_tau_is_in_the_production_member_display_text(self):
        text = member_parameter_text(_posterior(), 29)
        self.assertIn("constant delay τ", text)
        self.assertIn("0.037 s", text)
        self.assertIn("full inertia", text)
        self.assertIn("0.031", text)

    def test_scenario_assumption_is_not_rewritten_or_duplicated(self):
        backend_text = (
            "same recorded reference; same posterior member initial state; "
            "residual policy by bag: bag-a=posterior_replay; this is not a "
            "forecast of a new disturbance realization"
        )
        rendered = scenario_assumption_text(backend_text)
        self.assertEqual(
            rendered, "Counterfactual assumption: " + backend_text
        )
        self.assertEqual(rendered.count("residual policy"), 1)

    def test_production_gui_has_no_generated_data_fallback_or_development_wording(self):
        source_root = Path(__file__).resolve().parents[1] / "src"
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(source_root.rglob("*.py"))
        ).lower()
        numbered_stage = "".join(("ph", "ase", " ", "5"))
        for forbidden in (
            "synthetic", "fakeestimationrunner", "visual mock", numbered_stage
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn(
            "this view does not apply gains automatically", source
        )


if __name__ == "__main__":
    unittest.main()
