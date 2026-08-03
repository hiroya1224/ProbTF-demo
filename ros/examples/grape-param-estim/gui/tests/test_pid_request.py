from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from grape_param_estim.pid_evaluation_input import load_pid_evaluation_request
from grape_param_estim_gui.artifact_loader import AssimilationRun, SharedPosterior
from grape_param_estim_gui.pid_request import (
    PidEvaluationLaunchOptions,
    build_pid_evaluation_request,
)


def _run(root: Path) -> AssimilationRun:
    shared = SharedPosterior(
        member_id=np.asarray((11, 23), dtype=np.int64),
        parameter_coordinate=np.zeros((2, 1)),
        mass=np.ones(2),
        inertia=np.zeros((2, 3, 3)),
        cog=np.zeros((2, 3)),
        force_effectiveness=np.ones((2, 2)),
        torque_effectiveness=np.ones((2, 2)),
        constant_delay=np.asarray((0.01, 0.02)),
        ridge={},
        mode={"selected_mode_id": np.asarray(("nominal",))},
        iteration_diagnostics={},
    )
    return AssimilationRun(
        root=root,
        manifest={
            "schema": "grape-param-estim/assimilation-run/v1",
            "status": "complete",
            "run_id": "source-run",
            "project_request_fingerprint": "sha256:" + "a" * 64,
            "selected_bag_ids": ["bag-a", "bag-b"],
        },
        shared_posterior=shared,
        bag_results={},
        diagnostics={},
        warnings=(),
    )


class PidRequestBuilderTests(unittest.TestCase):
    def test_selected_member_builds_current_and_exact_member_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "assimilation_run"
            root.mkdir()
            payload = build_pid_evaluation_request(
                _run(root),
                "evaluation-a",
                PidEvaluationLaunchOptions(
                    source_member_id=23,
                    baseline_bag_id="bag-b",
                    residual_policy="zero",
                    cvar_level=0.85,
                ),
            )
            self.assertEqual(
                payload["candidates"],
                [
                    {"candidate_id": "current", "source": "current"},
                    {
                        "candidate_id": "member-23-exact",
                        "source": "member-derived",
                        "source_member_id": 23,
                    },
                ],
            )
            self.assertIsNone(payload["selected_candidate_id"])
            self.assertEqual(payload["residual_policy"], "zero")
            self.assertEqual(
                payload["thresholds"],
                {
                    "position": None,
                    "orientation": None,
                    "position_metric": "position_rmse",
                    "orientation_metric": "orientation_rmse",
                },
            )

            request_path = Path(directory) / "request.json"
            request_path.write_text(json.dumps(payload), encoding="utf-8")
            parsed = load_pid_evaluation_request(str(request_path))
            self.assertEqual(parsed.baseline_bag_id, "bag-b")
            self.assertEqual(parsed.candidates[1].source_member_id, 23)
            self.assertIsNone(parsed.selected_candidate_id)
            self.assertIsNone(parsed.thresholds.position)
            self.assertIsNone(parsed.thresholds.orientation)

    def test_exact_user_candidate_is_strict_and_can_be_selected_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "assimilation_run"
            root.mkdir()
            exact = tuple(
                tuple(float(10 * row + column + 1) for column in range(3))
                for row in range(4)
            )
            payload = build_pid_evaluation_request(
                _run(root),
                "evaluation-b",
                PidEvaluationLaunchOptions(
                    source_member_id=11,
                    baseline_bag_id="bag-a",
                    user_candidate_values=exact,
                    selected_candidate_source="user",
                ),
            )
            self.assertEqual(
                payload["selected_candidate_id"], "user-exact"
            )
            self.assertEqual(
                payload["candidates"][2],
                {
                    "candidate_id": "user-exact",
                    "source": "user",
                    "values": [list(row) for row in exact],
                },
            )
            request_path = Path(directory) / "request-user.json"
            request_path.write_text(json.dumps(payload), encoding="utf-8")
            parsed = load_pid_evaluation_request(str(request_path))
            np.testing.assert_array_equal(
                parsed.candidates[2].configuration.values, np.asarray(exact)
            )
            self.assertEqual(parsed.selected_candidate_id, "user-exact")

    def test_member_candidate_can_be_selected_without_an_automatic_representative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "assimilation_run"
            root.mkdir()
            payload = build_pid_evaluation_request(
                _run(root),
                "evaluation-member-selection",
                PidEvaluationLaunchOptions(
                    source_member_id=11,
                    baseline_bag_id="bag-a",
                    selected_candidate_source="member-derived",
                ),
            )
            self.assertEqual(
                payload["selected_candidate_id"], "member-11-exact"
            )

    def test_unknown_member_or_baseline_is_rejected_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "assimilation_run"
            root.mkdir()
            run = _run(root)
            with self.assertRaisesRegex(ValueError, "selected member"):
                build_pid_evaluation_request(
                    run,
                    "evaluation-c",
                    PidEvaluationLaunchOptions(99, "bag-a"),
                )
            with self.assertRaisesRegex(ValueError, "baseline_bag_id"):
                build_pid_evaluation_request(
                    run,
                    "evaluation-d",
                    PidEvaluationLaunchOptions(11, "bag-unknown"),
                )
            with self.assertRaisesRegex(ValueError, "finite non-negative 4x3"):
                PidEvaluationLaunchOptions(
                    11,
                    "bag-a",
                    user_candidate_values=((1.0, 2.0, 3.0),),
                )
            with self.assertRaisesRegex(ValueError, "must be included"):
                PidEvaluationLaunchOptions(
                    11,
                    "bag-a",
                    selected_candidate_source="user",
                )


if __name__ == "__main__":
    unittest.main()
