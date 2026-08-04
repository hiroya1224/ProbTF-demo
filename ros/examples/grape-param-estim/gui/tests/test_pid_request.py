from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.pid.request import validate_pid_evaluation_request
from grape_param_estim_gui.artifact_loader import (
    BatchEstimationRun,
    McmcPosterior,
    StaticParameterMap,
)
from grape_param_estim_gui.pid_request import (
    PidEvaluationLaunchOptions,
    build_pid_evaluation_request,
    sample_candidate_id,
)


def _run(root: Path) -> BatchEstimationRun:
    mcmc = McmcPosterior(
        sample_id=np.asarray(("chain-a:0001", "chain-b:0001")),
        chain_id=np.asarray(("chain-a", "chain-b")),
        draw_index=np.asarray((1, 1)),
        parameter_coordinate=np.zeros((2, 18)),
        mass=np.ones(2),
        inertia=np.repeat(np.eye(3)[None], 2, axis=0),
        cog=np.zeros((2, 3)),
        force_effectiveness=np.ones((2, 4)),
        torque_effectiveness=np.ones((2, 4)),
        delay=np.asarray((0.01, 0.02)),
        log_posterior=np.zeros(2),
        log_likelihood_approximation=np.zeros(2),
        log_determinant_term=np.zeros(2),
        accepted_kernel=np.asarray(("local", "ridge")),
        source_mode_id=np.asarray(("mode-a", "mode-a")),
    )
    static_map = StaticParameterMap(
        parameter_coordinate=np.zeros(18),
        mass=1.0,
        inertia=np.eye(3),
        cog=np.zeros(3),
        force_effectiveness=np.ones(4),
        torque_effectiveness=np.ones(4),
        delay=0.01,
        q_diagonal=np.ones(6),
        objective_components={},
        prior_objective=0.0,
        likelihood_objective=0.0,
        bag_objective={},
    )
    return BatchEstimationRun(
        root=root,
        manifest={
            "schema": "grape-param-estim/batch-estimation-run/v2",
            "status": "complete",
            "run_id": "source-run",
            "selected_bag_ids": ["bag-a", "bag-b"],
            "request_fingerprint": "sha256:" + "a" * 64,
        },
        static_map=static_map,
        q_em=None,  # type: ignore[arg-type]
        laplace=None,  # type: ignore[arg-type]
        diagnostics=None,  # type: ignore[arg-type]
        bags={},
        mcmc=mcmc,
        selected_trajectories={},
        warnings=(),
    )


class PidRequestBuilderTests(unittest.TestCase):
    def test_request_matches_backend_strict_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "estimation"
            run_root.mkdir()
            bag_a = root / "a.bag"
            bag_b = root / "b.bag"
            bag_a.write_bytes(b"a")
            bag_b.write_bytes(b"b")
            options = PidEvaluationLaunchOptions(
                source_sample_id="chain-b:0001",
                baseline_bag_id="bag-b",
                selected_mode_id="mode-a",
                bags=(
                    ("bag-a", str(bag_a), "1" * 64, True),
                    ("bag-b", str(bag_b), "2" * 64, False),
                ),
                fixed_linear_drag=(0.1, 0.2, 0.3),
                fixed_angular_drag=(0.01, 0.02, 0.03),
                model_discrepancy_policy="sample_model_discrepancy",
                base_seed=7,
                replicates=3,
                selected_candidate_source="sample-derived",
            )
            payload = build_pid_evaluation_request(
                _run(run_root), "evaluation-a", root / "output", options
            )
            parsed = validate_pid_evaluation_request(payload)
            self.assertEqual(parsed.evaluation_id, "evaluation-a")
            self.assertEqual(parsed.discrepancy_policy, "sample_model_discrepancy")
            self.assertEqual(parsed.plant_sample_subset_method, "all_equal_weight_mcmc_samples")
            self.assertEqual(len(parsed.candidates), 1)
            self.assertEqual(parsed.candidates[0].source, "current")
            self.assertEqual(parsed.derived_candidate_method, "deterministic_k_medoids")
            self.assertEqual(parsed.maximum_derived_candidates, 12)
            self.assertEqual(parsed.required_derived_sample_ids, ("chain-b:0001",))
            self.assertEqual(parsed.selected_candidate_id, sample_candidate_id("chain-b:0001"))
            self.assertEqual(tuple(parsed.fixed_linear_drag), (0.1, 0.2, 0.3))
            self.assertEqual(parsed.forecast_workers, "auto")

    def test_forecast_worker_count_is_explicit_and_bounded(self):
        common = dict(
            source_sample_id="sample-a",
            baseline_bag_id="bag-a",
            selected_mode_id="mode-a",
            bags=(("bag-a", __file__, "1" * 64, True),),
            fixed_linear_drag=(0.0, 0.0, 0.0),
            fixed_angular_drag=(0.0, 0.0, 0.0),
            model_discrepancy_policy="zero_model_discrepancy",
        )
        self.assertEqual(
            PidEvaluationLaunchOptions(forecast_workers=4, **common).forecast_workers,
            4,
        )
        for value in (0, 33, "automatic", True):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "forecast_workers"
            ):
                PidEvaluationLaunchOptions(forecast_workers=value, **common)
        self.assertIsNone(
            PidEvaluationLaunchOptions(
                maximum_derived_candidates=None, **common
            ).maximum_derived_candidates
        )
        for value in (0, -1, True):
            with self.subTest(maximum_candidates=value), self.assertRaisesRegex(
                ValueError, "maximum_derived_candidates"
            ):
                PidEvaluationLaunchOptions(
                    maximum_derived_candidates=value, **common
                )

    def test_unknown_sample_and_mode_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "estimation"
            run_root.mkdir()
            bag_a = root / "a.bag"
            bag_b = root / "b.bag"
            bag_a.write_bytes(b"a")
            bag_b.write_bytes(b"b")
            common = dict(
                baseline_bag_id="bag-a",
                selected_mode_id="mode-a",
                bags=(("bag-a", str(bag_a), "1" * 64, True), ("bag-b", str(bag_b), "2" * 64, True)),
                fixed_linear_drag=(0.0, 0.0, 0.0),
                fixed_angular_drag=(0.0, 0.0, 0.0),
                model_discrepancy_policy="zero_model_discrepancy",
            )
            with self.assertRaisesRegex(ValueError, "selected sample"):
                build_pid_evaluation_request(
                    _run(run_root),
                    "evaluation-b",
                    root / "out-b",
                    PidEvaluationLaunchOptions(source_sample_id="missing", **common),
                )
            with self.assertRaisesRegex(ValueError, "selected_mode_id"):
                bad = dict(common)
                bad["selected_mode_id"] = "mode-b"
                build_pid_evaluation_request(
                    _run(run_root),
                    "evaluation-c",
                    root / "out-c",
                    PidEvaluationLaunchOptions(source_sample_id="chain-a:0001", **bad),
                )


if __name__ == "__main__":
    unittest.main()
