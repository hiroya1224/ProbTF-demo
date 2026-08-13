from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from _support import synthetic_problem_parts
from single_bag_savgol_ablation import run_case_sequence
from single_bag_savgol_core import (
    COMMON_SCALE_DIRECTION,
    EstimationResult,
    SingleBagDynamicsProblem,
    ridge_analysis,
)
from single_bag_savgol_covariance import parameter_covariances, sum_mean_invariance
from single_bag_savgol_reports import (
    WRENCH_LINE_STYLES,
    output_run_directory,
    write_report_pdf,
)
from single_bag_wrench_replay import fit_external_wrench_replay


def _synthetic_result(dataset, model, actuator):
    problem = SingleBagDynamicsProblem(dataset, model, actuator)
    evaluation = problem.evaluate_physical(np.zeros(14), 0.0, 0.0)
    reference = problem.evaluate_physical(
        np.zeros(14), 0.0, 0.0, reference=True
    )
    ridge = ridge_analysis(evaluation.jacobian_matrix, COMMON_SCALE_DIRECTION)
    uncertainty = parameter_covariances(
        evaluation.acceleration_jacobian,
        dataset.covariance,
        COMMON_SCALE_DIRECTION,
    )
    return EstimationResult(
        physical_coordinate=np.zeros(14),
        rotor_lag_seconds=0.0,
        gimbal_lag_seconds=0.0,
        evaluation=evaluation,
        reference_evaluation=reference,
        success=True,
        status="completed",
        message="synthetic",
        stages=(),
        total_nfev=1,
        elapsed_seconds=0.0,
        ridge=ridge,
        uncertainty=uncertainty,
    )


class AblationReportTests(unittest.TestCase):
    def test_sum_mean_invariance_algebra(self):
        rng = np.random.default_rng(2)
        residual = rng.standard_normal((7, 6))
        jacobian = rng.standard_normal((7, 6, 4))
        result = sum_mean_invariance(residual, jacobian)
        self.assertTrue(
            np.isclose(result["loss_sum"], 7 * result["loss_mean"])
        )
        self.assertTrue(
            np.allclose(result["hessian_sum"], 7 * result["hessian_mean"])
        )

    def test_ablation_failure_continues_and_records_reason(self):
        executed = []

        def executor(name, _directory):
            executed.append(name)
            if name == "B":
                raise RuntimeError("deliberate failure")
            return {
                "status": "completed",
                "case_name": name,
                "elapsed_seconds": 0.0,
            }

        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            summaries = run_case_sequence(
                run_directory=tmp_path,
                case_names=("A", "B", "C"),
                source_revision="abc123",
                executor=executor,
            )
            self.assertEqual(executed, ["A", "B", "C"])
            self.assertEqual(
                [item["status"] for item in summaries],
                ["completed", "failed", "completed"],
            )
            failure = (tmp_path / "cases" / "B" / "result.json").read_text()
            self.assertIn("deliberate failure", failure)

    def test_output_commit_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            directory = output_run_directory(
                tmp_path, "default", "run-one", commit="deadbeef"
            )
            self.assertEqual(
                directory, tmp_path / "deadbeef" / "default" / "run-one"
            )

    def test_replay_isolation_and_report_smoke(self):
        dataset, model, actuator = synthetic_problem_parts()
        result = _synthetic_result(dataset, model, actuator)
        coordinate_before = result.physical_coordinate.copy()
        residual_before = result.evaluation.acceleration_residual.copy()
        replay = fit_external_wrench_replay(
            dataset=dataset, model=model, evaluation=result.evaluation
        )
        self.assertTrue(
            np.array_equal(result.physical_coordinate, coordinate_before)
        )
        self.assertTrue(
            np.array_equal(
                result.evaluation.acceleration_residual, residual_before
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.pdf"
            write_report_pdf(
                output,
                case_name="synthetic",
                dataset=dataset,
                model=model,
                result=result,
                replay=replay,
            )
            self.assertTrue(output.is_file() and output.stat().st_size > 0)
        self.assertNotEqual(
            WRENCH_LINE_STYLES["raw_sg_inverse_dynamics"],
            WRENCH_LINE_STYLES["trajectory_fitted_external"],
        )


if __name__ == "__main__":
    unittest.main()
