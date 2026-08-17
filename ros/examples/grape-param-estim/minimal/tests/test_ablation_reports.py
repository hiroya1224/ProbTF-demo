from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from _support import synthetic_problem_parts
from single_bag_savgol_ablation import (
    FIXED_CASE_NAMES,
    fixed_case_overrides,
    run_case_sequence,
    sweep_cases,
)
from single_bag_savgol_core import (
    COMMON_SCALE_DIRECTION,
    EstimationResult,
    SingleBagDynamicsProblem,
    acceleration_wrench_closure,
    covariance_weighting_diagnostics,
    estimation_diagnostics,
    ridge_analysis,
)
from rotor_lag import StrictZohCellGrid, local_strict_cell_descent
from single_bag_savgol_covariance import (
    parameter_covariances,
    residual_wrench_uncertainty,
    sum_mean_invariance,
)
from single_bag_savgol_reports import (
    WRENCH_LINE_STYLES,
    arrays_payload,
    output_run_directory,
    result_payload,
    write_completed_case,
)
from single_bag_wrench_replay import fit_external_wrench_replay
from single_bag_cross_bag_consensus import _pairwise_distance


def _synthetic_result(dataset, model, actuator):
    problem = SingleBagDynamicsProblem(dataset, model, actuator)
    evaluation = problem.evaluate_physical(np.zeros(14), 0.0)
    reference = problem.evaluate_physical(
        np.zeros(14), 0.0, reference=True
    )
    ridge = ridge_analysis(
        evaluation.jacobian_matrix,
        COMMON_SCALE_DIRECTION,
        evaluation.acceleration_jacobian.reshape(-1, 14),
    )
    residual_wrench = residual_wrench_uncertainty(
        raw_residual_wrench=evaluation.raw_residual_wrench,
        modeled_wrench=evaluation.modeled_wrench,
        required_wrench=evaluation.required_wrench,
        estimated_mass_kg=evaluation.parameters.mass,
        estimated_inertia_kg_m2=evaluation.parameters.inertia,
        fixed_mass_kg=model.parameters.mass,
        lever_arm_m=(
            dataset.pose_sensor_position_in_body
            - evaluation.parameters.cog_offset
        ),
        reference_sigma_z=dataset.reference_covariance.local_sigma_z,
    )
    uncertainty = parameter_covariances(
        evaluation.acceleration_jacobian,
        dataset.covariance,
        COMMON_SCALE_DIRECTION,
        residual_wrench.acceleration_model_discrepancy_covariance,
        evaluation.acceleration_residual,
    )
    diagnostics = estimation_diagnostics(
        problem=problem,
        final=evaluation,
        nominal_reference=reference,
        estimated_reference=reference,
        ridge=ridge,
        uncertainty=uncertainty,
        residual_wrench=residual_wrench,
        strict_lag={
            "mode": "zero",
            "data_support_lower_seconds": 0.0,
            "data_support_upper_seconds": dataset.rotor_lag_data_support_upper,
            "lag_reached_data_support_boundary": False,
            "final_smooth_rotor_lag_seconds": 0.0,
            "strict_lag_cell_lower_seconds": 0.0,
            "strict_lag_cell_upper_seconds": 0.1,
            "strict_lag_cell_representative_seconds": 0.0,
            "strict_lag_cell_width_seconds": 0.1,
            "candidate_table": [],
            "neighbor_cells": [],
        },
        continuation={
            "epsilon": np.empty(0),
            "rotor_lag_seconds": np.empty(0),
            "smooth_cost": np.empty(0),
            "strict_cost": np.empty(0),
            "absolute_cost_difference": np.empty(0),
            "command_max_error": np.empty(0),
            "rotor_lag_step": np.empty(0),
            "physical_step_norm": np.empty(0),
            "lag_jacobian_norm": np.empty(0),
            "physical_coordinate": np.empty((0, 14)),
        },
    )
    ridge.update(diagnostics["ridge"])
    return EstimationResult(
        physical_coordinate=np.zeros(14),
        rotor_lag_seconds=0.0,
        final_smooth_rotor_lag_seconds=0.0,
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
        residual_wrench_uncertainty=residual_wrench,
        diagnostics=diagnostics,
    )


class AblationReportTests(unittest.TestCase):
    def test_new_fixed_set_has_exactly_twenty_one_cases(self):
        self.assertEqual(len(FIXED_CASE_NAMES), 21)
        self.assertEqual(FIXED_CASE_NAMES[0], "default")
        self.assertIn("gimbal_command_replay", FIXED_CASE_NAMES)
        self.assertIn("lag_pow2_depth_12", FIXED_CASE_NAMES)
        self.assertNotIn("external_wrench_raw_only", FIXED_CASE_NAMES)

    def test_removed_numerical_jacobian_ablation(self):
        self.assertNotIn("jacobian_finite_difference", FIXED_CASE_NAMES)
        self.assertNotIn("jacobian_analytic", FIXED_CASE_NAMES)
        arguments = SimpleNamespace(kkt_scale_offsets=(-1.0, 0.0, 1.0))
        self.assertNotIn("jacobian_mode", fixed_case_overrides("naive_all", arguments))

    def test_focused_window_covariance_and_lag_seed_sweep_is_explicit(self):
        cases = sweep_cases(
            {
                "sg_window_covariance": [
                    {"window_seconds": 1.0, "covariance_mode": "full"},
                    {"window_seconds": 1.0, "covariance_mode": "identity"},
                ],
                "lag_initial_multipliers": [0.5],
            }
        )
        self.assertEqual(len(cases), 3)
        self.assertEqual(
            cases[0][1], {"sg_window": 1.0, "covariance_mode": "full"}
        )
        self.assertEqual(
            cases[-1][1],
            {"initial_rotor_lag": None, "initial_rotor_lag_multiplier": 0.5},
        )

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

    def test_cross_bag_distribution_distance_is_symmetric_and_nominal_free(self):
        coordinate = np.asarray(((0.0, 0.0), (1.0, -2.0), (0.5, 1.0)))
        covariance = np.stack((np.eye(2), 2.0 * np.eye(2), 0.5 * np.eye(2)))
        squared, distance = _pairwise_distance(coordinate, covariance)
        self.assertTrue(np.allclose(squared, squared.T))
        self.assertTrue(np.allclose(distance**2, squared))
        self.assertTrue(np.array_equal(np.diag(distance), np.zeros(3)))

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
        arrays = arrays_payload(dataset, result, replay)
        required = {
            "physical_coordinate",
            "sigma_z_eigenvalues",
            "whitening_gain",
            "mahalanobis_contribution_per_time",
            "whitened_physical_jacobian_singular_values",
            "parameter_displacement_ridge_coordinates",
            "uncertainty_variance_inflation_in_ridge_basis",
            "force_acceleration_closure_error",
            "lag_candidate_selected",
            "gimbal_raw_time",
            "lag_continuation_epsilon",
            "quotient_basis",
            "residual_wrench_nominal_mass_gauge",
            "residual_wrench_model_discrepancy_covariance",
            "residual_acceleration_model_discrepancy_covariance",
            "parameter_covariance_wrench_corrected",
            "parameter_sandwich_middle_total",
            "residual_wrench_uncentered_second_moment",
            "residual_acceleration_uncentered_second_moment",
            "residual_acceleration_recovered_from_nominal_mass_wrench",
            "residual_acceleration_recovery_error_from_nominal_mass_wrench",
            "parameter_sandwich_middle_residual_uncentered",
            "parameter_sandwich_middle_conservative_fusion",
            "parameter_covariance_conservative_fusion",
            "quotient_covariance_conservative_fusion",
            "nominal_mass_gauge_force_effectiveness_std_conservative_fusion",
        }
        self.assertTrue(required.issubset(arrays))
        payload = result_payload(
            case_name="synthetic",
            source_revision="abc123",
            model=model,
            result=result,
            replay=replay,
        )
        self.assertEqual(payload["ridge"]["dimension"], 14)
        self.assertIn("covariance", payload["diagnostics"])
        self.assertEqual(
            payload["diagnostics"]["overlap_correction"][
                "cross_time_covariance_model"
            ],
            "pairwise_mean_local_raw_pose_covariance",
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
            output = Path(temporary)
            write_completed_case(
                output,
                case_name="synthetic",
                source_revision="abc123",
                arguments={},
                dataset=dataset,
                model=model,
                result=result,
                replay=replay,
            )
            for name in (
                "report.pdf",
                "residual_wrench.pdf",
                "arrays.npz",
                "result.json",
            ):
                self.assertTrue(
                    (output / name).is_file()
                    and (output / name).stat().st_size > 0
                )
            with np.load(output / "arrays.npz") as archive:
                self.assertTrue(required.issubset(archive.files))
            result_json = (output / "result.json").read_text()
            self.assertIn('"residual_wrench"', result_json)
            self.assertIn(
                '"parameter_covariance_wrench_corrected"', result_json
            )
            self.assertIn(
                '"parameter_covariance_conservative_fusion"', result_json
            )
            self.assertIn('"conservative_fusion"', result_json)
            self.assertIn(
                '"sandwich_middle_residual_to_sg_trace_ratio"', result_json
            )
        self.assertNotEqual(
            WRENCH_LINE_STYLES["raw_sg_inverse_dynamics"],
            WRENCH_LINE_STYLES["trajectory_fitted_external"],
        )

    def test_covariance_mode_sum_and_ridge_basis_variance_consistency(self):
        dataset, model, actuator = synthetic_problem_parts()
        result = _synthetic_result(dataset, model, actuator)
        covariance = covariance_weighting_diagnostics(
            dataset.covariance, result.evaluation.acceleration_residual
        )
        self.assertTrue(
            np.allclose(
                np.sum(
                    covariance[
                        "covariance_eigenmode_mahalanobis_contribution"
                    ],
                    axis=1,
                ),
                covariance["mahalanobis_contribution_per_time"],
            )
        )
        right = result.ridge["whitened_right_singular_vectors"]
        expected = np.einsum(
            "ij,jk,ik->i", right, result.uncertainty.naive, right
        )
        self.assertTrue(
            np.allclose(
                expected,
                result.diagnostics["overlap_correction"][
                    "uncertainty_variance_naive_in_ridge_basis"
                ],
            )
        )
        closure = acceleration_wrench_closure(dataset, result.evaluation)
        self.assertLess(
            closure["force_acceleration_closure_error_max_abs"], 2e-13
        )
        self.assertLess(
            closure["torque_acceleration_closure_error_max_abs"], 2e-13
        )

    def test_exact_strict_cell_descent_selects_local_minimum(self):
        dataset, _model, _actuator = synthetic_problem_parts()
        grid = StrictZohCellGrid(
            dataset.time, dataset.rotor_history.times
        )
        target = min(3, grid.cell_count - 1)
        result = local_strict_cell_descent(
            grid,
            grid.cell(0).representative,
            lambda cell: ((cell.index - target) ** 2, cell.index),
        )
        self.assertEqual(result.selected.cell.index, target)
        self.assertEqual(result.selected.payload, target)


if __name__ == "__main__":
    unittest.main()
