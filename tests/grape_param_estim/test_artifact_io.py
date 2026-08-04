import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.artifact_io import (
    ASSIMILATION_RUN_SCHEMA,
    FLIGHT_INSPECTION_SCHEMA,
    INSPECTION_BUNDLE_SCHEMA,
    PID_PROPOSAL_EVALUATION_SCHEMA,
    ArtifactValidationError,
    IncompleteArtifactError,
    UnsupportedArtifactSchema,
    begin_bundle,
    load_assimilation_run,
    load_bundle,
    load_inspection_bundle,
    load_npz_strict,
    load_pid_proposal_evaluation,
    mark_bundle_cancelled,
    mark_bundle_complete,
    read_manifest,
    request_fingerprint,
)


class ArtifactIoTests(unittest.TestCase):
    def _save(self, path, **arrays):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **arrays)

    def _run_manifest(self, converged=False, termination="maximum_iterations"):
        return {
            "schema": ASSIMILATION_RUN_SCHEMA,
            "run_id": "run-a",
            "created_at": "2026-08-04T12:00:00+09:00",
            "estimator_revision": "test-revision",
            "request_path": "requests/run-a.json",
            "request_fingerprint": "sha256:test",
            "project_request_fingerprint": "sha256:" + "a" * 64,
            "selected_bag_ids": ["bag-a", "bag-b"],
            "selected_intervals": {
                "bag-a": [1.0, 2.0],
                "bag-b": [3.0, 5.0],
            },
            "configuration_fingerprint": "vehicle-a",
            "shared_member_count": 3,
            "termination_reason": termination,
            "converged": converged,
            "artifacts": {
                "shared_posterior": "shared_posterior.npz",
                "diagnostics": "diagnostics.npz",
                "bags": {
                    "bag-a": "bags/bag-a.npz",
                    "bag-b": "bags/bag-b.npz",
                },
            },
        }

    def _shared_arrays(self):
        members = np.asarray((41, 7, 99), dtype=np.int64)
        physical = np.zeros((3, 18))
        delay_coordinate = np.asarray((0.001, 0.007, 0.012))
        coordinates = np.column_stack((physical, delay_coordinate))
        covariance = np.cov(coordinates, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = np.zeros(19)
        direction[18] = 1.0
        return {
            "member_id": members,
            "parameter_coordinates": coordinates,
            "physical_parameter_coordinates": physical,
            "constant_delay_coordinate": delay_coordinate,
            "mass": np.asarray((2.3, 2.4, 2.5)),
            "inertia": np.repeat(np.eye(3)[None, :, :], 3, axis=0),
            "cog": np.zeros((3, 3)),
            "force_effectiveness": np.ones((3, 4)),
            "torque_effectiveness": np.ones((3, 4)),
            "constant_delay": delay_coordinate,
            "ridge_covariance": covariance,
            "ridge_eigenvalues": eigenvalues,
            "ridge_eigenvectors": eigenvectors,
            "ridge_expected_direction": direction,
            "ridge_expected_variance": np.asarray((covariance[18, 18],)),
            "mode_id": np.asarray(("nominal",)),
            "mode_weight": np.asarray((1.0,)),
            "selected_mode_id": np.asarray(("nominal",)),
        }

    def _run_bag_arrays(self, member_id=None, q_sufficient=True):
        members = (
            np.asarray((41, 7, 99), dtype=np.int64)
            if member_id is None
            else np.asarray(member_id, dtype=np.int64)
        )
        count = 4
        times = np.asarray((0.0, 0.1, 0.22, 0.4))
        position = np.zeros((count, 3))
        quaternion = np.zeros((count, 4))
        quaternion[:, 3] = 1.0
        member_position = np.zeros((members.size, count, 3))
        member_quaternion = np.zeros((members.size, count, 4))
        member_quaternion[:, :, 3] = 1.0
        zeros3 = np.zeros((count, 3))
        zeros4 = np.zeros((count, 4))
        zeros4[:, 3] = 1.0
        member_zeros3 = np.zeros((members.size, count, 3))
        member_zeros4 = np.zeros((members.size, count, 4))
        member_zeros4[:, :, 3] = 1.0
        pid_fields = np.asarray(
            (
                "p_gain",
                "i_gain",
                "d_gain",
                "limit_sum",
                "limit_p",
                "limit_i",
                "limit_d",
                "limit_error_p",
                "limit_error_i",
                "limit_error_d",
            )
        )
        return {
            "member_id": members,
            "times": times,
            "record_times": 100.0 + times,
            "observed_position": position,
            "observed_orientation_xyzw": quaternion,
            "observation_translation_covariance": np.eye(3) * 0.01,
            "observation_rotation_covariance": np.eye(3) * 0.02,
            "reference_position": position,
            "reference_linear_velocity": zeros3,
            "reference_linear_acceleration": zeros3,
            "reference_rpy": position,
            "reference_angular_velocity": zeros3,
            "reference_angular_acceleration": zeros3,
            "nominal_position": position,
            "nominal_orientation_xyzw": quaternion,
            "nominal_linear_velocity": zeros3,
            "nominal_angular_velocity": zeros3,
            "nominal_controller_integral": np.zeros((count, 6)),
            "nominal_commanded_thrust": np.ones((count, 4)),
            "nominal_commanded_gimbal_angle": np.zeros((count, 4)),
            "nominal_actuator_thrust": np.ones((count, 4)),
            "nominal_actuator_gimbal_angle": np.zeros((count, 4)),
            "nominal_body_wrench": np.zeros((count, 6)),
            "posterior_position": member_position,
            "posterior_orientation_xyzw": member_quaternion,
            "posterior_linear_velocity": member_zeros3,
            "posterior_angular_velocity": member_zeros3,
            "posterior_controller_integral": np.zeros(
                (members.size, count, 6)
            ),
            "posterior_commanded_thrust": np.ones(
                (members.size, count, 4)
            ),
            "posterior_commanded_gimbal_angle": np.zeros(
                (members.size, count, 4)
            ),
            "posterior_actuator_thrust": np.ones(
                (members.size, count, 4)
            ),
            "posterior_actuator_gimbal_angle": np.zeros(
                (members.size, count, 4)
            ),
            "posterior_body_wrench": np.zeros(
                (members.size, count, 6)
            ),
            "correction_translation": member_position,
            "correction_rotation_vector": member_position,
            "observed_correction_translation": position,
            "observed_correction_rotation_vector": position,
            "residual_wrench_interval": np.zeros(
                (members.size, count - 1, 6)
            ),
            "residual_wrench_knot": np.zeros((members.size, 2, 6)),
            "innovation_ensemble": np.zeros((members.size, 12)),
            "objective_contribution": np.arange(members.size, dtype=float),
            "pose_component_coverage": np.asarray((0.8,)),
            "initial_position": np.zeros((members.size, 3)),
            "initial_orientation_xyzw": np.tile(
                (0.0, 0.0, 0.0, 1.0), (members.size, 1)
            ),
            "initial_linear_velocity": np.zeros((members.size, 3)),
            "initial_angular_velocity": np.zeros((members.size, 3)),
            "initial_controller_integral": np.zeros((members.size, 6)),
            "initial_controller_roll_pitch_integration_active": np.ones(
                members.size, dtype=bool
            ),
            "initial_actuator_thrust": np.ones((members.size, 4)),
            "initial_actuator_gimbal_angle": np.zeros((members.size, 4)),
            "actuator_thrust_time_constant": np.asarray((0.0,)),
            "actuator_gimbal_time_constant": np.asarray((0.0,)),
            "actuator_minimum_thrust": np.asarray((0.0,)),
            "actuator_maximum_thrust": np.asarray((30.0,)),
            "actuator_maximum_gimbal_angle": np.asarray((3.14,)),
            "actuator_maximum_gimbal_rate": np.asarray((6.0,)),
            "q_knot_indices": np.asarray((0, count - 1), dtype=np.int64),
            "q_knot_times": np.asarray((times[0], times[-1])),
            "q_stationary_standard_deviation": np.ones(6),
            "q_correlation_time": np.asarray((0.5,)),
            "q_resolution_sufficient": np.asarray((q_sufficient,), dtype=bool),
            "controller_snapshot_groups": np.asarray(
                ("xy", "z", "roll_pitch", "yaw")
            ),
            "controller_snapshot_record_times": np.arange(4, dtype=float),
            "controller_snapshot_gains": np.ones((4, 3)),
            "controller_snapshot_pid_control_flags": np.ones(4, dtype=bool),
            "controller_snapshot_source_kinds": np.asarray(
                ("startup", "startup", "startup", "startup")
            ),
            "controller_pid_axis_names": np.asarray(
                ("x", "y", "z", "roll", "pitch", "yaw")
            ),
            "controller_pid_field_names": pid_fields,
            "controller_pid_configuration": np.ones((6, pid_fields.size)),
            "controller_xy_control_mode": np.asarray(("position",)),
            "controller_need_yaw_d_control": np.asarray((True,), dtype=bool),
            "controller_start_roll_pitch_integration_height": np.asarray(
                (0.01,)
            ),
            "controller_initial_height": np.asarray((0.0,)),
            "controller_source_compatible_gyro_term": np.asarray(
                (True,), dtype=bool
            ),
            "provenance_bag_path": np.asarray(("/archive/a.bag",)),
            "provenance_bag_sha256": np.asarray(("abc123",)),
            "provenance_bag_size_bytes": np.asarray((123,), dtype=np.int64),
            "provenance_time_basis": np.asarray(("record",)),
            "provenance_requested_window": np.asarray((1.0, 2.0)),
            "provenance_source_available_window": np.asarray((0.0, 3.0)),
            "provenance_selected_flight_state": np.asarray((5,), dtype=np.int64),
            "provenance_topic_names": np.asarray(("/odom",)),
            "provenance_topic_types": np.asarray(("nav_msgs/Odometry",)),
        }

    def _diagnostic_arrays(self, converged=False, termination="maximum_iterations"):
        return {
            "iteration": np.asarray((0, 1), dtype=np.int64),
            "objective": np.asarray((2.0, 1.5)),
            "accepted_objective": np.asarray((1.5, 1.25)),
            "gradient_norm": np.asarray((1.0, 0.5)),
            "step_norm": np.asarray((0.5, 0.25)),
            "accepted_fraction": np.asarray((1.0, 0.5)),
            "ensemble_rank": np.asarray((2,), dtype=np.int64),
            "converged": np.asarray((converged,), dtype=bool),
            "termination_reason": np.asarray((termination,)),
        }

    def _initial_prior_diagnostic_arrays(self):
        requested = np.asarray(((-1.0, 0.0), (1.0, -1.0), (0.0, 1.0)))
        effective = 0.5 * requested
        values = self._diagnostic_arrays()
        values.update(
            {
                "initial_prior_member_id": np.asarray(
                    (41, 7, 99), dtype=np.int64
                ),
                "requested_prior_control_ensemble": requested,
                "effective_prior_control_ensemble": effective,
                "initial_prior_radial_scale": np.asarray((0.5,)),
                "initial_prior_backoff_trials": np.asarray(
                    (1,), dtype=np.int64
                ),
                "initial_prior_maximum_backoff_trials": np.asarray(
                    (8,), dtype=np.int64
                ),
                "initial_prior_requested_rank": np.asarray(
                    (2,), dtype=np.int64
                ),
                "initial_prior_effective_rank": np.asarray(
                    (2,), dtype=np.int64
                ),
                "initial_prior_failed_scale": np.asarray((1.0,)),
                "initial_prior_failure_type": np.asarray(("ValueError",)),
                "initial_prior_failure_reason": np.asarray(
                    ("synthetic divergence",)
                ),
            }
        )
        return values

    def _initial_prior_manifest(self):
        return {
            "strategy": "global_radial_dyadic_backoff",
            "radial_scale": 0.5,
            "backoff_trials": 1,
            "maximum_backoff_trials": 8,
            "requested_member_count": 3,
            "effective_member_count": 3,
            "requested_rank": 2,
            "effective_rank": 2,
            "failed_attempts": [
                {
                    "radial_scale": 1.0,
                    "exception_type": "ValueError",
                    "reason": "synthetic divergence",
                }
            ],
            "effective_prior_source": (
                "diagnostics.npz:effective_prior_control_ensemble"
            ),
        }

    def _prepare_run(self, root, second_member_id=None, shared=None):
        begin_bundle(root, self._run_manifest())
        shared_arrays = self._shared_arrays() if shared is None else shared
        self._save(root / "shared_posterior.npz", **shared_arrays)
        self._save(root / "diagnostics.npz", **self._diagnostic_arrays())
        self._save(root / "bags" / "bag-a.npz", **self._run_bag_arrays())
        self._save(
            root / "bags" / "bag-b.npz",
            **self._run_bag_arrays(
                member_id=second_member_id, q_sufficient=False
            ),
        )

    def _inspection_manifest(self):
        return {
            "schema": INSPECTION_BUNDLE_SCHEMA,
            "bag_ids": ["bag-a"],
            "artifacts": {
                "bags": {
                    "bag-a": {
                        "inspection": "bags/bag-a.inspection.json",
                        "preview": "bags/bag-a.preview.npz",
                    }
                }
            },
        }

    def _prepare_inspection(self, root):
        begin_bundle(root, self._inspection_manifest())
        bags = root / "bags"
        bags.mkdir(parents=True)
        inspection = {
            "schema": FLIGHT_INSPECTION_SCHEMA,
            "bag_id": "bag-a",
            "bag_path": "/archive/bags/a.bag",
            "bag_size": 123,
            "bag_mtime": 1.5,
            "bag_sha256": "a" * 64,
            "record_time_start": 100.0,
            "record_time_end": 110.0,
            "topic_contract": [],
            "complete_episodes": [
                {
                    "episode_index": 0,
                    "start_local_time": 1.0,
                    "end_local_time": 9.0,
                    "state_intervals": [],
                }
            ],
            "state5_intervals": [],
            "recommended_interval": {
                "episode_index": 0,
                "reason": "preferred_state",
                "warnings": [],
                "interval": {
                    "start_local_time": 2.0,
                    "end_local_time": 8.0,
                },
            },
            "warnings": [],
            "controller_snapshot": {},
            "controller_flags": {},
            "configuration_fingerprint": {
                "value": "complete:vehicle-a",
                "complete": True,
                "missing_components": [],
            },
            "estimated_work_units": {
                "sample_count": 151,
                "knot_count": 151,
                "lag_profile_point_units": 42,
                "nonlinear_iteration_units": 1260,
                "mcmc_proposal_units": 0,
                "estimate_kind": (
                    "upper_bound_excluding_lm_retries_and_q_backtracking"
                ),
            },
            "status": "ready",
        }
        (bags / "bag-a.inspection.json").write_text(
            json.dumps(inspection), encoding="utf-8"
        )
        time = np.asarray((0.0, 0.1, 0.2))
        self._save(
            bags / "bag-a.preview.npz",
            time=time,
            position=np.zeros((3, 3)),
            orientation_xyzw=np.tile((0.0, 0.0, 0.0, 1.0), (3, 1)),
            reference_position=np.zeros((3, 3)),
            reference_rpy=np.zeros((3, 3)),
            flight_state=np.asarray((3, 5, 6), dtype=np.int32),
        )

    def _pid_manifest(self):
        return {
            "schema": PID_PROPOSAL_EVALUATION_SCHEMA,
            "evaluation_id": "pid-a",
            "source_run_id": "run-a",
            "created_at": "2026-08-04T12:30:00+09:00",
            "selected_bag_ids": ["bag-a"],
            "artifacts": {
                "proposal_ensemble": "proposal_ensemble.npz",
                "summary": "summary.npz",
                "proposed_yaml": "proposed_GimbalrotorControl.yaml",
                "proposed_diff_yaml": (
                    "proposed_GimbalrotorControl.diff.yaml"
                ),
                "bags": {"bag-a": "bags/bag-a.npz"},
            },
        }

    def _prepare_pid(self, root, inconsistent_failure=False, mass_key=False):
        begin_bundle(root, self._pid_manifest())
        member_id = np.asarray((41, 7), dtype=np.int64)
        group_scales = np.asarray(
            ((0.9, 0.95, 0.92, 0.98), (1.1, 1.05, 1.08, 1.02))
        )
        current_pid = np.ones((4, 3))
        proposed_pid = current_pid[None, :, :] * group_scales[:, :, None]
        proposal = {
            "source_run_id": np.asarray(("run-a",)),
            "source_member_id": member_id,
            "proposal_source_member_id": member_id.copy(),
            "source_mode_id": np.asarray(("nominal", "nominal")),
            "xy_scale": group_scales[:, 0],
            "z_scale": group_scales[:, 1],
            "roll_pitch_scale": group_scales[:, 2],
            "yaw_scale": group_scales[:, 3],
            "proposed_pid": proposed_pid,
            "current_pid": current_pid,
            "constant_delay": np.asarray((0.01, 0.02)),
            "acceleration_response": np.tile(np.eye(6), (2, 1, 1)),
            "proposal_range_50": np.percentile(
                proposed_pid, (25.0, 75.0), axis=0
            ),
            "proposal_range_95": np.percentile(
                proposed_pid, (2.5, 97.5), axis=0
            ),
        }
        if mass_key:
            proposal["controller_mass_scale"] = np.ones(2)
        self._save(root / "proposal_ensemble.npz", **proposal)
        candidate_id = np.asarray(("current", "member-41"))
        (root / "proposed_GimbalrotorControl.yaml").write_text(
            "xy: {}\n", encoding="utf-8"
        )
        (root / "proposed_GimbalrotorControl.diff.yaml").write_text(
            "xy: {}\n", encoding="utf-8"
        )
        success = np.asarray(((True, False), (True, True)), dtype=bool)
        reason = np.asarray((("", "unstable"), ("", "")))
        shape3 = (2, 2, 3, 3)
        shape4 = (2, 2, 3, 4)
        position = np.zeros(shape3)
        orientation = np.zeros(shape4)
        orientation[:, :, :, 3] = 1.0
        correction_t = np.zeros(shape3)
        correction_r = np.zeros(shape3)
        for value in (position, orientation, correction_t, correction_r):
            value[0, 1] = np.nan
        if inconsistent_failure:
            correction_t[0, 1] = 0.0
        position_error = position.copy()
        orientation_error = np.zeros(shape3)
        orientation_error[0, 1] = np.nan
        metric_values = np.asarray(((0.0, np.nan), (0.0, 0.0)))
        proposed_candidates = np.asarray((current_pid, proposed_pid[0]))
        summary = {
            "source_run_id": np.asarray(("run-a",)),
            "source_member_id": member_id,
            "source_mode_id": np.asarray(("nominal", "nominal")),
            "bag_id": np.asarray(("bag-a",)),
            "candidate_id": candidate_id,
            "candidate_source": np.asarray(("current", "member-derived")),
            "candidate_source_member_id": np.asarray((-1, 41)),
            "candidate_source_mode_id": np.asarray(("", "nominal")),
            "current_pid": current_pid,
            "current_pid_baseline_bag_id": np.asarray(("bag-a",)),
            "current_pid_snapshot_group": np.asarray(
                ("xy", "z", "roll_pitch", "yaw")
            ),
            "current_pid_snapshot_topic": np.asarray(
                ("/xy", "/z", "/roll_pitch", "/yaw")
            ),
            "current_pid_snapshot_record_time": np.asarray(
                (10.0, 10.3, 10.2, 10.1)
            ),
            "current_pid_snapshot_source_kind": np.asarray(
                ("recorded",) * 4
            ),
            "proposed_pid": proposed_candidates,
            "difference": proposed_candidates - current_pid[None, :, :],
            "ratio": proposed_candidates / current_pid[None, :, :],
            "ratio_configured": np.ones((2, 4, 3), dtype=bool),
            "member_bag_forecast_completion": success[:, None, :],
            "member_bag_failure_reason": reason[:, None, :],
            "member_bag_position_threshold_exceeded": np.full(
                (2, 1, 2), np.nan
            ),
            "member_bag_orientation_threshold_exceeded": np.full(
                (2, 1, 2), np.nan
            ),
            "position_threshold": np.asarray((np.nan,)),
            "orientation_threshold": np.asarray((np.nan,)),
            "position_threshold_configured": np.asarray((False,)),
            "orientation_threshold_configured": np.asarray((False,)),
            "position_threshold_metric": np.asarray(("position_rmse",)),
            "orientation_threshold_metric": np.asarray(
                ("orientation_rmse",)
            ),
            "cvar_level": np.asarray((0.9,)),
            "correction_coverage_interval": np.asarray((0.95,)),
            "log_gain_change": np.asarray(
                (
                    0.0,
                    np.linalg.norm(np.log(proposed_candidates[1].ravel())),
                )
            ),
            "forecast_completion": np.asarray((0.5, 1.0)),
            "numerical_failure_count": np.asarray((1, 0)),
            "per_bag_forecast_completion": np.asarray(((0.5,), (1.0,))),
            "per_bag_numerical_failure_count": np.asarray(((1,), (0,))),
            "per_bag_position_threshold_exceedance": np.full((2, 1), np.nan),
            "per_bag_orientation_threshold_exceedance": np.full(
                (2, 1), np.nan
            ),
            "aggregate_position_threshold_exceedance": np.full(2, np.nan),
            "aggregate_orientation_threshold_exceedance": np.full(2, np.nan),
            "pareto_dominated": np.asarray((False, True)),
            "pareto_non_dominated": np.asarray((True, False)),
            "improves_current": np.asarray((False, False)),
            "candidate_eligible": np.asarray((False, False)),
            "candidate_rejection_reason": np.asarray(
                ("current baseline", "Pareto dominated")
            ),
            "selected_candidate_id": np.asarray(("",)),
            "recommendation_available": np.asarray((False,)),
            "recommended_candidate_id": np.asarray(("",)),
            "rejection_reason": np.asarray(("no explicit selection",)),
            "improvement_rule": np.asarray(("componentwise",)),
            "scenario_assumption": np.asarray(("posterior replay",)),
            "per_bag_correction_translation_zero_coverage": np.ones((2, 1)),
            "per_bag_correction_rotation_zero_coverage": np.ones((2, 1)),
            "per_bag_correction_transform_zero_coverage": np.ones((2, 1)),
        }
        for metric in (
            "position_rmse",
            "orientation_rmse",
            "maximum_position_error",
            "maximum_orientation_error",
        ):
            summary["member_bag_{}".format(metric)] = metric_values[:, None, :]
            summary["per_bag_{}_mean".format(metric)] = np.zeros((2, 1))
            summary["per_bag_{}_upper_cvar".format(metric)] = np.zeros((2, 1))
            summary["aggregate_{}_mean".format(metric)] = np.zeros(2)
            summary["aggregate_{}_upper_cvar".format(metric)] = np.zeros(2)
        self._save(root / "summary.npz", **summary)
        self._save(
            root / "bags" / "bag-a.npz",
            member_id=member_id,
            candidate_id=candidate_id,
            times=np.asarray((0.0, 0.1, 0.3)),
            reference_position=np.zeros((3, 3)),
            reference_rpy=np.zeros((3, 3)),
            prediction_position=position,
            prediction_orientation_xyzw=orientation,
            correction_translation=correction_t,
            correction_rotation_vector=correction_r,
            position_error=position_error,
            orientation_error_rotation_vector=orientation_error,
            position_rmse=metric_values,
            orientation_rmse=metric_values,
            maximum_position_error=metric_values,
            maximum_orientation_error=metric_values,
            forecast_success=success,
            forecast_failure_reason=reason,
            residual_policy=np.asarray(
                ("posterior_replay", "posterior_replay")
            ),
            correction_coverage_interval=np.asarray((0.95,)),
            correction_translation_zero_coverage=np.ones(2),
            correction_rotation_zero_coverage=np.ones(2),
            correction_transform_zero_coverage=np.ones(2),
        )

    def test_request_fingerprint_is_canonical_and_finite(self):
        first = {
            "selected": ["bag-a", "bag-b"],
            "settings": {"iterations": 2, "seed": 4},
        }
        second = {
            "settings": {"seed": 4, "iterations": 2},
            "selected": ["bag-a", "bag-b"],
        }
        self.assertEqual(request_fingerprint(first), request_fingerprint(second))
        self.assertNotEqual(
            request_fingerprint(first),
            request_fingerprint({**first, "selected": ["bag-b", "bag-a"]}),
        )
        with self.assertRaises(ArtifactValidationError):
            request_fingerprint({"bad": float("nan")})

    def test_run_manifest_requires_project_request_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "missing-project-fingerprint"
            self._prepare_run(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["project_request_fingerprint"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                ArtifactValidationError, "project_request_fingerprint"
            ):
                mark_bundle_complete(root)

    def test_complete_run_loads_member_aligned_laws_and_q_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "assimilation_run"
            self._prepare_run(root)
            with self.assertRaises(IncompleteArtifactError):
                load_assimilation_run(root)
            mark_bundle_complete(root)
            bundle = load_assimilation_run(root)
            self.assertEqual(bundle.manifest["status"], "complete")
            np.testing.assert_array_equal(
                bundle.shared_posterior["member_id"], (41, 7, 99)
            )
            np.testing.assert_array_equal(
                bundle.bags["bag-b"]["member_id"], (41, 7, 99)
            )
            self.assertEqual(
                bundle.warnings, ("bag-b: Q resolution is insufficient",)
            )

    def test_initial_prior_backoff_is_strictly_audited_and_warned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "initial-prior-audit"
            manifest = self._run_manifest()
            manifest["initial_prior_forecast"] = (
                self._initial_prior_manifest()
            )
            begin_bundle(root, manifest)
            self._save(root / "shared_posterior.npz", **self._shared_arrays())
            self._save(
                root / "diagnostics.npz",
                **self._initial_prior_diagnostic_arrays(),
            )
            self._save(root / "bags" / "bag-a.npz", **self._run_bag_arrays())
            self._save(root / "bags" / "bag-b.npz", **self._run_bag_arrays())
            mark_bundle_complete(root)
            bundle = load_assimilation_run(root)
            self.assertIn("radial scale 0.5", bundle.warnings[0])
            self.assertIn("center, shape, and rank", bundle.warnings[0])
            np.testing.assert_array_equal(
                bundle.diagnostics["initial_prior_member_id"],
                (41, 7, 99),
            )

            broken = Path(directory) / "broken-initial-prior"
            begin_bundle(broken, manifest)
            self._save(
                broken / "shared_posterior.npz", **self._shared_arrays()
            )
            diagnostics = self._initial_prior_diagnostic_arrays()
            diagnostics["effective_prior_control_ensemble"] = diagnostics[
                "effective_prior_control_ensemble"
            ].copy()
            diagnostics["effective_prior_control_ensemble"][0, 0] += 0.01
            self._save(broken / "diagnostics.npz", **diagnostics)
            self._save(
                broken / "bags" / "bag-a.npz", **self._run_bag_arrays()
            )
            self._save(
                broken / "bags" / "bag-b.npz", **self._run_bag_arrays()
            )
            with self.assertRaisesRegex(
                ArtifactValidationError, "global radial transform"
            ):
                mark_bundle_complete(broken)

    def test_alignment_or_missing_tau_prevents_atomic_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "misaligned"
            self._prepare_run(root, second_member_id=(7, 41, 99))
            with self.assertRaisesRegex(
                ArtifactValidationError, "member_id order"
            ):
                mark_bundle_complete(root)
            self.assertEqual(read_manifest(root)["status"], "writing")

            no_tau = Path(directory) / "no-tau"
            shared = self._shared_arrays()
            del shared["constant_delay"]
            self._prepare_run(no_tau, shared=shared)
            with self.assertRaisesRegex(
                ArtifactValidationError, "constant_delay"
            ):
                mark_bundle_complete(no_tau)
            self.assertEqual(read_manifest(no_tau)["status"], "writing")

    def test_object_dtype_is_rejected_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object.npz"
            self._save(path, unsafe=np.asarray(({"x": 1},), dtype=object))
            with self.assertRaisesRegex(
                ArtifactValidationError, "without pickle|object dtype"
            ):
                load_npz_strict(path)

    def test_cancelled_manifest_is_authoritative_over_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cancelled"
            self._prepare_run(root)
            mark_bundle_cancelled(root, "user_requested")
            manifest = read_manifest(root)
            self.assertEqual(manifest["status"], "cancelled")
            self.assertEqual(manifest["termination_reason"], "cancelled")
            self.assertFalse(manifest["converged"])
            with self.assertRaises(IncompleteArtifactError):
                load_bundle(root)

    def test_maximum_iterations_cannot_be_published_as_converged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "false-convergence"
            begin_bundle(root, self._run_manifest(converged=True))
            self._save(root / "shared_posterior.npz", **self._shared_arrays())
            self._save(
                root / "diagnostics.npz",
                **self._diagnostic_arrays(converged=True),
            )
            self._save(root / "bags" / "bag-a.npz", **self._run_bag_arrays())
            self._save(root / "bags" / "bag-b.npz", **self._run_bag_arrays())
            with self.assertRaisesRegex(
                ArtifactValidationError, "cannot be labelled converged"
            ):
                mark_bundle_complete(root)

    def test_inspection_bundle_has_strict_metadata_and_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inspection"
            self._prepare_inspection(root)
            mark_bundle_complete(root)
            bundle = load_inspection_bundle(root)
            self.assertEqual(
                bundle.inspections["bag-a"]["bag_sha256"], "a" * 64
            )
            self.assertEqual(bundle.previews["bag-a"]["position"].shape, (3, 3))

    def test_pid_bundle_aligns_proposals_candidates_and_failure_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pid"
            self._prepare_pid(root)
            mark_bundle_complete(root)
            bundle = load_pid_proposal_evaluation(root)
            np.testing.assert_array_equal(
                bundle.proposal_ensemble["source_member_id"], (41, 7)
            )
            np.testing.assert_array_equal(
                bundle.summary["current_pid_snapshot_record_time"],
                (10.0, 10.3, 10.2, 10.1),
            )
            self.assertFalse(bundle.bags["bag-a"]["forecast_success"][0, 1])
            self.assertTrue(
                np.all(
                    np.isnan(
                        bundle.bags["bag-a"]["prediction_position"][0, 1]
                    )
                )
            )

    def test_pid_inconsistent_failure_and_controller_mass_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            inconsistent = Path(directory) / "inconsistent"
            self._prepare_pid(inconsistent, inconsistent_failure=True)
            with self.assertRaisesRegex(
                ArtifactValidationError, "all-NaN paths"
            ):
                mark_bundle_complete(inconsistent)

            mass = Path(directory) / "mass"
            self._prepare_pid(mass, mass_key=True)
            with self.assertRaisesRegex(
                ArtifactValidationError, "controller-mass"
            ):
                mark_bundle_complete(mass)

    def test_unknown_schema_and_escaping_artifact_path_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "unknown"
            with self.assertRaises(UnsupportedArtifactSchema):
                begin_bundle(root, {"schema": "unknown/v1"})

            inspection = Path(directory) / "escape"
            manifest = self._inspection_manifest()
            manifest["artifacts"]["bags"]["bag-a"]["inspection"] = (
                "../outside.json"
            )
            begin_bundle(inspection, manifest)
            with self.assertRaisesRegex(
                ArtifactValidationError, "inside the bundle"
            ):
                mark_bundle_complete(inspection)


if __name__ == "__main__":
    unittest.main()
