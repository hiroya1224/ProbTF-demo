import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.artifact_io import (
    ASSIMILATION_RUN_SCHEMA,
    ArtifactValidationError,
    IncompleteArtifactError,
    begin_bundle,
    mark_bundle_complete,
)
from grape_param_estim.controller import ControllerConfig
from grape_param_estim.pid_evaluation_input import (
    PID_EVALUATION_REQUEST_SCHEMA,
    candidates_from_request,
    input_from_assimilation_run,
    load_pid_evaluation_request,
)
from grape_param_estim.real_rosbag import (
    PID_AXIS_NAMES,
    PID_CONFIG_FIELD_NAMES,
)
from grape_param_estim.system import VehicleParameters


MEMBER_ID = np.asarray((11, 29), dtype=np.int64)


def _save(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays)


def _group_gains(bag_id):
    if bag_id == "bag-a":
        return np.asarray(
            (
                (3.0, 0.1, 1.0),
                (5.0, 1.0, 2.5),
                (20.0, 1.0, 8.0),
                (4.0, 1.0, 2.0),
            )
        )
    return np.asarray(
        (
            (4.0, 0.2, 1.2),
            (6.0, 1.1, 2.6),
            (18.0, 1.2, 7.0),
            (5.0, 1.1, 2.1),
        )
    )


def _controller_values(group_gains):
    base = ControllerConfig.grape()
    group_by_axis = (0, 0, 1, 2, 2, 3)
    values = []
    for pid, group_index in zip(base.pid, group_by_axis):
        fields = dict(pid.__dict__)
        fields.update(
            p_gain=float(group_gains[group_index, 0]),
            i_gain=float(group_gains[group_index, 1]),
            d_gain=float(group_gains[group_index, 2]),
        )
        values.append(
            [fields[name] for name in PID_CONFIG_FIELD_NAMES]
        )
    return np.asarray(values)


def _bag_arrays(bag_id):
    members = MEMBER_ID.size
    times = np.asarray((0.0, 0.05, 0.1))
    samples = times.size
    position = np.zeros((samples, 3))
    orientation = np.zeros((samples, 4))
    orientation[:, 3] = 1.0
    member_position = np.repeat(position[None, :, :], members, axis=0)
    member_orientation = np.repeat(
        orientation[None, :, :], members, axis=0
    )
    reference_velocity = np.zeros((samples, 3))
    reference_velocity[:, 0] = (0.0, 0.1, 0.2)
    reference_acceleration = np.zeros((samples, 3))
    reference_acceleration[:, 0] = 0.5
    reference_angular_velocity = np.zeros((samples, 3))
    reference_angular_velocity[:, 2] = (0.0, 0.2, 0.4)
    reference_angular_acceleration = np.zeros((samples, 3))
    reference_angular_acceleration[:, 2] = 1.0
    nominal_integral = np.zeros((samples, 6))
    nominal_thrust = np.full((samples, 4), 5.0)
    nominal_gimbal = np.zeros((samples, 4))
    nominal_wrench = np.zeros((samples, 6))
    group_gains = _group_gains(bag_id)
    return {
        "member_id": MEMBER_ID,
        "times": times,
        "record_times": 100.0 + times,
        "observed_position": position,
        "observed_orientation_xyzw": orientation,
        "observation_translation_covariance": np.eye(3) * 1.0e-4,
        "observation_rotation_covariance": np.eye(3) * 2.0e-4,
        "reference_position": position,
        "reference_linear_velocity": reference_velocity,
        "reference_linear_acceleration": reference_acceleration,
        "reference_rpy": position,
        "reference_angular_velocity": reference_angular_velocity,
        "reference_angular_acceleration": reference_angular_acceleration,
        "nominal_position": position,
        "nominal_orientation_xyzw": orientation,
        "nominal_linear_velocity": position,
        "nominal_angular_velocity": position,
        "nominal_controller_integral": nominal_integral,
        "nominal_commanded_thrust": nominal_thrust,
        "nominal_commanded_gimbal_angle": nominal_gimbal,
        "nominal_actuator_thrust": nominal_thrust,
        "nominal_actuator_gimbal_angle": nominal_gimbal,
        "nominal_body_wrench": nominal_wrench,
        "posterior_position": member_position,
        "posterior_orientation_xyzw": member_orientation,
        "posterior_linear_velocity": member_position,
        "posterior_angular_velocity": member_position,
        "posterior_controller_integral": np.zeros(
            (members, samples, 6)
        ),
        "posterior_commanded_thrust": np.full(
            (members, samples, 4), 5.0
        ),
        "posterior_commanded_gimbal_angle": np.zeros(
            (members, samples, 4)
        ),
        "posterior_actuator_thrust": np.full(
            (members, samples, 4), 5.0
        ),
        "posterior_actuator_gimbal_angle": np.zeros(
            (members, samples, 4)
        ),
        "posterior_body_wrench": np.zeros((members, samples, 6)),
        "correction_translation": member_position,
        "correction_rotation_vector": member_position,
        "observed_correction_translation": position,
        "observed_correction_rotation_vector": position,
        "residual_wrench_interval": np.zeros(
            (members, samples - 1, 6)
        ),
        "residual_wrench_knot": np.zeros((members, 2, 6)),
        "innovation_ensemble": np.zeros((members, 12)),
        "objective_contribution": np.asarray((0.1, 0.2)),
        "pose_component_coverage": np.asarray((1.0,)),
        "initial_position": np.zeros((members, 3)),
        "initial_orientation_xyzw": np.tile(
            (0.0, 0.0, 0.0, 1.0), (members, 1)
        ),
        "initial_linear_velocity": np.zeros((members, 3)),
        "initial_angular_velocity": np.zeros((members, 3)),
        "initial_controller_integral": np.zeros((members, 6)),
        "initial_controller_roll_pitch_integration_active": np.ones(
            members, dtype=bool
        ),
        "initial_actuator_thrust": np.full((members, 4), 5.0),
        "initial_actuator_gimbal_angle": np.zeros((members, 4)),
        "actuator_thrust_time_constant": np.asarray((0.01,)),
        "actuator_gimbal_time_constant": np.asarray((0.02,)),
        "actuator_minimum_thrust": np.asarray((1.5,)),
        "actuator_maximum_thrust": np.asarray((27.6145,)),
        "actuator_maximum_gimbal_angle": np.asarray((3.14,)),
        "actuator_maximum_gimbal_rate": np.asarray((6.0,)),
        "q_knot_indices": np.asarray((0, 2), dtype=np.int64),
        "q_knot_times": times[(0, 2),],
        "q_stationary_standard_deviation": np.ones(6) * 0.1,
        "q_correlation_time": np.asarray((0.2,)),
        "q_resolution_sufficient": np.asarray((True,), dtype=bool),
        "controller_snapshot_groups": np.asarray(
            ("xy", "z", "roll_pitch", "yaw")
        ),
        "controller_snapshot_record_times": np.asarray(
            (99.1, 99.2, 99.3, 99.4)
        ),
        "controller_snapshot_gains": group_gains,
        "controller_snapshot_pid_control_flags": np.zeros(4, dtype=bool),
        "controller_snapshot_source_kinds": np.asarray(
            (
                "recorded_startup_parameter_update",
                "recorded_startup_parameter_update",
                "recorded_startup_parameter_update",
                "recorded_startup_parameter_update",
            )
        ),
        "controller_pid_axis_names": np.asarray(PID_AXIS_NAMES),
        "controller_pid_field_names": np.asarray(PID_CONFIG_FIELD_NAMES),
        "controller_pid_configuration": _controller_values(group_gains),
        "controller_xy_control_mode": np.asarray(("position",)),
        "controller_need_yaw_d_control": np.asarray((True,), dtype=bool),
        "controller_start_roll_pitch_integration_height": np.asarray(
            (0.01,)
        ),
        "controller_initial_height": np.asarray((0.0,)),
        "controller_source_compatible_gyro_term": np.asarray(
            (True,), dtype=bool
        ),
        "provenance_bag_path": np.asarray(
            (("/archive/{}.bag".format(bag_id)),)
        ),
        "provenance_bag_sha256": np.asarray((bag_id[0] * 64,)),
        "provenance_bag_size_bytes": np.asarray((1024,), dtype=np.int64),
        "provenance_time_basis": np.asarray(("rosbag_record_time",)),
        "provenance_requested_window": np.asarray((0.0, 0.1)),
        "provenance_source_available_window": np.asarray((0.0, 0.1)),
        "provenance_selected_flight_state": np.asarray(
            (3,), dtype=np.int64
        ),
        "provenance_topic_names": np.asarray(("/pose", "/reference")),
        "provenance_topic_types": np.asarray(("Pose", "Reference")),
    }


def prepare_completed_run(root, complete=True, bag_ids=("bag-a", "bag-b")):
    manifest = {
        "schema": ASSIMILATION_RUN_SCHEMA,
        "run_id": "small-run",
        "created_at": "2026-08-04T12:00:00+09:00",
        "estimator_revision": "test",
        "request_path": "/archive/request.json",
        "request_fingerprint": "sha256:test",
        "project_request_fingerprint": "sha256:" + "a" * 64,
        "selected_bag_ids": list(bag_ids),
        "selected_intervals": {
            bag_id: [0.0, 0.1] for bag_id in bag_ids
        },
        "configuration_fingerprint": "vehicle-a",
        "shared_member_count": MEMBER_ID.size,
        "termination_reason": "converged",
        "converged": True,
        "artifacts": {
            "shared_posterior": "shared_posterior.npz",
            "diagnostics": "diagnostics.npz",
            "bags": {
                bag_id: "bags/{}.npz".format(bag_id)
                for bag_id in bag_ids
            },
        },
    }
    begin_bundle(root, manifest)
    nominal = VehicleParameters.nominal()
    physical = np.zeros((MEMBER_ID.size, 18))
    delay_coordinate = np.asarray((0.0, 0.1))
    coordinates = np.column_stack((physical, delay_coordinate))
    covariance = np.eye(19) * 0.01
    _save(
        root / "shared_posterior.npz",
        member_id=MEMBER_ID,
        parameter_coordinates=coordinates,
        physical_parameter_coordinates=physical,
        constant_delay_coordinate=delay_coordinate,
        mass=np.asarray((nominal.mass, nominal.mass * 1.02)),
        inertia=np.asarray((nominal.inertia, nominal.inertia * 1.01)),
        cog=np.asarray((nominal.cog_offset, nominal.cog_offset)),
        force_effectiveness=np.ones((MEMBER_ID.size, 4)),
        torque_effectiveness=np.ones((MEMBER_ID.size, 4)),
        constant_delay=np.asarray((0.01, 0.025)),
        ridge_covariance=covariance,
        ridge_eigenvalues=np.full(19, 0.01),
        ridge_eigenvectors=np.eye(19),
        ridge_expected_direction=np.eye(19)[0],
        ridge_expected_variance=np.asarray((0.01,)),
        mode_id=np.asarray(("actuator_wiring_nominal",)),
        mode_weight=np.asarray((1.0,)),
        selected_mode_id=np.asarray(("actuator_wiring_nominal",)),
    )
    _save(
        root / "diagnostics.npz",
        iteration=np.asarray((0,), dtype=np.int64),
        objective=np.asarray((1.0,)),
        accepted_objective=np.asarray((0.9,)),
        gradient_norm=np.asarray((0.1,)),
        step_norm=np.asarray((0.1,)),
        accepted_fraction=np.asarray((1.0,)),
        ensemble_rank=np.asarray((1,), dtype=np.int64),
        converged=np.asarray((True,), dtype=bool),
        termination_reason=np.asarray(("converged",)),
    )
    for bag_id in bag_ids:
        _save(root / "bags" / "{}.npz".format(bag_id), **_bag_arrays(bag_id))
    if complete:
        mark_bundle_complete(root)
    return root


def write_pid_request(path, run, selected="member-pick"):
    path.write_text(
        json.dumps(
            {
                "schema": PID_EVALUATION_REQUEST_SCHEMA,
                "evaluation_id": "small-pid-evaluation",
                "assimilation_run": str(run),
                "baseline_bag_id": "bag-b",
                "residual_policy": {
                    "bag-a": "posterior_replay",
                    "bag-b": "zero",
                },
                "cvar_level": 0.8,
                "thresholds": {
                    "position": 0.5,
                    "orientation": None,
                    "position_metric": "position_rmse",
                    "orientation_metric": "orientation_rmse",
                },
                "candidates": [
                    {"candidate_id": "current", "source": "current"},
                    {
                        "candidate_id": "member-pick",
                        "source": "member-derived",
                        "source_member_id": 11,
                    },
                    {
                        "candidate_id": "user-exact",
                        "source": "user",
                        "values": _group_gains("bag-a").tolist(),
                    },
                ],
                "selected_candidate_id": selected,
            }
        ),
        encoding="utf-8",
    )
    return path


class PidEvaluationInputTests(unittest.TestCase):
    def test_complete_run_restores_raw_members_and_baseline_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = prepare_completed_run(root / "run")
            predictive = input_from_assimilation_run(
                str(run),
                "bag-b",
                residual_policy={
                    "bag-a": "posterior_replay",
                    "bag-b": "zero",
                },
            )
            np.testing.assert_array_equal(predictive.member_id, MEMBER_ID)
            np.testing.assert_allclose(
                predictive.proposal_ensemble.constant_delay, (0.01, 0.025)
            )
            self.assertEqual(
                predictive.selected_mode_id, "actuator_wiring_nominal"
            )
            self.assertEqual(predictive.current.provenance.bag_id, "bag-b")
            np.testing.assert_array_equal(
                predictive.current.values, _group_gains("bag-b")
            )
            self.assertFalse(
                np.array_equal(
                    predictive.current.values,
                    0.5 * (_group_gains("bag-a") + _group_gains("bag-b")),
                )
            )
            bag_a = next(
                value for value in predictive.bags if value.bag_id == "bag-a"
            )
            self.assertEqual(bag_a.residual_policy, "posterior_replay")
            self.assertEqual(
                predictive.bags[1].residual_policy, "zero"
            )
            np.testing.assert_array_equal(
                bag_a.references[1].linear_velocity, (0.1, 0.0, 0.0)
            )
            np.testing.assert_array_equal(
                bag_a.references[1].linear_acceleration, (0.5, 0.0, 0.0)
            )
            self.assertEqual(len(bag_a.initial_states), MEMBER_ID.size)
            self.assertEqual(
                bag_a.posterior_residual_wrench.shape, (2, 2, 6)
            )

    def test_request_resolves_exact_candidate_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = prepare_completed_run(root / "run")
            request = load_pid_evaluation_request(
                str(write_pid_request(root / "request.json", run))
            )
            predictive = input_from_assimilation_run(
                request.assimilation_run,
                request.baseline_bag_id,
                request.residual_policies(("bag-a", "bag-b")),
            )
            candidates = candidates_from_request(request, predictive)
            self.assertEqual(
                tuple(value.candidate_id for value in candidates),
                ("current", "member-pick", "user-exact"),
            )
            self.assertEqual(
                tuple(value.source for value in candidates),
                ("current", "member-derived", "user"),
            )
            self.assertEqual(candidates[1].source_member_id, 11)
            np.testing.assert_array_equal(
                candidates[1].configuration.values,
                predictive.proposal_ensemble.exact_gain_values[0],
            )
            np.testing.assert_array_equal(
                candidates[2].configuration.values, _group_gains("bag-a")
            )

    def test_incomplete_or_missing_production_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writing = prepare_completed_run(
                root / "writing", complete=False
            )
            with self.assertRaises(IncompleteArtifactError):
                input_from_assimilation_run(str(writing), "bag-a")

            complete = prepare_completed_run(root / "complete")
            bag_path = complete / "bags" / "bag-a.npz"
            with np.load(str(bag_path), allow_pickle=False) as archive:
                arrays = {
                    key: archive[key]
                    for key in archive.files
                    if key != "reference_linear_acceleration"
                }
            _save(bag_path, **arrays)
            with self.assertRaisesRegex(
                ArtifactValidationError,
                "reference_linear_acceleration",
            ):
                input_from_assimilation_run(str(complete), "bag-a")


if __name__ == "__main__":
    unittest.main()
