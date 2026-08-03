import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from grape_param_estim.artifact_io import AssimilationRunBundle
from grape_param_estim.controller import ControllerConfig
from grape_param_estim.held_out_validation import (
    HELD_OUT_VALIDATION_REQUEST_SCHEMA,
    HeldOutBagRequest,
    HeldOutEvaluationTarget,
    HeldOutForecastScenario,
    HeldOutValidationRequest,
    RawPhysicalPosterior,
    forecast_held_out_posterior,
    load_held_out_validation,
    load_held_out_validation_request,
    prepare_held_out_episode,
    raw_physical_posterior_from_run,
    save_held_out_validation,
    score_held_out_forecasts,
)
from grape_param_estim.held_out_validation_cli import run_request
from grape_param_estim.real_rosbag import (
    ControllerGainSnapshot,
    EpisodeProvenance,
    RealFlightEpisode,
)
from grape_param_estim.system import (
    ActuatorState,
    ClosedLoopTrajectory,
    ControllerState,
    PoseObservations,
    ReferenceState,
    RigidBodyState,
)


SHA = "a" * 64
FINGERPRINT = "incomplete:test-configuration"


def reference(position=(0.0, 0.0, 0.0)):
    return ReferenceState(
        np.asarray(position),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
    )


def provenance():
    return EpisodeProvenance(
        bag_path="/tmp/held-out.bag",
        bag_sha256=SHA,
        bag_size_bytes=12,
        bag_record_start=10.0,
        bag_record_end=11.0,
        time_basis="rosbag_record_time",
        requested_window_start=10.0,
        requested_window_end=10.6,
        source_available_start=10.0,
        source_available_end=10.6,
        resample_period=0.1,
        selected_flight_state=5,
        flight_transition_record_times=np.asarray((10.0, 10.6)),
        flight_transition_states=np.asarray((5, 4)),
        static_window_start=9.0,
        static_window_end=9.5,
        static_position_samples=5,
        static_position_inliers=5,
        static_orientation_samples=5,
        static_orientation_inliers=5,
        static_position_center=np.zeros(3),
        static_orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        covariance_outlier_threshold=6.0,
        covariance_eigenvalue_floor=1.0e-12,
        controller_state_anchor_record_time=9.99,
        joint_anchor_record_time=9.99,
        thrust_anchor_record_time=9.99,
        thrust_anchor_kind="recorded_command",
        reference_acceleration_kind="recorded_pid_reference",
        controller_static_source="ControllerConfig.grape",
        controller_source_revision="test",
        topic_names=("/test",),
        topic_types=("test/Message",),
    )


def scenario():
    times = np.arange(7, dtype=float) * 0.1
    return HeldOutForecastScenario(
        bag_id="success",
        times=times,
        record_times=times + 10.0,
        references=tuple(reference() for _value in times),
        initial_state=RigidBodyState(
            np.zeros(3),
            np.asarray((0.0, 0.0, 0.0, 1.0)),
            np.zeros(3),
            np.zeros(3),
        ),
        initial_controller_state=ControllerState(np.arange(6, dtype=float)),
        initial_actuator_state=ActuatorState(
            np.full(4, 5.0), np.zeros(4)
        ),
        controller_configuration=ControllerConfig.grape(),
        controller_snapshot=ControllerGainSnapshot(
            groups=("xy", "z", "roll_pitch", "yaw"),
            record_times=np.asarray((9.0, 9.1, 9.2, 9.3)),
            gains=np.ones((4, 3)),
            pid_control_flags=np.ones(4, dtype=bool),
            source_kinds=("dynamic_reconfigure_applied",) * 4,
        ),
        provenance=provenance(),
    )


def posterior(configuration_fingerprint=FINGERPRINT):
    nominal_inertia = np.diag((0.06, 0.07, 0.12))
    return RawPhysicalPosterior(
        member_id=np.asarray((3, 8), dtype=np.int64),
        mass=np.asarray((1.9, 2.1)),
        inertia=np.asarray((nominal_inertia, nominal_inertia * 1.1)),
        cog=np.zeros((2, 3)),
        force_effectiveness=np.ones((2, 4)),
        torque_effectiveness=np.ones((2, 4)),
        constant_delay=np.asarray((0.0125, 0.0375)),
        source_run_id="source-run",
        source_configuration_fingerprint=configuration_fingerprint,
        source_root="/tmp/source-run",
    )


def fake_trajectory(times, mass):
    count = len(times)
    position = np.zeros((count, 3))
    position[:, 0] = np.asarray(times) + 0.01 * mass
    position[0] = 0.0
    quaternion = np.tile((0.0, 0.0, 0.0, 1.0), (count, 1))
    return ClosedLoopTrajectory(
        times=np.asarray(times),
        position=position,
        orientation_xyzw=quaternion,
        linear_velocity=np.zeros((count, 3)),
        angular_velocity=np.zeros((count, 3)),
        controller_integral=np.tile(np.arange(6, dtype=float), (count, 1)),
        commanded_thrust=np.zeros((count, 4)),
        commanded_gimbal_angle=np.zeros((count, 4)),
        actuator_thrust=np.full((count, 4), 5.0),
        actuator_gimbal_angle=np.zeros((count, 4)),
        body_wrench=np.zeros((count, 6)),
    )


def request():
    return HeldOutValidationRequest(
        validation_id="held-out-test",
        assimilation_run="/tmp/source-run",
        held_out_bag=HeldOutBagRequest(
            bag_id="success",
            path="/tmp/held-out.bag",
            sha256=SHA,
            episode_index=0,
            selected_interval=(0.0, 0.6),
            window_state=5,
            configuration_fingerprint=FINGERPRINT,
        ),
        sample_period=0.1,
    )


def request_payload():
    return {
        "schema": HELD_OUT_VALIDATION_REQUEST_SCHEMA,
        "validation_id": "held-out-test",
        "assimilation_run": "/tmp/source-run",
        "held_out_bag": {
            "bag_id": "success",
            "path": "/tmp/held-out.bag",
            "sha256": SHA,
            "episode_index": 0,
            "selected_interval": [0.0, 0.6],
            "window_state": 5,
            "configuration_fingerprint": FINGERPRINT,
        },
        "settings": {"sample_period": 0.1},
    }


class HeldOutValidationTest(unittest.TestCase):
    def _successful_forecast(self):
        calls = []

        def simulate(**kwargs):
            calls.append(kwargs)
            return fake_trajectory(
                kwargs["times"], kwargs["plant"].parameters.mass
            )

        with mock.patch(
            "grape_param_estim.held_out_validation.simulate_closed_loop",
            side_effect=simulate,
        ):
            forecasts = forecast_held_out_posterior(
                posterior(), scenario()
            )
        return forecasts, calls

    def test_forecast_contract_has_no_observations_and_q_is_always_zero(self):
        selected_scenario = scenario()
        self.assertFalse(hasattr(selected_scenario, "observations"))
        self.assertFalse(hasattr(selected_scenario, "observed_position"))
        forecasts, calls = self._successful_forecast()
        self.assertEqual(len(calls), 3)  # two raw members plus nominal
        self.assertTrue(np.all(forecasts.completed))
        self.assertTrue(forecasts.nominal_completed)
        self.assertEqual(
            [call["actuator_parameters"].delay for call in calls],
            [0.0125, 0.0375, 0.0],
        )
        for call in calls:
            self.assertNotIn("observations", call)
            np.testing.assert_array_equal(
                call["interval_residual_wrench"], np.zeros((6, 6))
            )
            np.testing.assert_array_equal(
                call["initial_controller_state"].integral_error,
                selected_scenario.initial_controller_state.integral_error,
            )
            np.testing.assert_array_equal(
                call["initial_actuator_state"].thrust,
                selected_scenario.initial_actuator_state.thrust,
            )

    def test_changing_post_initial_observations_only_changes_scores(self):
        forecasts, _calls = self._successful_forecast()
        selected_scenario = scenario()
        positions = np.zeros((7, 3))
        orientations = np.tile((0.0, 0.0, 0.0, 1.0), (7, 1))
        first_target = HeldOutEvaluationTarget(
            selected_scenario.times, positions, orientations
        )
        changed = positions.copy()
        changed[1:, 0] = 3.0
        second_target = HeldOutEvaluationTarget(
            selected_scenario.times, changed, orientations
        )
        first = score_held_out_forecasts(
            forecasts, selected_scenario, first_target
        )
        second = score_held_out_forecasts(
            forecasts, selected_scenario, second_target
        )
        for name in forecasts.trajectories:
            np.testing.assert_array_equal(
                first.forecasts.trajectories[name],
                second.forecasts.trajectories[name],
            )
        self.assertFalse(np.array_equal(first.metrics, second.metrics))
        np.testing.assert_array_equal(
            first.metrics[:, 4:], second.metrics[:, 4:]
        )

    def test_prepare_uses_pose_for_initial_condition_but_hides_target(self):
        times = np.arange(7, dtype=float) * 0.1
        positions = np.column_stack((times, times * 0.0, times * 0.0))
        orientations = np.tile((0.0, 0.0, 0.0, 1.0), (7, 1))
        episode = RealFlightEpisode(
            record_times=times + 10.0,
            window_start_record_time=10.0,
            window_end_record_time=10.6,
            window_start_local_time=0.0,
            window_end_local_time=0.6,
            observations=PoseObservations(
                times,
                positions,
                orientations,
                np.eye(3),
                np.eye(3),
            ),
            references=tuple(reference() for _value in times),
            controller_configuration=scenario().controller_configuration,
            initial_controller_state=scenario().initial_controller_state,
            initial_actuator_state=scenario().initial_actuator_state,
            controller_snapshot=scenario().controller_snapshot,
            provenance=provenance(),
        )
        forecast_scenario, target = prepare_held_out_episode("success", episode)
        np.testing.assert_array_equal(
            forecast_scenario.initial_state.position, positions[0]
        )
        self.assertFalse(hasattr(forecast_scenario, "observed_position"))
        np.testing.assert_array_equal(target.observed_position, positions)

    def test_source_adapter_ignores_all_bag_local_fields(self):
        selected = posterior()
        shared = {
            "member_id": selected.member_id,
            "mass": selected.mass,
            "inertia": selected.inertia,
            "cog": selected.cog,
            "force_effectiveness": selected.force_effectiveness,
            "torque_effectiveness": selected.torque_effectiveness,
            "constant_delay": selected.constant_delay,
        }
        bundle = AssimilationRunBundle(
            root=Path("/tmp/source-run"),
            manifest={
                "run_id": "source-run",
                "configuration_fingerprint": FINGERPRINT,
            },
            shared_posterior=shared,
            diagnostics={},
            bags={
                "failure": {
                    "initial_position": np.full((2, 3), 999.0),
                    "residual_wrench_interval": np.full((2, 4, 6), 999.0),
                }
            },
            warnings=tuple(),
        )
        restored = raw_physical_posterior_from_run(bundle)
        np.testing.assert_array_equal(restored.mass, selected.mass)
        np.testing.assert_array_equal(
            restored.constant_delay, selected.constant_delay
        )
        self.assertFalse(hasattr(restored, "residual_wrench_interval"))
        self.assertFalse(hasattr(restored, "initial_position"))

    def test_numerical_failure_is_preserved_with_nan_paths_and_reason(self):
        def simulate(**kwargs):
            mass = kwargs["plant"].parameters.mass
            if np.isclose(mass, 1.9):
                raise FloatingPointError("synthetic divergence")
            return fake_trajectory(kwargs["times"], mass)

        with mock.patch(
            "grape_param_estim.held_out_validation.simulate_closed_loop",
            side_effect=simulate,
        ):
            forecasts = forecast_held_out_posterior(posterior(), scenario())
        self.assertFalse(forecasts.completed[0])
        self.assertIn("FloatingPointError", forecasts.failure_reason[0])
        for value in forecasts.trajectories.values():
            self.assertTrue(np.all(np.isnan(value[0])))
        target = HeldOutEvaluationTarget(
            scenario().times,
            np.zeros((7, 3)),
            np.tile((0.0, 0.0, 0.0, 1.0), (7, 1)),
        )
        result = score_held_out_forecasts(forecasts, scenario(), target)
        self.assertTrue(np.all(np.isnan(result.metrics[0])))

    def test_pickle_free_artifact_round_trip_preserves_raw_paths(self):
        forecasts, _calls = self._successful_forecast()
        selected_scenario = scenario()
        target = HeldOutEvaluationTarget(
            selected_scenario.times,
            np.zeros((7, 3)),
            np.tile((0.0, 0.0, 0.0, 1.0), (7, 1)),
        )
        result = score_held_out_forecasts(
            forecasts, selected_scenario, target
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = save_held_out_validation(
                directory,
                request(),
                request_payload(),
                posterior(),
                selected_scenario,
                target,
                result,
            )
            loaded = load_held_out_validation(str(destination))
            np.testing.assert_array_equal(
                loaded.arrays["posterior_position"],
                forecasts.trajectories["position"],
            )
            with np.load(
                str(destination / "validation.npz"), allow_pickle=False
            ) as arrays:
                for name in arrays.files:
                    self.assertNotEqual(arrays[name].dtype.kind, "O")
            self.assertEqual(loaded.manifest["residual_policy"], "zero")
            self.assertEqual(
                loaded.manifest["observation_usage"],
                [
                    "initial_pose_velocity_anchor_from_leading_pose_samples",
                    "evaluation_only",
                ],
            )

    def test_loader_rejects_tampered_pose_error_consistency(self):
        forecasts, _calls = self._successful_forecast()
        selected_scenario = scenario()
        target = HeldOutEvaluationTarget(
            selected_scenario.times,
            np.zeros((7, 3)),
            np.tile((0.0, 0.0, 0.0, 1.0), (7, 1)),
        )
        result = score_held_out_forecasts(
            forecasts, selected_scenario, target
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = save_held_out_validation(
                directory,
                request(),
                request_payload(),
                posterior(),
                selected_scenario,
                target,
                result,
            )
            with np.load(
                str(destination / "validation.npz"), allow_pickle=False
            ) as source:
                arrays = {name: source[name].copy() for name in source.files}
            arrays["posterior_position"][0, 1, 0] += 0.5
            np.savez_compressed(str(destination / "validation.npz"), **arrays)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_held_out_validation(str(destination))

    def test_loader_rejects_non_nan_path_for_failed_member(self):
        def simulate(**kwargs):
            mass = kwargs["plant"].parameters.mass
            if np.isclose(mass, 1.9):
                raise FloatingPointError("synthetic divergence")
            return fake_trajectory(kwargs["times"], mass)

        selected_scenario = scenario()
        with mock.patch(
            "grape_param_estim.held_out_validation.simulate_closed_loop",
            side_effect=simulate,
        ):
            forecasts = forecast_held_out_posterior(
                posterior(), selected_scenario
            )
        target = HeldOutEvaluationTarget(
            selected_scenario.times,
            np.zeros((7, 3)),
            np.tile((0.0, 0.0, 0.0, 1.0), (7, 1)),
        )
        result = score_held_out_forecasts(
            forecasts, selected_scenario, target
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = save_held_out_validation(
                directory,
                request(),
                request_payload(),
                posterior(),
                selected_scenario,
                target,
                result,
            )
            load_held_out_validation(str(destination))
            with np.load(
                str(destination / "validation.npz"), allow_pickle=False
            ) as source:
                arrays = {name: source[name].copy() for name in source.files}
            arrays["posterior_position"][0, 0, 0] = 0.0
            np.savez_compressed(str(destination / "validation.npz"), **arrays)
            with self.assertRaisesRegex(ValueError, "failed member"):
                load_held_out_validation(str(destination))

    def test_request_parser_is_strict_and_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "request.json"
            payload = request_payload()
            payload["assimilation_run"] = "source-run"
            payload["held_out_bag"]["path"] = "success.bag"
            source.write_text(json.dumps(payload), encoding="utf-8")
            selected, _raw = load_held_out_validation_request(str(source))
            self.assertEqual(
                selected.assimilation_run,
                str((Path(directory) / "source-run").resolve()),
            )
            self.assertEqual(
                selected.held_out_bag.path,
                str((Path(directory) / "success.bag").resolve()),
            )
            payload["unexpected"] = True
            source.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                load_held_out_validation_request(str(source))

    def test_configuration_mismatch_rejects_before_reading_rosbag(self):
        selected_request = request()
        mismatched = posterior("different")
        with mock.patch(
            "grape_param_estim.held_out_validation_cli.load_held_out_validation_request",
            return_value=(selected_request, request_payload()),
        ), mock.patch(
            "grape_param_estim.held_out_validation_cli.load_assimilation_run",
            return_value=object(),
        ), mock.patch(
            "grape_param_estim.held_out_validation_cli.raw_physical_posterior_from_run",
            return_value=mismatched,
        ), mock.patch(
            "grape_param_estim.held_out_validation_cli.read_grape_rosbag_arrays"
        ) as reader:
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                run_request("request.json", "output")
        reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
