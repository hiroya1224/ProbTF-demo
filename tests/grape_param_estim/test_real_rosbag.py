from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.real_assimilation import (
    assimilate_real_episode,
    save_real_assimilation,
)
from grape_param_estim.real_rosbag import (
    ControllerGainEvents,
    FlightStateSeries,
    PidReferenceSeries,
    RosbagArrayData,
    TimedVectorSeries,
    build_real_flight_episode,
    quaternion_slerp_resample,
    robust_covariance,
    save_real_flight_episode,
    select_continuous_flight_window,
)


def _fake_arrays():
    bag_start = 100.0
    sample_times = np.arange(100.0, 113.01, 0.05)
    position = np.zeros((sample_times.size, 3))
    position[:, 2] = 0.4
    moving = sample_times >= 103.0
    position[moving, 0] = 0.1 * (sample_times[moving] - 103.0)
    orientation = np.zeros((sample_times.size, 4))
    orientation[:, 3] = 1.0

    pid_times = np.arange(103.0, 112.01, 0.05)
    count = pid_times.size
    target_position = np.column_stack(
        (0.2 * (pid_times - 103.0), np.zeros(count), np.ones(count))
    )
    target_velocity = np.column_stack(
        (np.full(count, 0.2), np.zeros((count, 2)))
    )
    target_rpy = np.zeros((count, 3))
    target_rpy[:, 2] = 0.1 * (pid_times - 103.0)
    target_omega = np.zeros((count, 3))
    p_term = np.full((count, 6), 0.2)
    i_term = np.tile(
        np.asarray((0.01, 0.02, 0.03, 0.04, 0.05, 0.06)),
        (count, 1),
    )
    d_term = np.full((count, 6), 0.1)
    feedforward = np.tile(
        np.asarray((1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
        (count, 1),
    )
    total = p_term + i_term + d_term + feedforward

    gain_times = np.asarray((100.10, 100.20, 100.30, 100.40))
    gains = np.asarray(
        ((4.0, 0.1, 2.0), (5.0, 1.0, 2.5),
         (13.0, 1.0, 20.0), (6.0, 1.0, 2.0))
    )
    joint_times = np.arange(102.0, 112.01, 0.1)
    joint_position = np.zeros((joint_times.size, 4))
    command_times = np.arange(103.0, 112.01, 0.05)
    command = np.tile(np.asarray((1.0, 2.0, 3.0, 4.0)),
                      (command_times.size, 1))
    topic_names = tuple("topic_{}".format(index) for index in range(10))
    topic_types = tuple("type_{}".format(index) for index in range(10))
    return RosbagArrayData(
        bag_path="/tmp/fake.bag",
        bag_sha256="abc",
        bag_size_bytes=123,
        bag_record_start=bag_start,
        bag_record_end=113.0,
        topic_names=topic_names,
        topic_types=topic_types,
        cog_position=TimedVectorSeries(sample_times, position),
        baselink_orientation=TimedVectorSeries(
            sample_times, orientation
        ),
        pid=PidReferenceSeries(
            pid_times,
            target_position,
            target_velocity,
            target_rpy,
            target_omega,
            total,
            p_term,
            i_term,
            d_term,
        ),
        flight_state=FlightStateSeries(
            np.asarray(
                (100.0, 101.0, 102.0, 103.0, 104.0, 105.0,
                 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0)
            ),
            np.asarray((0, 1, 2, 3, 5, 4, 6, 0, 1, 2, 3, 4, 6)),
        ),
        controller_gain_events=ControllerGainEvents(
            gain_times,
            ("xy", "z", "roll_pitch", "yaw"),
            gains,
            np.zeros(4, dtype=bool),
        ),
        joint_position=TimedVectorSeries(joint_times, joint_position),
        joint_names=("gimbal1", "gimbal2", "gimbal3", "gimbal4"),
        commanded_thrust=TimedVectorSeries(command_times, command),
    )


class RealRosbagArrayTests(unittest.TestCase):
    def test_window_is_one_complete_episode_and_crossing_is_rejected(self):
        arrays = _fake_arrays()
        selected = select_continuous_flight_window(
            arrays.flight_state, arrays.bag_record_start
        )
        self.assertEqual(selected[0], 104.0)
        self.assertEqual(selected[1], 105.0)
        np.testing.assert_array_equal(selected[3], (3, 5, 4, 6))
        with self.assertRaisesRegex(ValueError, "cannot be concatenated"):
            select_continuous_flight_window(
                arrays.flight_state,
                arrays.bag_record_start,
                start_local=3.2,
                end_local=11.5,
            )
        complete = select_continuous_flight_window(
            arrays.flight_state,
            arrays.bag_record_start,
            window_state=None,
        )
        self.assertEqual(complete[0], 103.0)
        self.assertEqual(complete[1], 106.0)

    def test_quaternion_slerp_is_sign_invariant(self):
        half = np.sqrt(0.5)
        source = np.asarray(
            ((0.0, 0.0, 0.0, 1.0),
             (0.0, 0.0, -half, -half))
        )
        result = quaternion_slerp_resample(
            (0.0, 1.0), source, (0.0, 0.5, 1.0)
        )
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0)
        self.assertGreater(np.dot(result[0], result[1]), 0.0)
        self.assertGreater(np.dot(result[1], result[2]), 0.0)
        np.testing.assert_allclose(
            result[1], (0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8))
        )

    def test_robust_covariance_rejects_a_large_outlier(self):
        generator = np.random.RandomState(4)
        samples = generator.normal(0.0, 0.01, size=(200, 3))
        samples = np.vstack((samples, np.asarray((50.0, -60.0, 70.0))))
        center, covariance, inliers = robust_covariance(samples)
        self.assertFalse(inliers[-1])
        self.assertLess(np.linalg.norm(center), 0.005)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0.0))
        self.assertLess(np.max(np.diag(covariance)), 2.0e-4)

    def test_array_builder_outputs_direct_estimator_contracts(self):
        episode = build_real_flight_episode(
            _fake_arrays(), sample_period=0.2
        )
        self.assertAlmostEqual(episode.window_start_local_time, 4.0)
        self.assertAlmostEqual(episode.window_end_local_time, 4.8)
        self.assertAlmostEqual(
            episode.provenance.requested_window_end, 105.0
        )
        self.assertEqual(episode.observations.position.shape, (5, 3))
        self.assertEqual(len(episode.references), 5)
        np.testing.assert_allclose(
            episode.references[3].linear_acceleration, (1.0, 2.0, 3.0)
        )
        np.testing.assert_allclose(
            episode.references[3].angular_acceleration, (4.0, 5.0, 6.0)
        )
        np.testing.assert_allclose(
            episode.controller_snapshot.axis_gains(),
            np.asarray(
                ((4.0, 0.1, 2.0), (4.0, 0.1, 2.0),
                 (5.0, 1.0, 2.5), (13.0, 1.0, 20.0),
                 (13.0, 1.0, 20.0), (6.0, 1.0, 2.0))
            ),
        )
        np.testing.assert_allclose(
            episode.initial_controller_state.integral_error,
            (0.1, 0.2, 0.03, 0.04, 0.05, 0.06),
        )
        np.testing.assert_allclose(
            episode.initial_actuator_state.thrust, (1.5, 2.0, 3.0, 4.0)
        )
        np.testing.assert_allclose(
            episode.initial_actuator_state.gimbal_angle, np.zeros(4)
        )
        self.assertEqual(
            episode.provenance.time_basis, "rosbag_record_time"
        )
        self.assertEqual(episode.provenance.selected_flight_state, 5)
        self.assertEqual(
            episode.provenance.thrust_anchor_kind,
            "clipped_four_axes_command_proxy",
        )
        self.assertTrue(
            np.all(np.linalg.eigvalsh(
                episode.observations.translation_covariance
            ) > 0.0)
        )

    def test_only_enabled_dynamic_reconfigure_events_change_effective_gains(self):
        arrays = _fake_arrays()
        base_events = arrays.controller_gain_events
        inactive_events = ControllerGainEvents(
            np.append(base_events.record_times, 102.0),
            base_events.groups + ("xy",),
            np.vstack((base_events.gains, (99.0, 98.0, 97.0))),
            np.append(base_events.pid_control_flags, False),
        )
        inactive_episode = build_real_flight_episode(
            replace(arrays, controller_gain_events=inactive_events),
            sample_period=0.2,
        )
        np.testing.assert_array_equal(
            inactive_episode.controller_snapshot.gains[0],
            (4.0, 0.1, 2.0),
        )
        self.assertEqual(
            inactive_episode.controller_snapshot.source_kinds[0],
            "static_controller_configuration",
        )

        active_events = ControllerGainEvents(
            np.append(base_events.record_times, 102.0),
            base_events.groups + ("xy",),
            np.vstack((base_events.gains, (7.0, 0.2, 3.0))),
            np.append(base_events.pid_control_flags, True),
        )
        active_episode = build_real_flight_episode(
            replace(arrays, controller_gain_events=active_events),
            sample_period=0.2,
        )
        np.testing.assert_array_equal(
            active_episode.controller_snapshot.gains[0],
            (7.0, 0.2, 3.0),
        )
        self.assertEqual(
            active_episode.controller_snapshot.source_kinds[0],
            "dynamic_reconfigure_applied",
        )

    def test_inactive_startup_event_must_confirm_static_configuration(self):
        arrays = _fake_arrays()
        events = arrays.controller_gain_events
        inconsistent = events.gains.copy()
        inconsistent[0] = (8.0, 0.3, 4.0)
        with self.assertRaisesRegex(ValueError, "inactive startup xy"):
            build_real_flight_episode(
                replace(
                    arrays,
                    controller_gain_events=ControllerGainEvents(
                        events.record_times,
                        events.groups,
                        inconsistent,
                        events.pid_control_flags,
                    ),
                ),
                sample_period=0.2,
            )

    def test_episode_artifact_is_pickle_free_and_complete(self):
        episode = build_real_flight_episode(
            _fake_arrays(), sample_period=0.2
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = save_real_flight_episode(
                str(Path(directory) / "real_episode.npz"), episode
            )
            with np.load(str(destination), allow_pickle=False) as artifact:
                self.assertEqual(
                    str(artifact["schema"][0]),
                    "grape-weak-constraint/phase5-real-episode",
                )
                self.assertEqual(
                    artifact["reference_position"].shape,
                    episode.observations.position.shape,
                )
                np.testing.assert_array_equal(
                    artifact["initial_actuator_thrust"],
                    episode.initial_actuator_state.thrust,
                )
                self.assertEqual(
                    str(artifact["provenance_time_basis"][0]),
                    "rosbag_record_time",
                )
                for key in artifact.files:
                    self.assertFalse(artifact[key].dtype.hasobject)

    def test_assimilation_artifact_carries_complete_episode_provenance(self):
        episode = build_real_flight_episode(
            _fake_arrays(), sample_period=0.1
        )
        result = assimilate_real_episode(
            episode,
            maximum_knots=2,
            maximum_iterations=1,
            seed=18,
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = save_real_assimilation(
                str(Path(directory) / "real_assimilation.npz"), result
            )
            with np.load(str(destination), allow_pickle=False) as artifact:
                self.assertEqual(
                    str(artifact["schema"][0]),
                    "grape-weak-constraint/phase5-real-assimilation",
                )
                self.assertEqual(
                    str(artifact["provenance_thrust_anchor_kind"][0]),
                    "clipped_four_axes_command_proxy",
                )
                self.assertEqual(
                    str(artifact["provenance_time_basis"][0]),
                    "rosbag_record_time",
                )
                np.testing.assert_array_equal(
                    artifact["controller_snapshot_source_kinds"],
                    episode.controller_snapshot.source_kinds,
                )
                np.testing.assert_array_equal(
                    artifact["residual_wrench_interval"],
                    result.posterior.residual_wrench_ensemble,
                )
                for key in artifact.files:
                    self.assertFalse(artifact[key].dtype.hasobject)


if __name__ == "__main__":
    unittest.main()
