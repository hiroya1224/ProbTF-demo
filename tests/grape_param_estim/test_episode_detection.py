import unittest

import numpy as np

from grape_param_estim.episode_detection import (
    EpisodeDetectionSettings,
    command_valid_mask,
    detect_control_episodes,
)
from grape_param_estim.failure_bag import FailureBagRecording


def _recording(
    support_height=0.2,
    airborne=True,
    command_gap=False,
    state_gap=False,
):
    step = 0.01
    timestamps = np.arange(0.0, 14.0 + 0.5 * step, step)
    flight_state = np.zeros(timestamps.size, dtype=int)
    flight_state[(timestamps >= 3.0) & (timestamps < 10.0)] = 3
    if state_gap:
        flight_state[
            (timestamps >= 7.0) & (timestamps < 7.04)
        ] = 6
    flight_state[timestamps >= 10.0] = 6

    position = np.zeros((timestamps.size, 3), dtype=float)
    position[:, 2] = support_height
    if airborne:
        ramp = np.clip((timestamps - 5.0) / 0.5, 0.0, 1.0)
        position[:, 2] += 0.30 * ramp
    velocity = np.gradient(position, step, axis=0)

    command_mask = (timestamps >= 3.0) & (timestamps <= 10.0)
    if command_gap:
        command_mask &= ~(
            (timestamps >= 7.0) & (timestamps <= 7.5)
        )
    command_times = timestamps[command_mask]
    command_wrench = np.zeros((command_times.size, 6), dtype=float)
    command_wrench[:, 2] = (
        20.0 + 2.0 * np.sin(command_times)
    )
    return FailureBagRecording(
        bag_path="/synthetic/automatic.bag",
        bag_sha256="b" * 64,
        bag_start_time=100.0,
        bag_duration_s=14.0,
        command_times=command_times,
        command_wrench=command_wrench,
        imu_times=timestamps,
        specific_force=np.zeros((timestamps.size, 3)),
        angular_velocity=np.zeros((timestamps.size, 3)),
        state_times=timestamps,
        position=position,
        linear_velocity=velocity,
        flight_state_times=timestamps,
        flight_state=flight_state,
    )


def _settings():
    return EpisodeDetectionSettings(
        active_flight_states=(3, 4, 5, 17),
        diagnostic_flight_states=(17,),
        baseline_window_s=2.0,
        minimum_active_duration_s=0.5,
        minimum_liftoff_height_m=0.02,
        minimum_airborne_duration_s=0.5,
        persistence_s=0.2,
        standardized_threshold=6.0,
    )


class EpisodeDetectionTests(unittest.TestCase):
    def test_recording_without_commands_can_report_no_episode(self):
        recording = _recording()
        empty = FailureBagRecording(
            **{
                **recording.__dict__,
                "command_times": np.empty(0),
                "command_wrench": np.empty((0, 6)),
                "flight_state": np.zeros_like(
                    recording.flight_state
                ),
            }
        )

        self.assertEqual(
            detect_control_episodes(empty, _settings()), []
        )
        np.testing.assert_array_equal(
            command_valid_mask(empty, [1.0, 2.0]),
            [False, False],
        )

    def test_relative_liftoff_is_invariant_to_support_height(self):
        low = detect_control_episodes(_recording(0.2), _settings())
        high = detect_control_episodes(_recording(2.3), _settings())

        self.assertEqual(len(low), 1)
        self.assertEqual(low[0].status, "candidate")
        self.assertAlmostEqual(low[0].support_height_m, 0.2, places=6)
        self.assertAlmostEqual(high[0].support_height_m, 2.3, places=6)
        self.assertAlmostEqual(
            low[0].liftoff_s, high[0].liftoff_s, places=6
        )
        self.assertAlmostEqual(low[0].liftoff_s, 5.04, delta=0.05)

    def test_controlled_but_supported_episode_is_not_identifiable(self):
        episodes = detect_control_episodes(
            _recording(airborne=False), _settings()
        )

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].status, "not_identifiable")
        self.assertEqual(
            episodes[0].reason, "no_persistent_liftoff"
        )

    def test_command_gap_is_a_mask_not_a_new_control_episode(self):
        recording = _recording(command_gap=True)
        episodes = detect_control_episodes(recording, _settings())
        query = np.asarray((6.5, 7.2, 7.8))

        self.assertEqual(len(episodes), 1)
        np.testing.assert_array_equal(
            command_valid_mask(recording, query),
            [True, False, True],
        )

    def test_short_flight_state_gap_does_not_split_episode(self):
        episodes = detect_control_episodes(
            _recording(state_gap=True), _settings()
        )

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].status, "candidate")


if __name__ == "__main__":
    unittest.main()
