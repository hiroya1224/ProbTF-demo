from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.automatic_analysis import (
    AutomaticAnalysisConfig,
    RESULT_SCHEMA,
    analyze_recordings,
    load_automatic_config,
)
from grape_param_estim.browser_report import render_browser_report
from grape_param_estim.episode_detection import (
    EpisodeDetectionSettings,
)
from grape_param_estim.failure_bag import FailureBagRecording


def _automatic_recording():
    step = 0.01
    timestamps = np.arange(0.0, 14.0 + 0.5 * step, step)
    flight_state = np.zeros(timestamps.size, dtype=int)
    flight_state[(timestamps >= 3.0) & (timestamps < 11.0)] = 3
    flight_state[(timestamps >= 10.5) & (timestamps < 11.0)] = 17
    flight_state[timestamps >= 11.0] = 6

    position = np.zeros((timestamps.size, 3), dtype=float)
    position[:, 2] = 1.7 + 0.3 * np.clip(
        (timestamps - 5.0) / 0.5, 0.0, 1.0
    )
    velocity = np.gradient(position, step, axis=0)

    command_mask = (timestamps >= 3.0) & (timestamps <= 11.0)
    command_times = timestamps[command_mask]
    wrench = np.column_stack(
        (
            2.0 * np.sin(2.0 * np.pi * 0.43 * command_times),
            1.5
            * np.sin(2.0 * np.pi * 0.57 * command_times + 0.2),
            20.0
            + 3.0 * np.sin(2.0 * np.pi * 0.31 * command_times),
            0.8
            * np.sin(2.0 * np.pi * 0.47 * command_times + 0.4),
            0.7
            * np.sin(2.0 * np.pi * 0.61 * command_times + 0.1),
            0.4
            * np.sin(2.0 * np.pi * 0.73 * command_times + 0.6),
        )
    )
    command_index = np.searchsorted(
        command_times, timestamps - 0.04, side="right"
    ) - 1
    command_index = np.clip(command_index, 0, command_times.size - 1)
    delayed = wrench[command_index]
    specific_force = delayed[:, :3] * np.asarray((1.2, 0.8, 0.5))
    angular_acceleration = delayed[:, 3:] * np.asarray((1.1, 0.9, 0.7))
    angular_velocity = np.cumsum(
        angular_acceleration * step, axis=0
    )
    rng = np.random.default_rng(9)
    specific_force += rng.normal(
        0.0, 0.005, specific_force.shape
    )
    angular_velocity += rng.normal(
        0.0, 0.0003, angular_velocity.shape
    )
    return FailureBagRecording(
        bag_path="/synthetic/automatic.bag",
        bag_sha256="c" * 64,
        bag_start_time=100.0,
        bag_duration_s=14.0,
        command_times=command_times,
        command_wrench=wrench,
        imu_times=timestamps,
        specific_force=specific_force,
        angular_velocity=angular_velocity,
        state_times=timestamps,
        position=position,
        linear_velocity=velocity,
        flight_state_times=timestamps,
        flight_state=flight_state,
    )


def _config():
    estimation = {
        "sample_rate_hz": 50.0,
        "smoothing_window_s": 0.04,
        "maximum_delay_s": 0.08,
        "delay_step_s": 0.01,
        "bootstrap_samples": 20,
        "bootstrap_block_s": 0.2,
        "huber_delta": 1.5,
        "ridge": 1.0e-8,
        "seed": 7,
        "minimum_input_std": 0.001,
        "minimum_r2": 0.05,
        "maximum_relative_interval_width": 4.0,
    }
    return AutomaticAnalysisConfig(
        raw={"schema": "test"},
        sha256="d" * 64,
        topics={
            "command": "/command",
            "gimbal": "/gimbal",
            "imu": "/imu",
            "odometry": "/odom",
            "flight_state": "/flight_state",
        },
        detection=EpisodeDetectionSettings(
            active_flight_states=(3, 4, 5, 17),
            diagnostic_flight_states=(17,),
            baseline_window_s=2.0,
            minimum_active_duration_s=0.5,
            minimum_liftoff_height_m=0.02,
            minimum_airborne_duration_s=0.5,
            persistence_s=0.2,
            standardized_threshold=6.0,
        ),
        estimation=estimation,
        parameter_trace_step_s=0.5,
    )


class AutomaticAnalysisTests(unittest.TestCase):
    def test_repository_automatic_config_loads(self):
        root = Path(__file__).resolve().parents[2]
        config = load_automatic_config(
            root
            / "ros/examples/grape-param-estim/config"
            / "automatic_failure_analysis.yaml"
        )

        self.assertEqual(
            config.detection.active_flight_states, (3, 4, 5, 17)
        )
        self.assertEqual(
            config.detection.diagnostic_flight_states, (17,)
        )

    def test_detection_estimation_trace_and_browser_report(self):
        result = analyze_recordings(
            [_automatic_recording()], _config()
        )

        self.assertEqual(result["schema"], RESULT_SCHEMA)
        self.assertEqual(result["bag_count"], 1)
        episode = result["bags"][0]["episodes"][0]
        self.assertEqual(episode["status"], "estimated")
        self.assertGreater(
            episode["estimate"]["fit_sample_count"], 100
        )
        self.assertGreater(len(episode["parameter_trace"]), 2)
        self.assertTrue(episode["selection"]["fit_intervals"])
        self.assertIn(
            "diagnostic_flight_state_17",
            {
                interval["reason"]
                for interval in episode["selection"][
                    "failure_diagnostic_intervals"
                ]
            },
        )
        self.assertAlmostEqual(
            episode["estimate"]["parameters"][
                "specific_force_x_gain"
            ]["estimate"],
            1.2,
            delta=0.10,
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.html"
            render_browser_report(result, destination)
            text = destination.read_text(encoding="utf-8")
            self.assertIn("Grape failure-bag automatic analysis", text)
            self.assertIn("<svg", text)
            self.assertIn("specific_force_x_gain", text)


if __name__ == "__main__":
    unittest.main()
