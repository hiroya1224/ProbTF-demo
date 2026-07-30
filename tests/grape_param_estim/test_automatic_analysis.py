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
from grape_param_estim.failure_bag import ControllerRecording


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
    position[:, 2] -= 0.3 * np.clip(
        (timestamps - 10.0) / 0.2, 0.0, 1.0
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
    angular_response = np.gradient(
        angular_velocity, step, axis=0
    )
    pid_total = np.column_stack(
        (
            specific_force / np.asarray((0.9, 1.1, 0.8)),
            angular_response / np.asarray((1.2, 0.95, 0.7)),
        )
    )
    orientation = np.zeros((timestamps.size, 4), dtype=float)
    orientation[:, 3] = 1.0
    controller = ControllerRecording(
        times=timestamps,
        total=pid_total,
        proportional=0.6 * pid_total,
        integral=0.1 * pid_total,
        derivative=0.3 * pid_total,
        gain_times={
            group: np.asarray((0.0,))
            for group in ("xy", "z", "roll_pitch", "yaw")
        },
        gain_values={
            "xy": np.asarray(((3.0, 0.1, 1.0),)),
            "z": np.asarray(((5.0, 1.0, 2.5),)),
            "roll_pitch": np.asarray(((20.0, 1.0, 8.0),)),
            "yaw": np.asarray(((4.0, 1.0, 2.0),)),
        },
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
        orientation_xyzw=orientation,
        controller=controller,
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
        self.assertEqual(
            config.controller_topics["pid_debug"],
            "/gimbalrotor/debug/pose/pid",
        )

    def test_detection_estimation_trace_and_browser_report(self):
        progress = []
        result = analyze_recordings(
            [_automatic_recording()],
            _config(),
            progress_callback=lambda fraction, phase: progress.append(
                (fraction, phase)
            ),
        )

        self.assertEqual(result["schema"], RESULT_SCHEMA)
        self.assertAlmostEqual(progress[-1][0], 1.0)
        self.assertTrue(
            np.all(np.diff([row[0] for row in progress]) >= 0.0)
        )
        self.assertIn(
            "block bootstrap", {row[1] for row in progress}
        )
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
        self.assertIn(
            "support_contact",
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
        advice = episode["controller_advice"]
        self.assertEqual(advice["status"], "available")
        self.assertTrue(advice["airborne_only"])
        self.assertEqual(
            advice["alignment_lag_s"],
            episode["estimate"]["selected_alignment_lag_s"],
        )
        z_advice = next(
            row for row in advice["groups"] if row["group"] == "z"
        )
        self.assertEqual(
            z_advice["status"], "proposal_available"
        )
        self.assertEqual(
            z_advice["minimum_log_change"]["decision"],
            "apply_bounded_first_step",
        )
        self.assertAlmostEqual(
            z_advice["response_scale"]["estimate"],
            0.8,
            delta=0.08,
        )
        self.assertTrue(
            z_advice["non_identifiability_ridge"]["points"]
        )
        self.assertGreater(
            z_advice["minimum_log_change"]["proposed_pid"]["p"],
            z_advice["current_pid"]["p"],
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.html"
            render_browser_report(result, destination)
            text = destination.read_text(encoding="utf-8")
            self.assertIn("Grape failure-bag automatic analysis", text)
            self.assertIn("<svg", text)
            self.assertIn("specific_force_x_gain", text)
            self.assertIn("PID / model first-step advice", text)
            self.assertIn(
                "response_scale = actuator_scale / "
                "physical_parameter_ratio",
                text,
            )


if __name__ == "__main__":
    unittest.main()
