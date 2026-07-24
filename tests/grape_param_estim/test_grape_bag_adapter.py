import csv
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from grape_param_estim.episode import stable_hash
from grape_param_estim.grape_bag_adapter import (
    BagIntervalData,
    analyze_interval,
    load_vertical_slice_config,
    write_vertical_slice_artifact,
)
import grape_param_estim.grape_bag_adapter as adapter


REPOSITORY = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPOSITORY
    / "ros/examples/grape-param-estim/config/counterfactual.yaml"
)


class GrapeBagAdapterTests(unittest.TestCase):
    def config(self):
        config = load_vertical_slice_config(CONFIG_PATH)
        config["trajectory_sample_count"] = 2
        config["sample_rate_hz"] = 10.0
        config["effective_response"] = {
            "delay_grid_s": [0.0],
            "time_constant_grid_s": [0.08],
            "posterior_sample_count": 8,
            "em_iterations": 1,
            "position_sigma": 0.03,
            "velocity_sigma": 0.08,
        }
        config["candidate_grid"] = {
            "roll_pitch_p": [10.0],
            "roll_pitch_d": [12.0],
            "allocation_scale": [1.0],
        }
        config["config_sha256"] = stable_hash(
            {
                key: value
                for key, value in config.items()
                if key != "config_sha256"
            }
        )
        return config

    def data(self):
        times = np.arange(0.0, 4.01, 0.05)
        position = np.zeros((times.size, 3))
        position[:, 0] = 0.02 * np.sin(times)
        quaternion = np.zeros((times.size, 4))
        quaternion[:, 3] = 1.0
        acceleration = np.zeros((times.size, 3))
        acceleration[:, 2] = 9.80665
        desired = np.zeros((times.size, 6))
        desired[:, 0] = 0.02 * np.sin(times)
        desired_velocity = np.zeros_like(desired)
        desired_velocity[:, 0] = 0.02 * np.cos(times)
        command = np.zeros_like(desired)
        command[:, 0] = -0.02 * np.sin(times)
        return BagIntervalData(
            episode_id="20260612-04",
            stratum="synthetic_unit",
            bag_path="/immutable/unit.bag",
            source_bag_sha256="a" * 64,
            bag_start_time=100.0,
            interval_start_offset_s=0.0,
            interval_end_offset_s=4.0,
            mocap_times=times,
            mocap_positions=position,
            mocap_quaternions=quaternion,
            imu_times=times,
            accelerometer=acceleration,
            gyro=np.zeros_like(acceleration),
            pid_times=times,
            desired_position_euler=desired,
            desired_velocity=desired_velocity,
            nominal_acceleration=command,
            four_axis_thrust=np.full((20, 4), 2.0),
            roll_pitch_gain={"p_gain": 20.0, "d_gain": 8.0},
            flight_states=(3, 5),
            topic_counts={"/mocap": times.size},
            header_record_offset_median_s={
                "mocap": 0.0,
                "imu": 0.0,
                "pid": 0.0,
            },
            observed_header_frames={
                "mocap_pose": ("world",),
                "imu": ("gimbalrotor/fc",),
                "controller_pid": ("",),
            },
        )

    def test_config_freezes_same_pipeline_for_bags_4_7_8(self):
        config = load_vertical_slice_config(CONFIG_PATH)
        self.assertEqual(
            {item["episode_id"] for item in config["episodes"]},
            {"20260612-04", "20260612-07", "20260612-08"},
        )
        self.assertEqual(
            config["exact_controller"]["status"], "ORACLE_UNAVAILABLE"
        )
        self.assertEqual(config["conventions"]["world_frame"], "ENU")
        self.assertNotIn(
            "verify_hash", inspect.signature(adapter.read_bag_interval).parameters
        )

    def test_source_commit_detects_staged_and_untracked_work(self):
        with mock.patch.object(
            adapter.subprocess,
            "check_output",
            side_effect=["abc123\n", "?? untracked.py\n"],
        ):
            with self.assertRaisesRegex(RuntimeError, "clean source tree"):
                adapter._source_commit(REPOSITORY)
        with mock.patch.object(
            adapter.subprocess,
            "check_output",
            side_effect=["abc123\n", ""],
        ):
            self.assertEqual(adapter._source_commit(REPOSITORY), "abc123")
        with mock.patch.object(
            adapter.subprocess,
            "check_output",
            side_effect=["abc123\n", ""],
        ):
            self.assertEqual(
                adapter._source_commit(REPOSITORY, explicit="abc123"),
                "abc123",
            )
        with mock.patch.object(
            adapter.subprocess,
            "check_output",
            side_effect=["abc123\n", ""],
        ):
            with self.assertRaisesRegex(ValueError, "does not match"):
                adapter._source_commit(
                    REPOSITORY, explicit="not-the-current-revision"
                )

    def test_vertical_slice_writes_honest_diagnostic_artifact(self):
        config = self.config()
        summary, arrays = analyze_interval(
            self.data(), config, source_commit="unit-test-commit"
        )
        self.assertEqual(summary["workflow_status"], "EXPERIMENTAL")
        self.assertFalse(summary["recommendation_available"])
        self.assertFalse(
            summary["gates"]["exact_controller_replay"]["passed"]
        )
        self.assertEqual(arrays["actual_position_mean"].shape[1], 6)
        with tempfile.TemporaryDirectory(
            prefix="grape-real-bag-artifact-"
        ) as directory:
            destination = write_vertical_slice_artifact(
                directory, summary, arrays, config
            )
            loaded = json.loads(
                (destination / "summary.json").read_text()
            )
            self.assertEqual(loaded["run_id"], summary["run_id"])
            self.assertFalse(loaded["recommendation_available"])
            with (destination / "candidate_grid.csv").open(
                newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["success_probability"], "")
            self.assertEqual(
                rows[0]["exact_evaluation_status"],
                "ORACLE_UNAVAILABLE",
            )
            self.assertTrue((destination / "trajectory.csv").is_file())
            self.assertIn(
                "Recommendation available: `false`",
                (destination / "REPORT.md").read_text(),
            )
            with self.assertRaises(FileExistsError):
                write_vertical_slice_artifact(
                    directory, summary, arrays, config
                )


if __name__ == "__main__":
    unittest.main()
