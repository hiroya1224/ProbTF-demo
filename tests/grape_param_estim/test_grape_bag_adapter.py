import csv
from dataclasses import replace
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from grape_param_estim.episode import sha256_file, stable_hash
from grape_param_estim.grape_bag_adapter import (
    BagIntervalData,
    analyze_interval,
    build_vertical_slice_analysis_records,
    load_vertical_slice_config,
    materialize_vertical_slice_analysis_bag,
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
            topic_counts={
                "/gimbalrotor/mocap/pose": times.size,
                "/gimbalrotor/sensor_plugin/imu1/ros_converted": times.size,
                "/gimbalrotor/debug/pose/pid": times.size,
                "/gimbalrotor/four_axes/command": 20,
                "/gimbalrotor/controller/roll_pitch/parameter_updates": 1,
                "/gimbalrotor/flight_state": 2,
            },
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

    def test_input_slice_hash_binds_diagnostics_and_metadata(self):
        data = self.data()
        baseline = adapter._input_slice_sha256(data)
        variants = (
            replace(
                data,
                four_axis_thrust=np.array(
                    data.four_axis_thrust, copy=True
                )
                + 0.01,
            ),
            replace(data, roll_pitch_gain={"p_gain": 21.0}),
            replace(data, flight_states=(3,)),
            replace(data, topic_counts=dict(data.topic_counts, extra=1)),
            replace(
                data,
                header_record_offset_median_s={
                    **data.header_record_offset_median_s,
                    "imu": 0.001,
                },
            ),
            replace(
                data,
                observed_header_frames={
                    **data.observed_header_frames,
                    "imu": ("other",),
                },
            ),
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(
                    adapter._input_slice_sha256(variant),
                    baseline,
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
            self.assertNotIn(
                b"\r\n",
                (destination / "trajectory.csv").read_bytes(),
            )
            self.assertNotIn(
                b"\r\n",
                (destination / "candidate_grid.csv").read_bytes(),
            )
            expected_files = {
                "REPORT.md",
                "artifact_manifest.json",
                "candidate_grid.csv",
                "summary.json",
                "trajectory.csv",
                "trajectory_particles.npz",
            }
            self.assertEqual(
                {path.name for path in destination.iterdir()},
                expected_files,
            )
            with np.load(
                str(destination / "trajectory_particles.npz"),
                allow_pickle=False,
            ) as archive:
                archived_arrays = {
                    name: archive[name]
                    for name in adapter._TRAJECTORY_EVIDENCE_FIELDS
                }
                self.assertEqual(
                    str(archive["trajectory_evidence_sha256"]),
                    summary["trajectory_evidence_sha256"],
                )
            self.assertEqual(
                adapter._trajectory_evidence_sha256(archived_arrays),
                summary["trajectory_evidence_sha256"],
            )
            manifest = json.loads(
                (destination / "artifact_manifest.json").read_text()
            )
            self.assertEqual(manifest["run_id"], summary["run_id"])
            self.assertEqual(
                set(manifest["files"]),
                expected_files - {"artifact_manifest.json"},
            )
            for name, metadata in manifest["files"].items():
                self.assertEqual(
                    metadata["sha256"],
                    sha256_file(destination / name),
                )
                self.assertEqual(
                    metadata["size_bytes"],
                    (destination / name).stat().st_size,
                )
            tampered_arrays = dict(arrays)
            tampered_arrays["actual_position_samples"] = np.array(
                arrays["actual_position_samples"], copy=True
            )
            tampered_arrays["actual_position_samples"][0, 0, 0] += 0.1
            with self.assertRaisesRegex(
                ValueError, "trajectory_evidence_sha256"
            ):
                write_vertical_slice_artifact(
                    directory,
                    summary,
                    tampered_arrays,
                    config,
                )
            tampered_config = dict(config)
            tampered_config["candidate_grid"] = {
                **config["candidate_grid"],
                "allocation_scale": [9.0],
            }
            with self.assertRaisesRegex(ValueError, "config_sha256"):
                write_vertical_slice_artifact(
                    directory,
                    summary,
                    arrays,
                    tampered_config,
                )
            self.assertIn(
                "Recommendation available: `false`",
                (destination / "REPORT.md").read_text(),
            )
            with self.assertRaises(FileExistsError):
                write_vertical_slice_artifact(
                    directory, summary, arrays, config
                )

    def test_ros_records_preserve_matched_samples_and_provenance(self):
        try:
            from grape_param_estim.msg import (
                ModelMismatch,
                TrajectoryParticleSet,
            )
        except ImportError as error:
            self.skipTest("built ROS 1 messages unavailable: {}".format(error))

        data = replace(self.data(), interval_end_offset_s=3.97)
        summary, arrays = analyze_interval(
            data,
            self.config(),
            source_commit="unit-test-commit",
        )

        records = build_vertical_slice_analysis_records(summary, arrays)
        trajectory_records = [
            item
            for item in records
            if isinstance(item.message, TrajectoryParticleSet)
        ]
        mismatch_records = [
            item for item in records if isinstance(item.message, ModelMismatch)
        ]
        self.assertEqual(len(trajectory_records), 3)
        self.assertEqual(
            len(mismatch_records), len(arrays["time_offset_s"])
        )
        self.assertFalse(
            any("counterfactual" in item.topic for item in records)
        )
        by_kind = {
            item.message.trajectory_id.rsplit(":", 1)[-1]: item.message
            for item in trajectory_records
        }
        actual = by_kind["actual_posterior"]
        nominal = by_kind["nominal"]
        desired = by_kind["desired"]
        point_count = len(arrays["time_offset_s"])
        sample_count = len(arrays["actual_sample_ids"])
        self.assertEqual(actual.trajectory_length, point_count)
        self.assertEqual(len(actual.stamps), point_count)
        self.assertEqual(len(actual.transforms), sample_count * point_count)
        self.assertEqual(len(actual.twists), sample_count * point_count)
        self.assertEqual(actual.sample_ids, nominal.sample_ids)
        self.assertEqual(actual.sample_weights, nominal.sample_weights)
        self.assertNotEqual(actual.model_version, nominal.model_version)
        self.assertNotEqual(desired.model_version, nominal.model_version)
        self.assertAlmostEqual(sum(actual.sample_weights), 1.0)
        self.assertEqual(desired.sample_ids, [0])
        self.assertEqual(desired.sample_weights, [1.0])
        time_index = point_count // 2
        flattened_index = point_count + time_index
        self.assertAlmostEqual(
            actual.transforms[flattened_index].translation.x,
            arrays["actual_position_samples"][1, time_index, 0],
        )
        quaternion = actual.transforms[flattened_index].rotation
        self.assertAlmostEqual(
            np.linalg.norm(
                [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
            ),
            1.0,
        )
        self.assertEqual(actual.header.frame_id, "world")
        self.assertEqual(actual.source_commit, "unit-test-commit")
        self.assertEqual(
            actual.source_bag_sha256, [summary["source_bag_sha256"]]
        )
        self.assertEqual(
            actual.normalized_dataset_sha256,
            [summary["input_slice_sha256"]],
        )
        self.assertIn(
            summary["trajectory_evidence_sha256"],
            actual.provenance.detail,
        )
        self.assertEqual(
            actual.stamps[0].to_nsec(),
            100_000_000_000,
        )
        self.assertEqual(
            actual.stamps[-1].to_nsec(),
            103_900_000_000,
        )
        self.assertEqual(
            actual.source_interval_end.to_nsec(),
            103_970_000_000,
        )
        self.assertTrue(
            all(
                100_000_000_000
                <= item.record_time_ns
                <= 103_970_000_000
                for item in records
            )
        )
        mismatch = mismatch_records[time_index].message
        model_samples = np.asarray(
            [
                adapter._se3_log_residual(
                    arrays["nominal_position_samples"][
                        sample_index, time_index
                    ],
                    arrays["actual_position_samples"][
                        sample_index, time_index
                    ],
                )
                for sample_index in range(sample_count)
            ]
        )
        weights = arrays["actual_sample_weights"]
        expected_mean = np.average(
            model_samples, axis=0, weights=weights
        )
        centered = model_samples - expected_mean
        expected_covariance = np.matmul(
            (centered * weights[:, None]).T,
            centered,
        )
        np.testing.assert_allclose(
            mismatch.model_residual_mean,
            expected_mean,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            mismatch.model_residual_covariance,
            expected_covariance.reshape(-1),
            atol=1.0e-12,
        )
        self.assertGreaterEqual(
            np.linalg.eigvalsh(expected_covariance).min(),
            -1.0e-12,
        )
        self.assertIn("matched_sample_SE3_log_residual", mismatch.diagnostics)
        self.assertIn("nominal_not_exact_pc_mcu_replay", mismatch.diagnostics)
        serialized = io.BytesIO()
        mismatch.serialize(serialized)
        self.assertGreater(len(serialized.getvalue()), 0)

        tampered_summary = dict(summary)
        tampered_summary["run_id"] = "wrong-run-id"
        with self.assertRaisesRegex(ValueError, "run_id"):
            build_vertical_slice_analysis_records(
                tampered_summary, arrays
            )
        tampered_arrays = dict(arrays)
        tampered_arrays["actual_position_samples"] = np.array(
            arrays["actual_position_samples"], copy=True
        )
        tampered_arrays["actual_position_samples"][0, 0, 0] += 1.0e-6
        with self.assertRaisesRegex(
            ValueError, "trajectory_evidence_sha256"
        ):
            build_vertical_slice_analysis_records(
                summary, tampered_arrays
            )
        invalid_arrays = dict(arrays)
        invalid_arrays["actual_position_samples"] = np.asarray(
            arrays["actual_position_samples"]
        )[:, :-1]
        with self.assertRaisesRegex(ValueError, "samples"):
            build_vertical_slice_analysis_records(summary, invalid_arrays)

    def test_se3_log_residual_uses_relative_transform_coordinates(self):
        reference = np.asarray([0.0, 0.0, -10.0, 0.0, 0.0, -0.2])
        actual = np.asarray([0.0, 0.0, -9.0, 0.0, 0.0, -0.1])
        np.testing.assert_allclose(
            adapter._se3_log_residual(reference, actual),
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.1],
            atol=1.0e-12,
        )

    def test_materialized_analysis_bag_preserves_source_and_record_order(self):
        try:
            import rosbag
            import rospy
            from grape_param_estim.msg import (
                ModelMismatch,
                TrajectoryParticleSet,
            )
            from std_msgs.msg import Float32
        except ImportError as error:
            self.skipTest("ROS 1 bag runtime unavailable: {}".format(error))

        with tempfile.TemporaryDirectory(
            prefix="grape-vertical-slice-bag-"
        ) as directory:
            source = Path(directory) / "source.bag"
            output = Path(directory) / "analysis.bag"
            with rosbag.Bag(str(source), "w") as bag:
                bag.write(
                    "/raw",
                    Float32(data=1.0),
                    t=rospy.Time.from_sec(100.0),
                )
                bag.write(
                    "/raw",
                    Float32(data=2.0),
                    t=rospy.Time.from_sec(105.0),
                )
            source_hash = sha256_file(source)
            data = replace(
                self.data(),
                bag_path=str(source),
                source_bag_sha256=source_hash,
                interval_end_offset_s=3.97,
            )
            summary, arrays = analyze_interval(
                data,
                self.config(),
                source_commit="unit-test-commit",
            )
            metadata = materialize_vertical_slice_analysis_bag(
                source, output, summary, arrays
            )
            sidecar = output.with_suffix(".json")
            self.assertTrue(sidecar.is_file())
            sidecar_payload = json.loads(sidecar.read_text())
            self.assertEqual(sidecar_payload["run_id"], summary["run_id"])
            self.assertEqual(
                sidecar_payload["analysis_bag_sha256"],
                sha256_file(output),
            )
            self.assertEqual(
                sidecar_payload["trajectory_evidence_sha256"],
                summary["trajectory_evidence_sha256"],
            )
            self.assertEqual(
                metadata["analysis_metadata"], str(sidecar)
            )
            self.assertEqual(sha256_file(source), source_hash)
            self.assertTrue(metadata["source_bag_unchanged"])
            self.assertEqual(
                metadata["analysis_record_count"],
                3 + len(arrays["time_offset_s"]),
            )
            with rosbag.Bag(str(output), "r") as bag:
                records = list(bag.read_messages())
            record_times = [stamp.to_nsec() for _, _, stamp in records]
            self.assertEqual(record_times, sorted(record_times))
            raw = [
                message.data
                for topic, message, _ in records
                if topic == "/raw"
            ]
            self.assertEqual(raw, [1.0, 2.0])
            analysis = [
                (topic, message)
                for topic, message, _ in records
                if topic.startswith("/analysis/")
            ]
            self.assertEqual(
                sum(
                    isinstance(message, TrajectoryParticleSet)
                    or getattr(message, "_type", "")
                    == "grape_param_estim/TrajectoryParticleSet"
                    for _, message in analysis
                ),
                3,
            )
            self.assertEqual(
                sum(
                    isinstance(message, ModelMismatch)
                    or getattr(message, "_type", "")
                    == "grape_param_estim/ModelMismatch"
                    for _, message in analysis
                ),
                len(arrays["time_offset_s"]),
            )
            self.assertFalse(
                any("counterfactual" in topic for topic, _ in analysis)
            )


if __name__ == "__main__":
    unittest.main()
