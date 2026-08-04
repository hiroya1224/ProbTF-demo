from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from grape_param_estim.sensor_models import (
    FlightData,
    FlightModeSeries,
    FlightProvenance,
    ImuPreflightCalibration,
    PidDebugSeries,
    PoseSeries,
    ReferenceSeries,
    SensorContract,
    TimeInterval,
    TimestampSource,
    TopicSensorContract,
    UsageDecision,
    VectorSeries,
)


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "ros"
    / "examples"
    / "grape-param-estim"
)


def _topic_contract(**overrides):
    values = {
        "topic": "/pose",
        "message_type": "geometry_msgs/PoseStamped",
        "timestamp_source": TimestampSource.HEADER,
        "usage": UsageDecision.USED,
        "frame_id": "world",
        "fields": ("position.x", "position.y", "position.z"),
        "units": ("m", "m", "m"),
        "sample_rate_hz": 100.0,
        "median_gap_seconds": 0.01,
        "maximum_gap_seconds": 0.025,
        "duplicate_timestamp_count": 0,
        "nonmonotonic_timestamp_count": 0,
        "covariance_provenance": "preflight static interval",
        "unavailable_reason": None,
        "mixed_frame_notes": None,
    }
    values.update(overrides)
    return TopicSensorContract(**values)


def _pose_series():
    return PoseSeries(
        times=np.asarray((10.0, 10.1)),
        record_times=np.asarray((10.002, 10.102)),
        positions=np.asarray(((1.0, 2.0, 3.0), (1.1, 2.1, 3.1))),
        orientations_xyzw=np.asarray(
            ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, -1.0))
        ),
        timestamp_source=TimestampSource.HEADER,
    )


def _reference_series():
    times = np.asarray((10.01, 10.06, 10.11))
    zeros = np.zeros((3, 3))
    return ReferenceSeries(
        times=times,
        record_times=times,
        position=zeros,
        linear_velocity=zeros,
        linear_acceleration=zeros,
        rpy=zeros,
        angular_velocity=zeros,
        angular_acceleration=zeros,
        timestamp_source=TimestampSource.RECORD,
    )


def _flight_mode_series():
    return FlightModeSeries(
        times=np.asarray((10.01, 10.06, 10.11)),
        record_times=np.asarray((100.01, 100.06, 100.11)),
        states=np.asarray((3, 3, 4), dtype=np.int64),
        initial_time=9.99,
        initial_record_time=99.99,
        initial_state=3,
        timestamp_source=TimestampSource.RECORD,
        source_topic="/flight_state",
        state_semantics="recorded controller mode",
    )


def _imu_preflight(with_accelerometer=False):
    return ImuPreflightCalibration(
        interval=TimeInterval(0.001, 6.4),
        state_value=0,
        imu_sample_count=1280,
        gyro_bias=np.asarray((1.0e-4, 2.0e-4, -2.0e-4)),
        gyro_standard_deviation=np.asarray((0.01, 0.01, 0.02)),
        specific_force_mean=np.asarray((0.0, 0.0, 9.8)),
        specific_force_standard_deviation=np.asarray((0.1, 0.1, 0.2)),
        specific_force_norm_mean=9.8,
        accelerometer_bias=(
            np.asarray((0.01, -0.02, -0.04))
            if with_accelerometer else None
        ),
        accelerometer_sample_count=(1200 if with_accelerometer else 0),
        gravity_magnitude=9.80665,
        frame_id="gimbalrotor/fc",
        timestamp_source=TimestampSource.HEADER,
        source_topic="/imu/converted",
        state_topic="/flight_state",
        orientation_topic=("/mocap/pose" if with_accelerometer else None),
        method="initial contiguous ARM_OFF arithmetic mean",
        accelerometer_unavailable_reason=(
            None
            if with_accelerometer
            else "physical_imu_origin_not_separately_calibrated"
        ),
    )


class TimeAndTopicContractTests(unittest.TestCase):
    def test_enums_and_interval_have_the_closed_planned_values(self):
        self.assertEqual(
            tuple(value.value for value in TimestampSource),
            ("header", "record"),
        )
        self.assertEqual(
            tuple(value.value for value in UsageDecision),
            (
                "used",
                "input",
                "initialization",
                "diagnostic",
                "disabled",
            ),
        )
        interval = TimeInterval(np.float64(18.0), 24.0)
        self.assertEqual(interval.duration, 6.0)
        self.assertIs(type(interval.start), float)
        with self.assertRaises(FrozenInstanceError):
            interval.start = 19.0
        for start, end in ((0.0, 0.0), (1.0, 0.0), (np.nan, 1.0)):
            with self.assertRaises((TypeError, ValueError)):
                TimeInterval(start, end)

    def test_topic_contract_retains_auditable_timing_and_provenance(self):
        contract = _topic_contract(
            duplicate_timestamp_count=np.int64(2),
            nonmonotonic_timestamp_count=np.int64(1),
            mixed_frame_notes="pose is world; twist is body-relative",
        )

        self.assertEqual(contract.topic, "/pose")
        self.assertEqual(contract.units, ("m", "m", "m"))
        self.assertEqual(contract.duplicate_timestamp_count, 2)
        self.assertIs(type(contract.duplicate_timestamp_count), int)
        self.assertEqual(
            contract.mixed_frame_notes,
            "pose is world; twist is body-relative",
        )

    def test_topic_contract_rejects_inconsistent_or_unusable_metadata(self):
        malformed = (
            {"timestamp_source": "header"},
            {"usage": "used"},
            {"fields": ("x", "y"), "units": ("m",)},
            {"fields": ("x", "x"), "units": ("m", "m")},
            {"sample_rate_hz": 0.0},
            {"median_gap_seconds": 0.02, "maximum_gap_seconds": 0.01},
            {"duplicate_timestamp_count": -1},
            {"nonmonotonic_timestamp_count": 1.5},
            {"covariance_provenance": ""},
            {"usage": UsageDecision.DISABLED, "unavailable_reason": None},
        )
        for overrides in malformed:
            with self.subTest(overrides=overrides):
                with self.assertRaises((TypeError, ValueError)):
                    _topic_contract(**overrides)

        disabled = _topic_contract(
            usage=UsageDecision.DISABLED,
            unavailable_reason="mixed frames are not a direct velocity",
        )
        self.assertEqual(disabled.usage, UsageDecision.DISABLED)

    def test_sensor_collection_is_unique_immutable_and_queryable(self):
        pose = _topic_contract()
        command = _topic_contract(
            topic="/command",
            message_type="std_msgs/Float64MultiArray",
            timestamp_source=TimestampSource.RECORD,
            usage=UsageDecision.INPUT,
            frame_id=None,
            fields=("rotor_1", "rotor_2"),
            units=("N", "N"),
            covariance_provenance=None,
        )
        contracts = SensorContract((pose, command))

        self.assertEqual(len(contracts), 2)
        self.assertEqual(tuple(contracts), (pose, command))
        self.assertIs(contracts.for_topic("/command"), command)
        with self.assertRaises(KeyError):
            contracts.for_topic("/missing")
        with self.assertRaises(FrozenInstanceError):
            contracts.topics = ()
        with self.assertRaises(ValueError):
            SensorContract((pose, pose))
        with self.assertRaises(TypeError):
            SensorContract([pose])


class AsynchronousSeriesTests(unittest.TestCase):
    def test_vector_series_preserves_both_clocks_and_freezes_copies(self):
        header_times = np.asarray((1.0, 1.1, 1.2))
        record_times = np.asarray((1.003, 1.104, 1.204))
        values = np.arange(6, dtype=float).reshape(3, 2)
        series = VectorSeries(
            times=header_times,
            record_times=record_times,
            values=values,
            field_names=("x", "y"),
            timestamp_source=TimestampSource.HEADER,
        )

        header_times[:] = -1.0
        record_times[:] = -2.0
        values[:] = -3.0
        np.testing.assert_array_equal(series.times, (1.0, 1.1, 1.2))
        np.testing.assert_array_equal(
            series.record_times, (1.003, 1.104, 1.204)
        )
        np.testing.assert_array_equal(
            series.values, np.arange(6, dtype=float).reshape(3, 2)
        )
        for value in (series.times, series.record_times, series.values):
            self.assertFalse(value.flags.writeable)

    def test_series_require_strict_time_shape_finite_values_and_clocks(self):
        base = {
            "times": np.asarray((1.0, 2.0)),
            "record_times": np.asarray((1.0, 2.0)),
            "values": np.zeros((2, 1)),
            "field_names": ("x",),
            "timestamp_source": TimestampSource.RECORD,
        }
        malformed = (
            {"times": np.asarray((1.0, 1.0))},
            {"record_times": np.asarray((1.0, 0.9))},
            {"times": np.asarray((1.0, np.nan))},
            {"record_times": np.asarray((1.0, 2.0, 3.0))},
            {"values": np.zeros((2, 2))},
            {"values": np.asarray(((0.0,), (np.inf,)))},
            {"timestamp_source": TimestampSource.HEADER,
             "times": np.asarray((1.0, 1.0))},
            {"times": np.asarray((1.0, 2.1))},
        )
        for overrides in malformed:
            values = dict(base)
            values.update(overrides)
            with self.subTest(overrides=overrides):
                with self.assertRaises((TypeError, ValueError)):
                    VectorSeries(**values)

        local_record_series = VectorSeries(
            times=np.asarray((0.0, 1.0)),
            record_times=np.asarray((100.0, 101.0)),
            values=np.zeros((2, 1)),
            field_names=("x",),
            timestamp_source=TimestampSource.RECORD,
        )
        np.testing.assert_array_equal(local_record_series.times, (0.0, 1.0))

    def test_pose_accepts_quaternion_sign_flips_but_requires_rotations(self):
        pose = _pose_series()
        np.testing.assert_array_equal(
            pose.orientations_xyzw[:, 3], (1.0, -1.0)
        )
        self.assertFalse(pose.positions.flags.writeable)
        self.assertFalse(pose.orientations_xyzw.flags.writeable)

        with self.assertRaisesRegex(ValueError, "unit quaternions"):
            PoseSeries(
                times=np.asarray((1.0,)),
                record_times=np.asarray((1.0,)),
                positions=np.zeros((1, 3)),
                orientations_xyzw=np.asarray(((0.0, 0.0, 0.0, 2.0),)),
                timestamp_source=TimestampSource.RECORD,
            )

    def test_pid_and_reference_fields_have_fixed_finite_shapes(self):
        times = np.asarray((1.0, 1.01))
        pid_values = np.arange(12, dtype=float).reshape(2, 6)
        pid = PidDebugSeries(
            times=times,
            record_times=times,
            axis_names=("x", "y", "z", "roll", "pitch", "yaw"),
            target_p=pid_values,
            error_p=pid_values,
            target_d=pid_values,
            error_d=pid_values,
            total=pid_values,
            p_term=pid_values,
            i_term=pid_values,
            d_term=pid_values,
            timestamp_source=TimestampSource.RECORD,
        )
        reference = _reference_series()

        for name in (
            "target_p",
            "error_p",
            "target_d",
            "error_d",
            "total",
            "p_term",
            "i_term",
            "d_term",
        ):
            self.assertEqual(getattr(pid, name).shape, (2, 6))
            self.assertFalse(getattr(pid, name).flags.writeable)
        for name in (
            "position",
            "linear_velocity",
            "linear_acceleration",
            "rpy",
            "angular_velocity",
            "angular_acceleration",
        ):
            self.assertEqual(getattr(reference, name).shape, (3, 3))
            self.assertFalse(getattr(reference, name).flags.writeable)

        with self.assertRaises(ValueError):
            PidDebugSeries(
                times=times,
                record_times=times,
                axis_names=("x", "y", "z", "roll", "pitch", "yaw"),
                target_p=np.zeros((2, 5)),
                error_p=pid_values,
                target_d=pid_values,
                error_d=pid_values,
                total=pid_values,
                p_term=pid_values,
                i_term=pid_values,
                d_term=pid_values,
                timestamp_source=TimestampSource.RECORD,
            )


class FlightDataTests(unittest.TestCase):
    def test_mode_and_preflight_contracts_are_immutable_and_sourced(self):
        mode = _flight_mode_series()
        calibration = _imu_preflight(with_accelerometer=True)

        self.assertFalse(mode.times.flags.writeable)
        self.assertFalse(mode.record_times.flags.writeable)
        self.assertFalse(mode.states.flags.writeable)
        self.assertEqual(mode.initial_state, 3)
        self.assertEqual(mode.source_topic, "/flight_state")
        for value in (
            calibration.gyro_bias,
            calibration.gyro_standard_deviation,
            calibration.specific_force_mean,
            calibration.specific_force_standard_deviation,
            calibration.accelerometer_bias,
        ):
            self.assertFalse(value.flags.writeable)
        self.assertEqual(calibration.accelerometer_sample_count, 1200)
        self.assertEqual(calibration.orientation_topic, "/mocap/pose")

        with self.assertRaisesRegex(ValueError, "causal"):
            FlightModeSeries(
                times=np.asarray((10.0, 10.1)),
                record_times=np.asarray((100.0, 100.1)),
                states=np.asarray((3, 3), dtype=np.int64),
                initial_time=10.05,
                initial_record_time=100.05,
                initial_state=3,
                timestamp_source=TimestampSource.RECORD,
                source_topic="/flight_state",
                state_semantics="controller mode",
            )
        with self.assertRaisesRegex(ValueError, "unavailable reason"):
            ImuPreflightCalibration(
                interval=TimeInterval(0.0, 1.0),
                state_value=0,
                imu_sample_count=10,
                gyro_bias=np.zeros(3),
                gyro_standard_deviation=np.zeros(3),
                specific_force_mean=np.asarray((0.0, 0.0, 9.8)),
                specific_force_standard_deviation=np.zeros(3),
                specific_force_norm_mean=9.8,
                accelerometer_bias=None,
                accelerometer_sample_count=0,
                gravity_magnitude=9.80665,
                frame_id="fc",
                timestamp_source=TimestampSource.HEADER,
                source_topic="/imu",
                state_topic="/flight_state",
                orientation_topic=None,
                method="mean",
                accelerometer_unavailable_reason=None,
            )

    def test_flight_data_keeps_streams_asynchronous_and_optional(self):
        pose = _pose_series()
        gyro_times = np.asarray((10.0, 10.025, 10.05, 10.075, 10.1))
        gyro = VectorSeries(
            times=gyro_times,
            record_times=gyro_times,
            values=np.zeros((5, 3)),
            field_names=("x", "y", "z"),
            timestamp_source=TimestampSource.RECORD,
        )
        reference = _reference_series()
        provenance = FlightProvenance(
            bag_path="/data/flight.bag",
            bag_sha256="0123456789abcdef",
            bag_size_bytes=np.int64(1024),
            bag_record_start=9.0,
            bag_record_end=12.0,
            adapter_revision="sensor-contract-v1",
            notes=("target interval selected by flight state",),
        )
        contract = SensorContract((_topic_contract(),))
        snapshot = object()
        flight = FlightData(
            bag_id="flight-a",
            interval=TimeInterval(10.0, 10.2),
            pose=pose,
            velocity=None,
            gyro=gyro,
            accelerometer=None,
            gimbal_position=None,
            gimbal_command=None,
            rotor_command=None,
            pid_debug=None,
            reference=reference,
            flight_mode=_flight_mode_series(),
            imu_preflight=_imu_preflight(),
            controller_snapshot=snapshot,
            sensor_contract=contract,
            provenance=provenance,
        )

        self.assertEqual(flight.pose.times.size, 2)
        self.assertEqual(flight.gyro.times.size, 5)
        self.assertEqual(flight.reference.times.size, 3)
        self.assertIsNone(flight.velocity)
        self.assertIs(flight.controller_snapshot, snapshot)
        self.assertEqual(provenance.bag_size_bytes, 1024)
        with self.assertRaises(FrozenInstanceError):
            flight.gyro = None

    def test_flight_data_rejects_wrong_nested_contract_types(self):
        base = {
            "bag_id": "flight-a",
            "interval": TimeInterval(10.0, 10.2),
            "pose": _pose_series(),
            "velocity": None,
            "gyro": None,
            "accelerometer": None,
            "gimbal_position": None,
            "gimbal_command": None,
            "rotor_command": None,
            "pid_debug": None,
            "reference": _reference_series(),
            "flight_mode": _flight_mode_series(),
            "imu_preflight": _imu_preflight(),
            "controller_snapshot": object(),
            "sensor_contract": SensorContract((_topic_contract(),)),
            "provenance": FlightProvenance(
                "/data/flight.bag", "hash", 1, 9.0, 12.0
            ),
        }
        malformed = (
            {"interval": (10.0, 10.2)},
            {"pose": object()},
            {"velocity": object()},
            {"pid_debug": object()},
            {"reference": object()},
            {"flight_mode": object()},
            {"imu_preflight": object()},
            {"controller_snapshot": None},
            {"sensor_contract": object()},
            {"provenance": object()},
        )
        for overrides in malformed:
            values = dict(base)
            values.update(overrides)
            with self.subTest(overrides=overrides):
                with self.assertRaises(TypeError):
                    FlightData(**values)

    def test_sensor_module_does_not_import_ros_or_legacy_adapter(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PACKAGE_ROOT / "src")
        script = (
            "import json,sys; import grape_param_estim.sensor_models; "
            "blocked=('grape_param_estim.real_rosbag',"
            "'grape_param_estim.controller','rospy','scipy'); "
            "print(json.dumps(sorted(name for name in sys.modules "
            "if any(name == root or name.startswith(root + '.') "
            "for root in blocked))))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=environment,
        )
        self.assertEqual(json.loads(completed.stdout), [])


if __name__ == "__main__":
    unittest.main()
