from dataclasses import FrozenInstanceError, replace
import unittest

import numpy as np

from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.geometry import quaternion_to_matrix
from grape_param_estim.initialization import build_flight_initialization
from grape_param_estim.sensor_models import (
    FlightData,
    FlightModeSeries,
    FlightProvenance,
    ImuPreflightCalibration,
    PidDebugSeries,
    PoseSeries,
    ReferenceSeries,
    SensorContract,
    SensorExtrinsics,
    TimeInterval,
    TimestampSource,
    TopicSensorContract,
    UsageDecision,
    VectorSeries,
)


_PID_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


class _ControllerSnapshot:
    def __init__(self):
        self._gains = np.zeros((6, 3), dtype=float)
        self._gains[:, 1] = (1.0, 2.0, 0.0, 4.0, 5.0, 6.0)

    def axis_gains(self):
        return self._gains.copy()


def _sensor_extrinsics():
    return SensorExtrinsics(
        body_frame="main_body",
        pose_sensor_frame="fc",
        velocity_sensor_frame="fc",
        gyro_sensor_frame="fc",
        pose_sensor_position_in_body=np.zeros(3),
        pose_sensor_to_body_rotation=np.eye(3),
        velocity_sensor_position_in_body=np.zeros(3),
        velocity_sensor_to_body_rotation=np.eye(3),
        gyro_sensor_position_in_body=np.zeros(3),
        body_to_gyro_sensor_rotation=np.eye(3),
        source="synthetic identity extrinsics",
    )


def _record_times(times):
    return 100.0 + np.asarray(times, dtype=float)


def _vector_series(times, values, field_names, timestamp_source):
    times = np.asarray(times, dtype=float)
    return VectorSeries(
        times=times,
        record_times=_record_times(times),
        values=np.asarray(values, dtype=float),
        field_names=tuple(field_names),
        timestamp_source=timestamp_source,
    )


def _pose_series(sign_flips=True):
    times = np.linspace(0.0, 1.0, 6)
    positions = np.column_stack((times, 2.0 * times, -times))
    yaw = 0.4 * times
    quaternions = np.column_stack(
        (
            np.zeros(times.size),
            np.zeros(times.size),
            np.sin(0.5 * yaw),
            np.cos(0.5 * yaw),
        )
    )
    if sign_flips:
        quaternions[1::2] *= -1.0
    return PoseSeries(
        times=times,
        record_times=_record_times(times),
        positions=positions,
        orientations_xyzw=quaternions,
        timestamp_source=TimestampSource.HEADER,
    )


def _pid_debug():
    times = np.asarray((0.01, 0.21, 0.41, 0.61, 0.81, 0.99))
    integral = times[:, None] * np.asarray((1.0, -2.0, 0.0, 0.5, 1.5, -1.0))
    integral_gains = np.asarray((1.0, 2.0, 0.0, 4.0, 5.0, 6.0))
    i_term = integral * integral_gains
    zeros = np.zeros((times.size, 6), dtype=float)
    return PidDebugSeries(
        times=times,
        record_times=_record_times(times),
        axis_names=_PID_AXES,
        target_p=zeros,
        error_p=zeros,
        target_d=zeros,
        error_d=zeros,
        total=i_term,
        p_term=zeros,
        i_term=i_term,
        d_term=zeros,
        timestamp_source=TimestampSource.RECORD,
    )


def _reference(times):
    times = np.asarray(times, dtype=float)
    zeros = np.zeros((times.size, 3), dtype=float)
    return ReferenceSeries(
        times=times,
        record_times=_record_times(times),
        position=zeros,
        linear_velocity=zeros,
        linear_acceleration=zeros,
        rpy=zeros,
        angular_velocity=zeros,
        angular_acceleration=zeros,
        timestamp_source=TimestampSource.RECORD,
    )


def _preflight(include_accelerometer=False):
    return ImuPreflightCalibration(
        interval=TimeInterval(0.001, 0.05),
        state_value=0,
        imu_sample_count=12,
        gyro_bias=np.asarray((0.01, -0.02, 0.03)),
        gyro_standard_deviation=np.asarray((0.001, 0.002, 0.003)),
        specific_force_mean=np.asarray((0.0, 0.0, 9.8)),
        specific_force_standard_deviation=np.asarray((0.1, 0.1, 0.1)),
        specific_force_norm_mean=9.8,
        accelerometer_bias=(
            np.asarray((0.1, -0.2, 0.3))
            if include_accelerometer else None
        ),
        accelerometer_sample_count=(10 if include_accelerometer else 0),
        gravity_magnitude=9.80665,
        frame_id="fc",
        timestamp_source=TimestampSource.HEADER,
        source_topic="/imu/converted",
        state_topic="/flight_state",
        orientation_topic=("/mocap/pose" if include_accelerometer else None),
        method="initial contiguous ARM_OFF arithmetic mean",
        accelerometer_unavailable_reason=(
            None
            if include_accelerometer
            else "physical_imu_origin_not_separately_calibrated"
        ),
    )


def _sensor_contract():
    return SensorContract(
        (
            TopicSensorContract(
                topic="/mocap/pose",
                message_type="geometry_msgs/PoseStamped",
                timestamp_source=TimestampSource.HEADER,
                usage=UsageDecision.USED,
                frame_id="world",
                fields=("position",),
                units=("m",),
                sample_rate_hz=5.0,
                median_gap_seconds=0.2,
                maximum_gap_seconds=0.2,
                duplicate_timestamp_count=0,
                nonmonotonic_timestamp_count=0,
            ),
        )
    )


def _flight_data(sign_flips=True, include_accelerometer=False):
    velocity_times = np.asarray((0.05, 0.35, 0.65, 0.95))
    velocity = _vector_series(
        velocity_times,
        np.tile((4.0, -3.0, 2.0), (velocity_times.size, 1)),
        ("x", "y", "z"),
        TimestampSource.HEADER,
    )
    gyro_times = np.asarray((0.02, 0.22, 0.42, 0.62, 0.82, 0.98))
    gyro_bias = np.asarray((0.01, -0.02, 0.03))
    gyro = _vector_series(
        gyro_times,
        np.tile(gyro_bias + np.asarray((0.3, -0.2, 0.1)),
                (gyro_times.size, 1)),
        ("x", "y", "z"),
        TimestampSource.HEADER,
    )
    gimbal_times = np.asarray((0.04, 0.24, 0.44, 0.64, 0.84, 0.96))
    gimbal = _vector_series(
        gimbal_times,
        gimbal_times[:, None] * np.asarray((1.0, 2.0, 3.0, 4.0)),
        ("gimbal1", "gimbal2", "gimbal3", "gimbal4"),
        TimestampSource.HEADER,
    )
    command_times = np.asarray((0.03, 0.23, 0.43, 0.63, 0.83, 0.97))
    command = _vector_series(
        command_times,
        command_times[:, None] + np.asarray((10.0, 20.0, 30.0, 40.0)),
        ("rotor_1", "rotor_2", "rotor_3", "rotor_4"),
        TimestampSource.RECORD,
    )
    pid = _pid_debug()
    accelerometer = None
    if include_accelerometer:
        accelerometer = _vector_series(
            gyro_times,
            np.tile((0.1, -0.2, 9.8), (gyro_times.size, 1)),
            ("x", "y", "z"),
            TimestampSource.HEADER,
        )
    return FlightData(
        bag_id="synthetic-flight",
        interval=TimeInterval(0.0, 1.0),
        pose=_pose_series(sign_flips=sign_flips),
        velocity=velocity,
        gyro=gyro,
        accelerometer=accelerometer,
        gimbal_position=gimbal,
        gimbal_command=None,
        rotor_command=command,
        pid_debug=pid,
        reference=_reference(pid.times),
        flight_mode=FlightModeSeries(
            times=np.asarray((0.1, 0.5, 0.9)),
            record_times=np.asarray((100.1, 100.5, 100.9)),
            states=np.asarray((3, 3, 3), dtype=np.int64),
            initial_time=0.0,
            initial_record_time=100.0,
            initial_state=3,
            timestamp_source=TimestampSource.RECORD,
            source_topic="/flight_state",
            state_semantics="controller mode ZOH",
        ),
        imu_preflight=_preflight(include_accelerometer),
        controller_snapshot=_ControllerSnapshot(),
        controller_configuration=object(),
        sensor_extrinsics=_sensor_extrinsics(),
        sensor_contract=_sensor_contract(),
        provenance=FlightProvenance(
            bag_path="/tmp/synthetic.bag",
            bag_sha256="synthetic-sha256",
            bag_size_bytes=100,
            bag_record_start=100.0,
            bag_record_end=101.1,
        ),
    )


def _knot_matrix(initialization, kind):
    return np.asarray(
        tuple(
            initialization.state.knot_value(
                initialization.bag_id, index, kind
            )
            for index in range(initialization.grid.count)
        )
    )


class FlightInitializationTests(unittest.TestCase):
    def test_direct_sources_build_canonical_immutable_batch_state(self):
        flight = _flight_data()
        original_pose_times = flight.pose.times.copy()
        original_pose_values = flight.pose.positions.copy()

        result = build_flight_initialization(
            flight, 0.1, pose_smoothing_window=1
        )

        np.testing.assert_allclose(
            result.grid.times, np.arange(0.1, 1.0, 0.1), atol=1.0e-15
        )
        self.assertEqual(result.grid.common_support.start, 0.05)
        self.assertEqual(result.grid.common_support.end, 0.95)
        np.testing.assert_allclose(
            _knot_matrix(result, VariableKind.POSITION),
            np.column_stack(
                (result.grid.times, 2.0 * result.grid.times,
                 -result.grid.times)
            ),
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            _knot_matrix(result, VariableKind.LINEAR_VELOCITY),
            np.tile((4.0, -3.0, 2.0), (result.grid.count, 1)),
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            _knot_matrix(result, VariableKind.ANGULAR_VELOCITY),
            np.tile((0.3, -0.2, 0.1), (result.grid.count, 1)),
            atol=1.0e-15,
        )
        integral_scale = np.asarray((1.0, -2.0, 0.0, 0.5, 1.5, -1.0))
        np.testing.assert_allclose(
            _knot_matrix(result, VariableKind.CONTROLLER_INTEGRAL),
            result.grid.times[:, None] * integral_scale,
            atol=2.0e-15,
        )
        expected_command_indices = np.searchsorted(
            flight.rotor_command.times, result.grid.times, side="right"
        ) - 1
        np.testing.assert_array_equal(
            _knot_matrix(result, VariableKind.ACTUATOR_THRUST),
            flight.rotor_command.values[expected_command_indices],
        )
        np.testing.assert_allclose(
            _knot_matrix(result, VariableKind.GIMBAL_ANGLE),
            result.grid.times[:, None] * np.asarray((1.0, 2.0, 3.0, 4.0)),
            atol=2.0e-15,
        )
        np.testing.assert_array_equal(
            result.state.value(VariableKey(VariableKind.STATIC_PARAMETERS)),
            np.zeros(18),
        )
        np.testing.assert_array_equal(
            result.state.value(
                VariableKey(VariableKind.GYRO_BIAS, bag_id=flight.bag_id)
            ),
            flight.imu_preflight.gyro_bias,
        )
        self.assertNotIn(
            VariableKey(
                VariableKind.ACCELEROMETER_BIAS, bag_id=flight.bag_id
            ),
            result.layout,
        )

        np.testing.assert_array_equal(flight.pose.times, original_pose_times)
        np.testing.assert_array_equal(flight.pose.positions, original_pose_values)
        self.assertIsNot(result.grid.times, flight.pose.times)
        self.assertIn(
            "remain asynchronous and unchanged",
            result.provenance.observation_policy,
        )
        self.assertEqual(
            result.provenance.for_field("actuator_thrust").method,
            "causal zero-order hold at recorded command issue time",
        )
        self.assertIsNone(
            result.provenance.for_field("linear_velocity").fallback_reason
        )
        self.assertFalse(result.grid.times.flags.writeable)
        with self.assertRaises(ValueError):
            result.grid.times[0] = 10.0
        with self.assertRaises(FrozenInstanceError):
            result.bag_id = "other"

    def test_orientation_path_is_sign_invariant_and_stays_on_so3(self):
        flipped = build_flight_initialization(
            _flight_data(sign_flips=True),
            0.1,
            pose_smoothing_window=3,
        )
        continuous = build_flight_initialization(
            _flight_data(sign_flips=False),
            0.1,
            pose_smoothing_window=3,
        )
        flipped_rotations = _knot_matrix(
            flipped, VariableKind.ORIENTATION_TANGENT
        )
        continuous_rotations = _knot_matrix(
            continuous, VariableKind.ORIENTATION_TANGENT
        )
        np.testing.assert_allclose(
            flipped_rotations, continuous_rotations, atol=2.0e-15
        )
        for rotation in flipped_rotations:
            np.testing.assert_allclose(
                rotation.T @ rotation, np.eye(3), atol=2.0e-15
            )
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=14)

    def test_gyro_initialization_applies_numeric_sensor_rotation(self):
        flight = _flight_data()
        body_omega = np.asarray((0.3, -0.2, 0.1))
        body_to_sensor = np.asarray(
            (
                (0.0, -1.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        sensor_values = np.tile(
            flight.imu_preflight.gyro_bias
            + body_to_sensor @ body_omega,
            (flight.gyro.times.size, 1),
        )
        rotated = replace(
            flight,
            gyro=replace(flight.gyro, values=sensor_values),
            sensor_extrinsics=replace(
                flight.sensor_extrinsics,
                body_to_gyro_sensor_rotation=body_to_sensor,
                source="synthetic rotated IMU extrinsics",
            ),
        )
        result = build_flight_initialization(
            rotated, 0.1, pose_smoothing_window=1
        )
        np.testing.assert_allclose(
            _knot_matrix(result, VariableKind.ANGULAR_VELOCITY),
            np.tile(body_omega, (result.grid.count, 1)),
            rtol=0.0,
            atol=2.0e-15,
        )
        self.assertIn(
            "C_SB transpose",
            result.provenance.for_field("angular_velocity").method,
        )

    def test_missing_velocity_and_gyro_use_only_initial_path_fallbacks(self):
        flight = replace(_flight_data(), velocity=None, gyro=None)
        result = build_flight_initialization(
            flight, 0.1, pose_smoothing_window=1
        )

        np.testing.assert_allclose(
            _knot_matrix(result, VariableKind.LINEAR_VELOCITY),
            np.tile((1.0, 2.0, -1.0), (result.grid.count, 1)),
            atol=3.0e-15,
        )
        np.testing.assert_allclose(
            _knot_matrix(result, VariableKind.ANGULAR_VELOCITY),
            np.tile((0.0, 0.0, 0.4), (result.grid.count, 1)),
            atol=3.0e-15,
        )
        self.assertIn(
            "direct velocity observation is unavailable",
            result.provenance.for_field(
                "linear_velocity"
            ).fallback_reason,
        )
        self.assertIn(
            "gyro observation is unavailable",
            result.provenance.for_field(
                "angular_velocity"
            ).fallback_reason,
        )
        self.assertNotIn(
            VariableKey(VariableKind.GYRO_BIAS, bag_id=flight.bag_id),
            result.layout,
        )

    def test_integral_replay_then_explicit_zero_are_audited_fallbacks(self):
        flight = replace(_flight_data(), pid_debug=None)

        with self.assertRaisesRegex(ValueError, "integral initialization"):
            build_flight_initialization(
                flight, 0.1, pose_smoothing_window=1
            )

        def replay(_flight, knot_times):
            self.assertFalse(knot_times.flags.writeable)
            return np.tile(
                (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                (knot_times.size, 1),
            )

        replayed = build_flight_initialization(
            flight,
            0.1,
            pose_smoothing_window=1,
            integral_replay=replay,
        )
        np.testing.assert_array_equal(
            _knot_matrix(replayed, VariableKind.CONTROLLER_INTEGRAL),
            np.tile((1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                    (replayed.grid.count, 1)),
        )
        replay_provenance = replayed.provenance.for_field(
            "controller_integral"
        )
        self.assertEqual(replay_provenance.source, "integral_replay callback")
        self.assertIn("PID debug is unavailable",
                      replay_provenance.fallback_reason)

        zero = build_flight_initialization(
            flight,
            0.1,
            pose_smoothing_window=1,
            allow_zero_integral_fallback=True,
        )
        np.testing.assert_array_equal(
            _knot_matrix(zero, VariableKind.CONTROLLER_INTEGRAL), 0.0
        )
        self.assertIn(
            "replay was not provided",
            zero.provenance.for_field(
                "controller_integral"
            ).fallback_reason,
        )

    def test_accelerometer_bias_block_requires_preflight_calibration(self):
        flight = _flight_data(include_accelerometer=True)
        result = build_flight_initialization(
            flight, 0.1, pose_smoothing_window=1
        )
        accelerometer_key = VariableKey(
            VariableKind.ACCELEROMETER_BIAS, bag_id=flight.bag_id
        )
        np.testing.assert_array_equal(
            result.state.value(accelerometer_key),
            flight.imu_preflight.accelerometer_bias,
        )
        self.assertEqual(
            result.provenance.for_field(
                "accelerometer_bias"
            ).source_sample_count,
            10,
        )

        malformed = replace(flight, imu_preflight=_preflight(False))
        with self.assertRaisesRegex(ValueError, "preflight bias"):
            build_flight_initialization(
                malformed, 0.1, pose_smoothing_window=1
            )

    def test_requested_grid_never_extrapolates_and_options_are_validated(self):
        flight = _flight_data()
        with self.assertRaisesRegex(ValueError, "requires extrapolation"):
            build_flight_initialization(
                flight,
                0.1,
                knot_interval=TimeInterval(0.0, 0.9),
                pose_smoothing_window=1,
            )
        narrowed = build_flight_initialization(
            flight,
            0.1,
            knot_interval=TimeInterval(0.1, 0.9),
            pose_smoothing_window=1,
        )
        np.testing.assert_allclose(
            narrowed.grid.times, np.arange(0.1, 1.0, 0.1)
        )
        for period in (0.0, -0.1, np.nan):
            with self.subTest(period=period):
                with self.assertRaises((TypeError, ValueError)):
                    build_flight_initialization(flight, period)
        with self.assertRaisesRegex(ValueError, "positive odd"):
            build_flight_initialization(
                flight, 0.1, pose_smoothing_window=2
            )
        no_command = replace(flight, rotor_command=None)
        with self.assertRaisesRegex(ValueError, "rotor command"):
            build_flight_initialization(no_command, 0.1)

    def test_bad_replay_shape_and_nonzero_zero_gain_pid_are_rejected(self):
        without_pid = replace(_flight_data(), pid_debug=None)
        with self.assertRaisesRegex(ValueError, "finite shape"):
            build_flight_initialization(
                without_pid,
                0.1,
                pose_smoothing_window=1,
                integral_replay=lambda _flight, times: np.zeros(
                    (times.size, 5)
                ),
            )

        flight = _flight_data()
        i_term = flight.pid_debug.i_term.copy()
        i_term[:, 2] = 1.0
        malformed_pid = replace(
            flight.pid_debug,
            i_term=i_term,
        )
        malformed = replace(flight, pid_debug=malformed_pid)
        with self.assertRaisesRegex(ValueError, "zero I gain"):
            build_flight_initialization(
                malformed, 0.1, pose_smoothing_window=1
            )

    def test_unsmoothed_orientation_matches_expected_geodesic(self):
        result = build_flight_initialization(
            _flight_data(), 0.1, pose_smoothing_window=1
        )
        rotations = _knot_matrix(result, VariableKind.ORIENTATION_TANGENT)
        for index, time in enumerate(result.grid.times):
            yaw = 0.4 * time
            expected_quaternion = np.asarray(
                (0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw))
            )
            np.testing.assert_allclose(
                rotations[index],
                quaternion_to_matrix(expected_quaternion),
                atol=2.0e-15,
            )


if __name__ == "__main__":
    unittest.main()
