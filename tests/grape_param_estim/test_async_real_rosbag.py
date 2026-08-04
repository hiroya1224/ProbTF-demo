import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from grape_param_estim.real_rosbag import (
    ACC_ONLY_TOPIC,
    ASYNC_TOPIC_TYPE_CONTRACT,
    BASELINK_ODOM_TOPIC,
    COG_ODOM_TOPIC,
    CONVERTED_IMU_TOPIC,
    DEFAULT_AUDITED_GRAPE_BAG,
    FLIGHT_STATE_TOPIC,
    FOUR_AXIS_COMMAND_TOPIC,
    GAIN_TOPICS,
    GIMBAL_COMMAND_TOPIC,
    HeaderDuplicatePolicy,
    JOINT_STATE_TOPIC,
    MAIN_BODY_TO_FC_TRANSLATION,
    NATIVE_IMU_TOPIC,
    PID_AXIS_NAMES,
    PID_TOPIC,
    RAW_MOCAP_POSE_TOPIC,
    TF_STATIC_TOPIC,
    build_flight_data_from_messages,
    load_flight_data,
)
from grape_param_estim.sensor_models import (
    TimestampSource,
    UsageDecision,
)


def _namespace(**values):
    return SimpleNamespace(**values)


class _Stamp:
    def __init__(self, value):
        self.value = float(value)

    def to_sec(self):
        return self.value


def _vector(x, y, z):
    return _namespace(x=float(x), y=float(y), z=float(z))


def _quaternion(x=0.0, y=0.0, z=0.0, w=1.0):
    return _namespace(x=float(x), y=float(y), z=float(z), w=float(w))


def _header(stamp, frame_id=""):
    return _namespace(stamp=_Stamp(stamp), frame_id=frame_id)


def _pose_message(header_time, position, quaternion, frame_id="world"):
    return _namespace(
        header=_header(header_time, frame_id),
        pose=_namespace(
            position=_vector(*position),
            orientation=_quaternion(*quaternion),
        ),
    )


def _odom_message(header_time, linear, child="gimbalrotor/fc"):
    return _namespace(
        header=_header(header_time, "/world"),
        child_frame_id=child,
        pose=_namespace(
            pose=_namespace(
                position=_vector(0.0, 0.0, 0.0),
                orientation=_quaternion(),
            ),
            covariance=[0.0] * 36,
        ),
        twist=_namespace(
            twist=_namespace(
                linear=_vector(*linear),
                angular=_vector(9.0, 9.0, 9.0),
            ),
            covariance=[0.0] * 36,
        ),
    )


def _imu_message(header_time, gyro, acceleration, frame="gimbalrotor/fc"):
    return _namespace(
        header=_header(header_time, frame),
        angular_velocity=_vector(*gyro),
        linear_acceleration=_vector(*acceleration),
        angular_velocity_covariance=[0.0] * 9,
        linear_acceleration_covariance=[0.0] * 9,
    )


def _joint_message(header_time, offset):
    return _namespace(
        header=_header(header_time),
        name=("gimbal4", "gimbal2", "gimbal1", "gimbal3"),
        position=(offset + 4.0, offset + 2.0, offset + 1.0, offset + 3.0),
        velocity=(0.0, 0.0, 0.0, 0.0),
        effort=(90.0, 91.0, 92.0, 93.0),
    )


def _pid_axis(axis_index, sample_index):
    target_p = 10.0 * sample_index + axis_index
    target_d = 20.0 * sample_index + axis_index
    p_term = 0.1 + axis_index
    i_term = 0.2 + axis_index
    d_term = 0.3 + axis_index
    feedforward = 1.0 + axis_index
    return _namespace(
        target_p=target_p,
        err_p=-0.5 - axis_index,
        target_d=target_d,
        err_d=-0.25 - axis_index,
        total=[p_term + i_term + d_term + feedforward],
        p_term=[p_term],
        i_term=[i_term],
        d_term=[d_term],
    )


def _pid_message(header_time, sample_index):
    values = {
        name: _pid_axis(axis_index, sample_index)
        for axis_index, name in enumerate(PID_AXIS_NAMES)
    }
    values["header"] = _header(header_time)
    return _namespace(**values)


def _gain_message(values):
    return _namespace(
        doubles=tuple(
            _namespace(name=name, value=value)
            for name, value in zip(
                ("p_gain", "i_gain", "d_gain"), values
            )
        ),
        bools=(_namespace(name="pid_control_flag", value=False),),
    )


def _tf_static_message():
    transform = _namespace(
        header=_header(0.0, "gimbalrotor/main_body"),
        child_frame_id="gimbalrotor/fc",
        transform=_namespace(
            translation=_vector(*MAIN_BODY_TO_FC_TRANSLATION),
            rotation=_quaternion(),
        ),
    )
    return _namespace(transforms=(transform,))


def _synthetic_records(duplicate_pose_header=False):
    records = []

    def add(topic, message, record_time):
        records.append((topic, message, _Stamp(record_time)))

    add(TF_STATIC_TOPIC, _tf_static_message(), 100.01)
    gains = (
        (4.0, 0.1, 2.0),
        (5.0, 1.0, 2.5),
        (13.0, 1.0, 20.0),
        (6.0, 1.0, 2.0),
    )
    for gain_index, ((_group, topic), values) in enumerate(
        zip(GAIN_TOPICS, gains)
    ):
        add(topic, _gain_message(values), 100.1 + 0.1 * gain_index)

    # Full-bag sources retained for causal mode and preflight provenance.
    for record_time, state in (
        (100.01, 0),
        (101.0, 0),
        (102.0, 0),
        (103.0, 0),
        (104.0, 1),
        (110.0, 2),
        (117.0, 3),
        (118.01, 3),
        (118.21, 3),
        (118.41, 3),
    ):
        add(FLIGHT_STATE_TOPIC, _namespace(data=state), record_time)
    for index, header_time in enumerate((100.1, 101.1, 102.1, 103.1)):
        add(
            RAW_MOCAP_POSE_TOPIC,
            _pose_message(
                header_time,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            header_time + 0.002,
        )
        add(
            CONVERTED_IMU_TOPIC,
            _imu_message(
                header_time,
                (0.01, 0.02, 0.03),
                (0.1, -0.2, 9.81),
            ),
            header_time + 0.001,
        )

    # These record times lie inside the requested window but their header
    # times do not; header-time selection must exclude both without clamping.
    add(
        RAW_MOCAP_POSE_TOPIC,
        _pose_message(117.99, (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        118.03,
    )
    pose_header_times = (
        118.1,
        118.1 if duplicate_pose_header else 118.2,
        118.4,
    )
    for index, header_time in enumerate(pose_header_times):
        quaternion = (0.0, 0.0, 0.0, -1.0 if index == 1 else 1.0)
        add(
            RAW_MOCAP_POSE_TOPIC,
            _pose_message(
                header_time,
                (float(index), 0.0, 1.0),
                quaternion,
            ),
            header_time + 0.002 + 0.001 * index,
        )
    add(
        RAW_MOCAP_POSE_TOPIC,
        _pose_message(119.01, (9.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        118.99,
    )

    for index, header_time in enumerate((118.05, 118.25)):
        add(
            BASELINK_ODOM_TOPIC,
            _odom_message(header_time, (index + 0.1, 0.0, 0.0)),
            header_time + 0.006,
        )
        add(
            COG_ODOM_TOPIC,
            _odom_message(
                header_time, (99.0, 99.0, 99.0), child="gimbalrotor/cog"
            ),
            header_time + 0.0065,
        )

    for index, header_time in enumerate((118.02, 118.12, 118.22, 118.32)):
        add(
            CONVERTED_IMU_TOPIC,
            _imu_message(
                header_time,
                (0.01 * index, 0.02 * index, 0.03 * index),
                (0.1, -0.2, 9.81),
            ),
            header_time + 0.001,
        )
        add(
            ACC_ONLY_TOPIC,
            _namespace(
                header=_header(header_time),
                acc=_vector(0.1, -0.2, 9.81),
            ),
            header_time + 0.0011,
        )
        add(NATIVE_IMU_TOPIC, _namespace(index=index), header_time + 0.0012)

    for index, header_time in enumerate((118.04, 118.24)):
        add(
            JOINT_STATE_TOPIC,
            _joint_message(header_time, 10.0 * index),
            header_time + 0.001,
        )

    add(
        GIMBAL_COMMAND_TOPIC,
        _namespace(
            header=_header(117.89),
            name=(),
            position=(0.0, 0.0, 0.0, 0.0),
            velocity=(),
            effort=(),
        ),
        117.9,
    )
    add(
        FOUR_AXIS_COMMAND_TOPIC,
        _namespace(
            base_thrust=(0.5, 0.5, 0.5, 0.5),
            angles=(0.0, 0.0, 0.0),
        ),
        117.91,
    )
    for index, header_time in enumerate(
        (118.03, 118.13, 118.23, 118.33)
    ):
        add(
            GIMBAL_COMMAND_TOPIC,
            _namespace(
                header=_header(header_time),
                name=(),
                position=tuple(
                    index + value for value in (0.1, 0.2, 0.3, 0.4)
                ),
                velocity=(),
                effort=(),
            ),
            header_time + 0.001,
        )

    for index, record_time in enumerate((118.06, 118.16, 118.26)):
        add(
            FOUR_AXIS_COMMAND_TOPIC,
            _namespace(
                base_thrust=(1.0, 2.0, 3.0, 4.0),
                angles=(0.0, 0.0, 0.0),
            ),
            record_time,
        )
        # The first two PID headers deliberately duplicate; record time is the
        # audited authoritative clock for this topic.
        pid_header = 118.0 if index < 2 else 118.2
        add(
            PID_TOPIC,
            _pid_message(pid_header, index),
            record_time + 0.0005,
        )

    return tuple(sorted(records, key=lambda value: value[2].to_sec()))


def _build_synthetic(**overrides):
    values = {
        "records": _synthetic_records(),
        "topic_types": dict(ASYNC_TOPIC_TYPE_CONTRACT),
        "bag_path": "/tmp/synthetic-flight.bag",
        "bag_sha256": "synthetic-sha256",
        "bag_size_bytes": 1234,
        "bag_record_start": 100.0,
        "bag_record_end": 130.0,
        "start_local": 18.0,
        "end_local": 19.0,
    }
    values.update(overrides)
    return build_flight_data_from_messages(**values)


class AsynchronousRosbagAdapterTests(unittest.TestCase):
    def test_audited_streams_keep_independent_clocks_and_fields(self):
        data = _build_synthetic()

        self.assertEqual(data.interval.start, 18.0)
        self.assertEqual(data.interval.end, 19.0)
        self.assertEqual(data.pose.times.size, 3)
        self.assertEqual(data.velocity.times.size, 2)
        self.assertEqual(data.gyro.times.size, 4)
        self.assertEqual(data.gimbal_position.times.size, 2)
        self.assertEqual(data.gimbal_command.times.size, 4)
        self.assertEqual(data.rotor_command.times.size, 3)
        self.assertEqual(data.pid_debug.times.size, 3)
        self.assertEqual(data.reference.times.size, 3)
        self.assertIsNone(data.accelerometer)
        self.assertEqual(data.flight_mode.times.size, 3)
        self.assertEqual(data.flight_mode.initial_state, 3)
        np.testing.assert_array_equal(data.flight_mode.states, (3, 3, 3))
        self.assertEqual(data.imu_preflight.imu_sample_count, 4)
        self.assertGreaterEqual(data.gimbal_command.history_times.size, 1)
        self.assertGreaterEqual(data.rotor_command.history_times.size, 1)
        self.assertLess(
            data.gimbal_command.history_times[-1],
            data.gimbal_command.times[0],
        )
        self.assertLess(
            data.rotor_command.history_times[-1],
            data.rotor_command.times[0],
        )
        np.testing.assert_array_equal(
            data.rotor_command.history_values[-1],
            (0.5, 0.5, 0.5, 0.5),
        )
        np.testing.assert_allclose(
            data.imu_preflight.gyro_bias, (0.01, 0.02, 0.03)
        )
        self.assertIsNone(data.imu_preflight.accelerometer_bias)
        self.assertEqual(
            data.imu_preflight.accelerometer_unavailable_reason,
            "physical_imu_origin_not_separately_calibrated",
        )

        self.assertGreater(data.pose.times[0], data.interval.start)
        self.assertLess(data.pose.times[-1], data.interval.end)
        np.testing.assert_array_equal(
            data.pose.orientations_xyzw[:, 3], (1.0, -1.0, 1.0)
        )
        np.testing.assert_allclose(
            data.rotor_command.times,
            data.rotor_command.record_times - 100.0,
        )
        np.testing.assert_array_equal(
            data.gimbal_position.values[0], (1.0, 2.0, 3.0, 4.0)
        )
        np.testing.assert_allclose(
            data.reference.linear_acceleration[0], (1.0, 2.0, 3.0)
        )
        np.testing.assert_allclose(
            data.reference.angular_acceleration[0], (4.0, 5.0, 6.0)
        )

        pose_contract = data.sensor_contract.for_topic(
            RAW_MOCAP_POSE_TOPIC
        )
        velocity_contract = data.sensor_contract.for_topic(
            BASELINK_ODOM_TOPIC
        )
        imu_contract = data.sensor_contract.for_topic(CONVERTED_IMU_TOPIC)
        command_contract = data.sensor_contract.for_topic(
            FOUR_AXIS_COMMAND_TOPIC
        )
        pid_contract = data.sensor_contract.for_topic(PID_TOPIC)
        gimbal_command_contract = data.sensor_contract.for_topic(
            GIMBAL_COMMAND_TOPIC
        )
        mode_contract = data.sensor_contract.for_topic(FLIGHT_STATE_TOPIC)
        self.assertEqual(
            pose_contract.timestamp_source, TimestampSource.HEADER
        )
        self.assertEqual(pose_contract.frame_id, "world")
        self.assertEqual(velocity_contract.frame_id, "/world")
        self.assertIn("gimbalrotor/fc", velocity_contract.mixed_frame_notes)
        self.assertEqual(imu_contract.fields[0], "angular_velocity.x")
        self.assertEqual(
            imu_contract.unavailable_reason,
            "physical_imu_origin_not_separately_calibrated",
        )
        self.assertEqual(
            command_contract.timestamp_source, TimestampSource.RECORD
        )
        self.assertEqual(command_contract.usage, UsageDecision.INPUT)
        self.assertEqual(pid_contract.timestamp_source, TimestampSource.RECORD)
        self.assertEqual(pid_contract.duplicate_timestamp_count, 0)
        self.assertEqual(gimbal_command_contract.usage, UsageDecision.INPUT)
        self.assertEqual(
            gimbal_command_contract.timestamp_source,
            TimestampSource.RECORD,
        )
        self.assertEqual(mode_contract.usage, UsageDecision.INPUT)
        self.assertEqual(
            mode_contract.timestamp_source, TimestampSource.RECORD
        )
        self.assertIn(
            "PoseControlPid_header_duplicates=1",
            " ".join(data.provenance.notes),
        )
        self.assertIn(
            "main_body_to_fc_translation=",
            " ".join(data.provenance.notes),
        )

        self.assertEqual(
            data.sensor_contract.for_topic(COG_ODOM_TOPIC).usage,
            UsageDecision.DISABLED,
        )
        self.assertEqual(
            data.sensor_contract.for_topic(NATIVE_IMU_TOPIC).usage,
            UsageDecision.DISABLED,
        )
        self.assertEqual(
            data.sensor_contract.for_topic(ACC_ONLY_TOPIC).usage,
            UsageDecision.DISABLED,
        )
        self.assertIsNone(
            data.sensor_contract.for_topic(TF_STATIC_TOPIC).sample_rate_hz
        )
        self.assertIn(
            "measurement stamp",
            data.sensor_contract.for_topic(
                NATIVE_IMU_TOPIC
            ).mixed_frame_notes,
        )
        np.testing.assert_array_equal(
            data.controller_snapshot.gains,
            (
                (4.0, 0.1, 2.0),
                (5.0, 1.0, 2.5),
                (13.0, 1.0, 20.0),
                (6.0, 1.0, 2.0),
            ),
        )
        self.assertAlmostEqual(
            data.controller_configuration.pid[0].p_gain, 4.0
        )
        self.assertAlmostEqual(
            data.controller_configuration.initial_height, 0.0
        )
        np.testing.assert_allclose(
            data.sensor_extrinsics.pose_sensor_position_in_body,
            MAIN_BODY_TO_FC_TRANSLATION,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            data.sensor_extrinsics.velocity_sensor_position_in_body,
            MAIN_BODY_TO_FC_TRANSLATION,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            data.sensor_extrinsics.body_to_gyro_sensor_rotation,
            np.eye(3),
        )
        self.assertIn(
            "initial_height_source=median_preflight_raw_FC_pose_z",
            " ".join(data.provenance.notes),
        )

    def test_specific_force_requires_explicit_fc_origin_acceptance(self):
        data = _build_synthetic(include_fc_specific_force=True)

        self.assertIsNotNone(data.accelerometer)
        np.testing.assert_allclose(
            data.accelerometer.values[:, 2], np.full(4, 9.81)
        )
        contract = data.sensor_contract.for_topic(CONVERTED_IMU_TOPIC)
        self.assertIsNone(contract.unavailable_reason)
        self.assertIn("linear_acceleration.z", contract.fields)
        self.assertIn(
            "accepted_C_SB=I_and_known_FC_origin",
            " ".join(data.provenance.notes),
        )
        np.testing.assert_allclose(
            data.imu_preflight.accelerometer_bias,
            (0.1, -0.2, 9.81 - 9.80665),
            atol=1.0e-12,
        )
        self.assertEqual(
            data.imu_preflight.orientation_topic, RAW_MOCAP_POSE_TOPIC
        )

    def test_duplicate_header_policy_is_explicit_and_never_sorts(self):
        records = _synthetic_records(duplicate_pose_header=True)
        with self.assertRaisesRegex(ValueError, "duplicate header"):
            _build_synthetic(records=records)

        data = _build_synthetic(
            records=records,
            header_duplicate_policy=HeaderDuplicatePolicy.KEEP_FIRST,
        )
        self.assertEqual(data.pose.times.size, 2)
        contract = data.sensor_contract.for_topic(RAW_MOCAP_POSE_TOPIC)
        self.assertEqual(contract.duplicate_timestamp_count, 1)
        self.assertEqual(
            data.provenance.notes[1], "header_duplicate_policy=keep_first"
        )

    def test_unexpected_sensor_frame_or_tf_geometry_is_rejected(self):
        records = list(_synthetic_records())
        for index, (topic, message, stamp) in enumerate(records):
            if topic == CONVERTED_IMU_TOPIC:
                message.header.frame_id = "gimbalrotor/unknown_sensor"
                break
        with self.assertRaisesRegex(ValueError, "sensor frame"):
            _build_synthetic(records=tuple(records))

        records = list(_synthetic_records())
        for topic, message, _stamp in records:
            if topic == TF_STATIC_TOPIC:
                message.transforms[0].transform.translation.z += 0.01
        with self.assertRaisesRegex(ValueError, "audited URDF"):
            _build_synthetic(records=tuple(records))

    def test_covariance_sentinel_pid_shape_and_missing_optional_contract(self):
        records = list(_synthetic_records())
        for topic, message, _stamp in records:
            if topic == CONVERTED_IMU_TOPIC:
                message.angular_velocity_covariance[0] = -1.0
        data = _build_synthetic(records=tuple(records))
        self.assertIn(
            "marks estimate unavailable",
            data.sensor_contract.for_topic(
                CONVERTED_IMU_TOPIC
            ).covariance_provenance,
        )

        records = list(_synthetic_records())
        for topic, message, _stamp in records:
            if topic == PID_TOPIC:
                message.roll.total = (1.0, 2.0)
                break
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _build_synthetic(records=tuple(records))

        omitted = {COG_ODOM_TOPIC, NATIVE_IMU_TOPIC, ACC_ONLY_TOPIC}
        topic_types = {
            topic: message_type
            for topic, message_type in ASYNC_TOPIC_TYPE_CONTRACT
            if topic not in omitted
        }
        records = tuple(
            record
            for record in _synthetic_records()
            if record[0] not in omitted
        )
        data = _build_synthetic(
            records=records, topic_types=topic_types
        )
        for topic in omitted:
            contract = data.sensor_contract.for_topic(topic)
            self.assertEqual(contract.usage, UsageDecision.DISABLED)
            self.assertIn("topic_not_present", contract.unavailable_reason)


_RUN_BAG_INTEGRATION = os.environ.get(
    "GRAPE_RUN_ROSBAG_INTEGRATION"
) == "1"


@unittest.skipUnless(
    _RUN_BAG_INTEGRATION and Path(DEFAULT_AUDITED_GRAPE_BAG).is_file(),
    "set GRAPE_RUN_ROSBAG_INTEGRATION=1 with the audited bag available",
)
class AuditedBagIntegrationTests(unittest.TestCase):
    def test_repo_sample_18_24_has_audited_support_and_positive_g(self):
        data = load_flight_data(
            DEFAULT_AUDITED_GRAPE_BAG,
            start_local=18.0,
            end_local=24.0,
            include_fc_specific_force=True,
            compute_sha256=True,
        )

        self.assertEqual(
            data.provenance.bag_sha256,
            "bd3fc7f71797c0f5cb665acc50832da93c590e540fa170f9977182ecedf93bf8",
        )
        self.assertEqual(data.pose.times.size, 719)
        self.assertEqual(data.velocity.times.size, 600)
        self.assertEqual(data.gyro.times.size, 1200)
        self.assertEqual(data.gimbal_position.times.size, 300)
        self.assertEqual(data.gimbal_command.times.size, 1200)
        self.assertEqual(data.rotor_command.times.size, 1200)
        self.assertEqual(data.pid_debug.times.size, 1200)
        self.assertEqual(data.reference.times.size, 1200)
        self.assertEqual(data.flight_mode.times.size, 1200)
        self.assertEqual(data.flight_mode.initial_state, 3)
        self.assertTrue(np.all(data.flight_mode.states == 3))
        self.assertEqual(data.imu_preflight.imu_sample_count, 1280)
        np.testing.assert_allclose(
            data.imu_preflight.gyro_bias,
            (1.44e-4, 2.23e-4, -2.03e-4),
            atol=8.0e-6,
        )
        self.assertIsNotNone(data.imu_preflight.accelerometer_bias)
        self.assertEqual(data.imu_preflight.accelerometer_sample_count, 1242)
        self.assertAlmostEqual(
            data.imu_preflight.specific_force_norm_mean, 9.75579, places=4
        )
        self.assertTrue(
            np.all(np.isfinite(data.imu_preflight.accelerometer_bias))
        )
        self.assertEqual(
            data.sensor_contract.for_topic(GIMBAL_COMMAND_TOPIC).usage,
            UsageDecision.INPUT,
        )
        self.assertEqual(
            data.sensor_contract.for_topic(FLIGHT_STATE_TOPIC).usage,
            UsageDecision.INPUT,
        )
        self.assertEqual(data.gyro.field_names, ("x", "y", "z"))
        self.assertTrue(np.all(data.pose.times >= 18.0))
        self.assertTrue(np.all(data.pose.times <= 24.0))
        self.assertTrue(np.all(data.velocity.times >= 18.0))
        self.assertTrue(np.all(data.velocity.times <= 24.0))
        np.testing.assert_allclose(
            data.rotor_command.times,
            data.rotor_command.record_times
            - data.provenance.bag_record_start,
            rtol=0.0,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            data.gimbal_command.times,
            data.gimbal_command.record_times
            - data.provenance.bag_record_start,
            rtol=0.0,
            atol=2.0e-7,
        )
        self.assertGreater(
            float(np.median(data.accelerometer.values[:, 2])), 8.0
        )
        self.assertIn(
            "PoseControlPid_header_duplicates=3",
            " ".join(data.provenance.notes),
        )
        imu_contract = data.sensor_contract.for_topic(CONVERTED_IMU_TOPIC)
        self.assertEqual(imu_contract.timestamp_source, TimestampSource.HEADER)
        self.assertGreater(imu_contract.sample_rate_hz, 199.0)
        self.assertLess(imu_contract.sample_rate_hz, 201.0)
        self.assertIn("unknown", imu_contract.covariance_provenance)
        np.testing.assert_array_equal(
            data.controller_snapshot.gains,
            (
                (3.0, 0.1, 1.0),
                (5.0, 1.0, 2.5),
                (20.0, 1.0, 8.0),
                (4.0, 1.0, 2.0),
            ),
        )
        np.testing.assert_allclose(
            data.sensor_extrinsics.pose_sensor_position_in_body,
            MAIN_BODY_TO_FC_TRANSLATION,
            rtol=0.0,
            atol=1.0e-15,
        )
        self.assertTrue(
            np.isfinite(data.controller_configuration.initial_height)
        )


if __name__ == "__main__":
    unittest.main()
