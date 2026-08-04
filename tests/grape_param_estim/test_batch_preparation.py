import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.batch.graph_builder import (
    build_fixed_batch_problem,
    build_initial_batch_state,
)
from grape_param_estim.batch.preparation import (
    PreparationSelection,
    prepare_fixed_batch_graph_data,
)
from grape_param_estim.batch.state import StateScaling
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.batch_request import (
    ACCELEROMETER_BIAS_PRIOR_COVARIANCE_BLOCKS,
    BATCH_ESTIMATION_REQUEST_SCHEMA,
    FIXED_FACTOR_COVARIANCE_BLOCKS,
    INITIAL_STATE_PRIOR_COVARIANCE_BLOCKS,
    OBSERVATION_COVARIANCE_BLOCKS,
    OBSERVATION_FACTOR_NAMES,
    validate_batch_estimation_request,
)
from grape_param_estim.batch_artifact import file_sha256
from grape_param_estim.controller import ControllerConfig
from grape_param_estim.initialization import build_flight_initialization
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.sensor_models import (
    CausalVectorSeries,
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
from grape_param_estim.system import (
    ActuatorParameters,
    GrapeGeometry,
    VehicleParameters,
)


_PID_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
_GIMBAL_FIELDS = ("gimbal1", "gimbal2", "gimbal3", "gimbal4")
_ROTOR_FIELDS = ("rotor_1", "rotor_2", "rotor_3", "rotor_4")


class _ControllerSnapshot:
    def __init__(self, configuration):
        self._gains = np.asarray(
            tuple(
                (value.p_gain, value.i_gain, value.d_gain)
                for value in configuration.pid
            ),
            dtype=float,
        )

    def axis_gains(self):
        return self._gains.copy()


def _record_times(times):
    return 100.0 + np.asarray(times, dtype=float)


def _vector(times, values, fields, source=TimestampSource.HEADER):
    times = np.asarray(times, dtype=float)
    return VectorSeries(
        times=times,
        record_times=_record_times(times),
        values=np.asarray(values, dtype=float),
        field_names=tuple(fields),
        timestamp_source=source,
    )


def _causal_command(times, values, history_times, history_values, fields):
    times = np.asarray(times, dtype=float)
    history_times = np.asarray(history_times, dtype=float)
    return CausalVectorSeries(
        times=times,
        record_times=_record_times(times),
        values=np.asarray(values, dtype=float),
        field_names=tuple(fields),
        timestamp_source=TimestampSource.RECORD,
        history_times=history_times,
        history_record_times=_record_times(history_times),
        history_values=np.asarray(history_values, dtype=float),
    )


def _sensor_contract():
    return SensorContract(
        (
            TopicSensorContract(
                topic="/pose",
                message_type="geometry_msgs/PoseStamped",
                timestamp_source=TimestampSource.HEADER,
                usage=UsageDecision.USED,
                frame_id="world",
                fields=("position", "orientation"),
                units=("m", "1"),
                sample_rate_hz=40.0,
                median_gap_seconds=0.025,
                maximum_gap_seconds=0.025,
                duplicate_timestamp_count=0,
                nonmonotonic_timestamp_count=0,
            ),
        )
    )


def _flight_data(
    path,
    digest,
    bag_id,
    mode_states=(3, 3, 3),
    include_accelerometer=False,
):
    pose_times = np.arange(0.0, 0.3000001, 0.025)
    pose = PoseSeries(
        times=pose_times,
        record_times=_record_times(pose_times),
        positions=np.column_stack(
            (
                0.1 * pose_times,
                -0.05 * pose_times,
                1.0 + 0.02 * pose_times,
            )
        ),
        orientations_xyzw=np.tile(
            np.asarray((0.0, 0.0, 0.0, 1.0)),
            (pose_times.size, 1),
        ),
        timestamp_source=TimestampSource.HEADER,
    )
    sensor_times = np.arange(0.0, 0.3000001, 0.05)
    velocity = _vector(
        sensor_times,
        np.tile((0.1, -0.05, 0.02), (sensor_times.size, 1)),
        ("x", "y", "z"),
    )
    gyro = _vector(
        sensor_times,
        np.tile((0.01, -0.02, 0.03), (sensor_times.size, 1)),
        ("x", "y", "z"),
    )
    accelerometer_bias = np.asarray((0.04, -0.03, 0.06))
    accelerometer = None
    if include_accelerometer:
        accelerometer = _vector(
            sensor_times,
            np.tile(
                accelerometer_bias + np.asarray((0.0, 0.0, 9.80665)),
                (sensor_times.size, 1),
            ),
            ("x", "y", "z"),
        )
    gimbal_position = _vector(
        sensor_times,
        sensor_times[:, None] * np.asarray((0.1, -0.1, 0.08, -0.08)),
        _GIMBAL_FIELDS,
    )
    command_times = np.asarray((0.0, 0.06, 0.12, 0.18, 0.24, 0.30))
    history_times = np.asarray((-0.12, -0.06))
    rotor_values = 6.0 + command_times[:, None] + np.asarray(
        (0.0, 0.1, 0.2, 0.3)
    )
    rotor_history = 6.0 + history_times[:, None] + np.asarray(
        (0.0, 0.1, 0.2, 0.3)
    )
    rotor = _causal_command(
        command_times,
        rotor_values,
        history_times,
        rotor_history,
        _ROTOR_FIELDS,
    )
    gimbal_values = command_times[:, None] * np.asarray(
        (0.2, -0.2, 0.15, -0.15)
    )
    gimbal_history = history_times[:, None] * np.asarray(
        (0.2, -0.2, 0.15, -0.15)
    )
    gimbal_command = _causal_command(
        command_times,
        gimbal_values,
        history_times,
        gimbal_history,
        _GIMBAL_FIELDS,
    )
    pid_times = np.arange(0.0, 0.3000001, 0.05)
    proxy = pid_times[:, None] * np.asarray(
        (0.1, -0.1, 0.2, 0.03, -0.04, 0.02)
    )
    configuration = ControllerConfig.grape()
    integral_gains = np.asarray(
        tuple(value.i_gain for value in configuration.pid)
    )
    zeros6 = np.zeros((pid_times.size, 6), dtype=float)
    pid = PidDebugSeries(
        times=pid_times,
        record_times=_record_times(pid_times),
        axis_names=_PID_AXES,
        target_p=np.column_stack(
            (
                np.zeros((pid_times.size, 2)),
                np.ones(pid_times.size),
                np.zeros((pid_times.size, 3)),
            )
        ),
        error_p=zeros6,
        target_d=zeros6,
        error_d=zeros6,
        total=proxy * integral_gains[None, :],
        p_term=zeros6,
        i_term=proxy * integral_gains[None, :],
        d_term=zeros6,
        timestamp_source=TimestampSource.RECORD,
    )
    reference = ReferenceSeries(
        times=pid_times,
        record_times=_record_times(pid_times),
        position=pid.target_p[:, :3],
        linear_velocity=np.zeros((pid_times.size, 3)),
        linear_acceleration=np.zeros((pid_times.size, 3)),
        rpy=np.zeros((pid_times.size, 3)),
        angular_velocity=np.zeros((pid_times.size, 3)),
        angular_acceleration=np.zeros((pid_times.size, 3)),
        timestamp_source=TimestampSource.RECORD,
    )
    extrinsics = SensorExtrinsics(
        body_frame="main_body",
        pose_sensor_frame="fc",
        velocity_sensor_frame="fc",
        gyro_sensor_frame="fc",
        pose_sensor_position_in_body=np.asarray((-0.0173, -0.0011, 0.0571)),
        pose_sensor_to_body_rotation=np.eye(3),
        velocity_sensor_position_in_body=np.asarray(
            (-0.0173, -0.0011, 0.0571)
        ),
        velocity_sensor_to_body_rotation=np.eye(3),
        gyro_sensor_position_in_body=np.asarray((-0.0173, -0.0011, 0.0571)),
        body_to_gyro_sensor_rotation=np.eye(3),
        source="synthetic numeric extrinsics",
    )
    preflight = ImuPreflightCalibration(
        interval=TimeInterval(-1.0, -0.2),
        state_value=0,
        imu_sample_count=20,
        gyro_bias=np.asarray((0.001, -0.002, 0.003)),
        gyro_standard_deviation=np.asarray((0.0001, 0.0002, 0.0003)),
        specific_force_mean=np.asarray((0.0, 0.0, 9.80665)),
        specific_force_standard_deviation=np.asarray((0.01, 0.01, 0.01)),
        specific_force_norm_mean=9.80665,
        accelerometer_bias=(
            accelerometer_bias if include_accelerometer else None
        ),
        accelerometer_sample_count=(20 if include_accelerometer else 0),
        gravity_magnitude=9.80665,
        frame_id="fc",
        timestamp_source=TimestampSource.HEADER,
        source_topic="/imu",
        state_topic="/flight_state",
        orientation_topic=("/pose" if include_accelerometer else None),
        method="synthetic preflight static mean",
        accelerometer_unavailable_reason=(
            None
            if include_accelerometer
            else "sensor origin not calibrated"
        ),
    )
    return FlightData(
        bag_id=bag_id,
        interval=TimeInterval(0.0, 0.3),
        pose=pose,
        velocity=velocity,
        gyro=gyro,
        accelerometer=accelerometer,
        gimbal_position=gimbal_position,
        gimbal_command=gimbal_command,
        rotor_command=rotor,
        pid_debug=pid,
        reference=reference,
        flight_mode=FlightModeSeries(
            times=np.asarray((0.0, 0.15, 0.3)),
            record_times=np.asarray((100.0, 100.15, 100.3)),
            states=np.asarray(mode_states, dtype=np.int64),
            initial_time=-0.1,
            initial_record_time=99.9,
            initial_state=int(mode_states[0]),
            timestamp_source=TimestampSource.RECORD,
            source_topic="/flight_state",
            state_semantics="synthetic controller mode ZOH",
        ),
        imu_preflight=preflight,
        controller_snapshot=_ControllerSnapshot(configuration),
        controller_configuration=configuration,
        sensor_extrinsics=extrinsics,
        sensor_contract=_sensor_contract(),
        provenance=FlightProvenance(
            bag_path=str(path),
            bag_sha256=digest,
            bag_size_bytes=path.stat().st_size,
            bag_record_start=100.0,
            bag_record_end=100.4,
            adapter_revision="synthetic-preparation/v1",
        ),
    )


def _covariance(contract, source="project_configuration", full=False):
    coordinates, units = contract
    dimension = len(coordinates)
    values = np.eye(dimension)
    if full and dimension >= 2:
        values[0, 1] = 0.2
        values[1, 0] = 0.2
        values[0, 0] = 2.0
    return {
        "source": source,
        "representation": "full" if full else "diagonal",
        "coordinates": list(coordinates),
        "units": list(units),
        "values": values.tolist() if full else np.diag(values).tolist(),
    }


def _request_payload(root, bag_specs):
    bags = []
    for bag_id, path, digest in bag_specs:
        factors = {
            name: {
                "enabled": True,
                "disabled_reason": None,
                "covariances": {
                    block_name: _covariance(contract)
                    for block_name, contract in (
                        OBSERVATION_COVARIANCE_BLOCKS[name].items()
                    )
                },
            }
            for name in OBSERVATION_FACTOR_NAMES
        }
        factors["accelerometer"] = {
            "enabled": False,
            "disabled_reason": "sensor origin not calibrated",
            "covariances": None,
        }
        bags.append(
            {
                "bag_id": bag_id,
                "path": str(path),
                "sha256": digest,
                "interval_seconds": [0.0, 0.3],
                "observation_factors": factors,
                "fixed_factor_covariances": {
                    name: _covariance(contract, "numerical_tolerance")
                    for name, contract in FIXED_FACTOR_COVARIANCE_BLOCKS.items()
                },
                "initial_state_prior_covariances": {
                    name: _covariance(contract, "project_configuration")
                    for name, contract in (
                        INITIAL_STATE_PRIOR_COVARIANCE_BLOCKS.items()
                    )
                },
            }
        )
    return {
        "schema": BATCH_ESTIMATION_REQUEST_SCHEMA,
        "run_id": "preparation-test",
        "run_mode": "estimate_only",
        "resume": False,
        "output_directory": str(root / "run"),
        "bags": bags,
        "q": {
            "update_policy": "laplace_em",
            "residual_quantity": "body_wrench",
            "interval_model": "continuous_spectral_density",
            "component_names": ["x", "y", "z", "roll", "pitch", "yaw"],
            "component_units": ["N", "N", "N", "N*m", "N*m", "N*m"],
            "initial_diagonal": [25.0, 25.0, 25.0, 1.0, 1.0, 1.0],
            "floor_diagonal": [1.0e-8] * 6,
        },
        "parameter_prior": {
            "kind": "gaussian",
            "mean_coordinate": [0.0] * 18,
            "covariance": np.eye(18).tolist(),
        },
        "delay": {
            "prior_kind": "uniform",
            "bounds_seconds": [0.0, 0.08],
            "initial_seconds": 0.035,
            "coarse_grid_points": 5,
            "refinement_tolerance_seconds": 1.0e-5,
            "maximum_refinement_evaluations": 12,
        },
        "actuator_model": {
            "source": "test actuator calibration",
            "thrust_time_constant_seconds": 0.04,
            "gimbal_time_constant_seconds": 0.03,
            "minimum_thrust_newtons": 1.5,
            "maximum_thrust_newtons": 27.6145,
            "maximum_gimbal_angle_radians": 3.14,
            "maximum_gimbal_rate_radians_per_second": 6.0,
        },
        "knot_policy": {
            "period_seconds": 0.1,
            "origin": "interval_start",
            "maximum_measurement_gap_seconds": 0.11,
        },
        "interpolation_policy": {
            "euclidean": "linear",
            "orientation": "so3_geodesic",
            "command": "zoh_record_issue_time",
            "allow_extrapolation": False,
        },
        "controller_snapshot_policy": {
            "source": "bag_startup_parameter_updates",
            "require_constant_within_interval": True,
        },
        "mode_hypotheses": [
            {
                "mode_id": "recorded-mode",
                "bag_schedules": {
                    bag_id: {
                        "flight_state_source": "recorded_causal_schedule",
                        "integration_gate_source": "deterministic_replay",
                    }
                    for bag_id, _path, _digest in bag_specs
                },
            }
        ],
        "solver_settings": {
            "maximum_iterations": 20,
            "maximum_factorization_retries": 4,
            "maximum_model_evaluation_retries": 4,
            "acceptance_ratio": 1.0e-4,
            "gradient_tolerance": 1.0e-6,
            "scaled_step_tolerance": 1.0e-7,
            "relative_objective_tolerance": 1.0e-8,
            "initial_damping": 1.0e-3,
            "minimum_damping": 1.0e-12,
            "maximum_damping": 1.0e12,
        },
        "em_settings": {
            "maximum_iterations": 4,
            "minimum_iterations": 1,
            "maximum_repeated_q_rejections": 2,
            "maximum_repeated_lag_profile_failures": 2,
            "log_q_tolerance": 1.0e-3,
            "lag_tolerance": 1.0e-5,
            "map_objective_tolerance": 1.0e-5,
            "marginal_objective_tolerance": 1.0e-5,
            "q_acceptance_objective_tolerance": 0.0,
            "q_minimum_alpha": 0.125,
        },
        "mcmc_settings": {"enabled": False},
    }


class BatchPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bag = self.root / "flight-a.bag"
        self.bag.write_bytes(b"synthetic flight a")
        self.digest = file_sha256(self.bag)
        self.flight = _flight_data(
            self.bag, self.digest, "flight-a"
        )
        self.initialization = build_flight_initialization(
            self.flight, 0.1, pose_smoothing_window=1
        )
        self.payload = _request_payload(
            self.root, (("flight-a", self.bag, self.digest),)
        )
        self.chart = VehicleParameterChart(VehicleParameters.nominal())
        self.geometry = GrapeGeometry.grape()
        self.actuators = ActuatorParameters(
            thrust_time_constant=0.04,
            gimbal_time_constant=0.03,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _prepare(
        self,
        payload=None,
        flight=None,
        initialization=None,
        selection=None,
    ):
        request = validate_batch_estimation_request(
            self.payload if payload is None else payload
        )
        return prepare_fixed_batch_graph_data(
            request=request,
            flight_data=(self.flight if flight is None else flight,),
            initializations=(
                self.initialization
                if initialization is None
                else initialization,
            ),
            parameter_chart=self.chart,
            geometry=self.geometry,
            actuator_parameters=self.actuators,
            scaling=StateScaling.unit(),
            selection=(
                PreparationSelection(
                    mode_id="recorded-mode",
                    fixed_delay_seconds=0.035,
                    q_diagonal=np.asarray(
                        (25.0, 25.0, 25.0, 1.0, 1.0, 1.0)
                    ),
                    initial_parameter_coordinates=np.zeros(18),
                )
                if selection is None
                else selection
            ),
        )

    def test_prepares_target_like_async_graph_and_exact_zoh_switches(self):
        prepared = self._prepare()
        bag = prepared.bags[0]
        self.assertEqual(len(bag.knots), 4)
        self.assertEqual(len(bag.pose_measurements), 13)
        self.assertEqual(len(bag.velocity_measurements), 7)
        self.assertEqual(len(bag.gyro_measurements), 7)
        self.assertEqual(len(bag.controller_integral_measurements), 7)
        self.assertEqual(len(bag.actual_gimbal_measurements), 7)
        self.assertEqual(
            bag.pose_measurements[1].bracket.left_knot_index, 0
        )
        self.assertAlmostEqual(
            bag.pose_measurements[1].bracket.interpolation_fraction, 0.25
        )
        durations = tuple(
            segment.duration
            for segment in bag.actuator_intervals[0].delayed_command_segments
        )
        np.testing.assert_allclose(durations, (0.035, 0.06, 0.005))
        np.testing.assert_allclose(
            prepared.dynamics.q,
            (25.0, 25.0, 25.0, 1.0, 1.0, 1.0),
        )
        problem = build_fixed_batch_problem(prepared)
        state = build_initial_batch_state(prepared)
        self.assertTrue(np.isfinite(problem.linearize(state).sparse.objective))

    def test_outer_point_rebuilds_delay_q_and_static_initialization(self):
        coordinates = np.linspace(-0.01, 0.01, 18)
        prepared = self._prepare(
            selection=PreparationSelection(
                mode_id="recorded-mode",
                fixed_delay_seconds=0.02,
                q_diagonal=np.asarray((2.0, 2.1, 2.2, 1.0, 1.1, 1.2)),
                initial_parameter_coordinates=coordinates,
            )
        )
        self.assertEqual(prepared.fixed_delay, 0.02)
        np.testing.assert_array_equal(prepared.dynamics.q, (2, 2.1, 2.2, 1, 1.1, 1.2))
        np.testing.assert_array_equal(
            prepared.initial_parameter_coordinates, coordinates
        )
        durations = tuple(
            segment.duration
            for segment in prepared.bags[0]
            .actuator_intervals[0]
            .delayed_command_segments
        )
        np.testing.assert_allclose(durations, (0.02, 0.06, 0.02))

    def test_full_covariance_and_disabled_stream_are_preserved_exactly(self):
        payload = copy.deepcopy(self.payload)
        velocity_factor = payload["bags"][0]["observation_factors"][
            "velocity"
        ]
        velocity_factor["covariances"]["velocity_observation"] = _covariance(
            OBSERVATION_COVARIANCE_BLOCKS["velocity"][
                "velocity_observation"
            ],
            full=True,
        )
        gyro_factor = payload["bags"][0]["observation_factors"]["gyro"]
        gyro_factor.update(
            enabled=False,
            disabled_reason="gyro excluded for this test",
            covariances=None,
        )
        flight = replace(self.flight, gyro=None)
        prepared = self._prepare(payload=payload, flight=flight)
        covariance = prepared.bags[0].covariances.velocity_observation.value
        np.testing.assert_allclose(
            covariance,
            ((2.0, 0.2, 0.0), (0.2, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        self.assertEqual(prepared.bags[0].gyro_measurements, ())
        self.assertIsNone(prepared.bags[0].covariances.gyro_observation)

    def test_command_history_is_required_instead_of_first_value_extension(self):
        late_history = _causal_command(
            self.flight.rotor_command.times,
            self.flight.rotor_command.values,
            (-0.01,),
            (self.flight.rotor_command.history_values[-1],),
            _ROTOR_FIELDS,
        )
        flight = replace(self.flight, rotor_command=late_history)
        with self.assertRaisesRegex(ValueError, "no causal event"):
            self._prepare(flight=flight)

    def test_adapter_raw_sha256_is_normalized_but_malformed_is_rejected(self):
        raw = self.digest.split(":", 1)[1]
        raw_flight = replace(
            self.flight,
            provenance=replace(self.flight.provenance, bag_sha256=raw),
        )
        self.assertEqual(self._prepare(flight=raw_flight).bags[0].bag_id, "flight-a")
        malformed = replace(
            self.flight,
            provenance=replace(
                self.flight.provenance,
                bag_sha256="sha256:" + raw.upper(),
            ),
        )
        with self.assertRaisesRegex(ValueError, "raw lowercase hex"):
            self._prepare(flight=malformed)

    def test_enabled_factor_field_frame_and_support_contracts_fail_closed(self):
        wrong_order = VectorSeries(
            times=self.flight.gyro.times,
            record_times=self.flight.gyro.record_times,
            values=self.flight.gyro.values,
            field_names=("z", "y", "x"),
            timestamp_source=TimestampSource.HEADER,
        )
        with self.assertRaisesRegex(ValueError, "fields must equal"):
            self._prepare(flight=replace(self.flight, gyro=wrong_order))

        wrong_frame = replace(
            self.flight.sensor_extrinsics, gyro_sensor_frame="imu"
        )
        with self.assertRaisesRegex(ValueError, "gyro frame"):
            self._prepare(
                flight=replace(
                    self.flight, sensor_extrinsics=wrong_frame
                )
            )

        late_velocity = _vector(
            self.flight.velocity.times[1:],
            self.flight.velocity.values[1:],
            ("x", "y", "z"),
        )
        with self.assertRaisesRegex(ValueError, "without extrapolation"):
            self._prepare(
                flight=replace(self.flight, velocity=late_velocity)
            )

    def test_calibrated_accelerometer_is_prepared_with_independent_bias(self):
        flight = _flight_data(
            self.bag,
            self.digest,
            "flight-a",
            include_accelerometer=True,
        )
        initialization = build_flight_initialization(
            flight, 0.1, pose_smoothing_window=1
        )
        payload = copy.deepcopy(self.payload)
        factor = payload["bags"][0]["observation_factors"][
            "accelerometer"
        ]
        factor.update(
            enabled=True,
            disabled_reason=None,
            covariances={
                name: _covariance(contract)
                for name, contract in (
                    OBSERVATION_COVARIANCE_BLOCKS["accelerometer"].items()
                )
            },
        )
        payload["bags"][0]["initial_state_prior_covariances"].update(
            {
                name: _covariance(contract)
                for name, contract in (
                    ACCELEROMETER_BIAS_PRIOR_COVARIANCE_BLOCKS.items()
                )
            }
        )
        prepared = self._prepare(
            payload=payload,
            flight=flight,
            initialization=initialization,
        )
        bag = prepared.bags[0]
        self.assertTrue(bag.accelerometer.enabled)
        self.assertEqual(len(bag.accelerometer_measurements), 7)
        self.assertIsNotNone(bag.covariances.accelerometer_observation)
        bias_keys = tuple(
            key
            for key in build_initial_batch_state(prepared).layout.variable_keys
            if key.kind is VariableKind.ACCELEROMETER_BIAS
        )
        self.assertEqual(len(bias_keys), 1)
        np.testing.assert_array_equal(
            build_initial_batch_state(prepared).value(bias_keys[0]),
            flight.imu_preflight.accelerometer_bias,
        )

    def test_disabled_calibrated_accelerometer_adds_no_graph_state(self):
        flight = _flight_data(
            self.bag,
            self.digest,
            "flight-a",
            include_accelerometer=True,
        )
        initialization = build_flight_initialization(
            flight, 0.1, pose_smoothing_window=1
        )
        prepared = self._prepare(
            flight=flight,
            initialization=initialization,
        )
        bag = prepared.bags[0]
        self.assertFalse(bag.accelerometer.enabled)
        self.assertEqual(bag.accelerometer_measurements, ())
        self.assertIsNone(bag.covariances.accelerometer_observation)
        self.assertFalse(
            any(
                key.kind is VariableKind.ACCELEROMETER_BIAS
                for key in build_initial_batch_state(prepared)
                .layout.variable_keys
            )
        )

    def test_mode_switch_marks_one_dynamics_interval_invalid_with_reason(self):
        flight = _flight_data(
            self.bag,
            self.digest,
            "flight-a",
            mode_states=(3, 6, 6),
        )
        initialization = build_flight_initialization(
            flight, 0.1, pose_smoothing_window=1
        )
        prepared = self._prepare(
            flight=flight, initialization=initialization
        )
        statuses = prepared.bags[0].dynamics_interval_statuses
        self.assertTrue(statuses[0].valid)
        self.assertFalse(statuses[1].valid)
        self.assertIn("switches", statuses[1].invalid_reason)
        self.assertFalse(statuses[2].valid)
        self.assertIn("not controller-active", statuses[2].invalid_reason)

    def test_multi_bag_preparation_keeps_local_graphs_separate(self):
        second_bag = self.root / "flight-b.bag"
        second_bag.write_bytes(b"synthetic flight b")
        second_digest = file_sha256(second_bag)
        second_flight = _flight_data(
            second_bag, second_digest, "flight-b"
        )
        second_initialization = build_flight_initialization(
            second_flight, 0.1, pose_smoothing_window=1
        )
        payload = _request_payload(
            self.root,
            (
                ("flight-a", self.bag, self.digest),
                ("flight-b", second_bag, second_digest),
            ),
        )
        request = validate_batch_estimation_request(payload)
        prepared = prepare_fixed_batch_graph_data(
            request,
            (self.flight, second_flight),
            (self.initialization, second_initialization),
            self.chart,
            self.geometry,
            self.actuators,
            StateScaling.unit(),
            PreparationSelection(
                "recorded-mode", 0.035, np.ones(6), np.zeros(18)
            ),
        )
        self.assertEqual(
            tuple(bag.bag_id for bag in prepared.bags),
            ("flight-a", "flight-b"),
        )
        problem = build_fixed_batch_problem(prepared)
        self.assertEqual(problem.layout.bag_ids, ("flight-a", "flight-b"))


if __name__ == "__main__":
    unittest.main()
