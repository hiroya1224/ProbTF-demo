"""Strict FlightData-to-fixed-batch-graph preparation.

This module is the sole boundary where an audited asynchronous ``FlightData``
object, its initialization, and one validated request become the ROS-free
contracts consumed by :mod:`grape_param_estim.batch.graph_builder`.  Sensor
observations retain their own timestamps.  Only controller/reference signals
and issued commands use causal record-time zero-order hold.

The preparation step never finite-differences a likelihood observation,
resamples a sensor stream onto the knot grid, invents a covariance, or extends
the first command backwards in time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.batch.factors.dynamics_factor import (
    BODY_WRENCH_QUANTITY,
    SPECIFIC_ACCELERATION_QUANTITY,
)
from grape_param_estim.batch.graph_builder import (
    AccelerometerFactorContract,
    GaussianCovariance,
    MeasurementBracket,
    OrientationGaussianPrior,
    PreparedActuatorInterval,
    PreparedBagGraphData,
    PreparedBagPriors,
    PreparedBatchGraphData,
    PreparedCommandSegment,
    PreparedControllerIntegralMeasurement,
    PreparedControllerInterval,
    PreparedDynamicsConfiguration,
    PreparedDynamicsIntervalStatus,
    PreparedFactorCovariances,
    PreparedGimbalMeasurement,
    PreparedGyroMeasurement,
    PreparedKnotPrior,
    PreparedKnotState,
    PreparedPoseMeasurement,
    PreparedSensorExtrinsics,
    PreparedVelocityMeasurement,
    VectorGaussianPrior,
)
from grape_param_estim.batch.laplace_em import (
    DiagonalQDefinition,
    QIntervalModel,
)
from grape_param_estim.batch.state import StateScaling
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.batch_request import BatchEstimationRequest
from grape_param_estim.controller import ControllerConfig, GrapeController
from grape_param_estim.geometry import quaternion_to_matrix
from grape_param_estim.initialization import FlightInitialization
from grape_param_estim.parameterization import (
    PARAMETER_DIMENSION,
    VehicleParameterChart,
)
from grape_param_estim.real_rosbag import CONTROL_ACTIVE_FLIGHT_STATES
from grape_param_estim.sensor_models import (
    CausalVectorSeries,
    FlightData,
    PidDebugSeries,
    ReferenceSeries,
    TimestampSource,
    VectorSeries,
)
from grape_param_estim.system import (
    ActuatorCommand,
    ActuatorParameters,
    GrapeGeometry,
    ReferenceState,
)


_TIME_TOLERANCE = 2.0e-9
_PID_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
_VELOCITY_FIELDS = ("x", "y", "z")
_GYRO_FIELDS = ("x", "y", "z")
_ROTOR_FIELDS = ("rotor_1", "rotor_2", "rotor_3", "rotor_4")
_GIMBAL_FIELDS = ("gimbal1", "gimbal2", "gimbal3", "gimbal4")
_KNOT_KINDS = (
    VariableKind.POSITION,
    VariableKind.ORIENTATION_TANGENT,
    VariableKind.LINEAR_VELOCITY,
    VariableKind.ANGULAR_VELOCITY,
    VariableKind.CONTROLLER_INTEGRAL,
    VariableKind.ACTUATOR_THRUST,
    VariableKind.GIMBAL_ANGLE,
)


@dataclass(frozen=True)
class PreparationSelection:
    """Explicit outer-loop values selecting one fixed graph."""

    mode_id: str
    fixed_delay_seconds: float
    q_diagonal: np.ndarray
    initial_parameter_coordinates: np.ndarray

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mode_id, str)
            or not self.mode_id
            or self.mode_id.strip() != self.mode_id
        ):
            raise ValueError("mode_id must be a canonical non-empty string")
        delay = float(self.fixed_delay_seconds)
        if not np.isfinite(delay) or delay < 0.0:
            raise ValueError(
                "fixed_delay_seconds must be finite and non-negative"
            )
        object.__setattr__(self, "fixed_delay_seconds", delay)
        q = np.asarray(self.q_diagonal, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)) or np.any(q <= 0.0):
            raise ValueError("q_diagonal must contain six positive values")
        q = q.copy()
        q.setflags(write=False)
        coordinates = np.asarray(
            self.initial_parameter_coordinates, dtype=float
        )
        if coordinates.shape != (PARAMETER_DIMENSION,) or not np.all(
            np.isfinite(coordinates)
        ):
            raise ValueError(
                "initial_parameter_coordinates must contain 18 finite values"
            )
        coordinates = coordinates.copy()
        coordinates.setflags(write=False)
        object.__setattr__(self, "q_diagonal", q)
        object.__setattr__(
            self, "initial_parameter_coordinates", coordinates
        )


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("{} must be a mapping".format(name))
    return value


def _request_bags(
    request: BatchEstimationRequest,
) -> Mapping[str, Mapping[str, object]]:
    bags = request.payload["bags"]
    result = {}
    for value in bags:
        bag = _as_mapping(value, "request bag")
        result[str(bag["bag_id"])] = bag
    if set(result) != set(request.bag_ids):
        raise ValueError("validated request bag identity is inconsistent")
    return result


def _mode_schedules(
    request: BatchEstimationRequest,
    mode_id: str,
) -> Mapping[str, Mapping[str, object]]:
    matches = tuple(
        value
        for value in request.payload["mode_hypotheses"]
        if value["mode_id"] == mode_id
    )
    if len(matches) != 1:
        raise ValueError(
            "mode_id {!r} does not select exactly one request hypothesis"
            .format(mode_id)
        )
    schedules = _as_mapping(matches[0]["bag_schedules"], "bag_schedules")
    if set(schedules) != set(request.bag_ids):
        raise ValueError("selected mode schedules do not match request bags")
    return schedules


def _covariance(value: object, name: str) -> GaussianCovariance:
    contract = _as_mapping(value, name)
    representation = contract["representation"]
    raw = np.asarray(contract["values"], dtype=float)
    if representation == "diagonal":
        if raw.ndim != 1:
            raise ValueError("{} diagonal values must be a vector".format(name))
        matrix = np.diag(raw)
    elif representation == "full":
        matrix = raw
    else:
        raise ValueError("{} has an unknown representation".format(name))
    return GaussianCovariance(matrix)


def _factor_contract(
    bag_request: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    factors = _as_mapping(
        bag_request["observation_factors"], "observation_factors"
    )
    return _as_mapping(factors[name], "factor {}".format(name))


def _factor_enabled(
    bag_request: Mapping[str, object],
    name: str,
) -> bool:
    return bool(_factor_contract(bag_request, name)["enabled"])


def _observation_covariance(
    bag_request: Mapping[str, object],
    factor_name: str,
    block_name: str,
) -> Optional[GaussianCovariance]:
    factor = _factor_contract(bag_request, factor_name)
    if not bool(factor["enabled"]):
        return None
    blocks = _as_mapping(factor["covariances"], "factor covariances")
    return _covariance(blocks[block_name], block_name)


def _covariance_block(
    parent: object,
    block_name: str,
    parent_name: str,
) -> GaussianCovariance:
    blocks = _as_mapping(parent, parent_name)
    return _covariance(blocks[block_name], block_name)


def _check_series_contract(
    series: VectorSeries,
    *,
    name: str,
    fields: Tuple[str, ...],
    timestamp_source: TimestampSource,
) -> None:
    if not isinstance(series, VectorSeries):
        raise TypeError("{} must be a VectorSeries".format(name))
    if series.field_names != fields:
        raise ValueError(
            "{} fields must equal {} in estimator order".format(
                name, fields
            )
        )
    if series.timestamp_source is not timestamp_source:
        raise ValueError(
            "{} must use {} timestamps".format(
                name, timestamp_source.value
            )
        )


def _check_measurement_support(
    times: np.ndarray,
    knot_times: np.ndarray,
    maximum_gap: float,
    name: str,
) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("{} requires at least two timestamps".format(name))
    if (
        values[0] > knot_times[0] + _TIME_TOLERANCE
        or values[-1] < knot_times[-1] - _TIME_TOLERANCE
    ):
        raise ValueError(
            "{} does not cover the latent knot interval without extrapolation"
            .format(name)
        )
    left = max(
        0,
        int(np.searchsorted(values, knot_times[0], side="right") - 1),
    )
    right = min(
        values.size,
        int(np.searchsorted(values, knot_times[-1], side="left") + 1),
    )
    support = values[left:right]
    if support.size < 2 or np.max(np.diff(support)) > maximum_gap:
        raise ValueError(
            "{} exceeds maximum_measurement_gap_seconds inside knot support"
            .format(name)
        )
    indices = np.flatnonzero(
        (values >= knot_times[0] - _TIME_TOLERANCE)
        & (values <= knot_times[-1] + _TIME_TOLERANCE)
    )
    if indices.size == 0:
        raise ValueError("{} has no samples inside knot support".format(name))
    return indices


def _measurement_bracket(
    knot_times: np.ndarray,
    measurement_time: float,
    maximum_gap: float,
    name: str,
) -> MeasurementBracket:
    timestamp = float(measurement_time)
    if (
        timestamp < knot_times[0] - _TIME_TOLERANCE
        or timestamp > knot_times[-1] + _TIME_TOLERANCE
    ):
        raise ValueError(
            "{} timestamp lies outside latent knot support".format(name)
        )
    if timestamp <= knot_times[0] + _TIME_TOLERANCE:
        left = 0
        fraction = 0.0
    elif timestamp >= knot_times[-1] - _TIME_TOLERANCE:
        left = knot_times.size - 2
        fraction = 1.0
    else:
        left = int(np.searchsorted(knot_times, timestamp, side="right") - 1)
        dt = float(knot_times[left + 1] - knot_times[left])
        fraction = (timestamp - float(knot_times[left])) / dt
    bracket_width = float(knot_times[left + 1] - knot_times[left])
    if bracket_width > maximum_gap + _TIME_TOLERANCE:
        raise ValueError(
            "{} adjacent-knot bracket exceeds maximum_measurement_gap_seconds"
            .format(name)
        )
    fraction = min(1.0, max(0.0, float(fraction)))
    return MeasurementBracket(left, fraction)


def _causal_index(
    times: np.ndarray,
    query_time: float,
    maximum_age: Optional[float],
    name: str,
) -> int:
    values = np.asarray(times, dtype=float)
    index = int(np.searchsorted(values, query_time, side="right") - 1)
    if index < 0:
        raise ValueError(
            "{} has no causal event at or before {:.9f}".format(
                name, query_time
            )
        )
    age = float(query_time - values[index])
    if age < -_TIME_TOLERANCE:
        raise ValueError("{} causal lookup selected a future event".format(name))
    if maximum_age is not None and age > maximum_age + _TIME_TOLERANCE:
        raise ValueError(
            "{} causal event age exceeds maximum_measurement_gap_seconds"
            .format(name)
        )
    return index


def _reference_at(
    reference: ReferenceSeries,
    query_time: float,
    maximum_age: float,
) -> ReferenceState:
    if not isinstance(reference, ReferenceSeries):
        raise TypeError("reference must be ReferenceSeries")
    if reference.timestamp_source is not TimestampSource.RECORD:
        raise ValueError("controller reference must use rosbag record time")
    index = _causal_index(
        reference.times,
        query_time,
        maximum_age,
        "controller reference",
    )
    return ReferenceState(
        position=reference.position[index],
        linear_velocity=reference.linear_velocity[index],
        linear_acceleration=reference.linear_acceleration[index],
        rpy=reference.rpy[index],
        angular_velocity=reference.angular_velocity[index],
        angular_acceleration=reference.angular_acceleration[index],
    )


def _command_events(
    series: Optional[VectorSeries],
    *,
    name: str,
    fields: Tuple[str, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(series, CausalVectorSeries):
        raise ValueError(
            "{} requires explicit pre-window CausalVectorSeries history"
            .format(name)
        )
    _check_series_contract(
        series,
        name=name,
        fields=fields,
        timestamp_source=TimestampSource.RECORD,
    )
    return series.all_times, series.all_values


def _command_at(
    times: np.ndarray,
    values: np.ndarray,
    query_time: float,
    maximum_age: float,
    name: str,
) -> np.ndarray:
    index = _causal_index(times, query_time, maximum_age, name)
    return values[index]


def _actuator_command(
    thrust: np.ndarray,
    gimbal: np.ndarray,
) -> ActuatorCommand:
    return ActuatorCommand(
        thrust=np.asarray(thrust, dtype=float),
        gimbal_angle=np.asarray(gimbal, dtype=float),
        virtual_force=np.zeros(8, dtype=float),
        desired_acceleration=np.zeros(6, dtype=float),
    )


def _delayed_command_segments(
    start: float,
    end: float,
    delay: float,
    rotor_times: np.ndarray,
    rotor_values: np.ndarray,
    gimbal_times: np.ndarray,
    gimbal_values: np.ndarray,
    maximum_age: float,
) -> Tuple[PreparedCommandSegment, ...]:
    query_start = start - delay
    query_end = end - delay
    _causal_index(
        rotor_times, query_start, maximum_age, "delayed rotor command"
    )
    _causal_index(
        gimbal_times, query_start, maximum_age, "delayed gimbal command"
    )
    open_query_end = float(np.nextafter(query_end, -np.inf))
    _causal_index(
        rotor_times,
        open_query_end,
        maximum_age,
        "delayed rotor command",
    )
    _causal_index(
        gimbal_times,
        open_query_end,
        maximum_age,
        "delayed gimbal command",
    )
    switch_times = [float(start), float(end)]
    for event_times in (rotor_times, gimbal_times):
        left = int(np.searchsorted(event_times, query_start, side="right"))
        right = int(np.searchsorted(event_times, query_end, side="left"))
        for issue_time in event_times[left:right]:
            plant_switch = float(issue_time + delay)
            if (
                plant_switch > start + _TIME_TOLERANCE
                and plant_switch < end - _TIME_TOLERANCE
            ):
                switch_times.append(plant_switch)
    boundaries = np.unique(np.asarray(switch_times, dtype=float))
    boundaries.sort()
    segments = []
    for index in range(boundaries.size - 1):
        left = float(boundaries[index])
        right = float(boundaries[index + 1])
        duration = right - left
        if duration <= 0.0:
            continue
        # Sampling strictly inside the segment avoids floating-point
        # ambiguity at the ZOH switch while preserving the constant value.
        issued_time = 0.5 * (left + right) - delay
        thrust = _command_at(
            rotor_times,
            rotor_values,
            issued_time,
            maximum_age,
            "delayed rotor command",
        )
        gimbal = _command_at(
            gimbal_times,
            gimbal_values,
            issued_time,
            maximum_age,
            "delayed gimbal command",
        )
        segments.append(
            PreparedCommandSegment(
                command=_actuator_command(thrust, gimbal),
                duration=duration,
            )
        )
    if not segments:
        raise ValueError("delayed command interval produced no segments")
    return tuple(segments)


def _flight_mode_at(flight_data: FlightData, query_time: float) -> int:
    schedule = flight_data.flight_mode
    if schedule.timestamp_source is not TimestampSource.RECORD:
        raise ValueError("flight mode schedule must use rosbag record time")
    times = np.concatenate(
        (np.asarray((schedule.initial_time,)), schedule.times)
    )
    states = np.concatenate(
        (np.asarray((schedule.initial_state,), dtype=np.int64), schedule.states)
    )
    index = _causal_index(times, query_time, None, "flight mode schedule")
    return int(states[index])


def _mode_switch_inside(
    flight_data: FlightData,
    start: float,
    end: float,
) -> bool:
    times = flight_data.flight_mode.times
    states = flight_data.flight_mode.states
    left_state = _flight_mode_at(flight_data, start)
    indices = np.flatnonzero(
        (times > start + _TIME_TOLERANCE)
        & (times < end - _TIME_TOLERANCE)
    )
    previous = left_state
    for index in indices:
        current = int(states[index])
        if current != previous:
            return True
        previous = current
    return False


def _integral_proxy(pid: PidDebugSeries, configuration: ControllerConfig):
    if not isinstance(pid, PidDebugSeries):
        raise TypeError("PID debug must be PidDebugSeries")
    if pid.axis_names != _PID_AXES:
        raise ValueError("PID debug axes are not canonical")
    if pid.timestamp_source is not TimestampSource.RECORD:
        raise ValueError("PID debug must use rosbag record time")
    gains = np.asarray(
        tuple(value.i_gain for value in configuration.pid), dtype=float
    )
    ambiguous = np.abs(gains) <= 1.0e-12
    if np.any(ambiguous):
        names = ", ".join(
            _PID_AXES[index] for index in np.flatnonzero(ambiguous)
        )
        raise ValueError(
            "controller integral proxy is not uniquely recoverable for "
            "zero-I-gain axes: {}".format(names)
        )
    values = pid.i_term / gains[None, :]
    if np.any(~np.isfinite(values)):
        raise ValueError("decoded controller integral proxy is non-finite")
    return values


def _integration_gate_schedule(
    flight_data: FlightData,
    initialization: FlightInitialization,
    source: str,
) -> Tuple[bool, ...]:
    configuration = flight_data.controller_configuration
    positions = tuple(
        initialization.state.knot_value(
            flight_data.bag_id, index, VariableKind.POSITION
        )
        for index in range(initialization.grid.count)
    )
    threshold = (
        configuration.initial_height
        + configuration.start_roll_pitch_integration_height
    )
    if source == "deterministic_replay":
        active = bool(positions[0][2] > threshold)
        result = []
        for position in positions[:-1]:
            result.append(active)
            if not active and position[2] > threshold:
                active = True
        return tuple(result)
    if source != "recorded_pid_debug":
        raise ValueError("unknown integration gate source")
    if flight_data.pid_debug is None:
        raise ValueError(
            "recorded_pid_debug gate source requires PID debug data"
        )
    proxy = _integral_proxy(
        flight_data.pid_debug, flight_data.controller_configuration
    )
    evidence = np.linalg.norm(proxy[:, 3:5], axis=1) > 1.0e-12
    if not np.any(evidence):
        raise ValueError(
            "recorded PID debug contains no unique roll/pitch integration "
            "activation evidence"
        )
    first_active = float(
        flight_data.pid_debug.times[int(np.flatnonzero(evidence)[0])]
    )
    return tuple(
        bool(time >= first_active - _TIME_TOLERANCE)
        for time in initialization.grid.times[:-1]
    )


def _prepared_knots(
    initialization: FlightInitialization,
) -> Tuple[PreparedKnotState, ...]:
    state = initialization.state
    result = []
    for index, time in enumerate(initialization.grid.times):
        result.append(
            PreparedKnotState(
                time=float(time),
                position=state.knot_value(
                    initialization.bag_id, index, VariableKind.POSITION
                ),
                rotation=state.knot_value(
                    initialization.bag_id,
                    index,
                    VariableKind.ORIENTATION_TANGENT,
                ),
                linear_velocity=state.knot_value(
                    initialization.bag_id,
                    index,
                    VariableKind.LINEAR_VELOCITY,
                ),
                angular_velocity=state.knot_value(
                    initialization.bag_id,
                    index,
                    VariableKind.ANGULAR_VELOCITY,
                ),
                controller_integral=state.knot_value(
                    initialization.bag_id,
                    index,
                    VariableKind.CONTROLLER_INTEGRAL,
                ),
                actuator_thrust=state.knot_value(
                    initialization.bag_id,
                    index,
                    VariableKind.ACTUATOR_THRUST,
                ),
                gimbal_angle=state.knot_value(
                    initialization.bag_id,
                    index,
                    VariableKind.GIMBAL_ANGLE,
                ),
            )
        )
    return tuple(result)


def _bag_priors(
    bag_request: Mapping[str, object],
    flight_data: FlightData,
    first_knot: PreparedKnotState,
) -> PreparedBagPriors:
    blocks = bag_request["initial_state_prior_covariances"]
    return PreparedBagPriors(
        gyro_bias=VectorGaussianPrior(
            mean=flight_data.imu_preflight.gyro_bias,
            covariance=_covariance_block(
                blocks, "gyro_bias", "initial state priors"
            ),
        ),
        initial_knot=PreparedKnotPrior(
            position=VectorGaussianPrior(
                first_knot.position,
                _covariance_block(
                    blocks, "position", "initial state priors"
                ),
            ),
            rotation=OrientationGaussianPrior(
                first_knot.rotation,
                _covariance_block(
                    blocks, "orientation", "initial state priors"
                ),
            ),
            linear_velocity=VectorGaussianPrior(
                first_knot.linear_velocity,
                _covariance_block(
                    blocks, "linear_velocity", "initial state priors"
                ),
            ),
            angular_velocity=VectorGaussianPrior(
                first_knot.angular_velocity,
                _covariance_block(
                    blocks, "angular_velocity", "initial state priors"
                ),
            ),
            controller_integral=VectorGaussianPrior(
                first_knot.controller_integral,
                _covariance_block(
                    blocks, "controller_integral", "initial state priors"
                ),
            ),
            actuator_thrust=VectorGaussianPrior(
                first_knot.actuator_thrust,
                _covariance_block(
                    blocks, "actuator_thrust", "initial state priors"
                ),
            ),
            gimbal_angle=VectorGaussianPrior(
                first_knot.gimbal_angle,
                _covariance_block(
                    blocks, "gimbal_angle", "initial state priors"
                ),
            ),
        ),
    )


def _prepared_covariances(
    bag_request: Mapping[str, object],
) -> PreparedFactorCovariances:
    fixed = bag_request["fixed_factor_covariances"]
    return PreparedFactorCovariances(
        position_observation=_observation_covariance(
            bag_request, "pose", "position_observation"
        ),
        orientation_observation=_observation_covariance(
            bag_request, "pose", "orientation_observation"
        ),
        velocity_observation=_observation_covariance(
            bag_request, "velocity", "velocity_observation"
        ),
        gyro_observation=_observation_covariance(
            bag_request, "gyro", "gyro_observation"
        ),
        issued_thrust_observation=_observation_covariance(
            bag_request,
            "issued_rotor_command",
            "issued_thrust_observation",
        ),
        issued_gimbal_observation=_observation_covariance(
            bag_request,
            "issued_gimbal_command",
            "issued_gimbal_observation",
        ),
        actual_gimbal_observation=_observation_covariance(
            bag_request,
            "actual_gimbal_position",
            "actual_gimbal_observation",
        ),
        controller_integral_observation=_observation_covariance(
            bag_request,
            "controller_integral",
            "controller_integral_observation",
        ),
        controller_integral_transition=_covariance_block(
            fixed, "controller_integral_transition", "fixed covariances"
        ),
        actuator_thrust_transition=_covariance_block(
            fixed, "actuator_thrust_transition", "fixed covariances"
        ),
        actuator_gimbal_transition=_covariance_block(
            fixed, "actuator_gimbal_transition", "fixed covariances"
        ),
        position_kinematic=_covariance_block(
            fixed, "position_kinematic", "fixed covariances"
        ),
        orientation_kinematic=_covariance_block(
            fixed, "orientation_kinematic", "fixed covariances"
        ),
    )


def _prepare_bag(
    bag_request: Mapping[str, object],
    flight_data: FlightData,
    initialization: FlightInitialization,
    mode_schedule: Mapping[str, object],
    selection: PreparationSelection,
    parameter_chart: VehicleParameterChart,
    geometry: GrapeGeometry,
    actuator_parameters: ActuatorParameters,
    maximum_gap: float,
) -> PreparedBagGraphData:
    bag_id = str(bag_request["bag_id"])
    if flight_data.bag_id != bag_id or initialization.bag_id != bag_id:
        raise ValueError("FlightData/initialization/request bag IDs disagree")
    if mode_schedule["flight_state_source"] != "recorded_causal_schedule":
        raise ValueError("flight state source must be recorded causal schedule")
    requested_interval = np.asarray(
        bag_request["interval_seconds"], dtype=float
    )
    actual_interval = np.asarray(
        (flight_data.interval.start, flight_data.interval.end), dtype=float
    )
    if not np.allclose(
        requested_interval, actual_interval, rtol=0.0, atol=_TIME_TOLERANCE
    ):
        raise ValueError("FlightData interval does not equal request interval")
    if flight_data.provenance.bag_sha256 != bag_request["sha256"]:
        raise ValueError("FlightData bag SHA-256 does not equal request")
    if Path(flight_data.provenance.bag_path).resolve() != Path(
        str(bag_request["path"])
    ).resolve():
        raise ValueError("FlightData bag path does not equal request")
    knot_times = initialization.grid.times
    if initialization.grid.count < 2:
        raise ValueError("batch graph requires at least two knots")
    initialized_interval = np.asarray(
        (
            initialization.grid.flight_interval.start,
            initialization.grid.flight_interval.end,
        ),
        dtype=float,
    )
    if not np.allclose(
        initialized_interval,
        actual_interval,
        rtol=0.0,
        atol=_TIME_TOLERANCE,
    ):
        raise ValueError(
            "initialization flight interval does not equal FlightData"
        )

    configuration = flight_data.controller_configuration
    if not isinstance(configuration, ControllerConfig):
        raise TypeError(
            "FlightData.controller_configuration must be ControllerConfig"
        )
    controller = GrapeController(
        configuration,
        parameter_chart.decode(np.zeros(PARAMETER_DIMENSION, dtype=float)),
        geometry,
    )

    extrinsics = flight_data.sensor_extrinsics
    if flight_data.imu_preflight.frame_id != extrinsics.gyro_sensor_frame:
        raise ValueError(
            "preflight gyro frame and numeric gyro extrinsics disagree"
        )
    prepared_extrinsics = PreparedSensorExtrinsics(
        pose_sensor_position_in_body=(
            extrinsics.pose_sensor_position_in_body
        ),
        pose_sensor_to_body_rotation=(
            extrinsics.pose_sensor_to_body_rotation
        ),
        velocity_sensor_position_in_body=(
            extrinsics.velocity_sensor_position_in_body
        ),
        body_to_gyro_sensor_rotation=(
            extrinsics.body_to_gyro_sensor_rotation
        ),
    )

    pose_measurements = ()
    if _factor_enabled(bag_request, "pose"):
        if flight_data.pose.timestamp_source is not TimestampSource.HEADER:
            raise ValueError("pose observation must use message header time")
        indices = _check_measurement_support(
            flight_data.pose.times, knot_times, maximum_gap, "pose"
        )
        pose_measurements = tuple(
            PreparedPoseMeasurement(
                bracket=_measurement_bracket(
                    knot_times,
                    flight_data.pose.times[index],
                    maximum_gap,
                    "pose",
                ),
                position=flight_data.pose.positions[index],
                rotation=quaternion_to_matrix(
                    flight_data.pose.orientations_xyzw[index]
                ),
            )
            for index in indices
        )

    velocity_measurements = ()
    if _factor_enabled(bag_request, "velocity"):
        if flight_data.velocity is None:
            raise ValueError("enabled velocity factor has no velocity stream")
        _check_series_contract(
            flight_data.velocity,
            name="velocity",
            fields=_VELOCITY_FIELDS,
            timestamp_source=TimestampSource.HEADER,
        )
        indices = _check_measurement_support(
            flight_data.velocity.times,
            knot_times,
            maximum_gap,
            "velocity",
        )
        velocity_measurements = tuple(
            PreparedVelocityMeasurement(
                _measurement_bracket(
                    knot_times,
                    flight_data.velocity.times[index],
                    maximum_gap,
                    "velocity",
                ),
                flight_data.velocity.values[index],
            )
            for index in indices
        )

    gyro_measurements = ()
    if _factor_enabled(bag_request, "gyro"):
        if flight_data.gyro is None:
            raise ValueError("enabled gyro factor has no gyro stream")
        _check_series_contract(
            flight_data.gyro,
            name="gyro",
            fields=_GYRO_FIELDS,
            timestamp_source=TimestampSource.HEADER,
        )
        indices = _check_measurement_support(
            flight_data.gyro.times, knot_times, maximum_gap, "gyro"
        )
        gyro_measurements = tuple(
            PreparedGyroMeasurement(
                _measurement_bracket(
                    knot_times,
                    flight_data.gyro.times[index],
                    maximum_gap,
                    "gyro",
                ),
                flight_data.gyro.values[index],
            )
            for index in indices
        )

    if _factor_enabled(bag_request, "accelerometer"):
        raise ValueError(
            "accelerometer factor is not implemented in batch graph v1"
        )
    accelerometer_reason = str(
        _factor_contract(bag_request, "accelerometer")["disabled_reason"]
    )

    actual_gimbal_measurements = ()
    if _factor_enabled(bag_request, "actual_gimbal_position"):
        if flight_data.gimbal_position is None:
            raise ValueError(
                "enabled actual gimbal factor has no joint-position stream"
            )
        _check_series_contract(
            flight_data.gimbal_position,
            name="actual gimbal position",
            fields=_GIMBAL_FIELDS,
            timestamp_source=TimestampSource.HEADER,
        )
        indices = _check_measurement_support(
            flight_data.gimbal_position.times,
            knot_times,
            maximum_gap,
            "actual gimbal position",
        )
        actual_gimbal_measurements = tuple(
            PreparedGimbalMeasurement(
                _measurement_bracket(
                    knot_times,
                    flight_data.gimbal_position.times[index],
                    maximum_gap,
                    "actual gimbal position",
                ),
                flight_data.gimbal_position.values[index],
            )
            for index in indices
        )

    controller_integral_measurements = ()
    if _factor_enabled(bag_request, "controller_integral"):
        if flight_data.pid_debug is None:
            raise ValueError(
                "enabled controller integral factor has no PID debug stream"
            )
        values = _integral_proxy(flight_data.pid_debug, configuration)
        indices = _check_measurement_support(
            flight_data.pid_debug.times,
            knot_times,
            maximum_gap,
            "controller integral proxy",
        )
        controller_integral_measurements = tuple(
            PreparedControllerIntegralMeasurement(
                _measurement_bracket(
                    knot_times,
                    flight_data.pid_debug.times[index],
                    maximum_gap,
                    "controller integral proxy",
                ),
                values[index],
            )
            for index in indices
        )

    rotor_times, rotor_values = _command_events(
        flight_data.rotor_command,
        name="rotor command",
        fields=_ROTOR_FIELDS,
    )
    gimbal_times, gimbal_values = _command_events(
        flight_data.gimbal_command,
        name="gimbal command",
        fields=_GIMBAL_FIELDS,
    )

    integration_gates = _integration_gate_schedule(
        flight_data,
        initialization,
        str(mode_schedule["integration_gate_source"]),
    )
    controller_intervals = []
    actuator_intervals = []
    dynamics_statuses = []
    for index in range(knot_times.size - 1):
        start = float(knot_times[index])
        end = float(knot_times[index + 1])
        issued_thrust = _command_at(
            rotor_times,
            rotor_values,
            start,
            maximum_gap,
            "issued rotor command",
        )
        issued_gimbal = _command_at(
            gimbal_times,
            gimbal_values,
            start,
            maximum_gap,
            "issued gimbal command",
        )
        controller_intervals.append(
            PreparedControllerInterval(
                left_knot_index=index,
                reference=_reference_at(
                    flight_data.reference, start, maximum_gap
                ),
                roll_pitch_integration_active=integration_gates[index],
                issued_thrust_observation=(
                    issued_thrust
                    if _factor_enabled(
                        bag_request, "issued_rotor_command"
                    )
                    else None
                ),
                issued_gimbal_observation=(
                    issued_gimbal
                    if _factor_enabled(
                        bag_request, "issued_gimbal_command"
                    )
                    else None
                ),
            )
        )
        actuator_intervals.append(
            PreparedActuatorInterval(
                left_knot_index=index,
                delayed_command_segments=_delayed_command_segments(
                    start,
                    end,
                    selection.fixed_delay_seconds,
                    rotor_times,
                    rotor_values,
                    gimbal_times,
                    gimbal_values,
                    maximum_gap,
                ),
            )
        )
        state = _flight_mode_at(flight_data, start)
        if _mode_switch_inside(flight_data, start, end):
            dynamics_statuses.append(
                PreparedDynamicsIntervalStatus(
                    index,
                    False,
                    "recorded flight mode switches inside knot interval",
                )
            )
        elif state not in CONTROL_ACTIVE_FLIGHT_STATES:
            dynamics_statuses.append(
                PreparedDynamicsIntervalStatus(
                    index,
                    False,
                    "recorded flight state {} is not controller-active"
                    .format(state),
                )
            )
        else:
            dynamics_statuses.append(
                PreparedDynamicsIntervalStatus(index, True, "")
            )

    knots = _prepared_knots(initialization)
    return PreparedBagGraphData(
        bag_id=bag_id,
        knots=knots,
        initial_gyro_bias=flight_data.imu_preflight.gyro_bias,
        priors=_bag_priors(bag_request, flight_data, knots[0]),
        controller=controller,
        controller_intervals=tuple(controller_intervals),
        actuator_parameters=actuator_parameters,
        actuator_intervals=tuple(actuator_intervals),
        dynamics_interval_statuses=tuple(dynamics_statuses),
        pose_measurements=pose_measurements,
        velocity_measurements=velocity_measurements,
        gyro_measurements=gyro_measurements,
        controller_integral_measurements=(
            controller_integral_measurements
        ),
        actual_gimbal_measurements=actual_gimbal_measurements,
        sensor_extrinsics=prepared_extrinsics,
        covariances=_prepared_covariances(bag_request),
        accelerometer=AccelerometerFactorContract(
            enabled=False,
            disabled_reason=accelerometer_reason,
        ),
    )


def prepare_fixed_batch_graph_data(
    request: BatchEstimationRequest,
    flight_data: Sequence[FlightData],
    initializations: Sequence[FlightInitialization],
    parameter_chart: VehicleParameterChart,
    geometry: GrapeGeometry,
    actuator_parameters: ActuatorParameters,
    scaling: StateScaling,
    selection: PreparationSelection,
) -> PreparedBatchGraphData:
    """Prepare one fixed-delay, fixed-Q graph without scientific defaults.

    ``selection`` is explicit because both mode hypothesis and continuous
    delay belong to outer loops.  Calling this function again at another
    delay rebuilds every delayed ZOH switch exactly; it never mutates or
    patches a previously prepared graph.
    """

    if not isinstance(request, BatchEstimationRequest):
        raise TypeError("request must be a validated BatchEstimationRequest")
    if not isinstance(parameter_chart, VehicleParameterChart):
        raise TypeError("parameter_chart must be VehicleParameterChart")
    if not isinstance(geometry, GrapeGeometry):
        raise TypeError("geometry must be GrapeGeometry")
    if not isinstance(actuator_parameters, ActuatorParameters):
        raise TypeError("actuator_parameters must be ActuatorParameters")
    if not isinstance(scaling, StateScaling):
        raise TypeError("scaling must be StateScaling")
    if not isinstance(selection, PreparationSelection):
        raise TypeError("selection must be PreparationSelection")
    flights_sequence = tuple(flight_data)
    initializations_sequence = tuple(initializations)
    if any(not isinstance(value, FlightData) for value in flights_sequence):
        raise TypeError("flight_data must contain only FlightData values")
    if any(
        not isinstance(value, FlightInitialization)
        for value in initializations_sequence
    ):
        raise TypeError(
            "initializations must contain only FlightInitialization values"
        )
    flights = {value.bag_id: value for value in flights_sequence}
    initials = {
        value.bag_id: value for value in initializations_sequence
    }
    if len(flights) != len(flights_sequence):
        raise ValueError("flight_data contains duplicate bag IDs")
    if len(initials) != len(initializations_sequence):
        raise ValueError("initializations contains duplicate bag IDs")
    expected = set(request.bag_ids)
    if set(flights) != expected or set(initials) != expected:
        raise ValueError(
            "flight_data and initializations must exactly match request bags"
        )
    body_frames = {
        value.sensor_extrinsics.body_frame for value in flights.values()
    }
    if len(body_frames) != 1:
        raise ValueError(
            "all bags must use one estimator body-frame convention"
        )

    delay_bounds = np.asarray(
        request.payload["delay"]["bounds_seconds"], dtype=float
    )
    if (
        selection.fixed_delay_seconds < delay_bounds[0]
        or selection.fixed_delay_seconds > delay_bounds[1]
    ):
        raise ValueError("fixed delay lies outside request bounds")
    maximum_gap = float(
        request.payload["knot_policy"]["maximum_measurement_gap_seconds"]
    )
    period = float(request.payload["knot_policy"]["period_seconds"])
    for initialization in initials.values():
        if not np.isclose(
            initialization.grid.period_seconds,
            period,
            rtol=0.0,
            atol=2.0e-12,
        ):
            raise ValueError(
                "initialization knot period does not equal request policy"
            )

    q = _as_mapping(request.payload["q"], "request q")
    q_floor = np.asarray(q["floor_diagonal"], dtype=float)
    if np.any(selection.q_diagonal < q_floor):
        raise ValueError("selected Q diagonal lies below the request floor")
    quantity = str(q["residual_quantity"])
    if quantity not in (
        BODY_WRENCH_QUANTITY,
        SPECIFIC_ACCELERATION_QUANTITY,
    ):
        raise ValueError("request Q residual quantity is unsupported")
    q_definition = DiagonalQDefinition(
        residual_quantity=quantity,
        component_names=tuple(q["component_names"]),
        component_units=tuple(q["component_units"]),
        interval_model=QIntervalModel(str(q["interval_model"])),
    )
    parameter_chart.decode(selection.initial_parameter_coordinates)

    bags_by_id = _request_bags(request)
    schedules = _mode_schedules(request, selection.mode_id)
    prepared_bags = tuple(
        _prepare_bag(
            bags_by_id[bag_id],
            flights[bag_id],
            initials[bag_id],
            _as_mapping(schedules[bag_id], "mode schedule"),
            selection,
            parameter_chart,
            geometry,
            actuator_parameters,
            maximum_gap,
        )
        for bag_id in request.bag_ids
    )

    gravity_values = np.asarray(
        tuple(
            flights[bag_id].imu_preflight.gravity_magnitude
            for bag_id in request.bag_ids
        ),
        dtype=float,
    )
    if not np.allclose(
        gravity_values,
        gravity_values[0],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("all bags must use one explicit gravity magnitude")
    initialization_static = initials[request.bag_ids[0]].state.value(
        VariableKey(VariableKind.STATIC_PARAMETERS)
    )
    for bag_id in request.bag_ids[1:]:
        if not np.array_equal(
            initials[bag_id].state.value(
                VariableKey(VariableKind.STATIC_PARAMETERS)
            ),
            initialization_static,
        ):
            raise ValueError(
                "all bag initializations must share static coordinates"
            )

    prior = _as_mapping(
        request.payload["parameter_prior"], "parameter prior"
    )
    return PreparedBatchGraphData(
        parameter_chart=parameter_chart,
        initial_parameter_coordinates=(
            selection.initial_parameter_coordinates
        ),
        static_parameter_prior=VectorGaussianPrior(
            np.asarray(prior["mean_coordinate"], dtype=float),
            GaussianCovariance(
                np.asarray(prior["covariance"], dtype=float)
            ),
        ),
        geometry=geometry,
        dynamics=PreparedDynamicsConfiguration(
            q_definition=q_definition,
            q=selection.q_diagonal,
            gravity_world=np.asarray(
                (0.0, 0.0, -gravity_values[0]), dtype=float
            ),
        ),
        fixed_delay=selection.fixed_delay_seconds,
        scaling=scaling,
        bags=prepared_bags,
    )


__all__ = [
    "PreparationSelection",
    "prepare_fixed_batch_graph_data",
]
