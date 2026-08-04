"""ROS-free prepared-data contract for one fixed-Q, fixed-delay graph.

The builder in this module only connects already-audited, bracketed data to
analytic factors.  It performs no rosbag access, timestamp interpolation
search, finite differencing, Q selection, or delay profiling.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.batch.factors.actuator import (
    evaluate_actuator_interval_factor,
    evaluate_gimbal_position_factor,
)
from grape_param_estim.batch.factors.controller import (
    evaluate_controller_step_factors,
)
from grape_param_estim.batch.factors.dynamics import (
    evaluate_raw_dynamics_residual,
)
from grape_param_estim.batch.factors.dynamics_factor import (
    BODY_WRENCH_QUANTITY,
    SPECIFIC_ACCELERATION_QUANTITY,
    body_wrench_statistical_residual,
    evaluate_dynamics_factor,
    specific_acceleration_statistical_residual,
)
from grape_param_estim.batch.factors.imu import evaluate_gyro_factor
from grape_param_estim.batch.factors.kinematics import (
    evaluate_orientation_kinematic_factor,
    evaluate_position_kinematic_factor,
)
from grape_param_estim.batch.factors.pose import (
    evaluate_pose_observation_factors,
)
from grape_param_estim.batch.factors.prior import (
    evaluate_orientation_prior_factor,
    evaluate_vector_prior_factor,
)
from grape_param_estim.batch.factors.velocity import (
    evaluate_world_sensor_velocity_factor,
)
from grape_param_estim.batch.laplace_em import DiagonalQDefinition
from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.problem import BatchProblem
from grape_param_estim.batch.state import BatchState, StateScaling
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.controller import GrapeController
from grape_param_estim.parameterization import (
    PARAMETER_DIMENSION,
    VehicleParameterChart,
)
from grape_param_estim.system import (
    ActuatorCommand,
    ActuatorParameters,
    GrapeGeometry,
    ReferenceState,
)


_KNOT_KINDS = (
    VariableKind.POSITION,
    VariableKind.ORIENTATION_TANGENT,
    VariableKind.LINEAR_VELOCITY,
    VariableKind.ANGULAR_VELOCITY,
    VariableKind.CONTROLLER_INTEGRAL,
    VariableKind.ACTUATOR_THRUST,
    VariableKind.GIMBAL_ANGLE,
)


def _immutable_vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(
            "{} must contain {} finite values".format(name, size)
        )
    result = result.copy()
    result.setflags(write=False)
    return result


def _immutable_rotation(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite 3 by 3 matrix".format(name))
    if not np.allclose(result.T @ result, np.eye(3), rtol=0.0, atol=1.0e-9):
        raise ValueError("{} must be orthonormal".format(name))
    if not np.isclose(np.linalg.det(result), 1.0, rtol=0.0, atol=1.0e-9):
        raise ValueError("{} must have determinant one".format(name))
    result = result.copy()
    result.setflags(write=False)
    return result


def _canonical_tuple(value, item_type, name: str):
    if type(value) is not tuple or any(
        not isinstance(item, item_type) for item in value
    ):
        raise TypeError(
            "{} must be a tuple of {} values".format(
                name, item_type.__name__
            )
        )
    return tuple(value)


@dataclass(frozen=True)
class GaussianCovariance:
    """One immutable symmetric positive-definite covariance matrix."""

    value: np.ndarray

    def __post_init__(self) -> None:
        covariance = np.asarray(self.value, dtype=float)
        if (
            covariance.ndim != 2
            or covariance.shape[0] == 0
            or covariance.shape[0] != covariance.shape[1]
            or not np.all(np.isfinite(covariance))
            or not np.allclose(
                covariance, covariance.T, rtol=0.0, atol=1.0e-12
            )
        ):
            raise ValueError("covariance must be a finite symmetric matrix")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("covariance must be positive definite") from error
        covariance = covariance.copy()
        covariance.setflags(write=False)
        object.__setattr__(self, "value", covariance)

    @property
    def dimension(self) -> int:
        return int(self.value.shape[0])

    @property
    def square_root_information(self) -> np.ndarray:
        """Return W with ``W.T @ W == covariance^-1``."""

        cholesky = np.linalg.cholesky(self.value)
        result = np.linalg.solve(cholesky, np.eye(self.dimension))
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class VectorGaussianPrior:
    mean: np.ndarray
    covariance: GaussianCovariance

    def __post_init__(self) -> None:
        if not isinstance(self.covariance, GaussianCovariance):
            raise TypeError("covariance must be GaussianCovariance")
        object.__setattr__(
            self,
            "mean",
            _immutable_vector(
                self.mean, self.covariance.dimension, "prior mean"
            ),
        )


@dataclass(frozen=True)
class OrientationGaussianPrior:
    mean_rotation: np.ndarray
    covariance: GaussianCovariance

    def __post_init__(self) -> None:
        if not isinstance(self.covariance, GaussianCovariance):
            raise TypeError("covariance must be GaussianCovariance")
        if self.covariance.dimension != 3:
            raise ValueError("orientation prior covariance must be 3 by 3")
        object.__setattr__(
            self,
            "mean_rotation",
            _immutable_rotation(self.mean_rotation, "prior mean_rotation"),
        )


@dataclass(frozen=True)
class PreparedKnotState:
    """Initial values for all 26 continuous variables at one knot."""

    time: float
    position: np.ndarray
    rotation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    controller_integral: np.ndarray
    actuator_thrust: np.ndarray
    gimbal_angle: np.ndarray

    def __post_init__(self) -> None:
        timestamp = float(self.time)
        if not np.isfinite(timestamp):
            raise ValueError("knot time must be finite")
        object.__setattr__(self, "time", timestamp)
        for name, size in (
            ("position", 3),
            ("linear_velocity", 3),
            ("angular_velocity", 3),
            ("controller_integral", 6),
            ("actuator_thrust", 4),
            ("gimbal_angle", 4),
        ):
            object.__setattr__(
                self,
                name,
                _immutable_vector(getattr(self, name), size, name),
            )
        object.__setattr__(
            self,
            "rotation",
            _immutable_rotation(self.rotation, "rotation"),
        )


@dataclass(frozen=True)
class PreparedKnotPrior:
    position: VectorGaussianPrior
    rotation: OrientationGaussianPrior
    linear_velocity: VectorGaussianPrior
    angular_velocity: VectorGaussianPrior
    controller_integral: VectorGaussianPrior
    actuator_thrust: VectorGaussianPrior
    gimbal_angle: VectorGaussianPrior

    def __post_init__(self) -> None:
        for name, size, prior_type in (
            ("position", 3, VectorGaussianPrior),
            ("rotation", 3, OrientationGaussianPrior),
            ("linear_velocity", 3, VectorGaussianPrior),
            ("angular_velocity", 3, VectorGaussianPrior),
            ("controller_integral", 6, VectorGaussianPrior),
            ("actuator_thrust", 4, VectorGaussianPrior),
            ("gimbal_angle", 4, VectorGaussianPrior),
        ):
            prior = getattr(self, name)
            if not isinstance(prior, prior_type):
                raise TypeError("{} prior has the wrong type".format(name))
            if prior.covariance.dimension != size:
                raise ValueError(
                    "{} prior must have dimension {}".format(name, size)
                )


@dataclass(frozen=True)
class PreparedBagPriors:
    gyro_bias: VectorGaussianPrior
    initial_knot: PreparedKnotPrior

    def __post_init__(self) -> None:
        if not isinstance(self.gyro_bias, VectorGaussianPrior):
            raise TypeError("gyro_bias must be VectorGaussianPrior")
        if self.gyro_bias.covariance.dimension != 3:
            raise ValueError("gyro bias prior must have dimension 3")
        if not isinstance(self.initial_knot, PreparedKnotPrior):
            raise TypeError("initial_knot must be PreparedKnotPrior")


@dataclass(frozen=True)
class MeasurementBracket:
    """A precomputed adjacent-knot bracket for one asynchronous sample."""

    left_knot_index: int
    interpolation_fraction: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.left_knot_index, (bool, np.bool_))
            or not isinstance(self.left_knot_index, (int, np.integer))
            or self.left_knot_index < 0
        ):
            raise ValueError("left_knot_index must be a non-negative integer")
        fraction = float(self.interpolation_fraction)
        if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
            raise ValueError(
                "interpolation_fraction must be finite and in [0, 1]"
            )
        object.__setattr__(self, "left_knot_index", int(self.left_knot_index))
        object.__setattr__(self, "interpolation_fraction", fraction)


@dataclass(frozen=True)
class PreparedPoseMeasurement:
    bracket: MeasurementBracket
    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.bracket, MeasurementBracket):
            raise TypeError("bracket must be MeasurementBracket")
        object.__setattr__(
            self, "position", _immutable_vector(self.position, 3, "position")
        )
        object.__setattr__(
            self,
            "rotation",
            _immutable_rotation(self.rotation, "rotation"),
        )


@dataclass(frozen=True)
class PreparedVelocityMeasurement:
    bracket: MeasurementBracket
    velocity_world: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.bracket, MeasurementBracket):
            raise TypeError("bracket must be MeasurementBracket")
        object.__setattr__(
            self,
            "velocity_world",
            _immutable_vector(self.velocity_world, 3, "velocity_world"),
        )


@dataclass(frozen=True)
class PreparedGyroMeasurement:
    bracket: MeasurementBracket
    angular_velocity_sensor: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.bracket, MeasurementBracket):
            raise TypeError("bracket must be MeasurementBracket")
        object.__setattr__(
            self,
            "angular_velocity_sensor",
            _immutable_vector(
                self.angular_velocity_sensor,
                3,
                "angular_velocity_sensor",
            ),
        )


@dataclass(frozen=True)
class PreparedGimbalMeasurement:
    bracket: MeasurementBracket
    gimbal_angle: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.bracket, MeasurementBracket):
            raise TypeError("bracket must be MeasurementBracket")
        object.__setattr__(
            self,
            "gimbal_angle",
            _immutable_vector(self.gimbal_angle, 4, "gimbal_angle"),
        )


@dataclass(frozen=True)
class PreparedControllerInterval:
    """Fixed reference and discrete controller schedule for one interval."""

    left_knot_index: int
    reference: ReferenceState
    roll_pitch_integration_active: bool
    issued_thrust_observation: Optional[np.ndarray]
    issued_gimbal_observation: Optional[np.ndarray]

    def __post_init__(self) -> None:
        if (
            isinstance(self.left_knot_index, (bool, np.bool_))
            or not isinstance(self.left_knot_index, (int, np.integer))
            or self.left_knot_index < 0
        ):
            raise ValueError("left_knot_index must be a non-negative integer")
        if not isinstance(self.reference, ReferenceState):
            raise TypeError("reference must be ReferenceState")
        if not isinstance(
            self.roll_pitch_integration_active, (bool, np.bool_)
        ):
            raise TypeError("roll_pitch_integration_active must be boolean")
        object.__setattr__(self, "left_knot_index", int(self.left_knot_index))
        object.__setattr__(
            self,
            "roll_pitch_integration_active",
            bool(self.roll_pitch_integration_active),
        )
        for name in (
            "issued_thrust_observation",
            "issued_gimbal_observation",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, _immutable_vector(value, 4, name)
                )


@dataclass(frozen=True)
class PreparedCommandSegment:
    command: ActuatorCommand
    duration: float

    def __post_init__(self) -> None:
        if not isinstance(self.command, ActuatorCommand):
            raise TypeError("command must be ActuatorCommand")
        duration = float(self.duration)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("command segment duration must be positive")
        object.__setattr__(self, "duration", duration)


@dataclass(frozen=True)
class PreparedActuatorInterval:
    left_knot_index: int
    delayed_command_segments: Tuple[PreparedCommandSegment, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.left_knot_index, (bool, np.bool_))
            or not isinstance(self.left_knot_index, (int, np.integer))
            or self.left_knot_index < 0
        ):
            raise ValueError("left_knot_index must be a non-negative integer")
        segments = _canonical_tuple(
            self.delayed_command_segments,
            PreparedCommandSegment,
            "delayed_command_segments",
        )
        if not segments:
            raise ValueError("delayed_command_segments cannot be empty")
        object.__setattr__(self, "left_knot_index", int(self.left_knot_index))
        object.__setattr__(self, "delayed_command_segments", segments)


@dataclass(frozen=True)
class PreparedSensorExtrinsics:
    pose_sensor_position_in_body: np.ndarray
    pose_sensor_to_body_rotation: np.ndarray
    velocity_sensor_position_in_body: np.ndarray
    body_to_gyro_sensor_rotation: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "pose_sensor_position_in_body",
            "velocity_sensor_position_in_body",
        ):
            object.__setattr__(
                self,
                name,
                _immutable_vector(getattr(self, name), 3, name),
            )
        for name in (
            "pose_sensor_to_body_rotation",
            "body_to_gyro_sensor_rotation",
        ):
            object.__setattr__(
                self,
                name,
                _immutable_rotation(getattr(self, name), name),
            )


@dataclass(frozen=True)
class PreparedFactorCovariances:
    """Independent sensor and fixed model-tolerance covariances."""

    position_observation: GaussianCovariance
    orientation_observation: GaussianCovariance
    velocity_observation: GaussianCovariance
    gyro_observation: GaussianCovariance
    issued_thrust_observation: GaussianCovariance
    issued_gimbal_observation: GaussianCovariance
    actual_gimbal_observation: GaussianCovariance
    controller_integral_transition: GaussianCovariance
    actuator_thrust_transition: GaussianCovariance
    actuator_gimbal_transition: GaussianCovariance
    position_kinematic: GaussianCovariance
    orientation_kinematic: GaussianCovariance

    def __post_init__(self) -> None:
        for name, dimension in (
            ("position_observation", 3),
            ("orientation_observation", 3),
            ("velocity_observation", 3),
            ("gyro_observation", 3),
            ("issued_thrust_observation", 4),
            ("issued_gimbal_observation", 4),
            ("actual_gimbal_observation", 4),
            ("controller_integral_transition", 6),
            ("actuator_thrust_transition", 4),
            ("actuator_gimbal_transition", 4),
            ("position_kinematic", 3),
            ("orientation_kinematic", 3),
        ):
            covariance = getattr(self, name)
            if not isinstance(covariance, GaussianCovariance):
                raise TypeError("{} must be GaussianCovariance".format(name))
            if covariance.dimension != dimension:
                raise ValueError(
                    "{} covariance must have dimension {}".format(
                        name, dimension
                    )
                )


@dataclass(frozen=True)
class AccelerometerFactorContract:
    """Explicit v1 accelerometer availability gate.

    The factor is not yet implemented.  Callers must preserve the audited
    disabled reason rather than silently omitting acceleration samples.
    """

    enabled: bool
    disabled_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, (bool, np.bool_)):
            raise TypeError("accelerometer enabled must be boolean")
        if bool(self.enabled):
            raise ValueError("accelerometer factor is not implemented in v1")
        if (
            not isinstance(self.disabled_reason, str)
            or not self.disabled_reason
            or self.disabled_reason.strip() != self.disabled_reason
        ):
            raise ValueError(
                "disabled accelerometer requires a canonical reason"
            )
        object.__setattr__(self, "enabled", False)


@dataclass(frozen=True)
class PreparedDynamicsConfiguration:
    """Explicit Q coordinates and physical residual settings."""

    q_definition: DiagonalQDefinition
    q: np.ndarray
    gravity_world: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.q_definition, DiagonalQDefinition):
            raise TypeError("q_definition must be DiagonalQDefinition")
        if self.q_definition.residual_quantity not in (
            BODY_WRENCH_QUANTITY,
            SPECIFIC_ACCELERATION_QUANTITY,
        ):
            raise ValueError(
                "q_definition residual_quantity must explicitly be "
                "body_wrench or specific_acceleration"
            )
        q = _immutable_vector(self.q, 6, "q")
        if np.any(q <= 0.0):
            raise ValueError("q must contain six positive values")
        object.__setattr__(self, "q", q)
        object.__setattr__(
            self,
            "gravity_world",
            _immutable_vector(self.gravity_world, 3, "gravity_world"),
        )


def _validate_measurement_brackets(measurements, knot_count: int, name: str):
    last = None
    for measurement in measurements:
        bracket = measurement.bracket
        if bracket.left_knot_index >= knot_count - 1:
            raise ValueError("{} bracket is outside knot support".format(name))
        order = (bracket.left_knot_index, bracket.interpolation_fraction)
        if last is not None and order < last:
            raise ValueError("{} must be in chronological bracket order".format(name))
        last = order


@dataclass(frozen=True)
class PreparedBagGraphData:
    """All solver/ROS-free inputs for one bag-local trajectory graph."""

    bag_id: str
    knots: Tuple[PreparedKnotState, ...]
    initial_gyro_bias: np.ndarray
    priors: PreparedBagPriors
    controller: GrapeController
    controller_intervals: Tuple[PreparedControllerInterval, ...]
    actuator_parameters: ActuatorParameters
    actuator_intervals: Tuple[PreparedActuatorInterval, ...]
    pose_measurements: Tuple[PreparedPoseMeasurement, ...]
    velocity_measurements: Tuple[PreparedVelocityMeasurement, ...]
    gyro_measurements: Tuple[PreparedGyroMeasurement, ...]
    actual_gimbal_measurements: Tuple[PreparedGimbalMeasurement, ...]
    sensor_extrinsics: PreparedSensorExtrinsics
    covariances: PreparedFactorCovariances
    accelerometer: AccelerometerFactorContract

    def __post_init__(self) -> None:
        if (
            not isinstance(self.bag_id, str)
            or not self.bag_id
            or self.bag_id.strip() != self.bag_id
        ):
            raise ValueError("bag_id must be a canonical non-empty string")
        knots = _canonical_tuple(self.knots, PreparedKnotState, "knots")
        if len(knots) < 2:
            raise ValueError("a prepared bag requires at least two knots")
        times = np.asarray(tuple(knot.time for knot in knots), dtype=float)
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("knot times must be strictly increasing")
        object.__setattr__(self, "knots", knots)
        object.__setattr__(
            self,
            "initial_gyro_bias",
            _immutable_vector(self.initial_gyro_bias, 3, "initial_gyro_bias"),
        )
        if not isinstance(self.priors, PreparedBagPriors):
            raise TypeError("priors must be PreparedBagPriors")
        if not isinstance(self.controller, GrapeController):
            raise TypeError("controller must be GrapeController")
        if not isinstance(self.actuator_parameters, ActuatorParameters):
            raise TypeError("actuator_parameters must be ActuatorParameters")
        controllers = _canonical_tuple(
            self.controller_intervals,
            PreparedControllerInterval,
            "controller_intervals",
        )
        actuators = _canonical_tuple(
            self.actuator_intervals,
            PreparedActuatorInterval,
            "actuator_intervals",
        )
        interval_count = len(knots) - 1
        if len(controllers) != interval_count or len(actuators) != interval_count:
            raise ValueError(
                "controller and actuator data must cover every knot interval"
            )
        for index, (controller, actuator) in enumerate(
            zip(controllers, actuators)
        ):
            if (
                controller.left_knot_index != index
                or actuator.left_knot_index != index
            ):
                raise ValueError(
                    "controller and actuator intervals must be contiguous"
                )
            time_step = knots[index + 1].time - knots[index].time
            duration = sum(
                segment.duration
                for segment in actuator.delayed_command_segments
            )
            tolerance = 1.0e-10 * max(1.0, abs(time_step))
            if abs(duration - time_step) > tolerance:
                raise ValueError(
                    "delayed command segment durations must equal knot dt"
                )
        object.__setattr__(self, "controller_intervals", controllers)
        object.__setattr__(self, "actuator_intervals", actuators)

        for name, item_type in (
            ("pose_measurements", PreparedPoseMeasurement),
            ("velocity_measurements", PreparedVelocityMeasurement),
            ("gyro_measurements", PreparedGyroMeasurement),
            ("actual_gimbal_measurements", PreparedGimbalMeasurement),
        ):
            measurements = _canonical_tuple(
                getattr(self, name), item_type, name
            )
            _validate_measurement_brackets(measurements, len(knots), name)
            object.__setattr__(self, name, measurements)
        if not isinstance(self.sensor_extrinsics, PreparedSensorExtrinsics):
            raise TypeError("sensor_extrinsics must be PreparedSensorExtrinsics")
        if not isinstance(self.covariances, PreparedFactorCovariances):
            raise TypeError("covariances must be PreparedFactorCovariances")
        if not isinstance(self.accelerometer, AccelerometerFactorContract):
            raise TypeError("accelerometer must be AccelerometerFactorContract")


@dataclass(frozen=True)
class PreparedBatchGraphData:
    """Complete fixed-Q/fixed-delay graph construction input."""

    parameter_chart: VehicleParameterChart
    initial_parameter_coordinates: np.ndarray
    static_parameter_prior: VectorGaussianPrior
    geometry: GrapeGeometry
    dynamics: PreparedDynamicsConfiguration
    fixed_delay: float
    scaling: StateScaling
    bags: Tuple[PreparedBagGraphData, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_chart, VehicleParameterChart):
            raise TypeError("parameter_chart must be VehicleParameterChart")
        object.__setattr__(
            self,
            "initial_parameter_coordinates",
            _immutable_vector(
                self.initial_parameter_coordinates,
                PARAMETER_DIMENSION,
                "initial_parameter_coordinates",
            ),
        )
        if not isinstance(self.static_parameter_prior, VectorGaussianPrior):
            raise TypeError(
                "static_parameter_prior must be VectorGaussianPrior"
            )
        if self.static_parameter_prior.covariance.dimension != PARAMETER_DIMENSION:
            raise ValueError("static parameter prior must have dimension 18")
        if not isinstance(self.geometry, GrapeGeometry):
            raise TypeError("geometry must be GrapeGeometry")
        if not isinstance(self.dynamics, PreparedDynamicsConfiguration):
            raise TypeError("dynamics must be PreparedDynamicsConfiguration")
        delay = float(self.fixed_delay)
        if not np.isfinite(delay) or delay < 0.0:
            raise ValueError("fixed_delay must be finite and non-negative")
        object.__setattr__(self, "fixed_delay", delay)
        if not isinstance(self.scaling, StateScaling):
            raise TypeError("scaling must be StateScaling")
        bags = _canonical_tuple(self.bags, PreparedBagGraphData, "bags")
        if not bags:
            raise ValueError("bags cannot be empty")
        if len({bag.bag_id for bag in bags}) != len(bags):
            raise ValueError("bag_id values must be unique")
        object.__setattr__(self, "bags", bags)


def _layout(prepared: PreparedBatchGraphData) -> VariableLayout:
    keys = [VariableKey(VariableKind.STATIC_PARAMETERS)]
    for bag in sorted(prepared.bags, key=lambda item: item.bag_id):
        keys.append(VariableKey(VariableKind.GYRO_BIAS, bag_id=bag.bag_id))
        for knot_index in range(len(bag.knots)):
            keys.extend(
                VariableKey(
                    kind,
                    bag_id=bag.bag_id,
                    knot_index=knot_index,
                )
                for kind in _KNOT_KINDS
            )
    return VariableLayout(tuple(keys))


def build_initial_batch_state(prepared: PreparedBatchGraphData) -> BatchState:
    """Materialize the prepared initial values in canonical layout order."""

    if not isinstance(prepared, PreparedBatchGraphData):
        raise TypeError("prepared must be PreparedBatchGraphData")
    layout = _layout(prepared)
    values = {
        VariableKey(VariableKind.STATIC_PARAMETERS): (
            prepared.initial_parameter_coordinates
        )
    }
    for bag in prepared.bags:
        values[VariableKey(VariableKind.GYRO_BIAS, bag_id=bag.bag_id)] = (
            bag.initial_gyro_bias
        )
        for knot_index, knot in enumerate(bag.knots):
            for kind, value in (
                (VariableKind.POSITION, knot.position),
                (VariableKind.ORIENTATION_TANGENT, knot.rotation),
                (VariableKind.LINEAR_VELOCITY, knot.linear_velocity),
                (VariableKind.ANGULAR_VELOCITY, knot.angular_velocity),
                (VariableKind.CONTROLLER_INTEGRAL, knot.controller_integral),
                (VariableKind.ACTUATOR_THRUST, knot.actuator_thrust),
                (VariableKind.GIMBAL_ANGLE, knot.gimbal_angle),
            ):
                values[
                    VariableKey(
                        kind,
                        bag_id=bag.bag_id,
                        knot_index=knot_index,
                    )
                ] = value
    return BatchState(layout, values)


def _key(kind: VariableKind, bag_id: str, knot_index: int) -> VariableKey:
    return VariableKey(kind, bag_id=bag_id, knot_index=knot_index)


def _knot_value(
    state: BatchState,
    bag_id: str,
    knot_index: int,
    kind: VariableKind,
):
    return state.knot_value(bag_id, knot_index, kind)


def _evaluate_prepared_factors(
    prepared: PreparedBatchGraphData,
    state: BatchState,
):
    factors = []
    static_key = VariableKey(VariableKind.STATIC_PARAMETERS)
    coordinates = state.value(static_key)
    parameters, parameter_jacobian = (
        prepared.parameter_chart.decode_with_jacobian(coordinates)
    )
    bags = sorted(prepared.bags, key=lambda item: item.bag_id)

    # 1. Priors: shared first, then bag bias and initial knot fields.
    prior = prepared.static_parameter_prior
    factors.append(
        evaluate_vector_prior_factor(
            static_key,
            coordinates,
            prior.mean,
            prior.covariance.square_root_information,
        )
    )
    for bag in bags:
        bias_key = VariableKey(VariableKind.GYRO_BIAS, bag_id=bag.bag_id)
        bias_prior = bag.priors.gyro_bias
        factors.append(
            evaluate_vector_prior_factor(
                bias_key,
                state.value(bias_key),
                bias_prior.mean,
                bias_prior.covariance.square_root_information,
            )
        )
        knot_prior = bag.priors.initial_knot
        for kind, selected_prior in (
            (VariableKind.POSITION, knot_prior.position),
            (VariableKind.ORIENTATION_TANGENT, knot_prior.rotation),
            (VariableKind.LINEAR_VELOCITY, knot_prior.linear_velocity),
            (VariableKind.ANGULAR_VELOCITY, knot_prior.angular_velocity),
            (VariableKind.CONTROLLER_INTEGRAL, knot_prior.controller_integral),
            (VariableKind.ACTUATOR_THRUST, knot_prior.actuator_thrust),
            (VariableKind.GIMBAL_ANGLE, knot_prior.gimbal_angle),
        ):
            key = _key(kind, bag.bag_id, 0)
            if kind is VariableKind.ORIENTATION_TANGENT:
                factors.append(
                    evaluate_orientation_prior_factor(
                        key,
                        state.value(key),
                        selected_prior.mean_rotation,
                        selected_prior.covariance.square_root_information,
                    )
                )
            else:
                factors.append(
                    evaluate_vector_prior_factor(
                        key,
                        state.value(key),
                        selected_prior.mean,
                        selected_prior.covariance.square_root_information,
                    )
                )

    # 2. Asynchronous pose observations, position before orientation.
    for bag in bags:
        covariance = bag.covariances
        extrinsics = bag.sensor_extrinsics
        for measurement in bag.pose_measurements:
            index = measurement.bracket.left_knot_index
            factors.extend(
                evaluate_pose_observation_factors(
                    bag_id=bag.bag_id,
                    left_knot_index=index,
                    interpolation_fraction=(
                        measurement.bracket.interpolation_fraction
                    ),
                    position_left=_knot_value(
                        state, bag.bag_id, index, VariableKind.POSITION
                    ),
                    position_right=_knot_value(
                        state, bag.bag_id, index + 1, VariableKind.POSITION
                    ),
                    rotation_left=_knot_value(
                        state,
                        bag.bag_id,
                        index,
                        VariableKind.ORIENTATION_TANGENT,
                    ),
                    rotation_right=_knot_value(
                        state,
                        bag.bag_id,
                        index + 1,
                        VariableKind.ORIENTATION_TANGENT,
                    ),
                    observed_sensor_position=measurement.position,
                    observed_sensor_rotation=measurement.rotation,
                    sensor_position_in_body=(
                        extrinsics.pose_sensor_position_in_body
                    ),
                    sensor_to_body_rotation=(
                        extrinsics.pose_sensor_to_body_rotation
                    ),
                    cog_offset_in_body=parameters.cog_offset,
                    cog_offset_chart_jacobian=(
                        parameter_jacobian.cog_offset
                    ),
                    position_square_root_information=(
                        covariance.position_observation
                        .square_root_information
                    ),
                    orientation_square_root_information=(
                        covariance.orientation_observation
                        .square_root_information
                    ),
                )
            )

    # 3. Asynchronous world-frame sensor velocity observations.
    for bag in bags:
        for measurement in bag.velocity_measurements:
            index = measurement.bracket.left_knot_index
            factors.append(
                evaluate_world_sensor_velocity_factor(
                    bag_id=bag.bag_id,
                    left_knot_index=index,
                    interpolation_fraction=(
                        measurement.bracket.interpolation_fraction
                    ),
                    velocity_left=_knot_value(
                        state,
                        bag.bag_id,
                        index,
                        VariableKind.LINEAR_VELOCITY,
                    ),
                    velocity_right=_knot_value(
                        state,
                        bag.bag_id,
                        index + 1,
                        VariableKind.LINEAR_VELOCITY,
                    ),
                    rotation_left=_knot_value(
                        state,
                        bag.bag_id,
                        index,
                        VariableKind.ORIENTATION_TANGENT,
                    ),
                    rotation_right=_knot_value(
                        state,
                        bag.bag_id,
                        index + 1,
                        VariableKind.ORIENTATION_TANGENT,
                    ),
                    angular_velocity_left=_knot_value(
                        state,
                        bag.bag_id,
                        index,
                        VariableKind.ANGULAR_VELOCITY,
                    ),
                    angular_velocity_right=_knot_value(
                        state,
                        bag.bag_id,
                        index + 1,
                        VariableKind.ANGULAR_VELOCITY,
                    ),
                    observed_sensor_velocity_world=(
                        measurement.velocity_world
                    ),
                    sensor_position_in_body=(
                        bag.sensor_extrinsics.velocity_sensor_position_in_body
                    ),
                    cog_offset_in_body=parameters.cog_offset,
                    cog_offset_chart_jacobian=(
                        parameter_jacobian.cog_offset
                    ),
                    square_root_information=(
                        bag.covariances.velocity_observation
                        .square_root_information
                    ),
                )
            )

    # 4. Gyroscope observations.  Accelerometer omission is explicit above.
    for bag in bags:
        gyro_bias = state.value(
            VariableKey(VariableKind.GYRO_BIAS, bag_id=bag.bag_id)
        )
        for measurement in bag.gyro_measurements:
            index = measurement.bracket.left_knot_index
            factors.append(
                evaluate_gyro_factor(
                    bag_id=bag.bag_id,
                    left_knot_index=index,
                    interpolation_fraction=(
                        measurement.bracket.interpolation_fraction
                    ),
                    angular_velocity_left=_knot_value(
                        state,
                        bag.bag_id,
                        index,
                        VariableKind.ANGULAR_VELOCITY,
                    ),
                    angular_velocity_right=_knot_value(
                        state,
                        bag.bag_id,
                        index + 1,
                        VariableKind.ANGULAR_VELOCITY,
                    ),
                    gyro_bias_sensor=gyro_bias,
                    observed_angular_velocity_sensor=(
                        measurement.angular_velocity_sensor
                    ),
                    body_to_sensor_rotation=(
                        bag.sensor_extrinsics.body_to_gyro_sensor_rotation
                    ),
                    square_root_information=(
                        bag.covariances.gyro_observation
                        .square_root_information
                    ),
                )
            )

    # 5. Fixed-schedule controller transitions and issued commands.
    for bag in bags:
        for interval in bag.controller_intervals:
            index = interval.left_knot_index
            dt = bag.knots[index + 1].time - bag.knots[index].time
            result = evaluate_controller_step_factors(
                controller=bag.controller,
                bag_id=bag.bag_id,
                knot_index=index,
                position=_knot_value(
                    state, bag.bag_id, index, VariableKind.POSITION
                ),
                rotation=_knot_value(
                    state,
                    bag.bag_id,
                    index,
                    VariableKind.ORIENTATION_TANGENT,
                ),
                world_velocity=_knot_value(
                    state,
                    bag.bag_id,
                    index,
                    VariableKind.LINEAR_VELOCITY,
                ),
                body_omega=_knot_value(
                    state,
                    bag.bag_id,
                    index,
                    VariableKind.ANGULAR_VELOCITY,
                ),
                integral_left=_knot_value(
                    state,
                    bag.bag_id,
                    index,
                    VariableKind.CONTROLLER_INTEGRAL,
                ),
                integral_right=_knot_value(
                    state,
                    bag.bag_id,
                    index + 1,
                    VariableKind.CONTROLLER_INTEGRAL,
                ),
                current_gimbal_angle=_knot_value(
                    state,
                    bag.bag_id,
                    index,
                    VariableKind.GIMBAL_ANGLE,
                ),
                reference=interval.reference,
                time_step=dt,
                roll_pitch_integration_active=(
                    interval.roll_pitch_integration_active
                ),
                integral_square_root_information=(
                    bag.covariances.controller_integral_transition
                    .square_root_information
                ),
                issued_thrust_observation=(
                    interval.issued_thrust_observation
                ),
                issued_thrust_square_root_information=(
                    None
                    if interval.issued_thrust_observation is None
                    else bag.covariances.issued_thrust_observation
                    .square_root_information
                ),
                issued_gimbal_observation=(
                    interval.issued_gimbal_observation
                ),
                issued_gimbal_square_root_information=(
                    None
                    if interval.issued_gimbal_observation is None
                    else bag.covariances.issued_gimbal_observation
                    .square_root_information
                ),
            )
            factors.extend(result.factors)

    # 6. Delayed actuator transitions, then actual gimbal observations.
    for bag in bags:
        for interval in bag.actuator_intervals:
            index = interval.left_knot_index
            factors.append(
                evaluate_actuator_interval_factor(
                    bag_id=bag.bag_id,
                    left_knot_index=index,
                    thrust_left=_knot_value(
                        state,
                        bag.bag_id,
                        index,
                        VariableKind.ACTUATOR_THRUST,
                    ),
                    thrust_right=_knot_value(
                        state,
                        bag.bag_id,
                        index + 1,
                        VariableKind.ACTUATOR_THRUST,
                    ),
                    gimbal_left=_knot_value(
                        state,
                        bag.bag_id,
                        index,
                        VariableKind.GIMBAL_ANGLE,
                    ),
                    gimbal_right=_knot_value(
                        state,
                        bag.bag_id,
                        index + 1,
                        VariableKind.GIMBAL_ANGLE,
                    ),
                    command_segments=tuple(
                        (segment.command, segment.duration)
                        for segment in interval.delayed_command_segments
                    ),
                    actuator_parameters=bag.actuator_parameters,
                    thrust_square_root_information=(
                        bag.covariances.actuator_thrust_transition
                        .square_root_information
                    ),
                    gimbal_square_root_information=(
                        bag.covariances.actuator_gimbal_transition
                        .square_root_information
                    ),
                )
            )
        for measurement in bag.actual_gimbal_measurements:
            index = measurement.bracket.left_knot_index
            factors.append(
                evaluate_gimbal_position_factor(
                    bag_id=bag.bag_id,
                    left_knot_index=index,
                    interpolation_fraction=(
                        measurement.bracket.interpolation_fraction
                    ),
                    gimbal_left=_knot_value(
                        state,
                        bag.bag_id,
                        index,
                        VariableKind.GIMBAL_ANGLE,
                    ),
                    gimbal_right=_knot_value(
                        state,
                        bag.bag_id,
                        index + 1,
                        VariableKind.GIMBAL_ANGLE,
                    ),
                    observed_gimbal_position=measurement.gimbal_angle,
                    square_root_information=(
                        bag.covariances.actual_gimbal_observation
                        .square_root_information
                    ),
                )
            )

    # 7. Fixed-covariance kinematic consistency.
    for bag in bags:
        for index in range(len(bag.knots) - 1):
            dt = bag.knots[index + 1].time - bag.knots[index].time
            factors.append(
                evaluate_position_kinematic_factor(
                    bag_id=bag.bag_id,
                    knot_index=index,
                    position_left=_knot_value(
                        state, bag.bag_id, index, VariableKind.POSITION
                    ),
                    position_right=_knot_value(
                        state, bag.bag_id, index + 1, VariableKind.POSITION
                    ),
                    velocity_left=_knot_value(
                        state,
                        bag.bag_id,
                        index,
                        VariableKind.LINEAR_VELOCITY,
                    ),
                    velocity_right=_knot_value(
                        state,
                        bag.bag_id,
                        index + 1,
                        VariableKind.LINEAR_VELOCITY,
                    ),
                    time_step=dt,
                    square_root_information=(
                        bag.covariances.position_kinematic
                        .square_root_information
                    ),
                )
            )
            factors.append(
                evaluate_orientation_kinematic_factor(
                    bag_id=bag.bag_id,
                    knot_index=index,
                    rotation_left=_knot_value(
                        state,
                        bag.bag_id,
                        index,
                        VariableKind.ORIENTATION_TANGENT,
                    ),
                    rotation_right=_knot_value(
                        state,
                        bag.bag_id,
                        index + 1,
                        VariableKind.ORIENTATION_TANGENT,
                    ),
                    angular_velocity_left=_knot_value(
                        state,
                        bag.bag_id,
                        index,
                        VariableKind.ANGULAR_VELOCITY,
                    ),
                    angular_velocity_right=_knot_value(
                        state,
                        bag.bag_id,
                        index + 1,
                        VariableKind.ANGULAR_VELOCITY,
                    ),
                    time_step=dt,
                    square_root_information=(
                        bag.covariances.orientation_kinematic
                        .square_root_information
                    ),
                )
            )

    # 8. Q-weighted dynamics.  There is intentionally no fallback quantity.
    for bag in bags:
        for index in range(len(bag.knots) - 1):
            dt = bag.knots[index + 1].time - bag.knots[index].time
            raw = evaluate_raw_dynamics_residual(
                rotation_left=_knot_value(
                    state,
                    bag.bag_id,
                    index,
                    VariableKind.ORIENTATION_TANGENT,
                ),
                rotation_right=_knot_value(
                    state,
                    bag.bag_id,
                    index + 1,
                    VariableKind.ORIENTATION_TANGENT,
                ),
                linear_velocity_left=_knot_value(
                    state,
                    bag.bag_id,
                    index,
                    VariableKind.LINEAR_VELOCITY,
                ),
                linear_velocity_right=_knot_value(
                    state,
                    bag.bag_id,
                    index + 1,
                    VariableKind.LINEAR_VELOCITY,
                ),
                angular_velocity_left=_knot_value(
                    state,
                    bag.bag_id,
                    index,
                    VariableKind.ANGULAR_VELOCITY,
                ),
                angular_velocity_right=_knot_value(
                    state,
                    bag.bag_id,
                    index + 1,
                    VariableKind.ANGULAR_VELOCITY,
                ),
                actuator_thrust_left=_knot_value(
                    state,
                    bag.bag_id,
                    index,
                    VariableKind.ACTUATOR_THRUST,
                ),
                actuator_thrust_right=_knot_value(
                    state,
                    bag.bag_id,
                    index + 1,
                    VariableKind.ACTUATOR_THRUST,
                ),
                gimbal_angle_left=_knot_value(
                    state,
                    bag.bag_id,
                    index,
                    VariableKind.GIMBAL_ANGLE,
                ),
                gimbal_angle_right=_knot_value(
                    state,
                    bag.bag_id,
                    index + 1,
                    VariableKind.GIMBAL_ANGLE,
                ),
                time_step=dt,
                parameter_chart=prepared.parameter_chart,
                parameter_coordinates=coordinates,
                geometry=prepared.geometry,
                gravity_world=prepared.dynamics.gravity_world,
            )
            quantity = prepared.dynamics.q_definition.residual_quantity
            if quantity == BODY_WRENCH_QUANTITY:
                statistical = body_wrench_statistical_residual(
                    bag.bag_id,
                    index,
                    raw,
                    prepared.dynamics.q_definition,
                )
            elif quantity == SPECIFIC_ACCELERATION_QUANTITY:
                statistical = specific_acceleration_statistical_residual(
                    bag.bag_id,
                    index,
                    raw,
                    prepared.dynamics.q_definition,
                    prepared.parameter_chart,
                    coordinates,
                )
            else:
                raise ValueError("unsupported dynamics residual quantity")
            factors.append(
                evaluate_dynamics_factor(statistical, prepared.dynamics.q, dt)
            )
    return tuple(factors)


def build_fixed_batch_problem(
    prepared: PreparedBatchGraphData,
) -> BatchProblem:
    """Build one deterministic analytic problem for fixed Q and delay data."""

    if not isinstance(prepared, PreparedBatchGraphData):
        raise TypeError("prepared must be PreparedBatchGraphData")
    layout = _layout(prepared)

    def evaluator(state):
        return _evaluate_prepared_factors(prepared, state)

    return BatchProblem(layout, prepared.scaling, evaluator)


__all__ = [
    "AccelerometerFactorContract",
    "GaussianCovariance",
    "MeasurementBracket",
    "OrientationGaussianPrior",
    "PreparedActuatorInterval",
    "PreparedBagGraphData",
    "PreparedBagPriors",
    "PreparedBatchGraphData",
    "PreparedCommandSegment",
    "PreparedControllerInterval",
    "PreparedDynamicsConfiguration",
    "PreparedFactorCovariances",
    "PreparedGimbalMeasurement",
    "PreparedGyroMeasurement",
    "PreparedKnotPrior",
    "PreparedKnotState",
    "PreparedPoseMeasurement",
    "PreparedSensorExtrinsics",
    "PreparedVelocityMeasurement",
    "VectorGaussianPrior",
    "build_fixed_batch_problem",
    "build_initial_batch_state",
]
