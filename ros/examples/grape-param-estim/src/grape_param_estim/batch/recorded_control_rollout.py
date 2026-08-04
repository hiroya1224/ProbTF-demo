"""Open-loop replay of the recorded actuator commands used by the batch graph.

The rollout starts once from a supplied bag-local state and then advances the
same rigid-body and actuator models used by the estimator.  It never invokes
the controller, resets from an observation, or injects a residual wrench.
Prepared command segments already include the selected command delay and ZOH
switch times, so replaying them is the exact recorded-control forward problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.batch.graph_builder import (
    PreparedActuatorInterval,
    PreparedBagGraphData,
)
from grape_param_estim.batch.state import BatchState
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.dynamics import (
    FullSixDofPlant,
    advance_actuators,
)
from grape_param_estim.geometry import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    so3_exp,
    so3_log,
)
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.system import (
    ActuatorState,
    GrapeGeometry,
    RigidBodyState,
)


def _immutable(value: object, shape: Tuple[int, ...], name: str) -> np.ndarray:
    selected = np.asarray(value)
    if selected.shape != shape:
        raise ValueError("{} must have shape {}".format(name, shape))
    if selected.dtype == np.bool_:
        result = selected.astype(bool, copy=True)
    else:
        result = selected.astype(float, copy=True)
        if np.any(~np.isfinite(result)):
            raise ValueError("{} must be finite".format(name))
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RecordedControlRollout:
    """Finite-prefix result of one recorded-control open-loop replay."""

    time: np.ndarray
    cog_position: np.ndarray
    body_orientation_xyzw: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    actuator_thrust: np.ndarray
    actuator_gimbal: np.ndarray
    sensor_position: np.ndarray
    sensor_orientation_xyzw: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        count = np.asarray(self.time).size
        for name, shape in (
            ("time", (count,)),
            ("cog_position", (count, 3)),
            ("body_orientation_xyzw", (count, 4)),
            ("linear_velocity", (count, 3)),
            ("angular_velocity", (count, 3)),
            ("actuator_thrust", (count, 4)),
            ("actuator_gimbal", (count, 4)),
            ("sensor_position", (count, 3)),
            ("sensor_orientation_xyzw", (count, 4)),
            ("valid", (count,)),
        ):
            object.__setattr__(
                self, name, _immutable(getattr(self, name), shape, name)
            )
        if count < 2 or np.any(np.diff(self.time) <= 0.0):
            raise ValueError("rollout time must be strictly increasing")
        valid = np.asarray(self.valid, dtype=bool)
        if not valid[0] or np.any(np.diff(valid.astype(np.int8)) > 0):
            raise ValueError("rollout validity must be one finite prefix")
        for name in ("body_orientation_xyzw", "sensor_orientation_xyzw"):
            norms = np.linalg.norm(getattr(self, name)[valid], axis=1)
            if np.any(np.abs(norms - 1.0) > 1.0e-9):
                raise ValueError("{} must contain unit quaternions".format(name))


def _initial_rigid_body(state: BatchState, bag_id: str) -> RigidBodyState:
    return RigidBodyState(
        position=state.knot_value(bag_id, 0, VariableKind.POSITION),
        orientation_xyzw=matrix_to_quaternion(
            state.knot_value(
                bag_id, 0, VariableKind.ORIENTATION_TANGENT
            )
        ),
        linear_velocity=state.knot_value(
            bag_id, 0, VariableKind.LINEAR_VELOCITY
        ),
        angular_velocity=state.knot_value(
            bag_id, 0, VariableKind.ANGULAR_VELOCITY
        ),
    )


def _initial_actuators(state: BatchState, bag_id: str) -> ActuatorState:
    return ActuatorState(
        thrust=state.knot_value(
            bag_id, 0, VariableKind.ACTUATOR_THRUST
        ),
        gimbal_angle=state.knot_value(
            bag_id, 0, VariableKind.GIMBAL_ANGLE
        ),
    )


def simulate_recorded_control_rollout(
    *,
    prepared_bag: PreparedBagGraphData,
    initial_state: BatchState,
    parameter_chart: VehicleParameterChart,
    parameter_coordinates: Sequence[float],
    geometry: GrapeGeometry,
    initial_state_parameter_coordinates: Optional[Sequence[float]] = None,
    actuator_intervals: Optional[
        Sequence[PreparedActuatorInterval]
    ] = None,
) -> RecordedControlRollout:
    """Replay prepared delayed commands without controller or state resets.

    When ``initial_state_parameter_coordinates`` is supplied, the candidate
    CoG state is rebased so that its initial sensor pose and twist equal the
    pose and twist represented by ``initial_state`` under those coordinates.
    If an unstable candidate ceases to produce finite states, the returned
    arrays retain their last finite sample and ``valid`` is false from the
    failed interval onward.  Artifact publication therefore reports a
    divergent forecast instead of failing after estimation has completed.
    """

    if not isinstance(prepared_bag, PreparedBagGraphData):
        raise TypeError("prepared_bag must be PreparedBagGraphData")
    if not isinstance(initial_state, BatchState):
        raise TypeError("initial_state must be BatchState")
    if not isinstance(parameter_chart, VehicleParameterChart):
        raise TypeError("parameter_chart must be VehicleParameterChart")
    if not isinstance(geometry, GrapeGeometry):
        raise TypeError("geometry must be GrapeGeometry")
    coordinates = np.asarray(parameter_coordinates, dtype=float)
    parameters = parameter_chart.decode(coordinates)
    intervals = (
        prepared_bag.actuator_intervals
        if actuator_intervals is None
        else tuple(actuator_intervals)
    )
    if (
        len(intervals) != len(prepared_bag.knots) - 1
        or any(
            not isinstance(value, PreparedActuatorInterval)
            or value.left_knot_index != index
            for index, value in enumerate(intervals)
        )
    ):
        raise ValueError(
            "actuator_intervals must cover the prepared knot grid"
        )
    times = np.asarray(
        tuple(knot.time for knot in prepared_bag.knots), dtype=float
    )
    count = times.size
    position = np.zeros((count, 3), dtype=float)
    orientation = np.zeros((count, 4), dtype=float)
    orientation[:, 3] = 1.0
    velocity = np.zeros((count, 3), dtype=float)
    omega = np.zeros((count, 3), dtype=float)
    thrust = np.zeros((count, 4), dtype=float)
    gimbal = np.zeros((count, 4), dtype=float)
    sensor_position = np.zeros((count, 3), dtype=float)
    sensor_orientation = np.zeros((count, 4), dtype=float)
    sensor_orientation[:, 3] = 1.0
    valid = np.zeros(count, dtype=bool)

    rigid = _initial_rigid_body(initial_state, prepared_bag.bag_id)
    if initial_state_parameter_coordinates is not None:
        anchor_parameters = parameter_chart.decode(
            initial_state_parameter_coordinates
        )
        body_rotation = quaternion_to_matrix(rigid.orientation_xyzw)
        angular_velocity = rigid.angular_velocity
        anchor_cog_to_sensor = (
            prepared_bag.sensor_extrinsics.pose_sensor_position_in_body
            - anchor_parameters.cog_offset
        )
        candidate_cog_to_sensor = (
            prepared_bag.sensor_extrinsics.pose_sensor_position_in_body
            - parameters.cog_offset
        )
        sensor_position_at_start = (
            rigid.position + body_rotation @ anchor_cog_to_sensor
        )
        sensor_velocity_at_start = (
            rigid.linear_velocity
            + body_rotation
            @ np.cross(angular_velocity, anchor_cog_to_sensor)
        )
        rigid = RigidBodyState(
            position=(
                sensor_position_at_start
                - body_rotation @ candidate_cog_to_sensor
            ),
            orientation_xyzw=rigid.orientation_xyzw,
            linear_velocity=(
                sensor_velocity_at_start
                - body_rotation
                @ np.cross(angular_velocity, candidate_cog_to_sensor)
            ),
            angular_velocity=angular_velocity,
        )
    actuators = _initial_actuators(initial_state, prepared_bag.bag_id)
    plant = FullSixDofPlant(parameters, geometry)
    cog_to_sensor = (
        prepared_bag.sensor_extrinsics.pose_sensor_position_in_body
        - parameters.cog_offset
    )
    body_to_sensor = (
        prepared_bag.sensor_extrinsics.pose_sensor_to_body_rotation
    )

    def store(index: int) -> None:
        body_rotation = quaternion_to_matrix(rigid.orientation_xyzw)
        position[index] = rigid.position
        orientation[index] = rigid.orientation_xyzw
        velocity[index] = rigid.linear_velocity
        omega[index] = rigid.angular_velocity
        thrust[index] = actuators.thrust
        gimbal[index] = actuators.gimbal_angle
        sensor_position[index] = (
            rigid.position + body_rotation @ cog_to_sensor
        )
        sensor_orientation[index] = matrix_to_quaternion(
            body_rotation @ body_to_sensor
        )
        valid[index] = True

    store(0)
    for interval_index, interval in enumerate(intervals):
        current_time = float(times[interval_index])
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                for segment in interval.delayed_command_segments:
                    half_step = 0.5 * segment.duration
                    midpoint_actuators = advance_actuators(
                        actuators,
                        segment.command,
                        prepared_bag.actuator_parameters,
                        half_step,
                    )
                    rigid = plant.step(
                        current_time,
                        rigid,
                        midpoint_actuators,
                        segment.duration,
                    )
                    actuators = advance_actuators(
                        midpoint_actuators,
                        segment.command,
                        prepared_bag.actuator_parameters,
                        half_step,
                    )
                    current_time += segment.duration
                store(interval_index + 1)
        except (
            ValueError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
        ):
            break

    last = int(np.flatnonzero(valid)[-1])
    for array in (
        position,
        orientation,
        velocity,
        omega,
        thrust,
        gimbal,
        sensor_position,
        sensor_orientation,
    ):
        array[last + 1 :] = array[last]
    return RecordedControlRollout(
        time=times,
        cog_position=position,
        body_orientation_xyzw=orientation,
        linear_velocity=velocity,
        angular_velocity=omega,
        actuator_thrust=thrust,
        actuator_gimbal=gimbal,
        sensor_position=sensor_position,
        sensor_orientation_xyzw=sensor_orientation,
        valid=valid,
    )


def interpolate_observed_pose(
    observation_time: Sequence[float],
    observation_position: np.ndarray,
    observation_orientation_xyzw: np.ndarray,
    query_time: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a sensor pose onto query times without extrapolation."""

    source_time = np.asarray(observation_time, dtype=float)
    source_position = np.asarray(observation_position, dtype=float)
    source_orientation = np.asarray(
        observation_orientation_xyzw, dtype=float
    )
    query = np.asarray(query_time, dtype=float)
    if (
        source_time.ndim != 1
        or source_time.size < 2
        or np.any(np.diff(source_time) <= 0.0)
        or source_position.shape != (source_time.size, 3)
        or source_orientation.shape != (source_time.size, 4)
        or query.ndim != 1
        or np.any(~np.isfinite(source_position))
        or np.any(~np.isfinite(source_orientation))
        or np.any(~np.isfinite(query))
    ):
        raise ValueError("observed pose interpolation inputs are invalid")
    position = np.zeros((query.size, 3), dtype=float)
    orientation = np.zeros((query.size, 4), dtype=float)
    orientation[:, 3] = 1.0
    tolerance = 2.0e-10 * max(
        1.0, abs(float(source_time[0])), abs(float(source_time[-1]))
    )
    valid = (query >= source_time[0] - tolerance) & (
        query <= source_time[-1] + tolerance
    )
    rotations = tuple(
        quaternion_to_matrix(value) for value in source_orientation
    )
    for output_index in np.flatnonzero(valid):
        selected_time = float(
            np.clip(query[output_index], source_time[0], source_time[-1])
        )
        right = int(np.searchsorted(source_time, selected_time, side="right"))
        if right == 0:
            left = right = 0
            alpha = 0.0
        elif right >= source_time.size:
            left = right = source_time.size - 1
            alpha = 0.0
        else:
            left = right - 1
            alpha = (selected_time - source_time[left]) / (
                source_time[right] - source_time[left]
            )
        if left == right:
            position[output_index] = source_position[left]
            rotation = rotations[left]
        else:
            position[output_index] = (
                (1.0 - alpha) * source_position[left]
                + alpha * source_position[right]
            )
            rotation = rotations[left] @ so3_exp(
                alpha * so3_log(rotations[left].T @ rotations[right])
            )
        orientation[output_index] = matrix_to_quaternion(rotation)
    return position, orientation, valid


__all__ = [
    "RecordedControlRollout",
    "interpolate_observed_pose",
    "simulate_recorded_control_rollout",
]
