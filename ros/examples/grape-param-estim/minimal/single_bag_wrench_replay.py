#!/usr/bin/env python3
"""Parameter-frozen trajectory-fitted external-wrench reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from grape_param_estim.dynamics import FullSixDofPlant, actuator_wrench  # noqa: E402
from grape_param_estim.geometry import matrix_to_quaternion, quaternion_to_matrix  # noqa: E402
from grape_param_estim.system import ActuatorState, RigidBodyState  # noqa: E402

try:  # noqa: E402
    from .single_bag_savgol_core import (
        GRAVITY_WORLD,
        DynamicsEvaluation,
        SingleBagDataset,
        VehicleModelInput,
    )
except ImportError:  # pragma: no cover
    from single_bag_savgol_core import (  # type: ignore
        GRAVITY_WORLD,
        DynamicsEvaluation,
        SingleBagDataset,
        VehicleModelInput,
    )


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class WrenchReplayResult:
    """Standardized evaluation-only replay products."""

    time: np.ndarray
    fitted_external_wrench: np.ndarray
    reconstructed_cog_position: np.ndarray
    reconstructed_body_orientation_xyzw: np.ndarray
    reconstructed_cog_velocity: np.ndarray
    reconstructed_body_angular_velocity: np.ndarray
    reconstructed_sensor_position: np.ndarray
    reconstructed_sensor_orientation_xyzw: np.ndarray
    free_cog_position: np.ndarray
    free_body_orientation_xyzw: np.ndarray
    free_cog_velocity: np.ndarray
    free_body_angular_velocity: np.ndarray
    free_sensor_position: np.ndarray
    free_sensor_orientation_xyzw: np.ndarray
    predicted_gyro: np.ndarray
    predicted_specific_force: np.ndarray

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        count = time.size
        shapes = {
            "fitted_external_wrench": (count, 6),
            "reconstructed_cog_position": (count, 3),
            "reconstructed_body_orientation_xyzw": (count, 4),
            "reconstructed_cog_velocity": (count, 3),
            "reconstructed_body_angular_velocity": (count, 3),
            "reconstructed_sensor_position": (count, 3),
            "reconstructed_sensor_orientation_xyzw": (count, 4),
            "free_cog_position": (count, 3),
            "free_body_orientation_xyzw": (count, 4),
            "free_cog_velocity": (count, 3),
            "free_body_angular_velocity": (count, 3),
            "free_sensor_position": (count, 3),
            "free_sensor_orientation_xyzw": (count, 4),
            "predicted_gyro": (count, 3),
            "predicted_specific_force": (count, 3),
        }
        if time.ndim != 1 or count < 2 or np.any(~np.isfinite(time)):
            raise ValueError("replay time is invalid")
        object.__setattr__(self, "time", _readonly(time))
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or np.any(~np.isfinite(value)):
                raise ValueError("{} is invalid".format(name))
            object.__setattr__(self, name, _readonly(value))


def _observed_cog_trajectory(
    dataset: SingleBagDataset, evaluation: DynamicsEvaluation
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sg = dataset.reference_sg
    lever = (
        np.asarray(dataset.pose_sensor_position_in_body)
        - np.asarray(evaluation.parameters.cog_offset)
    )
    position = sg.sensor_position - np.einsum(
        "nij,j->ni", sg.body_rotation, lever
    )
    velocity = sg.sensor_velocity_world - np.einsum(
        "nij,nj->ni",
        sg.body_rotation,
        np.cross(sg.body_angular_velocity, lever),
    )
    # scipy 1.5's Cython memoryview rejects read-only buffers.
    orientation = Rotation.from_matrix(
        np.asarray(sg.body_rotation, dtype=float).copy()
    ).as_quat()
    return position, orientation, velocity, sg.body_angular_velocity


def fit_external_wrench_replay(
    *,
    dataset: SingleBagDataset,
    model: VehicleModelInput,
    evaluation: DynamicsEvaluation,
) -> WrenchReplayResult:
    """Fit a per-interval wrench to discrete trajectory increments.

    Each six-dimensional interval wrench is the unregularized linear
    least-squares solution that jointly matches the next position/velocity and
    orientation/angular-velocity increment under a local constant-acceleration
    model.  Canonical RK4 rigid-body propagation then produces the reported
    reconstruction.  Parameters remain frozen throughout this function.
    """

    times = np.asarray(dataset.time, dtype=float)
    count = times.size
    target_p, target_q, target_v, target_omega = _observed_cog_trajectory(
        dataset, evaluation
    )
    parameters = evaluation.parameters
    plant = FullSixDofPlant(parameters, model.geometry)
    reconstructed_p = np.empty((count, 3))
    reconstructed_q = np.empty((count, 4))
    reconstructed_v = np.empty((count, 3))
    reconstructed_omega = np.empty((count, 3))
    sensor_p = np.empty((count, 3))
    sensor_q = np.empty((count, 4))
    gyro = np.empty((count, 3))
    specific = np.empty((count, 3))
    fitted = np.empty((count, 6))
    state = RigidBodyState(target_p[0], target_q[0], target_v[0], target_omega[0])
    pose_lever = (
        np.asarray(dataset.pose_sensor_position_in_body)
        - np.asarray(parameters.cog_offset)
    )
    imu_lever = (
        np.asarray(dataset.gyro_sensor_position_in_body)
        - np.asarray(parameters.cog_offset)
    )

    def actuator(index: int) -> ActuatorState:
        return ActuatorState(
            evaluation.actuator_history.actual_thrust[index],
            evaluation.actuator_history.actual_gimbal[index],
        )

    def store(index: int, external: np.ndarray) -> None:
        rotation = quaternion_to_matrix(state.orientation_xyzw)
        robot = actuator_wrench(actuator(index), parameters, model.geometry)
        total = robot + external
        alpha = np.linalg.solve(
            parameters.inertia,
            total[3:]
            - np.cross(
                state.angular_velocity,
                parameters.inertia @ state.angular_velocity,
            ),
        )
        specific_body = (
            total[:3] / parameters.mass
            + np.cross(alpha, imu_lever)
            + np.cross(
                state.angular_velocity,
                np.cross(state.angular_velocity, imu_lever),
            )
        )
        reconstructed_p[index] = state.position
        reconstructed_q[index] = state.orientation_xyzw
        reconstructed_v[index] = state.linear_velocity
        reconstructed_omega[index] = state.angular_velocity
        sensor_p[index] = state.position + rotation @ pose_lever
        sensor_q[index] = matrix_to_quaternion(
            rotation @ dataset.pose_sensor_to_body_rotation
        )
        gyro[index] = (
            dataset.body_to_gyro_sensor_rotation @ state.angular_velocity
            + dataset.gyro_bias
        )
        specific[index] = (
            dataset.body_to_gyro_sensor_rotation @ specific_body
            + dataset.accelerometer_bias
        )
        fitted[index] = external

    zero = np.zeros(6)
    for index in range(count - 1):
        dt = float(times[index + 1] - times[index])
        rotation = quaternion_to_matrix(state.orientation_xyzw)
        robot = actuator_wrench(actuator(index), parameters, model.geometry)
        base_linear_acceleration = GRAVITY_WORLD + rotation @ (
            robot[:3] / parameters.mass
        )
        force_response = rotation / parameters.mass
        linear_matrix = np.vstack(
            (0.5 * dt**2 * force_response, dt * force_response)
        )
        linear_target = np.concatenate(
            (
                target_p[index + 1]
                - (
                    state.position
                    + dt * state.linear_velocity
                    + 0.5 * dt**2 * base_linear_acceleration
                ),
                target_v[index + 1]
                - (state.linear_velocity + dt * base_linear_acceleration),
            )
        )
        external_force, *_ = np.linalg.lstsq(
            linear_matrix, linear_target, rcond=None
        )

        base_alpha = np.linalg.solve(
            parameters.inertia,
            robot[3:]
            - np.cross(
                state.angular_velocity,
                parameters.inertia @ state.angular_velocity,
            ),
        )
        target_rotation = quaternion_to_matrix(target_q[index + 1])
        local_rotation_increment = Rotation.from_matrix(
            rotation.T @ target_rotation
        ).as_rotvec()
        torque_response = np.linalg.solve(parameters.inertia, np.eye(3))
        angular_matrix = np.vstack(
            (0.5 * dt**2 * torque_response, dt * torque_response)
        )
        angular_target = np.concatenate(
            (
                local_rotation_increment
                - (dt * state.angular_velocity + 0.5 * dt**2 * base_alpha),
                target_omega[index + 1]
                - (state.angular_velocity + dt * base_alpha),
            )
        )
        external_torque, *_ = np.linalg.lstsq(
            angular_matrix, angular_target, rcond=None
        )
        external = np.concatenate((external_force, external_torque))
        store(index, external)
        state = plant.step(
            float(times[index]),
            state,
            actuator(index),
            dt,
            interval_model_discrepancy_wrench=external,
        )
    store(count - 1, fitted[count - 2] if count > 1 else zero)

    # Free rollout is retained only as a raw diagnostic array.  It is not
    # added to the standardized two-curve trajectory report.
    free_p = np.empty((count, 3))
    free_q = np.empty((count, 4))
    free_v = np.empty((count, 3))
    free_omega = np.empty((count, 3))
    free_sensor_p = np.empty((count, 3))
    free_sensor_q = np.empty((count, 4))
    free_state = RigidBodyState(
        target_p[0], target_q[0], target_v[0], target_omega[0]
    )
    for index in range(count):
        free_rotation = quaternion_to_matrix(free_state.orientation_xyzw)
        free_p[index] = free_state.position
        free_q[index] = free_state.orientation_xyzw
        free_v[index] = free_state.linear_velocity
        free_omega[index] = free_state.angular_velocity
        free_sensor_p[index] = free_state.position + free_rotation @ pose_lever
        free_sensor_q[index] = matrix_to_quaternion(
            free_rotation @ dataset.pose_sensor_to_body_rotation
        )
        if index + 1 < count:
            free_state = plant.step(
                float(times[index]),
                free_state,
                actuator(index),
                float(times[index + 1] - times[index]),
            )
    return WrenchReplayResult(
        time=times,
        fitted_external_wrench=fitted,
        reconstructed_cog_position=reconstructed_p,
        reconstructed_body_orientation_xyzw=reconstructed_q,
        reconstructed_cog_velocity=reconstructed_v,
        reconstructed_body_angular_velocity=reconstructed_omega,
        reconstructed_sensor_position=sensor_p,
        reconstructed_sensor_orientation_xyzw=sensor_q,
        free_cog_position=free_p,
        free_body_orientation_xyzw=free_q,
        free_cog_velocity=free_v,
        free_body_angular_velocity=free_omega,
        free_sensor_position=free_sensor_p,
        free_sensor_orientation_xyzw=free_sensor_q,
        predicted_gyro=gyro,
        predicted_specific_force=specific,
    )
