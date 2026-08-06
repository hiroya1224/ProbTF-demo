#!/usr/bin/env python3
"""Minimal direct-shooting identification from one recorded Grape flight.

This script deliberately does not use the sparse batch smoother, latent
states, Q, residual wrenches, EM, MCMC, or the GUI.  It reads the recorded
rotor/gimbal commands once, starts from the observed state once, advances the
same actuator and six-DoF rigid-body models open loop, and asks SciPy to make
that single rollout match recorded pose, velocity, gyro, and specific force.

The optimized thirteen-dimensional vector contains log mass, six symmetric
relative-log-inertia coordinates, the three-dimensional CoG offset, and
three relative rotor-force-effectiveness contrasts.  The four log
effectiveness values sum to zero, so their geometric mean remains exactly one
and the common effectiveness/mass scale ambiguity is not introduced.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/grape-minimal-matplotlib")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

from grape_param_estim.batch.recorded_control_rollout import (  # noqa: E402
    interpolate_observed_pose,
)
from grape_param_estim.dynamics import (  # noqa: E402
    FullSixDofPlant,
    advance_actuators,
)
from grape_param_estim.geometry import (  # noqa: E402
    matrix_to_euler_xyz,
    matrix_to_quaternion,
    quaternion_to_matrix,
    so3_log,
)
from grape_param_estim.parameterization import (  # noqa: E402
    PARAMETER_DIMENSION,
    VehicleParameterChart,
)
from grape_param_estim.real_rosbag import load_flight_data  # noqa: E402
from grape_param_estim.system import (  # noqa: E402
    ActuatorCommand,
    ActuatorParameters,
    ActuatorState,
    GrapeGeometry,
    RigidBodyState,
    VehicleParameters,
)


DEFAULT_BAG = (
    Path(__file__).resolve().parents[1]
    / "samples"
    / "rosbags"
    / "20260612_grape_hovering_4_2026-06-12-17-33-59.bag"
)
ACTIVE_PARAMETER_NAMES = (
    "log_mass_scale",
    "relative_log_inertia_xx",
    "relative_log_inertia_yy",
    "relative_log_inertia_zz",
    "relative_log_inertia_xy",
    "relative_log_inertia_xz",
    "relative_log_inertia_yz",
    "cog_offset_x_m",
    "cog_offset_y_m",
    "cog_offset_z_m",
    "force_effectiveness_contrast_1",
    "force_effectiveness_contrast_2",
    "force_effectiveness_contrast_3",
)
ACTIVE_PARAMETER_DIMENSION = len(ACTIVE_PARAMETER_NAMES)
FORCE_EFFECTIVENESS_CONTRAST_BASIS = np.asarray(
    (
        (
            1.0 / math.sqrt(2.0),
            1.0 / math.sqrt(6.0),
            1.0 / math.sqrt(12.0),
        ),
        (
            -1.0 / math.sqrt(2.0),
            1.0 / math.sqrt(6.0),
            1.0 / math.sqrt(12.0),
        ),
        (0.0, -2.0 / math.sqrt(6.0), 1.0 / math.sqrt(12.0)),
        (0.0, 0.0, -3.0 / math.sqrt(12.0)),
    ),
    dtype=float,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _linear_interpolate(
    source_time: np.ndarray,
    source_value: np.ndarray,
    query_time: np.ndarray,
) -> np.ndarray:
    source_time = np.asarray(source_time, dtype=float)
    source_value = np.asarray(source_value, dtype=float)
    query_time = np.asarray(query_time, dtype=float)
    if (
        source_time.ndim != 1
        or source_time.size < 2
        or np.any(np.diff(source_time) <= 0.0)
        or source_value.ndim != 2
        or source_value.shape[0] != source_time.size
        or query_time.ndim != 1
        or query_time[0] < source_time[0]
        or query_time[-1] > source_time[-1]
    ):
        raise ValueError("linear interpolation inputs are invalid")
    return np.column_stack(
        [
            np.interp(query_time, source_time, source_value[:, column])
            for column in range(source_value.shape[1])
        ]
    )


def _zoh_indices(source_time: np.ndarray, query_time: np.ndarray) -> np.ndarray:
    source_time = np.asarray(source_time, dtype=float)
    query_time = np.asarray(query_time, dtype=float)
    indices = np.searchsorted(source_time, query_time, side="right") - 1
    if np.any(indices < 0) or np.any(indices >= source_time.size):
        raise ValueError("recorded command does not cover delayed query time")
    return indices


def _rotation_series(quaternions: np.ndarray) -> np.ndarray:
    return np.asarray(
        [quaternion_to_matrix(value) for value in quaternions], dtype=float
    )


def _rpy_series(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(
        [
            matrix_to_euler_xyz(quaternion_to_matrix(value))
            for value in quaternions
        ],
        dtype=float,
    )
    return np.unwrap(values, axis=0)


def _vector_rmse(error: np.ndarray) -> float:
    value = np.asarray(error, dtype=float)
    return float(np.sqrt(np.mean(np.sum(value * value, axis=1))))


def _component_rmse(error: np.ndarray) -> list[float]:
    value = np.asarray(error, dtype=float)
    return np.sqrt(np.mean(value * value, axis=0)).tolist()


@dataclass(frozen=True)
class Observations:
    time: np.ndarray
    sensor_position: np.ndarray
    sensor_orientation_xyzw: np.ndarray
    sensor_velocity_world: np.ndarray
    angular_velocity_sensor: np.ndarray
    specific_force_sensor: np.ndarray


@dataclass(frozen=True)
class Simulation:
    time: np.ndarray
    sensor_position: np.ndarray
    sensor_orientation_xyzw: np.ndarray
    sensor_velocity_world: np.ndarray
    angular_velocity_sensor: np.ndarray
    specific_force_sensor: np.ndarray
    cog_position: np.ndarray
    cog_velocity_world: np.ndarray
    actuator_thrust: np.ndarray
    actuator_gimbal: np.ndarray


class DirectShootingProblem:
    """One fixed data set and one open-loop simulation per parameter vector."""

    def __init__(
        self,
        *,
        flight: Any,
        sample_step: float,
        integration_step: float,
        command_delay: float,
        prior_weight: float,
    ) -> None:
        ratio = int(round(sample_step / integration_step))
        if ratio < 1 or not np.isclose(
            ratio * integration_step,
            sample_step,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "sample-step must be an integer multiple of integration-step"
            )
        support_start = max(
            float(flight.pose.times[0]),
            float(flight.velocity.times[0]),
            float(flight.gyro.times[0]),
            float(flight.accelerometer.times[0]),
            float(flight.gimbal_position.times[0]),
        )
        support_end = min(
            float(flight.pose.times[-1]),
            float(flight.velocity.times[-1]),
            float(flight.gyro.times[-1]),
            float(flight.accelerometer.times[-1]),
        )
        sample_count = int(math.floor((support_end - support_start) / sample_step)) + 1
        if sample_count < 3:
            raise ValueError("common observation support is too short")
        output_time = support_start + sample_step * np.arange(sample_count)
        internal_count = (sample_count - 1) * ratio + 1
        internal_time = support_start + integration_step * np.arange(
            internal_count
        )

        pose_position, pose_orientation, pose_valid = (
            interpolate_observed_pose(
                flight.pose.times,
                flight.pose.positions,
                flight.pose.orientations_xyzw,
                output_time,
            )
        )
        if not np.all(pose_valid):
            raise ValueError("pose does not cover the common output grid")
        self.observations = Observations(
            time=output_time,
            sensor_position=pose_position,
            sensor_orientation_xyzw=pose_orientation,
            sensor_velocity_world=_linear_interpolate(
                flight.velocity.times,
                flight.velocity.values,
                output_time,
            ),
            angular_velocity_sensor=_linear_interpolate(
                flight.gyro.times,
                flight.gyro.values,
                output_time,
            ),
            specific_force_sensor=_linear_interpolate(
                flight.accelerometer.times,
                flight.accelerometer.values,
                output_time,
            ),
        )
        self.internal_time = internal_time
        self.output_stride = ratio
        self.integration_step = float(integration_step)
        self.command_delay = float(command_delay)
        self.prior_weight = float(prior_weight)
        self.actuator_parameters = ActuatorParameters(
            thrust_time_constant=0.01,
            gimbal_time_constant=0.02,
            delay=0.0,
        )
        self.chart = VehicleParameterChart(VehicleParameters.nominal())
        self.geometry = GrapeGeometry.grape()
        self.pose_sensor_position = np.asarray(
            flight.sensor_extrinsics.pose_sensor_position_in_body,
            dtype=float,
        )
        self.pose_body_to_sensor_rotation = np.asarray(
            flight.sensor_extrinsics.pose_sensor_to_body_rotation,
            dtype=float,
        )
        self.velocity_sensor_position = np.asarray(
            flight.sensor_extrinsics.velocity_sensor_position_in_body,
            dtype=float,
        )
        self.imu_sensor_position = np.asarray(
            flight.sensor_extrinsics.gyro_sensor_position_in_body,
            dtype=float,
        )
        self.body_to_imu_rotation = np.asarray(
            flight.sensor_extrinsics.body_to_gyro_sensor_rotation,
            dtype=float,
        )
        self.gyro_bias = np.asarray(
            flight.imu_preflight.gyro_bias, dtype=float
        )
        if flight.imu_preflight.accelerometer_bias is None:
            raise ValueError("accelerometer preflight bias is unavailable")
        self.accelerometer_bias = np.asarray(
            flight.imu_preflight.accelerometer_bias, dtype=float
        )

        command_query = (
            0.5 * (internal_time[:-1] + internal_time[1:])
            - command_delay
        )
        rotor_indices = _zoh_indices(
            flight.rotor_command.all_times, command_query
        )
        gimbal_indices = _zoh_indices(
            flight.gimbal_command.all_times, command_query
        )
        rotor_values = flight.rotor_command.all_values[rotor_indices]
        gimbal_values = flight.gimbal_command.all_values[gimbal_indices]
        self.commands = tuple(
            ActuatorCommand(
                thrust=rotor_values[index],
                gimbal_angle=gimbal_values[index],
                virtual_force=np.zeros(8),
                desired_acceleration=np.zeros(6),
            )
            for index in range(command_query.size)
        )
        initial_query = np.asarray((internal_time[0] - command_delay,))
        initial_rotor = flight.rotor_command.all_values[
            _zoh_indices(flight.rotor_command.all_times, initial_query)[0]
        ]
        initial_gimbal = _linear_interpolate(
            flight.gimbal_position.times,
            flight.gimbal_position.values,
            np.asarray((internal_time[0],)),
        )[0]
        self.initial_actuator_state = ActuatorState(
            thrust=initial_rotor,
            gimbal_angle=initial_gimbal,
        )
        self.initial_sensor_rotation = quaternion_to_matrix(
            self.observations.sensor_orientation_xyzw[0]
        )
        self.initial_body_rotation = (
            self.initial_sensor_rotation
            @ self.pose_body_to_sensor_rotation.T
        )
        initial_omega_sensor = (
            self.observations.angular_velocity_sensor[0] - self.gyro_bias
        )
        self.initial_omega_body = (
            self.body_to_imu_rotation.T @ initial_omega_sensor
        )
        self.initial_sensor_velocity = (
            self.observations.sensor_velocity_world[0]
        )
        self.observed_sensor_rotation = _rotation_series(
            self.observations.sensor_orientation_xyzw
        )
        self.observed_body_rotation = np.einsum(
            "nij,jk->nik",
            self.observed_sensor_rotation,
            self.pose_body_to_sensor_rotation.T,
        )
        observed_omega_sensor = (
            self.observations.angular_velocity_sensor - self.gyro_bias
        )
        self.observed_omega_body = (
            observed_omega_sensor @ self.body_to_imu_rotation
        )
        smoothing_window = min(11, self.output_time.size)
        if smoothing_window % 2 == 0:
            smoothing_window -= 1
        if smoothing_window >= 5:
            polynomial_order = min(3, smoothing_window - 2)
            self.smoothed_omega_body = savgol_filter(
                self.observed_omega_body,
                smoothing_window,
                polynomial_order,
                axis=0,
            )
            self.observed_angular_acceleration_body = savgol_filter(
                self.observed_omega_body,
                smoothing_window,
                polynomial_order,
                deriv=1,
                delta=sample_step,
                axis=0,
            )
        else:
            self.smoothed_omega_body = self.observed_omega_body.copy()
            self.observed_angular_acceleration_body = np.gradient(
                self.observed_omega_body,
                sample_step,
                axis=0,
                edge_order=1,
            )
        self._evaluation_count = 0
        self._best_cost = float("inf")
        self._last_progress_time = 0.0

        self.residual_scales = {
            "position_m": 0.05,
            "orientation_rad": 0.10,
            "velocity_m_per_s": 0.20,
            "angular_velocity_rad_per_s": 0.20,
            "specific_force_m_per_s2": 0.75,
        }
        self.prior_scales = np.asarray(
            (
                0.50,
                0.80,
                0.80,
                0.80,
                0.40,
                0.40,
                0.40,
                0.05,
                0.05,
                0.05,
                0.25,
                0.25,
                0.25,
            ),
            dtype=float,
        )

    @property
    def output_time(self) -> np.ndarray:
        return self.observations.time

    def full_coordinates(self, active: Sequence[float]) -> np.ndarray:
        active_array = np.asarray(active, dtype=float)
        if active_array.shape != (ACTIVE_PARAMETER_DIMENSION,):
            raise ValueError("active parameter vector has the wrong shape")
        result = np.zeros(PARAMETER_DIMENSION, dtype=float)
        result[:10] = active_array[:10]
        result[10:14] = (
            FORCE_EFFECTIVENESS_CONTRAST_BASIS @ active_array[10:13]
        )
        return result

    def _initial_rigid_state(self, parameters: VehicleParameters) -> RigidBodyState:
        pose_lever = self.pose_sensor_position - parameters.cog_offset
        velocity_lever = (
            self.velocity_sensor_position - parameters.cog_offset
        )
        cog_position = (
            self.observations.sensor_position[0]
            - self.initial_body_rotation @ pose_lever
        )
        cog_velocity = (
            self.initial_sensor_velocity
            - self.initial_body_rotation
            @ np.cross(self.initial_omega_body, velocity_lever)
        )
        return RigidBodyState(
            position=cog_position,
            orientation_xyzw=matrix_to_quaternion(
                self.initial_body_rotation
            ),
            linear_velocity=cog_velocity,
            angular_velocity=self.initial_omega_body,
        )

    def simulate(self, active: Sequence[float]) -> Simulation:
        return self.simulate_full_coordinates(self.full_coordinates(active))

    def simulate_full_coordinates(
        self, full_coordinates: Sequence[float]
    ) -> Simulation:
        coordinates = np.asarray(full_coordinates, dtype=float)
        if (
            coordinates.shape != (PARAMETER_DIMENSION,)
            or not np.all(np.isfinite(coordinates))
        ):
            raise ValueError("full parameter coordinates must be finite 18-D")
        parameters = self.chart.decode(coordinates)
        plant = FullSixDofPlant(parameters, self.geometry)
        rigid = self._initial_rigid_state(parameters)
        actuators = self.initial_actuator_state
        output_count = self.output_time.size
        arrays = {
            "sensor_position": np.empty((output_count, 3)),
            "sensor_orientation_xyzw": np.empty((output_count, 4)),
            "sensor_velocity_world": np.empty((output_count, 3)),
            "angular_velocity_sensor": np.empty((output_count, 3)),
            "specific_force_sensor": np.empty((output_count, 3)),
            "cog_position": np.empty((output_count, 3)),
            "cog_velocity_world": np.empty((output_count, 3)),
            "actuator_thrust": np.empty((output_count, 4)),
            "actuator_gimbal": np.empty((output_count, 4)),
        }

        def store(output_index: int, simulation_time: float) -> None:
            rotation = quaternion_to_matrix(rigid.orientation_xyzw)
            pose_lever = self.pose_sensor_position - parameters.cog_offset
            velocity_lever = (
                self.velocity_sensor_position - parameters.cog_offset
            )
            imu_lever = self.imu_sensor_position - parameters.cog_offset
            wrench = plant.total_body_wrench(
                simulation_time, rigid, actuators
            )
            angular_acceleration = np.linalg.solve(
                parameters.inertia,
                wrench[3:]
                - np.cross(
                    rigid.angular_velocity,
                    parameters.inertia @ rigid.angular_velocity,
                ),
            )
            specific_force_body = (
                wrench[:3] / parameters.mass
                + np.cross(angular_acceleration, imu_lever)
                + np.cross(
                    rigid.angular_velocity,
                    np.cross(rigid.angular_velocity, imu_lever),
                )
            )
            arrays["sensor_position"][output_index] = (
                rigid.position + rotation @ pose_lever
            )
            arrays["sensor_orientation_xyzw"][output_index] = (
                matrix_to_quaternion(
                    rotation @ self.pose_body_to_sensor_rotation
                )
            )
            arrays["sensor_velocity_world"][output_index] = (
                rigid.linear_velocity
                + rotation
                @ np.cross(rigid.angular_velocity, velocity_lever)
            )
            arrays["angular_velocity_sensor"][output_index] = (
                self.body_to_imu_rotation @ rigid.angular_velocity
                + self.gyro_bias
            )
            arrays["specific_force_sensor"][output_index] = (
                self.body_to_imu_rotation @ specific_force_body
                + self.accelerometer_bias
            )
            arrays["cog_position"][output_index] = rigid.position
            arrays["cog_velocity_world"][output_index] = (
                rigid.linear_velocity
            )
            arrays["actuator_thrust"][output_index] = actuators.thrust
            arrays["actuator_gimbal"][output_index] = (
                actuators.gimbal_angle
            )

        store(0, float(self.internal_time[0]))
        output_index = 1
        for step_index, command in enumerate(self.commands):
            start = float(self.internal_time[step_index])
            dt = self.integration_step
            midpoint_actuators = advance_actuators(
                actuators,
                command,
                self.actuator_parameters,
                0.5 * dt,
            )
            rigid = plant.step(start, rigid, midpoint_actuators, dt)
            actuators = advance_actuators(
                midpoint_actuators,
                command,
                self.actuator_parameters,
                0.5 * dt,
            )
            if (step_index + 1) % self.output_stride == 0:
                store(output_index, start + dt)
                output_index += 1
        if output_index != output_count:
            raise RuntimeError("internal/output simulation grids disagree")
        return Simulation(time=self.output_time, **arrays)

    def error_blocks(self, simulation: Simulation) -> dict[str, np.ndarray]:
        observed = self.observations
        orientation_error = np.asarray(
            [
                so3_log(
                    quaternion_to_matrix(
                        observed.sensor_orientation_xyzw[index]
                    ).T
                    @ quaternion_to_matrix(
                        simulation.sensor_orientation_xyzw[index]
                    )
                )
                for index in range(observed.time.size)
            ],
            dtype=float,
        )
        return {
            "position": simulation.sensor_position - observed.sensor_position,
            "orientation": orientation_error,
            "velocity": (
                simulation.sensor_velocity_world
                - observed.sensor_velocity_world
            ),
            "angular_velocity": (
                simulation.angular_velocity_sensor
                - observed.angular_velocity_sensor
            ),
            "specific_force": (
                simulation.specific_force_sensor
                - observed.specific_force_sensor
            ),
        }

    def residual(self, active: Sequence[float]) -> np.ndarray:
        active_array = np.asarray(active, dtype=float)
        self._evaluation_count += 1
        try:
            simulation = self.simulate(active_array)
            errors = self.error_blocks(simulation)
            count_scale = math.sqrt(self.output_time.size)
            residual = np.concatenate(
                (
                    (
                        errors["position"]
                        / self.residual_scales["position_m"]
                        / count_scale
                    ).ravel(),
                    (
                        errors["orientation"]
                        / self.residual_scales["orientation_rad"]
                        / count_scale
                    ).ravel(),
                    (
                        errors["velocity"]
                        / self.residual_scales["velocity_m_per_s"]
                        / count_scale
                    ).ravel(),
                    (
                        errors["angular_velocity"]
                        / self.residual_scales[
                            "angular_velocity_rad_per_s"
                        ]
                        / count_scale
                    ).ravel(),
                    (
                        errors["specific_force"]
                        / self.residual_scales[
                            "specific_force_m_per_s2"
                        ]
                        / count_scale
                    ).ravel(),
                    math.sqrt(self.prior_weight)
                    * active_array
                    / self.prior_scales,
                )
            )
            if not np.all(np.isfinite(residual)):
                raise FloatingPointError("non-finite residual")
        except (
            ValueError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
        ):
            residual = np.full(
                self.output_time.size * 15 + active_array.size,
                1.0e4 + float(np.linalg.norm(active_array)),
                dtype=float,
            )
        cost = 0.5 * float(residual @ residual)
        now = time.monotonic()
        if cost < self._best_cost:
            self._best_cost = cost
        if (
            self._evaluation_count == 1
            or self._evaluation_count % 10 == 0
            or now - self._last_progress_time >= 5.0
        ):
            print(
                "evaluation {:4d}: cost={:.8g}, best={:.8g}".format(
                    self._evaluation_count, cost, self._best_cost
                ),
                flush=True,
            )
            self._last_progress_time = now
        return residual

    def local_dynamics_residual(
        self,
        active: Sequence[float],
        actuator_reference: Simulation,
    ) -> np.ndarray:
        """Fit observed angular acceleration and IMU specific force.

        This inexpensive local problem supplies a stable physical starting
        point for the subsequent five-second single-shooting fit.  It does
        not replace the final open-loop trajectory objective.
        """

        active_array = np.asarray(active, dtype=float)
        residual_size = self.output_time.size * 6 + active_array.size
        try:
            parameters = self.chart.decode(self.full_coordinates(active_array))
            plant = FullSixDofPlant(parameters, self.geometry)
            predicted_alpha = np.empty(
                (self.output_time.size, 3), dtype=float
            )
            predicted_specific_force = np.empty_like(predicted_alpha)
            for index in range(self.output_time.size):
                rotation = self.observed_body_rotation[index]
                omega = self.smoothed_omega_body[index]
                pose_lever = self.pose_sensor_position - parameters.cog_offset
                velocity_lever = (
                    self.velocity_sensor_position - parameters.cog_offset
                )
                rigid = RigidBodyState(
                    position=(
                        self.observations.sensor_position[index]
                        - rotation @ pose_lever
                    ),
                    orientation_xyzw=matrix_to_quaternion(rotation),
                    linear_velocity=(
                        self.observations.sensor_velocity_world[index]
                        - rotation @ np.cross(omega, velocity_lever)
                    ),
                    angular_velocity=omega,
                )
                actuators = ActuatorState(
                    thrust=actuator_reference.actuator_thrust[index],
                    gimbal_angle=actuator_reference.actuator_gimbal[index],
                )
                wrench = plant.total_body_wrench(
                    float(self.output_time[index]), rigid, actuators
                )
                alpha = np.linalg.solve(
                    parameters.inertia,
                    wrench[3:]
                    - np.cross(omega, parameters.inertia @ omega),
                )
                imu_lever = self.imu_sensor_position - parameters.cog_offset
                specific_force_body = (
                    wrench[:3] / parameters.mass
                    + np.cross(alpha, imu_lever)
                    + np.cross(omega, np.cross(omega, imu_lever))
                )
                predicted_alpha[index] = alpha
                predicted_specific_force[index] = (
                    self.body_to_imu_rotation @ specific_force_body
                    + self.accelerometer_bias
                )
            count_scale = math.sqrt(self.output_time.size)
            residual = np.concatenate(
                (
                    (
                        predicted_alpha
                        - self.observed_angular_acceleration_body
                    ).ravel()
                    / (5.0 * count_scale),
                    (
                        predicted_specific_force
                        - self.observations.specific_force_sensor
                    ).ravel()
                    / (
                        self.residual_scales["specific_force_m_per_s2"]
                        * count_scale
                    ),
                    math.sqrt(self.prior_weight)
                    * active_array
                    / self.prior_scales,
                )
            )
            if not np.all(np.isfinite(residual)):
                raise FloatingPointError("non-finite local dynamics residual")
        except (
            ValueError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
        ):
            residual = np.full(
                residual_size,
                1.0e4 + float(np.linalg.norm(active_array)),
                dtype=float,
            )
        return residual


def _metrics(problem: DirectShootingProblem, simulation: Simulation) -> dict[str, Any]:
    errors = problem.error_blocks(simulation)
    orientation_norm = np.linalg.norm(errors["orientation"], axis=1)
    return {
        "position_rmse_m": _vector_rmse(errors["position"]),
        "position_component_rmse_m": _component_rmse(errors["position"]),
        "orientation_angle_rmse_rad": float(
            np.sqrt(np.mean(orientation_norm * orientation_norm))
        ),
        "orientation_angle_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(orientation_norm * orientation_norm)))
        ),
        "velocity_rmse_m_per_s": _vector_rmse(errors["velocity"]),
        "velocity_component_rmse_m_per_s": _component_rmse(
            errors["velocity"]
        ),
        "angular_velocity_rmse_rad_per_s": _vector_rmse(
            errors["angular_velocity"]
        ),
        "specific_force_rmse_m_per_s2": _vector_rmse(
            errors["specific_force"]
        ),
        "terminal_position_error_m": float(
            np.linalg.norm(errors["position"][-1])
        ),
        "terminal_orientation_error_deg": float(
            np.degrees(np.linalg.norm(errors["orientation"][-1]))
        ),
    }


def _physical_parameters(parameters: VehicleParameters) -> dict[str, Any]:
    return {
        "mass_kg": parameters.mass,
        "inertia_kg_m2": parameters.inertia.tolist(),
        "cog_offset_m": parameters.cog_offset.tolist(),
        "force_effectiveness": parameters.force_effectiveness.tolist(),
        "torque_effectiveness": parameters.torque_effectiveness.tolist(),
        "linear_drag": parameters.linear_drag.tolist(),
        "angular_drag": parameters.angular_drag.tolist(),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            _jsonable(payload),
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            sort_keys=True,
        )
        stream.write("\n")
    temporary.replace(path)


def _plot_vector_comparison(
    axes: Sequence[Any],
    time_axis: np.ndarray,
    observed: np.ndarray,
    nominal: np.ndarray,
    estimated: np.ndarray,
    labels: Sequence[str],
    ylabel: str,
) -> None:
    relative_time = time_axis - time_axis[0]
    for component, axis in enumerate(axes):
        axis.plot(
            relative_time,
            observed[:, component],
            color="#1e5abe",
            linewidth=2.0,
            linestyle="-",
            label="observed (rosbag)",
        )
        axis.plot(
            relative_time,
            nominal[:, component],
            color="#d2691e",
            linewidth=1.5,
            linestyle="--",
            label="nominal-parameter rollout",
        )
        axis.plot(
            relative_time,
            estimated[:, component],
            color="#1e965f",
            linewidth=1.7,
            linestyle=":",
            label="estimated-parameter rollout",
        )
        axis.set_ylabel("{} {}".format(labels[component], ylabel))
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time since first common sample [s]")
    axes[0].legend(loc="best", fontsize=8)


def _write_pdf(
    path: Path,
    problem: DirectShootingProblem,
    nominal: Simulation,
    estimated: Simulation,
    nominal_metrics: dict[str, Any],
    estimated_metrics: dict[str, Any],
) -> None:
    observed = problem.observations
    observed_rpy = _rpy_series(observed.sensor_orientation_xyzw)
    nominal_rpy = _rpy_series(nominal.sensor_orientation_xyzw)
    estimated_rpy = _rpy_series(estimated.sensor_orientation_xyzw)
    vector_specs = (
        (
            "Sensor position",
            observed.sensor_position,
            nominal.sensor_position,
            estimated.sensor_position,
            ("x", "y", "z"),
            "[m]",
        ),
        (
            "Sensor orientation (display only; fit uses SO(3) log)",
            observed_rpy,
            nominal_rpy,
            estimated_rpy,
            ("roll", "pitch", "yaw"),
            "[rad]",
        ),
        (
            "World-frame sensor velocity",
            observed.sensor_velocity_world,
            nominal.sensor_velocity_world,
            estimated.sensor_velocity_world,
            ("vx", "vy", "vz"),
            "[m/s]",
        ),
        (
            "IMU angular velocity",
            observed.angular_velocity_sensor,
            nominal.angular_velocity_sensor,
            estimated.angular_velocity_sensor,
            ("wx", "wy", "wz"),
            "[rad/s]",
        ),
        (
            "IMU specific force",
            observed.specific_force_sensor,
            nominal.specific_force_sensor,
            estimated.specific_force_sensor,
            ("fx", "fy", "fz"),
            "[m/s²]",
        ),
    )
    with PdfPages(path) as pdf:
        figure = plt.figure(figsize=(11.7, 8.3), constrained_layout=True)
        figure.suptitle(
            "Observed, nominal, and estimated recorded-control rollouts",
            fontsize=15,
        )
        grid = figure.add_gridspec(2, 2)
        axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
        for values, color, linestyle, linewidth, label in (
            (
                observed.sensor_position,
                "#1e5abe",
                "-",
                2.5,
                "observed (rosbag)",
            ),
            (
                nominal.sensor_position,
                "#d2691e",
                "--",
                1.5,
                "nominal-parameter rollout",
            ),
            (
                estimated.sensor_position,
                "#1e965f",
                ":",
                1.8,
                "estimated-parameter rollout",
            ),
        ):
            axis_3d.plot(
                values[:, 0],
                values[:, 1],
                values[:, 2],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
            )
        axis_3d.set_xlabel("x [m]")
        axis_3d.set_ylabel("y [m]")
        axis_3d.set_zlabel("z [m]")
        axis_3d.set_title("Recorded-control open-loop trajectory")
        axis_3d.legend(loc="best", fontsize=8)

        metric_axis = figure.add_subplot(grid[0, 1])
        metric_axis.axis("off")
        metric_lines = [
            "actual common support: {:.3f}–{:.3f} s".format(
                observed.time[0], observed.time[-1]
            ),
            "",
            "metric                           nominal    estimated",
        ]
        for key, label in (
            ("position_rmse_m", "position RMSE [m]"),
            ("orientation_angle_rmse_deg", "orientation RMSE [deg]"),
            ("velocity_rmse_m_per_s", "velocity RMSE [m/s]"),
            (
                "angular_velocity_rmse_rad_per_s",
                "angular velocity RMSE [rad/s]",
            ),
            ("specific_force_rmse_m_per_s2", "specific force RMSE [m/s²]"),
        ):
            metric_lines.append(
                "{:<32s} {:>9.5g}  {:>9.5g}".format(
                    label,
                    nominal_metrics[key],
                    estimated_metrics[key],
                )
            )
        metric_axis.text(
            0.0,
            1.0,
            "\n".join(metric_lines),
            va="top",
            family="monospace",
            fontsize=9,
        )

        error_axis = figure.add_subplot(grid[1, 1])
        relative_time = observed.time - observed.time[0]
        error_axis.plot(
            relative_time,
            np.linalg.norm(
                nominal.sensor_position - observed.sensor_position,
                axis=1,
            ),
            color="#d2691e",
            linestyle="--",
            label="nominal position error",
        )
        error_axis.plot(
            relative_time,
            np.linalg.norm(
                estimated.sensor_position - observed.sensor_position,
                axis=1,
            ),
            color="#1e965f",
            linestyle=":",
            linewidth=1.8,
            label="estimated position error",
        )
        error_axis.set_xlabel("time since first common sample [s]")
        error_axis.set_ylabel("position error norm [m]")
        error_axis.grid(True, alpha=0.25)
        error_axis.legend(loc="best", fontsize=8)
        pdf.savefig(figure)
        plt.close(figure)

        for (
            title,
            observed_value,
            nominal_value,
            estimated_value,
            labels,
            ylabel,
        ) in vector_specs:
            figure, axes = plt.subplots(
                3,
                1,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            figure.suptitle(title)
            _plot_vector_comparison(
                axes,
                observed.time,
                observed_value,
                nominal_value,
                estimated_value,
                labels,
                ylabel,
            )
            pdf.savefig(figure)
            plt.close(figure)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Directly fit mass, inertia, and CoG so one recorded-control "
            "open-loop rollout matches observed pose/velocity/IMU data."
        )
    )
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--start", type=float, default=19.0)
    parser.add_argument("--end", type=float, default=24.0)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--integration-step", type=float, default=0.025)
    parser.add_argument("--command-delay", type=float, default=0.01)
    parser.add_argument("--prior-weight", type=float, default=0.05)
    parser.add_argument("--initializer-max-nfev", type=int, default=100)
    parser.add_argument("--max-nfev", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    return parser


def parameter_bounds() -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(
        (
            -0.60,
            -1.0,
            -1.0,
            -1.0,
            -0.50,
            -0.50,
            -0.50,
            -0.08,
            -0.08,
            -0.08,
            -0.35,
            -0.35,
            -0.35,
        ),
        dtype=float,
    )
    return lower, -lower


def run(arguments: argparse.Namespace) -> int:
    bag = arguments.bag.expanduser().resolve()
    if not bag.is_file():
        raise SystemExit("bag does not exist: {}".format(bag))
    if not (
        np.isfinite(arguments.start)
        and np.isfinite(arguments.end)
        and arguments.start < arguments.end
        and arguments.sample_step > 0.0
        and arguments.integration_step > 0.0
        and arguments.command_delay >= 0.0
        and arguments.prior_weight >= 0.0
        and arguments.initializer_max_nfev >= 1
        and arguments.max_nfev >= 1
    ):
        raise SystemExit("invalid interval, integration, prior, or iteration setting")
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    print("loading {} [{:.3f}, {:.3f}] s".format(bag, arguments.start, arguments.end))
    started = time.perf_counter()
    flight = load_flight_data(
        str(bag),
        start_local=arguments.start,
        end_local=arguments.end,
        include_fc_specific_force=True,
        compute_sha256=False,
    )
    problem = DirectShootingProblem(
        flight=flight,
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        command_delay=arguments.command_delay,
        prior_weight=arguments.prior_weight,
    )
    initial = np.zeros(ACTIVE_PARAMETER_DIMENSION, dtype=float)
    lower, upper = parameter_bounds()
    nominal_simulation = problem.simulate(initial)
    nominal_metrics = _metrics(problem, nominal_simulation)
    print("nominal metrics: {}".format(json.dumps(nominal_metrics, sort_keys=True)))
    initializer_started = time.perf_counter()
    initializer_result = least_squares(
        lambda coordinates: problem.local_dynamics_residual(
            coordinates, nominal_simulation
        ),
        initial,
        bounds=(lower, upper),
        method="trf",
        jac="2-point",
        x_scale="jac",
        loss="soft_l1",
        ftol=1.0e-7,
        xtol=1.0e-7,
        gtol=1.0e-7,
        max_nfev=arguments.initializer_max_nfev,
        verbose=1,
    )
    initializer_elapsed = time.perf_counter() - initializer_started
    initialized_simulation = problem.simulate(initializer_result.x)
    initialized_metrics = _metrics(problem, initialized_simulation)
    print(
        "dynamics-initialized metrics: {}".format(
            json.dumps(initialized_metrics, sort_keys=True)
        )
    )
    refinement_started = time.perf_counter()
    result = least_squares(
        problem.residual,
        initializer_result.x,
        bounds=(lower, upper),
        method="trf",
        jac="2-point",
        x_scale="jac",
        loss="linear",
        ftol=1.0e-6,
        xtol=1.0e-6,
        gtol=1.0e-6,
        max_nfev=arguments.max_nfev,
        verbose=2,
    )
    refinement_elapsed = time.perf_counter() - refinement_started
    estimated_simulation = problem.simulate(result.x)
    estimated_metrics = _metrics(problem, estimated_simulation)
    full_coordinates = problem.full_coordinates(result.x)
    nominal_parameters = problem.chart.decode(np.zeros(PARAMETER_DIMENSION))
    estimated_parameters = problem.chart.decode(full_coordinates)
    elapsed = time.perf_counter() - started
    payload = {
        "schema": "grape-param-estim/minimal-direct-shooting/v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bag": {
            "path": str(bag),
            "sha256": _sha256(bag),
            "requested_interval_seconds": [arguments.start, arguments.end],
            "fitted_common_support_seconds": [
                float(problem.output_time[0]),
                float(problem.output_time[-1]),
            ],
            "sample_count": int(problem.output_time.size),
        },
        "model": {
            "method": (
                "local observed-dynamics initialization followed by "
                "single-shooting recorded-control open-loop RK4 refinement"
            ),
            "observation_resets": False,
            "latent_states": False,
            "q": None,
            "residual_wrench": None,
            "active_parameter_names": list(ACTIVE_PARAMETER_NAMES),
            "parameter_constraints": {
                "force_effectiveness_geometric_mean": 1.0,
            },
            "fixed_parameters": {
                "torque_effectiveness": [1.0] * 4,
                "rotor_thrust_model": (
                    "issued base_thrust with fixed first-order response"
                ),
                "gimbal_model": (
                    "issued gimbal command with fixed first-order response"
                ),
                "thrust_time_constant_seconds": (
                    problem.actuator_parameters.thrust_time_constant
                ),
                "gimbal_time_constant_seconds": (
                    problem.actuator_parameters.gimbal_time_constant
                ),
                "command_delay_seconds": arguments.command_delay,
            },
            "sample_step_seconds": arguments.sample_step,
            "integration_step_seconds": arguments.integration_step,
            "residual_scales": problem.residual_scales,
            "prior_weight": arguments.prior_weight,
        },
        "optimizer": {
            "name": "scipy.optimize.least_squares",
            "outputs_valid": True,
            "dynamics_initializer": {
                "success": bool(initializer_result.success),
                "status": int(initializer_result.status),
                "message": str(initializer_result.message),
                "cost": float(initializer_result.cost),
                "optimality": float(initializer_result.optimality),
                "nfev": int(initializer_result.nfev),
                "njev": (
                    None
                    if initializer_result.njev is None
                    else int(initializer_result.njev)
                ),
                "active_mask": initializer_result.active_mask.tolist(),
                "elapsed_seconds": initializer_elapsed,
            },
            "trajectory_refinement": {
                "converged": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
                "njev": None if result.njev is None else int(result.njev),
                "active_mask": result.active_mask.tolist(),
                "elapsed_seconds": refinement_elapsed,
            },
            "total_elapsed_seconds": elapsed,
        },
        "coordinates": {
            "nominal_full": np.zeros(PARAMETER_DIMENSION).tolist(),
            "dynamics_initialized_active": {
                name: float(value)
                for name, value in zip(
                    ACTIVE_PARAMETER_NAMES, initializer_result.x
                )
            },
            "estimated_full": full_coordinates.tolist(),
            "estimated_active": {
                name: float(value)
                for name, value in zip(ACTIVE_PARAMETER_NAMES, result.x)
            },
        },
        "parameters": {
            "nominal": _physical_parameters(nominal_parameters),
            "dynamics_initialized": _physical_parameters(
                problem.chart.decode(
                    problem.full_coordinates(initializer_result.x)
                )
            ),
            "estimated": _physical_parameters(estimated_parameters),
        },
        "metrics": {
            "nominal": nominal_metrics,
            "dynamics_initialized": initialized_metrics,
            "estimated": estimated_metrics,
            "position_rmse_improvement_fraction": (
                1.0
                - estimated_metrics["position_rmse_m"]
                / nominal_metrics["position_rmse_m"]
            ),
        },
        "outputs": {
            "json": "result.json",
            "pdf": "trajectory.pdf",
        },
    }
    json_path = output_directory / "result.json"
    pdf_path = output_directory / "trajectory.pdf"
    _write_json(json_path, payload)
    _write_pdf(
        pdf_path,
        problem,
        nominal_simulation,
        estimated_simulation,
        nominal_metrics,
        estimated_metrics,
    )
    print("estimated metrics: {}".format(json.dumps(estimated_metrics, sort_keys=True)))
    print("wrote {}".format(json_path))
    print("wrote {}".format(pdf_path))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
