from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


MINIMAL = Path(__file__).resolve().parents[1]
PROJECT = MINIMAL.parent
for path in (MINIMAL, PROJECT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grape_param_estim.system import ActuatorParameters, GrapeGeometry, VehicleParameters
from savgol_trajectory import GeometricSavitzkyGolayPose
from single_bag_savgol_core import SingleBagDataset, VehicleModelInput
from single_bag_savgol_covariance import build_sg_covariance
from smooth_command import QuinticSmoothZoh


def synthetic_problem_parts():
    raw_time = np.linspace(-1.0, 1.0, 81)
    rng = np.random.default_rng(42)
    position = np.column_stack(
        (
            0.1 + 0.2 * raw_time + 0.3 * raw_time**2,
            -0.2 + 0.1 * raw_time**3,
            0.4 - 0.15 * raw_time + 0.05 * raw_time**2,
        )
    )
    position += 1.0e-5 * rng.standard_normal(position.shape)
    theta = 0.2 * raw_time + 0.04 * raw_time**2
    orientation = np.zeros((raw_time.size, 4))
    orientation[:, 2] = np.sin(0.5 * theta)
    orientation[:, 3] = np.cos(0.5 * theta)
    trajectory = GeometricSavitzkyGolayPose(
        time_axis=raw_time,
        sensor_position=position,
        sensor_orientation_xyzw=orientation,
        pose_sensor_to_body_rotation=np.eye(3),
        window_seconds=0.4,
        degree=3,
    )
    time_axis = trajectory.centered_raw_times(
        support_start=-0.7, support_end=0.7, require_covariance_dof=True
    )[::4]
    sg = trajectory.evaluate(time_axis, centered=True)
    covariance = build_sg_covariance(sg, degree=3, mode="identity")
    reference_covariance = build_sg_covariance(sg, degree=3, mode="full")
    command_time = np.linspace(-1.2, 0.8, 11)
    rotor_values = np.tile(np.asarray((5.0, 5.1, 4.9, 5.0)), (11, 1))
    rotor_values[:, 0] += 0.2 * np.sin(command_time)
    gimbal_values = np.zeros((11, 4))
    gimbal_values[:, 0] = 0.05 * np.sin(command_time)
    dataset = SingleBagDataset(
        bag_id="synthetic",
        time=time_axis,
        sg=sg,
        covariance=covariance,
        reference_sg=sg,
        reference_covariance=reference_covariance,
        rotor_history=QuinticSmoothZoh(command_time, rotor_values),
        gimbal_history=QuinticSmoothZoh(command_time, gimbal_values),
        initial_gimbal=np.zeros(4),
        pose_sensor_position_in_body=np.asarray((0.03, -0.01, 0.02)),
        pose_sensor_to_body_rotation=np.eye(3),
        gyro_sensor_position_in_body=np.asarray((0.02, 0.0, 0.01)),
        body_to_gyro_sensor_rotation=np.eye(3),
        gyro_bias=np.zeros(3),
        accelerometer_bias=np.zeros(3),
        measured_gyro=np.zeros((time_axis.size, 3)),
        measured_specific_force=np.zeros((time_axis.size, 3)),
    )
    parameters = VehicleParameters.nominal()
    model = VehicleModelInput(
        source_path=Path("synthetic.json"),
        parameters=parameters,
        geometry=GrapeGeometry.grape(),
        raw={},
    )
    actuator = ActuatorParameters(
        thrust_time_constant=0.0,
        gimbal_time_constant=0.0,
        minimum_thrust=0.0,
        maximum_thrust=30.0,
        maximum_gimbal_angle=1.0,
        maximum_gimbal_rate=6.0,
    )
    return dataset, model, actuator
