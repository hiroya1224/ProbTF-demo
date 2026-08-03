"""Reproducible full closed-loop synthetic episodes for assimilation tests."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.controller import (
    ControllerConfig,
    GrapeController,
    initial_controller_state,
)
from grape_param_estim.dynamics import FullSixDofPlant, ResidualWrench
from grape_param_estim.dynamics import simulate_closed_loop
from grape_param_estim.geometry import (
    correction_transform_path,
    euler_xyz_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_matrix_from_vector,
)
from grape_param_estim.system import (
    ActuatorParameters,
    ClosedLoopTrajectory,
    GrapeGeometry,
    PoseObservations,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


@dataclass(frozen=True)
class SyntheticExperiment:
    """Synthetic output, including latent truth but pose-only observations."""

    references: Tuple[ReferenceState, ...]
    nominal: ClosedLoopTrajectory
    truth: ClosedLoopTrajectory
    observations: PoseObservations
    correction_translation: np.ndarray
    correction_rotation_vector: np.ndarray
    controller_parameters: VehicleParameters
    truth_parameters: VehicleParameters
    nominal_actuator_parameters: ActuatorParameters
    truth_actuator_parameters: ActuatorParameters

    def __post_init__(self) -> None:
        if not isinstance(
            self.nominal_actuator_parameters, ActuatorParameters
        ) or not isinstance(self.truth_actuator_parameters, ActuatorParameters):
            raise TypeError("synthetic actuator parameters are invalid")
        count = self.nominal.times.size
        if len(self.references) != count:
            raise ValueError("reference and trajectory lengths must agree")
        if not np.array_equal(self.nominal.times, self.truth.times):
            raise ValueError("nominal and truth trajectories must share time")
        if not np.array_equal(self.nominal.times, self.observations.times):
            raise ValueError("observations must share the trajectory time base")
        for name in (
            "correction_translation",
            "correction_rotation_vector",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (count, 3) or not np.all(np.isfinite(value)):
                raise ValueError("{} must be a finite path".format(name))
            object.__setattr__(self, name, value.copy())


def _angular_kinematics(rpy: np.ndarray, times: np.ndarray):
    rotations = np.asarray([euler_xyz_to_matrix(value) for value in rpy])
    rotation_rate = np.gradient(rotations, times, axis=0, edge_order=2)
    omega = np.empty((times.size, 3), dtype=float)
    for index, (rotation, rate) in enumerate(zip(rotations, rotation_rate)):
        omega_matrix = 0.5 * (
            rotation.T @ rate - rate.T @ rotation
        )
        omega[index] = np.asarray(
            (
                omega_matrix[2, 1],
                omega_matrix[0, 2],
                omega_matrix[1, 0],
            )
        )
    angular_acceleration = np.gradient(
        omega, times, axis=0, edge_order=2
    )
    return omega, angular_acceleration


def full_six_dof_reference(
    times: Sequence[float],
) -> Tuple[ReferenceState, ...]:
    """Create a smooth trajectory that excites all translational/rotational axes."""

    times = np.asarray(times, dtype=float)
    position = np.column_stack(
        (
            0.22 * np.sin(0.55 * times),
            0.18 * (np.cos(0.47 * times) - 1.0),
            1.0 + 0.12 * np.sin(0.63 * times),
        )
    )
    velocity = np.gradient(position, times, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, times, axis=0, edge_order=2)
    rpy = np.column_stack(
        (
            0.075 * np.sin(0.71 * times),
            0.060 * np.sin(0.53 * times + 0.35),
            0.24 * np.sin(0.39 * times),
        )
    )
    omega, angular_acceleration = _angular_kinematics(rpy, times)
    return tuple(
        ReferenceState(
            position[index],
            velocity[index],
            acceleration[index],
            rpy[index],
            omega[index],
            angular_acceleration[index],
        )
        for index in range(times.size)
    )


def generate_pose_observations(
    trajectory: ClosedLoopTrajectory,
    translation_standard_deviation: float,
    rotation_standard_deviation: float,
    seed: int,
) -> PoseObservations:
    """Observe pose only; velocity, IMU and commands never enter this object."""

    translation_sigma = float(translation_standard_deviation)
    rotation_sigma = float(rotation_standard_deviation)
    if translation_sigma < 0.0 or rotation_sigma < 0.0:
        raise ValueError("observation standard deviations cannot be negative")
    generator = np.random.RandomState(int(seed))
    position = trajectory.position + generator.normal(
        0.0, translation_sigma, trajectory.position.shape
    )
    orientation = np.empty_like(trajectory.orientation_xyzw)
    for index, quaternion in enumerate(trajectory.orientation_xyzw):
        noise = generator.normal(0.0, rotation_sigma, 3)
        noisy_rotation = quaternion_to_matrix(quaternion) @ (
            rotation_matrix_from_vector(noise)
        )
        orientation[index] = matrix_to_quaternion(noisy_rotation)
    return PoseObservations(
        times=trajectory.times,
        position=position,
        orientation_xyzw=orientation,
        translation_covariance=np.eye(3) * translation_sigma**2,
        rotation_covariance=np.eye(3) * rotation_sigma**2,
    )


def default_truth_parameters() -> VehicleParameters:
    nominal = VehicleParameters.nominal()
    inertia_scale = np.asarray((1.10, 0.91, 1.06))
    inertia_transform = np.diag(np.sqrt(inertia_scale))
    return VehicleParameters(
        mass=1.08 * nominal.mass,
        inertia=inertia_transform @ nominal.inertia @ inertia_transform,
        cog_offset=np.asarray((0.014, -0.010, 0.005)),
        force_effectiveness=np.asarray((0.97, 1.02, 0.95, 1.01)),
        torque_effectiveness=np.asarray((1.03, 0.96, 1.05, 0.98)),
        linear_drag=np.asarray((0.24, 0.20, 0.31)),
        angular_drag=np.asarray((0.012, 0.010, 0.018)),
    )


def default_residual_wrench(
    time: float, _state: RigidBodyState
) -> np.ndarray:
    """A deterministic truth-only wind/motor residual in the body frame."""

    return np.asarray(
        (
            0.34 * np.sin(0.73 * time),
            0.27 * np.cos(0.51 * time + 0.2),
            0.16 * np.sin(0.41 * time),
            0.012 * np.sin(0.67 * time),
            -0.010 * np.cos(0.59 * time),
            0.014 * np.sin(0.43 * time + 0.3),
        ),
        dtype=float,
    )


def run_synthetic_experiment(
    duration: float = 6.0,
    time_step: float = 0.02,
    truth_parameters: Optional[VehicleParameters] = None,
    truth_actuators: Optional[ActuatorParameters] = None,
    truth_residual_wrench: Optional[ResidualWrench] = None,
    translation_noise: float = 0.004,
    rotation_noise: float = np.deg2rad(0.25),
    seed: int = 7,
) -> SyntheticExperiment:
    """Run nominal and candidate-real closed loops from one initial state."""

    duration = float(duration)
    time_step = float(time_step)
    if duration <= 0.0 or time_step <= 0.0:
        raise ValueError("duration and time_step must be positive")
    sample_count = int(np.floor(duration / time_step + 0.5)) + 1
    times = np.linspace(0.0, duration, sample_count)
    references = full_six_dof_reference(times)
    geometry = GrapeGeometry.grape()
    articulated_model = GrapeArticulatedModel()
    nominal_parameters = VehicleParameters.nominal()
    controller_configuration = ControllerConfig.grape()
    initial_reference = references[0]
    initial_state = RigidBodyState(
        position=initial_reference.position,
        orientation_xyzw=matrix_to_quaternion(
            euler_xyz_to_matrix(initial_reference.rpy)
        ),
        linear_velocity=initial_reference.linear_velocity,
        angular_velocity=initial_reference.angular_velocity,
    )
    controller_initial = initial_controller_state(
        controller_configuration, trim_hover=True
    )

    nominal_actuators = ActuatorParameters()
    nominal = simulate_closed_loop(
        times=times,
        references=references,
        initial_state=initial_state,
        initial_controller_state=controller_initial,
        controller=GrapeController(
            controller_configuration,
            nominal_parameters,
            geometry,
            articulated_model=articulated_model,
        ),
        plant=FullSixDofPlant(nominal_parameters, geometry),
        actuator_parameters=nominal_actuators,
    )

    selected_truth = truth_parameters or default_truth_parameters()
    selected_actuators = truth_actuators or ActuatorParameters(
        thrust_time_constant=0.045,
        gimbal_time_constant=0.065,
        delay=0.02,
    )
    residual = (
        default_residual_wrench
        if truth_residual_wrench is None
        else truth_residual_wrench
    )
    truth = simulate_closed_loop(
        times=times,
        references=references,
        initial_state=initial_state,
        initial_controller_state=controller_initial,
        controller=GrapeController(
            controller_configuration,
            nominal_parameters,
            geometry,
            articulated_model=articulated_model,
        ),
        plant=FullSixDofPlant(
            selected_truth, geometry, residual_wrench=residual
        ),
        actuator_parameters=selected_actuators,
    )
    observations = generate_pose_observations(
        truth,
        translation_standard_deviation=translation_noise,
        rotation_standard_deviation=rotation_noise,
        seed=seed,
    )
    correction_translation, correction_rotation = correction_transform_path(
        nominal.position,
        nominal.orientation_xyzw,
        truth.position,
        truth.orientation_xyzw,
    )
    return SyntheticExperiment(
        references=references,
        nominal=nominal,
        truth=truth,
        observations=observations,
        correction_translation=correction_translation,
        correction_rotation_vector=correction_rotation,
        controller_parameters=nominal_parameters,
        truth_parameters=selected_truth,
        nominal_actuator_parameters=nominal_actuators,
        truth_actuator_parameters=selected_actuators,
    )


def run_perfect_model_experiment(
    duration: float = 3.0,
    time_step: float = 0.02,
) -> SyntheticExperiment:
    """Experiment-A base case: identical controller and plant models."""

    nominal = VehicleParameters.nominal()
    return run_synthetic_experiment(
        duration=duration,
        time_step=time_step,
        truth_parameters=nominal,
        truth_actuators=ActuatorParameters(),
        truth_residual_wrench=lambda _time, _state: np.zeros(6),
        translation_noise=0.0,
        rotation_noise=0.0,
        seed=0,
    )


def save_experiment(path: str, experiment: SyntheticExperiment) -> Path:
    """Persist synthetic arrays without pickle or a legacy schema."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    reference = experiment.references
    np.savez_compressed(
        str(destination),
        schema=np.asarray(("grape-param-estim/synthetic-closed-loop/v1",)),
        times=experiment.nominal.times,
        reference_position=np.asarray([item.position for item in reference]),
        reference_rpy=np.asarray([item.rpy for item in reference]),
        nominal_position=experiment.nominal.position,
        nominal_orientation_xyzw=experiment.nominal.orientation_xyzw,
        nominal_linear_velocity=experiment.nominal.linear_velocity,
        nominal_angular_velocity=experiment.nominal.angular_velocity,
        nominal_commanded_thrust=experiment.nominal.commanded_thrust,
        nominal_commanded_gimbal_angle=(
            experiment.nominal.commanded_gimbal_angle
        ),
        truth_position=experiment.truth.position,
        truth_orientation_xyzw=experiment.truth.orientation_xyzw,
        truth_linear_velocity=experiment.truth.linear_velocity,
        truth_angular_velocity=experiment.truth.angular_velocity,
        truth_commanded_thrust=experiment.truth.commanded_thrust,
        truth_commanded_gimbal_angle=experiment.truth.commanded_gimbal_angle,
        observed_position=experiment.observations.position,
        observed_orientation_xyzw=experiment.observations.orientation_xyzw,
        observation_translation_covariance=(
            experiment.observations.translation_covariance
        ),
        observation_rotation_covariance=(
            experiment.observations.rotation_covariance
        ),
        correction_translation=experiment.correction_translation,
        correction_rotation_vector=experiment.correction_rotation_vector,
        controller_mass=np.asarray(
            (experiment.controller_parameters.mass,), dtype=float
        ),
        controller_inertia=experiment.controller_parameters.inertia,
        truth_mass=np.asarray(
            (experiment.truth_parameters.mass,), dtype=float
        ),
        truth_inertia=experiment.truth_parameters.inertia,
        truth_cog_offset=experiment.truth_parameters.cog_offset,
        truth_force_effectiveness=(
            experiment.truth_parameters.force_effectiveness
        ),
        truth_torque_effectiveness=(
            experiment.truth_parameters.torque_effectiveness
        ),
        truth_linear_drag=experiment.truth_parameters.linear_drag,
        truth_angular_drag=experiment.truth_parameters.angular_drag,
        nominal_constant_delay=np.asarray(
            (experiment.nominal_actuator_parameters.delay,), dtype=float
        ),
        truth_constant_delay=np.asarray(
            (experiment.truth_actuator_parameters.delay,), dtype=float
        ),
    )
    return destination
