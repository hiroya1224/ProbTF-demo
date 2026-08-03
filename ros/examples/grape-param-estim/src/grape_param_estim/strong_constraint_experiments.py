"""Perfect-model and pose-noise strong-constraint IEnKS experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from grape_param_estim.controller import (
    ControllerConfig,
    initial_controller_state,
)
from grape_param_estim.geometry import (
    euler_xyz_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_vector_from_matrix,
)
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.strong_constraint import (
    CONTROL_DIMENSION,
    PARAMETER_OFFSET,
    IEnKSConfig,
    StrongConstraintIEnKS,
    StrongConstraintPosterior,
    StrongConstraintPrior,
    StrongConstraintProblem,
)
from grape_param_estim.synthetic import (
    SyntheticExperiment,
    run_synthetic_experiment,
)
from grape_param_estim.system import (
    ActuatorParameters,
    GrapeGeometry,
    PoseObservations,
    RigidBodyState,
    VehicleParameters,
)


@dataclass(frozen=True)
class StrongConstraintExperimentMetrics:
    prior_pose_rmse: float
    posterior_pose_rmse: float
    prior_velocity_rmse: float
    posterior_velocity_rmse: float
    prior_omega_rmse: float
    posterior_omega_rmse: float
    prior_identifiable_parameter_error: float
    posterior_identifiable_parameter_error: float
    truth_equivalence_mahalanobis: float
    ridge_variance_ratio: float
    truth_pose_component_coverage: float


@dataclass(frozen=True)
class StrongConstraintExperimentResult:
    label: str
    synthetic: SyntheticExperiment
    truth_control: np.ndarray
    posterior: StrongConstraintPosterior
    prior: StrongConstraintPrior
    metrics: StrongConstraintExperimentMetrics


def default_strong_constraint_truth_coordinates(
    chart: VehicleParameterChart,
) -> np.ndarray:
    """A non-nominal static plant inside the exact estimation model family."""

    value = np.zeros(18, dtype=float)
    value[0] = 0.08
    value[1:4] = np.asarray((0.05, 0.10, 0.07))
    value[4:7] = np.asarray((0.015, -0.010, 0.012))
    value[7:10] = np.asarray((0.004, -0.003, 0.002))
    value[10:14] = np.asarray((-0.04, 0.02, -0.03, 0.01))
    value[14:18] = np.asarray((0.03, -0.02, 0.025, -0.015))
    # Include a component along the known exact ridge.  Recovery must be
    # judged modulo this direction, never by demanding one arbitrary point.
    direction = chart.ridge_direction()
    value += 0.035 * direction / np.max(np.abs(direction))
    return value


def _problem_from_synthetic(
    experiment: SyntheticExperiment,
) -> StrongConstraintProblem:
    configuration = ControllerConfig.grape()
    first_reference = experiment.references[0]
    initial_state = RigidBodyState(
        position=first_reference.position,
        orientation_xyzw=matrix_to_quaternion(
            euler_xyz_to_matrix(first_reference.rpy)
        ),
        linear_velocity=first_reference.linear_velocity,
        angular_velocity=first_reference.angular_velocity,
    )
    return StrongConstraintProblem(
        references=experiment.references,
        observations=experiment.observations,
        nominal_trajectory=experiment.nominal,
        initial_state_anchor=initial_state,
        initial_controller_anchor=initial_controller_state(
            configuration, trim_hover=True
        ),
        controller_configuration=configuration,
        controller_parameters=experiment.controller_parameters,
        geometry=GrapeGeometry.grape(),
        actuator_parameters=experiment.nominal_actuator_parameters,
        parameter_chart=VehicleParameterChart(
            experiment.controller_parameters
        ),
    )


def _mean_trajectory(trajectories, field):
    return np.mean(
        np.asarray([getattr(value, field) for value in trajectories]), axis=0
    )


def _vector_rmse(candidate, truth):
    return float(np.sqrt(np.mean((candidate - truth) ** 2)))


def _pose_rmse(trajectories, truth):
    position = _mean_trajectory(trajectories, "position")
    return _vector_rmse(position, truth.position)


def _component_coverage(posterior, truth):
    position_error = np.asarray(
        [value.position for value in posterior.trajectory_ensemble]
    ) - truth.position[None, :, :]
    rotation_error = np.empty_like(position_error)
    for member, trajectory in enumerate(posterior.trajectory_ensemble):
        for index in range(truth.times.size):
            truth_rotation = quaternion_to_matrix(
                truth.orientation_xyzw[index]
            )
            member_rotation = quaternion_to_matrix(
                trajectory.orientation_xyzw[index]
            )
            rotation_error[member, index] = rotation_vector_from_matrix(
                truth_rotation.T @ member_rotation
            )
    residual = np.concatenate((position_error, rotation_error), axis=2)
    lower = np.percentile(residual, 2.5, axis=0)
    upper = np.percentile(residual, 97.5, axis=0)
    return float(np.mean((lower <= 0.0) & (upper >= 0.0)))


def _metrics(
    experiment: SyntheticExperiment,
    truth_control: np.ndarray,
    prior: StrongConstraintPrior,
    posterior: StrongConstraintPosterior,
) -> StrongConstraintExperimentMetrics:
    direction = posterior.ridge.expected_direction
    projector = np.eye(direction.size) - np.outer(direction, direction)
    truth_parameter = truth_control[PARAMETER_OFFSET:]
    prior_parameter = prior.mean[PARAMETER_OFFSET:]
    posterior_parameter = np.mean(
        posterior.parameter_ensemble.coordinates, axis=0
    )
    projected_error = projector @ (posterior_parameter - truth_parameter)
    projected_covariance = (
        projector @ posterior.ridge.covariance @ projector
    )
    eigenvalues, eigenvectors = np.linalg.eigh(projected_covariance)
    represented = eigenvalues > max(1.0e-12, 1.0e-10 * eigenvalues[-1])
    whitened_error = (
        eigenvectors[:, represented].T @ projected_error
    ) / np.sqrt(eigenvalues[represented])
    prior_variance = float(
        direction
        @ prior.covariance[PARAMETER_OFFSET:, PARAMETER_OFFSET:]
        @ direction
    )
    return StrongConstraintExperimentMetrics(
        prior_pose_rmse=_pose_rmse(
            posterior.prior_trajectory_ensemble, experiment.truth
        ),
        posterior_pose_rmse=_pose_rmse(
            posterior.trajectory_ensemble, experiment.truth
        ),
        prior_velocity_rmse=_vector_rmse(
            _mean_trajectory(
                posterior.prior_trajectory_ensemble, "linear_velocity"
            ),
            experiment.truth.linear_velocity,
        ),
        posterior_velocity_rmse=_vector_rmse(
            _mean_trajectory(
                posterior.trajectory_ensemble, "linear_velocity"
            ),
            experiment.truth.linear_velocity,
        ),
        prior_omega_rmse=_vector_rmse(
            _mean_trajectory(
                posterior.prior_trajectory_ensemble, "angular_velocity"
            ),
            experiment.truth.angular_velocity,
        ),
        posterior_omega_rmse=_vector_rmse(
            _mean_trajectory(
                posterior.trajectory_ensemble, "angular_velocity"
            ),
            experiment.truth.angular_velocity,
        ),
        prior_identifiable_parameter_error=float(
            np.linalg.norm(projector @ (prior_parameter - truth_parameter))
        ),
        posterior_identifiable_parameter_error=float(
            np.linalg.norm(projected_error)
        ),
        truth_equivalence_mahalanobis=float(
            np.dot(whitened_error, whitened_error)
        ),
        ridge_variance_ratio=float(
            posterior.ridge.expected_variance / prior_variance
        ),
        truth_pose_component_coverage=_component_coverage(
            posterior, experiment.truth
        ),
    )


def run_strong_constraint_experiment(
    label: str = "A",
    duration: float = 1.2,
    time_step: float = 0.04,
    ensemble_size: int = 48,
    maximum_iterations: int = 4,
    seed: int = 23,
    truth_parameter_coordinates: Optional[np.ndarray] = None,
) -> StrongConstraintExperimentResult:
    """Run Experiment A (zero realization) or B (pose noise only)."""

    normalized_label = str(label).upper()
    if normalized_label not in ("A", "B"):
        raise ValueError("strong-constraint experiment label must be A or B")
    nominal = VehicleParameters.nominal()
    chart = VehicleParameterChart(nominal)
    truth_parameter = (
        default_strong_constraint_truth_coordinates(chart)
        if truth_parameter_coordinates is None
        else np.asarray(truth_parameter_coordinates, dtype=float)
    )
    if truth_parameter.shape != (18,) or not np.all(np.isfinite(truth_parameter)):
        raise ValueError("truth parameter coordinates must be a finite 18-vector")
    translation_sigma = 0.003
    rotation_sigma = np.deg2rad(0.20)
    realized_translation_sigma = (
        0.0 if normalized_label == "A" else translation_sigma
    )
    realized_rotation_sigma = (
        0.0 if normalized_label == "A" else rotation_sigma
    )
    synthetic = run_synthetic_experiment(
        duration=duration,
        time_step=time_step,
        truth_parameters=chart.decode(truth_parameter),
        truth_actuators=ActuatorParameters(),
        truth_residual_wrench=lambda _time, _state: np.zeros(6),
        translation_noise=realized_translation_sigma,
        rotation_noise=realized_rotation_sigma,
        seed=seed + 1000,
    )
    if normalized_label == "A":
        # The realization is exactly zero, but likelihood covariance remains
        # positive definite so the variational problem is well-posed.
        observations = PoseObservations(
            times=synthetic.observations.times,
            position=synthetic.observations.position,
            orientation_xyzw=synthetic.observations.orientation_xyzw,
            translation_covariance=np.eye(3) * translation_sigma**2,
            rotation_covariance=np.eye(3) * rotation_sigma**2,
        )
        synthetic = SyntheticExperiment(
            references=synthetic.references,
            nominal=synthetic.nominal,
            truth=synthetic.truth,
            observations=observations,
            correction_translation=synthetic.correction_translation,
            correction_rotation_vector=(
                synthetic.correction_rotation_vector
            ),
            controller_parameters=synthetic.controller_parameters,
            truth_parameters=synthetic.truth_parameters,
            nominal_actuator_parameters=(
                synthetic.nominal_actuator_parameters
            ),
            truth_actuator_parameters=synthetic.truth_actuator_parameters,
        )
    problem = _problem_from_synthetic(synthetic)
    prior = StrongConstraintPrior.grape()
    posterior = StrongConstraintIEnKS(
        IEnKSConfig(
            ensemble_size=ensemble_size,
            maximum_iterations=maximum_iterations,
            seed=seed,
        )
    ).fit(problem, prior)
    truth_control = np.zeros(CONTROL_DIMENSION, dtype=float)
    truth_control[PARAMETER_OFFSET:] = truth_parameter
    return StrongConstraintExperimentResult(
        label=normalized_label,
        synthetic=synthetic,
        truth_control=truth_control,
        posterior=posterior,
        prior=prior,
        metrics=_metrics(synthetic, truth_control, prior, posterior),
    )


def save_strong_constraint_experiment(
    path: str, result: StrongConstraintExperimentResult
) -> Path:
    """Save raw members and member-aligned trajectory/path ensembles."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    posterior = result.posterior
    trajectories = posterior.trajectory_ensemble
    prior_trajectories = posterior.prior_trajectory_ensemble
    iterations = posterior.iterations
    np.savez_compressed(
        str(destination),
        schema=np.asarray(
            ("grape-param-estim/strong-constraint-experiment/v1",)
        ),
        experiment=np.asarray((result.label,)),
        times=result.synthetic.truth.times,
        reference_position=np.asarray(
            [value.position for value in result.synthetic.references]
        ),
        reference_linear_velocity=np.asarray(
            [value.linear_velocity for value in result.synthetic.references]
        ),
        reference_linear_acceleration=np.asarray(
            [value.linear_acceleration for value in result.synthetic.references]
        ),
        reference_rpy=np.asarray(
            [value.rpy for value in result.synthetic.references]
        ),
        reference_angular_velocity=np.asarray(
            [value.angular_velocity for value in result.synthetic.references]
        ),
        reference_angular_acceleration=np.asarray(
            [value.angular_acceleration for value in result.synthetic.references]
        ),
        observations_position=result.synthetic.observations.position,
        observations_orientation_xyzw=(
            result.synthetic.observations.orientation_xyzw
        ),
        observation_translation_covariance=(
            result.synthetic.observations.translation_covariance
        ),
        observation_rotation_covariance=(
            result.synthetic.observations.rotation_covariance
        ),
        truth_position=result.synthetic.truth.position,
        truth_orientation_xyzw=result.synthetic.truth.orientation_xyzw,
        truth_linear_velocity=result.synthetic.truth.linear_velocity,
        truth_angular_velocity=result.synthetic.truth.angular_velocity,
        nominal_position=result.synthetic.nominal.position,
        nominal_orientation_xyzw=(
            result.synthetic.nominal.orientation_xyzw
        ),
        truth_control=result.truth_control,
        prior_mean=result.prior.mean,
        prior_covariance=result.prior.covariance,
        prior_control_ensemble=posterior.prior_control_ensemble,
        posterior_control_ensemble=posterior.control_ensemble,
        center_control=posterior.center_control,
        center_position=posterior.center_trajectory.position,
        center_orientation_xyzw=(
            posterior.center_trajectory.orientation_xyzw
        ),
        center_linear_velocity=posterior.center_trajectory.linear_velocity,
        center_angular_velocity=posterior.center_trajectory.angular_velocity,
        posterior_parameter_coordinates=(
            posterior.parameter_ensemble.coordinates
        ),
        posterior_mass=posterior.parameter_ensemble.mass,
        posterior_inertia=posterior.parameter_ensemble.inertia,
        posterior_cog_offset=posterior.parameter_ensemble.cog_offset,
        posterior_force_effectiveness=(
            posterior.parameter_ensemble.force_effectiveness
        ),
        posterior_torque_effectiveness=(
            posterior.parameter_ensemble.torque_effectiveness
        ),
        prior_position=np.asarray([value.position for value in prior_trajectories]),
        prior_orientation_xyzw=np.asarray(
            [value.orientation_xyzw for value in prior_trajectories]
        ),
        posterior_position=np.asarray(
            [value.position for value in trajectories]
        ),
        posterior_orientation_xyzw=np.asarray(
            [value.orientation_xyzw for value in trajectories]
        ),
        posterior_linear_velocity=np.asarray(
            [value.linear_velocity for value in trajectories]
        ),
        posterior_angular_velocity=np.asarray(
            [value.angular_velocity for value in trajectories]
        ),
        posterior_controller_integral=np.asarray(
            [value.controller_integral for value in trajectories]
        ),
        posterior_commanded_thrust=np.asarray(
            [value.commanded_thrust for value in trajectories]
        ),
        posterior_commanded_gimbal_angle=np.asarray(
            [value.commanded_gimbal_angle for value in trajectories]
        ),
        posterior_actuator_thrust=np.asarray(
            [value.actuator_thrust for value in trajectories]
        ),
        posterior_actuator_gimbal_angle=np.asarray(
            [value.actuator_gimbal_angle for value in trajectories]
        ),
        posterior_body_wrench=np.asarray(
            [value.body_wrench for value in trajectories]
        ),
        correction_translation=posterior.correction_translation,
        correction_rotation_vector=(
            posterior.correction_rotation_vector
        ),
        ridge_covariance=posterior.ridge.covariance,
        ridge_eigenvalues=posterior.ridge.eigenvalues,
        expected_ridge_direction=posterior.ridge.expected_direction,
        truth_equivalence_mahalanobis=np.asarray(
            (result.metrics.truth_equivalence_mahalanobis,)
        ),
        iteration_objective=np.asarray(
            [value.objective for value in iterations]
        ),
        iteration_accepted_objective=np.asarray(
            [value.accepted_objective for value in iterations]
        ),
        iteration_gradient_norm=np.asarray(
            [value.gradient_norm for value in iterations]
        ),
        iteration_step_norm=np.asarray(
            [value.step_norm for value in iterations]
        ),
        iteration_accepted_fraction=np.asarray(
            [value.accepted_fraction for value in iterations]
        ),
        converged=np.asarray((posterior.converged,), dtype=bool),
        termination_reason=np.asarray((posterior.termination_reason,)),
    )
    return destination
