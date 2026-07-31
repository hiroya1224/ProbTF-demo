"""Experiment C: strong versus weak constraint under model error."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.controller import (
    ControllerConfig,
    GrapeController,
    initial_controller_state,
)
from grape_param_estim.dynamics import (
    FullSixDofPlant,
    actuator_wrench,
    advance_actuators,
)
from grape_param_estim.geometry import (
    normalise_quaternion,
    quaternion_to_matrix,
    rotation_vector_from_matrix,
)
from grape_param_estim.model_error import GaussMarkovWrenchProcess
from grape_param_estim.strong_constraint import (
    PARAMETER_OFFSET,
    IEnKSConfig,
    StrongConstraintIEnKS,
    StrongConstraintPosterior,
    StrongConstraintPrior,
)
from grape_param_estim.strong_constraint_experiments import (
    _problem_from_synthetic,
)
from grape_param_estim.synthetic import (
    SyntheticExperiment,
    default_residual_wrench,
    run_synthetic_experiment,
)
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    GrapeGeometry,
    RigidBodyState,
    VehicleParameters,
)
from grape_param_estim.weak_constraint import (
    WeakConstraintIEnKSQ,
    WeakConstraintPosterior,
    WeakConstraintPrior,
    WeakConstraintProblem,
)


@dataclass(frozen=True)
class Phase3Metrics:
    matched_strong_static_bias: float
    strong_static_bias: float
    weak_static_bias: float
    strong_pose_rmse: float
    weak_pose_rmse: float
    strong_rotation_rmse: float
    weak_rotation_rmse: float
    strong_velocity_rmse: float
    weak_velocity_rmse: float
    strong_omega_rmse: float
    weak_omega_rmse: float
    strong_path_coverage: float
    weak_path_coverage: float
    residual_acceleration_r_squared: float
    residual_excited_channel_correlation: float
    residual_component_coverage: float


@dataclass(frozen=True)
class Phase3Experiment:
    synthetic: SyntheticExperiment
    matched_synthetic: SyntheticExperiment
    truth_static_coordinates: np.ndarray
    oracle_residual_wrench: np.ndarray
    wrench_process: GaussMarkovWrenchProcess
    matched_strong_posterior: StrongConstraintPosterior
    strong_posterior: StrongConstraintPosterior
    weak_posterior: WeakConstraintPosterior
    metrics: Phase3Metrics


@dataclass(frozen=True)
class CounterfactualActuatorReplay:
    """Nominal closed-loop commands/actuators replayed on the truth states."""

    commanded_thrust: np.ndarray
    commanded_gimbal_angle: np.ndarray
    actuator_thrust: np.ndarray
    actuator_gimbal_angle: np.ndarray
    interval_midpoint_thrust: np.ndarray
    interval_midpoint_gimbal_angle: np.ndarray


def _static_truth_without_model_error(
    parameters: VehicleParameters,
) -> VehicleParameters:
    return VehicleParameters(
        mass=parameters.mass,
        inertia=parameters.inertia,
        cog_offset=parameters.cog_offset,
        force_effectiveness=parameters.force_effectiveness,
        torque_effectiveness=parameters.torque_effectiveness,
        linear_drag=np.zeros(3),
        angular_drag=np.zeros(3),
    )


def _midpoint_state(trajectory, index):
    quaternion = normalise_quaternion(
        trajectory.orientation_xyzw[index]
        + trajectory.orientation_xyzw[index + 1]
    )
    return RigidBodyState(
        position=0.5 * (
            trajectory.position[index] + trajectory.position[index + 1]
        ),
        orientation_xyzw=quaternion,
        linear_velocity=0.5 * (
            trajectory.linear_velocity[index]
            + trajectory.linear_velocity[index + 1]
        ),
        angular_velocity=0.5 * (
            trajectory.angular_velocity[index]
            + trajectory.angular_velocity[index + 1]
        ),
    )


def _trajectory_state(trajectory, index):
    return RigidBodyState(
        position=trajectory.position[index],
        orientation_xyzw=trajectory.orientation_xyzw[index],
        linear_velocity=trajectory.linear_velocity[index],
        angular_velocity=trajectory.angular_velocity[index],
    )


def replay_nominal_actuators_on_truth(
    experiment: SyntheticExperiment,
) -> CounterfactualActuatorReplay:
    """Replay the estimator's causal controller/actuator path on truth x_k.

    The replay is deliberately independent of the commands recorded in the
    truth episode.  In particular, the controller at sample k sees the
    nominal actuator's own gimbal state from sample k, just as a weak-model
    forecast would.  Only the rigid-body state is clamped to truth so the
    resulting wrench is the counterfactual base wrench for that path.
    """

    truth = experiment.truth
    sample_count = truth.times.size
    configuration = ControllerConfig.grape()
    controller = GrapeController(
        configuration,
        experiment.controller_parameters,
        GrapeGeometry.grape(),
        articulated_model=GrapeArticulatedModel(),
    )
    controller_state = initial_controller_state(
        configuration, trim_hover=True
    )
    actuator_parameters = ActuatorParameters()
    actuator_state = None
    commanded_thrust = np.empty((sample_count, 4), dtype=float)
    commanded_gimbal = np.empty((sample_count, 4), dtype=float)
    actuator_thrust = np.empty((sample_count, 4), dtype=float)
    actuator_gimbal = np.empty((sample_count, 4), dtype=float)
    midpoint_thrust = np.empty((sample_count - 1, 4), dtype=float)
    midpoint_gimbal = np.empty((sample_count - 1, 4), dtype=float)

    for index in range(sample_count):
        time_step = (
            truth.times[index + 1] - truth.times[index]
            if index + 1 < sample_count
            else truth.times[index] - truth.times[index - 1]
        )
        command, next_controller_state = controller.step(
            _trajectory_state(truth, index),
            experiment.references[index],
            controller_state,
            time_step,
            None if actuator_state is None else actuator_state.gimbal_angle,
        )
        commanded_thrust[index] = command.thrust
        commanded_gimbal[index] = command.gimbal_angle
        if actuator_state is None:
            actuator_state = ActuatorState(
                thrust=np.clip(
                    command.thrust,
                    actuator_parameters.minimum_thrust,
                    actuator_parameters.maximum_thrust,
                ),
                gimbal_angle=np.clip(
                    command.gimbal_angle,
                    -actuator_parameters.maximum_gimbal_angle,
                    actuator_parameters.maximum_gimbal_angle,
                ),
            )
        actuator_thrust[index] = actuator_state.thrust
        actuator_gimbal[index] = actuator_state.gimbal_angle
        if index + 1 == sample_count:
            break
        midpoint = advance_actuators(
            actuator_state,
            command,
            actuator_parameters,
            0.5 * time_step,
        )
        midpoint_thrust[index] = midpoint.thrust
        midpoint_gimbal[index] = midpoint.gimbal_angle
        actuator_state = advance_actuators(
            midpoint,
            command,
            actuator_parameters,
            0.5 * time_step,
        )
        controller_state = next_controller_state

    return CounterfactualActuatorReplay(
        commanded_thrust=commanded_thrust,
        commanded_gimbal_angle=commanded_gimbal,
        actuator_thrust=actuator_thrust,
        actuator_gimbal_angle=actuator_gimbal,
        interval_midpoint_thrust=midpoint_thrust,
        interval_midpoint_gimbal_angle=midpoint_gimbal,
    )


def oracle_effective_residual_wrench(
    experiment: SyntheticExperiment,
) -> np.ndarray:
    """Project drag, actuator mismatch and external force to body wrench.

    The base actuator path is a causal nominal counterfactual, not the truth
    command path.  Reusing truth commands would leak the truth actuator's
    lagged gimbal feedback into the weak model and give the residual estimator
    a physically inconsistent oracle target.
    """

    truth = experiment.truth
    geometry = GrapeGeometry.grape()
    static_truth = _static_truth_without_model_error(
        experiment.truth_parameters
    )
    truth_plant = FullSixDofPlant(
        experiment.truth_parameters,
        geometry,
        residual_wrench=default_residual_wrench,
    )
    counterfactual = replay_nominal_actuators_on_truth(experiment)
    result = np.empty((truth.times.size - 1, 6), dtype=float)
    for index in range(result.shape[0]):
        midpoint_time = 0.5 * (truth.times[index] + truth.times[index + 1])
        midpoint_state = _midpoint_state(truth, index)
        truth_actuator = ActuatorState(
            thrust=0.5 * (
                truth.actuator_thrust[index]
                + truth.actuator_thrust[index + 1]
            ),
            gimbal_angle=0.5 * (
                truth.actuator_gimbal_angle[index]
                + truth.actuator_gimbal_angle[index + 1]
            ),
        )
        nominal_actuator = ActuatorState(
            thrust=counterfactual.interval_midpoint_thrust[index],
            gimbal_angle=(
                counterfactual.interval_midpoint_gimbal_angle[index]
            ),
        )
        truth_wrench = truth_plant.total_body_wrench(
            midpoint_time, midpoint_state, truth_actuator
        )
        base_wrench = actuator_wrench(
            nominal_actuator, static_truth, geometry
        )
        result[index] = truth_wrench - base_wrench
    return result


def _ridge_quotient_bias(mean, truth, covariance, direction):
    inverse = np.linalg.inv(covariance)
    delta = mean - truth
    coefficient = float(
        direction @ inverse @ delta / (direction @ inverse @ direction)
    )
    quotient = delta - coefficient * direction
    return float(np.sqrt(quotient @ inverse @ quotient))


def _trajectory_errors(trajectories, truth):
    position = np.asarray([value.position for value in trajectories])
    position_mean_error = np.mean(position, axis=0) - truth.position
    rotation_error = np.empty((len(trajectories), truth.times.size, 3))
    for member, trajectory in enumerate(trajectories):
        for index in range(truth.times.size):
            truth_rotation = quaternion_to_matrix(
                truth.orientation_xyzw[index]
            )
            candidate_rotation = quaternion_to_matrix(
                trajectory.orientation_xyzw[index]
            )
            rotation_error[member, index] = rotation_vector_from_matrix(
                truth_rotation.T @ candidate_rotation
            )
    rotation_mean_error = np.mean(rotation_error, axis=0)
    velocity_mean = np.mean(
        np.asarray([value.linear_velocity for value in trajectories]), axis=0
    )
    omega_mean = np.mean(
        np.asarray([value.angular_velocity for value in trajectories]), axis=0
    )
    combined = np.concatenate(
        (position - truth.position[None, :, :], rotation_error), axis=2
    )
    lower = np.percentile(combined, 2.5, axis=0)
    upper = np.percentile(combined, 97.5, axis=0)
    return (
        float(np.sqrt(np.mean(position_mean_error**2))),
        float(np.sqrt(np.mean(rotation_mean_error**2))),
        float(np.sqrt(np.mean((velocity_mean - truth.linear_velocity) ** 2))),
        float(np.sqrt(np.mean((omega_mean - truth.angular_velocity) ** 2))),
        float(np.mean((lower <= 0.0) & (upper >= 0.0))),
    )


def _residual_metrics(
    posterior: WeakConstraintPosterior,
    oracle: np.ndarray,
    static_truth: VehicleParameters,
):
    estimated = np.mean(posterior.residual_wrench_ensemble, axis=0)
    inverse_inertia = np.linalg.inv(static_truth.inertia)

    def acceleration(wrench):
        result = np.empty_like(wrench)
        result[:, :3] = wrench[:, :3] / static_truth.mass
        result[:, 3:] = (inverse_inertia @ wrench[:, 3:].T).T
        return result

    oracle_acceleration = acceleration(oracle)
    estimated_acceleration = acceleration(estimated)
    denominator = float(np.sum(oracle_acceleration**2))
    r_squared = 1.0 - float(
        np.sum((estimated_acceleration - oracle_acceleration) ** 2)
        / denominator
    )
    correlations = []
    for axis in range(6):
        if np.std(oracle_acceleration[:, axis]) > 1.0e-5:
            correlations.append(
                np.corrcoef(
                    oracle_acceleration[:, axis],
                    estimated_acceleration[:, axis],
                )[0, 1]
            )
    lower = np.percentile(
        posterior.residual_wrench_ensemble, 2.5, axis=0
    )
    upper = np.percentile(
        posterior.residual_wrench_ensemble, 97.5, axis=0
    )
    coverage = float(np.mean((oracle >= lower) & (oracle <= upper)))
    return r_squared, float(np.median(correlations)), coverage


def _zero_residual_wrench(_time, _state):
    return np.zeros(6, dtype=float)


def run_phase3_experiment(
    duration: float = 0.8,
    time_step: float = 0.04,
    ensemble_size: Optional[int] = None,
    maximum_iterations: int = 3,
    seed: int = 31,
) -> Phase3Experiment:
    """Run the same model-error episode with strong and IEnKS-Q smoothers."""

    synthetic = run_synthetic_experiment(
        duration=duration,
        time_step=time_step,
        translation_noise=0.003,
        rotation_noise=np.deg2rad(0.20),
        seed=seed + 1000,
    )
    static_truth = _static_truth_without_model_error(
        synthetic.truth_parameters
    )
    # Matched control for Experiment C: same static theta, observation-noise
    # realization, reference and window, but no drag, actuator mismatch or
    # external wrench.  This makes the additional strong-constraint bias due
    # to model error distinguishable from short-window non-identifiability.
    matched_synthetic = run_synthetic_experiment(
        duration=duration,
        time_step=time_step,
        truth_parameters=static_truth,
        truth_actuators=ActuatorParameters(),
        truth_residual_wrench=_zero_residual_wrench,
        translation_noise=0.003,
        rotation_noise=np.deg2rad(0.20),
        seed=seed + 1000,
    )
    oracle_residual = oracle_effective_residual_wrench(synthetic)
    # Experiment C owns its truth generator, so Q is calibrated from the
    # known synthetic model-error RMS rather than adjusted to estimator output.
    # Real-data Q calibration is deliberately deferred to Phase 5.
    wrench_standard_deviation = np.maximum(
        2.0 * np.sqrt(np.mean(oracle_residual**2, axis=0)),
        np.asarray((0.05, 0.05, 0.05, 0.005, 0.005, 0.005)),
    )
    strong_problem = _problem_from_synthetic(synthetic)
    matched_problem = _problem_from_synthetic(matched_synthetic)
    process = GaussMarkovWrenchProcess(
        times=synthetic.observations.times[:-1],
        stationary_standard_deviation=wrench_standard_deviation,
        correlation_time=0.35,
    )
    weak_problem = WeakConstraintProblem(strong_problem, process)
    selected_size = (
        weak_problem.control_dimension + 2
        if ensemble_size is None
        else int(ensemble_size)
    )
    configuration = IEnKSConfig(
        ensemble_size=selected_size,
        maximum_iterations=maximum_iterations,
        seed=seed,
    )
    static_prior = StrongConstraintPrior.grape()
    matched_strong_posterior = StrongConstraintIEnKS(configuration).fit(
        matched_problem, static_prior
    )
    strong_posterior = StrongConstraintIEnKS(configuration).fit(
        strong_problem, static_prior
    )
    weak_posterior = WeakConstraintIEnKSQ(configuration).fit(
        weak_problem,
        WeakConstraintPrior(static_prior, process),
    )
    truth_coordinates = strong_problem.parameter_chart.encode(static_truth)
    prior_parameter_covariance = static_prior.covariance[
        PARAMETER_OFFSET:, PARAMETER_OFFSET:
    ]
    direction = strong_posterior.ridge.expected_direction
    matched_bias = _ridge_quotient_bias(
        np.mean(
            matched_strong_posterior.parameter_ensemble.coordinates,
            axis=0,
        ),
        truth_coordinates,
        prior_parameter_covariance,
        matched_strong_posterior.ridge.expected_direction,
    )
    strong_bias = _ridge_quotient_bias(
        np.mean(strong_posterior.parameter_ensemble.coordinates, axis=0),
        truth_coordinates,
        prior_parameter_covariance,
        direction,
    )
    weak_bias = _ridge_quotient_bias(
        np.mean(weak_posterior.parameter_ensemble.coordinates, axis=0),
        truth_coordinates,
        prior_parameter_covariance,
        direction,
    )
    strong_errors = _trajectory_errors(
        strong_posterior.trajectory_ensemble, synthetic.truth
    )
    weak_errors = _trajectory_errors(
        weak_posterior.trajectory_ensemble, synthetic.truth
    )
    residual_metrics = _residual_metrics(
        weak_posterior, oracle_residual, static_truth
    )
    metrics = Phase3Metrics(
        matched_strong_static_bias=matched_bias,
        strong_static_bias=strong_bias,
        weak_static_bias=weak_bias,
        strong_pose_rmse=strong_errors[0],
        weak_pose_rmse=weak_errors[0],
        strong_rotation_rmse=strong_errors[1],
        weak_rotation_rmse=weak_errors[1],
        strong_velocity_rmse=strong_errors[2],
        weak_velocity_rmse=weak_errors[2],
        strong_omega_rmse=strong_errors[3],
        weak_omega_rmse=weak_errors[3],
        strong_path_coverage=strong_errors[4],
        weak_path_coverage=weak_errors[4],
        residual_acceleration_r_squared=residual_metrics[0],
        residual_excited_channel_correlation=residual_metrics[1],
        residual_component_coverage=residual_metrics[2],
    )
    return Phase3Experiment(
        synthetic=synthetic,
        matched_synthetic=matched_synthetic,
        truth_static_coordinates=truth_coordinates,
        oracle_residual_wrench=oracle_residual,
        wrench_process=process,
        matched_strong_posterior=matched_strong_posterior,
        strong_posterior=strong_posterior,
        weak_posterior=weak_posterior,
        metrics=metrics,
    )


def save_phase3_experiment(path: str, result: Phase3Experiment) -> Path:
    """Save strong/weak comparison and the raw IEnKS-Q member law."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    matched = result.matched_strong_posterior
    strong = result.strong_posterior
    weak = result.weak_posterior
    counterfactual = replay_nominal_actuators_on_truth(result.synthetic)
    np.savez_compressed(
        str(destination),
        schema=np.asarray(("grape-weak-constraint/phase3",)),
        times=result.synthetic.truth.times,
        reference_position=np.asarray(
            [value.position for value in result.synthetic.references]
        ),
        reference_rpy=np.asarray(
            [value.rpy for value in result.synthetic.references]
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
        nominal_position=result.synthetic.nominal.position,
        nominal_orientation_xyzw=(
            result.synthetic.nominal.orientation_xyzw
        ),
        nominal_linear_velocity=result.synthetic.nominal.linear_velocity,
        nominal_angular_velocity=result.synthetic.nominal.angular_velocity,
        nominal_controller_integral=(
            result.synthetic.nominal.controller_integral
        ),
        nominal_commanded_thrust=(
            result.synthetic.nominal.commanded_thrust
        ),
        nominal_commanded_gimbal_angle=(
            result.synthetic.nominal.commanded_gimbal_angle
        ),
        nominal_actuator_thrust=result.synthetic.nominal.actuator_thrust,
        nominal_actuator_gimbal_angle=(
            result.synthetic.nominal.actuator_gimbal_angle
        ),
        nominal_body_wrench=result.synthetic.nominal.body_wrench,
        truth_position=result.synthetic.truth.position,
        truth_orientation_xyzw=result.synthetic.truth.orientation_xyzw,
        truth_linear_velocity=result.synthetic.truth.linear_velocity,
        truth_angular_velocity=result.synthetic.truth.angular_velocity,
        truth_controller_integral=(
            result.synthetic.truth.controller_integral
        ),
        truth_commanded_thrust=result.synthetic.truth.commanded_thrust,
        truth_commanded_gimbal_angle=(
            result.synthetic.truth.commanded_gimbal_angle
        ),
        truth_actuator_thrust=result.synthetic.truth.actuator_thrust,
        truth_actuator_gimbal_angle=(
            result.synthetic.truth.actuator_gimbal_angle
        ),
        truth_body_wrench=result.synthetic.truth.body_wrench,
        truth_static_coordinates=result.truth_static_coordinates,
        counterfactual_commanded_thrust=counterfactual.commanded_thrust,
        counterfactual_commanded_gimbal_angle=(
            counterfactual.commanded_gimbal_angle
        ),
        counterfactual_actuator_thrust=counterfactual.actuator_thrust,
        counterfactual_actuator_gimbal_angle=(
            counterfactual.actuator_gimbal_angle
        ),
        counterfactual_interval_midpoint_thrust=(
            counterfactual.interval_midpoint_thrust
        ),
        counterfactual_interval_midpoint_gimbal_angle=(
            counterfactual.interval_midpoint_gimbal_angle
        ),
        matched_truth_position=result.matched_synthetic.truth.position,
        matched_truth_orientation_xyzw=(
            result.matched_synthetic.truth.orientation_xyzw
        ),
        matched_observations_position=(
            result.matched_synthetic.observations.position
        ),
        matched_observations_orientation_xyzw=(
            result.matched_synthetic.observations.orientation_xyzw
        ),
        oracle_residual_wrench=result.oracle_residual_wrench,
        wrench_stationary_standard_deviation=(
            result.wrench_process.stationary_standard_deviation
        ),
        wrench_correlation_time=np.asarray(
            (result.wrench_process.correlation_time,)
        ),
        matched_strong_parameter_coordinates=(
            matched.parameter_ensemble.coordinates
        ),
        matched_strong_position=np.asarray(
            [value.position for value in matched.trajectory_ensemble]
        ),
        matched_strong_orientation_xyzw=np.asarray(
            [
                value.orientation_xyzw
                for value in matched.trajectory_ensemble
            ]
        ),
        matched_strong_static_bias=np.asarray(
            (result.metrics.matched_strong_static_bias,)
        ),
        strong_static_bias=np.asarray((result.metrics.strong_static_bias,)),
        weak_static_bias=np.asarray((result.metrics.weak_static_bias,)),
        strong_control_ensemble=strong.control_ensemble,
        strong_parameter_coordinates=strong.parameter_ensemble.coordinates,
        strong_position=np.asarray(
            [value.position for value in strong.trajectory_ensemble]
        ),
        strong_orientation_xyzw=np.asarray(
            [value.orientation_xyzw for value in strong.trajectory_ensemble]
        ),
        strong_correction_translation=strong.correction_translation,
        strong_correction_rotation_vector=(
            strong.correction_rotation_vector
        ),
        weak_control_ensemble=weak.control_ensemble,
        weak_parameter_coordinates=weak.parameter_ensemble.coordinates,
        weak_mass=weak.parameter_ensemble.mass,
        weak_inertia=weak.parameter_ensemble.inertia,
        weak_cog_offset=weak.parameter_ensemble.cog_offset,
        weak_force_effectiveness=(
            weak.parameter_ensemble.force_effectiveness
        ),
        weak_torque_effectiveness=(
            weak.parameter_ensemble.torque_effectiveness
        ),
        weak_innovation_ensemble=weak.innovation_ensemble,
        weak_residual_wrench_ensemble=weak.residual_wrench_ensemble,
        weak_position=np.asarray(
            [value.position for value in weak.trajectory_ensemble]
        ),
        weak_orientation_xyzw=np.asarray(
            [value.orientation_xyzw for value in weak.trajectory_ensemble]
        ),
        weak_linear_velocity=np.asarray(
            [value.linear_velocity for value in weak.trajectory_ensemble]
        ),
        weak_angular_velocity=np.asarray(
            [value.angular_velocity for value in weak.trajectory_ensemble]
        ),
        weak_controller_integral=np.asarray(
            [value.controller_integral for value in weak.trajectory_ensemble]
        ),
        weak_commanded_thrust=np.asarray(
            [value.commanded_thrust for value in weak.trajectory_ensemble]
        ),
        weak_commanded_gimbal_angle=np.asarray(
            [value.commanded_gimbal_angle for value in weak.trajectory_ensemble]
        ),
        weak_actuator_thrust=np.asarray(
            [value.actuator_thrust for value in weak.trajectory_ensemble]
        ),
        weak_actuator_gimbal_angle=np.asarray(
            [value.actuator_gimbal_angle for value in weak.trajectory_ensemble]
        ),
        weak_body_wrench=np.asarray(
            [value.body_wrench for value in weak.trajectory_ensemble]
        ),
        weak_correction_translation=weak.correction_translation,
        weak_correction_rotation_vector=weak.correction_rotation_vector,
        weak_ridge_covariance=weak.ridge.covariance,
        weak_iteration_objective=np.asarray(
            [value.objective for value in weak.iterations]
        ),
        weak_iteration_accepted_objective=np.asarray(
            [value.accepted_objective for value in weak.iterations]
        ),
        weak_converged=np.asarray((weak.converged,), dtype=bool),
        weak_termination_reason=np.asarray((weak.termination_reason,)),
    )
    return destination
