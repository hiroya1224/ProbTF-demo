"""Phase-5 weak-constraint assimilation of one continuous real flight."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.controller import GrapeController
from grape_param_estim.dynamics import FullSixDofPlant, simulate_closed_loop
from grape_param_estim.geometry import (
    correction_transform_path,
    quaternion_to_matrix,
    rotation_vector_from_matrix,
)
from grape_param_estim.model_error import KnotGaussMarkovWrenchProcess
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.real_calibration import (
    KnotResolution,
    ModelErrorCalibration,
    calibrate_model_error_from_closed_loop_pose,
    pose_derived_initial_state,
    select_ou_knot_resolution,
)
from grape_param_estim.real_rosbag import (
    DEFAULT_GRAPE_BAG,
    PID_AXIS_NAMES,
    PID_CONFIG_FIELD_NAMES,
    RealFlightEpisode,
    load_grape_rosbag_episode,
)
from grape_param_estim.strong_constraint import (
    PARAMETER_OFFSET,
    IEnKSConfig,
    StrongConstraintPrior,
    StrongConstraintProblem,
)
from grape_param_estim.system import (
    ActuatorParameters,
    ClosedLoopTrajectory,
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
class RealModeDiagnostic:
    """Explicit discrete-mode scope for the selected real experiment."""

    mode_ids: Tuple[str, ...]
    weights: np.ndarray
    selected_mode_id: str
    conditioning_source: str
    pose_mode_comparison_performed: bool

    def __post_init__(self) -> None:
        identifiers = tuple(str(value) for value in self.mode_ids)
        weights = np.asarray(self.weights, dtype=float)
        if (
            not identifiers
            or weights.shape != (len(identifiers),)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.isclose(np.sum(weights), 1.0)
            or self.selected_mode_id not in identifiers
        ):
            raise ValueError("real mode diagnostic is invalid")
        object.__setattr__(self, "mode_ids", identifiers)
        object.__setattr__(self, "weights", weights.copy())


@dataclass(frozen=True)
class RealAssimilationMetrics:
    nominal_position_rmse: float
    posterior_center_position_rmse: float
    nominal_rotation_rmse: float
    posterior_center_rotation_rmse: float
    observed_pose_component_coverage: float
    posterior_expected_ridge_variance_ratio: float


@dataclass(frozen=True)
class RealAssimilationResult:
    episode: RealFlightEpisode
    initial_state_anchor: RigidBodyState
    actuator_parameters: ActuatorParameters
    nominal_parameters: VehicleParameters
    nominal_trajectory: ClosedLoopTrajectory
    calibration: ModelErrorCalibration
    knot_resolution: KnotResolution
    wrench_process: KnotGaussMarkovWrenchProcess
    prior: WeakConstraintPrior
    posterior: WeakConstraintPosterior
    mode_diagnostic: RealModeDiagnostic
    metrics: RealAssimilationMetrics


def build_real_strong_problem(
    episode: RealFlightEpisode,
    actuator_parameters: Optional[ActuatorParameters] = None,
    nominal_parameters: Optional[VehicleParameters] = None,
):
    """Construct the real full-loop problem and its no-Q nominal replay."""

    selected_actuators = actuator_parameters or ActuatorParameters()
    parameters = nominal_parameters or VehicleParameters.nominal()
    geometry = GrapeGeometry.grape()
    initial_state = pose_derived_initial_state(
        episode.observations.times,
        episode.observations.position,
        episode.observations.orientation_xyzw,
    )
    controller = GrapeController(
        episode.controller_configuration,
        parameters,
        geometry,
        articulated_model=GrapeArticulatedModel(),
    )
    nominal = simulate_closed_loop(
        times=episode.observations.times,
        references=episode.references,
        initial_state=initial_state,
        initial_controller_state=episode.initial_controller_state,
        controller=controller,
        plant=FullSixDofPlant(parameters, geometry),
        actuator_parameters=selected_actuators,
        initial_actuator_state=episode.initial_actuator_state,
    )
    problem = StrongConstraintProblem(
        references=episode.references,
        observations=episode.observations,
        nominal_trajectory=nominal,
        initial_state_anchor=initial_state,
        initial_controller_anchor=episode.initial_controller_state,
        controller_configuration=episode.controller_configuration,
        controller_parameters=parameters,
        geometry=geometry,
        actuator_parameters=selected_actuators,
        parameter_chart=VehicleParameterChart(parameters),
        initial_actuator_state=episode.initial_actuator_state,
    )
    return problem, initial_state, nominal, selected_actuators, parameters


def _pose_rmse(trajectory, observations):
    position = float(np.sqrt(np.mean(
        (trajectory.position - observations.position) ** 2
    )))
    rotation = []
    for index in range(observations.times.size):
        observed = quaternion_to_matrix(
            observations.orientation_xyzw[index]
        )
        candidate = quaternion_to_matrix(
            trajectory.orientation_xyzw[index]
        )
        rotation.append(rotation_vector_from_matrix(observed.T @ candidate))
    return position, float(np.sqrt(np.mean(np.asarray(rotation) ** 2)))


def _observed_pose_coverage(posterior, observations):
    members = posterior.trajectory_ensemble
    translation = np.asarray([value.position for value in members])
    translation -= observations.position[None, :, :]
    rotation = np.empty_like(translation)
    for member, trajectory in enumerate(members):
        for index in range(observations.times.size):
            observed = quaternion_to_matrix(
                observations.orientation_xyzw[index]
            )
            candidate = quaternion_to_matrix(
                trajectory.orientation_xyzw[index]
            )
            rotation[member, index] = rotation_vector_from_matrix(
                observed.T @ candidate
            )
    residual = np.concatenate((translation, rotation), axis=2)
    lower = np.percentile(residual, 2.5, axis=0)
    upper = np.percentile(residual, 97.5, axis=0)
    return float(np.mean((lower <= 0.0) & (upper >= 0.0)))


def assimilate_real_episode(
    episode: RealFlightEpisode,
    maximum_knots: Optional[int] = 12,
    bridge_standard_deviation_fraction: float = 0.5,
    ensemble_size: Optional[int] = None,
    maximum_iterations: int = 1,
    seed: int = 53,
    actuator_parameters: Optional[ActuatorParameters] = None,
    nominal_parameters: Optional[VehicleParameters] = None,
) -> RealAssimilationResult:
    """Run sparse IEnKS-Q on the complete selected real episode."""

    (
        strong_problem,
        initial_state,
        nominal,
        selected_actuators,
        parameters,
    ) = build_real_strong_problem(
        episode, actuator_parameters, nominal_parameters
    )
    calibration = calibrate_model_error_from_closed_loop_pose(
        episode.observations.times,
        episode.observations.position,
        episode.observations.orientation_xyzw,
        episode.references,
        episode.controller_configuration,
        episode.initial_controller_state,
        episode.initial_actuator_state,
        selected_actuators,
        parameters,
        strong_problem.geometry,
    )
    resolution = select_ou_knot_resolution(
        episode.observations.times,
        calibration.correlation_time,
        bridge_standard_deviation_fraction,
        maximum_knots,
    )
    process = KnotGaussMarkovWrenchProcess(
        integration_times=episode.observations.times,
        knot_indices=resolution.knot_indices,
        stationary_standard_deviation=(
            calibration.stationary_standard_deviation
        ),
        correlation_time=calibration.correlation_time,
    )
    weak_problem = WeakConstraintProblem(strong_problem, process)
    static_prior = StrongConstraintPrior.grape()
    prior = WeakConstraintPrior(static_prior, process)
    members = (
        weak_problem.control_dimension + 2
        if ensemble_size is None
        else int(ensemble_size)
    )
    posterior = WeakConstraintIEnKSQ(
        IEnKSConfig(
            ensemble_size=members,
            maximum_iterations=maximum_iterations,
            seed=seed,
        )
    ).fit(weak_problem, prior)
    nominal_position, nominal_rotation = _pose_rmse(
        nominal, episode.observations
    )
    posterior_position, posterior_rotation = _pose_rmse(
        posterior.center_trajectory, episode.observations
    )
    ridge_direction = posterior.ridge.expected_direction
    prior_parameter_covariance = static_prior.covariance[
        PARAMETER_OFFSET:, PARAMETER_OFFSET:
    ]
    prior_ridge_variance = float(
        ridge_direction @ prior_parameter_covariance @ ridge_direction
    )
    metrics = RealAssimilationMetrics(
        nominal_position_rmse=nominal_position,
        posterior_center_position_rmse=posterior_position,
        nominal_rotation_rmse=nominal_rotation,
        posterior_center_rotation_rmse=posterior_rotation,
        observed_pose_component_coverage=_observed_pose_coverage(
            posterior, episode.observations
        ),
        posterior_expected_ridge_variance_ratio=float(
            posterior.ridge.expected_variance / prior_ridge_variance
        ),
    )
    # This bag records one audited controller/actuator channel convention.
    # No alternative discrete plant mode is mixed into this posterior.
    mode = RealModeDiagnostic(
        mode_ids=("actuator_wiring_nominal",),
        weights=np.asarray((1.0,)),
        selected_mode_id="actuator_wiring_nominal",
        conditioning_source=(
            "recorded canonical FourAxisCommand and gimbal joint channel order"
        ),
        pose_mode_comparison_performed=False,
    )
    return RealAssimilationResult(
        episode=episode,
        initial_state_anchor=initial_state,
        actuator_parameters=selected_actuators,
        nominal_parameters=parameters,
        nominal_trajectory=nominal,
        calibration=calibration,
        knot_resolution=resolution,
        wrench_process=process,
        prior=prior,
        posterior=posterior,
        mode_diagnostic=mode,
        metrics=metrics,
    )


def run_real_rosbag_assimilation(
    bag_path: str = DEFAULT_GRAPE_BAG,
    sample_period: float = 0.04,
    episode_index: int = 0,
    start_local: Optional[float] = None,
    end_local: Optional[float] = None,
    window_state: Optional[int] = 5,
    maximum_knots: Optional[int] = 12,
    ensemble_size: Optional[int] = None,
    maximum_iterations: int = 1,
    seed: int = 53,
    compute_sha256: bool = True,
) -> RealAssimilationResult:
    episode = load_grape_rosbag_episode(
        bag_path,
        sample_period=sample_period,
        episode_index=episode_index,
        start_local=start_local,
        end_local=end_local,
        window_state=window_state,
        compute_sha256=compute_sha256,
    )
    return assimilate_real_episode(
        episode,
        maximum_knots=maximum_knots,
        ensemble_size=ensemble_size,
        maximum_iterations=maximum_iterations,
        seed=seed,
    )


def _trajectory_field(trajectories, field):
    return np.asarray([getattr(value, field) for value in trajectories])


def save_real_assimilation(
    path: str, result: RealAssimilationResult
) -> Path:
    """Save the raw member law and all member-aligned Phase-5 paths."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    episode = result.episode
    posterior = result.posterior
    process = result.wrench_process
    trajectories = posterior.trajectory_ensemble
    prior_trajectories = posterior.prior_trajectory_ensemble
    references = episode.references
    observed_correction_translation, observed_correction_rotation = (
        correction_transform_path(
            result.nominal_trajectory.position,
            result.nominal_trajectory.orientation_xyzw,
            episode.observations.position,
            episode.observations.orientation_xyzw,
        )
    )
    knot_wrench = np.asarray(
        [process.decode_knots(value) for value in posterior.innovation_ensemble]
    )
    iteration = posterior.iterations
    configuration = episode.controller_configuration
    provenance = episode.provenance
    np.savez_compressed(
        str(destination),
        schema=np.asarray(("grape-weak-constraint/phase5-real-assimilation",)),
        member_id=np.arange(posterior.control_ensemble.shape[0], dtype=np.int64),
        control_ensemble=posterior.control_ensemble,
        prior_control_ensemble=posterior.prior_control_ensemble,
        center_control=posterior.center_control,
        times=episode.observations.times,
        record_times=episode.record_times,
        bag_path=np.asarray((provenance.bag_path,)),
        bag_sha256=np.asarray((provenance.bag_sha256,)),
        time_basis=np.asarray((provenance.time_basis,)),
        window_local=np.asarray(
            (episode.window_start_local_time, episode.window_end_local_time)
        ),
        window_record_time=np.asarray(
            (episode.window_start_record_time, episode.window_end_record_time)
        ),
        observations_position=episode.observations.position,
        observations_orientation_xyzw=episode.observations.orientation_xyzw,
        observation_translation_covariance=(
            episode.observations.translation_covariance
        ),
        observation_rotation_covariance=(
            episode.observations.rotation_covariance
        ),
        reference_position=np.asarray([value.position for value in references]),
        reference_linear_velocity=np.asarray(
            [value.linear_velocity for value in references]
        ),
        reference_linear_acceleration=np.asarray(
            [value.linear_acceleration for value in references]
        ),
        reference_rpy=np.asarray([value.rpy for value in references]),
        reference_angular_velocity=np.asarray(
            [value.angular_velocity for value in references]
        ),
        reference_angular_acceleration=np.asarray(
            [value.angular_acceleration for value in references]
        ),
        initial_state=result.initial_state_anchor.as_vector(),
        initial_controller_integral=(
            episode.initial_controller_state.integral_error
        ),
        initial_controller_roll_pitch_active=np.asarray(
            (episode.initial_controller_state.roll_pitch_integration_active,),
            dtype=bool,
        ),
        initial_actuator_thrust=episode.initial_actuator_state.thrust,
        initial_actuator_gimbal_angle=(
            episode.initial_actuator_state.gimbal_angle
        ),
        controller_pid_axis_names=np.asarray(PID_AXIS_NAMES),
        controller_pid_field_names=np.asarray(PID_CONFIG_FIELD_NAMES),
        controller_pid_configuration=np.asarray(
            [
                [getattr(pid, name) for name in PID_CONFIG_FIELD_NAMES]
                for pid in configuration.pid
            ]
        ),
        controller_xy_control_mode=np.asarray(
            (configuration.xy_control_mode,)
        ),
        controller_need_yaw_d_control=np.asarray(
            (configuration.need_yaw_d_control,), dtype=bool
        ),
        controller_start_roll_pitch_integration_height=np.asarray(
            (configuration.start_roll_pitch_integration_height,)
        ),
        controller_initial_height=np.asarray(
            (configuration.initial_height,)
        ),
        controller_source_compatible_gyro_term=np.asarray(
            (configuration.source_compatible_gyro_term,), dtype=bool
        ),
        controller_snapshot_groups=np.asarray(
            episode.controller_snapshot.groups
        ),
        controller_snapshot_record_times=(
            episode.controller_snapshot.record_times
        ),
        controller_snapshot_gains=episode.controller_snapshot.gains,
        controller_snapshot_pid_control_flags=(
            episode.controller_snapshot.pid_control_flags
        ),
        controller_snapshot_source_kinds=np.asarray(
            episode.controller_snapshot.source_kinds
        ),
        provenance_bag_path=np.asarray((provenance.bag_path,)),
        provenance_bag_sha256=np.asarray((provenance.bag_sha256,)),
        provenance_bag_size_bytes=np.asarray(
            (provenance.bag_size_bytes,), dtype=np.int64
        ),
        provenance_bag_record_start=np.asarray(
            (provenance.bag_record_start,)
        ),
        provenance_bag_record_end=np.asarray(
            (provenance.bag_record_end,)
        ),
        provenance_time_basis=np.asarray((provenance.time_basis,)),
        provenance_requested_window_start=np.asarray(
            (provenance.requested_window_start,)
        ),
        provenance_requested_window_end=np.asarray(
            (provenance.requested_window_end,)
        ),
        provenance_source_available_start=np.asarray(
            (provenance.source_available_start,)
        ),
        provenance_source_available_end=np.asarray(
            (provenance.source_available_end,)
        ),
        provenance_resample_period=np.asarray(
            (provenance.resample_period,)
        ),
        provenance_selected_flight_state=np.asarray(
            (provenance.selected_flight_state,), dtype=np.int64
        ),
        provenance_flight_transition_record_times=(
            provenance.flight_transition_record_times
        ),
        provenance_flight_transition_states=(
            provenance.flight_transition_states
        ),
        provenance_static_window_start=np.asarray(
            (provenance.static_window_start,)
        ),
        provenance_static_window_end=np.asarray(
            (provenance.static_window_end,)
        ),
        provenance_static_sample_counts=np.asarray(
            (
                provenance.static_position_samples,
                provenance.static_position_inliers,
                provenance.static_orientation_samples,
                provenance.static_orientation_inliers,
            ),
            dtype=np.int64,
        ),
        provenance_static_position_center=(
            provenance.static_position_center
        ),
        provenance_static_orientation_xyzw=(
            provenance.static_orientation_xyzw
        ),
        provenance_covariance_outlier_threshold=np.asarray(
            (provenance.covariance_outlier_threshold,)
        ),
        provenance_covariance_eigenvalue_floor=np.asarray(
            (provenance.covariance_eigenvalue_floor,)
        ),
        provenance_controller_state_anchor_record_time=np.asarray(
            (provenance.controller_state_anchor_record_time,)
        ),
        provenance_joint_anchor_record_time=np.asarray(
            (provenance.joint_anchor_record_time,)
        ),
        provenance_thrust_anchor_record_time=np.asarray(
            (provenance.thrust_anchor_record_time,)
        ),
        provenance_thrust_anchor_kind=np.asarray(
            (provenance.thrust_anchor_kind,)
        ),
        provenance_reference_acceleration_kind=np.asarray(
            (provenance.reference_acceleration_kind,)
        ),
        provenance_controller_static_source=np.asarray(
            (provenance.controller_static_source,)
        ),
        provenance_controller_source_revision=np.asarray(
            (provenance.controller_source_revision,)
        ),
        provenance_topic_names=np.asarray(provenance.topic_names),
        provenance_topic_types=np.asarray(provenance.topic_types),
        nominal_parameter_mass=np.asarray((result.nominal_parameters.mass,)),
        nominal_parameter_inertia=result.nominal_parameters.inertia,
        nominal_parameter_cog_offset=result.nominal_parameters.cog_offset,
        nominal_parameter_force_effectiveness=(
            result.nominal_parameters.force_effectiveness
        ),
        nominal_parameter_torque_effectiveness=(
            result.nominal_parameters.torque_effectiveness
        ),
        nominal_parameter_linear_drag=result.nominal_parameters.linear_drag,
        nominal_parameter_angular_drag=result.nominal_parameters.angular_drag,
        actuator_parameter_names=np.asarray(
            tuple(result.actuator_parameters.__dict__)
        ),
        actuator_parameter_values=np.asarray(
            tuple(result.actuator_parameters.__dict__.values())
        ),
        nominal_position=result.nominal_trajectory.position,
        nominal_orientation_xyzw=result.nominal_trajectory.orientation_xyzw,
        nominal_linear_velocity=result.nominal_trajectory.linear_velocity,
        nominal_angular_velocity=result.nominal_trajectory.angular_velocity,
        nominal_controller_integral=(
            result.nominal_trajectory.controller_integral
        ),
        nominal_commanded_thrust=result.nominal_trajectory.commanded_thrust,
        nominal_commanded_gimbal_angle=(
            result.nominal_trajectory.commanded_gimbal_angle
        ),
        nominal_actuator_thrust=result.nominal_trajectory.actuator_thrust,
        nominal_actuator_gimbal_angle=(
            result.nominal_trajectory.actuator_gimbal_angle
        ),
        nominal_body_wrench=result.nominal_trajectory.body_wrench,
        prior_position=_trajectory_field(prior_trajectories, "position"),
        prior_orientation_xyzw=_trajectory_field(
            prior_trajectories, "orientation_xyzw"
        ),
        posterior_position=_trajectory_field(trajectories, "position"),
        posterior_orientation_xyzw=_trajectory_field(
            trajectories, "orientation_xyzw"
        ),
        posterior_linear_velocity=_trajectory_field(
            trajectories, "linear_velocity"
        ),
        posterior_angular_velocity=_trajectory_field(
            trajectories, "angular_velocity"
        ),
        posterior_controller_integral=_trajectory_field(
            trajectories, "controller_integral"
        ),
        posterior_commanded_thrust=_trajectory_field(
            trajectories, "commanded_thrust"
        ),
        posterior_commanded_gimbal_angle=_trajectory_field(
            trajectories, "commanded_gimbal_angle"
        ),
        posterior_actuator_thrust=_trajectory_field(
            trajectories, "actuator_thrust"
        ),
        posterior_actuator_gimbal_angle=_trajectory_field(
            trajectories, "actuator_gimbal_angle"
        ),
        posterior_body_wrench=_trajectory_field(trajectories, "body_wrench"),
        center_position=posterior.center_trajectory.position,
        center_orientation_xyzw=posterior.center_trajectory.orientation_xyzw,
        parameter_coordinates=posterior.parameter_ensemble.coordinates,
        parameter_mass=posterior.parameter_ensemble.mass,
        parameter_inertia=posterior.parameter_ensemble.inertia,
        parameter_cog_offset=posterior.parameter_ensemble.cog_offset,
        parameter_force_effectiveness=(
            posterior.parameter_ensemble.force_effectiveness
        ),
        parameter_torque_effectiveness=(
            posterior.parameter_ensemble.torque_effectiveness
        ),
        innovation_ensemble=posterior.innovation_ensemble,
        residual_wrench_knot=knot_wrench,
        residual_wrench_interval=posterior.residual_wrench_ensemble,
        center_residual_wrench_interval=posterior.center_residual_wrench,
        knot_indices=process.knot_indices,
        knot_times=process.knot_times,
        interval_average_interpolation_matrix=process.interpolation_matrix,
        q_stationary_standard_deviation=(
            result.calibration.stationary_standard_deviation
        ),
        q_calibration_pilot_location=result.calibration.pilot_location,
        q_correlation_time=np.asarray((result.calibration.correlation_time,)),
        q_calibration_method=np.asarray((result.calibration.method,)),
        q_calibration_proxy_wrench=result.calibration.proxy_wrench,
        q_calibration_valid_mask=result.calibration.valid_mask,
        q_calibration_derivative_window_samples=np.asarray(
            (result.calibration.derivative_window_samples,), dtype=np.int64
        ),
        q_required_knot_count=np.asarray(
            (result.knot_resolution.required_knot_count,), dtype=np.int64
        ),
        q_maximum_bridge_gap=np.asarray(
            (result.knot_resolution.maximum_bridge_gap,)
        ),
        q_achieved_maximum_gap=np.asarray(
            (result.knot_resolution.achieved_maximum_gap,)
        ),
        q_bridge_standard_deviation_fraction=np.asarray(
            (result.knot_resolution.bridge_standard_deviation_fraction,)
        ),
        q_resolution_sufficient=np.asarray(
            (result.knot_resolution.resolution_sufficient,), dtype=bool
        ),
        correction_translation=posterior.correction_translation,
        correction_rotation_vector=posterior.correction_rotation_vector,
        observed_correction_translation=observed_correction_translation,
        observed_correction_rotation_vector=observed_correction_rotation,
        ridge_covariance=posterior.ridge.covariance,
        ridge_eigenvalues=posterior.ridge.eigenvalues,
        ridge_eigenvectors=posterior.ridge.eigenvectors,
        ridge_expected_direction=posterior.ridge.expected_direction,
        ridge_expected_variance=np.asarray(
            (posterior.ridge.expected_variance,)
        ),
        mode_ids=np.asarray(result.mode_diagnostic.mode_ids),
        mode_weights=result.mode_diagnostic.weights,
        selected_mode_id=np.asarray(
            (result.mode_diagnostic.selected_mode_id,)
        ),
        mode_conditioning_source=np.asarray(
            (result.mode_diagnostic.conditioning_source,)
        ),
        pose_mode_comparison_performed=np.asarray(
            (result.mode_diagnostic.pose_mode_comparison_performed,),
            dtype=bool,
        ),
        iteration_index=np.asarray(
            [value.iteration for value in iteration], dtype=np.int64
        ),
        iteration_objective=np.asarray(
            [value.objective for value in iteration]
        ),
        iteration_accepted_objective=np.asarray(
            [value.accepted_objective for value in iteration]
        ),
        iteration_gradient_norm=np.asarray(
            [value.gradient_norm for value in iteration]
        ),
        iteration_step_norm=np.asarray(
            [value.step_norm for value in iteration]
        ),
        iteration_accepted_fraction=np.asarray(
            [value.accepted_fraction for value in iteration]
        ),
        converged=np.asarray((posterior.converged,), dtype=bool),
        termination_reason=np.asarray((posterior.termination_reason,)),
        nominal_position_rmse=np.asarray(
            (result.metrics.nominal_position_rmse,)
        ),
        posterior_center_position_rmse=np.asarray(
            (result.metrics.posterior_center_position_rmse,)
        ),
        nominal_rotation_rmse=np.asarray(
            (result.metrics.nominal_rotation_rmse,)
        ),
        posterior_center_rotation_rmse=np.asarray(
            (result.metrics.posterior_center_rotation_rmse,)
        ),
        observed_pose_component_coverage=np.asarray(
            (result.metrics.observed_pose_component_coverage,)
        ),
        posterior_expected_ridge_variance_ratio=np.asarray(
            (result.metrics.posterior_expected_ridge_variance_ratio,)
        ),
    )
    return destination


__all__ = [
    "RealAssimilationMetrics",
    "RealAssimilationResult",
    "RealModeDiagnostic",
    "assimilate_real_episode",
    "build_real_strong_problem",
    "run_real_rosbag_assimilation",
    "save_real_assimilation",
]
