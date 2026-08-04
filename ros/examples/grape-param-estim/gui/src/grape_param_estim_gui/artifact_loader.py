"""Strict Qt-free views of batch-estimation artifacts.

The backend owns the on-disk contract.  The GUI accepts only a completed
``grape-param-estim/batch-estimation-run/v2`` directory validated by
``grape_param_estim.batch_artifact``.  This module converts that detached,
pickle-free bundle into immutable display models; it does not implement an
old assimilation or staged-artifact compatibility path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .project_io import (
    PROJECT_ARTIFACT_LOADER_ID,
    PROJECT_ARTIFACT_LOADER_VERSION,
)

try:
    from grape_param_estim import artifact_io
except ImportError as error:  # pragma: no cover - exercised by GUI startup
    artifact_io = None  # type: ignore[assignment]
    _INSPECTION_BACKEND_IMPORT_ERROR = error
else:
    _INSPECTION_BACKEND_IMPORT_ERROR = None

try:
    from grape_param_estim import batch_artifact as batch_artifact_io
except ImportError as error:  # pragma: no cover - exercised by GUI startup
    batch_artifact_io = None  # type: ignore[assignment]
    _BATCH_BACKEND_IMPORT_ERROR = error
else:
    _BATCH_BACKEND_IMPORT_ERROR = None

try:
    from grape_param_estim.pid import artifact as pid_artifact_io
except ImportError as error:  # pragma: no cover - exercised by GUI startup
    pid_artifact_io = None  # type: ignore[assignment]
    _PID_BACKEND_IMPORT_ERROR = error
else:
    _PID_BACKEND_IMPORT_ERROR = None


GUI_ARTIFACT_LOADER_ID = PROJECT_ARTIFACT_LOADER_ID
GUI_ARTIFACT_LOADER_VERSION = PROJECT_ARTIFACT_LOADER_VERSION


class GuiArtifactError(ValueError):
    """An artifact failed the backend contract or cannot be displayed."""


def _inspection_backend() -> Any:
    if artifact_io is None:
        raise GuiArtifactError(
            "grape_param_estim.artifact_io is unavailable; start the GUI "
            "from the package launcher or add the estimator src directory "
            "to PYTHONPATH"
        ) from _INSPECTION_BACKEND_IMPORT_ERROR
    return artifact_io


def _batch_backend() -> Any:
    if batch_artifact_io is None:
        raise GuiArtifactError(
            "grape_param_estim.batch_artifact is unavailable; start the GUI "
            "from the package launcher or add the estimator src directory "
            "to PYTHONPATH"
        ) from _BATCH_BACKEND_IMPORT_ERROR
    return batch_artifact_io


def _pid_backend() -> Any:
    if pid_artifact_io is None:
        raise GuiArtifactError(
            "grape_param_estim.pid.artifact is unavailable; start the GUI "
            "from the package launcher or add the estimator src directory "
            "to PYTHONPATH"
        ) from _PID_BACKEND_IMPORT_ERROR
    return pid_artifact_io


def _array(value: Any) -> np.ndarray:
    return np.asarray(value)


def _text_array(value: Any) -> np.ndarray:
    """Return display-safe text without changing backend ID alignment."""

    array = np.asarray(value).reshape(-1)
    result = np.asarray(
        [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in array.tolist()
        ]
    )
    result.setflags(write=False)
    return result


def _named_values(
    names: np.ndarray, values: np.ndarray
) -> Mapping[str, float]:
    return {
        str(name): float(value)
        for name, value in zip(_text_array(names), np.asarray(values))
    }


def _quaternion_xyzw_to_rpy(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=float)
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return np.stack((roll, pitch, yaw), axis=-1)


@dataclass(frozen=True)
class FlightResult:
    """Inspection preview data available before estimation."""

    bag_id: str
    time: np.ndarray
    record_time: np.ndarray
    reference_position: np.ndarray
    reference_rpy: np.ndarray
    observed_position: np.ndarray
    observed_orientation_xyzw: np.ndarray
    flight_state: np.ndarray | None
    provenance: Mapping[str, Any]

    @property
    def sample_count(self) -> int:
        return int(self.time.size)

    @property
    def duration(self) -> float:
        if self.time.size < 2:
            return 0.0
        return float(self.time[-1] - self.time[0])

    @property
    def observed_rpy(self) -> np.ndarray:
        return _quaternion_xyzw_to_rpy(self.observed_orientation_xyzw)


@dataclass(frozen=True)
class InspectionArtifact:
    root: Path
    manifest: Mapping[str, Any]
    inspections: Mapping[str, Mapping[str, Any]]
    previews: Mapping[str, FlightResult]


@dataclass(frozen=True)
class StaticParameterMap:
    parameter_coordinate: np.ndarray
    mass: float
    inertia: np.ndarray
    cog: np.ndarray
    force_effectiveness: np.ndarray
    torque_effectiveness: np.ndarray
    delay: float
    q_diagonal: np.ndarray
    objective_components: Mapping[str, float]
    prior_objective: float
    likelihood_objective: float
    bag_objective: Mapping[str, float]


@dataclass(frozen=True)
class QEmHistory:
    iteration: np.ndarray
    input_q: np.ndarray
    target_q: np.ndarray
    accepted_q: np.ndarray
    alpha: np.ndarray
    log_q_change: np.ndarray
    map_objective: np.ndarray
    approximate_marginal_objective: np.ndarray
    delay: np.ndarray
    accepted: np.ndarray
    reason: np.ndarray
    floor_activation: np.ndarray
    expected_residual_second_moment: np.ndarray
    map_residual_second_moment: np.ndarray
    covariance_correction: np.ndarray


@dataclass(frozen=True)
class LaplaceApproximation:
    reduced_likelihood_hessian: np.ndarray
    reduced_prior_information: np.ndarray
    reduced_posterior_hessian: np.ndarray
    fixed_delay_conditional_static_covariance: np.ndarray
    static_covariance_conditioning: str
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    effective_rank: int
    exact_ridge_direction: np.ndarray
    ridge_alignment: float
    condition_number: float
    delay_profile_available: bool
    delay_profile_grid: np.ndarray
    delay_profile_objective: np.ndarray
    delay_profile_approximate_marginal_objective: np.ndarray
    delay_profile_static_coordinate: np.ndarray
    delay_local_geometry_valid: bool
    delay_local_geometry_method: str
    delay_local_geometry_reason: str
    delay_profile_curvature: float | None
    delay_uncertainty_source: str
    joint_delay_marginal_standard_deviation: float | None
    fallback_delay_prior_standard_deviation: float | None
    mcmc_delay_proposal_scale_seconds: float | None
    parameter_delay_cross_covariance: np.ndarray | None
    joint_parameter_delay_information: np.ndarray | None
    joint_parameter_delay_covariance: np.ndarray | None
    joint_static_parameter_marginal_covariance: np.ndarray | None
    mcmc_quadratic_surrogate_method: str


@dataclass(frozen=True)
class StaticParameterSample:
    sample_id: str
    chain_id: str
    draw_index: int
    parameter_coordinate: np.ndarray
    mass: float
    inertia: np.ndarray
    cog: np.ndarray
    force_effectiveness: np.ndarray
    torque_effectiveness: np.ndarray
    delay: float
    log_posterior: float
    log_likelihood_approximation: float
    log_determinant_term: float
    accepted_kernel: str
    source_mode_id: str


@dataclass(frozen=True)
class McmcPosterior:
    sample_id: np.ndarray
    chain_id: np.ndarray
    draw_index: np.ndarray
    parameter_coordinate: np.ndarray
    mass: np.ndarray
    inertia: np.ndarray
    cog: np.ndarray
    force_effectiveness: np.ndarray
    torque_effectiveness: np.ndarray
    delay: np.ndarray
    log_posterior: np.ndarray
    log_likelihood_approximation: np.ndarray
    log_determinant_term: np.ndarray
    accepted_kernel: np.ndarray
    source_mode_id: np.ndarray

    @property
    def size(self) -> int:
        return int(self.sample_id.size)

    @property
    def equal_weights(self) -> np.ndarray:
        if self.size == 0:
            return np.empty((0,), dtype=float)
        return np.full((self.size,), 1.0 / float(self.size), dtype=float)

    def index_of(self, sample_id: str) -> int:
        matches = np.flatnonzero(self.sample_id == str(sample_id))
        if matches.size != 1:
            raise KeyError("unknown MCMC sample {!r}".format(sample_id))
        return int(matches[0])

    def sample(self, sample_id: str) -> StaticParameterSample:
        index = self.index_of(sample_id)
        return StaticParameterSample(
            sample_id=str(self.sample_id[index]),
            chain_id=str(self.chain_id[index]),
            draw_index=int(self.draw_index[index]),
            parameter_coordinate=self.parameter_coordinate[index],
            mass=float(self.mass[index]),
            inertia=self.inertia[index],
            cog=self.cog[index],
            force_effectiveness=self.force_effectiveness[index],
            torque_effectiveness=self.torque_effectiveness[index],
            delay=float(self.delay[index]),
            log_posterior=float(self.log_posterior[index]),
            log_likelihood_approximation=float(
                self.log_likelihood_approximation[index]
            ),
            log_determinant_term=float(self.log_determinant_term[index]),
            accepted_kernel=str(self.accepted_kernel[index]),
            source_mode_id=str(self.source_mode_id[index]),
        )


@dataclass(frozen=True)
class ReferenceTrajectory:
    time: np.ndarray
    record_time: np.ndarray
    position: np.ndarray
    linear_velocity: np.ndarray
    linear_acceleration: np.ndarray
    rpy: np.ndarray
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray


@dataclass(frozen=True)
class VectorObservation:
    time: np.ndarray
    record_time: np.ndarray
    value: np.ndarray
    valid: np.ndarray
    covariance: np.ndarray
    covariance_valid: np.ndarray


@dataclass(frozen=True)
class PoseObservation:
    time: np.ndarray
    record_time: np.ndarray
    position: np.ndarray
    orientation_xyzw: np.ndarray
    valid: np.ndarray
    covariance: np.ndarray
    covariance_valid: np.ndarray

    @property
    def rpy(self) -> np.ndarray:
        return _quaternion_xyzw_to_rpy(self.orientation_xyzw)


@dataclass(frozen=True)
class StateTrajectory:
    position: np.ndarray
    orientation_xyzw: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    controller_integral: np.ndarray
    actuator_thrust: np.ndarray
    actuator_gimbal: np.ndarray

    @property
    def rpy(self) -> np.ndarray:
        return _quaternion_xyzw_to_rpy(self.orientation_xyzw)


@dataclass(frozen=True)
class PoseTrajectory:
    """One sensor-pose path with an explicit finite-prefix mask."""

    position: np.ndarray
    orientation_xyzw: np.ndarray
    valid: np.ndarray

    @property
    def rpy(self) -> np.ndarray:
        return _quaternion_xyzw_to_rpy(self.orientation_xyzw)


@dataclass(frozen=True)
class BagEstimationResult:
    bag_id: str
    knot_time: np.ndarray
    knot_record_time: np.ndarray
    reference: ReferenceTrajectory
    pose: PoseObservation
    velocity: VectorObservation
    gyro: VectorObservation
    accelerometer: VectorObservation
    thrust_command: VectorObservation
    gimbal_command: VectorObservation
    gimbal_observation: VectorObservation
    controller_integral_observation: VectorObservation
    nominal: StateTrajectory
    map_trajectory: StateTrajectory
    initial_parameter_rollout: PoseTrajectory
    estimated_parameter_rollout: PoseTrajectory
    initial_parameter_rollout_correction_translation: np.ndarray
    initial_parameter_rollout_correction_rotation_vector: np.ndarray
    initial_parameter_rollout_correction_valid: np.ndarray
    estimated_parameter_rollout_correction_translation: np.ndarray
    estimated_parameter_rollout_correction_rotation_vector: np.ndarray
    estimated_parameter_rollout_correction_valid: np.ndarray
    map_dynamics_residual: np.ndarray
    map_dynamics_residual_valid: np.ndarray
    factor_residual_history: Mapping[str, np.ndarray]
    factor_normalized_residual_history: Mapping[str, np.ndarray]
    objective_components: Mapping[str, float]
    numerical_diagnostics: Mapping[str, float]
    sensor_contract: Mapping[str, Any]
    observation_factors: Mapping[str, Any]

    @property
    def sample_count(self) -> int:
        return int(self.knot_time.size)

    @property
    def duration(self) -> float:
        return float(self.knot_time[-1] - self.knot_time[0])


@dataclass(frozen=True)
class ConditionalTrajectory:
    sample_id: str
    knot_time: np.ndarray
    state: StateTrajectory
    recorded_control_rollout: PoseTrajectory
    observed_relative_correction_translation: np.ndarray
    observed_relative_correction_rotation_vector: np.ndarray
    observed_relative_correction_valid: np.ndarray
    dynamics_residual: np.ndarray
    dynamics_residual_valid: np.ndarray
    conditional_objective: float


@dataclass(frozen=True)
class SelectedTrajectorySet:
    bag_id: str
    sample_id: np.ndarray
    knot_time: np.ndarray
    conditional_position: np.ndarray
    conditional_orientation_xyzw: np.ndarray
    conditional_linear_velocity: np.ndarray
    conditional_angular_velocity: np.ndarray
    conditional_controller_integral: np.ndarray
    conditional_actuator_thrust: np.ndarray
    conditional_actuator_gimbal: np.ndarray
    recorded_control_rollout_position: np.ndarray
    recorded_control_rollout_orientation_xyzw: np.ndarray
    recorded_control_rollout_valid: np.ndarray
    observed_relative_correction_translation: np.ndarray
    observed_relative_correction_rotation_vector: np.ndarray
    observed_relative_correction_valid: np.ndarray
    dynamics_residual: np.ndarray
    dynamics_residual_valid: np.ndarray
    conditional_objective: np.ndarray

    def index_of(self, sample_id: str) -> int:
        matches = np.flatnonzero(self.sample_id == str(sample_id))
        if matches.size != 1:
            raise KeyError(
                "sample {!r} has no stored trajectory for bag {!r}".format(
                    sample_id, self.bag_id
                )
            )
        return int(matches[0])

    def trajectory(self, sample_id: str) -> ConditionalTrajectory:
        index = self.index_of(sample_id)
        return ConditionalTrajectory(
            sample_id=str(self.sample_id[index]),
            knot_time=self.knot_time,
            state=StateTrajectory(
                position=self.conditional_position[index],
                orientation_xyzw=self.conditional_orientation_xyzw[index],
                linear_velocity=self.conditional_linear_velocity[index],
                angular_velocity=self.conditional_angular_velocity[index],
                controller_integral=self.conditional_controller_integral[index],
                actuator_thrust=self.conditional_actuator_thrust[index],
                actuator_gimbal=self.conditional_actuator_gimbal[index],
            ),
            recorded_control_rollout=PoseTrajectory(
                position=self.recorded_control_rollout_position[index],
                orientation_xyzw=(
                    self.recorded_control_rollout_orientation_xyzw[index]
                ),
                valid=self.recorded_control_rollout_valid[index],
            ),
            observed_relative_correction_translation=(
                self.observed_relative_correction_translation[index]
            ),
            observed_relative_correction_rotation_vector=(
                self.observed_relative_correction_rotation_vector[index]
            ),
            observed_relative_correction_valid=(
                self.observed_relative_correction_valid[index]
            ),
            dynamics_residual=self.dynamics_residual[index],
            dynamics_residual_valid=self.dynamics_residual_valid[index],
            conditional_objective=float(self.conditional_objective[index]),
        )


@dataclass(frozen=True)
class McmcDiagnostics:
    chain_id: np.ndarray
    mode_id: str
    draws_per_chain: int
    split_rhat: np.ndarray
    effective_sample_size: np.ndarray
    integrated_autocorrelation_time: np.ndarray
    ridge_coordinate_trace: np.ndarray
    delay_trace: np.ndarray
    log_posterior_trace: np.ndarray
    kernel_names: np.ndarray
    kernel_attempts: np.ndarray
    kernel_stage_one_accepted: np.ndarray
    kernel_stage_two_attempted: np.ndarray
    kernel_stage_two_accepted: np.ndarray
    kernel_full_target_cache_hits: np.ndarray
    kernel_inner_solve_failures: np.ndarray
    kernel_inner_iterations: np.ndarray
    completed: bool
    converged: bool
    rhat_threshold: float
    minimum_effective_sample_size: float


@dataclass(frozen=True)
class RunDiagnostics:
    bag_id: np.ndarray
    knot_count: np.ndarray
    factor_count: np.ndarray
    residual_dimension: np.ndarray
    jacobian_nnz: np.ndarray
    assembly_seconds: np.ndarray
    factorization_seconds: np.ndarray
    schur_solve_seconds: np.ndarray
    nonlinear_iteration_seconds: np.ndarray
    em_iteration_seconds: np.ndarray
    mcmc_target_seconds: np.ndarray
    peak_memory_bytes: int
    mcmc: McmcDiagnostics | None


@dataclass(frozen=True)
class BatchEstimationRun:
    root: Path
    manifest: Mapping[str, Any]
    static_map: StaticParameterMap
    q_em: QEmHistory
    laplace: LaplaceApproximation
    diagnostics: RunDiagnostics
    bags: Mapping[str, BagEstimationResult]
    mcmc: McmcPosterior | None
    selected_trajectories: Mapping[str, SelectedTrajectorySet]
    warnings: tuple[str, ...]

    @property
    def request_fingerprint(self) -> str:
        return str(self.manifest["request_fingerprint"])

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def sample_ids(self) -> tuple[str, ...]:
        if self.mcmc is None:
            return ()
        return tuple(str(value) for value in self.mcmc.sample_id.tolist())

    @property
    def preferred_sample_id(self) -> str | None:
        """Prefer the first MCMC sample with a stored conditional path."""

        if self.mcmc is None:
            return None
        available = set(self.sample_ids)
        bag_order = tuple(
            str(value)
            for value in self.manifest.get("selected_bag_ids", ())
        )
        if not bag_order:
            bag_order = tuple(self.bags)
        for bag_id in bag_order:
            subset = self.selected_trajectories.get(bag_id)
            if subset is None:
                continue
            for sample_id in subset.sample_id.tolist():
                candidate = str(sample_id)
                if candidate in available:
                    return candidate
        return None if not self.sample_ids else self.sample_ids[0]

    def selected_trajectory(
        self, bag_id: str, sample_id: str
    ) -> ConditionalTrajectory | None:
        subset = self.selected_trajectories.get(bag_id)
        if subset is None or str(sample_id) not in set(subset.sample_id.tolist()):
            return None
        return subset.trajectory(str(sample_id))


@dataclass(frozen=True)
class PidProposalEvaluation:
    root: Path
    manifest: Mapping[str, Any]
    source_samples: Mapping[str, np.ndarray]
    candidate_particles: Mapping[str, np.ndarray]
    summary: Mapping[str, np.ndarray]
    bags: Mapping[str, Mapping[str, np.ndarray]]
    proposed_yaml: str | None
    proposed_diff_yaml: str | None


def _preview_result(
    bag_id: str,
    arrays: Mapping[str, np.ndarray],
    inspection: Mapping[str, Any],
) -> FlightResult:
    time_key = "times" if "times" in arrays else "time"
    record_key = "record_times" if "record_times" in arrays else time_key
    return FlightResult(
        bag_id=bag_id,
        time=_array(arrays[time_key]),
        record_time=_array(arrays[record_key]),
        reference_position=_array(arrays["reference_position"]),
        reference_rpy=_array(arrays["reference_rpy"]),
        observed_position=_array(arrays["position"]),
        observed_orientation_xyzw=_array(arrays["orientation_xyzw"]),
        flight_state=(
            None if "flight_state" not in arrays else _array(arrays["flight_state"])
        ),
        provenance=inspection,
    )


def load_inspection(path: str | Path) -> InspectionArtifact:
    """Load a complete inspection bundle through its strict validator."""

    backend = _inspection_backend()
    try:
        bundle = backend.load_inspection_bundle(path)
    except (backend.ArtifactValidationError, OSError) as error:
        raise GuiArtifactError("cannot load inspection: {}".format(error)) from error
    previews = {
        bag_id: _preview_result(
            bag_id, bundle.previews[bag_id], bundle.inspections[bag_id]
        )
        for bag_id in bundle.inspections
    }
    return InspectionArtifact(
        root=bundle.root,
        manifest=bundle.manifest,
        inspections=bundle.inspections,
        previews=previews,
    )


def _static_map(arrays: Mapping[str, np.ndarray]) -> StaticParameterMap:
    bag_ids = _text_array(arrays["bag_id"])
    return StaticParameterMap(
        parameter_coordinate=_array(arrays["parameter_coordinate_map"]),
        mass=float(arrays["mass"][0]),
        inertia=_array(arrays["inertia"]),
        cog=_array(arrays["cog"]),
        force_effectiveness=_array(arrays["force_effectiveness"]),
        torque_effectiveness=_array(arrays["torque_effectiveness"]),
        delay=float(arrays["delay"][0]),
        q_diagonal=_array(arrays["q_diagonal"]),
        objective_components=_named_values(
            arrays["objective_component_names"],
            arrays["objective_component_values"],
        ),
        prior_objective=float(arrays["prior_objective"][0]),
        likelihood_objective=float(arrays["likelihood_objective"][0]),
        bag_objective={
            str(bag_id): float(value)
            for bag_id, value in zip(bag_ids, arrays["bag_objective"])
        },
    )


def _q_em(arrays: Mapping[str, np.ndarray]) -> QEmHistory:
    return QEmHistory(
        iteration=_array(arrays["iteration"]),
        input_q=_array(arrays["input_q"]),
        target_q=_array(arrays["target_q"]),
        accepted_q=_array(arrays["accepted_q"]),
        alpha=_array(arrays["alpha"]),
        log_q_change=_array(arrays["log_q_change"]),
        map_objective=_array(arrays["map_objective"]),
        approximate_marginal_objective=_array(
            arrays["approximate_marginal_objective"]
        ),
        delay=_array(arrays["lag"]),
        accepted=_array(arrays["accepted"]),
        reason=_text_array(arrays["reason"]),
        floor_activation=_array(arrays["floor_activation"]),
        expected_residual_second_moment=_array(
            arrays["expected_residual_second_moment"]
        ),
        map_residual_second_moment=_array(
            arrays["map_residual_second_moment"]
        ),
        covariance_correction=_array(arrays["covariance_correction"]),
    )


def _symmetric_information(
    value: Any, name: str, shape: tuple[int, int]
) -> np.ndarray:
    information = np.asarray(value, dtype=float)
    if information.shape != shape:
        raise GuiArtifactError(
            "{} must have shape {}, got {}".format(
                name, shape, information.shape
            )
        )
    if not np.all(np.isfinite(information)):
        raise GuiArtifactError("{} must be finite".format(name))
    if not np.allclose(
        information, information.T, rtol=1.0e-9, atol=1.0e-11
    ):
        raise GuiArtifactError("{} must be symmetric".format(name))
    return information


def _laplace(
    arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any]
) -> LaplaceApproximation:
    likelihood = _symmetric_information(
        arrays["reduced_likelihood_hessian"],
        "reduced likelihood information",
        (18, 18),
    )
    posterior = _symmetric_information(
        arrays["reduced_posterior_hessian"],
        "reduced posterior information",
        (18, 18),
    )
    prior = _symmetric_information(
        posterior - likelihood,
        "reduced prior information (posterior - likelihood)",
        (18, 18),
    )
    prior_scale = max(1.0, float(np.linalg.norm(prior, ord=2)))
    if float(np.min(np.linalg.eigvalsh(prior))) < -1.0e-9 * prior_scale:
        raise GuiArtifactError(
            "reduced prior information (posterior - likelihood) must be "
            "positive semidefinite within numerical tolerance"
        )
    geometry_valid = bool(arrays["delay_local_geometry_valid"][0])
    joint_information: np.ndarray | None = None
    joint_covariance: np.ndarray | None = None
    joint_static_marginal: np.ndarray | None = None
    cross_covariance: np.ndarray | None = None
    joint_delay_sigma: float | None = None
    fallback_delay_prior_sigma: float | None = None
    raw_delay_scale = float(arrays["delay_local_uncertainty"][0])
    raw_curvature = float(arrays["delay_profile_curvature"][0])
    if geometry_valid:
        joint_information = _symmetric_information(
            arrays["joint_parameter_delay_information"],
            "joint parameter-delay information",
            (19, 19),
        )
        joint_covariance = _symmetric_information(
            arrays["joint_parameter_delay_covariance"],
            "joint parameter-delay covariance",
            (19, 19),
        )
        cross_covariance = _array(
            arrays["parameter_delay_cross_covariance"]
        )
        if cross_covariance.shape != (18,):
            raise GuiArtifactError(
                "valid parameter-delay cross covariance must have shape (18,)"
            )
        joint_static_marginal = joint_covariance[:18, :18]
        joint_delay_sigma = float(np.sqrt(joint_covariance[-1, -1]))
    else:
        fallback_delay_prior_sigma = raw_delay_scale
    mcmc_settings = manifest.get("mcmc_settings", {})
    proposal_value = (
        mcmc_settings.get("delay_scale_seconds")
        if isinstance(mcmc_settings, Mapping)
        else None
    )
    proposal_scale = (
        None if proposal_value is None else float(proposal_value)
    )
    if proposal_scale is not None and (
        not np.isfinite(proposal_scale) or proposal_scale <= 0.0
    ):
        raise GuiArtifactError(
            "manifest mcmc_settings.delay_scale_seconds must be positive"
        )

    return LaplaceApproximation(
        reduced_likelihood_hessian=likelihood,
        reduced_prior_information=prior,
        reduced_posterior_hessian=posterior,
        fixed_delay_conditional_static_covariance=_array(
            arrays["covariance"]
        ),
        static_covariance_conditioning=str(
            _text_array(arrays["static_covariance_conditioning"])[0]
        ),
        eigenvalues=_array(arrays["eigenvalues"]),
        eigenvectors=_array(arrays["eigenvectors"]),
        effective_rank=int(arrays["effective_rank"][0]),
        exact_ridge_direction=_array(arrays["exact_ridge_direction"]),
        ridge_alignment=float(arrays["ridge_alignment"][0]),
        condition_number=float(arrays["condition_number"][0]),
        delay_profile_available=bool(arrays["delay_profile_available"][0]),
        delay_profile_grid=_array(arrays["delay_profile_grid"]),
        delay_profile_objective=_array(arrays["delay_profile_objective"]),
        delay_profile_approximate_marginal_objective=_array(
            arrays["delay_profile_approximate_marginal_objective"]
        ),
        delay_profile_static_coordinate=_array(
            arrays["delay_profile_static_coordinate"]
        ),
        delay_local_geometry_valid=geometry_valid,
        delay_local_geometry_method=str(
            _text_array(arrays["delay_local_geometry_method"])[0]
        ),
        delay_local_geometry_reason=str(
            _text_array(arrays["delay_local_geometry_reason"])[0]
        ),
        delay_profile_curvature=(
            raw_curvature if geometry_valid else None
        ),
        delay_uncertainty_source=str(
            _text_array(arrays["delay_uncertainty_source"])[0]
        ),
        joint_delay_marginal_standard_deviation=joint_delay_sigma,
        fallback_delay_prior_standard_deviation=(
            fallback_delay_prior_sigma
        ),
        mcmc_delay_proposal_scale_seconds=proposal_scale,
        parameter_delay_cross_covariance=cross_covariance,
        joint_parameter_delay_information=joint_information,
        joint_parameter_delay_covariance=joint_covariance,
        joint_static_parameter_marginal_covariance=joint_static_marginal,
        mcmc_quadratic_surrogate_method=str(
            _text_array(arrays["mcmc_quadratic_surrogate_method"])[0]
        ),
    )


def _vector_observation(
    arrays: Mapping[str, np.ndarray], prefix: str, value_name: str
) -> VectorObservation:
    return VectorObservation(
        time=_array(arrays["{}_time".format(prefix)]),
        record_time=_array(arrays["{}_record_time".format(prefix)]),
        value=_array(arrays[value_name]),
        valid=_array(arrays["{}_valid".format(prefix)]),
        covariance=_array(arrays["{}_covariance".format(prefix)]),
        covariance_valid=_array(
            arrays["{}_covariance_valid".format(prefix)]
        ),
    )


def _state_trajectory(
    arrays: Mapping[str, np.ndarray], prefix: str
) -> StateTrajectory:
    return StateTrajectory(
        position=_array(arrays["{}_position".format(prefix)]),
        orientation_xyzw=_array(
            arrays["{}_orientation_xyzw".format(prefix)]
        ),
        linear_velocity=_array(
            arrays["{}_linear_velocity".format(prefix)]
        ),
        angular_velocity=_array(
            arrays["{}_angular_velocity".format(prefix)]
        ),
        controller_integral=_array(
            arrays["{}_controller_integral".format(prefix)]
        ),
        actuator_thrust=_array(
            arrays["{}_actuator_thrust".format(prefix)]
        ),
        actuator_gimbal=_array(
            arrays["{}_actuator_gimbal".format(prefix)]
        ),
    )


def _bag_result(
    bag_id: str,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> BagEstimationResult:
    factor_names = _text_array(arrays["factor_names"])
    residual = _array(arrays["factor_residual_history"])
    normalized = _array(arrays["factor_normalized_residual_history"])
    return BagEstimationResult(
        bag_id=bag_id,
        knot_time=_array(arrays["knot_time"]),
        knot_record_time=_array(arrays["knot_record_time"]),
        reference=ReferenceTrajectory(
            time=_array(arrays["reference_time"]),
            record_time=_array(arrays["reference_record_time"]),
            position=_array(arrays["reference_position"]),
            linear_velocity=_array(arrays["reference_linear_velocity"]),
            linear_acceleration=_array(
                arrays["reference_linear_acceleration"]
            ),
            rpy=_array(arrays["reference_rpy"]),
            angular_velocity=_array(arrays["reference_angular_velocity"]),
            angular_acceleration=_array(
                arrays["reference_angular_acceleration"]
            ),
        ),
        pose=PoseObservation(
            time=_array(arrays["pose_time"]),
            record_time=_array(arrays["pose_record_time"]),
            position=_array(arrays["pose_position"]),
            orientation_xyzw=_array(arrays["pose_orientation_xyzw"]),
            valid=_array(arrays["pose_valid"]),
            covariance=_array(arrays["pose_covariance"]),
            covariance_valid=_array(arrays["pose_covariance_valid"]),
        ),
        velocity=_vector_observation(arrays, "velocity", "velocity"),
        gyro=_vector_observation(arrays, "gyro", "gyro"),
        accelerometer=_vector_observation(
            arrays, "accelerometer", "accelerometer"
        ),
        thrust_command=_vector_observation(
            arrays, "thrust_command", "thrust_command"
        ),
        gimbal_command=_vector_observation(
            arrays, "gimbal_command", "gimbal_command"
        ),
        gimbal_observation=_vector_observation(
            arrays, "gimbal_observation", "gimbal_observation"
        ),
        controller_integral_observation=_vector_observation(
            arrays,
            "controller_integral",
            "controller_integral_observation",
        ),
        nominal=_state_trajectory(arrays, "nominal"),
        map_trajectory=_state_trajectory(arrays, "map"),
        initial_parameter_rollout=PoseTrajectory(
            position=_array(
                arrays["initial_parameter_rollout_position"]
            ),
            orientation_xyzw=_array(
                arrays["initial_parameter_rollout_orientation_xyzw"]
            ),
            valid=_array(arrays["initial_parameter_rollout_valid"]),
        ),
        estimated_parameter_rollout=PoseTrajectory(
            position=_array(
                arrays["estimated_parameter_rollout_position"]
            ),
            orientation_xyzw=_array(
                arrays["estimated_parameter_rollout_orientation_xyzw"]
            ),
            valid=_array(arrays["estimated_parameter_rollout_valid"]),
        ),
        initial_parameter_rollout_correction_translation=_array(
            arrays[
                "initial_parameter_rollout_correction_translation"
            ]
        ),
        initial_parameter_rollout_correction_rotation_vector=_array(
            arrays[
                "initial_parameter_rollout_correction_rotation_vector"
            ]
        ),
        initial_parameter_rollout_correction_valid=_array(
            arrays["initial_parameter_rollout_correction_valid"]
        ),
        estimated_parameter_rollout_correction_translation=_array(
            arrays[
                "estimated_parameter_rollout_correction_translation"
            ]
        ),
        estimated_parameter_rollout_correction_rotation_vector=_array(
            arrays[
                "estimated_parameter_rollout_correction_rotation_vector"
            ]
        ),
        estimated_parameter_rollout_correction_valid=_array(
            arrays["estimated_parameter_rollout_correction_valid"]
        ),
        map_dynamics_residual=_array(arrays["map_dynamics_residual"]),
        map_dynamics_residual_valid=_array(
            arrays["map_dynamics_residual_valid"]
        ),
        factor_residual_history={
            str(name): residual[:, index]
            for index, name in enumerate(factor_names)
        },
        factor_normalized_residual_history={
            str(name): normalized[:, index]
            for index, name in enumerate(factor_names)
        },
        objective_components=_named_values(
            arrays["objective_component_names"],
            arrays["objective_component_values"],
        ),
        numerical_diagnostics=_named_values(
            arrays["numerical_diagnostic_names"],
            arrays["numerical_diagnostic_values"],
        ),
        sensor_contract=manifest["sensor_contracts"][bag_id],
        observation_factors=manifest["observation_factors"][bag_id],
    )


def _mcmc(arrays: Mapping[str, np.ndarray]) -> McmcPosterior:
    return McmcPosterior(
        sample_id=_text_array(arrays["sample_id"]),
        chain_id=_text_array(arrays["chain_id"]),
        draw_index=_array(arrays["draw_index"]),
        parameter_coordinate=_array(arrays["parameter_coordinate"]),
        mass=_array(arrays["mass"]),
        inertia=_array(arrays["inertia"]),
        cog=_array(arrays["cog"]),
        force_effectiveness=_array(arrays["force_effectiveness"]),
        torque_effectiveness=_array(arrays["torque_effectiveness"]),
        delay=_array(arrays["delay"]),
        log_posterior=_array(arrays["log_posterior"]),
        log_likelihood_approximation=_array(
            arrays["log_likelihood_approximation"]
        ),
        log_determinant_term=_array(arrays["log_determinant_term"]),
        accepted_kernel=_text_array(arrays["accepted_kernel"]),
        source_mode_id=_text_array(arrays["source_mode_id"]),
    )


def _trajectory_set(
    bag_id: str, arrays: Mapping[str, np.ndarray]
) -> SelectedTrajectorySet:
    return SelectedTrajectorySet(
        bag_id=bag_id,
        sample_id=_text_array(arrays["sample_id"]),
        knot_time=_array(arrays["knot_time"]),
        conditional_position=_array(arrays["conditional_position"]),
        conditional_orientation_xyzw=_array(
            arrays["conditional_orientation_xyzw"]
        ),
        conditional_linear_velocity=_array(
            arrays["conditional_linear_velocity"]
        ),
        conditional_angular_velocity=_array(
            arrays["conditional_angular_velocity"]
        ),
        conditional_controller_integral=_array(
            arrays["conditional_controller_integral"]
        ),
        conditional_actuator_thrust=_array(
            arrays["conditional_actuator_thrust"]
        ),
        conditional_actuator_gimbal=_array(
            arrays["conditional_actuator_gimbal"]
        ),
        recorded_control_rollout_position=_array(
            arrays["recorded_control_rollout_position"]
        ),
        recorded_control_rollout_orientation_xyzw=_array(
            arrays["recorded_control_rollout_orientation_xyzw"]
        ),
        recorded_control_rollout_valid=_array(
            arrays["recorded_control_rollout_valid"]
        ),
        observed_relative_correction_translation=_array(
            arrays["observed_relative_correction_translation"]
        ),
        observed_relative_correction_rotation_vector=_array(
            arrays["observed_relative_correction_rotation_vector"]
        ),
        observed_relative_correction_valid=_array(
            arrays["observed_relative_correction_valid"]
        ),
        dynamics_residual=_array(arrays["dynamics_residual"]),
        dynamics_residual_valid=_array(arrays["dynamics_residual_valid"]),
        conditional_objective=_array(arrays["conditional_objective"]),
    )


def _mcmc_diagnostics(
    arrays: Mapping[str, np.ndarray]
) -> McmcDiagnostics | None:
    if "mcmc_chain_id" not in arrays:
        return None
    return McmcDiagnostics(
        chain_id=_text_array(arrays["mcmc_chain_id"]),
        mode_id=str(_text_array(arrays["mcmc_mode_id"])[0]),
        draws_per_chain=int(arrays["mcmc_draws_per_chain"][0]),
        split_rhat=_array(arrays["mcmc_split_rhat"]),
        effective_sample_size=_array(
            arrays["mcmc_effective_sample_size"]
        ),
        integrated_autocorrelation_time=_array(
            arrays["mcmc_integrated_autocorrelation_time"]
        ),
        ridge_coordinate_trace=_array(
            arrays["mcmc_ridge_coordinate_trace"]
        ),
        delay_trace=_array(arrays["mcmc_delay_trace"]),
        log_posterior_trace=_array(arrays["mcmc_log_posterior_trace"]),
        kernel_names=_text_array(arrays["mcmc_kernel_names"]),
        kernel_attempts=_array(arrays["mcmc_kernel_attempts"]),
        kernel_stage_one_accepted=_array(
            arrays["mcmc_kernel_stage_one_accepted"]
        ),
        kernel_stage_two_attempted=_array(
            arrays["mcmc_kernel_stage_two_attempted"]
        ),
        kernel_stage_two_accepted=_array(
            arrays["mcmc_kernel_stage_two_accepted"]
        ),
        kernel_full_target_cache_hits=_array(
            arrays["mcmc_kernel_full_target_cache_hits"]
        ),
        kernel_inner_solve_failures=_array(
            arrays["mcmc_kernel_inner_solve_failures"]
        ),
        kernel_inner_iterations=_array(
            arrays["mcmc_kernel_inner_iterations"]
        ),
        completed=bool(arrays["mcmc_completed"][0]),
        converged=bool(arrays["mcmc_converged"][0]),
        rhat_threshold=float(arrays["mcmc_rhat_threshold"][0]),
        minimum_effective_sample_size=float(
            arrays["mcmc_minimum_effective_sample_size"][0]
        ),
    )


def _diagnostics(arrays: Mapping[str, np.ndarray]) -> RunDiagnostics:
    return RunDiagnostics(
        bag_id=_text_array(arrays["bag_id"]),
        knot_count=_array(arrays["knot_count"]),
        factor_count=_array(arrays["factor_count"]),
        residual_dimension=_array(arrays["residual_dimension"]),
        jacobian_nnz=_array(arrays["jacobian_nnz"]),
        assembly_seconds=_array(arrays["assembly_seconds"]),
        factorization_seconds=_array(arrays["factorization_seconds"]),
        schur_solve_seconds=_array(arrays["schur_solve_seconds"]),
        nonlinear_iteration_seconds=_array(
            arrays["nonlinear_iteration_seconds"]
        ),
        em_iteration_seconds=_array(arrays["em_iteration_seconds"]),
        mcmc_target_seconds=_array(arrays["mcmc_target_seconds"]),
        peak_memory_bytes=int(arrays["peak_memory_bytes"][0]),
        mcmc=_mcmc_diagnostics(arrays),
    )


def load_batch_estimation_run(path: str | Path) -> BatchEstimationRun:
    """Load exactly one complete strict v1 sparse-batch estimation run."""

    backend = _batch_backend()
    try:
        bundle = backend.load_batch_estimation_run(path)
    except (backend.ArtifactValidationError, OSError) as error:
        raise GuiArtifactError(
            "cannot load batch estimation run: {}".format(error)
        ) from error

    mcmc = None if bundle.mcmc_samples is None else _mcmc(bundle.mcmc_samples)
    return BatchEstimationRun(
        root=bundle.root,
        manifest=bundle.manifest,
        static_map=_static_map(bundle.map_static),
        q_em=_q_em(bundle.q_em),
        laplace=_laplace(bundle.laplace, bundle.manifest),
        diagnostics=_diagnostics(bundle.diagnostics),
        bags={
            bag_id: _bag_result(bag_id, arrays, bundle.manifest)
            for bag_id, arrays in bundle.bags.items()
        },
        mcmc=mcmc,
        selected_trajectories={
            bag_id: _trajectory_set(bag_id, arrays)
            for bag_id, arrays in bundle.trajectories.items()
        },
        warnings=tuple(str(value) for value in bundle.manifest["warnings"]),
    )


def load_pid_evaluation(path: str | Path) -> PidProposalEvaluation:
    """Load a PID evaluation through its backend validator."""

    backend = _pid_backend()
    try:
        bundle = backend.load_pid_proposal_evaluation(path)
    except (backend.ArtifactValidationError, OSError) as error:
        raise GuiArtifactError(
            "cannot load PID proposal evaluation: {}".format(error)
        ) from error
    return PidProposalEvaluation(
        root=bundle.root,
        manifest=bundle.manifest,
        source_samples=bundle.source_samples,
        candidate_particles=bundle.candidate_particles,
        summary=bundle.summary,
        bags=bundle.bags,
        proposed_yaml=bundle.proposed_yaml,
        proposed_diff_yaml=bundle.proposed_diff_yaml,
    )


__all__ = [
    "BagEstimationResult",
    "BatchEstimationRun",
    "ConditionalTrajectory",
    "FlightResult",
    "GUI_ARTIFACT_LOADER_ID",
    "GUI_ARTIFACT_LOADER_VERSION",
    "GuiArtifactError",
    "InspectionArtifact",
    "LaplaceApproximation",
    "McmcDiagnostics",
    "McmcPosterior",
    "PidProposalEvaluation",
    "PoseTrajectory",
    "QEmHistory",
    "RunDiagnostics",
    "SelectedTrajectorySet",
    "StateTrajectory",
    "StaticParameterMap",
    "StaticParameterSample",
    "VectorObservation",
    "load_batch_estimation_run",
    "load_inspection",
    "load_pid_evaluation",
]
