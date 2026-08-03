"""Strong-constraint fixed-interval IEnKS for the full Grape loop."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import warnings

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.controller import ControllerConfig, GrapeController
from grape_param_estim.dynamics import FullSixDofPlant, simulate_closed_loop
from grape_param_estim.geometry import (
    correction_transform_path,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_matrix_from_vector,
    rotation_vector_from_matrix,
)
from grape_param_estim.parameterization import (
    PARAMETER_DIMENSION,
    VehicleParameterChart,
)
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    ClosedLoopTrajectory,
    ControllerState,
    GrapeGeometry,
    PoseObservations,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


INITIAL_STATE_DIMENSION = 12
CONTROLLER_STATE_DIMENSION = 6
PARAMETER_OFFSET = INITIAL_STATE_DIMENSION + CONTROLLER_STATE_DIMENSION
CONTROL_DIMENSION = PARAMETER_OFFSET + PARAMETER_DIMENSION

CONTROL_NAMES = (
    "initial_position_x",
    "initial_position_y",
    "initial_position_z",
    "initial_rotation_x",
    "initial_rotation_y",
    "initial_rotation_z",
    "initial_velocity_x",
    "initial_velocity_y",
    "initial_velocity_z",
    "initial_omega_x",
    "initial_omega_y",
    "initial_omega_z",
    "initial_pid_integral_x",
    "initial_pid_integral_y",
    "initial_pid_integral_z",
    "initial_pid_integral_roll",
    "initial_pid_integral_pitch",
    "initial_pid_integral_yaw",
) + tuple("plant_{}".format(index) for index in range(PARAMETER_DIMENSION))


def _finite_array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite {} array".format(name, shape))
    return result.copy()


def _symmetric_inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("matrix must be positive definite")
    return (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )


@dataclass(frozen=True)
class StrongConstraintPrior:
    """Proper Gaussian prior in the unconstrained control chart."""

    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        mean = _finite_array(self.mean, (CONTROL_DIMENSION,), "prior mean")
        covariance = _finite_array(
            self.covariance,
            (CONTROL_DIMENSION, CONTROL_DIMENSION),
            "prior covariance",
        )
        if not np.allclose(covariance, covariance.T, atol=1.0e-12):
            raise ValueError("prior covariance must be symmetric")
        if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
            raise ValueError("prior covariance must be positive definite")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    @classmethod
    def grape(cls):
        standard_deviation = np.asarray(
            (
                # Initial pose tangent state.
                0.015, 0.015, 0.015,
                0.012, 0.012, 0.012,
                # Latent initial velocity and body angular velocity.
                0.10, 0.10, 0.10,
                0.07, 0.07, 0.07,
                # Stateful PID integral snapshot.
                0.05, 0.05, 0.20,
                0.03, 0.03, 0.03,
                # Plant chart: mass; inertia diag/offdiag; CoG; force; torque.
                0.16,
                0.16, 0.16, 0.16,
                0.08, 0.08, 0.08,
                0.012, 0.012, 0.012,
                0.16, 0.16, 0.16, 0.16,
                0.18, 0.18, 0.18, 0.18,
            ),
            dtype=float,
        )
        if standard_deviation.shape != (CONTROL_DIMENSION,):
            raise AssertionError("default prior does not match control chart")
        return cls(
            mean=np.zeros(CONTROL_DIMENSION),
            covariance=np.diag(standard_deviation**2),
        )

    def ensemble(self, size: int, seed: int) -> np.ndarray:
        """Draw a reproducible ensemble with exact sample mean/covariance."""

        member_count = int(size)
        if member_count <= CONTROL_DIMENSION:
            raise ValueError(
                "ensemble size must exceed the 36-dimensional control"
            )
        generator = np.random.RandomState(int(seed))
        standard = generator.normal(
            size=(CONTROL_DIMENSION, member_count)
        )
        standard -= np.mean(standard, axis=1, keepdims=True)
        sample_covariance = standard @ standard.T / (member_count - 1.0)
        standard = _symmetric_inverse_sqrt(sample_covariance) @ standard
        factor = np.linalg.cholesky(self.covariance)
        values = self.mean[:, None] + factor @ standard
        return values.T


@dataclass(frozen=True)
class StrongConstraintProblem:
    """Black-box full-window forecast and pose-only residual contract."""

    references: Tuple[ReferenceState, ...]
    observations: PoseObservations
    nominal_trajectory: ClosedLoopTrajectory
    initial_state_anchor: RigidBodyState
    initial_controller_anchor: ControllerState
    controller_configuration: ControllerConfig
    controller_parameters: VehicleParameters
    geometry: GrapeGeometry
    actuator_parameters: ActuatorParameters
    parameter_chart: VehicleParameterChart
    initial_actuator_state: Optional[ActuatorState] = None

    def __post_init__(self) -> None:
        sample_count = self.observations.times.size
        if len(self.references) != sample_count:
            raise ValueError("reference and observation lengths must agree")
        if not np.array_equal(
            self.observations.times, self.nominal_trajectory.times
        ):
            raise ValueError("nominal and observed times must be identical")
        for covariance, name in (
            (
                self.observations.translation_covariance,
                "translation covariance",
            ),
            (self.observations.rotation_covariance, "rotation covariance"),
        ):
            if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
                raise ValueError("{} must be positive definite".format(name))
        if self.initial_actuator_state is not None:
            if not isinstance(self.initial_actuator_state, ActuatorState):
                raise ValueError(
                    "initial_actuator_state must be an ActuatorState or None"
                )
            object.__setattr__(
                self,
                "initial_actuator_state",
                ActuatorState(
                    self.initial_actuator_state.thrust,
                    self.initial_actuator_state.gimbal_angle,
                ),
            )

    def decode_control(
        self, control: Sequence[float]
    ) -> Tuple[RigidBodyState, ControllerState, VehicleParameters]:
        value = _finite_array(
            control, (CONTROL_DIMENSION,), "strong-constraint control"
        )
        anchor_rotation = quaternion_to_matrix(
            self.initial_state_anchor.orientation_xyzw
        )
        initial_rotation = anchor_rotation @ rotation_matrix_from_vector(
            value[3:6]
        )
        state = RigidBodyState(
            position=self.initial_state_anchor.position + value[0:3],
            orientation_xyzw=matrix_to_quaternion(initial_rotation),
            linear_velocity=(
                self.initial_state_anchor.linear_velocity + value[6:9]
            ),
            angular_velocity=(
                self.initial_state_anchor.angular_velocity + value[9:12]
            ),
        )
        controller_state = ControllerState(
            integral_error=(
                self.initial_controller_anchor.integral_error
                + value[INITIAL_STATE_DIMENSION:PARAMETER_OFFSET]
            ),
            roll_pitch_integration_active=(
                self.initial_controller_anchor.roll_pitch_integration_active
            ),
        )
        parameters = self.parameter_chart.decode(value[PARAMETER_OFFSET:])
        return state, controller_state, parameters

    def forecast(self, control: Sequence[float]) -> ClosedLoopTrajectory:
        """Integrate every sample once; observations never reset the member."""

        initial_state, controller_state, parameters = self.decode_control(
            control
        )
        controller = GrapeController(
            self.controller_configuration,
            self.controller_parameters,
            self.geometry,
            articulated_model=GrapeArticulatedModel(),
        )
        plant = FullSixDofPlant(parameters, self.geometry)
        return simulate_closed_loop(
            times=self.observations.times,
            references=self.references,
            initial_state=initial_state,
            initial_controller_state=controller_state,
            controller=controller,
            plant=plant,
            actuator_parameters=self.actuator_parameters,
            initial_actuator_state=self.initial_actuator_state,
        )

    def residual(self, trajectory: ClosedLoopTrajectory) -> np.ndarray:
        """Whiten full-window position and SO(3) residuals."""

        if not np.array_equal(trajectory.times, self.observations.times):
            raise ValueError("forecast and observation times must agree")
        translation_factor = np.linalg.cholesky(
            self.observations.translation_covariance
        )
        rotation_factor = np.linalg.cholesky(
            self.observations.rotation_covariance
        )
        translation = np.linalg.solve(
            translation_factor,
            (trajectory.position - self.observations.position).T,
        ).T
        rotation = np.empty_like(translation)
        for index in range(trajectory.times.size):
            observed = quaternion_to_matrix(
                self.observations.orientation_xyzw[index]
            )
            predicted = quaternion_to_matrix(
                trajectory.orientation_xyzw[index]
            )
            tangent = rotation_vector_from_matrix(observed.T @ predicted)
            rotation[index] = np.linalg.solve(rotation_factor, tangent)
        return np.concatenate((translation.reshape(-1), rotation.reshape(-1)))


@dataclass(frozen=True)
class IEnKSConfig:
    ensemble_size: int = 48
    maximum_iterations: int = 5
    convergence_tolerance: float = 1.0e-3
    minimum_line_search_step: float = 1.0 / 64.0
    seed: int = 23

    def __post_init__(self) -> None:
        if self.ensemble_size <= CONTROL_DIMENSION:
            raise ValueError("ensemble size must exceed control dimension")
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if (
            not np.isfinite(self.convergence_tolerance)
            or self.convergence_tolerance <= 0.0
            or not np.isfinite(self.minimum_line_search_step)
            or not 0.0 < self.minimum_line_search_step <= 1.0
        ):
            raise ValueError("invalid IEnKS convergence configuration")


@dataclass(frozen=True)
class IEnKSIteration:
    iteration: int
    objective: float
    accepted_objective: float
    gradient_norm: float
    step_norm: float
    accepted_fraction: float


@dataclass(frozen=True)
class StaticParameterEnsemble:
    coordinates: np.ndarray
    mass: np.ndarray
    inertia: np.ndarray
    cog_offset: np.ndarray
    force_effectiveness: np.ndarray
    torque_effectiveness: np.ndarray


@dataclass(frozen=True)
class ParameterRidge:
    covariance: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    expected_direction: np.ndarray
    expected_variance: float


@dataclass(frozen=True)
class StrongConstraintPosterior:
    """Raw member-preserving posterior and push-forward paths."""

    control_ensemble: np.ndarray
    prior_control_ensemble: np.ndarray
    parameter_ensemble: StaticParameterEnsemble
    trajectory_ensemble: Tuple[ClosedLoopTrajectory, ...]
    prior_trajectory_ensemble: Tuple[ClosedLoopTrajectory, ...]
    center_control: np.ndarray
    center_trajectory: ClosedLoopTrajectory
    correction_translation: np.ndarray
    correction_rotation_vector: np.ndarray
    ridge: ParameterRidge
    iterations: Tuple[IEnKSIteration, ...]
    ensemble_rank: int
    converged: bool
    termination_reason: str


class StrongConstraintIEnKS:
    """Derivative-free ensemble-space Gauss--Newton IEnKS bundle solver."""

    def __init__(self, configuration: Optional[IEnKSConfig] = None):
        self.configuration = configuration or IEnKSConfig()

    @staticmethod
    def _objective(coordinate: np.ndarray, residual: np.ndarray) -> float:
        return 0.5 * float(
            np.dot(coordinate, coordinate) + np.dot(residual, residual)
        )

    @staticmethod
    def _controls(
        mean: np.ndarray,
        basis: np.ndarray,
        coordinates: np.ndarray,
    ) -> np.ndarray:
        return (mean[:, None] + basis @ coordinates).T

    @staticmethod
    def _forecast_controls(
        problem: StrongConstraintProblem, controls: np.ndarray
    ) -> Tuple[Tuple[ClosedLoopTrajectory, ...], np.ndarray]:
        trajectories = tuple(problem.forecast(value) for value in controls)
        residuals = np.column_stack(
            [problem.residual(value) for value in trajectories]
        )
        return trajectories, residuals

    @staticmethod
    def _regression(
        residuals: np.ndarray, coordinate_deviations: np.ndarray
    ) -> np.ndarray:
        residual_deviations = residuals - np.mean(
            residuals, axis=1, keepdims=True
        )
        gram = coordinate_deviations @ coordinate_deviations.T
        return (
            residual_deviations
            @ coordinate_deviations.T
            @ np.linalg.inv(gram)
        )

    def fit(
        self,
        problem: StrongConstraintProblem,
        prior: Optional[StrongConstraintPrior] = None,
    ) -> StrongConstraintPosterior:
        selected_prior = prior or StrongConstraintPrior.grape()
        prior_ensemble = selected_prior.ensemble(
            self.configuration.ensemble_size, self.configuration.seed
        )
        prior_mean = np.mean(prior_ensemble, axis=0)
        deviations = (prior_ensemble - prior_mean).T
        left, singular_values, right_transpose = np.linalg.svd(
            deviations, full_matrices=False
        )
        tolerance = (
            np.max(deviations.shape)
            * np.finfo(float).eps
            * singular_values[0]
        )
        rank = int(np.count_nonzero(singular_values > tolerance))
        if rank != CONTROL_DIMENSION:
            raise ValueError("prior ensemble does not span the control chart")
        basis = (
            left[:, :rank]
            * singular_values[:rank][None, :]
            / np.sqrt(self.configuration.ensemble_size - 1.0)
        )
        ensemble_coordinates = (
            np.sqrt(self.configuration.ensemble_size - 1.0)
            * right_transpose[:rank]
        )

        center = np.zeros(rank, dtype=float)
        transform = np.eye(rank)
        center_trajectory = problem.forecast(prior_mean)
        center_residual = problem.residual(center_trajectory)
        objective = self._objective(center, center_residual)
        diagnostics = []
        prior_trajectories = None
        converged = False
        termination_reason = "maximum_iterations"

        for iteration in range(self.configuration.maximum_iterations):
            local_coordinates = (
                center[:, None] + transform @ ensemble_coordinates
            )
            local_controls = self._controls(
                prior_mean, basis, local_coordinates
            )
            local_trajectories, local_residuals = self._forecast_controls(
                problem, local_controls
            )
            if prior_trajectories is None:
                prior_trajectories = local_trajectories
            coordinate_deviations = transform @ ensemble_coordinates
            sensitivity = self._regression(
                local_residuals, coordinate_deviations
            )
            hessian = np.eye(rank) + sensitivity.T @ sensitivity
            gradient = center + sensitivity.T @ center_residual
            step = -np.linalg.solve(hessian, gradient)
            fraction = 1.0
            accepted = False
            candidate_trajectory = center_trajectory
            candidate_residual = center_residual
            candidate_objective = objective
            while fraction >= self.configuration.minimum_line_search_step:
                candidate = center + fraction * step
                trial_control = prior_mean + basis @ candidate
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        trial_trajectory = problem.forecast(trial_control)
                        trial_residual = problem.residual(trial_trajectory)
                    trial_objective = self._objective(
                        candidate, trial_residual
                    )
                except (
                    ValueError,
                    FloatingPointError,
                    RuntimeWarning,
                    np.linalg.LinAlgError,
                ):
                    # A nonlinear ensemble-space step may leave the domain in
                    # which the black-box rigid-body forecast is finite.  It
                    # has infinite objective; reduce the same line-search
                    # direction without clipping any physical parameter.
                    fraction *= 0.5
                    continue
                if trial_objective < objective:
                    accepted = True
                    candidate_trajectory = trial_trajectory
                    candidate_residual = trial_residual
                    candidate_objective = trial_objective
                    break
                fraction *= 0.5
            diagnostics.append(
                IEnKSIteration(
                    iteration=iteration,
                    objective=objective,
                    accepted_objective=candidate_objective,
                    gradient_norm=float(np.linalg.norm(gradient)),
                    step_norm=float(np.linalg.norm(step)),
                    accepted_fraction=float(fraction if accepted else 0.0),
                )
            )
            transform = _symmetric_inverse_sqrt(hessian)
            if not accepted:
                termination_reason = "line_search_failed"
                break
            center = candidate
            center_trajectory = candidate_trajectory
            center_residual = candidate_residual
            objective = candidate_objective
            if np.linalg.norm(fraction * step) <= (
                self.configuration.convergence_tolerance
                * (1.0 + np.linalg.norm(center))
            ):
                converged = True
                termination_reason = "step_tolerance"
                break

        # Re-linearise once at the accepted center.  This is another ensemble
        # cloud regression, not a coordinate-wise finite-difference Jacobian.
        local_coordinates = center[:, None] + transform @ ensemble_coordinates
        local_controls = self._controls(prior_mean, basis, local_coordinates)
        _local_trajectories, local_residuals = self._forecast_controls(
            problem, local_controls
        )
        sensitivity = self._regression(
            local_residuals, transform @ ensemble_coordinates
        )
        posterior_hessian = np.eye(rank) + sensitivity.T @ sensitivity
        posterior_transform = _symmetric_inverse_sqrt(posterior_hessian)
        posterior_coordinates = (
            center[:, None] + posterior_transform @ ensemble_coordinates
        )
        posterior_controls = self._controls(
            prior_mean, basis, posterior_coordinates
        )
        posterior_trajectories, _posterior_residuals = (
            self._forecast_controls(problem, posterior_controls)
        )
        center_control = prior_mean + basis @ center
        center_trajectory = problem.forecast(center_control)

        correction_translation = []
        correction_rotation = []
        for trajectory in posterior_trajectories:
            translation, rotation = correction_transform_path(
                problem.nominal_trajectory.position,
                problem.nominal_trajectory.orientation_xyzw,
                trajectory.position,
                trajectory.orientation_xyzw,
            )
            correction_translation.append(translation)
            correction_rotation.append(rotation)
        parameter_coordinates = posterior_controls[:, PARAMETER_OFFSET:]
        decoded = tuple(
            problem.parameter_chart.decode(value)
            for value in parameter_coordinates
        )
        parameter_ensemble = StaticParameterEnsemble(
            coordinates=parameter_coordinates.copy(),
            mass=np.asarray([value.mass for value in decoded]),
            inertia=np.asarray([value.inertia for value in decoded]),
            cog_offset=np.asarray([value.cog_offset for value in decoded]),
            force_effectiveness=np.asarray(
                [value.force_effectiveness for value in decoded]
            ),
            torque_effectiveness=np.asarray(
                [value.torque_effectiveness for value in decoded]
            ),
        )
        parameter_covariance = np.cov(parameter_coordinates, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(parameter_covariance)
        expected_direction = problem.parameter_chart.ridge_direction()
        expected_direction = expected_direction / np.linalg.norm(
            expected_direction
        )
        ridge = ParameterRidge(
            covariance=parameter_covariance,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            expected_direction=expected_direction,
            expected_variance=float(
                expected_direction @ parameter_covariance @ expected_direction
            ),
        )
        if prior_trajectories is None:
            raise AssertionError("IEnKS did not evaluate its prior ensemble")
        return StrongConstraintPosterior(
            control_ensemble=posterior_controls,
            prior_control_ensemble=prior_ensemble,
            parameter_ensemble=parameter_ensemble,
            trajectory_ensemble=posterior_trajectories,
            prior_trajectory_ensemble=prior_trajectories,
            center_control=center_control,
            center_trajectory=center_trajectory,
            correction_translation=np.asarray(correction_translation),
            correction_rotation_vector=np.asarray(correction_rotation),
            ridge=ridge,
            iterations=tuple(diagnostics),
            ensemble_rank=rank,
            converged=converged,
            termination_reason=termination_reason,
        )
