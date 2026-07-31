"""Full-block weak-constraint IEnKS-Q for additive residual wrench."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.controller import GrapeController
from grape_param_estim.dynamics import FullSixDofPlant, simulate_closed_loop
from grape_param_estim.geometry import correction_transform_path
from grape_param_estim.model_error import GaussMarkovWrenchProcess
from grape_param_estim.strong_constraint import (
    CONTROL_DIMENSION,
    PARAMETER_OFFSET,
    IEnKSConfig,
    IEnKSIteration,
    ParameterRidge,
    StaticParameterEnsemble,
    StrongConstraintIEnKS,
    StrongConstraintPrior,
    StrongConstraintProblem,
    _symmetric_inverse_sqrt,
)
from grape_param_estim.system import ClosedLoopTrajectory


@dataclass(frozen=True)
class WeakConstraintPrior:
    """Independent proper priors for static controls and every Q block."""

    static_prior: StrongConstraintPrior
    wrench_process: GaussMarkovWrenchProcess

    @property
    def control_dimension(self) -> int:
        return CONTROL_DIMENSION + self.wrench_process.innovation_dimension

    def ensemble(self, size: int, seed: int) -> np.ndarray:
        """Sample every augmented coordinate with exact sample covariance."""

        member_count = int(size)
        dimension = self.control_dimension
        if member_count <= dimension:
            raise ValueError(
                "IEnKS-Q ensemble size must exceed augmented control dimension"
            )
        generator = np.random.RandomState(int(seed))
        standard = generator.normal(size=(dimension, member_count))
        standard -= np.mean(standard, axis=1, keepdims=True)
        covariance = standard @ standard.T / (member_count - 1.0)
        standard = _symmetric_inverse_sqrt(covariance) @ standard
        values = standard
        static_factor = np.linalg.cholesky(self.static_prior.covariance)
        values[:CONTROL_DIMENSION] = (
            self.static_prior.mean[:, None]
            + static_factor @ standard[:CONTROL_DIMENSION]
        )
        # Remaining rows are independent N(0, 1) innovation blocks.  Because
        # the complete sample covariance was whitened jointly, their sample
        # cross-covariance with the static block is exactly zero.
        return values.T


@dataclass(frozen=True)
class WeakConstraintProblem:
    """A full-window Grape problem with one independent Q block per interval."""

    strong_problem: StrongConstraintProblem
    wrench_process: GaussMarkovWrenchProcess

    def __post_init__(self) -> None:
        expected_times = self.strong_problem.observations.times[:-1]
        if not np.array_equal(self.wrench_process.times, expected_times):
            raise ValueError(
                "wrench process must provide one value per integration interval"
            )

    @property
    def control_dimension(self) -> int:
        return CONTROL_DIMENSION + self.wrench_process.innovation_dimension

    @property
    def nominal_trajectory(self):
        return self.strong_problem.nominal_trajectory

    @property
    def parameter_chart(self):
        return self.strong_problem.parameter_chart

    def decode_control(self, control: Sequence[float]):
        value = np.asarray(control, dtype=float)
        if (
            value.shape != (self.control_dimension,)
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(
                "weak control must contain {} finite values".format(
                    self.control_dimension
                )
            )
        state, controller_state, parameters = (
            self.strong_problem.decode_control(value[:CONTROL_DIMENSION])
        )
        residual_wrench = self.wrench_process.decode(
            value[CONTROL_DIMENSION:]
        )
        return state, controller_state, parameters, residual_wrench

    def forecast(self, control: Sequence[float]) -> ClosedLoopTrajectory:
        state, controller_state, parameters, residual_wrench = (
            self.decode_control(control)
        )
        base = self.strong_problem
        controller = GrapeController(
            base.controller_configuration,
            base.controller_parameters,
            base.geometry,
            articulated_model=GrapeArticulatedModel(),
        )
        return simulate_closed_loop(
            times=base.observations.times,
            references=base.references,
            initial_state=state,
            initial_controller_state=controller_state,
            controller=controller,
            plant=FullSixDofPlant(parameters, base.geometry),
            actuator_parameters=base.actuator_parameters,
            interval_residual_wrench=residual_wrench,
        )

    def residual(self, trajectory: ClosedLoopTrajectory) -> np.ndarray:
        return self.strong_problem.residual(trajectory)


@dataclass(frozen=True)
class WeakConstraintPosterior:
    """Member-aligned static, Q, trajectory and transform path ensembles."""

    control_ensemble: np.ndarray
    prior_control_ensemble: np.ndarray
    parameter_ensemble: StaticParameterEnsemble
    innovation_ensemble: np.ndarray
    residual_wrench_ensemble: np.ndarray
    trajectory_ensemble: Tuple[ClosedLoopTrajectory, ...]
    prior_trajectory_ensemble: Tuple[ClosedLoopTrajectory, ...]
    center_control: np.ndarray
    center_trajectory: ClosedLoopTrajectory
    center_residual_wrench: np.ndarray
    correction_translation: np.ndarray
    correction_rotation_vector: np.ndarray
    ridge: ParameterRidge
    iterations: Tuple[IEnKSIteration, ...]
    ensemble_rank: int
    converged: bool
    termination_reason: str


class WeakConstraintIEnKSQ:
    """IEnKS-Q with explicit independent innovation blocks at every interval."""

    def __init__(self, configuration: Optional[IEnKSConfig] = None):
        self.configuration = configuration

    def fit(
        self,
        problem: WeakConstraintProblem,
        prior: Optional[WeakConstraintPrior] = None,
    ) -> WeakConstraintPosterior:
        selected_prior = prior or WeakConstraintPrior(
            static_prior=StrongConstraintPrior.grape(),
            wrench_process=problem.wrench_process,
        )
        # Dataclass equality with NumPy arrays is ambiguous, so compare each
        # factual process field explicitly.
        if (
            not np.array_equal(
                selected_prior.wrench_process.times,
                problem.wrench_process.times,
            )
            or not np.array_equal(
                selected_prior.wrench_process.stationary_standard_deviation,
                problem.wrench_process.stationary_standard_deviation,
            )
            or selected_prior.wrench_process.correlation_time
            != problem.wrench_process.correlation_time
        ):
            raise ValueError("prior and problem wrench processes must agree")
        configuration = self.configuration or IEnKSConfig(
            ensemble_size=problem.control_dimension + 2
        )
        if configuration.ensemble_size <= problem.control_dimension:
            raise ValueError(
                "full-block IEnKS-Q requires M greater than augmented dimension"
            )
        prior_ensemble = selected_prior.ensemble(
            configuration.ensemble_size, configuration.seed
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
        if rank != problem.control_dimension:
            raise ValueError(
                "IEnKS-Q prior ensemble must span every independent Q block"
            )
        basis = (
            left[:, :rank]
            * singular_values[:rank][None, :]
            / np.sqrt(configuration.ensemble_size - 1.0)
        )
        ensemble_coordinates = (
            np.sqrt(configuration.ensemble_size - 1.0)
            * right_transpose[:rank]
        )

        center = np.zeros(rank, dtype=float)
        transform = np.eye(rank)
        center_trajectory = problem.forecast(prior_mean)
        center_residual = problem.residual(center_trajectory)
        objective = StrongConstraintIEnKS._objective(
            center, center_residual
        )
        diagnostics = []
        prior_trajectories = None
        converged = False
        termination_reason = "maximum_iterations"

        for iteration in range(configuration.maximum_iterations):
            local_coordinates = (
                center[:, None] + transform @ ensemble_coordinates
            )
            local_controls = StrongConstraintIEnKS._controls(
                prior_mean, basis, local_coordinates
            )
            local_trajectories, local_residuals = (
                StrongConstraintIEnKS._forecast_controls(
                    problem, local_controls
                )
            )
            if prior_trajectories is None:
                prior_trajectories = local_trajectories
            coordinate_deviations = transform @ ensemble_coordinates
            sensitivity = StrongConstraintIEnKS._regression(
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
            while fraction >= configuration.minimum_line_search_step:
                candidate = center + fraction * step
                trial_control = prior_mean + basis @ candidate
                trial_trajectory = problem.forecast(trial_control)
                trial_residual = problem.residual(trial_trajectory)
                trial_objective = StrongConstraintIEnKS._objective(
                    candidate, trial_residual
                )
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
                configuration.convergence_tolerance
                * (1.0 + np.linalg.norm(center))
            ):
                converged = True
                termination_reason = "step_tolerance"
                break

        local_coordinates = center[:, None] + transform @ ensemble_coordinates
        local_controls = StrongConstraintIEnKS._controls(
            prior_mean, basis, local_coordinates
        )
        _local_trajectories, local_residuals = (
            StrongConstraintIEnKS._forecast_controls(problem, local_controls)
        )
        sensitivity = StrongConstraintIEnKS._regression(
            local_residuals, transform @ ensemble_coordinates
        )
        posterior_hessian = np.eye(rank) + sensitivity.T @ sensitivity
        posterior_transform = _symmetric_inverse_sqrt(posterior_hessian)
        posterior_coordinates = (
            center[:, None] + posterior_transform @ ensemble_coordinates
        )
        posterior_controls = StrongConstraintIEnKS._controls(
            prior_mean, basis, posterior_coordinates
        )
        posterior_trajectories, _posterior_residuals = (
            StrongConstraintIEnKS._forecast_controls(
                problem, posterior_controls
            )
        )
        center_control = prior_mean + basis @ center
        center_trajectory = problem.forecast(center_control)

        parameter_coordinates = posterior_controls[
            :, PARAMETER_OFFSET:CONTROL_DIMENSION
        ]
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
        innovation_ensemble = posterior_controls[:, CONTROL_DIMENSION:]
        residual_wrench_ensemble = np.asarray(
            [problem.wrench_process.decode(value) for value in innovation_ensemble]
        )
        center_residual_wrench = problem.wrench_process.decode(
            center_control[CONTROL_DIMENSION:]
        )
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
        parameter_covariance = np.cov(parameter_coordinates, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(parameter_covariance)
        expected_direction = problem.parameter_chart.ridge_direction()
        expected_direction /= np.linalg.norm(expected_direction)
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
            raise AssertionError("IEnKS-Q did not evaluate its prior ensemble")
        return WeakConstraintPosterior(
            control_ensemble=posterior_controls,
            prior_control_ensemble=prior_ensemble,
            parameter_ensemble=parameter_ensemble,
            innovation_ensemble=innovation_ensemble,
            residual_wrench_ensemble=residual_wrench_ensemble,
            trajectory_ensemble=posterior_trajectories,
            prior_trajectory_ensemble=prior_trajectories,
            center_control=center_control,
            center_trajectory=center_trajectory,
            center_residual_wrench=center_residual_wrench,
            correction_translation=np.asarray(correction_translation),
            correction_rotation_vector=np.asarray(correction_rotation),
            ridge=ridge,
            iterations=tuple(diagnostics),
            ensemble_rank=rank,
            converged=converged,
            termination_reason=termination_reason,
        )
