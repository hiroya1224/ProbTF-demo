"""Synthetic ground-truth validation of the exact Phase-2 parameter ridge.

The full closed-loop model has one exact common-scale invariance.  If mass,
the complete inertia tensor and every force-effectiveness coefficient are
multiplied by the same positive scale, the generated trajectory is unchanged.
This module validates that invariance and compares the raw IEnKS parameter and
correction-path members with the corresponding proper-prior ridge law.

No Gaussian fit replaces the ensemble here.  Covariances are used only for
coordinate-free diagnostics; all raw parameter, quotient and path samples are
retained in :class:`RidgeValidationReport`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from grape_param_estim.geometry import (
    quaternion_to_matrix,
    rotation_vector_from_matrix,
)
from grape_param_estim.strong_constraint import PARAMETER_OFFSET
from grape_param_estim.model_error import GaussMarkovWrenchProcess
from grape_param_estim.strong_constraint import (
    CONTROL_DIMENSION,
    IEnKSConfig,
    StrongConstraintPrior,
)
from grape_param_estim.strong_constraint_experiments import (
    Phase2Experiment,
    _problem_from_synthetic,
)
from grape_param_estim.weak_constraint import (
    WeakConstraintIEnKSQ,
    WeakConstraintPrior,
    WeakConstraintProblem,
)


def _finite_array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite {} array".format(name, shape))
    return result.copy()


def _symmetric_inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    symmetric = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("matrix must be positive definite")
    return (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )


def _lambda_and_quotient(
    samples: np.ndarray,
    reference: np.ndarray,
    inverse_prior_covariance: np.ndarray,
    direction: np.ndarray,
):
    """Split samples into the exact ridge coordinate and its quotient."""

    centered = samples - reference[None, :]
    inverse_direction = inverse_prior_covariance @ direction
    denominator = float(direction @ inverse_direction)
    ridge_coordinate = centered @ inverse_direction / denominator
    quotient = centered - np.outer(ridge_coordinate, direction)
    return ridge_coordinate, quotient, denominator


def _quotient_mahalanobis(
    samples: np.ndarray, truth: np.ndarray
) -> float:
    covariance = np.cov(samples, rowvar=False)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = float(max(1.0, eigenvalues[-1]))
    represented = eigenvalues > max(
        1.0e-12, np.finfo(float).eps * samples.shape[1] * scale
    )
    # One and only one parameter direction is removed by the quotient.
    if np.count_nonzero(represented) != samples.shape[1] - 1:
        raise ValueError("posterior quotient must represent 17 directions")
    difference = np.mean(samples, axis=0) - truth
    standard = (
        eigenvectors[:, represented].T @ difference
    ) / np.sqrt(eigenvalues[represented])
    return float(standard @ standard)


def _rotation_path_error(reference, candidate):
    result = np.empty((reference.shape[0], 3), dtype=float)
    for index in range(reference.shape[0]):
        reference_rotation = quaternion_to_matrix(reference[index])
        candidate_rotation = quaternion_to_matrix(candidate[index])
        result[index] = rotation_vector_from_matrix(
            reference_rotation.T @ candidate_rotation
        )
    return result


@dataclass(frozen=True)
class RidgeValidationReport:
    """Raw-member report for the exact synthetic parameter ridge."""

    experiment_label: str
    direction: np.ndarray
    rollout_lambdas: np.ndarray
    rollout_position_max_error: np.ndarray
    rollout_rotation_max_error: np.ndarray
    rollout_pose_cost: np.ndarray
    pose_cost_range: float
    prior_parameter_samples: np.ndarray
    posterior_parameter_samples: np.ndarray
    prior_lambda_samples: np.ndarray
    posterior_lambda_samples: np.ndarray
    prior_quotient_samples: np.ndarray
    posterior_quotient_samples: np.ndarray
    truth_quotient: np.ndarray
    theoretical_lambda_mean: float
    theoretical_lambda_variance: float
    posterior_lambda_mean_zscore: float
    posterior_lambda_variance_ratio: float
    lambda_wasserstein_ratio: float
    quotient_truth_mahalanobis: float
    prior_whitened_information_leak: float
    correction_translation_samples: np.ndarray
    correction_rotation_vector_samples: np.ndarray
    path_residual_samples: np.ndarray
    path_translation_coverage: float
    path_rotation_coverage: float
    path_component_coverage: float

    def __post_init__(self) -> None:
        vector_fields = (
            "direction",
            "rollout_lambdas",
            "rollout_position_max_error",
            "rollout_rotation_max_error",
            "rollout_pose_cost",
            "prior_lambda_samples",
            "posterior_lambda_samples",
            "truth_quotient",
        )
        matrix_or_path_fields = (
            "prior_parameter_samples",
            "posterior_parameter_samples",
            "prior_quotient_samples",
            "posterior_quotient_samples",
            "correction_translation_samples",
            "correction_rotation_vector_samples",
            "path_residual_samples",
        )
        for name in vector_fields + matrix_or_path_fields:
            value = np.asarray(getattr(self, name), dtype=float)
            if not np.all(np.isfinite(value)):
                raise ValueError("{} must contain finite values".format(name))
            object.__setattr__(self, name, value.copy())
        for name in (
            "pose_cost_range",
            "theoretical_lambda_mean",
            "theoretical_lambda_variance",
            "posterior_lambda_mean_zscore",
            "posterior_lambda_variance_ratio",
            "lambda_wasserstein_ratio",
            "quotient_truth_mahalanobis",
            "prior_whitened_information_leak",
            "path_translation_coverage",
            "path_rotation_coverage",
            "path_component_coverage",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError("{} must be finite".format(name))
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class WeakRidgeValidationReport:
    """Zero-realization IEnKS-Q ridge and augmented-symmetry audit."""

    direction: np.ndarray
    augmented_rollout_lambdas: np.ndarray
    augmented_position_max_error: np.ndarray
    augmented_rotation_max_error: np.ndarray
    augmented_pose_residual_max_error: np.ndarray
    prior_parameter_samples: np.ndarray
    posterior_parameter_samples: np.ndarray
    prior_lambda_samples: np.ndarray
    posterior_lambda_samples: np.ndarray
    posterior_innovation_samples: np.ndarray
    posterior_residual_wrench_samples: np.ndarray
    posterior_lambda_mean_zscore: float
    posterior_lambda_variance_ratio: float
    lambda_wasserstein_ratio: float
    prior_whitened_information_leak: float
    static_ridge_variance_ratio: float
    maximum_center_residual_wrench: float
    particle_correction_required: bool


def validate_phase2_ridge(
    experiment: Phase2Experiment,
    rollout_lambdas: Sequence[float] = (-0.30, -0.15, 0.0, 0.15, 0.30),
) -> RidgeValidationReport:
    """Validate one completed Phase-2 A/B experiment against its exact ridge."""

    if not isinstance(experiment, Phase2Experiment):
        raise TypeError("experiment must be a Phase2Experiment")
    lambdas = np.asarray(rollout_lambdas, dtype=float)
    if (
        lambdas.ndim != 1
        or lambdas.size < 3
        or not np.all(np.isfinite(lambdas))
        or not np.any(lambdas == 0.0)
    ):
        raise ValueError(
            "rollout_lambdas must be a finite vector containing zero"
        )

    posterior = experiment.posterior
    prior = experiment.prior
    problem = _problem_from_synthetic(experiment.synthetic)
    direction = problem.parameter_chart.ridge_direction()
    direction = direction / np.linalg.norm(direction)

    base_control = _finite_array(
        experiment.truth_control,
        (posterior.control_ensemble.shape[1],),
        "truth control",
    )
    trajectories = []
    pose_cost = np.empty(lambdas.size, dtype=float)
    for index, ridge_coordinate in enumerate(lambdas):
        control = base_control.copy()
        control[PARAMETER_OFFSET:] += ridge_coordinate * direction
        trajectory = problem.forecast(control)
        trajectories.append(trajectory)
        residual = problem.residual(trajectory)
        pose_cost[index] = 0.5 * float(residual @ residual)
    zero_index = int(np.flatnonzero(lambdas == 0.0)[0])
    baseline = trajectories[zero_index]
    position_error = np.asarray(
        [
            np.max(np.linalg.norm(value.position - baseline.position, axis=1))
            for value in trajectories
        ]
    )
    rotation_error = np.asarray(
        [
            np.max(
                np.linalg.norm(
                    _rotation_path_error(
                        baseline.orientation_xyzw, value.orientation_xyzw
                    ),
                    axis=1,
                )
            )
            for value in trajectories
        ]
    )

    prior_parameters = posterior.prior_control_ensemble[
        :, PARAMETER_OFFSET:
    ]
    posterior_parameters = posterior.parameter_ensemble.coordinates
    parameter_prior_mean = prior.mean[PARAMETER_OFFSET:]
    parameter_prior_covariance = prior.covariance[
        PARAMETER_OFFSET:, PARAMETER_OFFSET:
    ]
    inverse_prior = np.linalg.inv(parameter_prior_covariance)
    prior_lambda, prior_quotient, denominator = _lambda_and_quotient(
        prior_parameters,
        parameter_prior_mean,
        inverse_prior,
        direction,
    )
    posterior_lambda, posterior_quotient, _ = _lambda_and_quotient(
        posterior_parameters,
        parameter_prior_mean,
        inverse_prior,
        direction,
    )
    truth_parameter = base_control[PARAMETER_OFFSET:]
    _truth_lambda, truth_quotient_batch, _ = _lambda_and_quotient(
        truth_parameter[None, :],
        parameter_prior_mean,
        inverse_prior,
        direction,
    )
    truth_quotient = truth_quotient_batch[0]
    theoretical_mean = 0.0
    theoretical_variance = 1.0 / denominator
    lambda_standard_deviation = np.sqrt(theoretical_variance)
    posterior_mean_zscore = abs(
        float(np.mean(posterior_lambda)) - theoretical_mean
    ) / lambda_standard_deviation
    posterior_variance_ratio = float(
        np.var(posterior_lambda, ddof=1) / theoretical_variance
    )
    lambda_wasserstein = float(
        np.mean(
            np.abs(
                np.sort(prior_lambda) - np.sort(posterior_lambda)
            )
        )
        / lambda_standard_deviation
    )

    posterior_covariance = np.cov(posterior_parameters, rowvar=False)
    inverse_sqrt_prior = _symmetric_inverse_sqrt(
        parameter_prior_covariance
    )
    whitened_covariance = (
        inverse_sqrt_prior
        @ posterior_covariance
        @ inverse_sqrt_prior
    )
    whitened_direction = inverse_sqrt_prior @ direction
    whitened_direction /= np.linalg.norm(whitened_direction)
    information_leak = float(
        np.linalg.norm(
            (np.eye(direction.size) - whitened_covariance)
            @ whitened_direction
        )
    )

    correction_translation = posterior.correction_translation
    correction_rotation = posterior.correction_rotation_vector
    translation_residual = (
        correction_translation
        - experiment.synthetic.correction_translation[None, :, :]
    )
    rotation_residual = np.asarray(
        [
            _rotation_path_error(
                experiment.synthetic.truth.orientation_xyzw,
                trajectory.orientation_xyzw,
            )
            for trajectory in posterior.trajectory_ensemble
        ]
    )
    path_residual = np.concatenate(
        (translation_residual, rotation_residual), axis=2
    )
    lower = np.percentile(path_residual, 2.5, axis=0)
    upper = np.percentile(path_residual, 97.5, axis=0)
    covered = (lower <= 0.0) & (upper >= 0.0)

    return RidgeValidationReport(
        experiment_label=experiment.label,
        direction=direction,
        rollout_lambdas=lambdas,
        rollout_position_max_error=position_error,
        rollout_rotation_max_error=rotation_error,
        rollout_pose_cost=pose_cost,
        pose_cost_range=float(np.ptp(pose_cost)),
        prior_parameter_samples=prior_parameters,
        posterior_parameter_samples=posterior_parameters,
        prior_lambda_samples=prior_lambda,
        posterior_lambda_samples=posterior_lambda,
        prior_quotient_samples=prior_quotient,
        posterior_quotient_samples=posterior_quotient,
        truth_quotient=truth_quotient,
        theoretical_lambda_mean=theoretical_mean,
        theoretical_lambda_variance=theoretical_variance,
        posterior_lambda_mean_zscore=posterior_mean_zscore,
        posterior_lambda_variance_ratio=posterior_variance_ratio,
        lambda_wasserstein_ratio=lambda_wasserstein,
        quotient_truth_mahalanobis=_quotient_mahalanobis(
            posterior_quotient, truth_quotient
        ),
        prior_whitened_information_leak=information_leak,
        correction_translation_samples=correction_translation,
        correction_rotation_vector_samples=correction_rotation,
        path_residual_samples=path_residual,
        path_translation_coverage=float(np.mean(covered[:, :3])),
        path_rotation_coverage=float(np.mean(covered[:, 3:])),
        path_component_coverage=float(np.mean(covered)),
    )


def validate_weak_zero_realization_ridge(
    experiment: Phase2Experiment,
    maximum_iterations: int = 1,
    seed: int = 11,
    rollout_lambdas: Sequence[float] = (-0.30, 0.0, 0.30),
) -> WeakRidgeValidationReport:
    """Verify that IEnKS-Q preserves the ridge in its zero-Q realization.

    For a nonzero residual path the exact likelihood symmetry is augmented:
    adding ``lambda * v`` to static chart coordinates must be accompanied by
    multiplying every standard innovation, and hence every residual wrench,
    by ``exp(lambda)``.  The perfect-model Experiment-A realization keeps the
    posterior near zero residual, where its static proper-prior lambda law
    must also remain unaltered.
    """

    if not isinstance(experiment, Phase2Experiment):
        raise TypeError("experiment must be a Phase2Experiment")
    if experiment.label != "A":
        raise ValueError("weak ridge validation requires Experiment A")
    lambdas = np.asarray(rollout_lambdas, dtype=float)
    if (
        lambdas.ndim != 1
        or lambdas.size < 3
        or not np.all(np.isfinite(lambdas))
        or not np.any(lambdas == 0.0)
    ):
        raise ValueError("rollout_lambdas must contain a finite zero")
    strong_problem = _problem_from_synthetic(experiment.synthetic)
    process = GaussMarkovWrenchProcess(
        times=experiment.synthetic.observations.times[:-1],
        stationary_standard_deviation=np.asarray(
            (0.05, 0.05, 0.05, 0.005, 0.005, 0.005)
        ),
        correlation_time=0.35,
    )
    problem = WeakConstraintProblem(strong_problem, process)
    prior = StrongConstraintPrior.grape()
    posterior = WeakConstraintIEnKSQ(
        IEnKSConfig(
            ensemble_size=problem.control_dimension + 2,
            maximum_iterations=int(maximum_iterations),
            seed=int(seed),
        )
    ).fit(problem, WeakConstraintPrior(prior, process))

    raw_direction = strong_problem.parameter_chart.ridge_direction()
    direction = raw_direction / np.linalg.norm(raw_direction)
    parameter_prior_mean = prior.mean[PARAMETER_OFFSET:]
    parameter_prior_covariance = prior.covariance[
        PARAMETER_OFFSET:, PARAMETER_OFFSET:
    ]
    inverse_prior = np.linalg.inv(parameter_prior_covariance)
    prior_parameters = posterior.prior_control_ensemble[
        :, PARAMETER_OFFSET:CONTROL_DIMENSION
    ]
    posterior_parameters = posterior.parameter_ensemble.coordinates
    prior_lambda, _prior_quotient, denominator = _lambda_and_quotient(
        prior_parameters,
        parameter_prior_mean,
        inverse_prior,
        direction,
    )
    posterior_lambda, _posterior_quotient, _ = _lambda_and_quotient(
        posterior_parameters,
        parameter_prior_mean,
        inverse_prior,
        direction,
    )
    theoretical_variance = 1.0 / denominator
    lambda_sigma = np.sqrt(theoretical_variance)
    lambda_mean_zscore = abs(float(np.mean(posterior_lambda))) / lambda_sigma
    lambda_variance_ratio = float(
        np.var(posterior_lambda, ddof=1) / theoretical_variance
    )
    lambda_wasserstein_ratio = float(
        np.mean(
            np.abs(
                np.sort(prior_lambda) - np.sort(posterior_lambda)
            )
        )
        / lambda_sigma
    )
    posterior_covariance = np.cov(posterior_parameters, rowvar=False)
    inverse_sqrt_prior = _symmetric_inverse_sqrt(
        parameter_prior_covariance
    )
    whitened_covariance = (
        inverse_sqrt_prior
        @ posterior_covariance
        @ inverse_sqrt_prior
    )
    whitened_direction = inverse_sqrt_prior @ direction
    whitened_direction /= np.linalg.norm(whitened_direction)
    information_leak = float(
        np.linalg.norm(
            (np.eye(direction.size) - whitened_covariance)
            @ whitened_direction
        )
    )
    static_ridge_variance_ratio = float(
        posterior.ridge.expected_variance
        / (
            direction
            @ parameter_prior_covariance
            @ direction
        )
    )

    trajectories = []
    pose_residuals = []
    for ridge_coordinate in lambdas:
        control = posterior.center_control.copy()
        control[PARAMETER_OFFSET:CONTROL_DIMENSION] += (
            ridge_coordinate * raw_direction
        )
        control[CONTROL_DIMENSION:] *= np.exp(ridge_coordinate)
        trajectory = problem.forecast(control)
        trajectories.append(trajectory)
        pose_residuals.append(problem.residual(trajectory))
    zero_index = int(np.flatnonzero(lambdas == 0.0)[0])
    baseline = trajectories[zero_index]
    baseline_residual = pose_residuals[zero_index]
    position_error = np.asarray(
        [
            np.max(np.linalg.norm(value.position - baseline.position, axis=1))
            for value in trajectories
        ]
    )
    rotation_error = np.asarray(
        [
            np.max(
                np.linalg.norm(
                    _rotation_path_error(
                        baseline.orientation_xyzw, value.orientation_xyzw
                    ),
                    axis=1,
                )
            )
            for value in trajectories
        ]
    )
    pose_residual_error = np.asarray(
        [
            np.max(np.abs(value - baseline_residual))
            for value in pose_residuals
        ]
    )
    correction_required = bool(
        np.max(position_error) >= 3.0e-12
        or np.max(rotation_error) >= 3.0e-12
        or np.max(pose_residual_error) >= 1.0e-8
        or not 0.80 < lambda_variance_ratio < 1.20
        or lambda_wasserstein_ratio >= 0.05
        or information_leak >= 0.10
    )
    return WeakRidgeValidationReport(
        direction=direction,
        augmented_rollout_lambdas=lambdas,
        augmented_position_max_error=position_error,
        augmented_rotation_max_error=rotation_error,
        augmented_pose_residual_max_error=pose_residual_error,
        prior_parameter_samples=prior_parameters,
        posterior_parameter_samples=posterior_parameters,
        prior_lambda_samples=prior_lambda,
        posterior_lambda_samples=posterior_lambda,
        posterior_innovation_samples=posterior.innovation_ensemble,
        posterior_residual_wrench_samples=(
            posterior.residual_wrench_ensemble
        ),
        posterior_lambda_mean_zscore=lambda_mean_zscore,
        posterior_lambda_variance_ratio=lambda_variance_ratio,
        lambda_wasserstein_ratio=lambda_wasserstein_ratio,
        prior_whitened_information_leak=information_leak,
        static_ridge_variance_ratio=static_ridge_variance_ratio,
        maximum_center_residual_wrench=float(
            np.max(np.abs(posterior.center_residual_wrench))
        ),
        particle_correction_required=correction_required,
    )


def save_ridge_validation(
    path: str,
    reports: Sequence[RidgeValidationReport],
    weak_report: WeakRidgeValidationReport = None,
) -> Path:
    """Save raw ridge laws and diagnostics without Python-object arrays."""

    selected = tuple(reports)
    if not selected:
        raise ValueError("at least one ridge report is required")
    if len({value.experiment_label for value in selected}) != len(selected):
        raise ValueError("ridge report labels must be unique")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": np.asarray(("grape-weak-constraint/phase4-ridge",)),
        "experiment_label": np.asarray(
            [value.experiment_label for value in selected]
        ),
    }
    array_fields = (
        "direction",
        "rollout_lambdas",
        "rollout_position_max_error",
        "rollout_rotation_max_error",
        "rollout_pose_cost",
        "prior_parameter_samples",
        "posterior_parameter_samples",
        "prior_lambda_samples",
        "posterior_lambda_samples",
        "prior_quotient_samples",
        "posterior_quotient_samples",
        "truth_quotient",
        "correction_translation_samples",
        "correction_rotation_vector_samples",
        "path_residual_samples",
    )
    scalar_fields = (
        "pose_cost_range",
        "theoretical_lambda_mean",
        "theoretical_lambda_variance",
        "posterior_lambda_mean_zscore",
        "posterior_lambda_variance_ratio",
        "lambda_wasserstein_ratio",
        "quotient_truth_mahalanobis",
        "prior_whitened_information_leak",
        "path_translation_coverage",
        "path_rotation_coverage",
        "path_component_coverage",
    )
    for index, report in enumerate(selected):
        prefix = "report_{}_".format(index)
        for name in array_fields:
            payload[prefix + name] = np.asarray(getattr(report, name))
        for name in scalar_fields:
            payload[prefix + name] = np.asarray((getattr(report, name),))
    if weak_report is not None:
        weak_array_fields = (
            "direction",
            "augmented_rollout_lambdas",
            "augmented_position_max_error",
            "augmented_rotation_max_error",
            "augmented_pose_residual_max_error",
            "prior_parameter_samples",
            "posterior_parameter_samples",
            "prior_lambda_samples",
            "posterior_lambda_samples",
            "posterior_innovation_samples",
            "posterior_residual_wrench_samples",
        )
        weak_scalar_fields = (
            "posterior_lambda_mean_zscore",
            "posterior_lambda_variance_ratio",
            "lambda_wasserstein_ratio",
            "prior_whitened_information_leak",
            "static_ridge_variance_ratio",
            "maximum_center_residual_wrench",
            "particle_correction_required",
        )
        for name in weak_array_fields:
            payload["weak_" + name] = np.asarray(
                getattr(weak_report, name)
            )
        for name in weak_scalar_fields:
            payload["weak_" + name] = np.asarray(
                (getattr(weak_report, name),)
            )
    np.savez_compressed(str(destination), **payload)
    return destination


__all__ = [
    "RidgeValidationReport",
    "WeakRidgeValidationReport",
    "save_ridge_validation",
    "validate_phase2_ridge",
    "validate_weak_zero_realization_ridge",
]
