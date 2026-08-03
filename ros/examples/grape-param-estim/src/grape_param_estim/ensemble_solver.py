"""Minimal ensemble-space IEnKS core shared by single and joint problems."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import warnings

import numpy as np


ResidualBatch = Callable[[np.ndarray], np.ndarray]
ProgressCallback = Callable[[str, int, int, str], None]
CancelCallback = Callable[[], bool]


class EstimationCancelled(RuntimeError):
    """Raised at a forecast boundary after a cooperative cancel request."""


class InitialPriorForecastError(RuntimeError):
    """Every audited initial-prior radial-backoff attempt failed."""


@dataclass(frozen=True)
class EnsembleSpaceIteration:
    iteration: int
    objective: float
    accepted_objective: float
    gradient_norm: float
    step_norm: float
    accepted_fraction: float


@dataclass(frozen=True)
class InitialPriorForecastFailure:
    radial_scale: float
    exception_type: str
    reason: str


@dataclass(frozen=True)
class InitialPriorForecastDiagnostics:
    radial_scale: float
    backoff_trials: int
    maximum_backoff_trials: int
    requested_rank: int
    effective_rank: int
    failures: Tuple[InitialPriorForecastFailure, ...]


@dataclass(frozen=True)
class EnsembleSpaceResult:
    requested_prior_ensemble: np.ndarray
    prior_ensemble: np.ndarray
    posterior_ensemble: np.ndarray
    center_control: np.ndarray
    prior_residuals: np.ndarray
    posterior_residuals: np.ndarray
    center_residual: np.ndarray
    iterations: Tuple[EnsembleSpaceIteration, ...]
    ensemble_rank: int
    converged: bool
    termination_reason: str
    initial_prior_forecast: InitialPriorForecastDiagnostics


def _inverse_symmetric_sqrt(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("ensemble-space matrix must be positive definite")
    return (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )


def _validate_residuals(
    residuals: np.ndarray, member_count: int, name: str
) -> np.ndarray:
    values = np.asarray(residuals, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] != member_count
        or values.shape[1] == 0
        or np.any(~np.isfinite(values))
    ):
        raise ValueError(
            "{} must return one finite residual row per control".format(name)
        )
    return values


def _ensemble_geometry(ensemble: np.ndarray):
    member_count, dimension = ensemble.shape
    mean = np.mean(ensemble, axis=0)
    deviations = (ensemble - mean).T
    left, singular_values, right_transpose = np.linalg.svd(
        deviations, full_matrices=False
    )
    singular_tolerance = (
        max(deviations.shape)
        * np.finfo(float).eps
        * singular_values[0]
    )
    rank = int(np.count_nonzero(singular_values > singular_tolerance))
    expected_rank = min(dimension, member_count - 1)
    if rank != expected_rank:
        raise ValueError("prior ensemble has a degenerate ensemble subspace")
    basis = (
        left[:, :rank]
        * singular_values[:rank][None, :]
        / np.sqrt(member_count - 1.0)
    )
    ensemble_coordinates = (
        np.sqrt(member_count - 1.0) * right_transpose[:rank]
    )
    return mean, basis, ensemble_coordinates, rank


def run_ensemble_space_ienks(
    prior_ensemble: np.ndarray,
    forecast_residual_batch: ResidualBatch,
    configuration,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_requested: Optional[CancelCallback] = None,
) -> EnsembleSpaceResult:
    """Solve one nonlinear least-squares problem in the prior ensemble span.

    ``forecast_residual_batch`` receives controls with shape ``(M, D)`` and
    returns whitened pose residuals with shape ``(M, R)``.  The solver knows
    nothing about bags or trajectories; single-flight and joint callers keep
    those data-model choices outside this numerical core.
    """

    requested_prior = np.asarray(prior_ensemble, dtype=float)
    if (
        requested_prior.ndim != 2
        or requested_prior.shape[0] < 3
        or requested_prior.shape[1] < 1
        or np.any(~np.isfinite(requested_prior))
    ):
        raise ValueError(
            "prior ensemble must contain at least three finite members"
        )
    maximum_iterations = int(configuration.maximum_iterations)
    tolerance = float(configuration.convergence_tolerance)
    minimum_fraction = float(configuration.minimum_line_search_step)
    raw_maximum_initial_backoff_trials = getattr(
        configuration, "maximum_initial_prior_backoff_trials", 8
    )
    if isinstance(raw_maximum_initial_backoff_trials, (bool, np.bool_)):
        raise ValueError("invalid maximum initial-prior backoff trials")
    maximum_initial_backoff_trials = int(
        raw_maximum_initial_backoff_trials
    )
    if maximum_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("invalid IEnKS iteration configuration")
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("invalid minimum line-search step")
    if (
        maximum_initial_backoff_trials != raw_maximum_initial_backoff_trials
        or not 0 <= maximum_initial_backoff_trials <= 30
    ):
        raise ValueError("invalid maximum initial-prior backoff trials")

    def check_cancel() -> None:
        if cancel_requested is not None and bool(cancel_requested()):
            raise EstimationCancelled("estimation cancelled at forecast boundary")

    completed_batches = 0
    maximum_line_search_trials = int(
        np.floor(np.log2(1.0 / minimum_fraction) + 1.0e-12)
    ) + 1
    total_batches = (
        1
        + maximum_iterations * (1 + maximum_line_search_trials)
        + 3
        + maximum_initial_backoff_trials
    )

    def forecast(
        controls: np.ndarray, label: str, selected_stage_id: Optional[str] = None
    ) -> np.ndarray:
        nonlocal completed_batches
        check_cancel()
        if progress_callback is not None:
            if selected_stage_id is not None:
                stage_id = selected_stage_id
            elif "line search" in label:
                stage_id = "line_search_trial"
            elif label.startswith("posterior"):
                stage_id = "posterior_ensemble_forecast"
            elif label == "initial center":
                stage_id = "initial_forecast"
            else:
                stage_id = "ensemble_forecast"
            progress_callback(
                stage_id,
                completed_batches,
                total_batches,
                label,
            )
        try:
            return _validate_residuals(
                forecast_residual_batch(controls), controls.shape[0], label
            )
        finally:
            completed_batches += 1

    member_count, _dimension = requested_prior.shape
    requested_mean, _requested_basis, _requested_coordinates, requested_rank = (
        _ensemble_geometry(requested_prior)
    )

    def controls(coordinates: np.ndarray) -> np.ndarray:
        return (mean[:, None] + basis @ coordinates).T

    def objective(coordinate: np.ndarray, residual: np.ndarray) -> float:
        return 0.5 * float(
            np.dot(coordinate, coordinate) + np.dot(residual, residual)
        )

    def regression(
        residual_rows: np.ndarray, coordinate_deviations: np.ndarray
    ) -> np.ndarray:
        residual_columns = residual_rows.T
        residual_deviations = residual_columns - np.mean(
            residual_columns, axis=1, keepdims=True
        )
        gram = coordinate_deviations @ coordinate_deviations.T
        return (
            residual_deviations
            @ coordinate_deviations.T
            @ np.linalg.inv(gram)
        )

    center_residual = forecast(requested_mean[None, :], "initial center")[0]
    initial_failures = []
    prior = None
    prior_residuals = None
    selected_scale = None
    initial_numeric_errors = (
        ValueError,
        FloatingPointError,
        RuntimeWarning,
        np.linalg.LinAlgError,
    )
    for backoff_trial in range(maximum_initial_backoff_trials + 1):
        scale = float(2.0 ** (-backoff_trial))
        candidate_prior = requested_mean[None, :] + scale * (
            requested_prior - requested_mean[None, :]
        )
        if backoff_trial == 0:
            label = "iteration 1/{} ensemble".format(maximum_iterations)
            stage_id = None
        else:
            label = (
                "iteration 1/{} initial-prior backoff {}/{} "
                "(radial scale {:.12g})"
            ).format(
                maximum_iterations,
                backoff_trial,
                maximum_initial_backoff_trials,
                scale,
            )
            stage_id = "initial_prior_backoff_forecast"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                candidate_residuals = forecast(
                    candidate_prior, label, selected_stage_id=stage_id
                )
        except initial_numeric_errors as error:
            failure = InitialPriorForecastFailure(
                radial_scale=scale,
                exception_type=type(error).__name__,
                reason=str(error) or type(error).__name__,
            )
            initial_failures.append(failure)
            if progress_callback is not None:
                progress_callback(
                    "initial_prior_forecast_failed",
                    completed_batches,
                    total_batches,
                    (
                        "iteration 1/{} initial-prior attempt failed at "
                        "radial scale {:.12g}: {}: {}"
                    ).format(
                        maximum_iterations,
                        scale,
                        failure.exception_type,
                        failure.reason,
                    ),
                )
            if backoff_trial == maximum_initial_backoff_trials:
                attempted = "; ".join(
                    "scale {:.12g} {}: {}".format(
                        value.radial_scale,
                        value.exception_type,
                        value.reason,
                    )
                    for value in initial_failures
                )
                raise InitialPriorForecastError(
                    "initial prior ensemble remained non-finite after {} "
                    "global radial attempts ({})".format(
                        len(initial_failures), attempted
                    )
                ) from error
            continue
        prior = candidate_prior
        prior_residuals = candidate_residuals.copy()
        selected_scale = scale
        break
    if prior is None or prior_residuals is None or selected_scale is None:
        raise AssertionError("initial-prior radial backoff did not terminate")

    mean, basis, ensemble_coordinates, rank = _ensemble_geometry(prior)
    if rank != requested_rank:
        raise AssertionError("initial-prior radial backoff changed ensemble rank")

    center = np.zeros(rank, dtype=float)
    transform = np.eye(rank)
    current_objective = objective(center, center_residual)
    diagnostics = []
    converged = False
    termination_reason = "maximum_iterations"

    for iteration in range(maximum_iterations):
        if iteration == 0:
            local_residuals = prior_residuals
        else:
            local_coordinates = (
                center[:, None] + transform @ ensemble_coordinates
            )
            local_residuals = forecast(
                controls(local_coordinates),
                "iteration {}/{} ensemble".format(
                    iteration + 1, maximum_iterations
                ),
            )
        coordinate_deviations = transform @ ensemble_coordinates
        sensitivity = regression(local_residuals, coordinate_deviations)
        hessian = np.eye(rank) + sensitivity.T @ sensitivity
        gradient = center + sensitivity.T @ center_residual
        step = -np.linalg.solve(hessian, gradient)
        fraction = 1.0
        accepted = False
        candidate_residual = center_residual
        candidate_objective = current_objective
        while fraction >= minimum_fraction:
            check_cancel()
            candidate = center + fraction * step
            trial_control = mean + basis @ candidate
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    trial_residual = forecast(
                        trial_control[None, :],
                        "iteration {}/{} line search".format(
                            iteration + 1, maximum_iterations
                        ),
                    )[0]
                    trial_objective = objective(candidate, trial_residual)
            except (
                ValueError,
                FloatingPointError,
                RuntimeWarning,
                np.linalg.LinAlgError,
            ):
                fraction *= 0.5
                continue
            if trial_objective < current_objective:
                accepted = True
                candidate_residual = trial_residual
                candidate_objective = trial_objective
                break
            fraction *= 0.5
        diagnostics.append(
            EnsembleSpaceIteration(
                iteration=iteration,
                objective=current_objective,
                accepted_objective=candidate_objective,
                gradient_norm=float(np.linalg.norm(gradient)),
                step_norm=float(np.linalg.norm(step)),
                accepted_fraction=float(fraction if accepted else 0.0),
            )
        )
        transform = _inverse_symmetric_sqrt(hessian)
        if not accepted:
            termination_reason = "line_search_failed"
            break
        center = candidate
        center_residual = candidate_residual
        current_objective = candidate_objective
        if np.linalg.norm(fraction * step) <= tolerance * (
            1.0 + np.linalg.norm(center)
        ):
            converged = True
            termination_reason = "step_tolerance"
            break

    local_coordinates = center[:, None] + transform @ ensemble_coordinates
    local_residuals = forecast(
        controls(local_coordinates), "posterior linearization"
    )
    sensitivity = regression(
        local_residuals, transform @ ensemble_coordinates
    )
    posterior_hessian = np.eye(rank) + sensitivity.T @ sensitivity
    posterior_transform = _inverse_symmetric_sqrt(posterior_hessian)
    posterior_coordinates = (
        center[:, None] + posterior_transform @ ensemble_coordinates
    )
    posterior = controls(posterior_coordinates)
    posterior_residuals = forecast(posterior, "posterior ensemble")
    center_control = mean + basis @ center
    center_residual = forecast(center_control[None, :], "posterior center")[0]
    if progress_callback is not None:
        progress_callback(
            "posterior_ensemble_forecast",
            total_batches,
            total_batches,
            "complete",
        )
    return EnsembleSpaceResult(
        requested_prior_ensemble=requested_prior.copy(),
        prior_ensemble=prior.copy(),
        posterior_ensemble=posterior,
        center_control=center_control,
        prior_residuals=prior_residuals,
        posterior_residuals=posterior_residuals,
        center_residual=center_residual,
        iterations=tuple(diagnostics),
        ensemble_rank=rank,
        converged=converged,
        termination_reason=termination_reason,
        initial_prior_forecast=InitialPriorForecastDiagnostics(
            radial_scale=selected_scale,
            backoff_trials=len(initial_failures),
            maximum_backoff_trials=maximum_initial_backoff_trials,
            requested_rank=requested_rank,
            effective_rank=rank,
            failures=tuple(initial_failures),
        ),
    )


__all__ = [
    "EnsembleSpaceIteration",
    "EnsembleSpaceResult",
    "EstimationCancelled",
    "InitialPriorForecastDiagnostics",
    "InitialPriorForecastError",
    "InitialPriorForecastFailure",
    "run_ensemble_space_ienks",
]
