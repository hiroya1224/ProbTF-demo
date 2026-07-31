"""Raw-ensemble convergence diagnostics for Phase 4.

The posterior ensemble, rather than a fitted Gaussian, is the canonical law.
This module compares two numerical resolutions after removing the exact
static-parameter ridge and whitening parameter/path coordinates by the
declared prior and pose covariances.  Deterministic sliced Wasserstein-1
distances then compare the complete empirical member laws, including each
member's full correction-transform path.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.model_error import GaussMarkovWrenchProcess
from grape_param_estim.strong_constraint import (
    CONTROL_DIMENSION,
    PARAMETER_OFFSET,
    IEnKSConfig,
    StrongConstraintIEnKS,
    StrongConstraintPrior,
)
from grape_param_estim.strong_constraint_experiments import (
    _problem_from_synthetic,
)
from grape_param_estim.synthetic import (
    SyntheticExperiment,
    run_synthetic_experiment,
)
from grape_param_estim.weak_constraint import (
    WeakConstraintIEnKSQ,
    WeakConstraintPrior,
    WeakConstraintProblem,
)
from grape_param_estim.weak_constraint_experiments import (
    _static_truth_without_model_error,
    oracle_effective_residual_wrench,
)


@dataclass(frozen=True)
class RawEnsembleLaw:
    """Member-aligned raw law and secondary convergence diagnostics."""

    method: str
    ensemble_size: int
    control_dimension: int
    parameter_coordinates: np.ndarray
    identifiable_quotient: np.ndarray
    whitened_identifiable_quotient: np.ndarray
    correction_translation: np.ndarray
    correction_rotation_vector: np.ndarray
    whitened_correction_path: np.ndarray
    ridge_variance_ratio: float
    identifiable_mean_error: float
    path_mean_error: float
    nominal_path_error: float
    truth_path_component_coverage: float
    iteration_count: int
    initial_objective: float
    final_accepted_objective: float
    assimilation_reduced_objective: bool
    ridge_preserved: bool
    pose_mean_improved_over_nominal: bool
    truth_path_covered: bool


@dataclass(frozen=True)
class EnsembleSizeComparison:
    """Raw-law distance and qualitative changes between two resolutions."""

    method: str
    smaller_size: int
    larger_size: int
    identifiable_sliced_w1: float
    path_sliced_w1: float
    identifiable_mean_shift: float
    path_mean_shift: float
    ridge_variance_ratio_change: float
    path_coverage_change: float
    ridge_conclusion_stable: bool
    pose_mean_conclusion_stable: bool
    coverage_conclusion_stable: bool


@dataclass(frozen=True)
class WeakStrongConclusion:
    """Qualitative Experiment-C comparison at one resolution tier."""

    strong_size: int
    weak_size: int
    weak_has_lower_identifiable_bias: bool
    weak_has_lower_path_mean_error: bool
    weak_has_higher_path_coverage: bool

    @property
    def signature(self) -> Tuple[bool, bool, bool]:
        return (
            self.weak_has_lower_identifiable_bias,
            self.weak_has_lower_path_mean_error,
            self.weak_has_higher_path_coverage,
        )


@dataclass(frozen=True)
class EnsembleConvergenceReport:
    """Phase-4 strong/weak size sweep without Gaussian posterior fitting."""

    synthetic: SyntheticExperiment
    weak_control_dimension: int
    truth_static_coordinates: np.ndarray
    strong_laws: Tuple[RawEnsembleLaw, ...]
    weak_laws: Tuple[RawEnsembleLaw, ...]
    strong_endpoint_comparison: EnsembleSizeComparison
    weak_endpoint_comparison: EnsembleSizeComparison
    weak_strong_conclusions: Tuple[WeakStrongConclusion, ...]
    strong_size_conclusions_stable: bool
    weak_size_conclusions_stable: bool
    weak_strong_conclusion_stable: bool


def _finite_member_matrix(values, name):
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 2
        or result.shape[0] < 2
        or result.shape[1] < 1
        or not np.all(np.isfinite(result))
    ):
        raise ValueError("{} must be a finite member matrix".format(name))
    return result


def _deterministic_directions(dimension: int, count: int) -> np.ndarray:
    """Return fixed approximately uniform unit directions for sliced-W1."""

    dimension = int(dimension)
    count = int(count)
    if dimension <= 0 or count <= 0:
        raise ValueError("projection dimension and count must be positive")
    # A local fixed seed makes the diagnostic bit-reproducible and independent
    # of assimilation seeds.  It does not sample or alter either posterior.
    generator = np.random.RandomState(7919 + 104729 * dimension)
    directions = generator.normal(size=(count, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def empirical_wasserstein_1(first: Sequence[float], second: Sequence[float]):
    """Exact one-dimensional W1 for unequal, equally weighted samples."""

    first = np.sort(np.asarray(first, dtype=float))
    second = np.sort(np.asarray(second, dtype=float))
    if (
        first.ndim != 1
        or second.ndim != 1
        or first.size == 0
        or second.size == 0
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError("W1 inputs must be non-empty finite vectors")
    breakpoints = np.unique(
        np.concatenate(
            (
                np.arange(first.size + 1, dtype=float) / first.size,
                np.arange(second.size + 1, dtype=float) / second.size,
            )
        )
    )
    widths = np.diff(breakpoints)
    midpoints = 0.5 * (breakpoints[:-1] + breakpoints[1:])
    first_index = np.minimum(
        (midpoints * first.size).astype(int), first.size - 1
    )
    second_index = np.minimum(
        (midpoints * second.size).astype(int), second.size - 1
    )
    return float(
        np.sum(widths * np.abs(first[first_index] - second[second_index]))
    )


def deterministic_sliced_wasserstein_1(
    first: np.ndarray,
    second: np.ndarray,
    projection_count: int = 48,
) -> float:
    """Compare complete empirical laws on shared deterministic projections."""

    first = _finite_member_matrix(first, "first ensemble")
    second = _finite_member_matrix(second, "second ensemble")
    if first.shape[1] != second.shape[1]:
        raise ValueError("sliced-W1 ensembles must share a dimension")
    directions = _deterministic_directions(
        first.shape[1], int(projection_count)
    )
    first_projection = first @ directions.T
    second_projection = second @ directions.T
    distances = [
        empirical_wasserstein_1(
            first_projection[:, index], second_projection[:, index]
        )
        for index in range(directions.shape[0])
    ]
    return float(np.mean(distances))


def _prior_whitened_quotient(
    coordinates: np.ndarray,
    prior_covariance: np.ndarray,
    ridge_direction: np.ndarray,
):
    """Choose the minimum-prior-norm representative of every ridge class."""

    coordinates = np.asarray(coordinates, dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] < 1
        or coordinates.shape[1] < 1
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError("parameter ensemble must be a finite member matrix")
    covariance = np.asarray(prior_covariance, dtype=float)
    direction = np.asarray(ridge_direction, dtype=float)
    dimension = coordinates.shape[1]
    if covariance.shape != (dimension, dimension):
        raise ValueError("parameter prior covariance has the wrong shape")
    if direction.shape != (dimension,) or not np.all(np.isfinite(direction)):
        raise ValueError("ridge direction has the wrong shape")
    factor = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(factor, coordinates.T).T
    whitened_direction = np.linalg.solve(factor, direction)
    coefficients = (
        whitened @ whitened_direction
        / np.dot(whitened_direction, whitened_direction)
    )
    quotient = coordinates - coefficients[:, None] * direction[None, :]
    whitened_quotient = np.linalg.solve(factor, quotient.T).T
    return quotient, whitened_quotient


def _whiten_correction_path(
    translation: np.ndarray,
    rotation: np.ndarray,
    translation_covariance: np.ndarray,
    rotation_covariance: np.ndarray,
) -> np.ndarray:
    translation = np.asarray(translation, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    if (
        translation.ndim != 3
        or rotation.shape != translation.shape
        or translation.shape[2] != 3
        or not np.all(np.isfinite(translation))
        or not np.all(np.isfinite(rotation))
    ):
        raise ValueError("correction paths must have shape (M, N, 3)")
    translation_factor = np.linalg.cholesky(translation_covariance)
    rotation_factor = np.linalg.cholesky(rotation_covariance)
    translation_white = np.linalg.solve(
        translation_factor, translation.reshape(-1, 3).T
    ).T.reshape(translation.shape)
    rotation_white = np.linalg.solve(
        rotation_factor, rotation.reshape(-1, 3).T
    ).T.reshape(rotation.shape)
    return np.concatenate(
        (translation_white, rotation_white), axis=2
    ).reshape(translation.shape[0], -1)


def _law_snapshot(
    method,
    posterior,
    control_dimension,
    prior,
    synthetic,
    truth_static_coordinates,
):
    parameter_covariance = prior.covariance[
        PARAMETER_OFFSET:, PARAMETER_OFFSET:
    ]
    parameter_coordinates = (
        posterior.parameter_ensemble.coordinates.copy()
    )
    quotient, whitened_quotient = _prior_whitened_quotient(
        parameter_coordinates,
        parameter_covariance,
        posterior.ridge.expected_direction,
    )
    _truth_quotient, truth_whitened_quotient = (
        _prior_whitened_quotient(
            np.asarray(truth_static_coordinates, dtype=float)[None, :],
            parameter_covariance,
            posterior.ridge.expected_direction,
        )
    )
    translation = posterior.correction_translation.copy()
    rotation = posterior.correction_rotation_vector.copy()
    observations = synthetic.observations
    whitened_path = _whiten_correction_path(
        translation,
        rotation,
        observations.translation_covariance,
        observations.rotation_covariance,
    )
    truth_path = _whiten_correction_path(
        synthetic.correction_translation[None, :, :],
        synthetic.correction_rotation_vector[None, :, :],
        observations.translation_covariance,
        observations.rotation_covariance,
    )[0]
    quotient_scale = np.sqrt(max(1, whitened_quotient.shape[1] - 1))
    path_scale = np.sqrt(whitened_path.shape[1])
    identifiable_mean_error = float(
        np.linalg.norm(
            np.mean(whitened_quotient, axis=0)
            - truth_whitened_quotient[0]
        )
        / quotient_scale
    )
    path_mean_error = float(
        np.linalg.norm(np.mean(whitened_path, axis=0) - truth_path)
        / path_scale
    )
    nominal_path_error = float(np.linalg.norm(truth_path) / path_scale)
    lower = np.percentile(whitened_path, 2.5, axis=0)
    upper = np.percentile(whitened_path, 97.5, axis=0)
    coverage = float(np.mean((lower <= truth_path) & (truth_path <= upper)))
    ridge_direction = posterior.ridge.expected_direction
    prior_ridge_variance = float(
        ridge_direction @ parameter_covariance @ ridge_direction
    )
    ridge_ratio = float(
        posterior.ridge.expected_variance / prior_ridge_variance
    )
    return RawEnsembleLaw(
        method=str(method),
        ensemble_size=parameter_coordinates.shape[0],
        control_dimension=int(control_dimension),
        parameter_coordinates=parameter_coordinates,
        identifiable_quotient=quotient,
        whitened_identifiable_quotient=whitened_quotient,
        correction_translation=translation,
        correction_rotation_vector=rotation,
        whitened_correction_path=whitened_path,
        ridge_variance_ratio=ridge_ratio,
        identifiable_mean_error=identifiable_mean_error,
        path_mean_error=path_mean_error,
        nominal_path_error=nominal_path_error,
        truth_path_component_coverage=coverage,
        iteration_count=len(posterior.iterations),
        initial_objective=float(posterior.iterations[0].objective),
        final_accepted_objective=float(
            posterior.iterations[-1].accepted_objective
        ),
        assimilation_reduced_objective=bool(
            posterior.iterations[-1].accepted_objective
            < posterior.iterations[0].objective
        ),
        ridge_preserved=bool(ridge_ratio >= 0.75),
        pose_mean_improved_over_nominal=bool(
            path_mean_error < nominal_path_error
        ),
        truth_path_covered=bool(coverage >= 0.75),
    )


def _compare_endpoint_laws(
    smaller: RawEnsembleLaw,
    larger: RawEnsembleLaw,
    projection_count: int,
):
    if smaller.method != larger.method:
        raise ValueError("endpoint laws must use the same method")
    quotient_dimension = smaller.whitened_identifiable_quotient.shape[1]
    path_dimension = smaller.whitened_correction_path.shape[1]
    return EnsembleSizeComparison(
        method=smaller.method,
        smaller_size=smaller.ensemble_size,
        larger_size=larger.ensemble_size,
        identifiable_sliced_w1=deterministic_sliced_wasserstein_1(
            smaller.whitened_identifiable_quotient,
            larger.whitened_identifiable_quotient,
            projection_count,
        ),
        path_sliced_w1=deterministic_sliced_wasserstein_1(
            smaller.whitened_correction_path,
            larger.whitened_correction_path,
            projection_count,
        ),
        identifiable_mean_shift=float(
            np.linalg.norm(
                np.mean(smaller.whitened_identifiable_quotient, axis=0)
                - np.mean(larger.whitened_identifiable_quotient, axis=0)
            )
            / np.sqrt(max(1, quotient_dimension - 1))
        ),
        path_mean_shift=float(
            np.linalg.norm(
                np.mean(smaller.whitened_correction_path, axis=0)
                - np.mean(larger.whitened_correction_path, axis=0)
            )
            / np.sqrt(path_dimension)
        ),
        ridge_variance_ratio_change=float(
            abs(smaller.ridge_variance_ratio - larger.ridge_variance_ratio)
        ),
        path_coverage_change=float(
            abs(
                smaller.truth_path_component_coverage
                - larger.truth_path_component_coverage
            )
        ),
        ridge_conclusion_stable=(
            smaller.ridge_preserved == larger.ridge_preserved
        ),
        pose_mean_conclusion_stable=(
            smaller.pose_mean_improved_over_nominal
            == larger.pose_mean_improved_over_nominal
        ),
        coverage_conclusion_stable=(
            smaller.truth_path_covered == larger.truth_path_covered
        ),
    )


def _validated_sizes(values, minimum, name):
    result = tuple(sorted({int(value) for value in values}))
    if len(result) < 2 or result[0] <= int(minimum):
        raise ValueError(
            "{} needs at least two sizes greater than {}".format(
                name, minimum
            )
        )
    return result


def run_ensemble_size_convergence(
    duration: float = 0.28,
    time_step: float = 0.04,
    strong_sizes: Sequence[int] = (38, 46),
    weak_sizes: Optional[Sequence[int]] = None,
    maximum_iterations: int = 1,
    seed: int = 43,
    projection_count: int = 48,
) -> EnsembleConvergenceReport:
    """Assimilate one short Experiment-C episode at multiple real sizes."""

    synthetic = run_synthetic_experiment(
        duration=duration,
        time_step=time_step,
        translation_noise=0.003,
        rotation_noise=np.deg2rad(0.20),
        seed=seed + 1000,
    )
    strong_problem = _problem_from_synthetic(synthetic)
    oracle = oracle_effective_residual_wrench(synthetic)
    process = GaussMarkovWrenchProcess(
        times=synthetic.observations.times[:-1],
        stationary_standard_deviation=np.maximum(
            2.0 * np.sqrt(np.mean(oracle**2, axis=0)),
            np.asarray((0.05, 0.05, 0.05, 0.005, 0.005, 0.005)),
        ),
        correlation_time=0.35,
    )
    weak_problem = WeakConstraintProblem(strong_problem, process)
    selected_strong_sizes = _validated_sizes(
        strong_sizes, CONTROL_DIMENSION, "strong size sweep"
    )
    selected_weak_sizes = _validated_sizes(
        (
            (weak_problem.control_dimension + 2,
             weak_problem.control_dimension + 10)
            if weak_sizes is None
            else weak_sizes
        ),
        weak_problem.control_dimension,
        "weak size sweep",
    )
    if len(selected_strong_sizes) != len(selected_weak_sizes):
        raise ValueError(
            "strong and weak sweeps need the same number of resolution tiers"
        )

    prior = StrongConstraintPrior.grape()
    static_truth = _static_truth_without_model_error(
        synthetic.truth_parameters
    )
    truth_coordinates = strong_problem.parameter_chart.encode(static_truth)
    strong_laws = []
    for size in selected_strong_sizes:
        posterior = StrongConstraintIEnKS(
            IEnKSConfig(
                ensemble_size=size,
                maximum_iterations=maximum_iterations,
                seed=seed,
            )
        ).fit(strong_problem, prior)
        strong_laws.append(
            _law_snapshot(
                "strong",
                posterior,
                CONTROL_DIMENSION,
                prior,
                synthetic,
                truth_coordinates,
            )
        )

    weak_laws = []
    for size in selected_weak_sizes:
        posterior = WeakConstraintIEnKSQ(
            IEnKSConfig(
                ensemble_size=size,
                maximum_iterations=maximum_iterations,
                seed=seed,
            )
        ).fit(
            weak_problem,
            WeakConstraintPrior(prior, process),
        )
        weak_laws.append(
            _law_snapshot(
                "weak",
                posterior,
                weak_problem.control_dimension,
                prior,
                synthetic,
                truth_coordinates,
            )
        )

    strong_laws = tuple(strong_laws)
    weak_laws = tuple(weak_laws)
    strong_comparison = _compare_endpoint_laws(
        strong_laws[0], strong_laws[-1], projection_count
    )
    weak_comparison = _compare_endpoint_laws(
        weak_laws[0], weak_laws[-1], projection_count
    )
    method_conclusions = tuple(
        WeakStrongConclusion(
            strong_size=strong.ensemble_size,
            weak_size=weak.ensemble_size,
            weak_has_lower_identifiable_bias=(
                weak.identifiable_mean_error
                < strong.identifiable_mean_error
            ),
            weak_has_lower_path_mean_error=(
                weak.path_mean_error < strong.path_mean_error
            ),
            weak_has_higher_path_coverage=(
                weak.truth_path_component_coverage
                > strong.truth_path_component_coverage
            ),
        )
        for strong, weak in zip(strong_laws, weak_laws)
    )
    strong_signatures = tuple(
        (
            law.ridge_preserved,
            law.pose_mean_improved_over_nominal,
            law.truth_path_covered,
        )
        for law in strong_laws
    )
    weak_signatures = tuple(
        (
            law.ridge_preserved,
            law.pose_mean_improved_over_nominal,
            law.truth_path_covered,
        )
        for law in weak_laws
    )
    strong_stable = bool(
        all(value == strong_signatures[0] for value in strong_signatures[1:])
    )
    weak_stable = bool(
        all(value == weak_signatures[0] for value in weak_signatures[1:])
    )
    signatures = tuple(value.signature for value in method_conclusions)
    return EnsembleConvergenceReport(
        synthetic=synthetic,
        weak_control_dimension=weak_problem.control_dimension,
        truth_static_coordinates=truth_coordinates,
        strong_laws=strong_laws,
        weak_laws=weak_laws,
        strong_endpoint_comparison=strong_comparison,
        weak_endpoint_comparison=weak_comparison,
        weak_strong_conclusions=method_conclusions,
        strong_size_conclusions_stable=strong_stable,
        weak_size_conclusions_stable=weak_stable,
        weak_strong_conclusion_stable=bool(
            all(value == signatures[0] for value in signatures[1:])
        ),
    )


def save_ensemble_convergence(
    path: str, report: EnsembleConvergenceReport
) -> Path:
    """Persist every raw resolution law and its comparison diagnostics."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    synthetic = report.synthetic
    payload = {
        "schema": np.asarray(
            ("grape-weak-constraint/phase4-ensemble-convergence",)
        ),
        "times": synthetic.truth.times,
        "observations_position": synthetic.observations.position,
        "observations_orientation_xyzw": (
            synthetic.observations.orientation_xyzw
        ),
        "observation_translation_covariance": (
            synthetic.observations.translation_covariance
        ),
        "observation_rotation_covariance": (
            synthetic.observations.rotation_covariance
        ),
        "truth_static_coordinates": report.truth_static_coordinates,
        "truth_correction_translation": synthetic.correction_translation,
        "truth_correction_rotation_vector": (
            synthetic.correction_rotation_vector
        ),
        "weak_control_dimension": np.asarray(
            (report.weak_control_dimension,), dtype=np.int64
        ),
        "strong_size_conclusions_stable": np.asarray(
            (report.strong_size_conclusions_stable,), dtype=bool
        ),
        "weak_size_conclusions_stable": np.asarray(
            (report.weak_size_conclusions_stable,), dtype=bool
        ),
        "weak_strong_conclusion_stable": np.asarray(
            (report.weak_strong_conclusion_stable,), dtype=bool
        ),
    }
    law_array_fields = (
        "parameter_coordinates",
        "identifiable_quotient",
        "whitened_identifiable_quotient",
        "correction_translation",
        "correction_rotation_vector",
        "whitened_correction_path",
    )
    law_scalar_fields = (
        "ensemble_size",
        "control_dimension",
        "ridge_variance_ratio",
        "identifiable_mean_error",
        "path_mean_error",
        "nominal_path_error",
        "truth_path_component_coverage",
        "iteration_count",
        "initial_objective",
        "final_accepted_objective",
        "assimilation_reduced_objective",
        "ridge_preserved",
        "pose_mean_improved_over_nominal",
        "truth_path_covered",
    )
    for method, laws in (
        ("strong", report.strong_laws),
        ("weak", report.weak_laws),
    ):
        for index, law in enumerate(laws):
            prefix = "{}_{}_".format(method, index)
            for name in law_array_fields:
                payload[prefix + name] = np.asarray(getattr(law, name))
            for name in law_scalar_fields:
                payload[prefix + name] = np.asarray((getattr(law, name),))
    comparison_scalar_fields = (
        "smaller_size",
        "larger_size",
        "identifiable_sliced_w1",
        "path_sliced_w1",
        "identifiable_mean_shift",
        "path_mean_shift",
        "ridge_variance_ratio_change",
        "path_coverage_change",
        "ridge_conclusion_stable",
        "pose_mean_conclusion_stable",
        "coverage_conclusion_stable",
    )
    for method, comparison in (
        ("strong", report.strong_endpoint_comparison),
        ("weak", report.weak_endpoint_comparison),
    ):
        for name in comparison_scalar_fields:
            payload["{}_comparison_{}".format(method, name)] = np.asarray(
                (getattr(comparison, name),)
            )
    payload["weak_strong_signature"] = np.asarray(
        [value.signature for value in report.weak_strong_conclusions],
        dtype=bool,
    )
    np.savez_compressed(str(destination), **payload)
    return destination
