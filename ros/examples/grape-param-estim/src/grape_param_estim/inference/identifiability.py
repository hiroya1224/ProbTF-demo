"""Local excitation, gauge, and prior-dominance diagnostics."""

from dataclasses import dataclass, field
from typing import Any, Sequence, Tuple

import numpy as np


def _readonly_array(values: Any, shape: Tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0 and int(np.prod(shape)) == 0:
        array = np.empty(shape, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError("{} must have shape {} and be finite".format(label, shape))
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _canonicalize_directions(values: np.ndarray) -> np.ndarray:
    """Fix the arbitrary SVD sign so serialized coefficients are repeatable."""

    result = np.array(values, dtype=float, copy=True)
    for row in result:
        if not np.any(row):
            continue
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return result


def _svd_diagnostics(
    matrix: np.ndarray,
    parameter_count: int,
    structural_gauge_dimension: int,
    relative_tolerance: float,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, float]:
    # A tall trajectory Jacobian is the normal case.  ``full_matrices=True``
    # would construct a potentially enormous row-space basis even though only
    # the small parameter-space basis is needed.  It is required only for the
    # uncommon underdetermined case in order to retain every null direction.
    full_matrices = matrix.shape[0] < parameter_count
    _, singular, right = np.linalg.svd(
        matrix, full_matrices=full_matrices
    )
    right = _canonicalize_directions(right)
    threshold = (
        relative_tolerance * max(float(singular[0]), 1.0)
        if singular.size
        else relative_tolerance
    )
    numerical_rank = int(np.count_nonzero(singular > threshold))

    # A finite-difference Jacobian can spuriously lift a known structural
    # gauge (for example command scale versus plant authority).  The declared
    # model structure is authoritative: reserve those null dimensions even
    # when numerical noise makes the sampled matrix appear full rank.
    rank = min(
        numerical_rank,
        parameter_count - structural_gauge_dimension,
    )
    directions = right[:rank, :]
    null = right[rank:, :]
    condition = (
        float(singular[0] / singular[rank - 1])
        if rank > 0 and singular[rank - 1] > 0.0
        else float("inf")
    )
    return singular, rank, directions, null, condition


@dataclass(frozen=True)
class EpisodeExcitationReport:
    """Identifiable parameter directions excited by one failure maneuver.

    Rows in ``direction_coefficients`` are ordered like ``parameter_names``.
    ``nuisance_sample_ids`` and ``nuisance_sample_weights`` document every
    initial-state/nuisance realization included in the weighted Jacobian.
    """

    episode_id: str
    parameter_names: Tuple[str, ...]
    nuisance_sample_ids: Tuple[str, ...]
    nuisance_sample_weights: np.ndarray
    nuisance_sample_count: int
    jacobian_row_count: int
    jacobian_rank: int
    singular_values: np.ndarray
    direction_labels: Tuple[str, ...]
    direction_coefficients: np.ndarray
    null_directions: np.ndarray
    condition_number: float
    structural_gauge_dimension: int
    excitation_nullity: int

    def __post_init__(self) -> None:
        episode_id = str(self.episode_id)
        names = tuple(str(item) for item in self.parameter_names)
        sample_ids = tuple(str(item) for item in self.nuisance_sample_ids)
        labels = tuple(str(item) for item in self.direction_labels)
        sample_count = int(self.nuisance_sample_count)
        row_count = int(self.jacobian_row_count)
        rank = int(self.jacobian_rank)
        gauge = int(self.structural_gauge_dimension)
        nullity = int(self.excitation_nullity)
        if not episode_id:
            raise ValueError("episode_id is required")
        if not names or any(not item for item in names):
            raise ValueError("parameter_names must be non-empty")
        if (
            not sample_ids
            or any(not item for item in sample_ids)
            or sample_count != len(sample_ids)
        ):
            raise ValueError(
                "nuisance sample IDs/count must identify every sample"
            )
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("nuisance sample IDs must be unique per episode")
        if (
            row_count <= 0
            or rank < 0
            or rank > len(names) - gauge
            or gauge < 0
            or gauge > len(names)
            or nullity != max(0, len(names) - rank - gauge)
        ):
            raise ValueError("episode excitation dimensions are inconsistent")
        if len(labels) != rank or any(not item for item in labels):
            raise ValueError("direction labels must match the Jacobian rank")
        weights = _readonly_array(
            self.nuisance_sample_weights,
            (sample_count,),
            "nuisance_sample_weights",
        )
        if np.any(weights <= 0.0) or not np.isclose(
            float(np.sum(weights)), 1.0
        ):
            raise ValueError(
                "nuisance sample weights must be positive and normalized"
            )
        singular = np.asarray(self.singular_values, dtype=float).reshape(-1)
        if not np.all(np.isfinite(singular)):
            raise ValueError("singular_values must be finite")
        singular_copy = np.array(singular, copy=True)
        singular_copy.setflags(write=False)
        directions = _readonly_array(
            self.direction_coefficients,
            (rank, len(names)),
            "direction_coefficients",
        )
        null = _readonly_array(
            self.null_directions,
            (len(names) - rank, len(names)),
            "null_directions",
        )
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "nuisance_sample_ids", sample_ids)
        object.__setattr__(self, "nuisance_sample_weights", weights)
        object.__setattr__(self, "nuisance_sample_count", sample_count)
        object.__setattr__(self, "jacobian_row_count", row_count)
        object.__setattr__(self, "jacobian_rank", rank)
        object.__setattr__(self, "singular_values", singular_copy)
        object.__setattr__(self, "direction_labels", labels)
        object.__setattr__(self, "direction_coefficients", directions)
        object.__setattr__(self, "null_directions", null)
        object.__setattr__(self, "structural_gauge_dimension", gauge)
        object.__setattr__(self, "excitation_nullity", nullity)


@dataclass(frozen=True)
class IdentifiabilityReport:
    model_id: str
    parameter_names: Tuple[str, ...]
    jacobian_rank: int
    singular_values: np.ndarray
    null_directions: np.ndarray
    condition_number: float
    structural_gauge_dimension: int
    excitation_nullity: int
    prior_or_bound_dominated: Tuple[str, ...]
    identifiable_combinations: Tuple[str, ...]
    direction_coefficients: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0))
    )
    episode_excitation: Tuple[EpisodeExcitationReport, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(str(item) for item in self.parameter_names)
        rank = int(self.jacobian_rank)
        singular = np.asarray(self.singular_values, dtype=float).reshape(-1)
        null = np.asarray(self.null_directions, dtype=float)
        if (
            not np.all(np.isfinite(singular))
            or null.ndim != 2
            or not np.all(np.isfinite(null))
        ):
            raise ValueError("identifiability arrays must be finite")
        directions = np.asarray(self.direction_coefficients, dtype=float)
        if directions.size == 0:
            directions = np.empty((0, len(names)))
        if (
            directions.ndim != 2
            or directions.shape[1] != len(names)
            or directions.shape[0] not in (0, rank)
            or not np.all(np.isfinite(directions))
        ):
            raise ValueError(
                "direction coefficients must align with parameter names and rank"
            )
        episodes = tuple(self.episode_excitation)
        episode_ids = tuple(item.episode_id for item in episodes)
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("episode excitation entries must have unique IDs")
        if any(item.parameter_names != names for item in episodes):
            raise ValueError(
                "episode excitation parameter order must match global report"
            )
        singular_copy = np.array(singular, copy=True)
        null_copy = np.array(null, copy=True)
        direction_copy = np.array(directions, copy=True)
        singular_copy.setflags(write=False)
        null_copy.setflags(write=False)
        direction_copy.setflags(write=False)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "singular_values", singular_copy)
        object.__setattr__(self, "null_directions", null_copy)
        object.__setattr__(self, "direction_coefficients", direction_copy)
        object.__setattr__(self, "episode_excitation", episodes)


def episode_excitation_report(
    jacobian: Any,
    parameter_names: Sequence[str],
    episode_id: str,
    nuisance_sample_ids: Sequence[str],
    nuisance_sample_weights: Any,
    structural_gauge_dimension: int = 0,
    relative_tolerance: float = 1.0e-8,
) -> EpisodeExcitationReport:
    """Summarize one episode's weighted, nuisance-marginalized Jacobian."""

    matrix = np.asarray(jacobian, dtype=float)
    names = tuple(str(item) for item in parameter_names)
    if (
        matrix.ndim != 2
        or matrix.shape[0] <= 0
        or matrix.shape[1] != len(names)
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("jacobian and parameter names are incompatible")
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    gauge = int(structural_gauge_dimension)
    if gauge < 0 or gauge > len(names):
        raise ValueError(
            "structural gauge dimension must lie within parameter space"
        )
    sample_ids = tuple(str(item) for item in nuisance_sample_ids)
    weights = np.asarray(nuisance_sample_weights, dtype=float).reshape(-1)
    if (
        weights.shape != (len(sample_ids),)
        or weights.size == 0
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError(
            "nuisance sample IDs and positive finite weights are required"
        )
    weights = weights / float(np.sum(weights))
    singular, rank, directions, null, condition = _svd_diagnostics(
        matrix,
        len(names),
        gauge,
        tolerance,
    )
    labels = tuple(
        "sv_direction_{}".format(index) for index in range(rank)
    )
    return EpisodeExcitationReport(
        episode_id=str(episode_id),
        parameter_names=names,
        nuisance_sample_ids=sample_ids,
        nuisance_sample_weights=weights,
        nuisance_sample_count=len(sample_ids),
        jacobian_row_count=matrix.shape[0],
        jacobian_rank=rank,
        singular_values=singular,
        direction_labels=labels,
        direction_coefficients=directions,
        null_directions=null,
        condition_number=condition,
        structural_gauge_dimension=gauge,
        excitation_nullity=max(0, len(names) - rank - gauge),
    )


def local_identifiability(
    jacobian: Any,
    parameter_names: Sequence[str],
    model_id: str,
    structural_gauge_dimension: int = 0,
    posterior_particles: Any = None,
    prior_lower: Any = None,
    prior_upper: Any = None,
    relative_tolerance: float = 1.0e-8,
    episode_excitation: Sequence[EpisodeExcitationReport] = (),
) -> IdentifiabilityReport:
    matrix = np.asarray(jacobian, dtype=float)
    names = tuple(str(item) for item in parameter_names)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(names)
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("jacobian and parameter names are incompatible")
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    gauge = int(structural_gauge_dimension)
    if gauge < 0 or gauge > len(names):
        raise ValueError(
            "structural gauge dimension must lie within parameter space"
        )
    singular, rank, directions, null, condition = _svd_diagnostics(
        matrix,
        len(names),
        gauge,
        tolerance,
    )
    dominated = []
    if posterior_particles is not None:
        particles = np.asarray(posterior_particles, dtype=float)
        lower = np.asarray(prior_lower, dtype=float).reshape(-1)
        upper = np.asarray(prior_upper, dtype=float).reshape(-1)
        if (
            particles.ndim != 2
            or particles.shape[1] != len(names)
            or lower.shape != (len(names),)
            or upper.shape != lower.shape
        ):
            raise ValueError("posterior/prior arrays are incompatible")
        width_ratio = np.ptp(particles, axis=0) / (upper - lower)
        boundary = np.mean(
            (particles <= lower + 0.02 * (upper - lower))
            | (particles >= upper - 0.02 * (upper - lower)),
            axis=0,
        )
        dominated = [
            names[index]
            for index in range(len(names))
            if width_ratio[index] > 0.8 or boundary[index] > 0.5
        ]
    combinations = tuple(
        "sv_direction_{}".format(index) for index in range(rank)
    )
    return IdentifiabilityReport(
        model_id=str(model_id),
        parameter_names=names,
        jacobian_rank=rank,
        singular_values=singular,
        null_directions=null,
        condition_number=condition,
        structural_gauge_dimension=gauge,
        excitation_nullity=max(0, len(names) - rank - gauge),
        prior_or_bound_dominated=tuple(dominated),
        identifiable_combinations=combinations,
        direction_coefficients=directions,
        episode_excitation=tuple(episode_excitation),
    )


__all__ = [
    "EpisodeExcitationReport",
    "IdentifiabilityReport",
    "episode_excitation_report",
    "local_identifiability",
]
