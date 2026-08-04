"""Deterministic diagnostics for reduced static-parameter information.

This module only analyzes an already-formed 18-dimensional Schur complement.
It does not choose a model-discrepancy convention, add regularization, or
approximate derivatives.  In particular, a small likelihood eigenvalue is
reported as a ridge instead of being silently lifted to make an inverse exist.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Sequence, Tuple

import numpy as np

from grape_param_estim.parameterization import PARAMETER_DIMENSION


STATIC_PARAMETER_NAMES = (
    "log_mass",
    "relative_log_inertia_xx",
    "relative_log_inertia_yy",
    "relative_log_inertia_zz",
    "relative_log_inertia_xy",
    "relative_log_inertia_xz",
    "relative_log_inertia_yz",
    "cog_offset_x",
    "cog_offset_y",
    "cog_offset_z",
    "log_force_effectiveness_0",
    "log_force_effectiveness_1",
    "log_force_effectiveness_2",
    "log_force_effectiveness_3",
    "log_torque_effectiveness_0",
    "log_torque_effectiveness_1",
    "log_torque_effectiveness_2",
    "log_torque_effectiveness_3",
)


@dataclass(frozen=True)
class NamedParameterLoading:
    """One named coordinate loading in a numerical ridge direction."""

    coordinate_index: int
    parameter_name: str
    coefficient: float


@dataclass(frozen=True)
class RidgeDirection:
    """One eigenvector whose relative information is below the rank cutoff."""

    eigen_index: int
    eigenvalue: float
    relative_eigenvalue: float
    vector: np.ndarray
    loadings: Tuple[NamedParameterLoading, ...]


@dataclass(frozen=True)
class ReducedHessianAnalysis:
    """Immutable eigensystem and rank diagnostics for one reduced Hessian.

    Eigenvalues and eigenvector columns are ordered from least to most
    informative.  ``condition_number`` is infinite whenever the effective
    rank is deficient.  ``identified_condition_number`` only uses directions
    above ``relative_rank_tolerance`` and remains useful in that case.
    """

    parameter_names: Tuple[str, ...]
    hessian: np.ndarray
    eigenvalues: np.ndarray
    relative_eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    relative_rank_tolerance: float
    effective_rank: int
    condition_number: float
    identified_condition_number: float
    maximum_input_asymmetry: float
    ridge_directions: Tuple[RidgeDirection, ...]


@dataclass(frozen=True)
class ReducedInformationAnalysis:
    """Separate likelihood and posterior reduced-information diagnostics."""

    parameter_names: Tuple[str, ...]
    likelihood: ReducedHessianAnalysis
    posterior: ReducedHessianAnalysis
    prior_contribution: np.ndarray


def analyze_reduced_hessian(
    reduced_hessian: np.ndarray,
    parameter_names: Sequence[str] = STATIC_PARAMETER_NAMES,
    relative_rank_tolerance: float = 1.0e-10,
) -> ReducedHessianAnalysis:
    """Analyze one finite, positive-semidefinite 18-D information matrix.

    Roundoff-sized asymmetry is removed explicitly.  Material asymmetry and
    material negative curvature are rejected because both indicate that the
    supplied matrix is not a Gauss--Newton/Laplace information matrix.
    Eigenvector signs and numerically degenerate eigenspaces are canonicalized
    against chart-coordinate order so serialized diagnostics are repeatable.
    """

    names = _validated_parameter_names(parameter_names)
    tolerance = _validated_relative_tolerance(relative_rank_tolerance)
    value = np.asarray(reduced_hessian, dtype=float)
    if value.shape != (PARAMETER_DIMENSION, PARAMETER_DIMENSION):
        raise ValueError(
            "reduced_hessian must have shape ({0}, {0})".format(
                PARAMETER_DIMENSION
            )
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("reduced_hessian must be finite")
    value = value.copy()
    maximum_entry = float(np.max(np.abs(value))) if value.size else 0.0
    maximum_asymmetry = float(np.max(np.abs(value - value.T)))
    if maximum_asymmetry > 1.0e-10 * max(1.0, maximum_entry):
        raise ValueError("reduced_hessian must be symmetric up to roundoff")
    symmetric = 0.5 * (value + value.T)

    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    spectral_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    negative_tolerance = (
        256.0
        * np.finfo(float).eps
        * PARAMETER_DIMENSION
        * spectral_scale
    )
    if float(eigenvalues[0]) < -negative_tolerance:
        raise ValueError(
            "reduced_hessian has material negative information curvature"
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    eigenvectors = _canonical_eigenvectors(eigenvalues, eigenvectors)

    largest = float(eigenvalues[-1])
    if largest == 0.0:
        relative = np.zeros(PARAMETER_DIMENSION, dtype=float)
    else:
        relative = eigenvalues / largest
    identified = relative > tolerance
    effective_rank = int(np.count_nonzero(identified))
    if effective_rank == PARAMETER_DIMENSION:
        condition_number = float(largest / eigenvalues[0])
    else:
        condition_number = float("inf")
    if effective_rank:
        identified_condition_number = float(
            largest / eigenvalues[np.flatnonzero(identified)[0]]
        )
    else:
        identified_condition_number = float("inf")

    ridge_directions = tuple(
        _ridge_direction(
            eigen_index=index,
            eigenvalue=float(eigenvalues[index]),
            relative_eigenvalue=float(relative[index]),
            vector=eigenvectors[:, index],
            parameter_names=names,
        )
        for index in np.flatnonzero(~identified)
    )

    for array in (symmetric, eigenvalues, relative, eigenvectors):
        array.setflags(write=False)
    return ReducedHessianAnalysis(
        parameter_names=names,
        hessian=symmetric,
        eigenvalues=eigenvalues,
        relative_eigenvalues=relative,
        eigenvectors=eigenvectors,
        relative_rank_tolerance=tolerance,
        effective_rank=effective_rank,
        condition_number=condition_number,
        identified_condition_number=identified_condition_number,
        maximum_input_asymmetry=maximum_asymmetry,
        ridge_directions=ridge_directions,
    )


def analyze_reduced_information(
    likelihood_reduced_hessian: np.ndarray,
    posterior_reduced_hessian: np.ndarray,
    parameter_names: Sequence[str] = STATIC_PARAMETER_NAMES,
    relative_rank_tolerance: float = 1.0e-10,
) -> ReducedInformationAnalysis:
    """Analyze likelihood and posterior information without conflating them."""

    names = _validated_parameter_names(parameter_names)
    likelihood = analyze_reduced_hessian(
        likelihood_reduced_hessian,
        parameter_names=names,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    posterior = analyze_reduced_hessian(
        posterior_reduced_hessian,
        parameter_names=names,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    prior_contribution = posterior.hessian - likelihood.hessian
    prior_contribution = 0.5 * (
        prior_contribution + prior_contribution.T
    )
    prior_contribution.setflags(write=False)
    return ReducedInformationAnalysis(
        parameter_names=names,
        likelihood=likelihood,
        posterior=posterior,
        prior_contribution=prior_contribution,
    )


def _validated_parameter_names(
    parameter_names: Sequence[str],
) -> Tuple[str, ...]:
    try:
        names = tuple(parameter_names)
    except TypeError as error:
        raise TypeError("parameter_names must be an iterable of strings") from error
    if len(names) != PARAMETER_DIMENSION:
        raise ValueError(
            "parameter_names must contain {} entries".format(
                PARAMETER_DIMENSION
            )
        )
    if any(
        type(name) is not str or not name or name.strip() != name
        for name in names
    ):
        raise ValueError(
            "parameter_names must contain non-empty canonical strings"
        )
    if len(set(names)) != len(names):
        raise ValueError("parameter_names must be unique")
    return names


def _validated_relative_tolerance(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("relative_rank_tolerance must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result >= 1.0:
        raise ValueError(
            "relative_rank_tolerance must be finite and in [0, 1)"
        )
    return result


def _canonical_eigenvectors(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Choose repeatable bases for signs and degenerate eigenspaces."""

    result = np.asarray(eigenvectors, dtype=float).copy()
    scale = max(1.0, float(eigenvalues[-1]))
    cluster_tolerance = (
        512.0
        * np.finfo(float).eps
        * PARAMETER_DIMENSION
        * scale
    )
    start = 0
    while start < PARAMETER_DIMENSION:
        stop = start + 1
        while (
            stop < PARAMETER_DIMENSION
            and eigenvalues[stop] - eigenvalues[start] <= cluster_tolerance
        ):
            stop += 1
        if stop - start > 1:
            result[:, start:stop] = _canonical_subspace_basis(
                result[:, start:stop]
            )
        start = stop

    for column in range(PARAMETER_DIMENSION):
        vector = result[:, column]
        anchor = int(np.argmax(np.abs(vector)))
        if vector[anchor] < 0.0:
            result[:, column] = -vector
    return result


def _canonical_subspace_basis(basis: np.ndarray) -> np.ndarray:
    """Build a coordinate-ordered basis from an eigenspace projector."""

    projector = basis @ basis.T
    dimension = basis.shape[1]
    selected = []
    threshold = 2048.0 * np.finfo(float).eps * PARAMETER_DIMENSION
    for coordinate in range(PARAMETER_DIMENSION):
        candidate = projector[:, coordinate].copy()
        for _ in range(2):
            for existing in selected:
                candidate -= existing * float(existing @ candidate)
        norm = float(np.linalg.norm(candidate))
        if norm > threshold:
            selected.append(candidate / norm)
            if len(selected) == dimension:
                break
    if len(selected) != dimension:
        raise np.linalg.LinAlgError(
            "cannot construct a deterministic degenerate-eigenspace basis"
        )
    return np.column_stack(selected)


def _ridge_direction(
    eigen_index: int,
    eigenvalue: float,
    relative_eigenvalue: float,
    vector: np.ndarray,
    parameter_names: Tuple[str, ...],
) -> RidgeDirection:
    direction = np.asarray(vector, dtype=float).copy()
    direction.setflags(write=False)
    ordering = sorted(
        range(PARAMETER_DIMENSION),
        key=lambda index: (-abs(float(direction[index])), index),
    )
    loadings = tuple(
        NamedParameterLoading(
            coordinate_index=index,
            parameter_name=parameter_names[index],
            coefficient=float(direction[index]),
        )
        for index in ordering
    )
    return RidgeDirection(
        eigen_index=int(eigen_index),
        eigenvalue=eigenvalue,
        relative_eigenvalue=relative_eigenvalue,
        vector=direction,
        loadings=loadings,
    )


__all__ = [
    "NamedParameterLoading",
    "ReducedHessianAnalysis",
    "ReducedInformationAnalysis",
    "RidgeDirection",
    "STATIC_PARAMETER_NAMES",
    "analyze_reduced_hessian",
    "analyze_reduced_information",
]
