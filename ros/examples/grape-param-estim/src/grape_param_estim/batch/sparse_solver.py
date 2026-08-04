"""Scaled-coordinate LM steps for arrowhead sparse batch systems."""

from dataclasses import dataclass
from numbers import Real
from typing import Sequence, Tuple

import numpy as np
from scipy.sparse import csc_matrix, diags, eye, isspmatrix_csc
from scipy.sparse.linalg import splu

from grape_param_estim.batch.linearize import SparseBatchLinearization


@dataclass(frozen=True)
class BagFactorizationDiagnostics:
    """Sparse factorization diagnostics for one eliminated bag block."""

    bag_id: str
    local_slice: slice
    local_dimension: int
    undamped_hessian_nnz: int
    damped_hessian_nnz: int
    factor_l_nnz: int
    factor_u_nnz: int
    rhs_count: int


@dataclass(frozen=True)
class ScaledSchurStep:
    """One physical LM step and its scaled Schur diagnostics.

    The reduced system includes LM damping.  ``gradient_inf_norm`` and
    ``scaled_step_norm`` are measured in scaled coordinates, while
    ``predicted_reduction`` is evaluated with the undamped physical GN model.
    """

    delta: np.ndarray
    scaled_delta: np.ndarray
    reduced_hessian: np.ndarray
    reduced_rhs: np.ndarray
    gradient_inf_norm: float
    scaled_step_norm: float
    predicted_reduction: float
    bag_diagnostics: Tuple[BagFactorizationDiagnostics, ...]


def solve_scaled_lm_step(
    linearization: SparseBatchLinearization,
    coordinate_scale: Sequence[float],
    damping: float,
) -> ScaledSchurStep:
    """Solve an arrowhead LM system by eliminating each bag-local block.

    ``coordinate_scale`` contains positive physical units per scaled
    coordinate.  For ``D = diag(coordinate_scale)``, the solved system is
    ``(D H D + damping I) delta_scaled = -D g`` and the returned ``delta`` is
    ``D delta_scaled``.  The predicted reduction uses the undamped GN model.
    """

    if not isinstance(linearization, SparseBatchLinearization):
        raise TypeError("linearization must be a SparseBatchLinearization")
    if isinstance(damping, (bool, np.bool_)) or not isinstance(damping, Real):
        raise TypeError("damping must be a finite non-negative scalar")
    damping_value = float(damping)
    if not np.isfinite(damping_value) or damping_value < 0.0:
        raise ValueError("damping must be a finite non-negative scalar")

    layout = linearization.layout
    dimension = layout.total_dimension
    scale = np.asarray(coordinate_scale, dtype=float)
    if (
        scale.shape != (dimension,)
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise ValueError(
            "coordinate_scale must contain one finite positive value per "
            "physical coordinate"
        )

    hessian = linearization.hessian
    gradient = np.asarray(linearization.gradient, dtype=float)
    if (
        not isspmatrix_csc(hessian)
        or hessian.shape != (dimension, dimension)
        or gradient.shape != (dimension,)
        or not np.all(np.isfinite(hessian.data))
        or not np.all(np.isfinite(gradient))
    ):
        raise ValueError(
            "linearization must contain a finite CSC Hessian and gradient"
        )
    asymmetry = (hessian - hessian.T).tocsc()
    asymmetry.eliminate_zeros()
    if asymmetry.nnz:
        largest_asymmetry = float(np.max(np.abs(asymmetry.data)))
        largest_entry = (
            float(np.max(np.abs(hessian.data))) if hessian.nnz else 0.0
        )
        if largest_asymmetry > 1.0e-12 * max(1.0, largest_entry):
            raise ValueError("linearization Hessian must be symmetric")
    _reject_cross_bag_hessian_entries(hessian, layout)

    scaling = diags(scale, offsets=0, shape=hessian.shape, format="csc")
    scaled_hessian = (scaling @ hessian @ scaling).tocsc()
    scaled_hessian.sum_duplicates()
    scaled_hessian.eliminate_zeros()
    scaled_hessian.sort_indices()
    scaled_gradient = scale * gradient
    damped_hessian = (
        scaled_hessian
        + damping_value * eye(dimension, format="csc", dtype=float)
    ).tocsc()
    damped_hessian.sum_duplicates()
    damped_hessian.sort_indices()

    shared_slice = layout.shared_slice
    shared_dimension = shared_slice.stop - shared_slice.start
    if shared_slice != slice(0, 18) or shared_dimension != 18:
        raise ValueError("layout must begin with the shared 18-D block")
    reduced_hessian = damped_hessian[
        shared_slice, shared_slice
    ].toarray()
    reduced_rhs = -scaled_gradient[shared_slice].copy()
    local_solutions = {}
    diagnostics = []

    for bag_id in layout.bag_ids:
        local_slice = layout.bag_slice(bag_id)
        local_hessian_undamped = scaled_hessian[
            local_slice, local_slice
        ].tocsc()
        local_hessian = damped_hessian[
            local_slice, local_slice
        ].tocsc()
        local_to_shared = damped_hessian[
            local_slice, shared_slice
        ].tocsc()
        try:
            factorization = splu(local_hessian)
        except RuntimeError as error:
            raise np.linalg.LinAlgError(
                "bag-local LM block is singular for {!r}".format(bag_id)
            ) from error

        rhs = _multiple_rhs(
            scaled_gradient[local_slice],
            local_to_shared,
        )
        solved = np.asarray(factorization.solve(rhs), dtype=float)
        if solved.shape != rhs.shape or not np.all(np.isfinite(solved)):
            raise np.linalg.LinAlgError(
                "bag-local LM solve is non-finite for {!r}".format(bag_id)
            )
        solved_gradient = solved[:, 0]
        solved_cross = solved[:, 1:]
        shared_to_local = local_to_shared.T
        reduced_hessian -= np.asarray(
            shared_to_local.dot(solved_cross), dtype=float
        )
        reduced_rhs += np.asarray(
            shared_to_local.dot(solved_gradient), dtype=float
        ).reshape(-1)
        local_solutions[bag_id] = (solved_gradient, solved_cross)
        diagnostics.append(
            BagFactorizationDiagnostics(
                bag_id=bag_id,
                local_slice=local_slice,
                local_dimension=local_slice.stop - local_slice.start,
                undamped_hessian_nnz=local_hessian_undamped.nnz,
                damped_hessian_nnz=local_hessian.nnz,
                factor_l_nnz=factorization.L.nnz,
                factor_u_nnz=factorization.U.nnz,
                rhs_count=rhs.shape[1],
            )
        )

    reduced_hessian = 0.5 * (
        reduced_hessian + reduced_hessian.T
    )
    if (
        reduced_hessian.shape != (18, 18)
        or reduced_rhs.shape != (18,)
        or not np.all(np.isfinite(reduced_hessian))
        or not np.all(np.isfinite(reduced_rhs))
    ):
        raise np.linalg.LinAlgError("reduced LM system is not finite")
    try:
        shared_delta = np.linalg.solve(reduced_hessian, reduced_rhs)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError("reduced LM system is singular") from error
    if not np.all(np.isfinite(shared_delta)):
        raise np.linalg.LinAlgError("reduced LM solve is non-finite")

    scaled_delta = np.empty(dimension, dtype=float)
    scaled_delta[shared_slice] = shared_delta
    for bag_id in layout.bag_ids:
        solved_gradient, solved_cross = local_solutions[bag_id]
        scaled_delta[layout.bag_slice(bag_id)] = (
            -solved_gradient - solved_cross @ shared_delta
        )
    delta = scale * scaled_delta

    predicted_reduction = -float(
        gradient @ delta + 0.5 * delta @ hessian.dot(delta)
    )
    gradient_inf_norm = float(
        np.linalg.norm(scaled_gradient, ord=np.inf)
    )
    scaled_step_norm = float(np.linalg.norm(scaled_delta))
    for name, value in (
        ("delta", delta),
        ("scaled_delta", scaled_delta),
        ("reduced_hessian", reduced_hessian),
        ("reduced_rhs", reduced_rhs),
    ):
        if not np.all(np.isfinite(value)):
            raise np.linalg.LinAlgError("{} is non-finite".format(name))
    if not all(
        np.isfinite(value)
        for value in (
            predicted_reduction,
            gradient_inf_norm,
            scaled_step_norm,
        )
    ):
        raise np.linalg.LinAlgError("LM step diagnostics are non-finite")

    for value in (
        delta,
        scaled_delta,
        reduced_hessian,
        reduced_rhs,
    ):
        value.setflags(write=False)
    return ScaledSchurStep(
        delta=delta,
        scaled_delta=scaled_delta,
        reduced_hessian=reduced_hessian,
        reduced_rhs=reduced_rhs,
        gradient_inf_norm=gradient_inf_norm,
        scaled_step_norm=scaled_step_norm,
        predicted_reduction=predicted_reduction,
        bag_diagnostics=tuple(diagnostics),
    )


def _multiple_rhs(
    local_gradient: np.ndarray,
    local_to_shared: csc_matrix,
) -> np.ndarray:
    """Pack one local gradient and 18 sparse cross columns as dense RHS."""

    rhs = np.zeros(
        (local_gradient.size, 1 + local_to_shared.shape[1]),
        dtype=float,
    )
    rhs[:, 0] = local_gradient
    for column in range(local_to_shared.shape[1]):
        start = local_to_shared.indptr[column]
        stop = local_to_shared.indptr[column + 1]
        rhs[local_to_shared.indices[start:stop], 1 + column] = (
            local_to_shared.data[start:stop]
        )
    return rhs


def _reject_cross_bag_hessian_entries(hessian, layout) -> None:
    for first_index, first_bag in enumerate(layout.bag_ids):
        first_slice = layout.bag_slice(first_bag)
        for second_bag in layout.bag_ids[first_index + 1 :]:
            second_slice = layout.bag_slice(second_bag)
            cross_block = hessian[first_slice, second_slice].tocsc()
            cross_block.eliminate_zeros()
            if cross_block.nnz:
                raise ValueError(
                    "Hessian directly couples local variables from multiple "
                    "bags"
                )


__all__ = [
    "BagFactorizationDiagnostics",
    "ScaledSchurStep",
    "solve_scaled_lm_step",
]
