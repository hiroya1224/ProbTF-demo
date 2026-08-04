"""Information-form IEKS steps for adjacent-knot batch factor graphs.

The nonlinear iteration lives in :mod:`grape_param_estim.batch.lm`.  This
module solves each linearized Gaussian problem by eliminating the 26-D knot
states in chronological order and then back-substituting in reverse order.
Shared plant parameters and bag-local sensor biases remain as persistent
coordinates in the small reduced system.  Command lag is deliberately absent
from the state layout; it remains an outer fixed-lag profile parameter.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Sequence, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csc_matrix, diags, isspmatrix_csc

from grape_param_estim.batch.linearize import SparseBatchLinearization
from grape_param_estim.batch.variables import (
    VariableKey,
    VariableKind,
    VariableScope,
)


_KNOT_DIMENSION = 26


@dataclass(frozen=True)
class IeksBagSmoothingDiagnostics:
    """Forward-elimination and backward-smoothing counts for one bag."""

    bag_id: str
    local_dimension: int
    bias_dimension: int
    knot_count: int
    forward_factorizations: int
    backward_steps: int


@dataclass(frozen=True)
class ScaledIeksStep:
    """One physical step produced by an information-form IEKS pass."""

    delta: np.ndarray
    scaled_delta: np.ndarray
    gradient_inf_norm: float
    scaled_step_norm: float
    predicted_reduction: float
    bag_diagnostics: Tuple[IeksBagSmoothingDiagnostics, ...]


def _coordinate_indices(selected_slice: slice) -> np.ndarray:
    return np.arange(selected_slice.start, selected_slice.stop, dtype=np.int64)


def _dense_block(
    matrix: csc_matrix,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    if rows.size == 0 or columns.size == 0:
        return np.zeros((rows.size, columns.size), dtype=float)
    return np.asarray(matrix[rows, :][:, columns].toarray(), dtype=float)


def _bag_bias_indices(layout, bag_id: str) -> np.ndarray:
    parts = tuple(
        _coordinate_indices(layout.column_slice(key))
        for key in layout.variable_keys
        if key.scope is VariableScope.BAG and key.bag_id == bag_id
    )
    return (
        np.concatenate(parts)
        if parts
        else np.empty(0, dtype=np.int64)
    )


def _bag_knot_slices(layout, bag_id: str) -> Tuple[slice, ...]:
    knot_indices = tuple(
        sorted(
            {
                key.knot_index
                for key in layout.variable_keys
                if key.scope is VariableScope.KNOT
                and key.bag_id == bag_id
            }
        )
    )
    result = []
    for knot_index in knot_indices:
        first = layout.column_slice(
            VariableKey(
                VariableKind.POSITION,
                bag_id=bag_id,
                knot_index=knot_index,
            )
        )
        last = layout.column_slice(
            VariableKey(
                VariableKind.GIMBAL_ANGLE,
                bag_id=bag_id,
                knot_index=knot_index,
            )
        )
        selected = slice(first.start, last.stop)
        if selected.stop - selected.start != _KNOT_DIMENSION:
            raise ValueError("IEKS requires a complete 26-D knot state")
        result.append(selected)
    if not result:
        raise ValueError("IEKS requires at least one knot per bag")
    return tuple(result)


def validate_ieks_topology(linearization: SparseBatchLinearization) -> None:
    """Reject factors outside the supported Markov/arrowhead topology."""

    if not isinstance(linearization, SparseBatchLinearization):
        raise TypeError("linearization must be a SparseBatchLinearization")
    layout = linearization.layout
    if layout.shared_slice != slice(0, 18):
        raise ValueError("IEKS layout must begin with the shared 18-D block")
    for bag_id in layout.bag_ids:
        _bag_knot_slices(layout, bag_id)

    for provenance in linearization.factor_provenance:
        knots = tuple(
            key.knot_index
            for key in provenance.variable_keys
            if key.scope is VariableScope.KNOT
        )
        if knots and max(knots) - min(knots) > 1:
            raise ValueError(
                "IEKS factors may couple only the same or adjacent knots"
            )

    hessian = linearization.hessian
    dimension = layout.total_dimension
    coordinate_bag = np.empty(dimension, dtype=object)
    coordinate_bag[:] = None
    coordinate_knot = np.full(dimension, -1, dtype=np.int64)
    for key in layout.variable_keys:
        selected = layout.column_slice(key)
        if key.scope is not VariableScope.SHARED:
            coordinate_bag[selected] = key.bag_id
        if key.scope is VariableScope.KNOT:
            coordinate_knot[selected] = key.knot_index
    entries = hessian.tocoo()
    for row, column, value in zip(
        entries.row, entries.col, entries.data
    ):
        if value == 0.0:
            continue
        row_bag = coordinate_bag[row]
        column_bag = coordinate_bag[column]
        if (
            row_bag is not None
            and column_bag is not None
            and row_bag != column_bag
        ):
            raise ValueError("IEKS Hessian cannot directly couple bags")
        row_knot = coordinate_knot[row]
        column_knot = coordinate_knot[column]
        if (
            row_bag is not None
            and row_bag == column_bag
            and row_knot >= 0
            and column_knot >= 0
            and abs(int(row_knot) - int(column_knot)) > 1
        ):
            raise ValueError(
                "IEKS Hessian may couple only the same or adjacent knots"
            )


def _validated_scaled_system(
    linearization: SparseBatchLinearization,
    coordinate_scale: Sequence[float],
    damping: float,
) -> Tuple[csc_matrix, np.ndarray, np.ndarray, float]:
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
    validate_ieks_topology(linearization)

    scaling = diags(scale, offsets=0, shape=hessian.shape, format="csc")
    scaled_hessian = (scaling @ hessian @ scaling).tocsc()
    scaled_hessian.sum_duplicates()
    scaled_hessian.eliminate_zeros()
    scaled_hessian.sort_indices()
    return scaled_hessian, scale * gradient, scale, damping_value


def _persistent_indices(layout, optimize_shared: bool) -> np.ndarray:
    parts = []
    if optimize_shared:
        parts.append(_coordinate_indices(layout.shared_slice))
    parts.extend(
        _coordinate_indices(layout.column_slice(key))
        for key in layout.variable_keys
        if key.scope is VariableScope.BAG
    )
    return (
        np.concatenate(tuple(parts))
        if parts
        else np.empty(0, dtype=np.int64)
    )


def _solve_scaled_ieks_step(
    linearization: SparseBatchLinearization,
    coordinate_scale: Sequence[float],
    damping: float,
    *,
    optimize_shared: bool,
) -> ScaledIeksStep:
    hessian, scaled_gradient, scale, damping_value = (
        _validated_scaled_system(
            linearization, coordinate_scale, damping
        )
    )
    layout = linearization.layout
    persistent = _persistent_indices(layout, optimize_shared)
    persistent_dimension = int(persistent.size)
    reduced_hessian = _dense_block(hessian, persistent, persistent)
    if persistent_dimension:
        reduced_hessian += damping_value * np.eye(persistent_dimension)
    reduced_rhs = -scaled_gradient[persistent].copy()
    forward_records = {}
    diagnostics = []

    for bag_id in layout.bag_ids:
        knot_slices = _bag_knot_slices(layout, bag_id)
        bag_records = []
        previous_cross = None
        previous_solved_cross = None
        previous_solved_persistent = None
        previous_solved_rhs = None
        for knot_position, knot_slice in enumerate(knot_slices):
            knot = _coordinate_indices(knot_slice)
            information = _dense_block(hessian, knot, knot)
            information += damping_value * np.eye(_KNOT_DIMENSION)
            persistent_cross = _dense_block(
                hessian, knot, persistent
            )
            rhs = -scaled_gradient[knot].copy()
            if previous_cross is not None:
                information -= (
                    previous_cross.T @ previous_solved_cross
                )
                persistent_cross -= (
                    previous_cross.T @ previous_solved_persistent
                )
                rhs -= previous_cross.T @ previous_solved_rhs
            information = 0.5 * (information + information.T)
            try:
                factorization = cho_factor(
                    information,
                    lower=True,
                    overwrite_a=False,
                    check_finite=False,
                )
            except np.linalg.LinAlgError as error:
                raise np.linalg.LinAlgError(
                    "IEKS forward information block is singular for {!r} "
                    "at knot {}".format(bag_id, knot_position)
                ) from error

            solved_persistent = (
                cho_solve(
                    factorization,
                    persistent_cross,
                    check_finite=False,
                )
                if persistent_dimension
                else np.zeros((_KNOT_DIMENSION, 0), dtype=float)
            )
            solved_rhs = cho_solve(
                factorization, rhs, check_finite=False
            )
            if persistent_dimension:
                reduced_hessian -= (
                    persistent_cross.T @ solved_persistent
                )
                reduced_rhs -= persistent_cross.T @ solved_rhs

            if knot_position + 1 < len(knot_slices):
                next_knot = _coordinate_indices(
                    knot_slices[knot_position + 1]
                )
                next_cross = _dense_block(hessian, knot, next_knot)
                solved_next = cho_solve(
                    factorization, next_cross, check_finite=False
                )
            else:
                next_cross = None
                solved_next = None
            bag_records.append(
                (
                    knot,
                    factorization,
                    next_cross,
                    persistent_cross,
                    rhs,
                )
            )
            previous_cross = next_cross
            previous_solved_cross = solved_next
            previous_solved_persistent = solved_persistent
            previous_solved_rhs = solved_rhs
        forward_records[bag_id] = tuple(bag_records)
        bias_dimension = int(_bag_bias_indices(layout, bag_id).size)
        diagnostics.append(
            IeksBagSmoothingDiagnostics(
                bag_id=bag_id,
                local_dimension=(
                    layout.bag_slice(bag_id).stop
                    - layout.bag_slice(bag_id).start
                ),
                bias_dimension=bias_dimension,
                knot_count=len(knot_slices),
                forward_factorizations=len(knot_slices),
                backward_steps=len(knot_slices),
            )
        )

    if persistent_dimension:
        reduced_hessian = 0.5 * (
            reduced_hessian + reduced_hessian.T
        )
        try:
            reduced_factorization = cho_factor(
                reduced_hessian,
                lower=True,
                overwrite_a=False,
                check_finite=False,
            )
            persistent_delta = cho_solve(
                reduced_factorization,
                reduced_rhs,
                check_finite=False,
            )
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "IEKS persistent reduced system is singular"
            ) from error
    else:
        persistent_delta = np.empty(0, dtype=float)

    scaled_delta = np.zeros(layout.total_dimension, dtype=float)
    scaled_delta[persistent] = persistent_delta
    for bag_id in layout.bag_ids:
        next_delta = None
        for (
            knot,
            factorization,
            next_cross,
            persistent_cross,
            rhs,
        ) in reversed(forward_records[bag_id]):
            conditional_rhs = rhs - persistent_cross @ persistent_delta
            if next_cross is not None:
                conditional_rhs -= next_cross @ next_delta
            knot_delta = cho_solve(
                factorization, conditional_rhs, check_finite=False
            )
            scaled_delta[knot] = knot_delta
            next_delta = knot_delta

    delta = scale * scaled_delta
    predicted_reduction = -float(
        linearization.gradient @ delta
        + 0.5 * delta @ linearization.hessian.dot(delta)
    )
    optimized_gradient = (
        scaled_gradient
        if optimize_shared
        else scaled_gradient[layout.shared_slice.stop :]
    )
    gradient_inf_norm = float(
        np.linalg.norm(optimized_gradient, ord=np.inf)
    )
    scaled_step_norm = float(np.linalg.norm(scaled_delta))
    if (
        not np.all(np.isfinite(delta))
        or not np.all(np.isfinite(scaled_delta))
        or not np.isfinite(predicted_reduction)
        or not np.isfinite(gradient_inf_norm)
        or not np.isfinite(scaled_step_norm)
    ):
        raise np.linalg.LinAlgError("IEKS step diagnostics are non-finite")
    if not optimize_shared and np.any(
        delta[layout.shared_slice] != 0.0
    ):
        raise RuntimeError("conditional IEKS changed the fixed shared block")
    delta.setflags(write=False)
    scaled_delta.setflags(write=False)
    return ScaledIeksStep(
        delta=delta,
        scaled_delta=scaled_delta,
        gradient_inf_norm=gradient_inf_norm,
        scaled_step_norm=scaled_step_norm,
        predicted_reduction=predicted_reduction,
        bag_diagnostics=tuple(diagnostics),
    )


def solve_scaled_ieks_step(
    linearization: SparseBatchLinearization,
    coordinate_scale: Sequence[float],
    damping: float,
) -> ScaledIeksStep:
    """Solve one joint linearization by IEKS forward/backward recursions."""

    return _solve_scaled_ieks_step(
        linearization,
        coordinate_scale,
        damping,
        optimize_shared=True,
    )


def solve_scaled_conditional_ieks_step(
    linearization: SparseBatchLinearization,
    coordinate_scale: Sequence[float],
    damping: float,
) -> ScaledIeksStep:
    """Run IEKS with the shared 18-D plant coordinate held exactly fixed."""

    return _solve_scaled_ieks_step(
        linearization,
        coordinate_scale,
        damping,
        optimize_shared=False,
    )


__all__ = [
    "IeksBagSmoothingDiagnostics",
    "ScaledIeksStep",
    "solve_scaled_conditional_ieks_step",
    "solve_scaled_ieks_step",
    "validate_ieks_topology",
]
