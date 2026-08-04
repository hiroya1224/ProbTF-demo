"""Sparse assembly of whitened batch factor linearizations."""

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, isspmatrix_csc

from grape_param_estim.batch.factor import FactorEvaluation
from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.variables import VariableKey, VariableScope


@dataclass(frozen=True)
class FactorProvenance:
    """Input-factor identity, assembled rows, keys, and active-set flags."""

    factor_index: int
    row_slice: slice
    variable_keys: Tuple[VariableKey, ...]
    local_bag_id: Optional[str]
    active_set: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class SparseBatchLinearization:
    """Whitened sparse Jacobian, residual, and Gauss--Newton system."""

    layout: VariableLayout
    residual: np.ndarray
    jacobian: csc_matrix
    objective: float
    gradient: np.ndarray
    hessian: csc_matrix
    factor_provenance: Tuple[FactorProvenance, ...]

    @property
    def factor_row_slices(self) -> Tuple[slice, ...]:
        """Return row slices in input factor order."""

        return tuple(item.row_slice for item in self.factor_provenance)


def assemble_sparse_linearization(
    layout: VariableLayout,
    factor_evaluations: Sequence[FactorEvaluation],
) -> SparseBatchLinearization:
    """Assemble factor-local blocks without constructing a dense Jacobian.

    ``FactorEvaluation`` residuals and Jacobian blocks are assumed to have
    already been whitened by their owning factors.
    """

    if not isinstance(layout, VariableLayout):
        raise TypeError("layout must be a VariableLayout")
    try:
        factors = tuple(factor_evaluations)
    except TypeError as error:
        raise TypeError(
            "factor_evaluations must be an iterable of FactorEvaluation"
        ) from error
    if not factors:
        raise ValueError("factor_evaluations cannot be empty")
    if any(not isinstance(factor, FactorEvaluation) for factor in factors):
        raise TypeError(
            "factor_evaluations must contain only FactorEvaluation values"
        )

    total_rows = sum(factor.residual.size for factor in factors)
    residual = np.empty(total_rows, dtype=float)
    row_parts = []
    column_parts = []
    value_parts = []
    provenance = []
    referenced_keys = set()
    row_offset = 0

    for factor_index, factor in enumerate(factors):
        next_row = row_offset + factor.residual.size
        row_slice = slice(row_offset, next_row)
        residual[row_slice] = factor.residual
        factor_keys = tuple(
            block.variable_key for block in factor.jacobian_blocks
        )
        local_bag_ids = {
            key.bag_id
            for key in factor_keys
            if key.scope is not VariableScope.SHARED
        }
        if len(local_bag_ids) > 1:
            raise ValueError(
                "a factor cannot directly couple local variables from "
                "multiple bags"
            )
        local_bag_id = (
            next(iter(local_bag_ids)) if local_bag_ids else None
        )
        provenance.append(
            FactorProvenance(
                factor_index=factor_index,
                row_slice=row_slice,
                variable_keys=factor_keys,
                local_bag_id=local_bag_id,
                active_set=factor.active_set,
            )
        )

        for block in factor.jacobian_blocks:
            key = block.variable_key
            if key not in layout:
                raise ValueError(
                    "factor {} references a variable key absent from the layout"
                    .format(factor_index)
                )
            referenced_keys.add(key)
            column_slice = layout.column_slice(key)
            local_rows, local_columns = np.nonzero(block.value)
            if local_rows.size:
                row_parts.append(local_rows + row_offset)
                column_parts.append(local_columns + column_slice.start)
                value_parts.append(block.value[local_rows, local_columns])
        row_offset = next_row

    missing_keys = set(layout.variable_keys) - referenced_keys
    if missing_keys:
        raise ValueError(
            "factor_evaluations do not reference every layout key: {}".format(
                ", ".join(
                    key.kind.value
                    for key in layout.variable_keys
                    if key in missing_keys
                )
            )
        )

    if value_parts:
        rows = np.concatenate(row_parts)
        columns = np.concatenate(column_parts)
        values = np.concatenate(value_parts)
    else:
        rows = np.empty(0, dtype=np.int64)
        columns = np.empty(0, dtype=np.int64)
        values = np.empty(0, dtype=float)
    jacobian = coo_matrix(
        (values, (rows, columns)),
        shape=(total_rows, layout.total_dimension),
        dtype=float,
    ).tocsc()
    jacobian.sum_duplicates()
    jacobian.sort_indices()

    gradient = np.asarray(jacobian.T.dot(residual), dtype=float).reshape(-1)
    hessian = (jacobian.T @ jacobian).tocsc()
    hessian.sum_duplicates()
    hessian.eliminate_zeros()
    hessian.sort_indices()
    objective = 0.5 * math.fsum(
        factor.squared_error for factor in factors
    )

    if not isspmatrix_csc(jacobian) or not isspmatrix_csc(hessian):
        raise RuntimeError("sparse assembly did not produce CSC matrices")
    if (
        residual.shape != (total_rows,)
        or gradient.shape != (layout.total_dimension,)
        or not np.all(np.isfinite(residual))
        or not np.all(np.isfinite(gradient))
        or not np.all(np.isfinite(jacobian.data))
        or not np.all(np.isfinite(hessian.data))
        or not np.isfinite(objective)
    ):
        raise ValueError("assembled sparse linearization is not finite")

    residual.setflags(write=False)
    gradient.setflags(write=False)
    return SparseBatchLinearization(
        layout=layout,
        residual=residual,
        jacobian=jacobian,
        objective=objective,
        gradient=gradient,
        hessian=hessian,
        factor_provenance=tuple(provenance),
    )


__all__ = [
    "FactorProvenance",
    "SparseBatchLinearization",
    "assemble_sparse_linearization",
]
