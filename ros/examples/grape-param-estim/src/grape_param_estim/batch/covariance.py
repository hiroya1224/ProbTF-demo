"""Arrowhead Laplace covariance queries without forming a full inverse."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.sparse import csc_matrix, isspmatrix_csc
from scipy.sparse.linalg import splu

from grape_param_estim.batch.factor import JacobianBlock
from grape_param_estim.batch.linearize import SparseBatchLinearization
from grape_param_estim.batch.variables import VariableKey


@dataclass(frozen=True)
class LaplaceFactorizationDiagnostics:
    """Sparse factorization size and determinant evidence."""

    total_dimension: int
    hessian_nnz: int
    local_factor_nnz: Tuple[Tuple[str, int, int], ...]
    reduced_dimension: int
    log_determinant: float


def _sparse_columns_to_dense(matrix: csc_matrix) -> np.ndarray:
    result = np.zeros(matrix.shape, dtype=float)
    for column in range(matrix.shape[1]):
        start = matrix.indptr[column]
        stop = matrix.indptr[column + 1]
        result[matrix.indices[start:stop], column] = matrix.data[start:stop]
    return result


class ArrowheadLaplaceFactorization:
    """Reusable covariance action for one positive-definite MAP Hessian."""

    def __init__(self, linearization: SparseBatchLinearization):
        if not isinstance(linearization, SparseBatchLinearization):
            raise TypeError("linearization must be a SparseBatchLinearization")
        layout = linearization.layout
        hessian = linearization.hessian
        dimension = layout.total_dimension
        if (
            not isspmatrix_csc(hessian)
            or hessian.shape != (dimension, dimension)
            or not np.all(np.isfinite(hessian.data))
        ):
            raise ValueError("linearization Hessian must be finite CSC")
        asymmetry = (hessian - hessian.T).tocsc()
        asymmetry.eliminate_zeros()
        if asymmetry.nnz:
            largest = float(np.max(np.abs(asymmetry.data)))
            scale = (
                float(np.max(np.abs(hessian.data))) if hessian.nnz else 0.0
            )
            if largest > 1.0e-12 * max(1.0, scale):
                raise ValueError("Laplace Hessian must be symmetric")

        shared_slice = layout.shared_slice
        if shared_slice != slice(0, 18):
            raise ValueError("layout must begin with the shared 18-D block")
        reduced = hessian[shared_slice, shared_slice].toarray()
        local_factorizations = {}
        local_cross = {}
        local_cross_solutions = {}
        local_diagnostics = []
        log_determinant = 0.0
        for bag_id in layout.bag_ids:
            local_slice = layout.bag_slice(bag_id)
            local_hessian = hessian[local_slice, local_slice].tocsc()
            try:
                factorization = splu(local_hessian)
            except RuntimeError as error:
                raise np.linalg.LinAlgError(
                    "bag-local MAP Hessian is singular for {!r}".format(bag_id)
                ) from error
            cross = hessian[local_slice, shared_slice].tocsc()
            dense_cross = _sparse_columns_to_dense(cross)
            solved_cross = np.asarray(
                factorization.solve(dense_cross), dtype=float
            )
            if not np.all(np.isfinite(solved_cross)):
                raise np.linalg.LinAlgError(
                    "bag-local covariance solve is non-finite for {!r}".format(
                        bag_id
                    )
                )
            reduced -= dense_cross.T @ solved_cross
            diagonal = np.asarray(factorization.U.diagonal(), dtype=float)
            if (
                diagonal.shape != (local_slice.stop - local_slice.start,)
                or np.any(diagonal == 0.0)
                or not np.all(np.isfinite(diagonal))
            ):
                raise np.linalg.LinAlgError(
                    "bag-local determinant is singular for {!r}".format(bag_id)
                )
            log_determinant += float(np.sum(np.log(np.abs(diagonal))))
            local_factorizations[bag_id] = factorization
            local_cross[bag_id] = dense_cross
            local_cross_solutions[bag_id] = solved_cross
            local_diagnostics.append(
                (bag_id, factorization.L.nnz, factorization.U.nnz)
            )

        reduced = 0.5 * (reduced + reduced.T)
        try:
            reduced_cholesky = np.linalg.cholesky(reduced)
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "reduced MAP Hessian is not positive definite"
            ) from error
        log_determinant += 2.0 * float(
            np.sum(np.log(np.diag(reduced_cholesky)))
        )
        if not np.isfinite(log_determinant):
            raise np.linalg.LinAlgError("Laplace log determinant is not finite")

        self._layout = layout
        self._hessian = hessian
        self._local_factorizations = local_factorizations
        self._local_cross = local_cross
        self._local_cross_solutions = local_cross_solutions
        reduced.setflags(write=False)
        self._reduced_hessian = reduced
        self._reduced_cholesky = reduced_cholesky
        self._diagnostics = LaplaceFactorizationDiagnostics(
            total_dimension=dimension,
            hessian_nnz=hessian.nnz,
            local_factor_nnz=tuple(local_diagnostics),
            reduced_dimension=18,
            log_determinant=log_determinant,
        )

    @property
    def diagnostics(self) -> LaplaceFactorizationDiagnostics:
        return self._diagnostics

    @property
    def layout(self):
        return self._layout

    @property
    def reduced_hessian(self) -> np.ndarray:
        """Return the undamped physical 18-D Schur complement at the MAP."""

        return self._reduced_hessian

    def solve(self, right_hand_side: np.ndarray) -> np.ndarray:
        """Apply the MAP Hessian inverse to one or many right-hand sides."""

        rhs = np.asarray(right_hand_side, dtype=float)
        vector_input = rhs.ndim == 1
        if vector_input:
            rhs = rhs.reshape(-1, 1)
        if (
            rhs.ndim != 2
            or rhs.shape[0] != self._layout.total_dimension
            or rhs.shape[1] == 0
            or not np.all(np.isfinite(rhs))
        ):
            raise ValueError(
                "right_hand_side must have one finite row per layout column"
            )
        preliminary = {}
        reduced_rhs = rhs[self._layout.shared_slice, :].copy()
        for bag_id in self._layout.bag_ids:
            local_slice = self._layout.bag_slice(bag_id)
            local_solution = np.asarray(
                self._local_factorizations[bag_id].solve(rhs[local_slice, :]),
                dtype=float,
            )
            preliminary[bag_id] = local_solution
            reduced_rhs -= self._local_cross[bag_id].T @ local_solution
        shared_solution = np.linalg.solve(
            self._reduced_cholesky.T,
            np.linalg.solve(self._reduced_cholesky, reduced_rhs),
        )
        solution = np.empty_like(rhs)
        solution[self._layout.shared_slice, :] = shared_solution
        for bag_id in self._layout.bag_ids:
            local_slice = self._layout.bag_slice(bag_id)
            solution[local_slice, :] = (
                preliminary[bag_id]
                - self._local_cross_solutions[bag_id] @ shared_solution
            )
        if not np.all(np.isfinite(solution)):
            raise np.linalg.LinAlgError("Laplace covariance solve is non-finite")
        return solution[:, 0] if vector_input else solution

    def selected_covariance(
        self,
        variable_keys: Sequence[VariableKey],
    ) -> np.ndarray:
        """Return a dense marginal covariance for selected variable blocks."""

        try:
            keys = tuple(variable_keys)
        except TypeError as error:
            raise TypeError("variable_keys must be iterable") from error
        if not keys or any(not isinstance(key, VariableKey) for key in keys):
            raise TypeError("variable_keys must contain VariableKey values")
        if len(set(keys)) != len(keys):
            raise ValueError("variable_keys must be unique")
        columns = []
        for key in keys:
            if key not in self._layout:
                raise KeyError("selected variable key is absent from the layout")
            local_slice = self._layout.column_slice(key)
            columns.extend(range(local_slice.start, local_slice.stop))
        basis = np.zeros(
            (self._layout.total_dimension, len(columns)), dtype=float
        )
        basis[columns, np.arange(len(columns))] = 1.0
        solved = self.solve(basis)
        covariance = solved[columns, :]
        covariance = 0.5 * (covariance + covariance.T)
        return covariance

    def residual_covariance(
        self,
        jacobian_blocks: Tuple[JacobianBlock, ...],
    ) -> np.ndarray:
        """Return ``J H^-1 J.T`` for one local analytic residual Jacobian."""

        if type(jacobian_blocks) is not tuple or not jacobian_blocks:
            raise TypeError("jacobian_blocks must be a non-empty tuple")
        if any(not isinstance(block, JacobianBlock) for block in jacobian_blocks):
            raise TypeError("jacobian_blocks must contain JacobianBlock values")
        if len({block.variable_key for block in jacobian_blocks}) != len(
            jacobian_blocks
        ):
            raise ValueError("jacobian_blocks must have unique variable keys")
        residual_dimension = jacobian_blocks[0].value.shape[0]
        if any(
            block.value.shape[0] != residual_dimension
            for block in jacobian_blocks
        ):
            raise ValueError("jacobian block row counts must match")
        transpose = np.zeros(
            (self._layout.total_dimension, residual_dimension), dtype=float
        )
        for block in jacobian_blocks:
            if block.variable_key not in self._layout:
                raise KeyError("Jacobian variable key is absent from the layout")
            transpose[self._layout.column_slice(block.variable_key), :] = (
                block.value.T
            )
        solved = self.solve(transpose)
        covariance = np.zeros(
            (residual_dimension, residual_dimension), dtype=float
        )
        for block in jacobian_blocks:
            covariance += block.value @ solved[
                self._layout.column_slice(block.variable_key), :
            ]
        covariance = 0.5 * (covariance + covariance.T)
        if not np.all(np.isfinite(covariance)):
            raise np.linalg.LinAlgError("residual covariance is non-finite")
        return covariance


__all__ = [
    "ArrowheadLaplaceFactorization",
    "LaplaceFactorizationDiagnostics",
]
