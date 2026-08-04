"""Measured performance evidence for strict sparse-batch artifacts."""

import resource
import time
from typing import Tuple

import numpy as np
from scipy.sparse.linalg import splu

from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.linearize import assemble_sparse_linearization
from grape_param_estim.batch.variables import VariableScope
from grape_param_estim.batch_artifact_export import (
    BagPerformanceMeasurements,
    RunPerformanceMeasurements,
)
from grape_param_estim.estimation import FixedGraphLaplaceSolution
from grape_param_estim.real_estimation import RealEstimationResult


def _peak_memory_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB while macOS reports bytes.  ROS Noetic production is
    # Linux; retain a guarded branch so unit tests on macOS remain meaningful.
    return value if value > 10**10 else value * 1024


def _dense_sparse_columns(matrix) -> np.ndarray:
    result = np.zeros(matrix.shape, dtype=float)
    for column in range(matrix.shape[1]):
        start = matrix.indptr[column]
        stop = matrix.indptr[column + 1]
        result[matrix.indices[start:stop], column] = (
            matrix.data[start:stop]
        )
    return result


def measure_final_solution_bags(
    solution: FixedGraphLaplaceSolution,
) -> Tuple[BagPerformanceMeasurements, ...]:
    """Benchmark final factor assembly and local Schur work per bag.

    Factor evaluation has already happened in ``final_linearization``.  The
    assembly timing therefore measures the sparse Jacobian/normal-system
    construction itself.  Factorization and Schur timings use the undamped
    final Hessian that defines the Laplace geometry, never an LM matrix.
    """

    if not isinstance(solution, FixedGraphLaplaceSolution):
        raise TypeError("solution must be FixedGraphLaplaceSolution")
    final = solution.final_linearization
    layout = final.sparse.layout
    if layout != solution.lm.state.layout:
        raise ValueError("final linearization layout disagrees with MAP state")
    prepared = {value.bag_id: value for value in solution.prepared.bags}
    result = []
    for bag_id in layout.bag_ids:
        bag_provenance = tuple(
            value
            for value in final.sparse.factor_provenance
            if value.local_bag_id == bag_id
        )
        if not bag_provenance:
            raise ValueError("bag has no local factors")
        local_factor_indices = tuple(
            value.factor_index for value in bag_provenance
        )
        local_factors = tuple(
            final.factors[index] for index in local_factor_indices
        )
        shared_factors = tuple(
            final.factors[value.factor_index]
            for value in final.sparse.factor_provenance
            if value.local_bag_id is None
        )
        sublayout = VariableLayout(
            tuple(
                key
                for key in layout.variable_keys
                if key.scope is VariableScope.SHARED
                or key.bag_id == bag_id
            )
        )
        assembly_started = time.perf_counter()
        assembled = assemble_sparse_linearization(
            sublayout, shared_factors + local_factors
        )
        assembly_seconds = time.perf_counter() - assembly_started

        local_slice = layout.bag_slice(bag_id)
        shared_slice = layout.shared_slice
        local_hessian = final.sparse.hessian[
            local_slice, local_slice
        ].tocsc()
        factorization_started = time.perf_counter()
        factorization = splu(local_hessian)
        factorization_seconds = time.perf_counter() - factorization_started
        cross = final.sparse.hessian[local_slice, shared_slice].tocsc()
        dense_cross = _dense_sparse_columns(cross)
        rhs = np.column_stack(
            (final.sparse.gradient[local_slice], dense_cross)
        )
        schur_started = time.perf_counter()
        solved = np.asarray(factorization.solve(rhs), dtype=float)
        _ = dense_cross.T @ solved[:, 1:]
        _ = dense_cross.T @ solved[:, 0]
        schur_seconds = time.perf_counter() - schur_started
        if not np.all(np.isfinite(solved)):
            raise np.linalg.LinAlgError("measured Schur solve was non-finite")

        local_rows = np.concatenate(
            tuple(
                np.arange(value.row_slice.start, value.row_slice.stop)
                for value in bag_provenance
            )
        )
        result.append(
            BagPerformanceMeasurements(
                bag_id=bag_id,
                knot_count=len(prepared[bag_id].knots),
                factor_count=len(local_factors),
                residual_dimension=sum(
                    value.residual.size for value in local_factors
                ),
                jacobian_nnz=final.sparse.jacobian[
                    local_rows, :
                ].nnz,
                assembly_seconds=assembly_seconds,
                factorization_seconds=factorization_seconds,
                schur_solve_seconds=schur_seconds,
            )
        )
        if assembled.layout != sublayout:
            raise RuntimeError("measured sparse assembly changed its layout")
    return tuple(result)


def measure_run_performance(
    result: RealEstimationResult,
) -> RunPerformanceMeasurements:
    """Collect measured work-unit timings for the selected strict run."""

    if not isinstance(result, RealEstimationResult):
        raise TypeError("result must be RealEstimationResult")
    nonlinear = tuple(
        value
        for mode in result.modes
        for value in mode.nonlinear_iteration_seconds
    )
    em = tuple(
        value for mode in result.modes for value in mode.em_iteration_seconds
    )
    return RunPerformanceMeasurements(
        bags=measure_final_solution_bags(result.selected_mode.final_solution),
        nonlinear_iteration_seconds=nonlinear,
        em_iteration_seconds=em,
        mcmc_target_seconds=result.mcmc_target_seconds,
        peak_memory_bytes=_peak_memory_bytes(),
    )


__all__ = [
    "measure_final_solution_bags",
    "measure_run_performance",
]
