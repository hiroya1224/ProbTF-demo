"""Laplace-marginal target for static plant parameters and command delay.

The fixed-delay graph objective supplied here is the complete negative log
joint objective for the graph, including the static-parameter prior.  The
target therefore adds only the delay log prior before subtracting the graph
objective and the bag-local Laplace volume term::

    log_target = log_p_delay - graph_objective - 0.5 * logdet(H_zz)

No static prior is added a second time.  ``H_zz`` is the undamped final
Gauss--Newton Hessian with respect to bag-local variables only.  Its mutually
independent bag blocks are factorized sparsely; this module never forms a
full dense Hessian or a Hessian inverse.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Tuple

import numpy as np
from scipy.sparse import isspmatrix_csc
from scipy.sparse.linalg import splu

from grape_param_estim.batch.lm import (
    LMSettings,
    solve_conditional_batch_map,
)
from grape_param_estim.batch.linearize import SparseBatchLinearization
from grape_param_estim.batch.problem import (
    BatchProblem,
    RecoverableModelEvaluationError,
)
from grape_param_estim.batch.state import BatchState
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.posterior.delayed_acceptance import (
    PosteriorPoint,
    TargetEvaluation,
)


DelayLogPrior = Callable[[float], float]


def _float64_bits(value: np.ndarray) -> bytes:
    return np.asarray(value, dtype="<f8").tobytes(order="C")


def _scalar_float64_bits(value: float) -> bytes:
    return np.asarray((float(value),), dtype="<f8").tobytes(order="C")


def _static_value(state: BatchState) -> np.ndarray:
    key = state.layout.variable_keys[0]
    if key.kind is not VariableKind.STATIC_PARAMETERS:
        raise ValueError("batch layout must begin with static parameters")
    return state.value(key)


@dataclass(frozen=True)
class FixedDelayLaplaceProblem:
    """One factory product for an exact static-coordinate/delay proposal.

    ``graph_objective_includes_static_prior`` is deliberately explicit.
    Returning ``False`` is a target-contract failure rather than an invitation
    for this module to guess which prior term is missing.
    """

    fixed_delay: float
    problem: BatchProblem
    initial_state: BatchState
    graph_objective_includes_static_prior: bool

    def __post_init__(self) -> None:
        delay = float(self.fixed_delay)
        if not np.isfinite(delay):
            raise ValueError("fixed_delay must be finite")
        if not isinstance(self.problem, BatchProblem):
            raise TypeError("problem must be a BatchProblem")
        if not isinstance(self.initial_state, BatchState):
            raise TypeError("initial_state must be a BatchState")
        if self.initial_state.layout != self.problem.layout:
            raise ValueError("initial_state layout must match problem layout")
        if not isinstance(
            self.graph_objective_includes_static_prior,
            (bool, np.bool_),
        ):
            raise TypeError(
                "graph_objective_includes_static_prior must be boolean"
            )
        object.__setattr__(self, "fixed_delay", delay)
        object.__setattr__(
            self,
            "graph_objective_includes_static_prior",
            bool(self.graph_objective_includes_static_prior),
        )


FixedDelayProblemFactory = Callable[
    [PosteriorPoint], FixedDelayLaplaceProblem
]


@dataclass(frozen=True)
class ConditionalTrajectoryWarmStart:
    """Accepted conditional trajectory safe to rebase onto another graph.

    The exact source point key makes the payload self-auditing.  A subsequent
    target evaluation transfers only bag-local values and retains the new
    factory's shared coordinate.  If the graph layout changed, the payload is
    ignored and the factory initialization is used instead.
    """

    state: BatchState
    source_point_cache_key: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.state, BatchState):
            raise TypeError("state must be a BatchState")
        if not isinstance(self.source_point_cache_key, bytes):
            raise TypeError("source_point_cache_key must be bytes")
        expected_size = 19 * np.dtype("<f8").itemsize
        if len(self.source_point_cache_key) != expected_size:
            raise ValueError("source_point_cache_key has the wrong size")
        decoded_point = np.frombuffer(
            self.source_point_cache_key, dtype="<f8"
        )
        if not np.all(np.isfinite(decoded_point)):
            raise ValueError("source_point_cache_key must encode finite values")
        static_size = 18 * np.dtype("<f8").itemsize
        if (
            _float64_bits(_static_value(self.state))
            != self.source_point_cache_key[:static_size]
        ):
            raise ValueError(
                "warm-start state does not match its exact source point"
            )


@dataclass(frozen=True)
class BagLocalLogDeterminant:
    """Sparse factorization evidence for one bag-local Hessian block."""

    bag_id: str
    dimension: int
    hessian_nnz: int
    factor_l_nnz: int
    factor_u_nnz: int
    log_determinant: float


@dataclass(frozen=True)
class LocalLogDeterminant:
    """Sum of undamped bag-local Hessian log determinants."""

    value: float
    bags: Tuple[BagLocalLogDeterminant, ...]


class LaplaceTargetModelFailure(ValueError):
    """A proposal is outside a model domain declared by the graph factory."""

    def __init__(self, reason: str):
        if (
            not isinstance(reason, str)
            or not reason
            or reason.strip() != reason
            or any(character.isspace() for character in reason)
        ):
            raise ValueError(
                "model failure reason must be a canonical non-empty token"
            )
        self.reason = reason
        super().__init__(reason)


def _permutation_sign(permutation: np.ndarray) -> int:
    values = np.asarray(permutation, dtype=np.int64)
    size = values.size
    if values.shape != (size,) or not np.array_equal(
        np.sort(values), np.arange(size, dtype=np.int64)
    ):
        raise np.linalg.LinAlgError("sparse factor returned an invalid permutation")
    visited = np.zeros(size, dtype=bool)
    cycles = 0
    for start in range(size):
        if visited[start]:
            continue
        cycles += 1
        current = start
        while not visited[current]:
            visited[current] = True
            current = int(values[current])
    return -1 if (size - cycles) % 2 else 1


def factorize_bag_local_hessian(
    linearization: SparseBatchLinearization,
) -> LocalLogDeterminant:
    """Factor the undamped physical-coordinate ``H_zz`` bag by bag."""

    if not isinstance(linearization, SparseBatchLinearization):
        raise TypeError(
            "linearization must be a SparseBatchLinearization"
        )
    layout = linearization.layout
    hessian = linearization.hessian
    if (
        not isspmatrix_csc(hessian)
        or hessian.shape != (layout.total_dimension, layout.total_dimension)
        or not np.all(np.isfinite(hessian.data))
    ):
        raise ValueError("linearization Hessian must be a finite CSC matrix")
    if layout.shared_slice != slice(0, 18):
        raise ValueError("layout must begin with the shared 18-D block")
    asymmetry = (hessian - hessian.T).tocsc()
    asymmetry.eliminate_zeros()
    if asymmetry.nnz:
        largest_asymmetry = float(np.max(np.abs(asymmetry.data)))
        largest_entry = (
            float(np.max(np.abs(hessian.data))) if hessian.nnz else 0.0
        )
        if largest_asymmetry > 1.0e-12 * max(1.0, largest_entry):
            raise ValueError("local Laplace Hessian must be symmetric")

    diagnostics = []
    total = 0.0
    for first_index, bag_id in enumerate(layout.bag_ids):
        local_slice = layout.bag_slice(bag_id)
        for other_bag_id in layout.bag_ids[first_index + 1 :]:
            cross = hessian[
                local_slice, layout.bag_slice(other_bag_id)
            ].tocsc()
            cross.eliminate_zeros()
            if cross.nnz:
                raise ValueError(
                    "H_zz directly couples variables from different bags"
                )
        local = hessian[local_slice, local_slice].tocsc()
        local.sum_duplicates()
        local.eliminate_zeros()
        local.sort_indices()
        dimension = local_slice.stop - local_slice.start
        try:
            factorization = splu(local)
        except RuntimeError as error:
            raise np.linalg.LinAlgError(
                "bag-local Hessian is singular for {!r}".format(bag_id)
            ) from error
        diagonal = np.asarray(factorization.U.diagonal(), dtype=float)
        if (
            diagonal.shape != (dimension,)
            or not np.all(np.isfinite(diagonal))
            or np.any(diagonal == 0.0)
        ):
            raise np.linalg.LinAlgError(
                "bag-local Hessian determinant is singular for {!r}".format(
                    bag_id
                )
            )
        determinant_sign = (
            _permutation_sign(factorization.perm_r)
            * _permutation_sign(factorization.perm_c)
            * (-1 if np.count_nonzero(diagonal < 0.0) % 2 else 1)
        )
        if determinant_sign <= 0:
            raise np.linalg.LinAlgError(
                "bag-local Hessian is not positive definite for {!r}".format(
                    bag_id
                )
            )
        log_determinant = float(np.sum(np.log(np.abs(diagonal))))
        if not np.isfinite(log_determinant):
            raise np.linalg.LinAlgError(
                "bag-local Hessian log determinant is non-finite for {!r}"
                .format(bag_id)
            )
        diagnostics.append(
            BagLocalLogDeterminant(
                bag_id=bag_id,
                dimension=dimension,
                hessian_nnz=local.nnz,
                factor_l_nnz=factorization.L.nnz,
                factor_u_nnz=factorization.U.nnz,
                log_determinant=log_determinant,
            )
        )
        total += log_determinant
    if not diagnostics or not np.isfinite(total):
        raise np.linalg.LinAlgError(
            "local Laplace log determinant is unavailable"
        )
    return LocalLogDeterminant(float(total), tuple(diagnostics))


def _warm_started_state(
    initial_state: BatchState,
    warm_start: Any,
) -> BatchState:
    if not isinstance(warm_start, ConditionalTrajectoryWarmStart):
        return initial_state
    source = warm_start.state
    if source.layout != initial_state.layout:
        return initial_state
    return _copy_local_values(initial_state, source)


def _copy_local_values(
    shared_source: BatchState,
    local_source: BatchState,
) -> BatchState:
    if shared_source.layout != local_source.layout:
        raise ValueError("local state transfer requires identical layouts")
    values = {}
    for key in shared_source.layout.variable_keys:
        values[key] = (
            shared_source.value(key)
            if key.kind is VariableKind.STATIC_PARAMETERS
            else local_source.value(key)
        )
    result = BatchState(shared_source.layout, values)
    if _float64_bits(_static_value(result)) != _float64_bits(
        _static_value(shared_source)
    ):
        raise RuntimeError("local warm start changed the shared coordinate")
    return result


class LaplaceMarginalTarget:
    """Callable exact target used by delayed-acceptance MCMC."""

    def __init__(
        self,
        problem_factory: FixedDelayProblemFactory,
        delay_log_prior: DelayLogPrior,
        lm_settings: LMSettings = LMSettings(),
    ) -> None:
        if not callable(problem_factory):
            raise TypeError("problem_factory must be callable")
        if not callable(delay_log_prior):
            raise TypeError("delay_log_prior must be callable")
        if not isinstance(lm_settings, LMSettings):
            raise TypeError("lm_settings must be LMSettings")
        self.problem_factory = problem_factory
        self.delay_log_prior = delay_log_prior
        self.lm_settings = lm_settings

    def __call__(
        self,
        point: PosteriorPoint,
        warm_start: Any = None,
    ) -> TargetEvaluation:
        return self.evaluate(point, warm_start)

    def evaluate(
        self,
        point: PosteriorPoint,
        warm_start: Any = None,
    ) -> TargetEvaluation:
        if not isinstance(point, PosteriorPoint):
            raise TypeError("point must be a PosteriorPoint")
        try:
            delay_prior = self.delay_log_prior(point.delay)
        except (ArithmeticError, ValueError, LaplaceTargetModelFailure):
            return TargetEvaluation.failure(point, "delay_prior_failure")
        if isinstance(delay_prior, (bool, np.bool_)) or not isinstance(
            delay_prior, Real
        ):
            return TargetEvaluation.failure(point, "delay_prior_failure")
        delay_prior = float(delay_prior)
        if delay_prior == float("-inf"):
            return TargetEvaluation.failure(point, "delay_prior_out_of_support")
        if not np.isfinite(delay_prior):
            return TargetEvaluation.failure(point, "delay_prior_nonfinite")

        try:
            fixed = self.problem_factory(point)
        except LaplaceTargetModelFailure as error:
            return TargetEvaluation.failure(
                point,
                "problem_factory_model_failure:{}".format(error.reason),
            )
        except (ArithmeticError, ValueError):
            return TargetEvaluation.failure(
                point, "problem_factory_model_failure"
            )
        if not isinstance(fixed, FixedDelayLaplaceProblem):
            return TargetEvaluation.failure(
                point, "problem_factory_contract_failure"
            )
        if not fixed.graph_objective_includes_static_prior:
            return TargetEvaluation.failure(
                point, "graph_objective_missing_static_prior"
            )
        if _scalar_float64_bits(fixed.fixed_delay) != _scalar_float64_bits(
            point.delay
        ):
            return TargetEvaluation.failure(point, "factory_delay_mismatch")
        if _float64_bits(_static_value(fixed.initial_state)) != _float64_bits(
            point.static_coordinate
        ):
            return TargetEvaluation.failure(
                point, "factory_static_coordinate_mismatch"
            )

        initial_state = _warm_started_state(fixed.initial_state, warm_start)
        try:
            solution = solve_conditional_batch_map(
                fixed.problem,
                initial_state,
                self.lm_settings,
            )
        except RecoverableModelEvaluationError:
            return TargetEvaluation.failure(point, "lm_model_failure")
        except (ArithmeticError, ValueError, RuntimeError):
            return TargetEvaluation.failure(point, "lm_numerical_failure")
        iterations = len(solution.iterations)
        if not solution.converged:
            return TargetEvaluation.failure(
                point,
                "lm_nonconverged:{}".format(solution.reason.value),
                iterations,
            )
        # The conditional solver has an exact zero shared increment.  Restore
        # the factory's shared array nevertheless so even signed-zero bits are
        # preserved rather than being normalized by ``x + 0`` retraction.
        conditional_state = _copy_local_values(
            fixed.initial_state, solution.state
        )

        try:
            final = fixed.problem.linearize(conditional_state).sparse
        except RecoverableModelEvaluationError:
            return TargetEvaluation.failure(
                point, "final_model_failure", iterations
            )
        except (ArithmeticError, ValueError):
            return TargetEvaluation.failure(
                point, "final_numerical_failure", iterations
            )
        objective = float(final.objective)
        if not np.isfinite(objective):
            return TargetEvaluation.failure(
                point, "graph_objective_nonfinite", iterations
            )
        try:
            log_determinant = factorize_bag_local_hessian(final).value
        except (np.linalg.LinAlgError, ValueError):
            return TargetEvaluation.failure(
                point, "local_hessian_factorization_failure", iterations
            )
        log_density = delay_prior - objective - 0.5 * log_determinant
        if not np.isfinite(log_density):
            return TargetEvaluation.failure(
                point, "target_density_nonfinite", iterations
            )
        payload = ConditionalTrajectoryWarmStart(
            conditional_state,
            point.exact_cache_key,
        )
        return TargetEvaluation(
            point=point,
            log_density=log_density,
            successful=True,
            failure_reason="",
            inner_iterations=iterations,
            warm_start=payload,
            graph_objective=objective,
            local_log_determinant=log_determinant,
            delay_log_prior=delay_prior,
        )


__all__ = [
    "BagLocalLogDeterminant",
    "ConditionalTrajectoryWarmStart",
    "DelayLogPrior",
    "FixedDelayLaplaceProblem",
    "FixedDelayProblemFactory",
    "LaplaceMarginalTarget",
    "LaplaceTargetModelFailure",
    "LocalLogDeterminant",
    "factorize_bag_local_hessian",
]
