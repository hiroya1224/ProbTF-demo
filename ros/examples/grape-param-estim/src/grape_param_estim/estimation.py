"""Production orchestration for sparse MAP, lag profiling, and Laplace-EM.

The orchestration boundary is deliberately ROS-free.  A caller supplies a
factory that rebuilds one fixed graph for an explicit diagonal ``Q``, delay,
and 18-D static-coordinate initialization.  This is important for delayed
ZOH commands: changing delay must rebuild command segments rather than mutate
an already prepared graph or pretend that delay has an ordinary derivative.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Callable, Optional, Tuple

import numpy as np

from grape_param_estim.batch.covariance import (
    ArrowheadLaplaceFactorization,
)
from grape_param_estim.batch.dynamics_moments import (
    DynamicsLaplaceMoments,
    compute_expected_dynamics_moments,
    evaluate_prepared_dynamics_intervals,
)
from grape_param_estim.batch.em_loop import (
    EStepPhase,
    LaplaceEStepFailure,
    LaplaceEStepResult,
)
from grape_param_estim.batch.evidence import (
    MarginalObjectiveBreakdown,
    StaticLaplaceGeometry,
    approximate_marginal_objective,
    compute_static_laplace_geometry,
)
from grape_param_estim.batch.graph_builder import (
    PreparedBatchGraphData,
    build_fixed_batch_problem,
    build_initial_batch_state,
)
from grape_param_estim.batch.lag_profile import (
    LagObjectiveResult,
    LagProfileFailure,
    LagProfileResult,
    LagProfileSettings,
    optimize_lag_profile,
)
from grape_param_estim.batch.lm import (
    BatchMapCancelled,
    LMIterationRecord,
    LMSettings,
    LMSolveResult,
    solve_batch_map,
)
from grape_param_estim.batch.problem import (
    BatchProblem,
    ProblemLinearization,
    RecoverableModelEvaluationError,
)
from grape_param_estim.batch.state import BatchState
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.parameterization import PARAMETER_DIMENSION
from grape_param_estim.posterior.delayed_acceptance import PosteriorPoint
from grape_param_estim.posterior.laplace_target import (
    FixedDelayLaplaceProblem,
)


PreparedGraphFactory = Callable[
    [np.ndarray, float, np.ndarray], PreparedBatchGraphData
]
CancellationCheck = Callable[[], bool]
LMProgress = Callable[[LMIterationRecord], None]


def _positive_q(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (6,)
        or not np.all(np.isfinite(result))
        or np.any(result <= 0.0)
    ):
        raise ValueError("q must contain six positive finite values")
    copied = result.copy()
    copied.setflags(write=False)
    return copied


def _static_coordinate(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (PARAMETER_DIMENSION,)
        or not np.all(np.isfinite(result))
    ):
        raise ValueError("static coordinate must contain 18 finite values")
    copied = result.copy()
    copied.setflags(write=False)
    return copied


def _state_static_coordinate(state: BatchState) -> np.ndarray:
    key = state.layout.variable_keys[0]
    if key.kind is not VariableKind.STATIC_PARAMETERS:
        raise ValueError("batch layout must begin with static parameters")
    return state.value(key)


def _warm_started_state(
    prepared_initial: BatchState,
    warm_start: Optional[BatchState],
) -> BatchState:
    if warm_start is None or warm_start.layout != prepared_initial.layout:
        return prepared_initial
    return BatchState(
        prepared_initial.layout,
        {
            key: warm_start.value(key)
            for key in prepared_initial.layout.variable_keys
        },
    )


@dataclass(frozen=True)
class FixedGraphLaplaceSolution:
    """One converged fixed-Q, fixed-delay joint MAP and its covariance pass."""

    prepared: PreparedBatchGraphData
    problem: BatchProblem
    lm: LMSolveResult
    final_linearization: ProblemLinearization
    factorization: ArrowheadLaplaceFactorization
    dynamics: DynamicsLaplaceMoments
    marginal_objective: MarginalObjectiveBreakdown

    def __post_init__(self) -> None:
        expected = (
            (self.prepared, PreparedBatchGraphData, "prepared"),
            (self.problem, BatchProblem, "problem"),
            (self.lm, LMSolveResult, "lm"),
            (
                self.final_linearization,
                ProblemLinearization,
                "final_linearization",
            ),
            (
                self.factorization,
                ArrowheadLaplaceFactorization,
                "factorization",
            ),
            (self.dynamics, DynamicsLaplaceMoments, "dynamics"),
            (
                self.marginal_objective,
                MarginalObjectiveBreakdown,
                "marginal_objective",
            ),
        )
        for value, expected_type, name in expected:
            if not isinstance(value, expected_type):
                raise TypeError("{} has an invalid type".format(name))
        if not self.lm.converged:
            raise ValueError("a fixed graph Laplace solution must converge")
        if self.lm.state.layout != self.problem.layout:
            raise ValueError("LM state layout does not match its problem")
        if self.factorization.layout != self.problem.layout:
            raise ValueError("factorization layout does not match its problem")
        if self.dynamics.definition != self.prepared.dynamics.q_definition:
            raise ValueError("dynamics Q definition changed during solve")

    def as_e_step_result(
        self,
        *,
        inner_iterations: Optional[int] = None,
        termination_reason: Optional[str] = None,
    ) -> LaplaceEStepResult:
        count = (
            len(self.lm.iterations)
            if inner_iterations is None
            else int(inner_iterations)
        )
        reason = (
            self.lm.reason.value
            if termination_reason is None
            else str(termination_reason)
        )
        return LaplaceEStepResult(
            q=self.prepared.dynamics.q,
            lag=self.prepared.fixed_delay,
            state=self.lm.state,
            moments=self.dynamics.moments,
            map_objective=self.lm.objective,
            approximate_marginal_objective=(
                self.marginal_objective.value
            ),
            inner_iterations=count,
            termination_reason=reason,
        )

    def static_geometry(
        self,
        *,
        relative_rank_tolerance: float = 1.0e-10,
    ) -> StaticLaplaceGeometry:
        return compute_static_laplace_geometry(
            self.factorization,
            self.prepared.static_parameter_prior.covariance
            .square_root_information,
            self.prepared.parameter_chart.ridge_direction(),
            relative_rank_tolerance=relative_rank_tolerance,
        )


class FixedGraphSolveFailure(RuntimeError):
    """A fixed graph could not produce a converged undamped Laplace point."""

    def __init__(self, reason: str, inner_iterations: int = 0):
        if not isinstance(reason, str) or not reason:
            raise ValueError("failure reason must be a non-empty string")
        self.reason = reason
        self.inner_iterations = int(inner_iterations)
        super().__init__(reason)


class EstimationCancelled(RuntimeError):
    """Cancellation observed at an outer solver boundary."""


def solve_fixed_graph_laplace(
    graph_factory: PreparedGraphFactory,
    q: np.ndarray,
    fixed_delay: float,
    static_initial_coordinate: np.ndarray,
    lm_settings: LMSettings = LMSettings(),
    *,
    warm_start: Optional[BatchState] = None,
    cancellation_requested: Optional[CancellationCheck] = None,
    lm_progress: Optional[LMProgress] = None,
) -> FixedGraphLaplaceSolution:
    """Solve and factor one graph without retaining LM damping in precision."""

    if not callable(graph_factory):
        raise TypeError("graph_factory must be callable")
    selected_q = _positive_q(q)
    delay = float(fixed_delay)
    if not np.isfinite(delay) or delay < 0.0:
        raise ValueError("fixed_delay must be finite and non-negative")
    static = _static_coordinate(static_initial_coordinate)
    if not isinstance(lm_settings, LMSettings):
        raise TypeError("lm_settings must be LMSettings")
    if warm_start is not None and not isinstance(warm_start, BatchState):
        raise TypeError("warm_start must be BatchState or None")
    for callback, name in (
        (cancellation_requested, "cancellation_requested"),
        (lm_progress, "lm_progress"),
    ):
        if callback is not None and not callable(callback):
            raise TypeError("{} must be callable".format(name))

    try:
        prepared = graph_factory(selected_q.copy(), delay, static.copy())
    except (ArithmeticError, ValueError) as error:
        raise FixedGraphSolveFailure("graph_preparation_failure") from error
    if not isinstance(prepared, PreparedBatchGraphData):
        raise TypeError("graph_factory must return PreparedBatchGraphData")
    if not np.array_equal(prepared.dynamics.q, selected_q):
        raise ValueError("prepared graph Q differs from the requested Q")
    if np.asarray((prepared.fixed_delay,), dtype="<f8").tobytes() != np.asarray(
        (delay,), dtype="<f8"
    ).tobytes():
        raise ValueError("prepared graph delay differs from the requested delay")
    if not np.array_equal(prepared.initial_parameter_coordinates, static):
        raise ValueError(
            "prepared graph static initialization differs from the request"
        )

    problem = build_fixed_batch_problem(prepared)
    initial = _warm_started_state(
        build_initial_batch_state(prepared), warm_start
    )
    try:
        lm = solve_batch_map(
            problem,
            initial,
            lm_settings,
            cancellation_requested=cancellation_requested,
            progress=lm_progress,
        )
    except BatchMapCancelled as error:
        raise EstimationCancelled(str(error)) from error
    except RecoverableModelEvaluationError as error:
        raise FixedGraphSolveFailure("lm_model_failure") from error
    except (ArithmeticError, ValueError, RuntimeError) as error:
        raise FixedGraphSolveFailure("lm_numerical_failure") from error
    if not lm.converged:
        raise FixedGraphSolveFailure(
            "lm_nonconverged:{}".format(lm.reason.value),
            len(lm.iterations),
        )
    try:
        final = problem.linearize(lm.state)
        factorization = ArrowheadLaplaceFactorization(final.sparse)
        dynamics_linearizations = evaluate_prepared_dynamics_intervals(
            prepared, lm.state
        )
        dynamics = compute_expected_dynamics_moments(
            dynamics_linearizations, factorization
        )
        marginal = approximate_marginal_objective(
            lm.objective,
            factorization,
            prepared.dynamics.q_definition,
            prepared.dynamics.q,
            dynamics.time_step,
        )
    except (ArithmeticError, ValueError, RuntimeError) as error:
        raise FixedGraphSolveFailure(
            "laplace_factorization_failure", len(lm.iterations)
        ) from error
    return FixedGraphLaplaceSolution(
        prepared=prepared,
        problem=problem,
        lm=lm,
        final_linearization=final,
        factorization=factorization,
        dynamics=dynamics,
        marginal_objective=marginal,
    )


class SparseLaplaceEStepSolver:
    """Adapter implementing the E-step protocol used by ``run_laplace_em``."""

    def __init__(
        self,
        graph_factory: PreparedGraphFactory,
        initial_static_coordinate: np.ndarray,
        lm_settings: LMSettings,
        wide_lag_settings: LagProfileSettings,
        *,
        cancellation_requested: Optional[CancellationCheck] = None,
        lm_progress: Optional[LMProgress] = None,
    ) -> None:
        if not callable(graph_factory):
            raise TypeError("graph_factory must be callable")
        if not isinstance(lm_settings, LMSettings):
            raise TypeError("lm_settings must be LMSettings")
        if not isinstance(wide_lag_settings, LagProfileSettings):
            raise TypeError("wide_lag_settings must be LagProfileSettings")
        if cancellation_requested is not None and not callable(
            cancellation_requested
        ):
            raise TypeError("cancellation_requested must be callable")
        if lm_progress is not None and not callable(lm_progress):
            raise TypeError("lm_progress must be callable")
        self.graph_factory = graph_factory
        self.initial_static_coordinate = _static_coordinate(
            initial_static_coordinate
        )
        self.lm_settings = lm_settings
        self.wide_lag_settings = wide_lag_settings
        self.cancellation_requested = cancellation_requested
        self.lm_progress = lm_progress
        self.profile_history = []  # type: list
        # A profile objective is comparable only with profiles evaluated at
        # the same Q.  Keep the exact Q beside every chronological profile so
        # downstream delay-curvature analysis cannot accidentally mix EM
        # iterations with different normalizing terms.
        self.profile_q_history = []  # type: list
        # E-step results intentionally expose only the immutable quantities
        # needed by EM.  Retain the corresponding full solve until the outer
        # driver selects its final step, so it can export that exact Laplace
        # point without running LM a second time and silently changing it.
        self._returned_solutions = {}  # type: dict

    def _remember_solution(
        self,
        result: LaplaceEStepResult,
        solution: FixedGraphLaplaceSolution,
    ) -> LaplaceEStepResult:
        self._returned_solutions[id(result)] = (result, solution)
        return result

    def take_solution_for_result(
        self, result: LaplaceEStepResult
    ) -> FixedGraphLaplaceSolution:
        """Return the exact solve behind an E-step and release the others."""

        if not isinstance(result, LaplaceEStepResult):
            raise TypeError("result must be LaplaceEStepResult")
        selected = self._returned_solutions.get(id(result))
        if selected is None or selected[0] is not result:
            raise ValueError("E-step result was not produced by this solver")
        solution = selected[1]
        self._returned_solutions.clear()
        return solution

    def _check_cancelled(self) -> None:
        if (
            self.cancellation_requested is not None
            and self.cancellation_requested()
        ):
            raise EstimationCancelled("estimation cancelled")

    def _solve(
        self,
        q: np.ndarray,
        lag: float,
        warm_start: Optional[BatchState],
    ) -> FixedGraphLaplaceSolution:
        self._check_cancelled()
        static = (
            self.initial_static_coordinate
            if warm_start is None
            else _state_static_coordinate(warm_start)
        )
        return solve_fixed_graph_laplace(
            self.graph_factory,
            q,
            lag,
            static,
            self.lm_settings,
            warm_start=warm_start,
            cancellation_requested=self.cancellation_requested,
            lm_progress=self.lm_progress,
        )

    def _local_lag_settings(self, center: float) -> LagProfileSettings:
        wide = self.wide_lag_settings
        coarse_spacing = (
            (wide.maximum_lag - wide.minimum_lag)
            / float(wide.coarse_grid_points - 1)
        )
        half_width = max(
            coarse_spacing, 2.0 * wide.refinement_tolerance
        )
        lower = max(wide.minimum_lag, float(center) - half_width)
        upper = min(wide.maximum_lag, float(center) + half_width)
        if upper - lower <= wide.refinement_tolerance:
            return wide
        return LagProfileSettings(
            minimum_lag=lower,
            maximum_lag=upper,
            coarse_grid_points=3,
            refinement_tolerance=wide.refinement_tolerance,
            maximum_refinement_evaluations=(
                wide.maximum_refinement_evaluations
            ),
        )

    def __call__(
        self,
        q: np.ndarray,
        phase: EStepPhase,
        lag: float,
        warm_start: Optional[BatchState],
    ) -> LaplaceEStepResult:
        if not isinstance(phase, EStepPhase):
            raise TypeError("phase must be EStepPhase")
        selected_q = _positive_q(q)
        if phase is EStepPhase.FIXED_LAG:
            try:
                solution = self._solve(selected_q, lag, warm_start)
                result = solution.as_e_step_result()
                return self._remember_solution(result, solution)
            except FixedGraphSolveFailure as error:
                raise LaplaceEStepFailure(
                    error.reason, error.inner_iterations
                ) from error

        settings = (
            self.wide_lag_settings
            if phase is EStepPhase.WIDE_LAG_PROFILE
            else self._local_lag_settings(lag)
        )
        solved = {}  # type: dict
        failure_iterations = {}  # type: dict

        def evaluator(
            candidate_lag: float,
            candidate_warm_start: Optional[BatchState],
        ) -> LagObjectiveResult:
            try:
                solution = self._solve(
                    selected_q, candidate_lag, candidate_warm_start
                )
            except FixedGraphSolveFailure as error:
                failure_iterations[float(candidate_lag)] = (
                    error.inner_iterations
                )
                return LagObjectiveResult(
                    objective=None,
                    converged=False,
                    state=None,
                    inner_iterations=error.inner_iterations,
                    termination_reason=error.reason,
                )
            solved[float(candidate_lag)] = solution
            return LagObjectiveResult(
                objective=solution.marginal_objective.value,
                converged=True,
                state=solution.lm.state,
                inner_iterations=len(solution.lm.iterations),
                termination_reason=solution.lm.reason.value,
            )

        try:
            profile = optimize_lag_profile(
                evaluator, settings, initial_warm_start=warm_start
            )
        except LagProfileFailure as error:
            total = sum(failure_iterations.values())
            raise LaplaceEStepFailure(
                "lag_profile_failure", total
            ) from error
        self.profile_history.append(profile)
        profile_q = selected_q.copy()
        profile_q.setflags(write=False)
        self.profile_q_history.append(profile_q)
        best = solved[profile.best_lag]
        total_iterations = sum(
            point.inner_iterations for point in profile.points
        )
        result = best.as_e_step_result(
            inner_iterations=total_iterations,
            termination_reason=phase.value,
        )
        return self._remember_solution(result, best)


def make_fixed_q_laplace_problem_factory(
    graph_factory: PreparedGraphFactory,
    q: np.ndarray,
) -> Callable[[PosteriorPoint], FixedDelayLaplaceProblem]:
    """Adapt the same graph factory to the exact conditional MCMC target."""

    if not callable(graph_factory):
        raise TypeError("graph_factory must be callable")
    selected_q = _positive_q(q)

    def factory(point: PosteriorPoint) -> FixedDelayLaplaceProblem:
        if not isinstance(point, PosteriorPoint):
            raise TypeError("point must be PosteriorPoint")
        prepared = graph_factory(
            selected_q.copy(),
            point.delay,
            point.static_coordinate.copy(),
        )
        if not isinstance(prepared, PreparedBatchGraphData):
            raise TypeError(
                "graph_factory must return PreparedBatchGraphData"
            )
        if not np.array_equal(prepared.dynamics.q, selected_q):
            raise ValueError("prepared MCMC graph changed fixed Q")
        initial = build_initial_batch_state(prepared)
        if not np.array_equal(
            _state_static_coordinate(initial), point.static_coordinate
        ):
            raise ValueError(
                "prepared MCMC graph changed the proposed static coordinate"
            )
        return FixedDelayLaplaceProblem(
            fixed_delay=prepared.fixed_delay,
            problem=build_fixed_batch_problem(prepared),
            initial_state=initial,
            graph_objective_includes_static_prior=True,
        )

    return factory


__all__ = [
    "CancellationCheck",
    "EstimationCancelled",
    "FixedGraphLaplaceSolution",
    "FixedGraphSolveFailure",
    "LMProgress",
    "PreparedGraphFactory",
    "SparseLaplaceEStepSolver",
    "make_fixed_q_laplace_problem_factory",
    "solve_fixed_graph_laplace",
]
