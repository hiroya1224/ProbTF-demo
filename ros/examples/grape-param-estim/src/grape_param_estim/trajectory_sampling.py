"""Bounded, auditable conditional trajectories for retained MCMC draws.

The MCMC chain state contains only static plant coordinates and delay.  This
module deliberately re-solves the conditional sparse MAP for a deterministic
subset after the chains have completed.  It never trusts or serializes a
``TargetEvaluation`` warm start retained by an in-memory sampler, because that
payload is absent after checkpoint/resume and is not an artifact contract.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.batch.dynamics_moments import (
    evaluate_prepared_dynamics_intervals,
)
from grape_param_estim.batch.graph_builder import build_initial_batch_state
from grape_param_estim.batch.lm import LMSettings
from grape_param_estim.batch.preparation import (
    PreparationSelection,
    prepare_fixed_batch_graph_data,
)
from grape_param_estim.batch_artifact_export import (
    CONDITIONAL_TRAJECTORY_EVALUATION_METHOD,
    CONDITIONAL_TRAJECTORY_SAMPLE_ORDER,
    CONDITIONAL_TRAJECTORY_SELECTION_POLICY,
    CONDITIONAL_TRAJECTORY_WARM_START_POLICY,
    SelectedConditionalTrajectory,
)
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.estimation import (
    FixedGraphLaplaceSolution,
    make_fixed_q_laplace_problem_factory,
)
from grape_param_estim.posterior.delayed_acceptance import PosteriorPoint
from grape_param_estim.posterior.laplace_target import (
    ConditionalTrajectoryWarmStart,
    LaplaceMarginalTarget,
)
from grape_param_estim.posterior.mcmc import McmcChainResult
from grape_param_estim.real_estimation import RealEstimationInputs


CONDITIONAL_TRAJECTORY_MAXIMUM_SAMPLE_COUNT = 8

CancellationCheck = Callable[[], bool]
TrajectoryProgress = Callable[[int, int, str], None]


def _canonical_positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError("{} must be a positive integer".format(name))
    return int(value)


@dataclass(frozen=True)
class ConditionalTrajectorySelection:
    """The exact deterministic subset saved in one completed run."""

    available_sample_count: int
    maximum_sample_count: int
    selected_sample_ids: Tuple[str, ...]
    selected_bag_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        available = _canonical_positive_integer(
            self.available_sample_count, "available_sample_count"
        )
        maximum = _canonical_positive_integer(
            self.maximum_sample_count, "maximum_sample_count"
        )
        if (
            type(self.selected_sample_ids) is not tuple
            or not self.selected_sample_ids
            or any(
                not isinstance(value, str)
                or not value
                or value.strip() != value
                for value in self.selected_sample_ids
            )
            or len(set(self.selected_sample_ids)) != len(self.selected_sample_ids)
        ):
            raise ValueError("selected_sample_ids must be unique canonical text")
        if len(self.selected_sample_ids) > min(available, maximum):
            raise ValueError("selected sample count exceeds its audited bound")
        if (
            type(self.selected_bag_ids) is not tuple
            or not self.selected_bag_ids
            or any(
                not isinstance(value, str)
                or not value
                or value.strip() != value
                for value in self.selected_bag_ids
            )
            or len(set(self.selected_bag_ids)) != len(self.selected_bag_ids)
        ):
            raise ValueError("selected_bag_ids must be unique canonical text")
        object.__setattr__(self, "available_sample_count", available)
        object.__setattr__(self, "maximum_sample_count", maximum)

    @property
    def manifest_payload(self) -> Mapping[str, object]:
        """Return the strict JSON-compatible policy audit."""

        return {
            "policy": CONDITIONAL_TRAJECTORY_SELECTION_POLICY,
            "sample_order": CONDITIONAL_TRAJECTORY_SAMPLE_ORDER,
            "available_sample_count": self.available_sample_count,
            "maximum_sample_count": self.maximum_sample_count,
            "selected_sample_ids": list(self.selected_sample_ids),
            "selected_bag_ids": list(self.selected_bag_ids),
            "conditional_evaluation_method": (
                CONDITIONAL_TRAJECTORY_EVALUATION_METHOD
            ),
            "warm_start_policy": CONDITIONAL_TRAJECTORY_WARM_START_POLICY,
        }


@dataclass(frozen=True)
class ConditionalTrajectorySamplingResult:
    """Fresh conditional MAP products and their selection provenance."""

    trajectories: Tuple[SelectedConditionalTrajectory, ...]
    selection: ConditionalTrajectorySelection

    def __post_init__(self) -> None:
        if (
            type(self.trajectories) is not tuple
            or not self.trajectories
            or any(
                not isinstance(value, SelectedConditionalTrajectory)
                for value in self.trajectories
            )
        ):
            raise ValueError(
                "trajectories must contain fresh selected conditional values"
            )
        if not isinstance(self.selection, ConditionalTrajectorySelection):
            raise TypeError("selection must be ConditionalTrajectorySelection")
        expected = {
            (sample_id, bag_id)
            for sample_id in self.selection.selected_sample_ids
            for bag_id in self.selection.selected_bag_ids
        }
        actual = {
            (value.sample_id, value.bag_id) for value in self.trajectories
        }
        if actual != expected or len(actual) != len(self.trajectories):
            raise ValueError(
                "conditional trajectories must cover every selected sample/bag pair"
            )


@dataclass(frozen=True)
class _RetainedDraw:
    sample_id: str
    point: PosteriorPoint
    graph_objective: float


def _flatten_retained_draws(
    chains: Sequence[McmcChainResult],
) -> Tuple[_RetainedDraw, ...]:
    if not isinstance(chains, (tuple, list)) or not chains:
        raise ValueError("chains must contain completed MCMC chains")
    if any(not isinstance(value, McmcChainResult) for value in chains):
        raise TypeError("chains must contain McmcChainResult values")
    result = []
    for chain in chains:
        if chain.graph_objective is None:
            raise ValueError(
                "conditional trajectory sampling requires graph objectives"
            )
        for index, sample_id in enumerate(chain.sample_id.tolist()):
            result.append(
                _RetainedDraw(
                    str(sample_id),
                    PosteriorPoint(
                        chain.static_coordinate[index], chain.delay[index]
                    ),
                    float(chain.graph_objective[index]),
                )
            )
    identifiers = tuple(value.sample_id for value in result)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("retained MCMC sample IDs must be globally unique")
    return tuple(result)


def select_conditional_trajectory_draws(
    chains: Sequence[McmcChainResult],
    maximum_sample_count: int = CONDITIONAL_TRAJECTORY_MAXIMUM_SAMPLE_COUNT,
) -> Tuple[_RetainedDraw, ...]:
    """Select bounded quantiles of canonical chain/draw order exactly."""

    flattened = _flatten_retained_draws(chains)
    maximum = _canonical_positive_integer(
        maximum_sample_count, "maximum_sample_count"
    )
    count = min(len(flattened), maximum)
    if count == len(flattened):
        indices = tuple(range(count))
    elif count == 1:
        indices = (0,)
    else:
        # Integer arithmetic defines the policy without platform-dependent
        # rounding.  With population >= count, these indices are unique and
        # include both endpoints of the flattened retained sequence.
        indices = tuple(
            index * (len(flattened) - 1) // (count - 1)
            for index in range(count)
        )
    if len(set(indices)) != count:  # pragma: no cover - arithmetic invariant
        raise RuntimeError("conditional trajectory subset indices are not unique")
    return tuple(flattened[index] for index in indices)


def _uniform_delay_log_prior(bounds: Tuple[float, float]):
    lower, upper = bounds
    log_density = -float(np.log(upper - lower))

    def evaluate(delay: float) -> float:
        value = float(delay)
        return log_density if lower <= value <= upper else float("-inf")

    return evaluate


def _prepared_graph_factory(inputs: RealEstimationInputs, mode_id: str):
    def factory(q, delay, static_coordinate):
        return prepare_fixed_batch_graph_data(
            request=inputs.request,
            flight_data=inputs.flight_data,
            initializations=inputs.initializations,
            parameter_chart=inputs.parameter_chart,
            geometry=inputs.geometry,
            actuator_parameters=inputs.actuator_parameters,
            scaling=inputs.scaling,
            selection=PreparationSelection(
                mode_id=mode_id,
                fixed_delay_seconds=delay,
                q_diagonal=q,
                initial_parameter_coordinates=static_coordinate,
            ),
        )

    return factory


def _dynamics_paths(prepared, state):
    collection = evaluate_prepared_dynamics_intervals(prepared, state)
    result = {}
    for bag in prepared.bags:
        interval_count = len(bag.knots) - 1
        result[bag.bag_id] = (
            np.zeros((interval_count, 6), dtype=float),
            np.zeros(interval_count, dtype=bool),
        )
    for interval in collection.intervals:
        residual, valid = result[interval.bag_id]
        residual[interval.left_knot_index] = interval.residual
        valid[interval.left_knot_index] = True
    for excluded in collection.excluded_intervals:
        _residual, valid = result[excluded.bag_id]
        if valid[excluded.left_knot_index]:
            raise RuntimeError("valid and excluded dynamics intervals overlap")
    return result


def sample_selected_conditional_trajectories(
    inputs: RealEstimationInputs,
    mode_id: str,
    final_solution: FixedGraphLaplaceSolution,
    chains: Sequence[McmcChainResult],
    *,
    maximum_sample_count: int = CONDITIONAL_TRAJECTORY_MAXIMUM_SAMPLE_COUNT,
    cancellation_requested: Optional[CancellationCheck] = None,
    progress: Optional[TrajectoryProgress] = None,
) -> ConditionalTrajectorySamplingResult:
    """Freshly solve and materialize only the deterministic retained subset."""

    if not isinstance(inputs, RealEstimationInputs):
        raise TypeError("inputs must be RealEstimationInputs")
    if not isinstance(mode_id, str) or not mode_id or mode_id.strip() != mode_id:
        raise ValueError("mode_id must be canonical non-empty text")
    if not isinstance(final_solution, FixedGraphLaplaceSolution):
        raise TypeError("final_solution must be FixedGraphLaplaceSolution")
    if cancellation_requested is not None and not callable(
        cancellation_requested
    ):
        raise TypeError("cancellation_requested must be callable")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")

    selected = select_conditional_trajectory_draws(
        chains, maximum_sample_count
    )
    selection = ConditionalTrajectorySelection(
        available_sample_count=sum(chain.sample_id.size for chain in chains),
        maximum_sample_count=maximum_sample_count,
        selected_sample_ids=tuple(value.sample_id for value in selected),
        selected_bag_ids=inputs.request.bag_ids,
    )
    total = len(selected)
    if progress is not None:
        progress(0, total, "selecting conditional trajectory subset")

    q = np.asarray(final_solution.prepared.dynamics.q, dtype=float)
    graph_factory = _prepared_graph_factory(inputs, mode_id)
    target_factory = make_fixed_q_laplace_problem_factory(graph_factory, q)
    delay_bounds = tuple(
        float(value)
        for value in inputs.request.payload["delay"]["bounds_seconds"]
    )
    lm_settings = LMSettings(
        **dict(inputs.request.payload["solver_settings"])
    )
    target = LaplaceMarginalTarget(
        target_factory,
        _uniform_delay_log_prior(delay_bounds),
        lm_settings,
    )
    map_static = final_solution.lm.state.value(
        VariableKey(VariableKind.STATIC_PARAMETERS)
    )
    map_point = PosteriorPoint(
        map_static, final_solution.prepared.fixed_delay
    )
    deterministic_warm_start = ConditionalTrajectoryWarmStart(
        final_solution.lm.state, map_point.exact_cache_key
    )
    objective_relative_tolerance = max(
        2.0e-8, 5.0 * lm_settings.relative_objective_tolerance
    )

    trajectories = []
    for index, draw in enumerate(selected):
        if cancellation_requested is not None and cancellation_requested():
            raise RuntimeError("conditional_trajectory_sampling_cancelled")
        # The selected-mode MAP state is checkpointed and reproducible.  It is
        # rebased onto every proposed shared coordinate; an ephemeral retained
        # TargetEvaluation warm start is never used.
        evaluation = target.evaluate(draw.point, deterministic_warm_start)
        if not evaluation.successful:
            raise RuntimeError(
                "fresh conditional target failed for {}: {}".format(
                    draw.sample_id, evaluation.failure_reason
                )
            )
        if not isinstance(
            evaluation.warm_start, ConditionalTrajectoryWarmStart
        ):
            raise RuntimeError(
                "fresh conditional target omitted its audited state"
            )
        if evaluation.graph_objective is None or not np.isclose(
            evaluation.graph_objective,
            draw.graph_objective,
            rtol=objective_relative_tolerance,
            atol=(
                objective_relative_tolerance
                * max(1.0, abs(draw.graph_objective))
            ),
        ):
            raise RuntimeError(
                "fresh conditional objective disagrees with retained sample {}"
                .format(draw.sample_id)
            )
        prepared = graph_factory(
            q.copy(), draw.point.delay, draw.point.static_coordinate.copy()
        )
        state = evaluation.warm_start.state
        expected_layout = build_initial_batch_state(prepared).layout
        if state.layout != expected_layout:
            raise RuntimeError(
                "fresh conditional state layout disagrees with prepared graph"
            )
        paths = _dynamics_paths(prepared, state)
        for bag_id in selection.selected_bag_ids:
            residual, valid = paths[bag_id]
            trajectories.append(
                SelectedConditionalTrajectory(
                    sample_id=draw.sample_id,
                    bag_id=bag_id,
                    state=state,
                    dynamics_residual=residual,
                    dynamics_residual_valid=valid,
                    conditional_objective=float(evaluation.graph_objective),
                )
            )
        if cancellation_requested is not None and cancellation_requested():
            raise RuntimeError("conditional_trajectory_sampling_cancelled")
        if progress is not None:
            progress(
                index + 1,
                total,
                "materialized conditional trajectory {}".format(draw.sample_id),
            )
    return ConditionalTrajectorySamplingResult(tuple(trajectories), selection)


__all__ = [
    "CONDITIONAL_TRAJECTORY_EVALUATION_METHOD",
    "CONDITIONAL_TRAJECTORY_MAXIMUM_SAMPLE_COUNT",
    "CONDITIONAL_TRAJECTORY_SAMPLE_ORDER",
    "CONDITIONAL_TRAJECTORY_SELECTION_POLICY",
    "CONDITIONAL_TRAJECTORY_WARM_START_POLICY",
    "ConditionalTrajectorySamplingResult",
    "ConditionalTrajectorySelection",
    "sample_selected_conditional_trajectories",
    "select_conditional_trajectory_draws",
]
