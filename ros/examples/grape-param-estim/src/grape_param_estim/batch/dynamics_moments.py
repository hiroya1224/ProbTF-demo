"""Selected Laplace moments for unwhitened dynamics residuals.

This module is the single bridge between the fixed prepared graph, the
analytic six-dimensional dynamics residuals, and the diagonal-Q Laplace
E-step.  It deliberately keeps the residual and Jacobian in Q's unwhitened
statistical coordinates.  Covariance queries use only six right-hand sides
per valid interval; a full trajectory covariance or dense Hessian inverse is
never formed.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from grape_param_estim.batch.covariance import ArrowheadLaplaceFactorization
from grape_param_estim.batch.factor import JacobianBlock
from grape_param_estim.batch.factors.dynamics import (
    evaluate_raw_dynamics_residual,
)
from grape_param_estim.batch.factors.dynamics_factor import (
    BODY_WRENCH_QUANTITY,
    SPECIFIC_ACCELERATION_QUANTITY,
    StatisticalDynamicsResidual,
    body_wrench_statistical_residual,
    specific_acceleration_statistical_residual,
)
from grape_param_estim.batch.laplace_em import (
    DiagonalQDefinition,
    ExpectedResidualMoments,
)
from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.state import BatchState
from grape_param_estim.batch.variables import (
    VariableKey,
    VariableKind,
    VariableScope,
)


def _canonical_bag_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
    ):
        raise ValueError("bag_id must be a canonical non-empty string")
    return value


def _left_knot_index(value: object) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError("left_knot_index must be a non-negative integer")
    return int(value)


def _positive_time_step(value: object) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("time_step must be finite and positive")
    return result


@dataclass(frozen=True)
class DynamicsIntervalExclusion:
    """One audited interval omitted from both likelihood and Q moments."""

    bag_id: str
    left_knot_index: int
    time_step: float
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bag_id", _canonical_bag_id(self.bag_id))
        object.__setattr__(
            self,
            "left_knot_index",
            _left_knot_index(self.left_knot_index),
        )
        object.__setattr__(
            self, "time_step", _positive_time_step(self.time_step)
        )
        if (
            not isinstance(self.reason, str)
            or not self.reason
            or self.reason.strip() != self.reason
        ):
            raise ValueError("excluded interval requires a canonical reason")

    @property
    def key(self) -> Tuple[str, int]:
        return (self.bag_id, self.left_knot_index)


@dataclass(frozen=True)
class DynamicsIntervalLinearization:
    """Unwhitened analytic ``e_k`` and ``G_k`` for one valid interval."""

    bag_id: str
    left_knot_index: int
    time_step: float
    statistical_residual: StatisticalDynamicsResidual

    def __post_init__(self) -> None:
        bag_id = _canonical_bag_id(self.bag_id)
        left_knot_index = _left_knot_index(self.left_knot_index)
        object.__setattr__(self, "bag_id", bag_id)
        object.__setattr__(self, "left_knot_index", left_knot_index)
        object.__setattr__(
            self, "time_step", _positive_time_step(self.time_step)
        )
        if not isinstance(
            self.statistical_residual, StatisticalDynamicsResidual
        ):
            raise TypeError(
                "statistical_residual must be StatisticalDynamicsResidual"
            )
        if (
            self.statistical_residual.bag_id != bag_id
            or self.statistical_residual.left_knot_index != left_knot_index
        ):
            raise ValueError(
                "statistical residual identity disagrees with its interval"
            )
        _validate_local_blocks(self.statistical_residual, bag_id)

    @property
    def key(self) -> Tuple[str, int]:
        return (self.bag_id, self.left_knot_index)

    @property
    def residual(self) -> np.ndarray:
        """Return the immutable unwhitened statistical residual ``e_k``."""

        return self.statistical_residual.residual

    @property
    def jacobian_blocks(self) -> Tuple[JacobianBlock, ...]:
        """Return analytic unwhitened Jacobian blocks making up ``G_k``."""

        return self.statistical_residual.jacobian_blocks


def _validate_local_blocks(
    residual: StatisticalDynamicsResidual,
    bag_id: str,
) -> None:
    local_bag_ids = {
        block.variable_key.bag_id
        for block in residual.jacobian_blocks
        if block.variable_key.scope is not VariableScope.SHARED
    }
    if local_bag_ids != {bag_id}:
        raise ValueError(
            "a dynamics interval cannot reference another bag's variables"
        )


@dataclass(frozen=True)
class DynamicsLinearizationCollection:
    """All valid and explicitly excluded intervals of one fixed graph."""

    layout: VariableLayout
    definition: DiagonalQDefinition
    intervals: Tuple[DynamicsIntervalLinearization, ...]
    excluded_intervals: Tuple[DynamicsIntervalExclusion, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layout, VariableLayout):
            raise TypeError("layout must be a VariableLayout")
        if not isinstance(self.definition, DiagonalQDefinition):
            raise TypeError("definition must be DiagonalQDefinition")
        if type(self.intervals) is not tuple or any(
            not isinstance(item, DynamicsIntervalLinearization)
            for item in self.intervals
        ):
            raise TypeError(
                "intervals must be a tuple of DynamicsIntervalLinearization"
            )
        if type(self.excluded_intervals) is not tuple or any(
            not isinstance(item, DynamicsIntervalExclusion)
            for item in self.excluded_intervals
        ):
            raise TypeError(
                "excluded_intervals must be a tuple of "
                "DynamicsIntervalExclusion"
            )
        ordered = tuple(item.key for item in self.intervals)
        excluded = tuple(item.key for item in self.excluded_intervals)
        if ordered != tuple(sorted(ordered)) or excluded != tuple(
            sorted(excluded)
        ):
            raise ValueError("dynamics intervals must use canonical order")
        all_keys = ordered + excluded
        if not all_keys:
            raise ValueError(
                "dynamics collection must audit at least one interval"
            )
        if len(set(all_keys)) != len(all_keys):
            raise ValueError("valid and excluded dynamics intervals overlap")
        for bag_id in {key[0] for key in all_keys}:
            indices = sorted(
                key[1] for key in all_keys if key[0] == bag_id
            )
            if indices != list(range(indices[-1] + 1)):
                raise ValueError(
                    "dynamics interval identities must be contiguous"
                )
        layout_bags = set(self.layout.bag_ids)
        for item in self.intervals:
            if item.statistical_residual.definition != self.definition:
                raise ValueError(
                    "all dynamics intervals must use the collection Q definition"
                )
            if item.bag_id not in layout_bags:
                raise ValueError("dynamics interval bag is absent from layout")
            for block in item.jacobian_blocks:
                if block.variable_key not in self.layout:
                    raise ValueError(
                        "dynamics Jacobian key is absent from collection layout"
                    )
        for item in self.excluded_intervals:
            if item.bag_id not in layout_bags:
                raise ValueError(
                    "excluded dynamics interval bag is absent from layout"
                )

    @property
    def valid_interval_count(self) -> int:
        return len(self.intervals)

    @property
    def excluded_interval_count(self) -> int:
        return len(self.excluded_intervals)

    @property
    def time_step(self) -> np.ndarray:
        result = np.asarray(
            tuple(item.time_step for item in self.intervals), dtype=float
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class DynamicsLaplaceMoments:
    """Laplace E-step result retaining interval and exclusion provenance."""

    linearizations: DynamicsLinearizationCollection
    moments: ExpectedResidualMoments

    def __post_init__(self) -> None:
        if not isinstance(
            self.linearizations, DynamicsLinearizationCollection
        ):
            raise TypeError(
                "linearizations must be DynamicsLinearizationCollection"
            )
        if not isinstance(self.moments, ExpectedResidualMoments):
            raise TypeError("moments must be ExpectedResidualMoments")
        if (
            self.moments.interval_count
            != self.linearizations.valid_interval_count
        ):
            raise ValueError("moment count must match valid interval count")

    @property
    def definition(self) -> DiagonalQDefinition:
        return self.linearizations.definition

    @property
    def time_step(self) -> np.ndarray:
        return self.linearizations.time_step

    @property
    def excluded_intervals(self) -> Tuple[DynamicsIntervalExclusion, ...]:
        return self.linearizations.excluded_intervals


def evaluate_prepared_dynamics_intervals(
    prepared: object,
    state: BatchState,
) -> DynamicsLinearizationCollection:
    """Evaluate valid fixed-graph intervals without Q whitening.

    Invalid intervals are returned as exclusions and are never evaluated.
    The prepared validity status is therefore shared by the dynamics
    likelihood and the Laplace-EM moment calculation.
    """

    # Local imports avoid a module cycle: graph_builder itself calls this
    # function when it appends its Q-weighted dynamics factors.
    from grape_param_estim.batch.graph_builder import (  # pylint: disable=C0415
        PreparedBatchGraphData,
        _layout,
    )

    if not isinstance(prepared, PreparedBatchGraphData):
        raise TypeError("prepared must be PreparedBatchGraphData")
    if not isinstance(state, BatchState):
        raise TypeError("state must be BatchState")
    expected_layout = _layout(prepared)
    if state.layout != expected_layout:
        raise ValueError("state layout does not match prepared graph")

    static_key = VariableKey(VariableKind.STATIC_PARAMETERS)
    coordinates = state.value(static_key)
    valid = []
    excluded = []
    for bag in sorted(prepared.bags, key=lambda item: item.bag_id):
        for status in bag.dynamics_interval_statuses:
            index = status.left_knot_index
            dt = bag.knots[index + 1].time - bag.knots[index].time
            if not status.valid:
                excluded.append(
                    DynamicsIntervalExclusion(
                        bag_id=bag.bag_id,
                        left_knot_index=index,
                        time_step=dt,
                        reason=status.invalid_reason,
                    )
                )
                continue
            raw = evaluate_raw_dynamics_residual(
                rotation_left=state.knot_value(
                    bag.bag_id, index, VariableKind.ORIENTATION_TANGENT
                ),
                rotation_right=state.knot_value(
                    bag.bag_id, index + 1, VariableKind.ORIENTATION_TANGENT
                ),
                linear_velocity_left=state.knot_value(
                    bag.bag_id, index, VariableKind.LINEAR_VELOCITY
                ),
                linear_velocity_right=state.knot_value(
                    bag.bag_id, index + 1, VariableKind.LINEAR_VELOCITY
                ),
                angular_velocity_left=state.knot_value(
                    bag.bag_id, index, VariableKind.ANGULAR_VELOCITY
                ),
                angular_velocity_right=state.knot_value(
                    bag.bag_id, index + 1, VariableKind.ANGULAR_VELOCITY
                ),
                actuator_thrust_left=state.knot_value(
                    bag.bag_id, index, VariableKind.ACTUATOR_THRUST
                ),
                actuator_thrust_right=state.knot_value(
                    bag.bag_id, index + 1, VariableKind.ACTUATOR_THRUST
                ),
                gimbal_angle_left=state.knot_value(
                    bag.bag_id, index, VariableKind.GIMBAL_ANGLE
                ),
                gimbal_angle_right=state.knot_value(
                    bag.bag_id, index + 1, VariableKind.GIMBAL_ANGLE
                ),
                time_step=dt,
                parameter_chart=prepared.parameter_chart,
                parameter_coordinates=coordinates,
                geometry=prepared.geometry,
                gravity_world=prepared.dynamics.gravity_world,
            )
            quantity = prepared.dynamics.q_definition.residual_quantity
            if quantity == BODY_WRENCH_QUANTITY:
                statistical = body_wrench_statistical_residual(
                    bag.bag_id,
                    index,
                    raw,
                    prepared.dynamics.q_definition,
                )
            elif quantity == SPECIFIC_ACCELERATION_QUANTITY:
                statistical = specific_acceleration_statistical_residual(
                    bag.bag_id,
                    index,
                    raw,
                    prepared.dynamics.q_definition,
                    prepared.parameter_chart,
                    coordinates,
                )
            else:
                raise ValueError("unsupported dynamics residual quantity")
            valid.append(
                DynamicsIntervalLinearization(
                    bag_id=bag.bag_id,
                    left_knot_index=index,
                    time_step=dt,
                    statistical_residual=statistical,
                )
            )
    return DynamicsLinearizationCollection(
        layout=state.layout,
        definition=prepared.dynamics.q_definition,
        intervals=tuple(valid),
        excluded_intervals=tuple(excluded),
    )


def compute_expected_dynamics_moments(
    linearizations: DynamicsLinearizationCollection,
    factorization: ArrowheadLaplaceFactorization,
) -> DynamicsLaplaceMoments:
    """Compute ``e_k^2 + diag(G_k H^-1 G_k.T)`` interval by interval."""

    if not isinstance(linearizations, DynamicsLinearizationCollection):
        raise TypeError(
            "linearizations must be DynamicsLinearizationCollection"
        )
    if not isinstance(factorization, ArrowheadLaplaceFactorization):
        raise TypeError(
            "factorization must be ArrowheadLaplaceFactorization"
        )
    if factorization.layout != linearizations.layout:
        raise ValueError(
            "Laplace factorization layout does not match dynamics layout"
        )
    if not linearizations.intervals:
        raise ValueError(
            "Laplace-EM needs at least one valid dynamics interval"
        )

    residual_rows = []
    correction_rows = []
    for interval in linearizations.intervals:
        covariance = factorization.residual_covariance(
            interval.jacobian_blocks
        )
        if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
            raise np.linalg.LinAlgError(
                "selected dynamics residual covariance must be finite 6 by 6"
            )
        diagonal = np.diag(covariance).copy()
        tolerance = 1.0e-12 * max(
            1.0, float(np.max(np.abs(diagonal)))
        )
        if np.any(diagonal < -tolerance):
            raise np.linalg.LinAlgError(
                "selected dynamics covariance has a negative diagonal"
            )
        residual_rows.append(interval.residual)
        correction_rows.append(np.maximum(diagonal, 0.0))

    return DynamicsLaplaceMoments(
        linearizations=linearizations,
        moments=ExpectedResidualMoments(
            map_residual=np.vstack(residual_rows),
            covariance_correction=np.vstack(correction_rows),
        ),
    )


__all__ = [
    "DynamicsIntervalExclusion",
    "DynamicsIntervalLinearization",
    "DynamicsLaplaceMoments",
    "DynamicsLinearizationCollection",
    "compute_expected_dynamics_moments",
    "evaluate_prepared_dynamics_intervals",
]
