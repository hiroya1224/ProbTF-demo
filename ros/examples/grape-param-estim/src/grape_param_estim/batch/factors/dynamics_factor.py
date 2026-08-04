"""Explicit statistical coordinates and Q whitening for dynamics factors."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

import numpy as np

from grape_param_estim.batch.factor import FactorEvaluation, JacobianBlock
from grape_param_estim.batch.factors.dynamics import (
    DynamicsResidualEvaluation,
    DynamicsResidualJacobian,
)
from grape_param_estim.batch.laplace_em import (
    BODY_WRENCH_COMPONENT_NAMES,
    BODY_WRENCH_COMPONENT_UNITS,
    BODY_WRENCH_QUANTITY,
    DiagonalQDefinition,
)
from grape_param_estim.batch.variables import VariableKey, VariableKind


_JACOBIAN_FIELDS = (
    ("static_parameters", VariableKind.STATIC_PARAMETERS, 0),
    ("rotation_left", VariableKind.ORIENTATION_TANGENT, 0),
    ("rotation_right", VariableKind.ORIENTATION_TANGENT, 1),
    ("linear_velocity_left", VariableKind.LINEAR_VELOCITY, 0),
    ("linear_velocity_right", VariableKind.LINEAR_VELOCITY, 1),
    ("angular_velocity_left", VariableKind.ANGULAR_VELOCITY, 0),
    ("angular_velocity_right", VariableKind.ANGULAR_VELOCITY, 1),
    ("actuator_thrust_left", VariableKind.ACTUATOR_THRUST, 0),
    ("actuator_thrust_right", VariableKind.ACTUATOR_THRUST, 1),
    ("gimbal_angle_left", VariableKind.GIMBAL_ANGLE, 0),
    ("gimbal_angle_right", VariableKind.GIMBAL_ANGLE, 1),
)


def _immutable_six(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError("{} must contain six finite values".format(name))
    result = result.copy()
    result.setflags(write=False)
    return result


def _validate_identity(definition: DiagonalQDefinition) -> None:
    if not isinstance(definition, DiagonalQDefinition):
        raise TypeError("definition must be DiagonalQDefinition")
    if definition.residual_quantity != BODY_WRENCH_QUANTITY:
        raise ValueError(
            "Q definition residual_quantity must be 'body_wrench'"
        )


@dataclass(frozen=True)
class StatisticalDynamicsResidual:
    """One tagged six-vector in the exact coordinate system owned by Q."""

    bag_id: str
    left_knot_index: int
    definition: DiagonalQDefinition
    residual: np.ndarray
    jacobian: DynamicsResidualJacobian
    raw_body_wrench_residual: np.ndarray
    branch_diagnostics: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.bag_id, str)
            or not self.bag_id
            or self.bag_id.strip() != self.bag_id
        ):
            raise ValueError("bag_id must be a canonical non-empty string")
        if (
            isinstance(self.left_knot_index, (bool, np.bool_))
            or not isinstance(self.left_knot_index, (int, np.integer))
            or self.left_knot_index < 0
        ):
            raise ValueError("left_knot_index must be a non-negative integer")
        if not isinstance(self.definition, DiagonalQDefinition):
            raise TypeError("definition must be DiagonalQDefinition")
        if not isinstance(self.jacobian, DynamicsResidualJacobian):
            raise TypeError("jacobian must be DynamicsResidualJacobian")
        object.__setattr__(self, "left_knot_index", int(self.left_knot_index))
        object.__setattr__(
            self, "residual", _immutable_six(self.residual, "residual")
        )
        object.__setattr__(
            self,
            "raw_body_wrench_residual",
            _immutable_six(
                self.raw_body_wrench_residual,
                "raw_body_wrench_residual",
            ),
        )
        diagnostics = {}
        for name, value in self.branch_diagnostics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("branch diagnostic names must be non-empty")
            mask = np.asarray(value)
            if mask.ndim != 1 or mask.size == 0 or mask.dtype != np.bool_:
                raise ValueError("branch diagnostics must be boolean vectors")
            mask = mask.copy()
            mask.setflags(write=False)
            diagnostics[name] = mask
        object.__setattr__(
            self, "branch_diagnostics", MappingProxyType(diagnostics)
        )

    @property
    def jacobian_blocks(self) -> Tuple[JacobianBlock, ...]:
        """Return unwhitened blocks for selected Laplace covariance queries."""

        result = []
        for field, kind, knot_offset in _JACOBIAN_FIELDS:
            if kind is VariableKind.STATIC_PARAMETERS:
                key = VariableKey(kind)
            else:
                key = VariableKey(
                    kind,
                    bag_id=self.bag_id,
                    knot_index=self.left_knot_index + knot_offset,
                )
            result.append(JacobianBlock(key, getattr(self.jacobian, field)))
        return tuple(result)


def body_wrench_statistical_residual(
    bag_id: str,
    left_knot_index: int,
    raw_evaluation: DynamicsResidualEvaluation,
    definition: DiagonalQDefinition,
) -> StatisticalDynamicsResidual:
    """Tag the raw N/Nm balance without changing its coordinate system."""

    if not isinstance(raw_evaluation, DynamicsResidualEvaluation):
        raise TypeError("raw_evaluation must be DynamicsResidualEvaluation")
    _validate_identity(definition)
    return StatisticalDynamicsResidual(
        bag_id=bag_id,
        left_knot_index=left_knot_index,
        definition=definition,
        residual=raw_evaluation.residual,
        jacobian=raw_evaluation.jacobian,
        raw_body_wrench_residual=raw_evaluation.residual,
        branch_diagnostics=raw_evaluation.branch_diagnostics,
    )

def _positive_q(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (6,)
        or not np.all(np.isfinite(result))
        or np.any(result <= 0.0)
    ):
        raise ValueError("q must contain six positive finite values")
    return result


def dynamics_square_root_information(
    q: np.ndarray,
    time_step: float,
    definition: DiagonalQDefinition,
) -> np.ndarray:
    """Return ``diag(sqrt(dt / Q))`` for body-wrench spectral density."""

    if not isinstance(definition, DiagonalQDefinition):
        raise TypeError("definition must be DiagonalQDefinition")
    selected_q = _positive_q(q)
    weight = definition.interval_weights(
        np.asarray((float(time_step),), dtype=float)
    )[0]
    return np.diag(np.sqrt(weight / selected_q))


def evaluate_dynamics_factor(
    statistical_residual: StatisticalDynamicsResidual,
    q: np.ndarray,
    time_step: float,
) -> FactorEvaluation:
    """Whiten one explicitly tagged residual for sparse GN assembly."""

    if not isinstance(statistical_residual, StatisticalDynamicsResidual):
        raise TypeError(
            "statistical_residual must be StatisticalDynamicsResidual"
        )
    whitening = dynamics_square_root_information(
        q,
        time_step,
        statistical_residual.definition,
    )
    residual = whitening @ statistical_residual.residual
    blocks = tuple(
        JacobianBlock(block.variable_key, whitening @ block.value)
        for block in statistical_residual.jacobian_blocks
    )
    return FactorEvaluation(
        residual=residual,
        jacobian_blocks=blocks,
        squared_error=float(residual @ residual),
        active_set=statistical_residual.branch_diagnostics,
    )


def diagonal_q_log_normalization(
    q: np.ndarray,
    time_step: np.ndarray,
    definition: DiagonalQDefinition,
) -> float:
    """Return the Gaussian negative-log normalization across intervals."""

    if not isinstance(definition, DiagonalQDefinition):
        raise TypeError("definition must be DiagonalQDefinition")
    selected_q = _positive_q(q)
    time_steps = np.asarray(time_step, dtype=float)
    weights = definition.interval_weights(time_steps)
    log_determinant = np.sum(
        np.log(selected_q)[None, :] - np.log(weights)[:, None]
    )
    return float(
        0.5
        * (
            time_steps.size * 6 * np.log(2.0 * np.pi)
            + log_determinant
        )
    )


__all__ = [
    "BODY_WRENCH_COMPONENT_NAMES",
    "BODY_WRENCH_COMPONENT_UNITS",
    "BODY_WRENCH_QUANTITY",
    "StatisticalDynamicsResidual",
    "body_wrench_statistical_residual",
    "diagonal_q_log_normalization",
    "dynamics_square_root_information",
    "evaluate_dynamics_factor",
]
