"""Canonical variable-block keys for sparse batch estimation.

The key space is deliberately closed: one shared 18-dimensional parameter
block, two bag-local bias blocks, and seven fields making up each 26-
dimensional knot state.  Solvers may assign storage offsets separately; a key
only describes scientific identity and canonical block dimension.
"""

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Optional


class VariableScope(Enum):
    """Lifetime of one variable block in a batch problem."""

    SHARED = "shared"
    BAG = "bag"
    KNOT = "knot"


class VariableKind(Enum):
    """Closed set of variable-block kinds in the planned batch state."""

    STATIC_PARAMETERS = "static_parameters"
    GYRO_BIAS = "gyro_bias"
    ACCELEROMETER_BIAS = "accelerometer_bias"
    POSITION = "position"
    ORIENTATION_TANGENT = "orientation_tangent"
    LINEAR_VELOCITY = "linear_velocity"
    ANGULAR_VELOCITY = "angular_velocity"
    CONTROLLER_INTEGRAL = "controller_integral"
    ACTUATOR_THRUST = "actuator_thrust"
    GIMBAL_ANGLE = "gimbal_angle"


_VARIABLE_SPECS = {
    VariableKind.STATIC_PARAMETERS: (VariableScope.SHARED, 18),
    VariableKind.GYRO_BIAS: (VariableScope.BAG, 3),
    VariableKind.ACCELEROMETER_BIAS: (VariableScope.BAG, 3),
    VariableKind.POSITION: (VariableScope.KNOT, 3),
    VariableKind.ORIENTATION_TANGENT: (VariableScope.KNOT, 3),
    VariableKind.LINEAR_VELOCITY: (VariableScope.KNOT, 3),
    VariableKind.ANGULAR_VELOCITY: (VariableScope.KNOT, 3),
    VariableKind.CONTROLLER_INTEGRAL: (VariableScope.KNOT, 6),
    VariableKind.ACTUATOR_THRUST: (VariableScope.KNOT, 4),
    VariableKind.GIMBAL_ANGLE: (VariableScope.KNOT, 4),
}


@dataclass(frozen=True)
class VariableKey:
    """Immutable identity of one canonical variable block.

    Shared keys have neither ``bag_id`` nor ``knot_index``.  Bag-local keys
    require only ``bag_id``.  Knot keys require both a non-empty canonical bag
    identifier and a non-negative integer knot index.
    """

    kind: VariableKind
    bag_id: Optional[str] = None
    knot_index: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, VariableKind):
            raise TypeError("kind must be a VariableKind")

        scope = self.scope
        if scope is VariableScope.SHARED:
            if self.bag_id is not None or self.knot_index is not None:
                raise ValueError(
                    "shared variable keys cannot have bag_id or knot_index"
                )
            return

        if (
            type(self.bag_id) is not str
            or not self.bag_id
            or self.bag_id.strip() != self.bag_id
        ):
            raise ValueError(
                "bag-local and knot variable keys require a canonical bag_id"
            )

        if scope is VariableScope.BAG:
            if self.knot_index is not None:
                raise ValueError("bag-local variable keys cannot have knot_index")
            return

        if (
            isinstance(self.knot_index, bool)
            or not isinstance(self.knot_index, Integral)
            or self.knot_index < 0
        ):
            raise ValueError(
                "knot variable keys require a non-negative integer knot_index"
            )
        object.__setattr__(self, "knot_index", int(self.knot_index))

    @property
    def scope(self) -> VariableScope:
        """Return the canonical scope of this block kind."""

        return _VARIABLE_SPECS[self.kind][0]

    @property
    def dimension(self) -> int:
        """Return the canonical local dimension of this variable block."""

        return _VARIABLE_SPECS[self.kind][1]


__all__ = ["VariableKey", "VariableKind", "VariableScope"]
