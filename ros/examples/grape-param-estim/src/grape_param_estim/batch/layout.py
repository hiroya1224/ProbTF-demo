"""Deterministic column layout for canonical sparse batch variables."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Tuple

from grape_param_estim.batch.variables import (
    VariableKey,
    VariableKind,
    VariableScope,
)


_BAG_KIND_ORDER = (
    VariableKind.GYRO_BIAS,
    VariableKind.ACCELEROMETER_BIAS,
)
_KNOT_KIND_ORDER = (
    VariableKind.POSITION,
    VariableKind.ORIENTATION_TANGENT,
    VariableKind.LINEAR_VELOCITY,
    VariableKind.ANGULAR_VELOCITY,
    VariableKind.CONTROLLER_INTEGRAL,
    VariableKind.ACTUATOR_THRUST,
    VariableKind.GIMBAL_ANGLE,
)


@dataclass(frozen=True)
class VariableLayout:
    """Canonical immutable ordering and offsets for one batch problem.

    Ordering is independent of input order: the shared static block is first,
    followed by lexicographically ordered bags.  Within each bag, present bias
    blocks use gyro/accelerometer order, then knots use increasing index and
    the planned 26-D state-field order.

    Every knot must contain all seven state fields.  Bias blocks are optional
    independently because the corresponding sensor can be disabled.
    """

    variable_keys: Tuple[VariableKey, ...]
    _column_slices: Mapping[VariableKey, slice] = field(
        init=False,
        repr=False,
        compare=False,
    )
    total_dimension: int = field(init=False)

    def __post_init__(self) -> None:
        try:
            supplied_keys = tuple(self.variable_keys)
        except TypeError as error:
            raise TypeError(
                "variable_keys must be an iterable of VariableKey"
            ) from error
        if not supplied_keys:
            raise ValueError("variable_keys cannot be empty")
        if any(not isinstance(key, VariableKey) for key in supplied_keys):
            raise TypeError("variable_keys must contain only VariableKey values")
        if len(set(supplied_keys)) != len(supplied_keys):
            raise ValueError("variable_keys must be unique")

        static_key = VariableKey(VariableKind.STATIC_PARAMETERS)
        if static_key not in supplied_keys:
            raise ValueError("variable_keys must include the shared static block")

        bag_ids = sorted(
            {
                key.bag_id
                for key in supplied_keys
                if key.scope is not VariableScope.SHARED
            }
        )
        if not bag_ids:
            raise ValueError("variable_keys must include at least one bag")

        supplied_set = set(supplied_keys)
        canonical_keys = [static_key]
        expected_set = {static_key}
        for bag_id in bag_ids:
            knot_indices = sorted(
                {
                    key.knot_index
                    for key in supplied_keys
                    if key.scope is VariableScope.KNOT
                    and key.bag_id == bag_id
                }
            )
            if not knot_indices:
                raise ValueError(
                    "every bag in variable_keys must contain knot variables"
                )
            if knot_indices != list(range(knot_indices[-1] + 1)):
                raise ValueError(
                    "knot indices must be contiguous and start at zero"
                )

            for kind in _BAG_KIND_ORDER:
                key = VariableKey(kind, bag_id=bag_id)
                if key in supplied_set:
                    canonical_keys.append(key)
                    expected_set.add(key)
            for knot_index in knot_indices:
                for kind in _KNOT_KIND_ORDER:
                    key = VariableKey(
                        kind,
                        bag_id=bag_id,
                        knot_index=knot_index,
                    )
                    canonical_keys.append(key)
                    expected_set.add(key)

        missing_keys = expected_set - supplied_set
        unexpected_keys = supplied_set - expected_set
        if missing_keys:
            raise ValueError(
                "variable_keys are missing canonical knot fields: {}".format(
                    _format_keys(missing_keys)
                )
            )
        if unexpected_keys:
            raise ValueError(
                "variable_keys contain keys outside the canonical layout: {}"
                .format(_format_keys(unexpected_keys))
            )

        offsets = {}
        offset = 0
        for key in canonical_keys:
            next_offset = offset + key.dimension
            offsets[key] = slice(offset, next_offset)
            offset = next_offset
        object.__setattr__(self, "variable_keys", tuple(canonical_keys))
        object.__setattr__(self, "_column_slices", MappingProxyType(offsets))
        object.__setattr__(self, "total_dimension", offset)

    def column_slice(self, variable_key: VariableKey) -> slice:
        """Return the deterministic column slice for a known key."""

        if not isinstance(variable_key, VariableKey):
            raise TypeError("variable_key must be a VariableKey")
        try:
            return self._column_slices[variable_key]
        except KeyError as error:
            raise KeyError("variable_key is not present in this layout") from error

    def column_offset(self, variable_key: VariableKey) -> int:
        """Return the first deterministic column for a known key."""

        return self.column_slice(variable_key).start

    def __contains__(self, variable_key: object) -> bool:
        return variable_key in self._column_slices

    def __len__(self) -> int:
        return len(self.variable_keys)


def _format_keys(keys) -> str:
    def key_tuple(key):
        return (
            key.bag_id or "",
            -1 if key.knot_index is None else key.knot_index,
            key.kind.value,
        )

    return ", ".join(
        "{}:{}:{}".format(key.kind.value, key.bag_id, key.knot_index)
        for key in sorted(keys, key=key_tuple)
    )


__all__ = ["VariableLayout"]
