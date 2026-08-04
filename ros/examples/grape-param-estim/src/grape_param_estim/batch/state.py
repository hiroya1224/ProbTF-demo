"""Canonical nonlinear batch state, right retraction, and solver scaling."""

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping

import numpy as np

from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.geometry import so3_exp


def _proper_rotation(value: object) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("orientation state must be a finite 3 by 3 matrix")
    if not np.allclose(result.T @ result, np.eye(3), rtol=0.0, atol=1.0e-9):
        raise ValueError("orientation state must be orthonormal")
    if not np.isclose(np.linalg.det(result), 1.0, rtol=0.0, atol=1.0e-9):
        raise ValueError("orientation state must have determinant one")
    copied = result.copy()
    copied.setflags(write=False)
    return copied


def _state_vector(value: object, dimension: int, kind: VariableKind) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (dimension,) or not np.all(np.isfinite(result)):
        raise ValueError(
            "{} state must contain {} finite values".format(
                kind.value,
                dimension,
            )
        )
    copied = result.copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, eq=False)
class BatchState:
    """One complete nonlinear state for a :class:`VariableLayout`.

    The key named ``ORIENTATION_TANGENT`` stores an absolute proper rotation
    matrix at a nonlinear point.  Its canonical dimension remains three
    because only a right-tangent increment enters a linear solve.  No global
    absolute rotation vector or freely optimized quaternion is constructed.
    """

    layout: VariableLayout
    values: Mapping[VariableKey, np.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.layout, VariableLayout):
            raise TypeError("layout must be a VariableLayout")
        if not isinstance(self.values, MappingABC):
            raise TypeError("values must be a mapping")
        supplied_keys = set(self.values.keys())
        expected_keys = set(self.layout.variable_keys)
        if supplied_keys != expected_keys:
            missing = expected_keys - supplied_keys
            extra = supplied_keys - expected_keys
            raise ValueError(
                "values must contain every layout key exactly once "
                "(missing={}, extra={})".format(len(missing), len(extra))
            )
        copied = {}
        for key in self.layout.variable_keys:
            if not isinstance(key, VariableKey):
                raise TypeError("layout contains a non-VariableKey value")
            value = self.values[key]
            if key.kind is VariableKind.ORIENTATION_TANGENT:
                copied[key] = _proper_rotation(value)
            else:
                copied[key] = _state_vector(value, key.dimension, key.kind)
        object.__setattr__(self, "values", MappingProxyType(copied))

    def value(self, variable_key: VariableKey) -> np.ndarray:
        """Return the immutable array stored for one exact layout key."""

        if not isinstance(variable_key, VariableKey):
            raise TypeError("variable_key must be a VariableKey")
        try:
            return self.values[variable_key]
        except KeyError as error:
            raise KeyError("variable_key is not present in this state") from error

    def knot_value(
        self,
        bag_id: str,
        knot_index: int,
        kind: VariableKind,
    ) -> np.ndarray:
        """Return one knot field without relying on positional tuple order."""

        return self.value(
            VariableKey(kind, bag_id=bag_id, knot_index=knot_index)
        )

    def retract(self, physical_delta: np.ndarray) -> "BatchState":
        """Apply a complete physical-coordinate tangent step immutably."""

        delta = np.asarray(physical_delta, dtype=float)
        if (
            delta.shape != (self.layout.total_dimension,)
            or not np.all(np.isfinite(delta))
        ):
            raise ValueError(
                "physical_delta must contain one finite value per layout column"
            )
        retracted = {}
        for key in self.layout.variable_keys:
            local_delta = delta[self.layout.column_slice(key)]
            current = self.values[key]
            if key.kind is VariableKind.ORIENTATION_TANGENT:
                retracted[key] = current @ so3_exp(local_delta)
            else:
                retracted[key] = current + local_delta
        return BatchState(self.layout, retracted)


@dataclass(frozen=True)
class StateScaling:
    """Positive physical units per scaled coordinate, grouped by kind."""

    kind_scales: Mapping[VariableKind, float]

    def __post_init__(self) -> None:
        if not isinstance(self.kind_scales, MappingABC):
            raise TypeError("kind_scales must be a mapping")
        supplied = set(self.kind_scales.keys())
        expected = set(VariableKind)
        if supplied != expected:
            raise ValueError(
                "kind_scales must contain every VariableKind exactly once"
            )
        copied = {}
        for kind in VariableKind:
            value = self.kind_scales[kind]
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
            ):
                raise TypeError("state scales must be finite positive scalars")
            scale = float(value)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError("state scales must be finite and positive")
            copied[kind] = scale
        object.__setattr__(self, "kind_scales", MappingProxyType(copied))

    @classmethod
    def unit(cls) -> "StateScaling":
        """Return an explicit all-one scaling, primarily for tests."""

        return cls({kind: 1.0 for kind in VariableKind})

    def vector_for(self, layout: VariableLayout) -> np.ndarray:
        """Expand kind scales into deterministic layout-column order."""

        if not isinstance(layout, VariableLayout):
            raise TypeError("layout must be a VariableLayout")
        result = np.empty(layout.total_dimension, dtype=float)
        for key in layout.variable_keys:
            result[layout.column_slice(key)] = self.kind_scales[key.kind]
        result.setflags(write=False)
        return result


__all__ = ["BatchState", "StateScaling"]
