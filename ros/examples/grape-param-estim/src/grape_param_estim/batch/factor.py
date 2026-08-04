"""ROS- and solver-free contracts for local sparse factor evaluations."""

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Tuple

import numpy as np

from grape_param_estim.batch.variables import VariableKey


@dataclass(frozen=True)
class JacobianBlock:
    """One residual Jacobian block for one canonical variable key."""

    variable_key: VariableKey
    value: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.variable_key, VariableKey):
            raise TypeError("variable_key must be a VariableKey")
        value = np.asarray(self.value, dtype=float)
        expected_columns = self.variable_key.dimension
        if (
            value.ndim != 2
            or value.shape[0] == 0
            or value.shape[1] != expected_columns
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(
                "value must be a finite non-empty 2-D array with {} columns"
                .format(expected_columns)
            )
        result = value.copy()
        result.setflags(write=False)
        object.__setattr__(self, "value", result)


@dataclass(frozen=True)
class FactorEvaluation:
    """Residual and factor-local Jacobian blocks from one evaluation.

    Jacobian storage remains factor-local.  Every block must have exactly one
    row per residual component and the canonical number of columns for its
    variable key; duplicate keys are rejected rather than implicitly summed.
    """

    residual: np.ndarray
    jacobian_blocks: Tuple[JacobianBlock, ...]
    squared_error: float
    active_set: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        residual = np.asarray(self.residual, dtype=float)
        if (
            residual.ndim != 1
            or residual.size == 0
            or not np.all(np.isfinite(residual))
        ):
            raise ValueError("residual must be a finite non-empty 1-D array")
        residual = residual.copy()
        residual.setflags(write=False)

        if type(self.jacobian_blocks) is not tuple or not self.jacobian_blocks:
            raise TypeError("jacobian_blocks must be a non-empty tuple")
        seen_keys = set()
        for block in self.jacobian_blocks:
            if not isinstance(block, JacobianBlock):
                raise TypeError(
                    "jacobian_blocks must contain only JacobianBlock values"
                )
            if block.variable_key in seen_keys:
                raise ValueError("jacobian_blocks must have unique variable keys")
            seen_keys.add(block.variable_key)
            if block.value.shape[0] != residual.size:
                raise ValueError(
                    "every Jacobian block must have one row per residual entry"
                )

        if (
            isinstance(self.squared_error, (bool, np.bool_))
            or not isinstance(self.squared_error, Real)
        ):
            raise TypeError("squared_error must be a finite scalar")
        squared_error = float(self.squared_error)
        with np.errstate(over="ignore", invalid="ignore"):
            expected_squared_error = float(residual @ residual)
        if (
            not np.isfinite(squared_error)
            or squared_error < 0.0
            or not np.isfinite(expected_squared_error)
            or not np.isclose(
                squared_error,
                expected_squared_error,
                rtol=1.0e-12,
                atol=1.0e-15,
            )
        ):
            raise ValueError(
                "squared_error must equal the residual squared Euclidean norm"
            )

        if not isinstance(self.active_set, MappingABC):
            raise TypeError("active_set must be a mapping")
        active_set = {}
        for name, mask in self.active_set.items():
            if type(name) is not str or not name or name.strip() != name:
                raise ValueError(
                    "active_set names must be non-empty canonical strings"
                )
            if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
                raise TypeError("active_set values must be boolean NumPy arrays")
            if mask.ndim == 0 or mask.size == 0:
                raise ValueError(
                    "active_set values must be non-empty arrays with rank >= 1"
                )
            copied_mask = mask.copy()
            copied_mask.setflags(write=False)
            active_set[name] = copied_mask

        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "squared_error", squared_error)
        object.__setattr__(self, "active_set", MappingProxyType(active_set))


__all__ = ["FactorEvaluation", "JacobianBlock"]
