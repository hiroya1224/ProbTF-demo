"""Frozen controller configuration and nominal-model snapshots."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Tuple

from .contracts import (
    FrozenMapping,
    SerializableContract,
    _boolean,
    _matrix,
    _sha256,
    _vector,
    deep_freeze,
)


STATIC_OPTION_FIELDS = (
    "gimbal_dof",
    "gimbal_calc_in_fc",
    "hovering_approximate",
    "underactuate",
    "need_yaw_d_control",
    "integration_start_height",
    "force_landing_descending_rate",
    "estimate_mode",
)


@dataclass(frozen=True)
class ControllerStaticOptions(SerializableContract):
    gimbal_dof: int
    gimbal_calc_in_fc: bool
    hovering_approximate: bool
    underactuate: bool
    need_yaw_d_control: bool
    integration_start_height: float
    force_landing_descending_rate: float
    estimate_mode: int

    def __post_init__(self) -> None:
        gimbal_dof = int(self.gimbal_dof)
        if gimbal_dof not in (1, 2):
            raise ValueError("gimbal_dof must be one or two")
        object.__setattr__(self, "gimbal_dof", gimbal_dof)
        for name in (
            "gimbal_calc_in_fc",
            "hovering_approximate",
            "underactuate",
            "need_yaw_d_control",
        ):
            object.__setattr__(
                self, name, _boolean(getattr(self, name), name)
            )
        for name in (
            "integration_start_height",
            "force_landing_descending_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError("{} must be finite".format(name))
            object.__setattr__(self, name, value)
        object.__setattr__(self, "estimate_mode", int(self.estimate_mode))

    def to_frozen_mapping(self) -> FrozenMapping:
        return FrozenMapping(self.to_mapping())

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "ControllerStaticOptions":
        normalized = dict(values)
        if (
            "integration_start_height" not in normalized
            and "start_rp_integration_height" in normalized
        ):
            normalized["integration_start_height"] = normalized.pop(
                "start_rp_integration_height"
            )
        if (
            "integration_start_height" not in normalized
            and "start_roll_pitch_integration_height" in normalized
        ):
            normalized["integration_start_height"] = normalized.pop(
                "start_roll_pitch_integration_height"
            )
        return cls(**normalized)


def _static_options(value: Any) -> FrozenMapping:
    if isinstance(value, ControllerStaticOptions):
        return value.to_frozen_mapping()
    if not isinstance(value, Mapping):
        raise TypeError(
            "static_options must be ControllerStaticOptions or a mapping"
        )
    normalized = dict(value)
    if (
        "integration_start_height" not in normalized
        and "start_rp_integration_height" in normalized
    ):
        normalized["integration_start_height"] = normalized[
            "start_rp_integration_height"
        ]
    if (
        "integration_start_height" not in normalized
        and "start_roll_pitch_integration_height" in normalized
    ):
        normalized["integration_start_height"] = normalized[
            "start_roll_pitch_integration_height"
        ]
    missing = tuple(name for name in STATIC_OPTION_FIELDS if name not in normalized)
    if missing:
        raise ValueError(
            "controller static_options lack required fields: {}".format(
                ", ".join(missing)
            )
        )
    # Validate the plan-mandated values while retaining additional frozen
    # upstream options for provenance.
    ControllerStaticOptions.from_mapping(
        {name: normalized[name] for name in STATIC_OPTION_FIELDS}
    )
    frozen = deep_freeze(normalized)
    if not isinstance(frozen, FrozenMapping):
        raise TypeError("static_options did not normalize to a mapping")
    return frozen


def _mapping(value: Any, name: str) -> FrozenMapping:
    if not isinstance(value, Mapping):
        raise TypeError("{} must be a mapping".format(name))
    frozen = deep_freeze(value)
    if not isinstance(frozen, FrozenMapping):
        raise TypeError("{} did not normalize to a mapping".format(name))
    return frozen


@dataclass(frozen=True)
class ControllerSnapshot(SerializableContract):
    """Controller-owned model/configuration fixed across plant particles."""

    backend_id: str
    source_commit: str
    artifact_sha256: str
    nominal_model_sha256: str
    parameter_dump_sha256: str
    controller_rate_hz: float
    gains: FrozenMapping
    limits: FrozenMapping
    static_options: FrozenMapping
    nominal_mass: float
    nominal_cog: Tuple[float, float, float]
    nominal_inertia: Tuple[Tuple[float, float, float], ...]
    nominal_geometry: FrozenMapping
    schema: str = "grape.controller-snapshot/v2"

    def __post_init__(self) -> None:
        backend_id = str(self.backend_id)
        source_commit = str(self.source_commit)
        if not backend_id:
            raise ValueError("controller snapshot backend_id is required")
        if not source_commit or source_commit == "UNKNOWN":
            raise ValueError("controller snapshot source_commit is required")
        rate = float(self.controller_rate_hz)
        mass = float(self.nominal_mass)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("controller_rate_hz must be finite and positive")
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError("nominal_mass must be finite and positive")
        object.__setattr__(self, "backend_id", backend_id)
        object.__setattr__(self, "source_commit", source_commit)
        for name in (
            "artifact_sha256",
            "nominal_model_sha256",
            "parameter_dump_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), name)
            )
        object.__setattr__(self, "controller_rate_hz", rate)
        object.__setattr__(self, "gains", _mapping(self.gains, "gains"))
        object.__setattr__(self, "limits", _mapping(self.limits, "limits"))
        object.__setattr__(
            self, "static_options", _static_options(self.static_options)
        )
        object.__setattr__(self, "nominal_mass", mass)
        object.__setattr__(
            self, "nominal_cog", _vector(self.nominal_cog, "nominal_cog", 3)
        )
        object.__setattr__(
            self,
            "nominal_inertia",
            _matrix(self.nominal_inertia, "nominal_inertia", 3, 3),
        )
        object.__setattr__(
            self,
            "nominal_geometry",
            _mapping(self.nominal_geometry, "nominal_geometry"),
        )
        object.__setattr__(self, "schema", str(self.schema))

    @property
    def snapshot_id(self) -> str:
        return self.content_sha256

    def with_updates(self, **changes: Any) -> "ControllerSnapshot":
        return replace(self, **changes)

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "ControllerSnapshot":
        return cls(**dict(values))


__all__ = [
    "ControllerSnapshot",
    "ControllerStaticOptions",
    "STATIC_OPTION_FIELDS",
]
