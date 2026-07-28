"""Immutable contracts for the controller side of a plant rollout.

This module intentionally contains no Grape controller equations.  It defines
the serialization boundary shared by a live C++ wrapper, an external replay
oracle, and explicitly approximate Python adapters.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import dataclass, field, fields
from enum import Enum
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Dict, Iterator, Mapping, Optional, Protocol, Sequence, Tuple
from typing import runtime_checkable


class ControllerFidelity(str, Enum):
    """Named fidelity boundaries from the redesign plan."""

    PC_EXACT = "pc_exact"
    PC_MCU_EXACT = "pc_mcu_exact"
    ACTUATOR_CALIBRATED = "actuator_calibrated"
    PLANT_CLOSED_LOOP = "plant_closed_loop"


class ControllerTask(str, Enum):
    FACTUAL_CONTROLLER_REPLAY = "factual_controller_replay"
    OPEN_LOOP_PLANT_IDENTIFICATION = "open_loop_plant_identification"
    CLOSED_LOOP_PLANT_IDENTIFICATION = "closed_loop_plant_identification"
    POSTERIOR_CONTROLLER_EVALUATION = "posterior_controller_evaluation"


FIDELITY_PC_EXACT = ControllerFidelity.PC_EXACT.value
FIDELITY_PC_MCU_EXACT = ControllerFidelity.PC_MCU_EXACT.value
FIDELITY_ACTUATOR_CALIBRATED = ControllerFidelity.ACTUATOR_CALIBRATED.value
FIDELITY_PLANT_CLOSED_LOOP = ControllerFidelity.PLANT_CLOSED_LOOP.value
SUPPORTED_FIDELITIES = tuple(item.value for item in ControllerFidelity)

CAPABILITY_PC_CLOSED_LOOP_REPLAY = "pc_closed_loop_replay"
CAPABILITY_PC_MCU_CLOSED_LOOP_REPLAY = "pc_mcu_closed_loop_replay"
CAPABILITY_PID_TERMS = "pid_terms"
CAPABILITY_COMMAND_TIMESTAMP = "command_timestamp"
CAPABILITY_FOUR_AXIS_COMMAND = "four_axis_command"
CAPABILITY_VECTORING_FORCE = "vectoring_force"
CAPABILITY_GIMBAL_COMMAND = "gimbal_command"
CAPABILITY_ALLOCATION_INTERNAL = "allocation_internal"
CAPABILITY_TORQUE_ALLOCATION_MATRIX_INVERSE = (
    "torque_allocation_matrix_inverse"
)
CAPABILITY_PWM = "pwm"
CAPABILITY_MODE_AND_SATURATION_EVENTS = "mode_and_saturation_events"
CAPABILITY_ACTUATOR_CALIBRATED = "actuator_calibrated"
CAPABILITY_REALIZED_WRENCH = "realized_wrench"
CAPABILITY_PLANT_CLOSED_LOOP = "plant_closed_loop"
CAPABILITY_PLANT_STATE = "plant_state"

PC_EXACT_REQUIRED_CAPABILITIES = (
    CAPABILITY_PC_CLOSED_LOOP_REPLAY,
    CAPABILITY_COMMAND_TIMESTAMP,
    CAPABILITY_PID_TERMS,
    CAPABILITY_FOUR_AXIS_COMMAND,
    CAPABILITY_VECTORING_FORCE,
    CAPABILITY_GIMBAL_COMMAND,
    CAPABILITY_ALLOCATION_INTERNAL,
    CAPABILITY_TORQUE_ALLOCATION_MATRIX_INVERSE,
    CAPABILITY_MODE_AND_SATURATION_EVENTS,
)
PC_MCU_EXACT_REQUIRED_CAPABILITIES = (
    *PC_EXACT_REQUIRED_CAPABILITIES,
    CAPABILITY_PC_MCU_CLOSED_LOOP_REPLAY,
    CAPABILITY_PWM,
)
ACTUATOR_CALIBRATED_REQUIRED_CAPABILITIES = (
    CAPABILITY_ACTUATOR_CALIBRATED,
    CAPABILITY_REALIZED_WRENCH,
)
PLANT_CLOSED_LOOP_REQUIRED_CAPABILITIES = (
    CAPABILITY_PLANT_CLOSED_LOOP,
    CAPABILITY_PLANT_STATE,
)


def normalize_fidelity(value: Any) -> str:
    fidelity = value.value if isinstance(value, ControllerFidelity) else str(value)
    if fidelity not in SUPPORTED_FIDELITIES:
        raise ValueError("unsupported controller fidelity: {}".format(fidelity))
    return fidelity


def normalize_task(value: Any) -> str:
    task = value.value if isinstance(value, ControllerTask) else str(value)
    supported = tuple(item.value for item in ControllerTask)
    if task not in supported:
        raise ValueError("unsupported controller task: {}".format(task))
    return task


def expand_capabilities(values: Sequence[str]) -> Tuple[str, ...]:
    """Return capabilities plus implications of a higher-fidelity boundary.

    The implications make the pre-redesign PC+MCU capability declaration
    compatible with the new PC boundary: a backend that can replay through the
    MCU necessarily crosses the PC and gimbal-command boundaries as well.
    """

    capabilities = {str(item) for item in values}
    if CAPABILITY_PC_MCU_CLOSED_LOOP_REPLAY in capabilities:
        capabilities.update(
            (
                CAPABILITY_PC_CLOSED_LOOP_REPLAY,
                CAPABILITY_GIMBAL_COMMAND,
            )
        )
    if CAPABILITY_PLANT_CLOSED_LOOP in capabilities:
        capabilities.add(CAPABILITY_PLANT_STATE)
    return tuple(sorted(capabilities))


def required_capabilities_for_fidelity(fidelity: Any) -> Tuple[str, ...]:
    normalized = normalize_fidelity(fidelity)
    if normalized == FIDELITY_PC_EXACT:
        return PC_EXACT_REQUIRED_CAPABILITIES
    if normalized == FIDELITY_PC_MCU_EXACT:
        return PC_MCU_EXACT_REQUIRED_CAPABILITIES
    if normalized == FIDELITY_ACTUATOR_CALIBRATED:
        return ACTUATOR_CALIBRATED_REQUIRED_CAPABILITIES
    return PLANT_CLOSED_LOOP_REQUIRED_CAPABILITIES


def required_capabilities_for_task(
    task: Any, fidelity: Optional[Any] = None
) -> Tuple[str, ...]:
    normalized_task = normalize_task(task)
    if normalized_task == ControllerTask.FACTUAL_CONTROLLER_REPLAY.value:
        task_required = PC_EXACT_REQUIRED_CAPABILITIES
    elif normalized_task == ControllerTask.OPEN_LOOP_PLANT_IDENTIFICATION.value:
        task_required = ()
    elif normalized_task == ControllerTask.CLOSED_LOOP_PLANT_IDENTIFICATION.value:
        task_required = (
            *PC_EXACT_REQUIRED_CAPABILITIES,
            *PLANT_CLOSED_LOOP_REQUIRED_CAPABILITIES,
        )
    else:
        task_required = (
            *PC_EXACT_REQUIRED_CAPABILITIES,
            *PLANT_CLOSED_LOOP_REQUIRED_CAPABILITIES,
        )
    if fidelity is None:
        return tuple(dict.fromkeys(task_required))
    return tuple(
        dict.fromkeys(
            (
                *task_required,
                *required_capabilities_for_fidelity(fidelity),
            )
        )
    )


def _scalar(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def deep_freeze(value: Any) -> Any:
    """Convert supported JSON-like data into recursively immutable values."""

    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, MappingABC):
        return FrozenMapping(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return deep_freeze(value.tolist())
        except (AttributeError, TypeError, ValueError):
            pass
    scalar = _scalar(value)
    if scalar is not value:
        return deep_freeze(scalar)
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((deep_freeze(item) for item in value), key=repr))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("serialized controller values must be finite")
        return float(value)
    raise TypeError(
        "unsupported controller serialization value: {}".format(
            type(value).__name__
        )
    )


def deep_thaw(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return {key: deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [deep_thaw(item) for item in value]
    if isinstance(value, SerializableContract):
        return value.to_mapping()
    return value


class FrozenMapping(MappingABC):
    """Small hashable mapping used inside frozen controller contracts."""

    __slots__ = ("_items", "_lookup", "_hash")

    def __init__(self, values: Optional[Mapping[str, Any]] = None, **kwargs: Any):
        source: Dict[str, Any] = {}
        if values is not None:
            source.update({str(key): value for key, value in values.items()})
        source.update({str(key): value for key, value in kwargs.items()})
        self._items = tuple(
            sorted(
                ((key, deep_freeze(value)) for key, value in source.items()),
                key=lambda item: item[0],
            )
        )
        self._lookup = dict(self._items)
        self._hash = hash(self._items)

    def __getitem__(self, key: str) -> Any:
        return self._lookup[key]

    def __iter__(self) -> Iterator[str]:
        return (item[0] for item in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return "FrozenMapping({})".format(dict(self._items))

    def to_mapping(self) -> Mapping[str, Any]:
        return {key: deep_thaw(value) for key, value in self._items}


class SerializableContract:
    """Mixin providing deterministic JSON and content hashing."""

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            item.name: deep_thaw(getattr(self, item.name))
            for item in fields(self)
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError("{} must be a built-in bool".format(name))
    return value


def _vector(
    values: Any, name: str, size: Optional[int] = None
) -> Tuple[float, ...]:
    frozen = deep_freeze(values)
    if not isinstance(frozen, tuple):
        raise ValueError("{} must be a vector".format(name))
    result = tuple(_finite(item, name) for item in frozen)
    if size is not None and len(result) != size:
        raise ValueError("{} must contain {} values".format(name, size))
    return result


def _matrix(
    values: Any,
    name: str,
    rows: Optional[int] = None,
    columns: Optional[int] = None,
) -> Tuple[Tuple[float, ...], ...]:
    frozen = deep_freeze(values)
    if not isinstance(frozen, tuple):
        raise ValueError("{} must be a matrix".format(name))
    result = tuple(_vector(row, name, columns) for row in frozen)
    if rows is not None and len(result) != rows:
        raise ValueError("{} must contain {} rows".format(name, rows))
    if result and columns is None:
        width = len(result[0])
        if any(len(row) != width for row in result):
            raise ValueError("{} rows must have equal length".format(name))
    return result


def _sha256(value: Any, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("{} must be a lowercase SHA-256".format(name))
    return digest


@dataclass(frozen=True)
class PidCoreState(SerializableContract):
    error_p: float = 0.0
    error_i: float = 0.0
    previous_error_i: float = 0.0
    error_d: float = 0.0
    result: float = 0.0
    p_term: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "error_p",
            "error_i",
            "previous_error_i",
            "error_d",
            "result",
            "p_term",
            "i_term",
            "d_term",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PidCoreState":
        return cls(**dict(values))


def _zero_pid_state() -> Tuple[PidCoreState, ...]:
    return tuple(PidCoreState() for _ in range(6))


@dataclass(frozen=True)
class ControllerCoreState(SerializableContract):
    pid: Tuple[PidCoreState, ...] = _zero_pid_state()
    start_roll_pitch_integration: bool = False
    previous_stamp: float = -1.0
    previous_flight_state: int = -1
    target_gimbal_angles: Tuple[float, ...] = ()
    target_roll: float = 0.0
    target_pitch: float = 0.0
    previous_control_mode: Tuple[int, ...] = ()
    previous_force_landing: Optional[bool] = None
    pending_events: int = 0

    def __post_init__(self) -> None:
        pid = tuple(
            item
            if isinstance(item, PidCoreState)
            else PidCoreState.from_mapping(item)
            for item in self.pid
        )
        if len(pid) != 6:
            raise ValueError("ControllerCoreState.pid must contain six axes")
        object.__setattr__(self, "pid", pid)
        object.__setattr__(
            self,
            "start_roll_pitch_integration",
            _boolean(
                self.start_roll_pitch_integration,
                "start_roll_pitch_integration",
            ),
        )
        object.__setattr__(
            self, "previous_stamp", _finite(self.previous_stamp, "previous_stamp")
        )
        object.__setattr__(
            self, "previous_flight_state", int(self.previous_flight_state)
        )
        object.__setattr__(
            self,
            "target_gimbal_angles",
            _vector(
                self.target_gimbal_angles,
                "target_gimbal_angles",
            ),
        )
        object.__setattr__(
            self, "target_roll", _finite(self.target_roll, "target_roll")
        )
        object.__setattr__(
            self, "target_pitch", _finite(self.target_pitch, "target_pitch")
        )
        modes = tuple(int(item) for item in self.previous_control_mode)
        if modes and len(modes) != 6:
            raise ValueError(
                "previous_control_mode must be empty or contain six axes"
            )
        object.__setattr__(self, "previous_control_mode", modes)
        object.__setattr__(
            self,
            "previous_force_landing",
            None
            if self.previous_force_landing is None
            else _boolean(
                self.previous_force_landing,
                "previous_force_landing",
            ),
        )
        if (
            isinstance(self.pending_events, bool)
            or not isinstance(self.pending_events, Integral)
            or not 0 <= int(self.pending_events) <= 0xFFFFFFFF
        ):
            raise ValueError(
                "pending_events must be a non-negative uint32 integer"
            )
        object.__setattr__(
            self, "pending_events", int(self.pending_events)
        )

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "ControllerCoreState":
        return cls(**dict(values))


@dataclass(frozen=True)
class ControllerCoreInput(SerializableContract):
    stamp: float
    dt: float
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    orientation: Tuple[Tuple[float, float, float], ...]
    angular_velocity: Tuple[float, float, float]
    target_position: Tuple[float, float, float]
    target_velocity: Tuple[float, float, float]
    target_acceleration: Tuple[float, float, float]
    target_orientation: Tuple[Tuple[float, float, float], ...]
    target_angular_velocity: Tuple[float, float, float]
    target_angular_acceleration: Tuple[float, float, float]
    control_mode: Tuple[int, int, int, int, int, int]
    integration_enabled: Tuple[bool, bool, bool, bool, bool, bool]
    flight_state: int
    force_landing: bool
    joint_positions: Tuple[float, ...] = ()
    initial_height: float = 0.0
    reset: bool = False
    current_rpy: Optional[Tuple[float, float, float]] = None
    target_rpy: Optional[Tuple[float, float, float]] = None
    pid_config: Optional[
        Tuple[Tuple[float, ...], ...]
    ] = None
    state_previous_stamp: Optional[float] = None
    allocation_geometry: FrozenMapping = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp", _finite(self.stamp, "stamp"))
        delta = _finite(self.dt, "dt")
        if delta < 0.0:
            raise ValueError("dt must be non-negative")
        object.__setattr__(self, "dt", delta)
        for name in (
            "position",
            "velocity",
            "angular_velocity",
            "target_position",
            "target_velocity",
            "target_acceleration",
            "target_angular_velocity",
            "target_angular_acceleration",
        ):
            object.__setattr__(
                self, name, _vector(getattr(self, name), name, 3)
            )
        object.__setattr__(
            self, "orientation", _matrix(self.orientation, "orientation", 3, 3)
        )
        object.__setattr__(
            self,
            "target_orientation",
            _matrix(self.target_orientation, "target_orientation", 3, 3),
        )
        modes = tuple(int(item) for item in self.control_mode)
        if len(modes) != 6:
            raise ValueError("control_mode must contain six axes")
        object.__setattr__(self, "control_mode", modes)
        integration = tuple(
            _boolean(item, "integration_enabled") for item in self.integration_enabled
        )
        if len(integration) != 6:
            raise ValueError("integration_enabled must contain six axes")
        object.__setattr__(self, "integration_enabled", integration)
        object.__setattr__(self, "flight_state", int(self.flight_state))
        object.__setattr__(
            self,
            "force_landing",
            _boolean(self.force_landing, "force_landing"),
        )
        object.__setattr__(
            self,
            "joint_positions",
            _vector(self.joint_positions, "joint_positions"),
        )
        object.__setattr__(
            self,
            "initial_height",
            _finite(self.initial_height, "initial_height"),
        )
        object.__setattr__(self, "reset", _boolean(self.reset, "reset"))
        object.__setattr__(
            self,
            "current_rpy",
            None
            if self.current_rpy is None
            else _vector(self.current_rpy, "current_rpy", 3),
        )
        object.__setattr__(
            self,
            "target_rpy",
            None
            if self.target_rpy is None
            else _vector(self.target_rpy, "target_rpy", 3),
        )
        pid_config = self.pid_config
        if pid_config is not None:
            pid_config = _matrix(
                pid_config, "pid_config", 6, 10
            )
            if any(
                value < 0.0
                for row in pid_config
                for value in row[3:]
            ):
                raise ValueError(
                    "PID configuration limits must be non-negative"
                )
        object.__setattr__(self, "pid_config", pid_config)
        object.__setattr__(
            self,
            "state_previous_stamp",
            None
            if self.state_previous_stamp is None
            else _finite(
                self.state_previous_stamp, "state_previous_stamp"
            ),
        )
        if not isinstance(self.allocation_geometry, MappingABC):
            raise TypeError("allocation_geometry must be a mapping")
        object.__setattr__(
            self,
            "allocation_geometry",
            FrozenMapping(self.allocation_geometry),
        )

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "ControllerCoreInput":
        return cls(**dict(values))


@dataclass(frozen=True)
class ControllerCommand(SerializableContract):
    stamp: float
    base_thrust: Tuple[float, ...]
    gimbal_angle: Tuple[float, ...]
    generalized_wrench: Optional[Tuple[float, ...]] = None
    events: Tuple[int, ...] = ()
    saturated: bool = False
    four_axis_angles: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_vectoring_force: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp", _finite(self.stamp, "stamp"))
        object.__setattr__(
            self, "base_thrust", _vector(self.base_thrust, "base_thrust")
        )
        object.__setattr__(
            self, "gimbal_angle", _vector(self.gimbal_angle, "gimbal_angle")
        )
        wrench = self.generalized_wrench
        object.__setattr__(
            self,
            "generalized_wrench",
            None
            if wrench is None
            else _vector(wrench, "generalized_wrench", 6),
        )
        object.__setattr__(
            self, "events", tuple(int(item) for item in self.events)
        )
        object.__setattr__(
            self, "saturated", _boolean(self.saturated, "saturated")
        )
        object.__setattr__(
            self,
            "four_axis_angles",
            _vector(self.four_axis_angles, "four_axis_angles", 3),
        )
        object.__setattr__(
            self,
            "target_vectoring_force",
            _vector(self.target_vectoring_force, "target_vectoring_force"),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ControllerCommand":
        return cls(**dict(values))


@dataclass(frozen=True)
class ControllerCoreOutput(SerializableContract):
    pid_result: Tuple[float, float, float, float, float, float]
    pid_p_term: Tuple[float, float, float, float, float, float]
    pid_i_term: Tuple[float, float, float, float, float, float]
    pid_d_term: Tuple[float, float, float, float, float, float]
    target_vectoring_force: Tuple[float, ...]
    base_thrust: Tuple[float, ...]
    gimbal_angle: Tuple[float, ...]
    torque_allocation_matrix_inverse: Tuple[Tuple[float, ...], ...]
    target_roll: float
    target_pitch: float
    candidate_yaw_term: float
    events: Tuple[int, ...]
    stamp: float = 0.0
    saturated: bool = False
    four_axis_angles: Tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    generalized_wrench: Optional[Tuple[float, ...]] = None
    effective_target_acceleration: Tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )

    def __post_init__(self) -> None:
        for name in ("pid_result", "pid_p_term", "pid_i_term", "pid_d_term"):
            object.__setattr__(
                self, name, _vector(getattr(self, name), name, 6)
            )
        for name in ("target_vectoring_force", "base_thrust", "gimbal_angle"):
            object.__setattr__(
                self, name, _vector(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "torque_allocation_matrix_inverse",
            _matrix(
                self.torque_allocation_matrix_inverse,
                "torque_allocation_matrix_inverse",
            ),
        )
        for name in ("target_roll", "target_pitch", "candidate_yaw_term", "stamp"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(
            self, "events", tuple(int(item) for item in self.events)
        )
        object.__setattr__(
            self, "saturated", _boolean(self.saturated, "saturated")
        )
        object.__setattr__(
            self,
            "four_axis_angles",
            _vector(
                self.four_axis_angles, "four_axis_angles", 3
            ),
        )
        object.__setattr__(
            self,
            "generalized_wrench",
            None
            if self.generalized_wrench is None
            else _vector(self.generalized_wrench, "generalized_wrench", 6),
        )
        object.__setattr__(
            self,
            "effective_target_acceleration",
            _vector(
                self.effective_target_acceleration,
                "effective_target_acceleration",
                3,
            ),
        )

    @property
    def command(self) -> ControllerCommand:
        return ControllerCommand(
            stamp=self.stamp,
            base_thrust=self.base_thrust,
            gimbal_angle=self.gimbal_angle,
            generalized_wrench=self.generalized_wrench,
            events=self.events,
            saturated=self.saturated,
            four_axis_angles=self.four_axis_angles,
            target_vectoring_force=self.target_vectoring_force,
        )

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "ControllerCoreOutput":
        return cls(**dict(values))


@dataclass(frozen=True)
class ControllerBackendIdentity(SerializableContract):
    backend_id: str
    fidelity: str
    is_exact: bool
    capabilities: Tuple[str, ...]
    implementation_language: str = "unknown"
    source_commit: str = "UNKNOWN"
    artifact_sha256: str = "0" * 64
    protocol: str = "grape.controller-backend/v2"

    def __post_init__(self) -> None:
        backend_id = str(self.backend_id)
        if not backend_id:
            raise ValueError("controller backend_id is required")
        fidelity = normalize_fidelity(self.fidelity)
        is_exact = self.is_exact
        if type(is_exact) is not bool:
            raise TypeError("controller backend is_exact must be a built-in bool")
        language = str(self.implementation_language)
        if is_exact and language.strip().lower() not in ("c++", "cpp"):
            raise ValueError("an exact controller backend must be implemented in C++")
        if is_exact and (
            "python" in backend_id.lower() or "surrogate" in backend_id.lower()
        ):
            raise ValueError("an exact backend cannot identify a Python surrogate")
        capabilities = tuple(dict.fromkeys(str(item) for item in self.capabilities))
        object.__setattr__(self, "backend_id", backend_id)
        object.__setattr__(self, "fidelity", fidelity)
        object.__setattr__(self, "is_exact", is_exact)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "implementation_language", language)
        object.__setattr__(self, "source_commit", str(self.source_commit))
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(self.artifact_sha256, "artifact_sha256"),
        )
        object.__setattr__(self, "protocol", str(self.protocol))

    @property
    def expanded_capabilities(self) -> Tuple[str, ...]:
        return expand_capabilities(self.capabilities)

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "ControllerBackendIdentity":
        return cls(**dict(values))


@runtime_checkable
class ControllerBackend(Protocol):
    identity: ControllerBackendIdentity

    def reset(
        self,
        snapshot: Any,
        initial_state: ControllerCoreState,
    ) -> None:
        ...

    def step(self, item: ControllerCoreInput) -> ControllerCoreOutput:
        ...


__all__ = [
    "ACTUATOR_CALIBRATED_REQUIRED_CAPABILITIES",
    "CAPABILITY_ACTUATOR_CALIBRATED",
    "CAPABILITY_ALLOCATION_INTERNAL",
    "CAPABILITY_COMMAND_TIMESTAMP",
    "CAPABILITY_FOUR_AXIS_COMMAND",
    "CAPABILITY_GIMBAL_COMMAND",
    "CAPABILITY_MODE_AND_SATURATION_EVENTS",
    "CAPABILITY_PC_CLOSED_LOOP_REPLAY",
    "CAPABILITY_PC_MCU_CLOSED_LOOP_REPLAY",
    "CAPABILITY_PID_TERMS",
    "CAPABILITY_PLANT_CLOSED_LOOP",
    "CAPABILITY_PLANT_STATE",
    "CAPABILITY_PWM",
    "CAPABILITY_REALIZED_WRENCH",
    "CAPABILITY_TORQUE_ALLOCATION_MATRIX_INVERSE",
    "CAPABILITY_VECTORING_FORCE",
    "ControllerBackend",
    "ControllerBackendIdentity",
    "ControllerCommand",
    "ControllerCoreInput",
    "ControllerCoreOutput",
    "ControllerCoreState",
    "ControllerFidelity",
    "ControllerTask",
    "FIDELITY_ACTUATOR_CALIBRATED",
    "FIDELITY_PC_EXACT",
    "FIDELITY_PC_MCU_EXACT",
    "FIDELITY_PLANT_CLOSED_LOOP",
    "FrozenMapping",
    "PC_EXACT_REQUIRED_CAPABILITIES",
    "PC_MCU_EXACT_REQUIRED_CAPABILITIES",
    "PLANT_CLOSED_LOOP_REQUIRED_CAPABILITIES",
    "PidCoreState",
    "SerializableContract",
    "SUPPORTED_FIDELITIES",
    "deep_freeze",
    "deep_thaw",
    "expand_capabilities",
    "normalize_fidelity",
    "normalize_task",
    "required_capabilities_for_fidelity",
    "required_capabilities_for_task",
]
