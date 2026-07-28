"""External exact-controller transports at the v2 package boundary.

The process/C-ABI implementations remain source-compatible with their legacy
location while new code can import them from the role-specific controller
package.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import threading
import time
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np
from scipy.spatial.transform import Rotation

from grape_param_estim.alternative_backends import (
    CtypesExactControllerOracle,
    EXACT_ORACLE_PROTOCOL,
    ExactOracleError,
    ExactOracleIdentity,
    ExactOracleProtocolError,
    ExactOracleReplayOutput,
    ExactOracleUnavailable,
    PersistentSubprocessExactControllerOracle,
    SubprocessExactControllerOracle,
)

from .contracts import (
    ControllerBackendIdentity,
    ControllerCoreInput,
    ControllerCoreOutput,
    ControllerCoreState,
    FIDELITY_PC_MCU_EXACT,
    FrozenMapping,
    PidCoreState,
    deep_thaw,
)
from .snapshot import ControllerSnapshot


class ExternalControllerOracle(Protocol):
    is_exact: bool
    identity: ExactOracleIdentity

    def replay(self, payload: Mapping[str, Any]) -> ExactOracleReplayOutput:
        ...


@dataclass
class _PendingReplay:
    payload: Mapping[str, Any]
    batch_key: str
    tick_count: int
    job_count: int
    completed: bool = False
    output: Optional[ExactOracleReplayOutput] = None
    error: Optional[BaseException] = None


def _batchable_replay_request(
    payload: Mapping[str, Any],
) -> Tuple[str, int, int]:
    if not isinstance(payload, Mapping):
        raise TypeError("exact replay payload must be a mapping")
    jobs = payload.get("jobs")
    if (
        not isinstance(jobs, (list, tuple))
        or not jobs
        or any(not isinstance(job, Mapping) for job in jobs)
    ):
        raise ExactOracleProtocolError(
            "exact replay batching requires one or more mapping jobs"
        )
    tick_count = 0
    for index, job in enumerate(jobs):
        ticks = job.get("ticks")
        if not isinstance(ticks, (list, tuple)):
            raise ExactOracleProtocolError(
                "exact replay job {} lacks a tick sequence".format(index)
            )
        tick_count += len(ticks)
    header = {
        str(name): value
        for name, value in payload.items()
        if name != "jobs"
    }
    try:
        batch_key = json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ExactOracleProtocolError(
            "exact replay batch header must be finite JSON data"
        ) from exc
    return batch_key, tick_count, len(jobs)


class BatchingExactControllerOracle:
    """Coalesce compatible concurrent replays onto one exact transport.

    Closed-loop rollouts each retain a private
    :class:`StatefulExactOracleControllerBackend`, while this wrapper is
    shared by all of them.  The first waiting caller becomes the batch leader,
    briefly collects requests with the same snapshot/evidence header, and
    submits their jobs in FIFO order.  No worker process or background thread
    is created.
    """

    is_exact = True
    transport_is_persistent = True

    def __init__(
        self,
        oracle: ExternalControllerOracle,
        max_batch_size: int = 32,
        batch_wait_s: float = 0.001,
    ) -> None:
        identity = getattr(oracle, "identity", None)
        if isinstance(oracle, SubprocessExactControllerOracle):
            raise ExactOracleProtocolError(
                "one-shot subprocess oracles cannot back closed-loop batches"
            )
        if (
            getattr(oracle, "is_exact", None) is not True
            or not isinstance(identity, ExactOracleIdentity)
            or not callable(getattr(oracle, "replay", None))
        ):
            raise TypeError(
                "batching requires an exact replay-capable oracle"
            )
        batch_size = int(max_batch_size)
        wait = float(batch_wait_s)
        if batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if not np.isfinite(wait) or wait < 0.0:
            raise ValueError(
                "batch_wait_s must be finite and non-negative"
            )
        self.identity = identity
        self.max_batch_size = batch_size
        self.batch_wait_s = wait
        self._oracle = oracle
        self._condition = threading.Condition(threading.RLock())
        self._queue = deque()
        self._leader_active = False

    @property
    def underlying(self) -> ExternalControllerOracle:
        return self._oracle

    def _collect_leader_batch(
        self, leader: _PendingReplay
    ) -> Tuple[_PendingReplay, ...]:
        deadline = time.monotonic() + self.batch_wait_s
        with self._condition:
            while len(self._queue) < self.max_batch_size:
                if (
                    len(self._queue) > 1
                    and self._queue[1].batch_key
                    != leader.batch_key
                ):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            batch = []
            while (
                self._queue
                and len(batch) < self.max_batch_size
                and self._queue[0].batch_key
                == leader.batch_key
            ):
                batch.append(self._queue.popleft())
            if not batch or batch[0] is not leader:
                raise RuntimeError(
                    "exact replay batch queue lost its leader"
                )
            return tuple(batch)

    def _split_reply(
        self,
        batch: Sequence[_PendingReplay],
        reply: ExactOracleReplayOutput,
    ) -> Tuple[ExactOracleReplayOutput, ...]:
        if (
            not isinstance(reply, ExactOracleReplayOutput)
            or reply.identity != self.identity
        ):
            raise ExactOracleProtocolError(
                "batched oracle returned the wrong replay identity"
            )
        total_ticks = sum(item.tick_count for item in batch)
        total_jobs = sum(item.job_count for item in batch)
        if any(
            values.shape[0] != total_ticks
            for values in reply.continuous.values()
        ) or reply.events.shape[0] != total_ticks:
            raise ExactOracleProtocolError(
                "batched oracle reply row count does not match its jobs"
            )
        if (
            reply.final_states
            and len(reply.final_states) != total_jobs
        ):
            raise ExactOracleProtocolError(
                "batched oracle final-state count does not match its jobs"
            )
        outputs = []
        tick_offset = 0
        job_offset = 0
        for item in batch:
            continuous = {}
            tick_stop = tick_offset + item.tick_count
            job_stop = job_offset + item.job_count
            for name, values in reply.continuous.items():
                selected = np.array(
                    values[tick_offset:tick_stop], copy=True
                )
                if name == "job_tick" and selected.shape[1] >= 1:
                    selected[:, 0] -= job_offset
                continuous[name] = selected
            final_states = (
                reply.final_states[job_offset:job_stop]
                if reply.final_states
                else ()
            )
            outputs.append(
                ExactOracleReplayOutput(
                    identity=self.identity,
                    continuous=continuous,
                    events=reply.events[tick_offset:tick_stop],
                    final_states=final_states,
                )
            )
            tick_offset = tick_stop
            job_offset = job_stop
        return tuple(outputs)

    def _invoke_batch(
        self, batch: Sequence[_PendingReplay]
    ) -> Tuple[ExactOracleReplayOutput, ...]:
        if len(batch) == 1:
            return (self._oracle.replay(batch[0].payload),)
        combined = dict(batch[0].payload)
        combined["jobs"] = [
            job
            for item in batch
            for job in item.payload["jobs"]
        ]
        return self._split_reply(
            batch, self._oracle.replay(combined)
        )

    def replay(
        self, payload: Mapping[str, Any]
    ) -> ExactOracleReplayOutput:
        batch_key, tick_count, job_count = (
            _batchable_replay_request(payload)
        )
        pending = _PendingReplay(
            payload=payload,
            batch_key=batch_key,
            tick_count=tick_count,
            job_count=job_count,
        )
        with self._condition:
            self._queue.append(pending)
            self._condition.notify_all()
            while True:
                if pending.completed:
                    if pending.error is not None:
                        raise pending.error
                    return pending.output
                if (
                    not self._leader_active
                    and self._queue
                    and self._queue[0] is pending
                ):
                    self._leader_active = True
                    break
                self._condition.wait()

        batch = (pending,)
        try:
            batch = self._collect_leader_batch(pending)
            outputs = self._invoke_batch(batch)
            if len(outputs) != len(batch):
                raise RuntimeError(
                    "exact replay batch produced the wrong output count"
                )
        except BaseException as exc:
            outputs = ()
            error = exc
        else:
            error = None

        with self._condition:
            for index, item in enumerate(batch):
                if (
                    self._queue
                    and self._queue[0] is item
                ):
                    self._queue.popleft()
                item.output = (
                    None if error is not None else outputs[index]
                )
                item.error = error
                item.completed = True
            self._leader_active = False
            self._condition.notify_all()
        if error is not None:
            raise error
        return pending.output


def controller_backend_identity(
    identity: ExactOracleIdentity,
) -> ControllerBackendIdentity:
    """Translate the legacy transport identity into the v2 domain identity."""

    if not isinstance(identity, ExactOracleIdentity):
        raise TypeError("identity must be ExactOracleIdentity")
    return ControllerBackendIdentity(
        backend_id=identity.backend_id,
        fidelity=getattr(identity, "fidelity", FIDELITY_PC_MCU_EXACT),
        is_exact=True,
        capabilities=identity.capabilities,
        implementation_language=identity.implementation_language,
        source_commit=identity.source_commit,
        artifact_sha256=identity.artifact_sha256,
        protocol=identity.protocol,
    )


def _finite_vector(
    value: Any, name: str, size: int = None
) -> Tuple[float, ...]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if (
        (size is not None and array.size != size)
        or not np.all(np.isfinite(array))
    ):
        suffix = "" if size is None else " with length {}".format(size)
        raise ExactOracleProtocolError(
            "{} must be a finite vector{}".format(name, suffix)
        )
    return tuple(float(item) for item in array)


def _finite_matrix(
    value: Any, name: str, shape: Tuple[int, int]
) -> Tuple[Tuple[float, ...], ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ExactOracleProtocolError(
            "{} must have finite shape {}".format(name, shape)
        )
    return tuple(
        tuple(float(item) for item in row) for row in array
    )


def _mapping_value(
    values: Mapping[str, Any],
    name: str,
    aliases: Sequence[str] = (),
) -> Any:
    for candidate in (name,) + tuple(aliases):
        if candidate in values:
            return values[candidate]
    raise ExactOracleProtocolError(
        "controller snapshot lacks {}".format(name)
    )


def _axis_values(
    values: Mapping[str, Any],
    name: str,
    aliases: Sequence[str] = (),
) -> Tuple[float, ...]:
    return _finite_vector(
        _mapping_value(values, name, aliases),
        name,
        6,
    )


def _geometry_payload(
    snapshot: ControllerSnapshot,
    override: Mapping[str, Any] = None,
) -> Mapping[str, Any]:
    geometry: Dict[str, Any] = dict(
        deep_thaw(snapshot.nominal_geometry)
    )
    if override:
        geometry.update(dict(deep_thaw(override)))
    origins = _mapping_value(
        geometry, "rotor_origins_from_cog"
    )
    directions = tuple(
        int(item)
        for item in _mapping_value(geometry, "rotor_directions")
    )
    rotations = _mapping_value(
        geometry, "thrust_coordinate_rotations"
    )
    origin_rows = tuple(
        _finite_vector(item, "rotor_origins_from_cog", 3)
        for item in origins
    )
    rotation_rows = tuple(
        _finite_matrix(
            item, "thrust_coordinate_rotations", (3, 3)
        )
        for item in rotations
    )
    if (
        not origin_rows
        or len(origin_rows) != len(directions)
        or len(origin_rows) != len(rotation_rows)
    ):
        raise ExactOracleProtocolError(
            "controller geometry rotor arrays are not aligned"
        )
    moment_force_rate = float(
        _mapping_value(geometry, "moment_force_rate")
    )
    mass = float(geometry.get("mass", snapshot.nominal_mass))
    inertia = _finite_matrix(
        geometry.get("inertia", snapshot.nominal_inertia),
        "inertia",
        (3, 3),
    )
    if (
        not np.isfinite(moment_force_rate)
        or not np.isfinite(mass)
        or mass <= 0.0
    ):
        raise ExactOracleProtocolError(
            "controller geometry mass/rate must be finite and valid"
        )
    return {
        "mass": mass,
        "inertia": [list(row) for row in inertia],
        "moment_force_rate": moment_force_rate,
        "rotor_origins_from_cog": [
            list(row) for row in origin_rows
        ],
        "rotor_directions": list(directions),
        "thrust_coordinate_rotations": [
            [list(row) for row in matrix]
            for matrix in rotation_rows
        ],
    }


def _snapshot_payload(
    snapshot: ControllerSnapshot,
) -> Mapping[str, Any]:
    gains = snapshot.gains
    limits = snapshot.limits
    p_gain = _axis_values(gains, "p_gain")
    i_gain = _axis_values(gains, "i_gain")
    d_gain = _axis_values(gains, "d_gain")
    limit_sum = _axis_values(limits, "limit_sum")
    limit_p = _axis_values(limits, "limit_p")
    limit_i = _axis_values(limits, "limit_i")
    limit_d = _axis_values(limits, "limit_d")
    limit_error_p = _axis_values(
        limits, "limit_error_p", ("limit_err_p",)
    )
    limit_error_i = _axis_values(
        limits, "limit_error_i", ("limit_err_i",)
    )
    limit_error_d = _axis_values(
        limits, "limit_error_d", ("limit_err_d",)
    )
    axis_names = tuple(
        snapshot.gains.get(
            "axis_names",
            ("x", "y", "z", "roll", "pitch", "yaw"),
        )
    )
    if len(axis_names) != 6:
        raise ExactOracleProtocolError(
            "controller snapshot axis_names must contain six values"
        )
    pid = []
    for index in range(6):
        pid.append(
            {
                "name": str(axis_names[index]),
                "p_gain": p_gain[index],
                "i_gain": i_gain[index],
                "d_gain": d_gain[index],
                "limit_sum": limit_sum[index],
                "limit_p": limit_p[index],
                "limit_i": limit_i[index],
                "limit_d": limit_d[index],
                "limit_error_p": limit_error_p[index],
                "limit_error_i": limit_error_i[index],
                "limit_error_d": limit_error_d[index],
            }
        )
    options = snapshot.static_options
    geometry = dict(deep_thaw(snapshot.nominal_geometry))
    gravity = float(geometry.get("gravity", 9.797))
    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ExactOracleProtocolError(
            "controller allocation gravity must be finite and positive"
        )
    return {
        "snapshot_id": snapshot.snapshot_id,
        "pid": pid,
        "pose_config": {
            "need_yaw_d_control": bool(
                options["need_yaw_d_control"]
            ),
            "start_roll_pitch_integration_height": float(
                options["integration_start_height"]
            ),
            "force_landing_descending_rate": float(
                options["force_landing_descending_rate"]
            ),
        },
        "allocation_options": {
            "gimbal_dof": int(options["gimbal_dof"]),
            "gimbal_calc_in_fc": bool(
                options["gimbal_calc_in_fc"]
            ),
            "hovering_approximate": bool(
                options["hovering_approximate"]
            ),
            "underactuate": bool(options["underactuate"]),
            "gravity": gravity,
        },
        "geometry": _geometry_payload(snapshot),
    }


def _xy_control_mode(values: Sequence[int], name: str) -> int:
    modes = tuple(int(item) for item in values)
    if not modes:
        return -1
    if (
        len(modes) != 6
        or modes[0] != modes[1]
        or modes[2:] != (0, 0, 0, 0)
    ):
        raise ExactOracleProtocolError(
            "{} requires one shared X/Y mode and POSITION mode on "
            "axes Z/ROLL/PITCH/YAW".format(name)
        )
    return modes[0]


def _pose_state_payload(
    state: ControllerCoreState,
) -> Mapping[str, Any]:
    previous_xy = _xy_control_mode(
        state.previous_control_mode, "previous_control_mode"
    )
    return {
        "pid": [item.to_mapping() for item in state.pid],
        "start_roll_pitch_integration": (
            state.start_roll_pitch_integration
        ),
        "previous_stamp": state.previous_stamp,
        "previous_flight_state": state.previous_flight_state,
        "previous_xy_control_mode": previous_xy,
        "previous_force_landing": bool(
            state.previous_force_landing
        ),
        "has_previous_force_landing": (
            state.previous_force_landing is not None
        ),
        "pending_events": state.pending_events,
    }


def _allocation_state_payload(
    state: ControllerCoreState,
) -> Mapping[str, Any]:
    return {
        "target_gimbal_angles": list(
            state.target_gimbal_angles
        ),
        "target_roll": state.target_roll,
        "target_pitch": state.target_pitch,
    }


def _rpy(
    explicit: Any,
    orientation: Any,
    name: str,
) -> Tuple[float, ...]:
    if explicit is not None:
        return _finite_vector(explicit, name, 3)
    try:
        values = Rotation.from_matrix(
            np.asarray(orientation, dtype=float)
        ).as_euler("xyz")
    except ValueError as exc:
        raise ExactOracleProtocolError(
            "{} orientation cannot be converted to RPY".format(name)
        ) from exc
    return _finite_vector(values, name, 3)


def _tick_payload(
    snapshot: ControllerSnapshot,
    item: ControllerCoreInput,
) -> Mapping[str, Any]:
    integration = tuple(item.integration_enabled)
    if (
        integration[0:3] != (True, True, True)
        or integration[5] is not True
        or integration[3] != integration[4]
    ):
        raise ExactOracleProtocolError(
            "the PC replay protocol requires enabled X/Y/Z/YAW "
            "integration and one shared ROLL/PITCH integration state"
        )
    allocation_geometry = dict(
        deep_thaw(item.allocation_geometry)
    )
    if item.joint_positions and not allocation_geometry:
        raise ExactOracleProtocolError(
            "joint-dependent allocation requires explicit tick geometry"
        )
    result = {
        "stamp": item.stamp,
        "dt": item.dt,
        "position": list(item.position),
        "velocity": list(item.velocity),
        "rpy": list(
            _rpy(item.current_rpy, item.orientation, "current_rpy")
        ),
        "angular_velocity": list(item.angular_velocity),
        "target_position": list(item.target_position),
        "target_velocity": list(item.target_velocity),
        "target_acceleration": list(item.target_acceleration),
        "target_rpy": list(
            _rpy(
                item.target_rpy,
                item.target_orientation,
                "target_rpy",
            )
        ),
        "target_angular_velocity": list(
            item.target_angular_velocity
        ),
        "target_angular_acceleration": list(
            item.target_angular_acceleration
        ),
        "xy_control_mode": _xy_control_mode(
            item.control_mode, "control_mode"
        ),
        "control_mode": list(item.control_mode),
        "flight_state": item.flight_state,
        "force_landing": item.force_landing,
        "reset": item.reset,
        "initial_height": item.initial_height,
        "orientation": [list(row) for row in item.orientation],
        "integration_enabled": list(item.integration_enabled),
        "joint_positions": list(item.joint_positions),
    }
    if item.pid_config is not None:
        result["pid_config"] = [
            list(row) for row in item.pid_config
        ]
    if item.state_previous_stamp is not None:
        result["state_previous_stamp"] = (
            item.state_previous_stamp
        )
    if allocation_geometry:
        result["geometry"] = _geometry_payload(
            snapshot, allocation_geometry
        )
    return result


def build_exact_replay_payload(
    snapshot: ControllerSnapshot,
    initial_state: ControllerCoreState,
    inputs: Sequence[ControllerCoreInput],
    *,
    evidence_binding: Mapping[str, Any] = None,
) -> Mapping[str, Any]:
    """Build the canonical multi-tick PC replay request.

    The builder is shared by factual conformance evidence and the stateful
    inference adapter.  It rejects typed fields the PC core cannot represent,
    and checks that the recorded roll/pitch integration flags describe the
    imported state before every tick.
    """

    if not isinstance(snapshot, ControllerSnapshot):
        raise TypeError("snapshot must be ControllerSnapshot")
    if not isinstance(initial_state, ControllerCoreState):
        raise TypeError("initial_state must be ControllerCoreState")
    items = tuple(inputs)
    if not items or any(
        not isinstance(item, ControllerCoreInput) for item in items
    ):
        raise TypeError(
            "exact replay inputs must be non-empty typed controller inputs"
        )
    snapshot_payload = _snapshot_payload(snapshot)
    expected_gimbal_size = (
        len(snapshot_payload["geometry"]["rotor_origins_from_cog"])
        * int(snapshot_payload["allocation_options"]["gimbal_dof"])
    )
    if len(initial_state.target_gimbal_angles) != expected_gimbal_size:
        raise ExactOracleProtocolError(
            "initial allocation state must contain every gimbal angle"
        )
    integration_enabled = bool(
        initial_state.start_roll_pitch_integration
    )
    ticks = []
    threshold = float(
        snapshot.static_options["integration_start_height"]
    )
    for index, item in enumerate(items):
        # A live wrapper reset happens before the next controller tick is
        # captured.  Its ReplayFrame therefore carries the post-reset
        # integration gate (false) plus a pending RESET event.  Apply that
        # transition before checking the imported gate so a mid-episode reset
        # can be replayed from the preceding continuous state.
        if item.reset:
            integration_enabled = False
        if (
            item.integration_enabled[3] is not integration_enabled
            or item.integration_enabled[4] is not integration_enabled
        ):
            raise ExactOracleProtocolError(
                "tick {} ROLL/PITCH integration flags do not match "
                "the imported/replayed controller state".format(index)
            )
        ticks.append(_tick_payload(snapshot, item))
        if (
            not integration_enabled
            and item.position[2] - item.initial_height > threshold
        ):
            integration_enabled = True
    payload: Dict[str, Any] = {
        "snapshot": snapshot_payload,
        "jobs": [
            {
                "reset_before_first_tick": False,
                "initial_pose_state": _pose_state_payload(
                    initial_state
                ),
                "initial_allocation_state": (
                    _allocation_state_payload(initial_state)
                ),
                "ticks": ticks,
            }
        ],
    }
    if evidence_binding is not None:
        if not isinstance(evidence_binding, Mapping):
            raise TypeError("evidence_binding must be a mapping")
        payload["evidence_binding"] = deep_thaw(
            FrozenMapping(evidence_binding)
        )
    return payload


def _state_from_final(
    value: Mapping[str, Any],
) -> ControllerCoreState:
    try:
        pose = value["pose"]
        allocation = value["allocation"]
        pid = tuple(
            PidCoreState.from_mapping(item) for item in pose["pid"]
        )
        previous_xy = int(pose["previous_xy_control_mode"])
        previous_force = (
            bool(pose["previous_force_landing"])
            if pose["has_previous_force_landing"] is True
            else None
        )
        return ControllerCoreState(
            pid=pid,
            start_roll_pitch_integration=bool(
                pose["start_roll_pitch_integration"]
            ),
            previous_stamp=float(pose["previous_stamp"]),
            previous_flight_state=int(
                pose["previous_flight_state"]
            ),
            target_gimbal_angles=tuple(
                allocation["target_gimbal_angles"]
            ),
            target_roll=float(allocation["target_roll"]),
            target_pitch=float(allocation["target_pitch"]),
            previous_control_mode=(
                previous_xy,
                previous_xy,
                0,
                0,
                0,
                0,
            ),
            previous_force_landing=previous_force,
            pending_events=int(pose["pending_events"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExactOracleProtocolError(
            "oracle final controller state is incomplete"
        ) from exc


def _event_codes(value: Any) -> Tuple[int, ...]:
    scalar = np.asarray(value)
    if scalar.size != 1:
        raise ExactOracleProtocolError(
            "one-tick oracle reply must contain one event bitmask"
        )
    numeric = float(scalar.reshape(-1)[0])
    mask = int(numeric)
    if (
        not np.isfinite(numeric)
        or numeric != mask
        or mask < 0
    ):
        raise ExactOracleProtocolError(
            "oracle event bitmask must be a non-negative integer"
        )
    return tuple(
        1 << bit
        for bit in range(mask.bit_length())
        if mask & (1 << bit)
    )


class StatefulExactOracleControllerBackend:
    """State-continuous one-tick adapter for the C++ replay oracle.

    Each :meth:`step` submits exactly one replay job/tick, imports the prior
    reply's complete ``final_states`` payload, and maps the PC-side channels
    into the controller-domain output contract.  Unsupported fixture fields
    fail closed instead of being silently dropped.
    """

    def __init__(self, oracle: ExternalControllerOracle) -> None:
        legacy_identity = getattr(oracle, "identity", None)
        if isinstance(oracle, SubprocessExactControllerOracle):
            raise ExactOracleProtocolError(
                "the one-shot subprocess oracle is not permitted for "
                "closed-loop ticks; use the persistent or C-ABI transport"
            )
        if (
            getattr(oracle, "is_exact", None) is not True
            or not isinstance(legacy_identity, ExactOracleIdentity)
            or not callable(getattr(oracle, "replay", None))
        ):
            raise TypeError(
                "stateful adapter requires an exact external oracle"
            )
        self._oracle = oracle
        self._oracle_identity = legacy_identity
        self.identity = controller_backend_identity(legacy_identity)
        self._snapshot = None
        self._snapshot_request = None
        self._pose_state = None
        self._allocation_state = None
        self._expected_gimbal_state_size = None
        self._state = None

    @property
    def state(self) -> ControllerCoreState:
        if self._state is None:
            raise RuntimeError(
                "controller backend has not been reset"
            )
        return self._state

    @property
    def transport(self) -> ExternalControllerOracle:
        """Underlying shared transport, exposed for factory preflight."""

        return self._oracle

    def reset(
        self,
        snapshot: ControllerSnapshot,
        initial_state: ControllerCoreState,
    ) -> None:
        if not isinstance(snapshot, ControllerSnapshot):
            raise TypeError("snapshot must be ControllerSnapshot")
        if not isinstance(initial_state, ControllerCoreState):
            raise TypeError(
                "initial_state must be ControllerCoreState"
            )
        if (
            snapshot.backend_id != self.identity.backend_id
            or snapshot.source_commit != self.identity.source_commit
            or snapshot.artifact_sha256
            != self.identity.artifact_sha256
        ):
            raise ExactOracleProtocolError(
                "controller snapshot does not match exact oracle identity"
            )
        self._snapshot = snapshot
        self._snapshot_request = _snapshot_payload(snapshot)
        expected_gimbal_state_size = (
            len(
                self._snapshot_request["geometry"][
                    "rotor_origins_from_cog"
                ]
            )
            * int(
                self._snapshot_request["allocation_options"][
                    "gimbal_dof"
                ]
            )
        )
        if (
            len(initial_state.target_gimbal_angles)
            != expected_gimbal_state_size
        ):
            raise ExactOracleProtocolError(
                "initial allocation state must contain every gimbal angle"
            )
        self._expected_gimbal_state_size = expected_gimbal_state_size
        self._pose_state = _pose_state_payload(initial_state)
        self._allocation_state = _allocation_state_payload(
            initial_state
        )
        self._state = initial_state

    def step(self, item: ControllerCoreInput) -> ControllerCoreOutput:
        if not isinstance(item, ControllerCoreInput):
            raise TypeError("item must be ControllerCoreInput")
        if (
            self._snapshot is None
            or self._snapshot_request is None
            or self._pose_state is None
            or self._allocation_state is None
        ):
            raise RuntimeError(
                "controller backend must be reset before stepping"
            )
        expected_roll_pitch_integration = bool(
            self._state.start_roll_pitch_integration
        )
        if (
            item.integration_enabled[3]
            is not expected_roll_pitch_integration
            or item.integration_enabled[4]
            is not expected_roll_pitch_integration
        ):
            raise ExactOracleProtocolError(
                "ROLL/PITCH integration flags do not match imported "
                "controller state"
            )
        payload = {
            "snapshot": self._snapshot_request,
            "jobs": [
                {
                    "reset_before_first_tick": False,
                    "initial_pose_state": self._pose_state,
                    "initial_allocation_state": (
                        self._allocation_state
                    ),
                    "ticks": [
                        _tick_payload(self._snapshot, item)
                    ],
                }
            ],
        }
        reply = self._oracle.replay(payload)
        if (
            not isinstance(reply, ExactOracleReplayOutput)
            or reply.identity != self._oracle_identity
        ):
            raise ExactOracleProtocolError(
                "oracle returned the wrong replay identity"
            )
        required_channels = (
            "command_timestamp",
            "pid_terms",
            "four_axis_command",
            "vectoring_force",
            "gimbal_command",
            "torque_allocation_matrix_inverse",
            "allocation_internal",
            "job_tick",
        )
        missing = tuple(
            name
            for name in required_channels
            if name not in reply.continuous
        )
        if missing:
            raise ExactOracleProtocolError(
                "one-tick oracle reply lacks channels: {}".format(
                    ", ".join(missing)
                )
            )
        if len(reply.final_states) != 1:
            raise ExactOracleProtocolError(
                "one-tick oracle reply requires one final state"
            )
        pid = reply.continuous["pid_terms"]
        command_timestamp = reply.continuous["command_timestamp"]
        four_axis = reply.continuous["four_axis_command"]
        vectoring = reply.continuous["vectoring_force"]
        gimbal = reply.continuous["gimbal_command"]
        inverse_flat = reply.continuous[
            "torque_allocation_matrix_inverse"
        ]
        allocation_internal = reply.continuous[
            "allocation_internal"
        ]
        job_tick = reply.continuous["job_tick"]
        if (
            command_timestamp.shape != (1, 1)
            or not np.array_equal(
                command_timestamp[0],
                np.asarray((item.stamp,), dtype=float),
            )
            or pid.shape != (1, 24)
            or four_axis.shape[0] != 1
            or four_axis.shape[1] < 4
            or vectoring.shape[0] != 1
            or vectoring.shape[1] == 0
            or gimbal.shape[0] != 1
            or gimbal.shape[1] == 0
            or inverse_flat.shape[0] != 1
            or inverse_flat.shape[1] == 0
            or inverse_flat.shape[1] % 3 != 0
            or allocation_internal.shape != (1, 3)
            or job_tick.shape != (1, 2)
            or not np.array_equal(job_tick[0], (0.0, 0.0))
        ):
            raise ExactOracleProtocolError(
                "oracle one-tick continuous channel shapes are invalid"
            )
        final = deep_thaw(reply.final_states[0])
        state = _state_from_final(final)
        if (
            len(state.target_gimbal_angles)
            != self._expected_gimbal_state_size
        ):
            raise ExactOracleProtocolError(
                "oracle final allocation state has the wrong gimbal size"
            )
        self._pose_state = dict(final["pose"])
        self._allocation_state = dict(final["allocation"])
        self._state = state
        if item.pid_config is not None:
            names = (
                "p_gain",
                "i_gain",
                "d_gain",
                "limit_sum",
                "limit_p",
                "limit_i",
                "limit_d",
                "limit_error_p",
                "limit_error_i",
                "limit_error_d",
            )
            for index, row in enumerate(item.pid_config):
                for name, value in zip(names, row):
                    self._snapshot_request["pid"][index][
                        name
                    ] = value
        event_codes = _event_codes(reply.events)
        effective_acceleration = list(item.target_acceleration)
        if item.force_landing:
            effective_acceleration[2] = 0.0
        return ControllerCoreOutput(
            pid_result=pid[0, 0:6],
            pid_p_term=pid[0, 6:12],
            pid_i_term=pid[0, 12:18],
            pid_d_term=pid[0, 18:24],
            target_vectoring_force=vectoring[0],
            base_thrust=four_axis[0, 3:],
            gimbal_angle=gimbal[0],
            torque_allocation_matrix_inverse=(
                inverse_flat[0].reshape((-1, 3))
            ),
            target_roll=allocation_internal[0, 0],
            target_pitch=allocation_internal[0, 1],
            candidate_yaw_term=allocation_internal[0, 2],
            events=event_codes,
            stamp=item.stamp,
            saturated=32 in event_codes,
            four_axis_angles=four_axis[0, 0:3],
            generalized_wrench=None,
            effective_target_acceleration=effective_acceleration,
        )


def batched_exact_controller_backend_factory(
    backend_factory: Any,
    *,
    max_batch_size: int,
    batch_wait_s: float,
) -> Any:
    """Wrap fresh stateful adapters around one shared batching transport."""

    if not callable(backend_factory):
        raise TypeError("backend_factory must be callable")
    batch_size = int(max_batch_size)
    wait = float(batch_wait_s)
    if batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    if not np.isfinite(wait) or wait < 0.0:
        raise ValueError(
            "batch_wait_s must be finite and non-negative"
        )
    if batch_size == 1:
        return backend_factory

    lock = threading.Lock()
    batching_oracle = None

    def factory() -> StatefulExactOracleControllerBackend:
        nonlocal batching_oracle
        backend = backend_factory()
        if not isinstance(
            backend, StatefulExactOracleControllerBackend
        ):
            raise TypeError(
                "exact replay batching requires stateful oracle adapters"
            )
        transport = backend.transport
        with lock:
            if batching_oracle is None:
                batching_oracle = BatchingExactControllerOracle(
                    transport,
                    max_batch_size=batch_size,
                    batch_wait_s=wait,
                )
            elif batching_oracle.underlying is not transport:
                raise ExactOracleProtocolError(
                    "batched stateful adapters must share one oracle transport"
                )
            shared = batching_oracle
        return StatefulExactOracleControllerBackend(shared)

    return factory


__all__ = [
    "BatchingExactControllerOracle",
    "CtypesExactControllerOracle",
    "EXACT_ORACLE_PROTOCOL",
    "ExactOracleError",
    "ExactOracleIdentity",
    "ExactOracleProtocolError",
    "ExactOracleReplayOutput",
    "ExactOracleUnavailable",
    "ExternalControllerOracle",
    "PersistentSubprocessExactControllerOracle",
    "StatefulExactOracleControllerBackend",
    "SubprocessExactControllerOracle",
    "batched_exact_controller_backend_factory",
    "build_exact_replay_payload",
    "controller_backend_identity",
]
