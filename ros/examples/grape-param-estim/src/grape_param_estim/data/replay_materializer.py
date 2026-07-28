"""Materialize future controller replay frames into exact, gated inputs.

The live replay messages are ROS transport objects.  This module consumes a
canonical, content-addressed JSON extraction of those messages so that the
conversion and conformance step is testable without a ROS master or rosbag
Python bindings.  No controller input, state, nominal model, or recorded
output is synthesized when it is absent.
"""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from grape_param_estim.alternative_backends import (
    EXACT_ORACLE_PROTOCOL,
    PC_EXACT_ORACLE_CAPABILITIES,
    ExactOracleConformanceFixture,
    ExactOracleFixtureProvenance,
    ExactOracleIdentity,
    PersistentSubprocessExactControllerOracle,
    evaluate_exact_oracle_conformance,
)
from grape_param_estim.controller.contracts import (
    ControllerCoreInput,
    ControllerCoreState,
    PidCoreState,
)
from grape_param_estim.controller.exact_inputs import (
    EXACT_EPISODE_CONTROLLER_INPUT_SCHEMA,
    EXACT_EPISODE_REQUEST_BINDING_SCHEMA,
    ExactEpisodeConformanceBundle,
    ExactEpisodeConformanceEvidence,
    FIXTURE_BUNDLE_SCHEMA,
    SNAPSHOT_BUNDLE_SCHEMA,
    STATE_BUNDLE_SCHEMA,
)
from grape_param_estim.controller.external_oracle import (
    build_exact_replay_payload,
)
from grape_param_estim.controller.snapshot import (
    STATIC_OPTION_FIELDS,
    ControllerSnapshot,
)
from grape_param_estim.data.controller_fixture import (
    CONTROLLER_REPLAY_FIXTURE_SCHEMA,
    ControllerReplayFixture,
    EpisodeTimeGrids,
)
from grape_param_estim.data.bag_reader import (
    read_bag_topic_inventory,
)
from grape_param_estim.data.event_scheduler import EventGrid
from grape_param_estim.episode import stable_hash


CANONICAL_REPLAY_STREAM_SCHEMA = (
    "grape.controller-replay-canonical-stream/v1"
)
EXACT_REQUEST_BUNDLE_SCHEMA = "grape.exact-replay-request-bundle/v1"
EXACT_CONFORMANCE_FIXTURE_BUNDLE_SCHEMA = (
    "grape.exact-conformance-fixture-bundle/v1"
)
MATERIALIZER_POLICY_SCHEMA = "grape.replay-materializer-policy/v1"
NOMINAL_GEOMETRY_SCHEMA = (
    "grape.gimbalrotor-allocation-geometry/v1"
)
REPLAY_METADATA_SCHEMA = (
    "grape/gimbalrotor_controller_replay_metadata/v1"
)
REPLAY_FRAME_SCHEMA = "grape/gimbalrotor_controller_replay_frame/v1"
REPLAY_METADATA_TOPIC = "/gimbalrotor/controller_replay/metadata"
REPLAY_FRAME_TOPIC = "/gimbalrotor/controller_replay/frame"
REPLAY_METADATA_TYPE = (
    "grape_param_estim/GimbalrotorControllerReplayMetadata"
)
REPLAY_FRAME_TYPE = "grape_param_estim/GimbalrotorControllerReplayFrame"

MATERIALIZED_EXACT_FILES = (
    "controller_replay_fixture_bundle.json",
    "controller_snapshot_bundle.json",
    "controller_state_bundle.json",
    "exact_replay_request_bundle.json",
    "exact_conformance_fixture_bundle.json",
    "exact_episode_conformance_bundle.json",
)

_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
_GAIN_FIELDS = ("p_gain", "i_gain", "d_gain")
_LIMIT_FIELDS = (
    "limit_sum",
    "limit_p",
    "limit_i",
    "limit_d",
    "limit_err_p",
    "limit_err_i",
    "limit_err_d",
)
_PID_CONFIG_FIELDS = _GAIN_FIELDS + _LIMIT_FIELDS
_GRID_FIELDS = (
    ("controller_tick_grid", "controller_tick"),
    ("plant_integration_grid", "plant_integration"),
    ("observation_grid", "observation"),
    ("likelihood_grid", "likelihood"),
    ("report_grid", "report"),
)
_RESET_EVENT = 1
_SATURATED_EVENT = 1 << 5


def _sha256(value: Any, name: str) -> str:
    digest = str(value)
    if (
        len(digest) != 64
        or digest != digest.lower()
        or any(item not in "0123456789abcdef" for item in digest)
    ):
        raise ValueError("{} must be a lowercase SHA-256".format(name))
    return digest


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("{} must be a mapping".format(name))
    return value


def _sequence(value: Any, name: str) -> Tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(
        value, SequenceABC
    ):
        raise TypeError("{} must be an array".format(name))
    return tuple(value)


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError("{} must be an integer".format(name))
    result = int(value)
    if float(value) != result:
        raise ValueError("{} must be an integer".format(name))
    return result


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError("{} must be a boolean".format(name))
    return value


def _vector(value: Any, size: int, name: str) -> Tuple[float, ...]:
    if isinstance(value, Mapping):
        names = ("x", "y", "z", "w")[:size]
        if set(value) != set(names):
            raise ValueError(
                "{} mapping must contain {}".format(name, ", ".join(names))
            )
        items = tuple(value[item] for item in names)
    else:
        items = _sequence(value, name)
    if len(items) != size:
        raise ValueError("{} must contain {} values".format(name, size))
    return tuple(_finite(item, name) for item in items)


def _quaternion_matrix(value: Any, name: str) -> Tuple[Tuple[float, ...], ...]:
    quaternion = np.asarray(_vector(value, 4, name), dtype=float)
    if not np.isclose(
        np.linalg.norm(quaternion), 1.0, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError("{} must be a normalized quaternion".format(name))
    matrix = Rotation.from_quat(quaternion).as_matrix()
    return tuple(tuple(float(item) for item in row) for row in matrix)


def _strings(value: Any, name: str) -> Tuple[str, ...]:
    items = tuple(str(item) for item in _sequence(value, name))
    if any(not item for item in items):
        raise ValueError("{} may not contain empty values".format(name))
    return items


def _content_hashed_payload(
    schema: str, field: str, values: Sequence[Any]
) -> Mapping[str, Any]:
    payload = {"schema": schema, field: list(values)}
    payload["content_sha256"] = stable_hash(payload)
    return payload


def _parse_scalar(value: Any) -> Any:
    text = str(value).strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        integer = int(text)
        if str(integer) == text or (
            text.startswith("+") and str(integer) == text[1:]
        ):
            return integer
    except ValueError:
        pass
    try:
        numeric = float(text)
    except ValueError:
        return text
    if not np.isfinite(numeric):
        raise ValueError("metadata options must be finite")
    return numeric


def _time_seconds(value: Any, name: str) -> float:
    converter = getattr(value, "to_sec", None)
    return _finite(
        converter() if callable(converter) else value,
        name,
    )


def _header_record_timing(
    message: Any,
    *,
    bag_start_time_s: float,
    record_time_s: Any,
    allow_header_before_origin: bool = False,
) -> Mapping[str, Any]:
    header = getattr(message, "header", None)
    if header is None or not hasattr(header, "stamp"):
        raise ValueError("replay message requires a Header timestamp")
    header_time = _time_seconds(header.stamp, "header stamp")
    record_time = _time_seconds(record_time_s, "bag record stamp")
    bag_start = _finite(bag_start_time_s, "bag start record stamp")
    if record_time < bag_start - 1.0e-9:
        raise ValueError(
            "replay bag record stamp predates the bag record-time origin"
        )
    if header_time <= 0.0:
        raise ValueError(
            "replay event time must be a positive Header timestamp; "
            "bag-record-time fallback is forbidden"
        )
    relative_ns = int(round(1.0e9 * (header_time - bag_start)))
    if relative_ns < 0 and not allow_header_before_origin:
        raise ValueError(
            "replay Header timestamp predates the bag record-time origin"
        )
    relative = float(relative_ns) / 1.0e9
    return {
        "stamp": relative,
        "relative_event_time_ns": relative_ns,
        "header_time_s": header_time,
        "bag_record_time_s": record_time,
        "bag_start_record_time_s": bag_start,
        "event_time_source": "message_header_stamp",
        "header_predates_bag_origin": relative_ns < 0,
        "frame_id": str(getattr(header, "frame_id", "")),
    }


def _message_vector(value: Any, size: int, name: str) -> list:
    names = ("x", "y", "z", "w")[:size]
    try:
        return [
            _finite(getattr(value, field), name) for field in names
        ]
    except AttributeError as exc:
        raise ValueError(
            "{} is not a {}-component ROS geometry value".format(
                name, size
            )
        ) from exc


def _message_array(message: Any, name: str) -> list:
    if not hasattr(message, name):
        raise ValueError("replay message lacks {}".format(name))
    return list(getattr(message, name))


def replay_metadata_message_to_mapping(
    message: Any,
    *,
    bag_start_time_s: float,
    record_time_s: Any,
) -> Mapping[str, Any]:
    """Convert one ROS replay-metadata message without losing time origin."""

    declared_type = getattr(message, "_type", REPLAY_METADATA_TYPE)
    if declared_type != REPLAY_METADATA_TYPE:
        raise TypeError(
            "metadata topic has unexpected message type {}".format(
                declared_type
            )
        )
    result = dict(
        _header_record_timing(
            message,
            bag_start_time_s=bag_start_time_s,
            record_time_s=record_time_s,
            allow_header_before_origin=True,
        )
    )
    scalar_names = (
        "schema",
        "source_commit",
        "backend_id",
        "fidelity",
        "controller_artifact_sha256",
        "controller_snapshot_sha256",
        "nominal_model_sha256",
        "parameter_dump_sha256",
        "nominal_geometry_sha256",
        "controller_rate_hz",
        "nominal_mass",
    )
    for name in scalar_names:
        if not hasattr(message, name):
            raise ValueError(
                "replay metadata lacks {}".format(name)
            )
        result[name] = getattr(message, name)
    for name in (
        "gain_names",
        "gain_values",
        "limit_names",
        "limit_values",
        "static_option_names",
        "static_option_values",
        "motor_order",
        "nominal_inertia",
        "geometry_names",
        "geometry_offsets",
        "geometry_values",
    ):
        result[name] = _message_array(message, name)
    result["nominal_cog"] = _message_vector(
        message.nominal_cog, 3, "nominal_cog"
    )
    return result


def _relative_state_stamp(value: Any, bag_start_time_s: float) -> float:
    stamp = _finite(value, "controller state previous_stamp")
    if stamp == 0.0:
        return 0.0
    relative_ns = int(round(1.0e9 * (stamp - bag_start_time_s)))
    if relative_ns < 0:
        raise ValueError(
            "controller state timestamp does not use the replay Header/"
            "bag-record time origin"
        )
    return float(relative_ns) / 1.0e9


def replay_frame_message_to_mapping(
    message: Any,
    *,
    bag_start_time_s: float,
    record_time_s: Any,
) -> Mapping[str, Any]:
    """Convert one ROS replay frame; Header time is the sole tick time."""

    declared_type = getattr(message, "_type", REPLAY_FRAME_TYPE)
    if declared_type != REPLAY_FRAME_TYPE:
        raise TypeError(
            "frame topic has unexpected message type {}".format(
                declared_type
            )
        )
    result = dict(
        _header_record_timing(
            message,
            bag_start_time_s=bag_start_time_s,
            record_time_s=record_time_s,
        )
    )
    scalar_names = (
        "schema",
        "controller_snapshot_sha256",
        "dt",
        "flight_state",
        "force_landing",
        "reset",
        "initial_height",
        "start_roll_pitch_integration_before",
        "start_roll_pitch_integration_after",
        "previous_flight_state_before",
        "previous_flight_state_after",
        "previous_force_landing_before",
        "previous_force_landing_after",
        "has_previous_force_landing_before",
        "has_previous_force_landing_after",
        "pending_events_before",
        "pending_events_after",
        "target_roll_before",
        "target_roll_after",
        "target_pitch_before",
        "target_pitch_after",
        "torque_allocation_rows",
        "torque_allocation_columns",
        "target_roll",
        "target_pitch",
        "candidate_yaw_term",
        "saturated",
    )
    for name in scalar_names:
        if not hasattr(message, name):
            raise ValueError("replay frame lacks {}".format(name))
        result[name] = getattr(message, name)
    for name in (
        "control_mode",
        "integration_enabled",
        "joint_names",
        "joint_positions",
        "allocation_geometry_names",
        "allocation_geometry_offsets",
        "allocation_geometry_values",
        "pid_state_before",
        "pid_state_after",
        "previous_control_mode_before",
        "previous_control_mode_after",
        "target_gimbal_angles_before",
        "target_gimbal_angles_after",
        "pid_result",
        "pid_p_term",
        "pid_i_term",
        "pid_d_term",
        "target_vectoring_force",
        "four_axis_angles",
        "base_thrust",
        "gimbal_angle",
        "torque_allocation_matrix_inverse",
        "events",
    ):
        result[name] = _message_array(message, name)
    for name in (
        "position",
        "velocity",
        "angular_velocity",
        "target_position",
        "target_velocity",
        "target_acceleration",
        "target_rpy",
        "target_angular_velocity",
        "target_angular_acceleration",
        "current_rpy",
        "effective_target_acceleration",
    ):
        result[name] = _message_vector(
            getattr(message, name), 3, name
        )
    for name in ("orientation", "target_orientation"):
        result[name] = _message_vector(
            getattr(message, name), 4, name
        )
    for suffix in ("before", "after"):
        result["previous_stamp_{}".format(suffix)] = (
            _relative_state_stamp(
                getattr(
                    message,
                    "previous_stamp_{}".format(suffix),
                ),
                bag_start_time_s,
            )
        )
    return result


def _named_values(
    names: Any, values: Any, expected: Sequence[str], label: str
) -> Mapping[str, Tuple[float, ...]]:
    keys = _strings(names, "{} names".format(label))
    items = tuple(
        _finite(item, "{} values".format(label))
        for item in _sequence(values, "{} values".format(label))
    )
    if len(keys) != len(items) or len(set(keys)) != len(keys):
        raise ValueError("{} names and values must align uniquely".format(label))
    parsed: Dict[str, Dict[str, float]] = {
        field: {} for field in expected
    }
    for key, item in zip(keys, items):
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError(
                "{} name must be axis.field: {}".format(label, key)
            )
        axis, field = parts
        if axis not in _AXES or field not in parsed:
            raise ValueError(
                "unsupported {} entry: {}".format(label, key)
            )
        if axis in parsed[field]:
            raise ValueError("{} repeats {}".format(label, key))
        parsed[field][axis] = item
    missing = tuple(
        "{}.{}".format(axis, field)
        for field in expected
        for axis in _AXES
        if axis not in parsed[field]
    )
    if missing:
        raise ValueError(
            "{} lacks {}".format(label, ", ".join(missing))
        )
    return MappingProxyType(
        {
            field: tuple(parsed[field][axis] for axis in _AXES)
            for field in expected
        }
    )


def _static_options(names: Any, values: Any) -> Mapping[str, Any]:
    keys = _strings(names, "static option names")
    items = _sequence(values, "static option values")
    if len(keys) != len(items) or len(set(keys)) != len(keys):
        raise ValueError("static option names and values must align uniquely")
    result = {
        key: _parse_scalar(value) for key, value in zip(keys, items)
    }
    aliases = (
        "start_roll_pitch_integration_height",
        "start_rp_integration_height",
    )
    present = tuple(name for name in aliases if name in result)
    if "integration_start_height" not in result and len(present) == 1:
        result["integration_start_height"] = result[present[0]]
    elif "integration_start_height" not in result and len(present) > 1:
        values = {result[name] for name in present}
        if len(values) != 1:
            raise ValueError(
                "integration-height metadata aliases disagree"
            )
        result["integration_start_height"] = next(iter(values))
    missing = tuple(
        name for name in STATIC_OPTION_FIELDS if name not in result
    )
    if missing:
        raise ValueError(
            "metadata lacks controller-used static options: {}".format(
                ", ".join(missing)
            )
        )
    typed = {
        "gimbal_dof": _integer(result["gimbal_dof"], "gimbal_dof"),
        "gimbal_calc_in_fc": _boolean(
            result["gimbal_calc_in_fc"], "gimbal_calc_in_fc"
        ),
        "hovering_approximate": _boolean(
            result["hovering_approximate"], "hovering_approximate"
        ),
        "underactuate": _boolean(
            result["underactuate"], "underactuate"
        ),
        "need_yaw_d_control": _boolean(
            result["need_yaw_d_control"], "need_yaw_d_control"
        ),
        "integration_start_height": _finite(
            result["integration_start_height"],
            "integration_start_height",
        ),
        "force_landing_descending_rate": _finite(
            result["force_landing_descending_rate"],
            "force_landing_descending_rate",
        ),
        "estimate_mode": _integer(
            result["estimate_mode"], "estimate_mode"
        ),
    }
    for key, value in result.items():
        if key not in aliases and key not in typed:
            typed[key] = value
    return MappingProxyType(typed)


def _flattened_geometry(
    names: Any, offsets: Any, values: Any, label: str
) -> Mapping[str, Any]:
    keys = _strings(names, "{} names".format(label))
    starts = tuple(
        _integer(item, "{} offsets".format(label))
        for item in _sequence(offsets, "{} offsets".format(label))
    )
    data = tuple(
        _finite(item, "{} values".format(label))
        for item in _sequence(values, "{} values".format(label))
    )
    if (
        not keys
        or len(keys) != len(starts)
        or len(set(keys)) != len(keys)
        or starts[0] != 0
        or any(left >= right for left, right in zip(starts, starts[1:]))
        or starts[-1] >= len(data)
    ):
        raise ValueError("{} names/offsets are not canonical".format(label))
    pieces = {}
    ends = starts[1:] + (len(data),)
    for key, start, end in zip(keys, starts, ends):
        pieces[key] = data[start:end]
    required = {
        "mass",
        "inertia_row_major",
        "rotor_origins_from_cog_xyz",
        "rotor_directions",
        "moment_force_rate",
        "thrust_coordinate_rotations_row_major",
    }
    if set(pieces) != required:
        raise ValueError(
            "{} fields must be exactly {}".format(
                label, ", ".join(sorted(required))
            )
        )
    directions = tuple(
        _integer(item, "rotor direction")
        for item in pieces["rotor_directions"]
    )
    rotor_count = len(directions)
    if (
        rotor_count < 1
        or len(pieces["mass"]) != 1
        or len(pieces["moment_force_rate"]) != 1
        or len(pieces["inertia_row_major"]) != 9
        or len(pieces["rotor_origins_from_cog_xyz"]) != 3 * rotor_count
        or len(pieces["thrust_coordinate_rotations_row_major"])
        != 9 * rotor_count
    ):
        raise ValueError("{} dimensions do not align".format(label))
    origins = np.asarray(
        pieces["rotor_origins_from_cog_xyz"], dtype=float
    ).reshape(rotor_count, 3)
    rotations = np.asarray(
        pieces["thrust_coordinate_rotations_row_major"], dtype=float
    ).reshape(rotor_count, 3, 3)
    inertia = np.asarray(
        pieces["inertia_row_major"], dtype=float
    ).reshape(3, 3)
    return {
        "mass": pieces["mass"][0],
        "inertia": inertia.tolist(),
        "moment_force_rate": pieces["moment_force_rate"][0],
        "rotor_origins_from_cog": origins.tolist(),
        "rotor_directions": list(directions),
        "thrust_coordinate_rotations": rotations.tolist(),
    }


def _metadata_snapshot(
    value: Any,
) -> Tuple[
    float,
    str,
    str,
    ControllerSnapshot,
    Tuple[str, ...],
]:
    item = _mapping(value, "metadata record")
    if item.get("schema") != REPLAY_METADATA_SCHEMA:
        raise ValueError("unsupported replay metadata schema")
    stamp = _finite(item.get("stamp"), "metadata stamp")
    frame_id = str(item.get("frame_id", "")).strip()
    if not frame_id:
        raise ValueError("metadata frame_id is required")
    if item.get("fidelity") != "pc_exact":
        raise ValueError("materializer requires pc_exact metadata")
    gains = dict(
        _named_values(
            item.get("gain_names"),
            item.get("gain_values"),
            _GAIN_FIELDS,
            "gains",
        )
    )
    gains["axis_names"] = _AXES
    limits = dict(
        _named_values(
            item.get("limit_names"),
            item.get("limit_values"),
            _LIMIT_FIELDS,
            "limits",
        )
    )
    options = dict(
        _static_options(
            item.get("static_option_names"),
            item.get("static_option_values"),
        )
    )
    geometry = dict(
        _flattened_geometry(
            item.get("geometry_names"),
            item.get("geometry_offsets"),
            item.get("geometry_values"),
            "nominal geometry",
        )
    )
    if "gravity" in options:
        geometry["gravity"] = _finite(options["gravity"], "gravity")
    mass = _finite(item.get("nominal_mass"), "nominal_mass")
    cog = _vector(item.get("nominal_cog"), 3, "nominal_cog")
    inertia_flat = _vector(
        item.get("nominal_inertia"), 9, "nominal_inertia"
    )
    inertia = tuple(
        tuple(inertia_flat[3 * row + column] for column in range(3))
        for row in range(3)
    )
    if (
        not np.isclose(mass, geometry["mass"], rtol=0.0, atol=0.0)
        or not np.array_equal(
            np.asarray(inertia), np.asarray(geometry["inertia"])
        )
    ):
        raise ValueError(
            "metadata nominal model and allocation geometry disagree"
        )
    motor_order = _strings(
        item.get("motor_order"), "metadata motor_order"
    )
    if (
        len(motor_order) != len(geometry["rotor_directions"])
        or len(set(motor_order)) != len(motor_order)
    ):
        raise ValueError(
            "metadata motor_order must uniquely name every allocation rotor"
        )
    snapshot = ControllerSnapshot(
        backend_id=str(item.get("backend_id", "")),
        source_commit=str(item.get("source_commit", "")),
        artifact_sha256=_sha256(
            item.get("controller_artifact_sha256"),
            "controller_artifact_sha256",
        ),
        nominal_model_sha256=_sha256(
            item.get("nominal_model_sha256"),
            "nominal_model_sha256",
        ),
        parameter_dump_sha256=_sha256(
            item.get("parameter_dump_sha256"),
            "parameter_dump_sha256",
        ),
        controller_rate_hz=_finite(
            item.get("controller_rate_hz"), "controller_rate_hz"
        ),
        gains=gains,
        limits=limits,
        static_options=options,
        nominal_mass=mass,
        nominal_cog=cog,
        nominal_inertia=inertia,
        nominal_geometry=geometry,
    )
    declared = _sha256(
        item.get("controller_snapshot_sha256"),
        "controller_snapshot_sha256",
    )
    if snapshot.snapshot_id != declared:
        raise ValueError(
            "recorded controller snapshot hash does not match its content"
        )
    declared_geometry_sha256 = _sha256(
        item.get("nominal_geometry_sha256"),
        "nominal_geometry_sha256",
    )
    computed_geometry_sha256 = stable_hash(
        {
            "schema": NOMINAL_GEOMETRY_SCHEMA,
            "geometry": geometry,
        }
    )
    if declared_geometry_sha256 != computed_geometry_sha256:
        raise ValueError(
            "recorded nominal geometry hash does not match canonical "
            "allocation geometry content"
        )
    return stamp, frame_id, declared, snapshot, motor_order


def _pid_state(values: Any, label: str) -> Tuple[PidCoreState, ...]:
    flat = _vector(values, 48, label)
    result = []
    for axis in range(6):
        row = flat[8 * axis : 8 * axis + 8]
        result.append(
            PidCoreState(
                error_p=row[0],
                error_i=row[1],
                previous_error_i=row[2],
                error_d=row[3],
                result=row[4],
                p_term=row[5],
                i_term=row[6],
                d_term=row[7],
            )
        )
    return tuple(result)


def _event_mask(values: Any, label: str) -> int:
    mask = 0
    for value in _sequence(values, label):
        event = _integer(value, label)
        if event <= 0 or event > 0x80000000 or event & (event - 1):
            raise ValueError(
                "{} must contain positive uint32 bit values".format(label)
            )
        if mask & event:
            raise ValueError("{} repeats an event bit".format(label))
        mask |= event
    return mask


def _frame_state(
    frame: Mapping[str, Any], suffix: str
) -> ControllerCoreState:
    has_force = _boolean(
        frame.get("has_previous_force_landing_{}".format(suffix)),
        "has_previous_force_landing_{}".format(suffix),
    )
    previous_force = (
        _boolean(
            frame.get("previous_force_landing_{}".format(suffix)),
            "previous_force_landing_{}".format(suffix),
        )
        if has_force
        else None
    )
    return ControllerCoreState(
        pid=_pid_state(
            frame.get("pid_state_{}".format(suffix)),
            "pid_state_{}".format(suffix),
        ),
        start_roll_pitch_integration=_boolean(
            frame.get(
                "start_roll_pitch_integration_{}".format(suffix)
            ),
            "start_roll_pitch_integration_{}".format(suffix),
        ),
        previous_stamp=_finite(
            frame.get("previous_stamp_{}".format(suffix)),
            "previous_stamp_{}".format(suffix),
        ),
        previous_flight_state=_integer(
            frame.get("previous_flight_state_{}".format(suffix)),
            "previous_flight_state_{}".format(suffix),
        ),
        target_gimbal_angles=tuple(
            _finite(item, "target_gimbal_angles_{}".format(suffix))
            for item in _sequence(
                frame.get("target_gimbal_angles_{}".format(suffix)),
                "target_gimbal_angles_{}".format(suffix),
            )
        ),
        target_roll=_finite(
            frame.get("target_roll_{}".format(suffix)),
            "target_roll_{}".format(suffix),
        ),
        target_pitch=_finite(
            frame.get("target_pitch_{}".format(suffix)),
            "target_pitch_{}".format(suffix),
        ),
        previous_control_mode=tuple(
            _integer(item, "previous_control_mode_{}".format(suffix))
            for item in _sequence(
                frame.get("previous_control_mode_{}".format(suffix)),
                "previous_control_mode_{}".format(suffix),
            )
        ),
        previous_force_landing=previous_force,
        pending_events=_integer(
            frame.get("pending_events_{}".format(suffix)),
            "pending_events_{}".format(suffix),
        ),
    )


@dataclass(frozen=True)
class _MaterializedEpisode:
    fixture: ControllerReplayFixture
    snapshot: ControllerSnapshot
    initial_state: ControllerCoreState
    recorded_continuous: Mapping[str, np.ndarray]
    recorded_events: np.ndarray
    source_topics: Tuple[str, ...]
    frame_conventions: Mapping[str, str]
    unit_conventions: Mapping[str, str]
    motor_order: Tuple[str, ...]
    interval_start_time_ns: int
    interval_end_time_ns: int


def _grids(value: Any) -> EpisodeTimeGrids:
    source = _mapping(value, "episode grids")
    if set(source) != {item[0] for item in _GRID_FIELDS}:
        raise ValueError("canonical stream must contain exactly five grids")
    grids = []
    for field, name in _GRID_FIELDS:
        grids.append(
            EventGrid(
                name,
                tuple(
                    _finite(item, field)
                    for item in _sequence(source[field], field)
                ),
            )
        )
    return EpisodeTimeGrids(*grids)


def extract_canonical_replay_stream(
    bag_root: Any,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Read future replay topics directly from every configured ROS bag.

    Controller event time is always the message Header stamp expressed
    relative to the bag record-time origin.  Bag record timestamps are kept
    beside it for provenance and are never substituted for controller time.
    """

    try:
        import rosbag
    except ImportError as exc:  # pragma: no cover - ROS runtime boundary
        raise RuntimeError(
            "direct replay extraction requires ROS 1 rosbag"
        ) from exc
    if not isinstance(config, Mapping):
        raise TypeError("assimilation config must be a mapping")
    configured = _sequence(config.get("episodes"), "config episodes")
    if not configured:
        raise ValueError("assimilation config requires episodes")
    root = Path(bag_root).expanduser().resolve()
    episodes = []
    for raw in configured:
        episode_config = _mapping(raw, "configured episode")
        episode_id = str(
            episode_config.get("episode_id", "")
        ).strip()
        if not episode_id:
            raise ValueError("configured episode_id is required")
        bag_path = (
            root / str(episode_config.get("bag", ""))
        ).resolve()
        expected_source = _sha256(
            episode_config.get("source_bag_sha256"),
            "{} source_bag_sha256".format(episode_id),
        )
        inventory = read_bag_topic_inventory(
            bag_path, source_bag_sha256=expected_source
        )
        for topic, expected_type in (
            (REPLAY_METADATA_TOPIC, REPLAY_METADATA_TYPE),
            (REPLAY_FRAME_TOPIC, REPLAY_FRAME_TYPE),
        ):
            entry = inventory.topics.get(topic)
            if (
                entry is None
                or entry.message_type != expected_type
                or entry.message_count <= 0
            ):
                raise ValueError(
                    "{} lacks non-empty {} ({})".format(
                        episode_id, topic, expected_type
                    )
                )
        metadata = []
        frames = []
        with rosbag.Bag(str(bag_path), "r") as bag:
            bag_start = _finite(
                bag.get_start_time(), "bag start record time"
            )
            for topic, message, record_time in bag.read_messages(
                topics=(REPLAY_METADATA_TOPIC, REPLAY_FRAME_TOPIC)
            ):
                if topic == REPLAY_METADATA_TOPIC:
                    metadata.append(
                        replay_metadata_message_to_mapping(
                            message,
                            bag_start_time_s=bag_start,
                            record_time_s=record_time,
                        )
                    )
                elif topic == REPLAY_FRAME_TOPIC:
                    frames.append(
                        replay_frame_message_to_mapping(
                            message,
                            bag_start_time_s=bag_start,
                            record_time_s=record_time,
                        )
                    )
        replay_start = _finite(
            episode_config.get("replay_start_offset_s"),
            "replay_start_offset_s",
        )
        score_start = _finite(
            episode_config.get("score_start_offset_s"),
            "score_start_offset_s",
        )
        score_end = _finite(
            episode_config.get("score_end_offset_s"),
            "score_end_offset_s",
        )
        ordered_frames = tuple(
            sorted(frames, key=lambda item: float(item["stamp"]))
        )
        ordered_frame_times = np.asarray(
            [float(item["stamp"]) for item in ordered_frames],
            dtype=float,
        )
        before_replay = np.flatnonzero(
            ordered_frame_times <= replay_start
        )
        after_score = np.flatnonzero(
            ordered_frame_times >= score_end
        )
        if before_replay.size == 0 or after_score.size == 0:
            raise ValueError(
                "{} ReplayFrame events do not bracket the configured "
                "replay/score interval".format(episode_id)
            )
        first_index = int(before_replay[-1])
        last_index = int(after_score[0])
        selected_frames = ordered_frames[
            first_index : last_index + 1
        ]
        if len(selected_frames) < 2:
            raise ValueError(
                "{} has fewer than two bracketing ReplayFrame events".format(
                    episode_id
                )
            )
        first_frame = float(selected_frames[0]["stamp"])
        last_frame = float(selected_frames[-1]["stamp"])
        ordered_metadata = tuple(
            sorted(metadata, key=lambda item: float(item["stamp"]))
        )
        before = tuple(
            item
            for item in ordered_metadata
            if float(item["stamp"]) <= first_frame
        )
        if not before:
            raise ValueError(
                "{} lacks ReplayMetadata at or before its first "
                "ReplayFrame".format(episode_id)
            )
        selected_metadata = (before[-1],) + tuple(
            item
            for item in ordered_metadata
            if first_frame < float(item["stamp"]) <= last_frame
        )
        motor_orders = {
            tuple(str(value) for value in item["motor_order"])
            for item in selected_metadata
        }
        if len(motor_orders) != 1:
            raise ValueError(
                "{} changes motor_order inside one episode".format(
                    episode_id
                )
            )
        frame_ids = {
            str(item["frame_id"]) for item in selected_frames
        }
        if len(frame_ids) != 1 or not next(iter(frame_ids)):
            raise ValueError(
                "{} ReplayFrame Header frame_id is not constant/explicit".format(
                    episode_id
                )
            )
        episodes.append(
            {
                "episode_id": episode_id,
                "source_bag_sha256": inventory.source_bag_sha256,
                "topic_inventory_sha256": (
                    inventory.inventory_sha256
                ),
                "replay_start_offset_s": replay_start,
                "score_start_offset_s": score_start,
                "score_end_offset_s": score_end,
                "metadata_records": list(selected_metadata),
                "frames": list(selected_frames),
                "source_topics": [
                    REPLAY_METADATA_TOPIC,
                    REPLAY_FRAME_TOPIC,
                ],
                "frame_conventions": {
                    "event_time": (
                        "message.header.stamp minus "
                        "bag_start_record_time"
                    ),
                    "bag_record_time": (
                        "preserved as provenance; never controller time"
                    ),
                    "input_frame": next(iter(frame_ids)),
                    "quaternion_storage": "xyzw",
                },
                "unit_conventions": {
                    "time": "s",
                    "position": "m",
                    "velocity": "m/s",
                    "acceleration": "m/s^2",
                    "angle": "rad",
                    "angular_velocity": "rad/s",
                    "base_thrust": (
                        "FourAxisCommand.base_thrust controller-native"
                    ),
                },
                "motor_order": list(next(iter(motor_orders))),
                "time_semantics": {
                    "event_time_source": "message_header_stamp",
                    "relative_origin": "bag_start_record_time",
                    "record_time_is_fallback": False,
                },
            }
        )
    payload = {
        "schema": CANONICAL_REPLAY_STREAM_SCHEMA,
        "episodes": episodes,
    }
    return {
        **payload,
        "content_sha256": stable_hash(payload),
    }


def write_canonical_replay_stream(
    payload: Mapping[str, Any],
    path: Any,
    *,
    prepared: Optional[Mapping[str, Any]] = None,
    assimilation_config_sha256: Optional[str] = None,
) -> Path:
    """Atomically create, but never replace, one canonical stream JSON."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(str(destination))
    # Validate both its schema and content hash before persisting.
    load_canonical_replay_stream(
        payload,
        prepared=prepared,
        assimilation_config_sha256=assimilation_config_sha256,
    )
    handle, temporary_name = tempfile.mkstemp(
        prefix=".grape-replay-stream-",
        suffix=".json",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(str(temporary), str(destination))
        except FileExistsError:
            raise FileExistsError(str(destination))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def _unchanged_non_pid_snapshot_fields(
    initial: ControllerSnapshot, current: ControllerSnapshot
) -> bool:
    return bool(
        initial.backend_id == current.backend_id
        and initial.source_commit == current.source_commit
        and initial.artifact_sha256 == current.artifact_sha256
        and initial.nominal_model_sha256
        == current.nominal_model_sha256
        and initial.controller_rate_hz == current.controller_rate_hz
        and initial.static_options == current.static_options
        and initial.nominal_mass == current.nominal_mass
        and initial.nominal_cog == current.nominal_cog
        and initial.nominal_inertia == current.nominal_inertia
        and initial.nominal_geometry == current.nominal_geometry
    )


def _prepared_grids(
    *,
    episode_id: str,
    source_bag_sha256: str,
    replay_start_offset_s: float,
    score_start_offset_s: float,
    score_end_offset_s: float,
    frame_times: Tuple[float, ...],
    prepared_episode: Any,
    declared_grids: Any,
) -> Tuple[EpisodeTimeGrids, str]:
    """Bind ReplayFrame ticks to the independently prepared bag episode."""

    config = getattr(prepared_episode, "config", None)
    observations = getattr(prepared_episode, "observations", None)
    original = getattr(prepared_episode, "grids", None)
    if (
        not isinstance(config, Mapping)
        or observations is None
        or not isinstance(original, EpisodeTimeGrids)
    ):
        raise TypeError(
            "prepared episode {} has the wrong runtime type".format(
                episode_id
            )
        )
    expected_id = str(config.get("episode_id", "")).strip()
    expected_source = _sha256(
        getattr(observations, "source_bag_sha256", None),
        "prepared source_bag_sha256",
    )
    normalized_sha256 = _sha256(
        getattr(observations, "normalized_episode_sha256", None),
        "prepared normalized_episode_sha256",
    )
    expected_offsets = (
        _finite(
            config.get("replay_start_offset_s"),
            "prepared replay_start_offset_s",
        ),
        _finite(
            config.get("score_start_offset_s"),
            "prepared score_start_offset_s",
        ),
        _finite(
            config.get("score_end_offset_s"),
            "prepared score_end_offset_s",
        ),
    )
    if (
        expected_id != episode_id
        or expected_source != source_bag_sha256
        or expected_offsets
        != (
            replay_start_offset_s,
            score_start_offset_s,
            score_end_offset_s,
        )
    ):
        raise ValueError(
            "canonical replay episode {} is not bound to the prepared "
            "bag/window".format(episode_id)
        )
    prepared_integration = np.asarray(
        original.plant_integration_grid.timestamps, dtype=float
    )
    factual_controller = np.asarray(frame_times, dtype=float)
    controller = factual_controller[
        (factual_controller >= prepared_integration[0])
        & (factual_controller <= prepared_integration[-1])
    ]
    if (
        prepared_integration.size == 0
        or controller.size == 0
    ):
        raise ValueError(
            "ReplayFrame ticks for {} do not intersect prepared "
            "trajectory/config support".format(episode_id)
        )
    prepared_integration = prepared_integration[
        prepared_integration >= controller[0]
    ]
    if prepared_integration.size == 0:
        raise ValueError(
            "ReplayFrame ticks for {} leave no prepared plant support".format(
                episode_id
            )
        )
    integration = tuple(
        float(item)
        for item in np.unique(
            np.concatenate((prepared_integration, controller))
        )
    )
    result = EpisodeTimeGrids(
        controller_tick_grid=EventGrid(
            "controller_tick", tuple(float(item) for item in controller)
        ),
        plant_integration_grid=EventGrid(
            "plant_integration", integration
        ),
        observation_grid=original.observation_grid,
        likelihood_grid=original.likelihood_grid,
        report_grid=original.report_grid,
    )
    if declared_grids is not None:
        supplied = _grids(declared_grids)
        for (field, _), expected in zip(
            _GRID_FIELDS, result.as_tuple()
        ):
            if getattr(supplied, field).timestamps != expected.timestamps:
                raise ValueError(
                    "canonical replay {} differs from the prepared/"
                    "ReplayFrame-derived grid".format(field)
                )
    return result, normalized_sha256


def _materialize_episode(
    value: Any,
    stream_sha256: str,
    prepared_episode: Any = None,
    assimilation_config_sha256: Optional[str] = None,
) -> _MaterializedEpisode:
    episode = _mapping(value, "canonical replay episode")
    episode_id = str(episode.get("episode_id", "")).strip()
    if not episode_id:
        raise ValueError("canonical replay episode_id is required")
    source_bag_sha256 = _sha256(
        episode.get("source_bag_sha256"), "source_bag_sha256"
    )
    topic_inventory_sha256 = _sha256(
        episode.get("topic_inventory_sha256"),
        "topic_inventory_sha256",
    )
    replay_start_offset_s = _finite(
        episode.get("replay_start_offset_s"),
        "replay_start_offset_s",
    )
    score_start_offset_s = _finite(
        episode.get("score_start_offset_s"),
        "score_start_offset_s",
    )
    score_end_offset_s = _finite(
        episode.get("score_end_offset_s"),
        "score_end_offset_s",
    )
    metadata_records = tuple(
        sorted(
            (
                _metadata_snapshot(item)
                for item in _sequence(
                    episode.get("metadata_records"),
                    "metadata_records",
                )
            ),
            key=lambda item: item[0],
        )
    )
    if (
        not metadata_records
        or len({item[0] for item in metadata_records})
        != len(metadata_records)
    ):
        raise ValueError(
            "metadata records must have unique increasing timestamps"
        )
    frames = tuple(
        sorted(
            (
                _mapping(item, "replay frame")
                for item in _sequence(episode.get("frames"), "frames")
            ),
            key=lambda item: _finite(item.get("stamp"), "frame stamp"),
        )
    )
    if len(frames) < 2:
        raise ValueError(
            "exact conformance requires at least two replay frames"
        )
    frame_times = tuple(
        _finite(item.get("stamp"), "frame stamp") for item in frames
    )
    if len(set(frame_times)) != len(frame_times):
        raise ValueError(
            "replay frames must have unique increasing timestamps"
        )
    replay_boundary = tuple(
        stamp for stamp in frame_times if stamp <= replay_start_offset_s
    )
    score_boundary = tuple(
        stamp for stamp in frame_times if stamp >= score_end_offset_s
    )
    if not replay_boundary or not score_boundary:
        raise ValueError(
            "factual ReplayFrame ticks must bracket the configured "
            "replay/score interval"
        )
    if prepared_episode is None:
        grids = _grids(episode.get("grids"))
        normalized_episode_sha256 = _sha256(
            episode.get("normalized_episode_sha256"),
            "normalized_episode_sha256",
        )
        if frame_times != grids.controller_tick_grid.timestamps:
            raise ValueError(
                "frames must align one-to-one with controller_tick_grid"
            )
    else:
        grids, normalized_episode_sha256 = _prepared_grids(
            episode_id=episode_id,
            source_bag_sha256=source_bag_sha256,
            replay_start_offset_s=replay_start_offset_s,
            score_start_offset_s=score_start_offset_s,
            score_end_offset_s=score_end_offset_s,
            frame_times=frame_times,
            prepared_episode=prepared_episode,
            declared_grids=episode.get("grids"),
        )
    if metadata_records[0][0] > frame_times[0]:
        raise ValueError(
            "initial metadata must precede the first replay frame"
        )
    initial_snapshot = metadata_records[0][3]
    if any(
        not _unchanged_non_pid_snapshot_fields(
            initial_snapshot, item[3]
        )
        for item in metadata_records[1:]
    ):
        raise ValueError(
            "one fixture may contain dynamic PID gain/limit changes only; "
            "controller options/model/geometry/rate changes require a "
            "new episode"
        )

    controller_inputs = []
    continuous = {
        "command_timestamp": [],
        "pid_terms": [],
        "four_axis_command": [],
        "vectoring_force": [],
        "gimbal_command": [],
        "allocation_internal": [],
        "torque_allocation_matrix_inverse": [],
    }
    events = []
    initial_state = None
    previous_metadata_sha = metadata_records[0][2]
    metadata_index = 0
    previous_after_state = None
    for index, (stamp, frame) in enumerate(zip(frame_times, frames)):
        if frame.get("schema") != REPLAY_FRAME_SCHEMA:
            raise ValueError("unsupported replay frame schema")
        while (
            metadata_index + 1 < len(metadata_records)
            and metadata_records[metadata_index + 1][0] <= stamp
        ):
            metadata_index += 1
        (
            _,
            frame_id,
            active_sha,
            active_snapshot,
            _,
        ) = metadata_records[metadata_index]
        if (
            str(frame.get("frame_id", "")) != frame_id
            or _sha256(
                frame.get("controller_snapshot_sha256"),
                "frame controller_snapshot_sha256",
            )
            != active_sha
        ):
            raise ValueError(
                "frame identity does not match active replay metadata"
            )
        before_state = _frame_state(frame, "before")
        after_state = _frame_state(frame, "after")
        if initial_state is None:
            initial_state = before_state
        elif (
            previous_after_state != before_state
            and not (
                before_state.pending_events & _RESET_EVENT
            )
        ):
            raise ValueError(
                "controller state is discontinuous without a recorded reset"
            )
        previous_after_state = after_state
        frame_events = _event_mask(frame.get("events"), "events")
        saturated = _boolean(frame.get("saturated"), "saturated")
        if saturated != bool(frame_events & _SATURATED_EVENT):
            raise ValueError(
                "frame saturated flag and event bitmask disagree"
            )
        reset = bool(
            _boolean(frame.get("reset"), "reset")
            or before_state.pending_events & _RESET_EVENT
        )
        pid_config_event = None
        if active_sha != previous_metadata_sha:
            pid_config_event = tuple(
                tuple(
                    (
                        active_snapshot.gains[field][axis]
                        if field in _GAIN_FIELDS
                        else active_snapshot.limits[field][axis]
                    )
                    for field in _PID_CONFIG_FIELDS
                )
                for axis in range(6)
            )
            previous_metadata_sha = active_sha
        geometry = _flattened_geometry(
            frame.get("allocation_geometry_names"),
            frame.get("allocation_geometry_offsets"),
            frame.get("allocation_geometry_values"),
            "frame allocation geometry",
        )
        core_input = ControllerCoreInput(
            stamp=stamp,
            dt=_finite(frame.get("dt"), "frame dt"),
            position=_vector(frame.get("position"), 3, "position"),
            velocity=_vector(frame.get("velocity"), 3, "velocity"),
            orientation=_quaternion_matrix(
                frame.get("orientation"), "orientation"
            ),
            angular_velocity=_vector(
                frame.get("angular_velocity"), 3, "angular_velocity"
            ),
            target_position=_vector(
                frame.get("target_position"), 3, "target_position"
            ),
            target_velocity=_vector(
                frame.get("target_velocity"), 3, "target_velocity"
            ),
            target_acceleration=_vector(
                frame.get("target_acceleration"),
                3,
                "target_acceleration",
            ),
            target_orientation=_quaternion_matrix(
                frame.get("target_orientation"), "target_orientation"
            ),
            target_angular_velocity=_vector(
                frame.get("target_angular_velocity"),
                3,
                "target_angular_velocity",
            ),
            target_angular_acceleration=_vector(
                frame.get("target_angular_acceleration"),
                3,
                "target_angular_acceleration",
            ),
            control_mode=tuple(
                _integer(item, "control_mode")
                for item in _sequence(
                    frame.get("control_mode"), "control_mode"
                )
            ),
            integration_enabled=tuple(
                _boolean(item, "integration_enabled")
                for item in _sequence(
                    frame.get("integration_enabled"),
                    "integration_enabled",
                )
            ),
            flight_state=_integer(
                frame.get("flight_state"), "flight_state"
            ),
            force_landing=_boolean(
                frame.get("force_landing"), "force_landing"
            ),
            joint_positions=tuple(
                _finite(item, "joint_positions")
                for item in _sequence(
                    frame.get("joint_positions"), "joint_positions"
                )
            ),
            initial_height=_finite(
                frame.get("initial_height"), "initial_height"
            ),
            reset=reset,
            current_rpy=_vector(
                frame.get("current_rpy"), 3, "current_rpy"
            ),
            target_rpy=_vector(
                frame.get("target_rpy"), 3, "target_rpy"
            ),
            pid_config=pid_config_event,
            state_previous_stamp=after_state.previous_stamp,
            allocation_geometry=geometry,
        )
        controller_inputs.append(core_input)
        pid = []
        for field in (
            "pid_result",
            "pid_p_term",
            "pid_i_term",
            "pid_d_term",
        ):
            pid.extend(_vector(frame.get(field), 6, field))
        four_axis = list(
            _vector(frame.get("four_axis_angles"), 3, "four_axis_angles")
        )
        base = tuple(
            _finite(item, "base_thrust")
            for item in _sequence(
                frame.get("base_thrust"), "base_thrust"
            )
        )
        gimbal = tuple(
            _finite(item, "gimbal_angle")
            for item in _sequence(
                frame.get("gimbal_angle"), "gimbal_angle"
            )
        )
        vectoring = tuple(
            _finite(item, "target_vectoring_force")
            for item in _sequence(
                frame.get("target_vectoring_force"),
                "target_vectoring_force",
            )
        )
        rows = _integer(
            frame.get("torque_allocation_rows"),
            "torque_allocation_rows",
        )
        columns = _integer(
            frame.get("torque_allocation_columns"),
            "torque_allocation_columns",
        )
        matrix = tuple(
            _finite(item, "torque_allocation_matrix_inverse")
            for item in _sequence(
                frame.get("torque_allocation_matrix_inverse"),
                "torque_allocation_matrix_inverse",
            )
        )
        if rows < 1 or columns != 3 or len(matrix) != rows * columns:
            raise ValueError(
                "torque allocation matrix dimensions do not align"
            )
        if not base or not gimbal or not vectoring:
            raise ValueError(
                "recorded PC command channels may not be empty"
            )
        continuous["command_timestamp"].append((stamp,))
        continuous["pid_terms"].append(tuple(pid))
        continuous["four_axis_command"].append(
            tuple(four_axis) + base
        )
        continuous["vectoring_force"].append(vectoring)
        continuous["gimbal_command"].append(gimbal)
        continuous["allocation_internal"].append(
            (
                _finite(frame.get("target_roll"), "target_roll"),
                _finite(frame.get("target_pitch"), "target_pitch"),
                _finite(
                    frame.get("candidate_yaw_term"),
                    "candidate_yaw_term",
                ),
            )
        )
        continuous["torque_allocation_matrix_inverse"].append(
            matrix
        )
        events.append(frame_events)

    arrays = {}
    for name, rows in continuous.items():
        try:
            arrays[name] = np.asarray(rows, dtype=float)
        except ValueError as exc:
            raise ValueError(
                "recorded {} width changes within the episode".format(name)
            ) from exc
        if arrays[name].ndim != 2:
            raise ValueError(
                "recorded {} must be a matrix".format(name)
            )
    source_topics = _strings(
        episode.get("source_topics"), "source_topics"
    )
    required_topics = {
        "/gimbalrotor/controller_replay/metadata",
        "/gimbalrotor/controller_replay/frame",
    }
    if not required_topics.issubset(set(source_topics)):
        raise ValueError(
            "source_topics must include future replay metadata and frame"
        )
    frame_conventions = {
        str(key): str(item)
        for key, item in _mapping(
            episode.get("frame_conventions"), "frame_conventions"
        ).items()
    }
    unit_conventions = {
        str(key): str(item)
        for key, item in _mapping(
            episode.get("unit_conventions"), "unit_conventions"
        ).items()
    }
    if not frame_conventions or not unit_conventions:
        raise ValueError(
            "frame and unit conventions must be explicit"
        )
    motor_order = _strings(
        episode.get("motor_order"), "motor_order"
    )
    metadata_motor_orders = {
        item[4] for item in metadata_records
    }
    base_width = arrays["four_axis_command"].shape[1] - 3
    if (
        len(set(motor_order)) != len(motor_order)
        or len(metadata_motor_orders) != 1
        or next(iter(metadata_motor_orders)) != motor_order
        or len(motor_order) != base_width
    ):
        raise ValueError(
            "motor_order must match replay metadata and base_thrust width"
        )
    start_ns = int(round(1.0e9 * frame_times[0]))
    end_ns = int(round(1.0e9 * frame_times[-1]))
    if start_ns < 0 or end_ns <= start_ns:
        raise ValueError(
            "replay frame interval must map to increasing non-negative ns"
        )
    metadata = {
        "canonical_stream_sha256": stream_sha256,
        "metadata_snapshot_sha256": initial_snapshot.snapshot_id,
        "metadata_record_sha256": [
            item[2] for item in metadata_records
        ],
        "normalized_episode_sha256": normalized_episode_sha256,
        "assimilation_config_sha256": (
            assimilation_config_sha256
            if assimilation_config_sha256 is not None
            else ""
        ),
        "future_replay_topics": list(source_topics),
        "controller_output_data_sha256": stable_hash(
            {
                "continuous": arrays,
                "events": np.asarray(events, dtype=np.uint32),
            }
        ),
        "controller_tick_domains": {
            "schema": "grape_controller_tick_domains/v1",
            "factual_replay_tick_grid": list(frame_times),
            "inference_controller_tick_grid": list(
                grids.controller_tick_grid.timestamps
            ),
            "pre_replay_boundary_tick_s": replay_boundary[-1],
            "post_score_boundary_tick_s": score_boundary[0],
            "inference_is_clipped_to_prepared_plant_support": (
                prepared_episode is not None
            ),
        },
    }
    fixture = ControllerReplayFixture(
        schema=CONTROLLER_REPLAY_FIXTURE_SCHEMA,
        episode_id=episode_id,
        source_bag_sha256=source_bag_sha256,
        topic_inventory_sha256=topic_inventory_sha256,
        replay_start_offset_s=replay_start_offset_s,
        score_start_offset_s=score_start_offset_s,
        score_end_offset_s=score_end_offset_s,
        grids=grids,
        controller_inputs=tuple(controller_inputs),
        metadata=metadata,
    )
    return _MaterializedEpisode(
        fixture=fixture,
        snapshot=initial_snapshot,
        initial_state=initial_state,
        recorded_continuous=MappingProxyType(arrays),
        recorded_events=np.asarray(events, dtype=np.uint32),
        source_topics=source_topics,
        frame_conventions=MappingProxyType(frame_conventions),
        unit_conventions=MappingProxyType(unit_conventions),
        motor_order=motor_order,
        interval_start_time_ns=start_ns,
        interval_end_time_ns=end_ns,
    )


def load_canonical_replay_stream(
    path: Any,
    *,
    prepared: Optional[Mapping[str, Any]] = None,
    assimilation_config_sha256: Optional[str] = None,
) -> Tuple[Mapping[str, _MaterializedEpisode], str]:
    """Load and fully validate one canonical future-message extraction."""

    if isinstance(path, Mapping):
        values = dict(path)
    else:
        source = Path(path).expanduser().resolve()
        try:
            values = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "canonical replay stream must be readable UTF-8 JSON"
            ) from exc
    root = _mapping(values, "canonical replay stream")
    if set(root) != {"schema", "episodes", "content_sha256"}:
        raise ValueError(
            "canonical replay stream requires schema, episodes, content_sha256"
        )
    if root["schema"] != CANONICAL_REPLAY_STREAM_SCHEMA:
        raise ValueError("unsupported canonical replay stream schema")
    digest = _sha256(
        root["content_sha256"], "canonical stream content_sha256"
    )
    payload = {"schema": root["schema"], "episodes": root["episodes"]}
    if stable_hash(payload) != digest:
        raise ValueError("canonical replay stream content hash mismatch")
    raw_episodes = _sequence(root["episodes"], "episodes")
    if not raw_episodes:
        raise ValueError("canonical replay stream requires episodes")
    parsed = tuple(
        _mapping(item, "canonical replay episode")
        for item in raw_episodes
    )
    episode_ids = tuple(
        str(item.get("episode_id", "")).strip() for item in parsed
    )
    if (
        any(not item for item in episode_ids)
        or len(set(episode_ids)) != len(episode_ids)
    ):
        raise ValueError(
            "canonical replay stream requires unique non-empty episode IDs"
        )
    prepared_by_id = None
    config_sha256 = None
    if prepared is not None:
        if not isinstance(prepared, Mapping) or not prepared:
            raise TypeError("prepared must be a non-empty episode mapping")
        prepared_by_id = {
            str(key): value for key, value in prepared.items()
        }
        if set(prepared_by_id) != set(episode_ids):
            raise ValueError(
                "canonical replay stream must cover exactly the prepared "
                "episodes"
            )
        config_sha256 = _sha256(
            assimilation_config_sha256,
            "assimilation_config_sha256",
        )
    elif assimilation_config_sha256 is not None:
        raise ValueError(
            "assimilation_config_sha256 requires prepared episodes"
        )
    episodes = tuple(
        _materialize_episode(
            item,
            digest,
            None
            if prepared_by_id is None
            else prepared_by_id[episode_id],
            config_sha256,
        )
        for episode_id, item in zip(episode_ids, parsed)
    )
    result = {item.fixture.episode_id: item for item in episodes}
    if len(result) != len(episodes):
        raise ValueError("canonical replay stream repeats an episode_id")
    return MappingProxyType(dict(sorted(result.items()))), digest


def _request_binding(
    fixture: ControllerReplayFixture, snapshot: ControllerSnapshot
) -> Mapping[str, Any]:
    controller_inputs = {
        "schema": EXACT_EPISODE_CONTROLLER_INPUT_SCHEMA,
        "episode_id": fixture.episode_id,
        "controller_inputs": [
            item.to_mapping() for item in fixture.controller_inputs
        ],
    }
    return {
        "schema": EXACT_EPISODE_REQUEST_BINDING_SCHEMA,
        "episode_id": fixture.episode_id,
        "source_bag_sha256": fixture.source_bag_sha256,
        "controller_replay_fixture_sha256": fixture.fixture_sha256,
        "controller_input_sha256": stable_hash(controller_inputs),
        "controller_snapshot_sha256": snapshot.snapshot_id,
    }


def _identity(episodes: Mapping[str, _MaterializedEpisode]) -> ExactOracleIdentity:
    identities = {
        (
            item.snapshot.backend_id,
            item.snapshot.source_commit,
            item.snapshot.artifact_sha256,
        )
        for item in episodes.values()
    }
    if len(identities) != 1:
        raise ValueError(
            "all materialized episodes must use one controller artifact/source"
        )
    backend_id, source_commit, artifact_sha256 = next(iter(identities))
    return ExactOracleIdentity(
        protocol=EXACT_ORACLE_PROTOCOL,
        backend_id=backend_id,
        implementation_language="c++",
        source_commit=source_commit,
        artifact_sha256=artifact_sha256,
        capabilities=PC_EXACT_ORACLE_CAPABILITIES,
        fidelity="pc_exact",
    )


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def materialize_exact_replay_inputs(
    *,
    stream_path: Any,
    prepared: Mapping[str, Any],
    assimilation_config_sha256: str,
    exact_replay_executable: Any,
    output_root: Any,
    run_id: str,
    timeout_s: float = 30.0,
    oracle_factory: Any = PersistentSubprocessExactControllerOracle,
) -> Path:
    """Run factual conformance and atomically write all six exact inputs."""

    run_name = str(run_id)
    if (
        not run_name.strip()
        or Path(run_name).name != run_name
    ):
        raise ValueError("run_id must be one non-empty path component")
    root = Path(output_root).expanduser().resolve()
    destination = root / run_name
    if destination.exists():
        raise FileExistsError(
            "materialized exact run already exists: {}".format(destination)
        )
    episodes, stream_sha256 = load_canonical_replay_stream(
        stream_path,
        prepared=prepared,
        assimilation_config_sha256=assimilation_config_sha256,
    )
    identity = _identity(episodes)
    timeout = _finite(timeout_s, "timeout_s")
    if timeout <= 0.0:
        raise ValueError("timeout_s must be positive")
    executable = Path(exact_replay_executable).expanduser().resolve()
    if not executable.is_file():
        raise ValueError("exact replay executable does not exist")
    if not callable(oracle_factory):
        raise TypeError("oracle_factory must be callable")
    oracle = oracle_factory(
        (
            str(executable),
            "--artifact-sha256",
            identity.artifact_sha256,
        ),
        identity,
        timeout_s=timeout,
    )
    requests = []
    conformance_fixtures = []
    evidences = {}
    policy_sha256 = stable_hash(
        {
            "schema": MATERIALIZER_POLICY_SCHEMA,
            "canonical_stream_sha256": stream_sha256,
            "assimilation_config_sha256": _sha256(
                assimilation_config_sha256,
                "assimilation_config_sha256",
            ),
            "timestamp_tolerance_s": 0.0,
            "continuous_rmse_threshold": 0.01,
            "continuous_maximum_error_threshold": 0.03,
            "event_agreement_threshold": 1.0,
            "required_channels": (
                "command_timestamp",
                "pid_terms",
                "four_axis_command",
                "vectoring_force",
                "gimbal_command",
                "allocation_internal",
                "torque_allocation_matrix_inverse",
            ),
        }
    )
    try:
        for episode_id, item in episodes.items():
            binding = _request_binding(item.fixture, item.snapshot)
            request = build_exact_replay_payload(
                item.snapshot,
                item.initial_state,
                item.fixture.controller_inputs,
                evidence_binding=binding,
            )
            provenance = ExactOracleFixtureProvenance.create(
                source_bag_sha256=item.fixture.source_bag_sha256,
                source_topics=item.source_topics,
                interval_start_time_ns=item.interval_start_time_ns,
                interval_end_time_ns=item.interval_end_time_ns,
                frame_conventions=item.frame_conventions,
                unit_conventions=item.unit_conventions,
                motor_order=item.motor_order,
                request_payload=request,
                continuous=item.recorded_continuous,
                events=item.recorded_events,
                extraction_config_sha256=policy_sha256,
                source_commit=item.snapshot.source_commit,
            )
            conformance_fixture = ExactOracleConformanceFixture(
                continuous=item.recorded_continuous,
                events=item.recorded_events,
                provenance=provenance,
                fidelity="pc_exact",
            )
            report = evaluate_exact_oracle_conformance(
                oracle, request, conformance_fixture
            )
            if not report.passed:
                raise RuntimeError(
                    "episode {} factual conformance failed: {}".format(
                        episode_id,
                        "; ".join(report.reasons) or report.status,
                    )
                )
            evidence = ExactEpisodeConformanceEvidence.create(
                fixture=item.fixture,
                snapshot=item.snapshot,
                initial_controller_state=item.initial_state,
                conformance_fixture=conformance_fixture,
                conformance_report=report,
            )
            requests.append(
                {
                    "episode_id": episode_id,
                    "request_payload_sha256": stable_hash(request),
                    "request_payload": request,
                }
            )
            conformance_fixtures.append(
                {
                    "episode_id": episode_id,
                    "fixture_content_sha256": (
                        provenance.content_sha256
                    ),
                    "conformance_fixture": (
                        conformance_fixture.to_mapping()
                    ),
                }
            )
            evidences[episode_id] = evidence
    finally:
        close = getattr(oracle, "close", None)
        if callable(close):
            close()

    fixture_bundle = _content_hashed_payload(
        FIXTURE_BUNDLE_SCHEMA,
        "fixtures",
        [
            item.fixture.to_dict()
            for item in episodes.values()
        ],
    )
    snapshot_bundle = _content_hashed_payload(
        SNAPSHOT_BUNDLE_SCHEMA,
        "snapshots",
        [
            {
                "episode_id": episode_id,
                "snapshot": item.snapshot.to_mapping(),
                "snapshot_sha256": item.snapshot.snapshot_id,
            }
            for episode_id, item in episodes.items()
        ],
    )
    state_bundle = _content_hashed_payload(
        STATE_BUNDLE_SCHEMA,
        "episodes",
        [
            {
                "episode_id": episode_id,
                "controller_state": (
                    item.initial_state.to_mapping()
                ),
            }
            for episode_id, item in episodes.items()
        ],
    )
    request_bundle = _content_hashed_payload(
        EXACT_REQUEST_BUNDLE_SCHEMA, "episodes", requests
    )
    conformance_fixture_bundle = _content_hashed_payload(
        EXACT_CONFORMANCE_FIXTURE_BUNDLE_SCHEMA,
        "episodes",
        conformance_fixtures,
    )
    conformance_bundle = ExactEpisodeConformanceBundle(
        episodes=evidences
    ).to_mapping()
    payloads = dict(
        zip(
            MATERIALIZED_EXACT_FILES,
            (
                fixture_bundle,
                snapshot_bundle,
                state_bundle,
                request_bundle,
                conformance_fixture_bundle,
                conformance_bundle,
            ),
        )
    )

    root.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            "materialized exact run already exists: {}".format(destination)
        )
    temporary = Path(
        tempfile.mkdtemp(prefix=".grape-exact-", dir=str(root))
    )
    try:
        for name in MATERIALIZED_EXACT_FILES:
            _json_write(temporary / name, payloads[name])
        os.rename(str(temporary), str(destination))
    except BaseException:
        shutil.rmtree(str(temporary), ignore_errors=True)
        raise
    return destination


__all__ = [
    "CANONICAL_REPLAY_STREAM_SCHEMA",
    "EXACT_CONFORMANCE_FIXTURE_BUNDLE_SCHEMA",
    "EXACT_REQUEST_BUNDLE_SCHEMA",
    "MATERIALIZED_EXACT_FILES",
    "MATERIALIZER_POLICY_SCHEMA",
    "NOMINAL_GEOMETRY_SCHEMA",
    "REPLAY_FRAME_SCHEMA",
    "REPLAY_FRAME_TOPIC",
    "REPLAY_FRAME_TYPE",
    "REPLAY_METADATA_SCHEMA",
    "REPLAY_METADATA_TOPIC",
    "REPLAY_METADATA_TYPE",
    "extract_canonical_replay_stream",
    "load_canonical_replay_stream",
    "materialize_exact_replay_inputs",
    "replay_frame_message_to_mapping",
    "replay_metadata_message_to_mapping",
    "write_canonical_replay_stream",
]
