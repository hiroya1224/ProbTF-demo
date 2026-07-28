"""Fail-closed sufficiency audit for factual Grape controller replay."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .bag_reader import BagTopicInventory
from .provenance import stable_hash, validated_sha256


AUDIT_AVAILABLE = "AVAILABLE"
AUDIT_DERIVABLE = "DERIVABLE"
AUDIT_MISSING = "MISSING"
_AUDIT_STATUSES = (AUDIT_AVAILABLE, AUDIT_DERIVABLE, AUDIT_MISSING)

CONTROLLER_REPLAY_AUDIT_SCHEMA = "grape_controller_replay_sufficiency/v1"
CONTROLLER_REPLAY_AUDIT_BUNDLE_SCHEMA = "grape_controller_replay_audit/v1"

REPLAY_FRAME_TOPICS = (
    "/gimbalrotor/controller_replay/frame",
    "/gimbalrotor/controller/replay_frame",
)
REPLAY_METADATA_TOPICS = (
    "/gimbalrotor/controller_replay/metadata",
    "/gimbalrotor/controller/replay_metadata",
)

_PID_GROUPS = ("xy", "z", "roll_pitch", "yaw")
REPLAY_AUDIT_FIELDS = (
    "controller_tick",
    "controller_estimator_state",
    "navigator_target",
    "control_mode",
    "integration_enable_reset_event",
    "force_landing_flight_state",
    "pid_gain_limit_changes",
    "nominal_model_geometry_snapshot",
    "joint_state",
    "recorded_four_axis_command",
    "recorded_gimbal_command",
    "pid_debug_term",
    "target_vectoring_force",
    "torque_allocation_matrix",
    "spinal_pwm_channel",
)


@dataclass(frozen=True)
class ReplayAuditItem:
    """Availability decision for one required replay input."""

    field: str
    status: str
    required: bool
    observed_topics: Tuple[str, ...]
    evidence: str
    action: str

    def __post_init__(self) -> None:
        if not self.field or self.status not in _AUDIT_STATUSES:
            raise ValueError("replay audit item field/status is invalid")
        if type(self.required) is not bool:
            raise TypeError("replay audit item required must be bool")
        object.__setattr__(
            self,
            "observed_topics",
            tuple(sorted(set(str(topic) for topic in self.observed_topics))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControllerReplayAudit:
    """One bag's complete §8.3 audit and fail-closed gate decision."""

    episode_id: str
    bag_path: str
    source_bag_sha256: str
    topic_inventory_sha256: str
    fields: Tuple[ReplayAuditItem, ...]
    schema: str = CONTROLLER_REPLAY_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CONTROLLER_REPLAY_AUDIT_SCHEMA:
            raise ValueError("unsupported replay audit schema")
        if not self.episode_id or not self.fields:
            raise ValueError("replay audit requires episode and field decisions")
        names = tuple(item.field for item in self.fields)
        if names != REPLAY_AUDIT_FIELDS:
            raise ValueError(
                "replay audit must classify every plan section 8.3 field in order"
            )
        object.__setattr__(
            self,
            "source_bag_sha256",
            validated_sha256(self.source_bag_sha256, "source_bag_sha256"),
        )
        object.__setattr__(
            self,
            "topic_inventory_sha256",
            validated_sha256(
                self.topic_inventory_sha256, "topic_inventory_sha256"
            ),
        )

    @property
    def missing_required_fields(self) -> Tuple[str, ...]:
        return tuple(
            item.field
            for item in self.fields
            if item.required and item.status == AUDIT_MISSING
        )

    @property
    def derivation_required_fields(self) -> Tuple[str, ...]:
        return tuple(
            item.field
            for item in self.fields
            if item.required and item.status == AUDIT_DERIVABLE
        )

    @property
    def fixture_inputs_resolvable(self) -> bool:
        return not self.missing_required_fields

    @property
    def exact_replay_ready(self) -> bool:
        # The audit never treats an unmaterialized derivation as an exact input.
        return not self.missing_required_fields and not self.derivation_required_fields

    @property
    def decision(self) -> str:
        if self.missing_required_fields:
            return "BLOCKED_MISSING_INPUTS"
        if self.derivation_required_fields:
            return "BLOCKED_DERIVATION_REQUIRED"
        return "READY"

    @property
    def audit_sha256(self) -> str:
        return stable_hash(self.to_dict(include_audit_hash=False))

    def to_dict(self, include_audit_hash: bool = True) -> Dict[str, Any]:
        counts = {
            status: sum(item.status == status for item in self.fields)
            for status in _AUDIT_STATUSES
        }
        result = {
            "schema": self.schema,
            "episode_id": self.episode_id,
            "bag_path": self.bag_path,
            "source_bag_sha256": self.source_bag_sha256,
            "topic_inventory_sha256": self.topic_inventory_sha256,
            "status_counts": counts,
            "fixture_inputs_resolvable": self.fixture_inputs_resolvable,
            "exact_replay_ready": self.exact_replay_ready,
            "decision": self.decision,
            "missing_required_fields": list(self.missing_required_fields),
            "derivation_required_fields": list(self.derivation_required_fields),
            "fields": [item.to_dict() for item in self.fields],
        }
        if include_audit_hash:
            result["audit_sha256"] = self.audit_sha256
        return result


def _present(inventory: BagTopicInventory, candidates: Iterable[str]) -> Tuple[str, ...]:
    return tuple(
        topic
        for topic in candidates
        if topic in inventory.topics and inventory.topics[topic].message_count > 0
    )


def _typed_topics(inventory: BagTopicInventory, suffix: str) -> Tuple[str, ...]:
    return tuple(
        topic
        for topic, entry in inventory.topics.items()
        if entry.message_count > 0 and entry.message_type.endswith(suffix)
    )


def _item(
    field: str,
    status: str,
    observed: Iterable[str],
    evidence: str,
    action: str,
    *,
    required: bool = True,
) -> ReplayAuditItem:
    return ReplayAuditItem(
        field=field,
        status=status,
        required=required,
        observed_topics=tuple(observed),
        evidence=evidence,
        action=action,
    )


def audit_controller_replay_inventory(
    inventory: BagTopicInventory,
    *,
    episode_id: str,
) -> ControllerReplayAudit:
    """Classify every redesign-plan §8.3 replay input conservatively."""

    if not isinstance(inventory, BagTopicInventory):
        raise TypeError("inventory must be BagTopicInventory")
    replay_frames = _present(inventory, REPLAY_FRAME_TOPICS) + _typed_topics(
        inventory, "GimbalrotorControllerReplayFrame"
    )
    replay_metadata = _present(inventory, REPLAY_METADATA_TOPICS) + _typed_topics(
        inventory, "GimbalrotorControllerReplayMetadata"
    )
    command = _present(inventory, ("/gimbalrotor/four_axes/command",))
    pid_debug = _present(inventory, ("/gimbalrotor/debug/pose/pid",))
    controller_ticks = replay_frames or tuple(sorted(set(command + pid_debug)))
    if replay_frames:
        controller_tick_item = _item(
            "controller_tick",
            AUDIT_AVAILABLE,
            replay_frames,
            "ReplayFrame provides the controller tick and explicit dt.",
            "",
        )
    elif controller_ticks:
        controller_tick_item = _item(
            "controller_tick",
            AUDIT_DERIVABLE,
            controller_ticks,
            "Controller output/debug timestamps can define a candidate tick grid.",
            "Materialize and validate tick/dt against the live controller rate.",
        )
    else:
        controller_tick_item = _item(
            "controller_tick",
            AUDIT_MISSING,
            (),
            "No ReplayFrame, controller command, or PID debug tick was recorded.",
            "Record GimbalrotorControllerReplayFrame at every controller tick.",
        )

    estimator_topics = _present(
        inventory,
        (
            "/gimbalrotor/uav/full_state",
            "/gimbalrotor/kf/imu1/data",
            "/gimbalrotor/kf/mocap1/data",
        ),
    )
    estimator_item = (
        _item(
            "controller_estimator_state",
            AUDIT_AVAILABLE,
            replay_frames,
            "ReplayFrame stores the exact feedback passed to the controller.",
            "",
        )
        if replay_frames
        else _item(
            "controller_estimator_state",
            AUDIT_DERIVABLE if estimator_topics else AUDIT_MISSING,
            estimator_topics,
            (
                "Published onboard estimator products exist, but equivalence to the "
                "controller's in-process state must be validated."
                if estimator_topics
                else "No controller ReplayFrame or onboard estimator state was recorded."
            ),
            (
                "Bind the selected onboard state topic and prove it matches controller input."
                if estimator_topics
                else "Record exact controller feedback in ReplayFrame."
            ),
        )
    )

    navigator = _present(inventory, ("/gimbalrotor/uav/nav",))
    navigator_item = (
        _item(
            "navigator_target",
            AUDIT_AVAILABLE,
            replay_frames or navigator,
            (
                "ReplayFrame stores the per-tick target."
                if replay_frames
                else "FlightNav target events were recorded and can be causally held."
            ),
            "",
        )
        if replay_frames or navigator
        else _item(
            "navigator_target",
            AUDIT_MISSING,
            pid_debug,
            "PID target terms are partial and cannot reconstruct the full navigator target.",
            "Record FlightNav and preferably the per-tick ReplayFrame target.",
        )
    )

    control_mode_item = (
        _item(
            "control_mode",
            AUDIT_AVAILABLE,
            replay_frames,
            "ReplayFrame stores all six controller modes per tick.",
            "",
        )
        if replay_frames
        else _item(
            "control_mode",
            AUDIT_DERIVABLE if navigator else AUDIT_MISSING,
            navigator,
            (
                "FlightNav contains axis navigation modes; materialization must define "
                "causal hold and mode transitions."
                if navigator
                else "No full six-axis controller mode source was recorded."
            ),
            (
                "Materialize deterministic mode segments from FlightNav."
                if navigator
                else "Record control_mode in ReplayFrame."
            ),
        )
    )

    flight_config = _present(inventory, ("/gimbalrotor/flight_config_cmd",))
    flight_state = _present(inventory, ("/gimbalrotor/flight_state",))
    integration_sources = tuple(sorted(set(flight_config + flight_state + pid_debug)))
    integration_item = (
        _item(
            "integration_enable_reset_event",
            AUDIT_AVAILABLE,
            replay_frames,
            "ReplayFrame stores integration gates and reset events explicitly.",
            "",
        )
        if replay_frames
        else _item(
            "integration_enable_reset_event",
            AUDIT_DERIVABLE if flight_config and flight_state else AUDIT_MISSING,
            integration_sources,
            (
                "FlightConfigCmd and flight-state transitions permit a candidate event "
                "timeline, but reset semantics require validation."
                if flight_config and flight_state
                else "Integration enable/reset history is incomplete."
            ),
            (
                "Materialize and validate integration/reset transitions against PID terms."
                if flight_config and flight_state
                else "Record integration_enabled and reset events in ReplayFrame."
            ),
        )
    )

    force_landing_item = (
        _item(
            "force_landing_flight_state",
            AUDIT_AVAILABLE,
            replay_frames or tuple(sorted(set(flight_config + flight_state))),
            (
                "ReplayFrame stores force-landing and flight-state input."
                if replay_frames
                else "FlightConfigCmd and flight_state channels were both recorded."
            ),
            "",
        )
        if replay_frames or (flight_config and flight_state)
        else _item(
            "force_landing_flight_state",
            AUDIT_MISSING,
            tuple(sorted(set(flight_config + flight_state))),
            "Force-landing command and flight state are not both available.",
            "Record both inputs or store them in ReplayFrame.",
        )
    )

    gain_topics = tuple(
        "/gimbalrotor/controller/{}/parameter_updates".format(group)
        for group in _PID_GROUPS
    )
    description_topics = tuple(
        "/gimbalrotor/controller/{}/parameter_descriptions".format(group)
        for group in _PID_GROUPS
    )
    observed_gain = _present(inventory, gain_topics + description_topics)
    all_gain_topics = all(topic in observed_gain for topic in gain_topics)
    all_descriptions = all(topic in observed_gain for topic in description_topics)
    gain_item = (
        _item(
            "pid_gain_limit_changes",
            AUDIT_AVAILABLE,
            replay_metadata or observed_gain,
            (
                "ReplayMetadata contains the initial gains/limits and all changes."
                if replay_metadata
                else "All four dynamic-reconfigure update and description channels exist."
            ),
            "",
        )
        if replay_metadata or (all_gain_topics and all_descriptions)
        else _item(
            "pid_gain_limit_changes",
            AUDIT_MISSING,
            observed_gain,
            "The complete initial gain/limit set and change history is not available.",
            "Record ReplayMetadata and every gain/limit change.",
        )
    )

    partial_model = _present(
        inventory,
        (
            "/tf_static",
            "/gimbalrotor/uav_info",
            "/gimbalrotor/joint_profiles",
            "/gimbalrotor/motor_info",
        ),
    )
    nominal_model_item = (
        _item(
            "nominal_model_geometry_snapshot",
            AUDIT_AVAILABLE,
            replay_metadata,
            "ReplayMetadata stores the nominal model, geometry, options, and hashes.",
            "",
        )
        if replay_metadata
        else _item(
            "nominal_model_geometry_snapshot",
            AUDIT_MISSING,
            partial_model,
            "TF/profile topics are partial and do not bind nominal mass, inertia, CoG, URDF, and options.",
            "Record GimbalrotorControllerReplayMetadata with model and geometry hashes.",
        )
    )

    joint_state = _present(inventory, ("/gimbalrotor/joint_states",))
    joint_item = _item(
        "joint_state",
        AUDIT_AVAILABLE if replay_frames or joint_state else AUDIT_MISSING,
        replay_frames or joint_state,
        (
            "ReplayFrame stores joint positions."
            if replay_frames
            else (
                "JointState was recorded."
                if joint_state
                else "No joint state was recorded."
            )
        ),
        "" if replay_frames or joint_state else "Record joint positions at replay time.",
    )

    command_item = _item(
        "recorded_four_axis_command",
        AUDIT_AVAILABLE if replay_frames or command else AUDIT_MISSING,
        replay_frames or command,
        (
            "ReplayFrame stores controller output."
            if replay_frames
            else (
                "FourAxisCommand was recorded."
                if command
                else "No FourAxisCommand was recorded."
            )
        ),
        "" if replay_frames or command else "Record FourAxisCommand.",
    )

    gimbal = _present(
        inventory,
        (
            "/gimbalrotor/gimbals_ctrl",
            "/gimbalrotor/servo/target_states",
            "/gimbalrotor/mujoco/ctrl_input",
        ),
    )
    gimbal_item = _item(
        "recorded_gimbal_command",
        AUDIT_AVAILABLE if replay_frames or gimbal else AUDIT_MISSING,
        replay_frames or gimbal,
        (
            "ReplayFrame stores gimbal targets."
            if replay_frames
            else (
                "At least one gimbal command channel was recorded."
                if gimbal
                else "No gimbal command channel was recorded."
            )
        ),
        "" if replay_frames or gimbal else "Record the PC-side gimbal target.",
    )

    pid_item = _item(
        "pid_debug_term",
        AUDIT_AVAILABLE if replay_frames or pid_debug else AUDIT_MISSING,
        replay_frames or pid_debug,
        (
            "ReplayFrame stores PID terms."
            if replay_frames
            else (
                "PoseControlPid was recorded."
                if pid_debug
                else "No PID term channel was recorded."
            )
        ),
        "" if replay_frames or pid_debug else "Record per-axis PID terms.",
    )

    vectoring = _present(
        inventory, ("/gimbalrotor/debug/target_vectoring_force",)
    )
    vectoring_item = _item(
        "target_vectoring_force",
        AUDIT_AVAILABLE if replay_frames or vectoring else AUDIT_MISSING,
        replay_frames or vectoring,
        (
            "ReplayFrame stores target vectoring force."
            if replay_frames
            else (
                "Target vectoring force was recorded."
                if vectoring
                else "No target vectoring force was recorded."
            )
        ),
        "" if replay_frames or vectoring else "Record target vectoring force.",
    )

    torque_matrix = _present(
        inventory,
        (
            "/gimbalrotor/debug/torque_allocation_matrix",
            "/gimbalrotor/debug/torque_allocation_matrix_inverse",
        ),
    )
    torque_item = _item(
        "torque_allocation_matrix",
        AUDIT_AVAILABLE if replay_frames or torque_matrix else AUDIT_MISSING,
        replay_frames or torque_matrix,
        (
            "ReplayFrame stores the torque allocation matrix inverse."
            if replay_frames
            else (
                "Torque allocation matrix was recorded."
                if torque_matrix
                else "No torque allocation matrix channel was recorded."
            )
        ),
        "" if replay_frames or torque_matrix else "Record the per-tick allocation matrix.",
    )

    spinal = _present(
        inventory,
        (
            "/gimbalrotor/motor_pwms",
            "/gimbalrotor/rpy/pid",
            "/gimbalrotor/rpy/gain",
            "/gimbalrotor/servo/target_states",
            "/gimbalrotor/esc_telem",
        ),
    )
    spinal_item = _item(
        "spinal_pwm_channel",
        AUDIT_AVAILABLE if spinal else AUDIT_MISSING,
        spinal,
        (
            "Spinal/PWM-side command or telemetry channels were recorded."
            if spinal
            else "No spinal/PWM-side channel was recorded."
        ),
        "" if spinal else "Record motor PWM and spinal controller channels.",
        required=False,
    )

    fields = (
        controller_tick_item,
        estimator_item,
        navigator_item,
        control_mode_item,
        integration_item,
        force_landing_item,
        gain_item,
        nominal_model_item,
        joint_item,
        command_item,
        gimbal_item,
        pid_item,
        vectoring_item,
        torque_item,
        spinal_item,
    )
    return ControllerReplayAudit(
        episode_id=str(episode_id),
        bag_path=inventory.bag_path,
        source_bag_sha256=inventory.source_bag_sha256,
        topic_inventory_sha256=inventory.inventory_sha256,
        fields=fields,
    )


def build_replay_audit_bundle(
    audits: Sequence[ControllerReplayAudit],
) -> Dict[str, Any]:
    """Build a deterministic multi-bag controller_replay_audit.json payload."""

    ordered = tuple(sorted(audits, key=lambda item: item.episode_id))
    if not ordered:
        raise ValueError("replay audit bundle requires at least one audit")
    if len({item.episode_id for item in ordered}) != len(ordered):
        raise ValueError("replay audit bundle episode IDs must be unique")
    result: Dict[str, Any] = {
        "schema": CONTROLLER_REPLAY_AUDIT_BUNDLE_SCHEMA,
        "audit_count": len(ordered),
        "overall_exact_replay_ready": all(
            item.exact_replay_ready for item in ordered
        ),
        "overall_fixture_inputs_resolvable": all(
            item.fixture_inputs_resolvable for item in ordered
        ),
        "audits": [item.to_dict() for item in ordered],
    }
    result["bundle_sha256"] = stable_hash(result)
    return result


def write_replay_audit_bundle(
    bundle: Mapping[str, Any],
    path: Any,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write the canonical audit artifact without silent overwrite."""

    payload = dict(bundle)
    if payload.get("schema") != CONTROLLER_REPLAY_AUDIT_BUNDLE_SCHEMA:
        raise ValueError("unsupported replay audit bundle schema")
    expected_hash = payload.pop("bundle_sha256", None)
    if expected_hash != stable_hash(payload):
        raise ValueError("replay audit bundle hash mismatch")
    payload["bundle_sha256"] = expected_hash
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(destination))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


__all__ = [
    "AUDIT_AVAILABLE",
    "AUDIT_DERIVABLE",
    "AUDIT_MISSING",
    "CONTROLLER_REPLAY_AUDIT_BUNDLE_SCHEMA",
    "CONTROLLER_REPLAY_AUDIT_SCHEMA",
    "ControllerReplayAudit",
    "REPLAY_AUDIT_FIELDS",
    "ReplayAuditItem",
    "audit_controller_replay_inventory",
    "build_replay_audit_bundle",
    "write_replay_audit_bundle",
]
