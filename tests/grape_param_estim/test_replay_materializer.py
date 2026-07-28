import hashlib
from dataclasses import dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from grape_param_estim.alternative_backends import (
    ExactOracleReplayOutput,
)
from grape_param_estim.controller.exact_grid_alignment import (
    align_prepared_exact_grids,
)
from grape_param_estim.controller.exact_inputs import (
    load_controller_state_bundle,
    load_episode_conformance_bundle,
    load_fixture_bundle,
    load_snapshot_bundle,
)
from grape_param_estim.controller.snapshot import ControllerSnapshot
from grape_param_estim.data.controller_fixture import EpisodeTimeGrids
from grape_param_estim.data.event_scheduler import EventGrid
from grape_param_estim.data.replay_materializer import (
    CANONICAL_REPLAY_STREAM_SCHEMA,
    MATERIALIZED_EXACT_FILES,
    NOMINAL_GEOMETRY_SCHEMA,
    REPLAY_FRAME_SCHEMA,
    REPLAY_FRAME_TOPIC,
    REPLAY_FRAME_TYPE,
    REPLAY_METADATA_SCHEMA,
    REPLAY_METADATA_TOPIC,
    REPLAY_METADATA_TYPE,
    extract_canonical_replay_stream,
    materialize_exact_replay_inputs,
    replay_frame_message_to_mapping,
    replay_metadata_message_to_mapping,
)
from grape_param_estim.episode import stable_hash


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
_MOTORS = tuple(
    "allocation_rotor_index_{}".format(index) for index in range(4)
)


class _Stamp:
    def __init__(self, value):
        self.value = float(value)

    def to_sec(self):
        return self.value


@dataclass(frozen=True)
class _Prepared:
    config: object
    grids: EpisodeTimeGrids
    observations: object
    data: object = None
    nuisance_samples: tuple = ()
    commands: object = None
    trajectory_posterior: object = None


def _geometry():
    rotations = [
        np.eye(3, dtype=float).tolist() for _ in range(4)
    ]
    return {
        "mass": 2.0,
        "inertia": np.diag((0.2, 0.3, 0.4)).tolist(),
        "moment_force_rate": 0.01,
        "rotor_origins_from_cog": [
            [0.2, 0.2, 0.0],
            [-0.2, 0.2, 0.0],
            [-0.2, -0.2, 0.0],
            [0.2, -0.2, 0.0],
        ],
        "rotor_directions": [1, -1, 1, -1],
        "thrust_coordinate_rotations": rotations,
        "gravity": 9.80665,
    }


def _flatten_geometry(geometry):
    pieces = (
        ("mass", [geometry["mass"]]),
        (
            "inertia_row_major",
            np.asarray(geometry["inertia"]).reshape(-1).tolist(),
        ),
        (
            "rotor_origins_from_cog_xyz",
            np.asarray(
                geometry["rotor_origins_from_cog"]
            ).reshape(-1).tolist(),
        ),
        ("rotor_directions", geometry["rotor_directions"]),
        ("moment_force_rate", [geometry["moment_force_rate"]]),
        (
            "thrust_coordinate_rotations_row_major",
            np.asarray(
                geometry["thrust_coordinate_rotations"]
            ).reshape(-1).tolist(),
        ),
    )
    names = []
    offsets = []
    values = []
    for name, row in pieces:
        names.append(name)
        offsets.append(len(values))
        values.extend(row)
    return names, offsets, values


def _snapshot_metadata(
    artifact_sha256,
    *,
    stamp,
    gain_delta=0.0,
    limit_delta=0.0,
    parameter_dump_sha256="d" * 64,
):
    geometry = _geometry()
    gains = {
        "axis_names": _AXES,
        "p_gain": tuple(
            1.0 + gain_delta + index for index in range(6)
        ),
        "i_gain": tuple(0.1 + index for index in range(6)),
        "d_gain": tuple(0.01 + index for index in range(6)),
    }
    limits = {
        field: tuple(
            10.0 + limit_delta + field_index + axis
            for axis in range(6)
        )
        for field_index, field in enumerate(_LIMIT_FIELDS)
    }
    options = {
        "gimbal_dof": 1,
        "gimbal_calc_in_fc": False,
        "hovering_approximate": False,
        "underactuate": False,
        "need_yaw_d_control": True,
        "integration_start_height": 0.2,
        "force_landing_descending_rate": 0.1,
        "estimate_mode": 0,
        "gravity": geometry["gravity"],
    }
    snapshot = ControllerSnapshot(
        backend_id="gimbalrotor_pc_exact_test",
        source_commit="0123456789abcdef",
        artifact_sha256=artifact_sha256,
        nominal_model_sha256="b" * 64,
        parameter_dump_sha256=parameter_dump_sha256,
        controller_rate_hz=10.0,
        gains=gains,
        limits=limits,
        static_options=options,
        nominal_mass=geometry["mass"],
        nominal_cog=(0.0, 0.0, 0.0),
        nominal_inertia=geometry["inertia"],
        nominal_geometry=geometry,
    )
    gain_names = []
    gain_values = []
    for axis_index, axis in enumerate(_AXES):
        for field in _GAIN_FIELDS:
            gain_names.append("{}.{}".format(axis, field))
            gain_values.append(gains[field][axis_index])
    limit_names = []
    limit_values = []
    for axis_index, axis in enumerate(_AXES):
        for field in _LIMIT_FIELDS:
            limit_names.append("{}.{}".format(axis, field))
            limit_values.append(limits[field][axis_index])
    geometry_names, geometry_offsets, geometry_values = (
        _flatten_geometry(geometry)
    )
    metadata = {
        "schema": REPLAY_METADATA_SCHEMA,
        "stamp": float(stamp),
        "frame_id": "world",
        "source_commit": snapshot.source_commit,
        "backend_id": snapshot.backend_id,
        "fidelity": "pc_exact",
        "controller_artifact_sha256": artifact_sha256,
        "controller_snapshot_sha256": snapshot.snapshot_id,
        "nominal_model_sha256": snapshot.nominal_model_sha256,
        "parameter_dump_sha256": parameter_dump_sha256,
        "nominal_geometry_sha256": stable_hash(
            {
                "schema": NOMINAL_GEOMETRY_SCHEMA,
                "geometry": geometry,
            }
        ),
        "controller_rate_hz": snapshot.controller_rate_hz,
        "gain_names": gain_names,
        "gain_values": gain_values,
        "limit_names": limit_names,
        "limit_values": limit_values,
        "static_option_names": [
            "gimbal_dof",
            "gimbal_calc_in_fc",
            "hovering_approximate",
            "underactuate",
            "need_yaw_d_control",
            "start_roll_pitch_integration_height",
            "force_landing_descending_rate",
            "estimate_mode",
            "gravity",
        ],
        "static_option_values": [
            "1",
            "false",
            "false",
            "false",
            "true",
            "0.2",
            "0.1",
            "0",
            "9.80665",
        ],
        "motor_order": list(_MOTORS),
        "nominal_mass": geometry["mass"],
        "nominal_cog": [0.0, 0.0, 0.0],
        "nominal_inertia": (
            np.asarray(geometry["inertia"]).reshape(-1).tolist()
        ),
        "geometry_names": geometry_names,
        "geometry_offsets": geometry_offsets,
        "geometry_values": geometry_values,
    }
    return snapshot, metadata


def _frame(stamp, snapshot_sha256, before_stamp, after_stamp):
    geometry_names, geometry_offsets, geometry_values = (
        _flatten_geometry(_geometry())
    )
    return {
        "schema": REPLAY_FRAME_SCHEMA,
        "stamp": float(stamp),
        "frame_id": "world",
        "controller_snapshot_sha256": snapshot_sha256,
        "dt": 0.1,
        "position": [0.0, 0.0, 1.0],
        "velocity": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "angular_velocity": [0.0, 0.0, 0.0],
        "target_position": [0.0, 0.0, 1.0],
        "target_velocity": [0.0, 0.0, 0.0],
        "target_acceleration": [0.0, 0.0, 0.0],
        "target_orientation": [0.0, 0.0, 0.0, 1.0],
        "target_rpy": [0.0, 0.0, 0.0],
        "target_angular_velocity": [0.0, 0.0, 0.0],
        "target_angular_acceleration": [0.0, 0.0, 0.0],
        "control_mode": [0] * 6,
        "integration_enabled": [True] * 6,
        "flight_state": 1,
        "force_landing": False,
        "reset": False,
        "initial_height": 0.0,
        "joint_names": ["gimbal{}".format(i) for i in range(1, 5)],
        "joint_positions": [0.0] * 4,
        "allocation_geometry_names": geometry_names,
        "allocation_geometry_offsets": geometry_offsets,
        "allocation_geometry_values": geometry_values,
        "pid_state_before": [0.0] * 48,
        "pid_state_after": [0.0] * 48,
        "start_roll_pitch_integration_before": True,
        "start_roll_pitch_integration_after": True,
        "previous_stamp_before": float(before_stamp),
        "previous_stamp_after": float(after_stamp),
        "previous_flight_state_before": 1,
        "previous_flight_state_after": 1,
        "previous_control_mode_before": [0] * 6,
        "previous_control_mode_after": [0] * 6,
        "previous_force_landing_before": False,
        "previous_force_landing_after": False,
        "has_previous_force_landing_before": False,
        "has_previous_force_landing_after": False,
        "pending_events_before": 0,
        "pending_events_after": 0,
        "target_gimbal_angles_before": [0.0] * 4,
        "target_gimbal_angles_after": [0.0] * 4,
        "target_roll_before": 0.0,
        "target_roll_after": 0.0,
        "target_pitch_before": 0.0,
        "target_pitch_after": 0.0,
        "current_rpy": [0.0, 0.0, 0.0],
        "pid_result": [0.0] * 6,
        "pid_p_term": [0.0] * 6,
        "pid_i_term": [0.0] * 6,
        "pid_d_term": [0.0] * 6,
        "target_vectoring_force": [0.0] * 4,
        "four_axis_angles": [0.0] * 3,
        "base_thrust": [1.0] * 4,
        "gimbal_angle": [0.0] * 4,
        "torque_allocation_rows": 4,
        "torque_allocation_columns": 3,
        "torque_allocation_matrix_inverse": [0.0] * 12,
        "target_roll": 0.0,
        "target_pitch": 0.0,
        "candidate_yaw_term": 0.0,
        "effective_target_acceleration": [0.0, 0.0, 0.0],
        "saturated": False,
        "events": [],
    }


def _ros_geometry(values):
    names = ("x", "y", "z", "w")[: len(values)]
    return SimpleNamespace(**dict(zip(names, values)))


def _ros_metadata(values, origin=100.0):
    fields = {
        key: value
        for key, value in values.items()
        if key not in ("stamp", "frame_id", "nominal_cog")
    }
    fields.update(
        {
            "_type": REPLAY_METADATA_TYPE,
            "header": SimpleNamespace(
                stamp=_Stamp(origin + values["stamp"]),
                frame_id=values["frame_id"],
            ),
            "nominal_cog": _ros_geometry(values["nominal_cog"]),
        }
    )
    return SimpleNamespace(**fields)


def _ros_frame(values, origin=100.0):
    vector_fields = (
        "position",
        "velocity",
        "orientation",
        "angular_velocity",
        "target_position",
        "target_velocity",
        "target_acceleration",
        "target_orientation",
        "target_rpy",
        "target_angular_velocity",
        "target_angular_acceleration",
        "current_rpy",
        "effective_target_acceleration",
    )
    fields = {
        key: value
        for key, value in values.items()
        if key not in ("stamp", "frame_id") + vector_fields
    }
    for name in vector_fields:
        fields[name] = _ros_geometry(values[name])
    for suffix in ("before", "after"):
        name = "previous_stamp_{}".format(suffix)
        if fields[name] != 0.0:
            fields[name] += origin
    fields.update(
        {
            "_type": REPLAY_FRAME_TYPE,
            "header": SimpleNamespace(
                stamp=_Stamp(origin + values["stamp"]),
                frame_id=values["frame_id"],
            ),
        }
    )
    return SimpleNamespace(**fields)


def _prepared(source_sha256):
    grids = EpisodeTimeGrids(
        EventGrid("controller_tick", (0.0, 0.2)),
        EventGrid(
            "plant_integration", (0.0, 0.05, 0.1, 0.15, 0.2)
        ),
        EventGrid("observation", (0.0, 0.1, 0.2)),
        EventGrid("likelihood", (0.1, 0.2)),
        EventGrid("report", (0.1, 0.2)),
    )
    return {
        "episode-1": _Prepared(
            config={
                "episode_id": "episode-1",
                "replay_start_offset_s": 0.0,
                "score_start_offset_s": 0.1,
                "score_end_offset_s": 0.2,
            },
            grids=grids,
            observations=SimpleNamespace(
                source_bag_sha256=source_sha256,
                normalized_episode_sha256="9" * 64,
            ),
        )
    }


def _irregular_prepared(source_sha256, **overrides):
    grids = EpisodeTimeGrids(
        EventGrid("controller_tick", (0.02, 0.18)),
        EventGrid(
            "plant_integration", (0.02, 0.05, 0.1, 0.15, 0.18)
        ),
        EventGrid("observation", (0.02, 0.1, 0.18)),
        EventGrid("likelihood", (0.12, 0.18)),
        EventGrid("report", (0.12, 0.18)),
    )
    values = {
        "config": {
            "episode_id": "episode-1",
            "replay_start_offset_s": 0.02,
            "score_start_offset_s": 0.12,
            "score_end_offset_s": 0.18,
        },
        "grids": grids,
        "observations": SimpleNamespace(
            source_bag_sha256=source_sha256,
            normalized_episode_sha256="9" * 64,
        ),
    }
    values.update(overrides)
    return {"episode-1": _Prepared(**values)}


def _stream(source_sha256, metadata, frames):
    payload = {
        "schema": CANONICAL_REPLAY_STREAM_SCHEMA,
        "episodes": [
            {
                "episode_id": "episode-1",
                "source_bag_sha256": source_sha256,
                "topic_inventory_sha256": "8" * 64,
                "replay_start_offset_s": 0.0,
                "score_start_offset_s": 0.1,
                "score_end_offset_s": 0.2,
                "metadata_records": metadata,
                "frames": frames,
                "source_topics": [
                    REPLAY_METADATA_TOPIC,
                    REPLAY_FRAME_TOPIC,
                ],
                "frame_conventions": {
                    "event_time": "message.header.stamp",
                },
                "unit_conventions": {"time": "s"},
                "motor_order": list(_MOTORS),
            }
        ],
    }
    return {
        **payload,
        "content_sha256": stable_hash(payload),
    }


def _continuous(stamps=(0.0, 0.1, 0.2)):
    count = len(stamps)
    return {
        "command_timestamp": np.asarray(
            [[float(stamp)] for stamp in stamps]
        ),
        "pid_terms": np.zeros((count, 24)),
        "four_axis_command": np.asarray(
            [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]] * count
        ),
        "vectoring_force": np.zeros((count, 4)),
        "gimbal_command": np.zeros((count, 4)),
        "allocation_internal": np.zeros((count, 3)),
        "torque_allocation_matrix_inverse": np.zeros((count, 12)),
    }


def _check_message_conversion_keeps_header_and_record_time_distinct(
    tmp_path,
):
    executable = tmp_path / "oracle"
    executable.write_bytes(b"fake exact oracle")
    artifact = hashlib.sha256(executable.read_bytes()).hexdigest()
    _, metadata = _snapshot_metadata(artifact, stamp=0.2)
    frame = _frame(0.2, metadata["controller_snapshot_sha256"], 0.1, 0.2)

    converted_metadata = replay_metadata_message_to_mapping(
        _ros_metadata(metadata),
        bag_start_time_s=100.0,
        record_time_s=_Stamp(100.25),
    )
    converted_frame = replay_frame_message_to_mapping(
        _ros_frame(frame),
        bag_start_time_s=100.0,
        record_time_s=_Stamp(100.27),
    )

    assert np.isclose(converted_metadata["stamp"], 0.2)
    assert np.isclose(converted_metadata["header_time_s"], 100.2)
    assert converted_metadata["bag_record_time_s"] == 100.25
    assert converted_metadata["event_time_source"] == "message_header_stamp"
    assert np.isclose(converted_frame["stamp"], 0.2)
    assert converted_frame["bag_record_time_s"] == 100.27
    assert np.isclose(converted_frame["previous_stamp_before"], 0.1)
    assert np.isclose(converted_frame["previous_stamp_after"], 0.2)

    latched = replay_metadata_message_to_mapping(
        _ros_metadata(metadata, origin=99.5),
        bag_start_time_s=100.0,
        record_time_s=_Stamp(100.01),
    )
    assert np.isclose(latched["stamp"], -0.3)
    assert latched["header_predates_bag_origin"] is True
    assert latched["bag_record_time_s"] == 100.01

    pre_origin_frame = _ros_frame(frame, origin=99.5)
    try:
        replay_frame_message_to_mapping(
            pre_origin_frame,
            bag_start_time_s=100.0,
            record_time_s=_Stamp(100.02),
        )
    except ValueError as exc:
        assert "predates the bag record-time origin" in str(exc)
    else:
        raise AssertionError("pre-origin ReplayFrame Header was accepted")

    try:
        replay_metadata_message_to_mapping(
            _ros_metadata(metadata),
            bag_start_time_s=100.0,
            record_time_s=_Stamp(99.9),
        )
    except ValueError as exc:
        assert "bag record stamp predates" in str(exc)
    else:
        raise AssertionError("invalid pre-origin bag record stamp was accepted")

    broken = _ros_frame(frame)
    broken.header.stamp = _Stamp(0.0)
    try:
        replay_frame_message_to_mapping(
            broken,
            bag_start_time_s=100.0,
            record_time_s=_Stamp(100.3),
        )
    except ValueError as exc:
        assert "fallback is forbidden" in str(exc)
    else:
        raise AssertionError("zero Header timestamp was accepted")


def _check_direct_bag_extraction_and_materializer_loader_alignment_chain(
    tmp_path,
):
    executable = tmp_path / "oracle"
    executable.write_bytes(b"fake exact oracle")
    executable.chmod(0o755)
    artifact = hashlib.sha256(executable.read_bytes()).hexdigest()
    _, first_metadata = _snapshot_metadata(artifact, stamp=-0.02)
    _, changed_metadata = _snapshot_metadata(
        artifact,
        stamp=0.1,
        gain_delta=0.5,
        limit_delta=1.0,
        parameter_dump_sha256="e" * 64,
    )
    frame_times = (0.01, 0.05, 0.1, 0.15, 0.19)
    frames = [
        _frame(
            0.01,
            first_metadata["controller_snapshot_sha256"],
            0.0,
            0.01,
        ),
        _frame(
            0.05,
            first_metadata["controller_snapshot_sha256"],
            0.01,
            0.05,
        ),
        _frame(
            0.1,
            changed_metadata["controller_snapshot_sha256"],
            0.05,
            0.1,
        ),
        _frame(
            0.15,
            changed_metadata["controller_snapshot_sha256"],
            0.1,
            0.15,
        ),
        _frame(
            0.19,
            changed_metadata["controller_snapshot_sha256"],
            0.15,
            0.19,
        ),
    ]
    source_sha256 = "7" * 64
    prepared = _irregular_prepared(source_sha256)
    bag_path = tmp_path / "episode.bag"
    bag_path.write_bytes(b"fixture")
    config = {
        "episodes": [
            {
                "episode_id": "episode-1",
                "bag": bag_path.name,
                "source_bag_sha256": source_sha256,
                "replay_start_offset_s": 0.02,
                "score_start_offset_s": 0.12,
                "score_end_offset_s": 0.18,
            }
        ]
    }
    records = [
        (
            REPLAY_METADATA_TOPIC,
            _ros_metadata(first_metadata),
            _Stamp(100.001),
        ),
        (
            REPLAY_FRAME_TOPIC,
            _ros_frame(frames[0]),
            _Stamp(100.002),
        ),
        (
            REPLAY_FRAME_TOPIC,
            _ros_frame(frames[1]),
            _Stamp(100.052),
        ),
        (
            REPLAY_METADATA_TOPIC,
            _ros_metadata(changed_metadata),
            _Stamp(100.101),
        ),
        (
            REPLAY_FRAME_TOPIC,
            _ros_frame(frames[2]),
            _Stamp(100.102),
        ),
        (
            REPLAY_FRAME_TOPIC,
            _ros_frame(frames[3]),
            _Stamp(100.152),
        ),
        (
            REPLAY_FRAME_TOPIC,
            _ros_frame(frames[4]),
            _Stamp(100.192),
        ),
    ]

    class FakeBag:
        def __init__(self, path, mode):
            assert Path(path) == bag_path
            assert mode == "r"

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            return None

        def get_start_time(self):
            return 100.0

        def read_messages(self, topics):
            assert tuple(topics) == (
                REPLAY_METADATA_TOPIC,
                REPLAY_FRAME_TOPIC,
            )
            return iter(records)

    inventory = SimpleNamespace(
        source_bag_sha256=source_sha256,
        inventory_sha256="8" * 64,
        topics={
            REPLAY_METADATA_TOPIC: SimpleNamespace(
                message_type=REPLAY_METADATA_TYPE,
                message_count=2,
            ),
            REPLAY_FRAME_TOPIC: SimpleNamespace(
                message_type=REPLAY_FRAME_TYPE,
                message_count=5,
            ),
        },
    )
    with mock.patch.dict(
        "sys.modules", {"rosbag": SimpleNamespace(Bag=FakeBag)}
    ), mock.patch(
        "grape_param_estim.data.replay_materializer."
        "read_bag_topic_inventory",
        return_value=inventory,
    ):
        canonical = extract_canonical_replay_stream(tmp_path, config)

    episode = canonical["episodes"][0]
    assert episode["time_semantics"]["record_time_is_fallback"] is False
    assert tuple(item["stamp"] for item in episode["frames"]) == frame_times
    assert episode["metadata_records"][0]["stamp"] == -0.02
    assert episode["metadata_records"][0][
        "header_predates_bag_origin"
    ] is True
    assert episode["frames"][2]["bag_record_time_s"] == 100.102

    continuous = _continuous(frame_times)

    class FakeOracle:
        is_exact = True

        def __init__(self, command, identity, timeout_s):
            assert hashlib.sha256(
                Path(command[0]).read_bytes()
            ).hexdigest() == identity.artifact_sha256
            assert timeout_s == 2.0
            self.identity = identity
            self.closed = False

        def replay(self, payload):
            ticks = payload["jobs"][0]["ticks"]
            assert ticks[2]["pid_config"] == [
                [
                    changed_metadata["gain_values"][
                        changed_metadata["gain_names"].index(
                            "{}.{}".format(axis, field)
                        )
                    ]
                    if field in _GAIN_FIELDS
                    else changed_metadata["limit_values"][
                        changed_metadata["limit_names"].index(
                            "{}.{}".format(axis, field)
                        )
                    ]
                    for field in _GAIN_FIELDS + _LIMIT_FIELDS
                ]
                for axis in _AXES
            ]
            assert "pid_config" not in ticks[0]
            assert "pid_config" not in ticks[1]
            assert "pid_config" not in ticks[3]
            assert "pid_config" not in ticks[4]
            return ExactOracleReplayOutput(
                identity=self.identity,
                continuous=continuous,
                events=np.zeros(5, dtype=np.uint32),
            )

        def close(self):
            self.closed = True

    destination = materialize_exact_replay_inputs(
        stream_path=canonical,
        prepared=prepared,
        assimilation_config_sha256="6" * 64,
        exact_replay_executable=executable,
        output_root=tmp_path / "runs",
        run_id="exact-1",
        timeout_s=2.0,
        oracle_factory=FakeOracle,
    )
    assert tuple(
        path.name for path in sorted(destination.iterdir())
    ) == tuple(sorted(MATERIALIZED_EXACT_FILES))

    fixtures, _ = load_fixture_bundle(
        destination / MATERIALIZED_EXACT_FILES[0]
    )
    snapshots, _ = load_snapshot_bundle(
        destination / MATERIALIZED_EXACT_FILES[1]
    )
    states, _ = load_controller_state_bundle(
        destination / MATERIALIZED_EXACT_FILES[2]
    )
    conformance = load_episode_conformance_bundle(
        destination / MATERIALIZED_EXACT_FILES[5],
        fixtures,
        snapshots,
    )
    aligned = align_prepared_exact_grids(prepared, fixtures)

    assert set(states) == {"episode-1"}
    assert set(conformance.episodes) == {"episode-1"}
    fixture = fixtures["episode-1"]
    assert (
        fixture.metadata["normalized_episode_sha256"] == "9" * 64
    )
    assert fixture.metadata["assimilation_config_sha256"] == "6" * 64
    assert (
        aligned["episode-1"].grids.controller_tick_grid.timestamps
        == (0.05, 0.1, 0.15)
    )
    assert (
        aligned["episode-1"].grids.plant_integration_grid.timestamps
        == (0.05, 0.1, 0.15, 0.18)
    )
    assert fixture.factual_controller_tick_grid.timestamps == frame_times
    assert len(fixture.controller_inputs) == 5
    assert fixture.metadata["controller_tick_domains"] == {
        "schema": "grape_controller_tick_domains/v1",
        "factual_replay_tick_grid": frame_times,
        "inference_controller_tick_grid": (0.05, 0.1, 0.15),
        "pre_replay_boundary_tick_s": 0.01,
        "post_score_boundary_tick_s": 0.19,
        "inference_is_clipped_to_prepared_plant_support": True,
    }

    old_nuisance = object()
    new_nuisance = object()
    stateful_prepared = _irregular_prepared(
        source_sha256,
        nuisance_samples=(old_nuisance,),
        trajectory_posterior=SimpleNamespace(
            timestamps=np.asarray((0.02, 0.05, 0.1, 0.18))
        ),
    )
    with mock.patch(
        "grape_param_estim.controller.exact_grid_alignment."
        "initial_state_posterior",
        return_value=SimpleNamespace(samples=(new_nuisance,)),
    ) as derive:
        state_aligned = align_prepared_exact_grids(
            stateful_prepared, fixtures
        )
    assert state_aligned["episode-1"].nuisance_samples == (
        new_nuisance,
    )
    assert derive.call_args.args[2] == 0.05
    assert derive.call_args.kwargs["maximum_samples"] == 1

    try:
        materialize_exact_replay_inputs(
            stream_path=canonical,
            prepared=prepared,
            assimilation_config_sha256="6" * 64,
            exact_replay_executable=executable,
            output_root=tmp_path / "runs",
            run_id="exact-1",
            timeout_s=2.0,
            oracle_factory=FakeOracle,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("materializer overwrote an existing run")


def _check_geometry_identity_mismatch_is_rejected(tmp_path):
    executable = tmp_path / "oracle"
    executable.write_bytes(b"fake exact oracle")
    artifact = hashlib.sha256(executable.read_bytes()).hexdigest()
    _, metadata = _snapshot_metadata(artifact, stamp=0.0)
    metadata["nominal_geometry_sha256"] = "f" * 64
    frames = [
        _frame(0.0, metadata["controller_snapshot_sha256"], 0.0, 0.0),
        _frame(0.2, metadata["controller_snapshot_sha256"], 0.0, 0.2),
    ]
    canonical = _stream("7" * 64, [metadata], frames)
    try:
        materialize_exact_replay_inputs(
            stream_path=canonical,
            prepared=_prepared("7" * 64),
            assimilation_config_sha256="6" * 64,
            exact_replay_executable=executable,
            output_root=tmp_path / "runs",
            run_id="bad",
            oracle_factory=lambda *args, **kwargs: None,
        )
    except ValueError as exc:
        assert "geometry hash" in str(exc)
    else:
        raise AssertionError("invalid nominal geometry identity was accepted")
    assert not (tmp_path / "runs" / "bad").exists()


class ReplayMaterializerTest(unittest.TestCase):
    def test_message_conversion_keeps_header_and_record_time_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_message_conversion_keeps_header_and_record_time_distinct(
                Path(directory)
            )

    def test_direct_bag_to_materializer_loader_alignment_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_direct_bag_extraction_and_materializer_loader_alignment_chain(
                Path(directory)
            )

    def test_geometry_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_geometry_identity_mismatch_is_rejected(
                Path(directory)
            )


if __name__ == "__main__":
    unittest.main()
