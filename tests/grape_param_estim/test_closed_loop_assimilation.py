import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import yaml

from grape_param_estim.alternative_backends import (
    EXACT_ORACLE_PROTOCOL,
    PC_EXACT_ORACLE_CAPABILITIES,
    ExactOracleConformanceFixture,
    ExactOracleFixtureProvenance,
    ExactOracleIdentity,
    ExactOracleReplayOutput,
    evaluate_exact_oracle_conformance,
    required_conformance_channels,
)
from grape_param_estim.controller import (
    ControllerCoreInput,
    ControllerCoreState,
    ControllerSnapshot,
    ControllerStaticOptions,
    evaluate_exact_closed_loop_gate,
)
from grape_param_estim.controller.external_oracle import (
    StatefulExactOracleControllerBackend,
    build_exact_replay_payload,
    controller_backend_identity,
)
from grape_param_estim.controller.exact_inputs import (
    ExactEpisodeConformanceBundle,
    ExactEpisodeConformanceEvidence,
)
from grape_param_estim.data import (
    ControllerReplayFixture,
    EpisodeTimeGrids,
    EventGrid,
)
from grape_param_estim.forward import ClosedLoopGateError
from grape_param_estim.episode import stable_hash
from grape_param_estim.inference import (
    ControllerEventObservations,
    ObservationDataset,
)
from grape_param_estim.output.manifest import verify_run_manifest
from grape_param_estim.plant import EpisodeNuisance
from grape_param_estim.plant_assimilation import (
    CLOSED_LOOP_PLANT_BACKEND_ID,
    CLOSED_LOOP_POSTERIOR_MODEL_ID,
    ExactClosedLoopDependencies,
    PreparedEpisode,
    validate_posterior,
    write_assimilation_run,
)


REPOSITORY = Path(__file__).resolve().parents[2]
CURRENT_CONFIG = (
    REPOSITORY
    / "ros/examples/grape-param-estim/config/plant_assimilation.yaml"
)
EPISODE_ID = "synthetic-exact"
SOURCE_BAG_SHA256 = "1" * 64
NORMALIZED_EPISODE_SHA256 = "2" * 64
SUCCESS_EPISODE_ID = "synthetic-success"
SUCCESS_SOURCE_BAG_SHA256 = "4" * 64
SUCCESS_NORMALIZED_EPISODE_SHA256 = "5" * 64


def _identity():
    return controller_backend_identity(_oracle_identity())


def _oracle_identity():
    return ExactOracleIdentity(
        protocol=EXACT_ORACLE_PROTOCOL,
        backend_id="gimbalrotor_controller_cpp/v2",
        implementation_language="c++",
        source_commit="2786cc3e",
        artifact_sha256="a" * 64,
        capabilities=PC_EXACT_ORACLE_CAPABILITIES,
        fidelity="pc_exact",
    )


def _factual_evidence(identity, timestamp_delta=0.0):
    payload = {"factual_replay": "closed-loop-assimilation-test"}
    continuous = {
        channel: (
            np.asarray([[1.0]])
            if channel == "command_timestamp"
            else np.zeros((1, 1))
        )
        for channel in required_conformance_channels(identity.fidelity)
    }
    events = np.zeros(1, dtype=int)
    provenance = ExactOracleFixtureProvenance.create(
        source_bag_sha256=SOURCE_BAG_SHA256,
        source_topics=("/controller/factual_fixture",),
        interval_start_time_ns=1,
        interval_end_time_ns=2,
        frame_conventions={"controller": "body_flu"},
        unit_conventions={"controller": "SI"},
        motor_order=("1", "2", "3", "4"),
        request_payload=payload,
        continuous=continuous,
        events=events,
        extraction_config_sha256="e" * 64,
        source_commit="fixture-source",
    )
    fixture = ExactOracleConformanceFixture(
        continuous=continuous,
        events=events,
        provenance=provenance,
        fidelity=identity.fidelity,
    )

    class Oracle:
        is_exact = True

        def __init__(self):
            self.identity = identity

        def replay(self, request):
            replayed = dict(continuous)
            replayed["command_timestamp"] = (
                continuous["command_timestamp"]
                + float(timestamp_delta)
            )
            return ExactOracleReplayOutput(
                identity=self.identity,
                continuous=replayed,
                events=events,
            )

    return evaluate_exact_oracle_conformance(
        Oracle(), payload, fixture
    )


def _episode_evidence(
    identity,
    fixture,
    snapshot,
    initial_state,
    timestamp_delta=0.0,
):
    request = build_exact_replay_payload(
        snapshot,
        initial_state,
        fixture.controller_inputs,
        evidence_binding={
            "schema": "grape_exact_episode_request_binding/v1",
            "episode_id": fixture.episode_id,
            "source_bag_sha256": fixture.source_bag_sha256,
            "controller_replay_fixture_sha256": fixture.fixture_sha256,
            "controller_input_sha256": stable_hash(
                {
                    "schema": (
                        "grape_exact_episode_controller_inputs/v1"
                    ),
                    "episode_id": fixture.episode_id,
                    "controller_inputs": [
                        item.to_mapping()
                        for item in fixture.controller_inputs
                    ],
                }
            ),
            "controller_snapshot_sha256": snapshot.snapshot_id,
        },
    )
    count = len(fixture.controller_inputs)
    widths = {
        "command_timestamp": 1,
        "pid_terms": 24,
        "four_axis_command": 7,
        "vectoring_force": 8,
        "gimbal_command": 4,
        "allocation_internal": 3,
        "torque_allocation_matrix_inverse": 24,
        "pwm": 4,
    }
    continuous = {
        channel: (
            np.asarray(
                [[item.stamp] for item in fixture.controller_inputs],
                dtype=float,
            )
            if channel == "command_timestamp"
            else np.zeros((count, widths[channel]))
        )
        for channel in required_conformance_channels(identity.fidelity)
    }
    events = np.zeros(count, dtype=int)
    provenance = ExactOracleFixtureProvenance.create(
        source_bag_sha256=fixture.source_bag_sha256,
        source_topics=("/controller/factual_fixture",),
        interval_start_time_ns=1,
        interval_end_time_ns=max(2, count),
        frame_conventions={"controller": "body_flu"},
        unit_conventions={"controller": "SI"},
        motor_order=("1", "2", "3", "4"),
        request_payload=request,
        continuous=continuous,
        events=events,
        extraction_config_sha256="e" * 64,
        source_commit="fixture-source",
    )
    conformance_fixture = ExactOracleConformanceFixture(
        continuous=continuous,
        events=events,
        provenance=provenance,
        fidelity=identity.fidelity,
    )

    class Oracle:
        is_exact = True

        def __init__(self):
            self.identity = identity

        def replay(self, payload):
            replayed = dict(continuous)
            replayed["command_timestamp"] = (
                continuous["command_timestamp"]
                + float(timestamp_delta)
            )
            return ExactOracleReplayOutput(
                identity=self.identity,
                continuous=replayed,
                events=events,
            )

    report = evaluate_exact_oracle_conformance(
        Oracle(), request, conformance_fixture
    )
    return ExactEpisodeConformanceEvidence.create(
        fixture=fixture,
        snapshot=snapshot,
        initial_controller_state=initial_state,
        conformance_fixture=conformance_fixture,
        conformance_report=report,
    )


def _conformance_bundle(
    identity,
    fixtures,
    snapshots,
    initial_state=None,
):
    state = _controller_state() if initial_state is None else initial_state
    return ExactEpisodeConformanceBundle(
        episodes={
            episode_id: _episode_evidence(
                identity,
                fixture,
                snapshots[episode_id],
                state,
            )
            for episode_id, fixture in fixtures.items()
        }
    )


def _snapshot(identity):
    identity_matrix = np.eye(3).tolist()
    return ControllerSnapshot(
        backend_id=identity.backend_id,
        source_commit=identity.source_commit,
        artifact_sha256=identity.artifact_sha256,
        nominal_model_sha256="b" * 64,
        parameter_dump_sha256="c" * 64,
        controller_rate_hz=10.0,
        gains={
            "p_gain": [1.0] * 6,
            "i_gain": [0.0] * 6,
            "d_gain": [0.0] * 6,
        },
        limits={
            "limit_sum": [100.0] * 6,
            "limit_p": [100.0] * 6,
            "limit_i": [100.0] * 6,
            "limit_d": [100.0] * 6,
            "limit_err_p": [100.0] * 6,
            "limit_err_i": [100.0] * 6,
            "limit_err_d": [100.0] * 6,
        },
        static_options=ControllerStaticOptions(
            gimbal_dof=1,
            gimbal_calc_in_fc=False,
            hovering_approximate=False,
            underactuate=False,
            need_yaw_d_control=True,
            integration_start_height=0.01,
            force_landing_descending_rate=-0.1,
            estimate_mode=0,
        ),
        nominal_mass=2.0,
        nominal_cog=[0.0, 0.0, 0.0],
        nominal_inertia=np.diag([0.3, 0.4, 0.5]),
        nominal_geometry={
            "gravity": 9.797,
            "moment_force_rate": 0.01,
            "rotor_origins_from_cog": [
                [0.3, 0.2, 0.0],
                [-0.3, 0.2, 0.0],
                [-0.3, -0.2, 0.0],
                [0.3, -0.2, 0.0],
            ],
            "rotor_directions": [1, -1, 1, -1],
            "thrust_coordinate_rotations": [
                identity_matrix,
                identity_matrix,
                identity_matrix,
                identity_matrix,
            ],
        },
    )


def _core_input(stamp):
    roll_pitch_enabled = float(stamp) > 0.0
    return ControllerCoreInput(
        stamp=stamp,
        dt=0.1,
        position=[0.0, 0.0, 1.0],
        velocity=[0.0, 0.0, 0.0],
        orientation=np.eye(3),
        angular_velocity=[0.0, 0.0, 0.0],
        target_position=[0.0, 0.0, 1.0],
        target_velocity=[0.0, 0.0, 0.0],
        target_acceleration=[0.0, 0.0, 0.0],
        target_orientation=np.eye(3),
        target_angular_velocity=[0.0, 0.0, 0.0],
        target_angular_acceleration=[0.0, 0.0, 0.0],
        control_mode=[0] * 6,
        integration_enabled=[
            True,
            True,
            True,
            roll_pitch_enabled,
            roll_pitch_enabled,
            True,
        ],
        flight_state=5,
        force_landing=False,
        current_rpy=[0.0, 0.0, 0.0],
        target_rpy=[0.0, 0.0, 0.0],
    )


def _grids():
    return EpisodeTimeGrids(
        controller_tick_grid=EventGrid(
            "controller_tick", (0.0, 0.1, 0.2)
        ),
        plant_integration_grid=EventGrid(
            "plant_integration", (0.0, 0.1, 0.2)
        ),
        observation_grid=EventGrid(
            "observation", (0.0, 0.1, 0.2)
        ),
        likelihood_grid=EventGrid("likelihood", (0.1, 0.2)),
        report_grid=EventGrid("report", (0.1, 0.2)),
    )


def _fixture(
    episode_id=EPISODE_ID,
    source_bag_sha256=SOURCE_BAG_SHA256,
):
    return ControllerReplayFixture(
        episode_id=episode_id,
        source_bag_sha256=source_bag_sha256,
        topic_inventory_sha256="3" * 64,
        replay_start_offset_s=0.0,
        score_start_offset_s=0.1,
        score_end_offset_s=0.2,
        grids=_grids(),
        controller_inputs=tuple(
            _core_input(stamp) for stamp in (0.0, 0.1, 0.2)
        ),
        metadata={"source": "synthetic_exact_unit_fixture"},
    )


def _controller_state():
    return ControllerCoreState(
        target_gimbal_angles=(0.0, 0.0, 0.0, 0.0)
    )


class _PersistentAssimilationOracle:
    is_exact = True
    transport_is_persistent = True

    def __init__(self, identity):
        self.identity = identity

    def replay(self, payload):
        command_timestamps = []
        pid_terms = []
        job_ticks = []
        final_states = []
        for job_index, raw_job in enumerate(payload["jobs"]):
            job = copy.deepcopy(raw_job)
            pose = job["initial_pose_state"]
            allocation = job["initial_allocation_state"]
            for tick_index, tick in enumerate(job["ticks"]):
                pose["previous_stamp"] = tick["stamp"]
                pose["previous_flight_state"] = tick[
                    "flight_state"
                ]
                pose["previous_xy_control_mode"] = tick[
                    "xy_control_mode"
                ]
                pose["previous_force_landing"] = tick[
                    "force_landing"
                ]
                pose["has_previous_force_landing"] = True
                pose["start_roll_pitch_integration"] = True
                command_timestamps.append([tick["stamp"]])
                pid_terms.append(
                    [0.0, 0.0, 9.80665, 0.0, 0.0, 0.0]
                    + [0.0] * 18
                )
                job_ticks.append([job_index, tick_index])
            final_states.append(
                {"pose": pose, "allocation": allocation}
            )
        row_count = len(command_timestamps)
        return ExactOracleReplayOutput(
            identity=self.identity,
            continuous={
                "command_timestamp": np.asarray(
                    command_timestamps
                ),
                "pid_terms": np.asarray(pid_terms),
                "four_axis_command": np.zeros((row_count, 7)),
                "vectoring_force": np.zeros((row_count, 8)),
                "gimbal_command": np.zeros((row_count, 4)),
                "allocation_internal": np.zeros((row_count, 3)),
                "torque_allocation_matrix_inverse": np.zeros(
                    (row_count, 24)
                ),
                "job_tick": np.asarray(job_ticks, dtype=float),
            },
            events=np.zeros(row_count, dtype=int),
            final_states=tuple(final_states),
        )


class _BatchOnlyExactOracle:
    """Exact identity without the reset/step continuity contract."""

    def __init__(self, identity):
        self.identity = identity

    def run_batch(self, items):
        return tuple(items)


class _RecordedCommandsForbidden:
    @property
    def content_sha256(self):
        raise AssertionError(
            "closed-loop inference must not inspect recorded commands"
        )


def _prepared(controller_state=None):
    grids = _grids()

    def episode(
        episode_id,
        role,
        source_bag_sha256,
        normalized_episode_sha256,
    ):
        observations = ObservationDataset(
            episode_id=episode_id,
            role=role,
            timestamps=np.asarray([0.1, 0.2]),
            position_world=np.asarray(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
            ),
            orientation_xyzw=np.asarray(
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
            ),
            velocity_world=np.zeros((2, 3)),
            source_bag_sha256=source_bag_sha256,
            normalized_episode_sha256=normalized_episode_sha256,
        )
        nuisance = EpisodeNuisance(
            initial_plant_state=np.asarray(
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            initial_actuator_state=np.zeros(0),
            disturbance_parameters=np.zeros(6),
            sensor_bias=np.zeros(6),
            controller_state=controller_state,
            state_sample_id="{}-state".format(episode_id),
        )
        return PreparedEpisode(
            config={
                "episode_id": episode_id,
                "role": role,
            },
            data=SimpleNamespace(
                episode_id=episode_id,
                source_bag_sha256=source_bag_sha256,
                normalized_episode_sha256=normalized_episode_sha256,
            ),
            grids=grids,
            observations=observations,
            nuisance_samples=(nuisance,),
            commands=_RecordedCommandsForbidden(),
            trajectory_posterior=object(),
        )
    return {
        EPISODE_ID: episode(
            EPISODE_ID,
            "inference_failure",
            SOURCE_BAG_SHA256,
            NORMALIZED_EPISODE_SHA256,
        ),
        SUCCESS_EPISODE_ID: episode(
            SUCCESS_EPISODE_ID,
            "validation_success",
            SUCCESS_SOURCE_BAG_SHA256,
            SUCCESS_NORMALIZED_EPISODE_SHA256,
        ),
    }


def _closed_loop_config():
    config = copy.deepcopy(
        yaml.safe_load(CURRENT_CONFIG.read_text(encoding="utf-8"))
    )
    config["mode"] = "closed_loop_plant_identification"
    config["controller"].update(
        {
            "backend": "gimbalrotor_controller_cpp/v2",
            "fidelity": "pc_exact",
            "snapshot_policy": "injected_hash_bound",
            "nominal_model_policy": "frozen",
            "require_factual_replay_pass": True,
        }
    )
    config["observation"][
        "require_controller_event_evidence"
    ] = True
    config["inference"].update(
        {
            "particle_count": 32,
            "mcmc_steps": 0,
            "chain_count": 1,
            "rollout_cache_entries": 1024,
        }
    )
    return config


def _dependencies(creations):
    oracle_identity = _oracle_identity()
    identity = controller_backend_identity(oracle_identity)
    snapshot = _snapshot(identity)
    fixtures = {
        EPISODE_ID: _fixture(),
        SUCCESS_EPISODE_ID: _fixture(
            SUCCESS_EPISODE_ID,
            SUCCESS_SOURCE_BAG_SHA256,
        ),
    }
    snapshots = {
        EPISODE_ID: snapshot,
        SUCCESS_EPISODE_ID: snapshot,
    }
    conformance_bundle = _conformance_bundle(
        oracle_identity, fixtures, snapshots
    )
    gate = evaluate_exact_closed_loop_gate(
        identity, conformance_bundle.representative_report
    )
    oracle = _PersistentAssimilationOracle(oracle_identity)

    def factory():
        backend = StatefulExactOracleControllerBackend(oracle)
        creations.append(backend)
        return backend

    return ExactClosedLoopDependencies(
        controller_backend_factory=factory,
        fixtures=fixtures,
        snapshots=snapshots,
        gate_report=gate,
        conformance_bundle=conformance_bundle,
    )


class ExactClosedLoopAssimilationTests(unittest.TestCase):
    def test_current_style_missing_exact_evidence_rejects_before_smc(self):
        config = copy.deepcopy(
            yaml.safe_load(CURRENT_CONFIG.read_text(encoding="utf-8"))
        )
        config["mode"] = "closed_loop_plant_identification"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ClosedLoopGateError, "injected exact fixtures"
            ):
                write_assimilation_run(
                    config=config,
                    config_sha256="d" * 64,
                    prepared={},
                    output_root=directory,
                    run_id="must-not-exist",
                    source_commit="test-source",
                )
            self.assertEqual(tuple(Path(directory).iterdir()), ())

    def test_nonpassing_factual_gate_cannot_form_dependencies(self):
        identity = _identity()
        snapshot = _snapshot(identity)
        fixtures = {
            EPISODE_ID: _fixture(),
            SUCCESS_EPISODE_ID: _fixture(
                SUCCESS_EPISODE_ID,
                SUCCESS_SOURCE_BAG_SHA256,
            ),
        }
        snapshots = {
            EPISODE_ID: snapshot,
            SUCCESS_EPISODE_ID: snapshot,
        }
        conformance_bundle = _conformance_bundle(
            _oracle_identity(), fixtures, snapshots
        )
        with self.assertRaisesRegex(
            ClosedLoopGateError, "passing factual gate"
        ):
            ExactClosedLoopDependencies(
                controller_backend_factory=lambda: (
                    _BatchOnlyExactOracle(identity)
                ),
                fixtures=fixtures,
                snapshots=snapshots,
                gate_report=evaluate_exact_closed_loop_gate(
                    identity,
                    _factual_evidence(
                        _oracle_identity(), timestamp_delta=1.0
                    ),
                ),
                conformance_bundle=conformance_bundle,
            )

    def test_other_source_bag_conformance_cannot_authorize_runtime_fixture(
        self,
    ):
        oracle_identity = _oracle_identity()
        identity = controller_backend_identity(oracle_identity)
        snapshot = _snapshot(identity)
        runtime_fixtures = {
            EPISODE_ID: _fixture(),
            SUCCESS_EPISODE_ID: _fixture(
                SUCCESS_EPISODE_ID,
                SUCCESS_SOURCE_BAG_SHA256,
            ),
        }
        foreign_fixtures = {
            EPISODE_ID: _fixture(
                EPISODE_ID,
                "9" * 64,
            ),
            SUCCESS_EPISODE_ID: runtime_fixtures[SUCCESS_EPISODE_ID],
        }
        snapshots = {
            EPISODE_ID: snapshot,
            SUCCESS_EPISODE_ID: snapshot,
        }
        foreign_bundle = _conformance_bundle(
            oracle_identity,
            foreign_fixtures,
            snapshots,
        )
        gate = evaluate_exact_closed_loop_gate(
            identity,
            foreign_bundle.representative_report,
        )
        with self.assertRaisesRegex(
            ClosedLoopGateError,
            "not bound",
        ):
            ExactClosedLoopDependencies(
                controller_backend_factory=lambda: (
                    _BatchOnlyExactOracle(identity)
                ),
                fixtures=runtime_fixtures,
                snapshots=snapshots,
                gate_report=gate,
                conformance_bundle=foreign_bundle,
            )

    def test_unsupported_snapshot_actuator_dimensions_reject_before_smc(self):
        oracle_identity = _oracle_identity()
        identity = controller_backend_identity(oracle_identity)
        base_snapshot = _snapshot(identity)
        fixtures = {
            EPISODE_ID: _fixture(),
            SUCCESS_EPISODE_ID: _fixture(
                SUCCESS_EPISODE_ID,
                SUCCESS_SOURCE_BAG_SHA256,
            ),
        }
        for changed_option in (
            {"gimbal_calc_in_fc": True},
            {"gimbal_dof": 2},
        ):
            with self.subTest(**changed_option):
                options = dict(base_snapshot.static_options)
                options.update(changed_option)
                unsupported = base_snapshot.with_updates(
                    static_options=options
                )
                snapshots = {
                    EPISODE_ID: unsupported,
                    SUCCESS_EPISODE_ID: unsupported,
                }
                conformance_bundle = _conformance_bundle(
                    oracle_identity,
                    fixtures,
                    snapshots,
                    initial_state=ControllerCoreState(
                        target_gimbal_angles=(
                            (0.0,) * 8
                            if changed_option.get("gimbal_dof") == 2
                            else (0.0,) * 4
                        )
                    ),
                )
                gate = evaluate_exact_closed_loop_gate(
                    identity,
                    conformance_bundle.representative_report,
                )
                with self.assertRaisesRegex(
                    ClosedLoopGateError, "actuator dimensions"
                ):
                    ExactClosedLoopDependencies(
                        controller_backend_factory=lambda: (
                            _BatchOnlyExactOracle(identity)
                        ),
                        fixtures=fixtures,
                        snapshots=snapshots,
                        gate_report=gate,
                        conformance_bundle=conformance_bundle,
                    )

    def test_missing_controller_state_rejects_before_backend_rollout(self):
        creations = []
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ClosedLoopGateError, "controller state"
            ):
                write_assimilation_run(
                    config=_closed_loop_config(),
                    config_sha256="d" * 64,
                    prepared=_prepared(controller_state=None),
                    output_root=directory,
                    run_id="must-not-exist",
                    source_commit="test-source",
                    closed_loop_dependencies=_dependencies(creations),
                )
            # One backend is constructed only for preflight identity checking
            # after every evidence/state check has passed.
            self.assertEqual(creations, [])
            self.assertEqual(tuple(Path(directory).iterdir()), ())

    def test_batch_only_oracle_without_state_continuity_is_rejected(self):
        identity = _identity()
        snapshot = _snapshot(identity)
        fixtures = {
            EPISODE_ID: _fixture(),
            SUCCESS_EPISODE_ID: _fixture(
                SUCCESS_EPISODE_ID,
                SUCCESS_SOURCE_BAG_SHA256,
            ),
        }
        snapshots = {
            EPISODE_ID: snapshot,
            SUCCESS_EPISODE_ID: snapshot,
        }
        conformance_bundle = _conformance_bundle(
            _oracle_identity(), fixtures, snapshots
        )
        dependencies = ExactClosedLoopDependencies(
            controller_backend_factory=lambda: _BatchOnlyExactOracle(
                identity
            ),
            fixtures=fixtures,
            snapshots=snapshots,
            gate_report=evaluate_exact_closed_loop_gate(
                identity, conformance_bundle.representative_report
            ),
            conformance_bundle=conformance_bundle,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ClosedLoopGateError, "stateful backend"
            ):
                write_assimilation_run(
                    config=_closed_loop_config(),
                    config_sha256="d" * 64,
                    prepared=_prepared(
                        controller_state=_controller_state()
                    ),
                    output_root=directory,
                    run_id="must-not-exist",
                    source_commit="test-source",
                    closed_loop_dependencies=dependencies,
                )
            self.assertEqual(tuple(Path(directory).iterdir()), ())

    def test_persistent_factory_must_share_one_oracle_process(self):
        oracle_identity = _oracle_identity()
        identity = controller_backend_identity(oracle_identity)
        snapshot = _snapshot(identity)
        fixtures = {
            EPISODE_ID: _fixture(),
            SUCCESS_EPISODE_ID: _fixture(
                SUCCESS_EPISODE_ID,
                SUCCESS_SOURCE_BAG_SHA256,
            ),
        }
        snapshots = {
            EPISODE_ID: snapshot,
            SUCCESS_EPISODE_ID: snapshot,
        }
        conformance_bundle = _conformance_bundle(
            oracle_identity, fixtures, snapshots
        )
        dependencies = ExactClosedLoopDependencies(
            controller_backend_factory=lambda: (
                StatefulExactOracleControllerBackend(
                    _PersistentAssimilationOracle(oracle_identity)
                )
            ),
            fixtures=fixtures,
            snapshots=snapshots,
            gate_report=evaluate_exact_closed_loop_gate(
                identity, conformance_bundle.representative_report
            ),
            conformance_bundle=conformance_bundle,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ClosedLoopGateError, "share one oracle process"
            ):
                write_assimilation_run(
                    config=_closed_loop_config(),
                    config_sha256="d" * 64,
                    prepared=_prepared(
                        controller_state=_controller_state()
                    ),
                    output_root=directory,
                    run_id="must-not-exist",
                    source_commit="test-source",
                    closed_loop_dependencies=dependencies,
                )
            self.assertEqual(tuple(Path(directory).iterdir()), ())

    def test_passing_injected_exact_dependencies_run_smc_and_writer(self):
        creations = []
        dependencies = _dependencies(creations)
        with tempfile.TemporaryDirectory() as directory:
            destination = write_assimilation_run(
                config=_closed_loop_config(),
                config_sha256="d" * 64,
                prepared=_prepared(controller_state=_controller_state()),
                output_root=directory,
                run_id="synthetic-closed-loop",
                source_commit="test-source",
                closed_loop_dependencies=dependencies,
            )
            self.assertTrue(creations)
            summary = json.loads(
                (destination / "posterior_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            factual = json.loads(
                (destination / "factual_replay_report.json").read_text(
                    encoding="utf-8"
                )
            )
            audit = json.loads(
                (destination / "controller_replay_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = verify_run_manifest(destination)
            with (
                destination / "likelihood_components.csv"
            ).open(newline="", encoding="utf-8") as stream:
                likelihood_rows = tuple(csv.DictReader(stream))
            self.assertEqual(
                summary["model_id"], CLOSED_LOOP_POSTERIOR_MODEL_ID
            )
            self.assertTrue(likelihood_rows)
            self.assertTrue(
                all(
                    int(row["scored_event_sample_count"]) == 2
                    and int(row["censored_event_sample_count"]) == 0
                    and row["controller_event_evidence_status"]
                    == "scored"
                    and float(row["saturation_mode_event"]) < 0.0
                    for row in likelihood_rows
                )
            )
            self.assertTrue(factual["passed"])
            self.assertTrue(factual["closed_loop_exact_allowed"])
            self.assertEqual(
                factual["factual_evidence_sha256"],
                dependencies.gate_report.conformance_report.evidence_sha256,
            )
            self.assertEqual(
                factual["conformance_report"]["evidence_sha256"],
                factual["factual_evidence_sha256"],
            )
            self.assertTrue(audit["overall_exact_replay_ready"])
            self.assertEqual(
                manifest["provenance"]["plant_backend_id"],
                CLOSED_LOOP_PLANT_BACKEND_ID,
            )
            self.assertEqual(
                manifest["provenance"]["fixture_sha256"],
                stable_hash(
                    {
                        "exact_controller_dependencies_sha256": (
                            dependencies.content_sha256
                        ),
                        "episode_nuisance_evidence": {
                            key: [
                                {
                                    "state_sample_id": (
                                        nuisance.state_sample_id
                                    ),
                                    "weight": nuisance.weight,
                                    "initial_plant_state": (
                                        nuisance.initial_plant_state
                                    ),
                                    "initial_actuator_state": (
                                        nuisance.initial_actuator_state
                                    ),
                                    "disturbance_model_id": (
                                        nuisance.disturbance_model_id
                                    ),
                                    "disturbance_parameters": (
                                        nuisance.disturbance_parameters
                                    ),
                                    "sensor_bias": nuisance.sensor_bias,
                                }
                                for nuisance
                                in item.nuisance_samples
                            ]
                            for key, item in _prepared(
                                controller_state=_controller_state()
                            ).items()
                        },
                        "episode_controller_state_evidence": {
                            key: [
                                {
                                    "state_sample_id": (
                                        nuisance.state_sample_id
                                    ),
                                    "controller_state_sha256": (
                                        nuisance.controller_state.content_sha256
                                    ),
                                }
                                for nuisance
                                in item.nuisance_samples
                            ]
                            for key, item in _prepared(
                                controller_state=_controller_state()
                            ).items()
                        },
                    }
                ),
            )

    def test_caller_event_observations_cannot_replace_exact_evidence(self):
        creations = []
        dependencies = _dependencies(creations)
        prepared = _prepared(
            controller_state=_controller_state()
        )
        episode = prepared[EPISODE_ID]
        observations = replace(
            episode.observations,
            event_observations=ControllerEventObservations(
                timestamps=np.asarray([0.1, 0.2]),
                event_bitmasks=np.asarray([32, 32], dtype=np.uint32),
            ),
        )
        prepared[EPISODE_ID] = replace(
            episode, observations=observations
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ClosedLoopGateError,
                "differ from the exact conformance evidence",
            ):
                write_assimilation_run(
                    config=_closed_loop_config(),
                    config_sha256="d" * 64,
                    prepared=prepared,
                    output_root=directory,
                    run_id="must-not-exist",
                    source_commit="test-source",
                    closed_loop_dependencies=dependencies,
                )
            self.assertEqual(tuple(Path(directory).iterdir()), ())

    def test_open_loop_cannot_promote_controller_event_frames(self):
        config = copy.deepcopy(
            yaml.safe_load(CURRENT_CONFIG.read_text(encoding="utf-8"))
        )
        prepared = _prepared(
            controller_state=_controller_state()
        )
        episode = prepared[EPISODE_ID]
        prepared[EPISODE_ID] = replace(
            episode,
            observations=replace(
                episode.observations,
                event_observations=ControllerEventObservations(
                    timestamps=np.asarray([0.1, 0.2]),
                    event_bitmasks=np.asarray(
                        [0, 0], dtype=np.uint32
                    ),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "open-loop inference cannot accept",
            ):
                write_assimilation_run(
                    config=config,
                    config_sha256="d" * 64,
                    prepared=prepared,
                    output_root=directory,
                    run_id="must-not-exist",
                    source_commit="test-source",
                )
            self.assertEqual(tuple(Path(directory).iterdir()), ())

    def test_validation_cannot_promote_unscored_controller_events(self):
        prepared = _prepared(
            controller_state=_controller_state()
        )
        episode = prepared[EPISODE_ID]
        prepared[EPISODE_ID] = replace(
            episode,
            observations=replace(
                episode.observations,
                event_observations=ControllerEventObservations(
                    timestamps=np.asarray([0.1, 0.2]),
                    event_bitmasks=np.asarray(
                        [0, 0], dtype=np.uint32
                    ),
                ),
            ),
        )
        likelihood = SimpleNamespace(
            config=SimpleNamespace(
                require_controller_event_evidence=False
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "event-unscored validation cannot accept",
        ):
            validate_posterior(
                posterior=None,
                prepared=prepared,
                rollout=None,
                component_likelihood=likelihood,
            )


if __name__ == "__main__":
    unittest.main()
