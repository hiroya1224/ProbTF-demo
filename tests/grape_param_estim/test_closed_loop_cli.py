import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

from grape_param_estim.alternative_backends import (
    EXACT_ORACLE_PROTOCOL,
    PC_EXACT_ORACLE_CAPABILITIES,
    ExactOracleConformanceFixture,
    ExactOracleConformanceReport,
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
)
from grape_param_estim.controller.exact_inputs import (
    ControllerStateSelection,
    ExactEpisodeConformanceBundle,
    ExactEpisodeConformanceEvidence,
    FIXTURE_BUNDLE_SCHEMA,
    SNAPSHOT_BUNDLE_SCHEMA,
    STATE_BUNDLE_SCHEMA,
    inject_controller_states,
    load_conformance_report,
    load_controller_state_bundle,
    load_fixture_bundle,
    load_snapshot_bundle,
)
from grape_param_estim.controller.external_oracle import (
    build_exact_replay_payload,
)
from grape_param_estim.data import (
    ControllerReplayFixture,
    EpisodeTimeGrids,
    EventGrid,
)
from grape_param_estim.episode import stable_hash
from grape_param_estim.forward import ClosedLoopGateError
from grape_param_estim.plant import EpisodeNuisance
from grape_param_estim.plant_assimilation import PreparedEpisode
from grape_param_estim.controller_replay import ReplayMetrics


REPOSITORY = Path(__file__).resolve().parents[2]
PACKAGE = REPOSITORY / "ros/examples/grape-param-estim"
CONFIG = PACKAGE / "config/plant_assimilation.yaml"
SCRIPT = PACKAGE / "scripts/estimate_grape_plant.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "estimate_grape_plant_cli_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _core_input(stamp):
    roll_pitch_enabled = float(stamp) > 0.0
    return ControllerCoreInput(
        stamp=stamp,
        dt=0.1,
        position=(0.0, 0.0, 1.0),
        velocity=(0.0, 0.0, 0.0),
        orientation=np.eye(3),
        angular_velocity=(0.0, 0.0, 0.0),
        target_position=(0.0, 0.0, 1.0),
        target_velocity=(0.0, 0.0, 0.0),
        target_acceleration=(0.0, 0.0, 0.0),
        target_orientation=np.eye(3),
        target_angular_velocity=(0.0, 0.0, 0.0),
        target_angular_acceleration=(0.0, 0.0, 0.0),
        control_mode=(0, 0, 0, 0, 0, 0),
        integration_enabled=(
            True,
            True,
            True,
            roll_pitch_enabled,
            roll_pitch_enabled,
            True,
        ),
        flight_state=5,
        force_landing=False,
        current_rpy=(0.0, 0.0, 0.0),
        target_rpy=(0.0, 0.0, 0.0),
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


def _fixture():
    return ControllerReplayFixture(
        episode_id="episode",
        source_bag_sha256="1" * 64,
        topic_inventory_sha256="2" * 64,
        replay_start_offset_s=0.0,
        score_start_offset_s=0.1,
        score_end_offset_s=0.2,
        grids=_grids(),
        controller_inputs=tuple(
            _core_input(stamp) for stamp in (0.0, 0.1, 0.2)
        ),
        metadata={
            "origin": "unit-test",
            "normalized_episode_sha256": "9" * 64,
        },
    )


def _snapshot(identity):
    rotation = np.eye(3).tolist()
    return ControllerSnapshot(
        backend_id=identity.backend_id,
        source_commit=identity.source_commit,
        artifact_sha256=identity.artifact_sha256,
        nominal_model_sha256="d" * 64,
        parameter_dump_sha256="e" * 64,
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
        nominal_cog=(0.0, 0.0, 0.0),
        nominal_inertia=np.diag((0.3, 0.4, 0.5)),
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
            "thrust_coordinate_rotations": [rotation] * 4,
        },
    )


def _write_hashed_bundle(path, schema, field, values):
    payload = {"schema": schema, field: values}
    payload["content_sha256"] = stable_hash(payload)
    path.write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )


def _conformance_report(identity):
    request = {"fixture": "request"}
    channel_widths = {
        "command_timestamp": 1,
        "pid_terms": 24,
        "four_axis_command": 7,
        "vectoring_force": 8,
        "gimbal_command": 4,
        "allocation_internal": 3,
        "torque_allocation_matrix_inverse": 24,
    }
    continuous = {
        name: np.zeros((1, width))
        for name, width in channel_widths.items()
    }
    provenance = ExactOracleFixtureProvenance.create(
        source_bag_sha256="b" * 64,
        source_topics=("/controller/replay",),
        interval_start_time_ns=1,
        interval_end_time_ns=2,
        frame_conventions={"controller": "base_link"},
        unit_conventions={"time": "seconds"},
        motor_order=("m1", "m2", "m3", "m4"),
        request_payload=request,
        continuous=continuous,
        events=np.zeros(1),
        extraction_config_sha256="c" * 64,
        source_commit="fixture-source",
    )
    metrics = {}
    for name, width in channel_widths.items():
        timestamp = name == "command_timestamp"
        metrics[name] = ReplayMetrics(
            normalized_rmse=np.zeros(width),
            normalized_maximum_error=np.zeros(width),
            event_agreement=1.0,
            passed=True,
            rmse_threshold=0.0 if timestamp else 0.01,
            maximum_error_threshold=0.0 if timestamp else 0.03,
            event_agreement_threshold=1.0,
        )
    return ExactOracleConformanceReport(
        passed=True,
        status="PASS",
        reasons=(),
        channel_metrics=metrics,
        identity=identity,
        fixture_provenance=provenance,
        fixture_content_sha256=provenance.content_sha256,
        request_payload_sha256=stable_hash(request),
        fidelity="pc_exact",
    )


def _episode_conformance_bundle(identity, fixture, snapshot, state):
    input_payload = {
        "schema": "grape_exact_episode_controller_inputs/v1",
        "episode_id": fixture.episode_id,
        "controller_inputs": [
            item.to_mapping() for item in fixture.controller_inputs
        ],
    }
    request = build_exact_replay_payload(
        snapshot,
        state,
        fixture.controller_inputs,
        evidence_binding={
            "schema": "grape_exact_episode_request_binding/v1",
            "episode_id": fixture.episode_id,
            "source_bag_sha256": fixture.source_bag_sha256,
            "controller_replay_fixture_sha256": fixture.fixture_sha256,
            "controller_input_sha256": stable_hash(input_payload),
            "controller_snapshot_sha256": snapshot.snapshot_id,
        },
    )
    widths = {
        "command_timestamp": 1,
        "pid_terms": 24,
        "four_axis_command": 7,
        "vectoring_force": 8,
        "gimbal_command": 4,
        "allocation_internal": 3,
        "torque_allocation_matrix_inverse": 24,
    }
    count = len(fixture.controller_inputs)
    continuous = {
        channel: (
            np.asarray(
                [[item.stamp] for item in fixture.controller_inputs]
            )
            if channel == "command_timestamp"
            else np.zeros((count, widths[channel]))
        )
        for channel in required_conformance_channels(identity.fidelity)
    }
    events = np.zeros(count, dtype=int)
    provenance = ExactOracleFixtureProvenance.create(
        source_bag_sha256=fixture.source_bag_sha256,
        source_topics=("/controller/replay",),
        interval_start_time_ns=1,
        interval_end_time_ns=3,
        frame_conventions={"controller": "base_link"},
        unit_conventions={"time": "seconds"},
        motor_order=("m1", "m2", "m3", "m4"),
        request_payload=request,
        continuous=continuous,
        events=events,
        extraction_config_sha256="c" * 64,
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
            return ExactOracleReplayOutput(
                identity=identity,
                continuous=continuous,
                events=events,
            )

    report = evaluate_exact_oracle_conformance(
        Oracle(), request, conformance_fixture
    )
    evidence = ExactEpisodeConformanceEvidence.create(
        fixture=fixture,
        snapshot=snapshot,
        initial_controller_state=state,
        conformance_fixture=conformance_fixture,
        conformance_report=report,
    )
    return ExactEpisodeConformanceBundle(
        episodes={fixture.episode_id: evidence}
    )


class _FakePersistentOracle:
    is_exact = True
    transport_is_persistent = True

    def __init__(self, command, expected_identity, timeout_s):
        self.command = tuple(command)
        self.identity = expected_identity
        self.timeout_s = timeout_s
        self.closed = False

    def replay(self, payload):
        ticks = payload["jobs"][0]["ticks"]
        count = len(ticks)
        widths = {
            "pid_terms": 24,
            "four_axis_command": 7,
            "vectoring_force": 8,
            "gimbal_command": 4,
            "allocation_internal": 3,
            "torque_allocation_matrix_inverse": 24,
        }
        continuous = {
            channel: (
                np.asarray([[tick["stamp"]] for tick in ticks])
                if channel == "command_timestamp"
                else np.zeros((count, widths[channel]))
            )
            for channel in required_conformance_channels(
                self.identity.fidelity
            )
        }
        return ExactOracleReplayOutput(
            identity=self.identity,
            continuous=continuous,
            events=np.zeros(count, dtype=int),
        )

    def close(self):
        self.closed = True


def _arguments(**updates):
    values = {
        "exact_replay_executable": None,
        "controller_fixture_bundle": None,
        "controller_snapshot_bundle": None,
        "controller_state_bundle": None,
        "factual_conformance_report": None,
        "exact_oracle_timeout_s": None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _nuisance(sample_id):
    return EpisodeNuisance(
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
        initial_actuator_state=np.empty(0),
        controller_state=None,
        state_sample_id=sample_id,
    )


def _prepared(samples):
    return {
        "episode": PreparedEpisode(
            config={
                "episode_id": "episode",
                "replay_start_offset_s": 0.0,
                "score_start_offset_s": 0.1,
                "score_end_offset_s": 0.2,
            },
            data=object(),
            grids=_grids(),
            observations=SimpleNamespace(
                source_bag_sha256="1" * 64,
                normalized_episode_sha256="9" * 64,
            ),
            nuisance_samples=tuple(samples),
            commands=object(),
            trajectory_posterior=object(),
        )
    }


class ExactClosedLoopCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = _load_script()
        cls.open_config = yaml.safe_load(
            CONFIG.read_text(encoding="utf-8")
        )

    def test_parser_exposes_all_explicit_exact_inputs(self):
        parsed = self.script._arguments(
            [
                "--bag-root",
                "/tmp/bags",
                "--output-root",
                "/tmp/output",
                "--exact-replay-executable",
                "/tmp/oracle",
                "--controller-fixture-bundle",
                "/tmp/fixtures.json",
                "--controller-snapshot-bundle",
                "/tmp/snapshots.json",
                "--controller-state-bundle",
                "/tmp/states.json",
                "--factual-conformance-report",
                "/tmp/report.json",
            ]
        )
        self.assertEqual(parsed.exact_replay_executable, Path("/tmp/oracle"))
        self.assertEqual(
            parsed.factual_conformance_report, Path("/tmp/report.json")
        )

    def test_open_loop_rejects_exact_inputs_instead_of_ignoring_them(self):
        with self.assertRaisesRegex(ValueError, "open-loop"):
            self.script._validate_exact_mode_arguments(
                self.open_config,
                _arguments(controller_fixture_bundle=Path("fixture.json")),
            )

    def test_closed_loop_rejects_missing_inputs_and_current_policy(self):
        closed = copy.deepcopy(self.open_config)
        closed["mode"] = "closed_loop_plant_identification"
        with self.assertRaisesRegex(
            ClosedLoopGateError, "--exact-replay-executable"
        ):
            self.script._validate_exact_mode_arguments(
                closed, _arguments()
            )
        complete = _arguments(
            **{
                destination: Path("/tmp/{}".format(destination))
                for destination, _ in self.script._EXACT_PATH_OPTIONS
            }
        )
        with self.assertRaisesRegex(
            ClosedLoopGateError, "not eligible"
        ):
            self.script._validate_exact_mode_arguments(closed, complete)

    def test_fixture_bundle_is_typed_and_both_hash_layers_are_checked(self):
        fixture = _fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixtures.json"
            _write_hashed_bundle(
                path,
                FIXTURE_BUNDLE_SCHEMA,
                "fixtures",
                [fixture.to_dict()],
            )
            loaded, _ = load_fixture_bundle(path)
            self.assertIsInstance(
                loaded["episode"], ControllerReplayFixture
            )
            self.assertEqual(
                loaded["episode"].fixture_sha256,
                fixture.fixture_sha256,
            )

            outer = json.loads(path.read_text(encoding="utf-8"))
            outer["fixtures"][0]["metadata"]["origin"] = "tampered"
            path.write_text(json.dumps(outer), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                load_fixture_bundle(path)

            outer["content_sha256"] = stable_hash(
                {
                    "schema": outer["schema"],
                    "fixtures": outer["fixtures"],
                }
            )
            path.write_text(json.dumps(outer), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixture hash mismatch"):
                load_fixture_bundle(path)

    def test_state_bundle_injects_every_sample_without_zero_default(self):
        first = ControllerCoreState(
            previous_stamp=1.0,
            target_gimbal_angles=(0.0, 0.0, 0.0, 0.0),
        )
        second = replace(first, previous_stamp=2.0)
        records = [
            {
                "episode_id": "episode",
                "sample_states": [
                    {
                        "state_sample_id": "sample-a",
                        "controller_state": first.to_mapping(),
                    },
                    {
                        "state_sample_id": "sample-b",
                        "controller_state": second.to_mapping(),
                    },
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "states.json"
            _write_hashed_bundle(
                path, STATE_BUNDLE_SCHEMA, "episodes", records
            )
            selections, _ = load_controller_state_bundle(path)
            original = _prepared(
                (_nuisance("sample-a"), _nuisance("sample-b"))
            )
            injected = inject_controller_states(
                original, selections, "9" * 64
            )
            self.assertIsNone(
                original["episode"].nuisance_samples[0].controller_state
            )
            states = tuple(
                item.controller_state
                for item in injected[
                    "episode"
                ].nuisance_samples
            )
            self.assertEqual(states, (first, second))
            self.assertEqual(
                tuple(
                    item.state_sample_id
                    for item in injected[
                        "episode"
                    ].nuisance_samples
                ),
                (
                    stable_hash(
                        {
                            "schema": (
                                "grape.controller-state-bound-nuisance/v1"
                            ),
                            "original_state_sample_id": "sample-a",
                            "controller_state_sha256": (
                                first.content_sha256
                            ),
                            "controller_state_bundle_sha256": "9" * 64,
                        }
                    ),
                    stable_hash(
                        {
                            "schema": (
                                "grape.controller-state-bound-nuisance/v1"
                            ),
                            "original_state_sample_id": "sample-b",
                            "controller_state_sha256": (
                                second.content_sha256
                            ),
                            "controller_state_bundle_sha256": "9" * 64,
                        }
                    ),
                ),
            )

            incomplete = {
                "episode": ControllerStateSelection(
                    sample_states={"sample-a": first}
                )
            }
            with self.assertRaisesRegex(
                ValueError, "do not exactly match"
            ):
                inject_controller_states(
                    original, incomplete, "9" * 64
                )

    def test_changed_controller_state_changes_cache_and_provenance_ids(self):
        original = _prepared((_nuisance("sample-a"),))
        first = ControllerCoreState(
            previous_stamp=1.0,
            target_gimbal_angles=(0.0, 0.0, 0.0, 0.0),
        )
        changed = replace(first, previous_stamp=1.1)
        first_prepared = inject_controller_states(
            original,
            {
                "episode": ControllerStateSelection(
                    shared_state=first
                )
            },
            "8" * 64,
        )
        changed_prepared = inject_controller_states(
            original,
            {
                "episode": ControllerStateSelection(
                    shared_state=changed
                )
            },
            "8" * 64,
        )
        first_nuisance = first_prepared[
            "episode"
        ].nuisance_samples[0]
        changed_nuisance = changed_prepared[
            "episode"
        ].nuisance_samples[0]
        # This ID is the RolloutCacheKey.initial_state_sample_id.
        self.assertNotEqual(
            first_nuisance.state_sample_id,
            changed_nuisance.state_sample_id,
        )

        def provenance_hash(nuisance):
            return stable_hash(
                {
                    "exact_controller_dependencies_sha256": "7" * 64,
                    "episode_controller_state_evidence": {
                        "episode": [
                            {
                                "state_sample_id": (
                                    nuisance.state_sample_id
                                ),
                                "controller_state_sha256": (
                                    nuisance.controller_state.content_sha256
                                ),
                            }
                        ]
                    },
                }
            )

        self.assertNotEqual(
            provenance_hash(first_nuisance),
            provenance_hash(changed_nuisance),
        )

    def test_snapshot_bundle_returns_typed_hash_bound_snapshots(self):
        identity = ExactOracleIdentity(
            protocol=EXACT_ORACLE_PROTOCOL,
            backend_id="jsk_aerial_robot/gimbalrotor_controller_replay",
            implementation_language="c++",
            source_commit="test-source",
            artifact_sha256="a" * 64,
            capabilities=PC_EXACT_ORACLE_CAPABILITIES,
            fidelity="pc_exact",
        )
        snapshot = _snapshot(identity)
        records = [
            {
                "episode_id": "episode",
                "snapshot": snapshot.to_mapping(),
                "snapshot_sha256": snapshot.snapshot_id,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshots.json"
            _write_hashed_bundle(
                path,
                SNAPSHOT_BUNDLE_SCHEMA,
                "snapshots",
                records,
            )
            loaded, _ = load_snapshot_bundle(path)
            self.assertIsInstance(
                loaded["episode"], ControllerSnapshot
            )
            self.assertEqual(
                loaded["episode"].snapshot_id, snapshot.snapshot_id
            )

            changed = json.loads(path.read_text(encoding="utf-8"))
            changed["snapshots"][0]["snapshot_sha256"] = "f" * 64
            changed["content_sha256"] = stable_hash(
                {
                    "schema": changed["schema"],
                    "snapshots": changed["snapshots"],
                }
            )
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "snapshot hash mismatch"
            ):
                load_snapshot_bundle(path)

    def test_conformance_report_loader_rejects_rebound_evidence(self):
        identity = ExactOracleIdentity(
            protocol=EXACT_ORACLE_PROTOCOL,
            backend_id="jsk_aerial_robot/gimbalrotor_controller_replay",
            implementation_language="c++",
            source_commit="test-source",
            artifact_sha256="a" * 64,
            capabilities=PC_EXACT_ORACLE_CAPABILITIES,
            fidelity="pc_exact",
        )
        report = _conformance_report(identity)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(
                json.dumps(report.to_mapping(), sort_keys=True),
                encoding="utf-8",
            )
            loaded = load_conformance_report(path)
            self.assertEqual(
                loaded.evidence_sha256, report.evidence_sha256
            )

            rebound = json.loads(path.read_text(encoding="utf-8"))
            rebound["identity"]["source_commit"] = "different-source"
            path.write_text(json.dumps(rebound), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema/hash"):
                load_conformance_report(path)

    def test_runtime_loads_genuine_v2_evidence_and_shared_transport(self):
        identity = ExactOracleIdentity(
            protocol=EXACT_ORACLE_PROTOCOL,
            backend_id="jsk_aerial_robot/gimbalrotor_controller_replay",
            implementation_language="c++",
            source_commit="test-source",
            artifact_sha256="a" * 64,
            capabilities=PC_EXACT_ORACLE_CAPABILITIES,
            fidelity="pc_exact",
        )
        fixture = _fixture()
        snapshot = _snapshot(identity)
        state = ControllerCoreState(
            target_gimbal_angles=(0.0, 0.0, 0.0, 0.0)
        )
        conformance_bundle = _episode_conformance_bundle(
            identity, fixture, snapshot, state
        )
        report = conformance_bundle.representative_report
        config = copy.deepcopy(self.open_config)
        config["mode"] = "closed_loop_plant_identification"
        config["controller"].update(
            {
                "backend": identity.backend_id,
                "fidelity": "pc_exact",
                "snapshot_policy": "injected_hash_bound",
                "nominal_model_policy": "frozen",
                "require_factual_replay_pass": True,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_path = root / "fixtures.json"
            snapshot_path = root / "snapshots.json"
            state_path = root / "states.json"
            report_path = root / "conformance.json"
            executable_path = root / "replay"
            _write_hashed_bundle(
                fixture_path,
                FIXTURE_BUNDLE_SCHEMA,
                "fixtures",
                [fixture.to_dict()],
            )
            _write_hashed_bundle(
                snapshot_path,
                SNAPSHOT_BUNDLE_SCHEMA,
                "snapshots",
                [
                    {
                        "episode_id": "episode",
                        "snapshot": snapshot.to_mapping(),
                        "snapshot_sha256": snapshot.snapshot_id,
                    }
                ],
            )
            _write_hashed_bundle(
                state_path,
                STATE_BUNDLE_SCHEMA,
                "episodes",
                [
                    {
                        "episode_id": "episode",
                        "controller_state": state.to_mapping(),
                    }
                ],
            )
            report_path.write_text(
                json.dumps(
                    conformance_bundle.to_mapping(), sort_keys=True
                ),
                encoding="utf-8",
            )
            arguments = _arguments(
                exact_replay_executable=executable_path,
                controller_fixture_bundle=fixture_path,
                controller_snapshot_bundle=snapshot_path,
                controller_state_bundle=state_path,
                factual_conformance_report=report_path,
            )
            with mock.patch.object(
                self.script,
                "PersistentSubprocessExactControllerOracle",
                _FakePersistentOracle,
            ):
                injected, dependencies, oracle, evidence = (
                    self.script._load_closed_loop_runtime(
                        config,
                        arguments,
                        _prepared((_nuisance("sample-a"),)),
                    )
                )
            try:
                self.assertTrue(dependencies.gate_report.passed)
                self.assertEqual(
                    dependencies.gate_report.factual_evidence_sha256,
                    report.evidence_sha256,
                )
                self.assertEqual(
                    evidence["factual_conformance_evidence_sha256"],
                    report.evidence_sha256,
                )
                self.assertEqual(
                    evidence["factual_conformance_bundle_sha256"],
                    conformance_bundle.content_sha256,
                )
                self.assertEqual(
                    evidence["exact_input_bundle_sha256"],
                    stable_hash(
                        {
                            name: digest
                            for name, digest in evidence.items()
                            if name != "exact_input_bundle_sha256"
                        }
                    ),
                )
                first = dependencies.controller_backend_factory()
                second = dependencies.controller_backend_factory()
                self.assertIsNot(first, second)
                self.assertIs(first.transport, oracle)
                self.assertIs(second.transport, oracle)
                self.assertEqual(
                    injected["episode"].nuisance_samples[
                        0
                    ].controller_state,
                    state,
                )
            finally:
                oracle.close()
            self.assertTrue(oracle.closed)

    def test_factory_creates_fresh_adapters_on_one_shared_transport(self):
        identity = ExactOracleIdentity(
            protocol=EXACT_ORACLE_PROTOCOL,
            backend_id="jsk_aerial_robot/gimbalrotor_controller_replay",
            implementation_language="c++",
            source_commit="test-source",
            artifact_sha256="a" * 64,
            capabilities=PC_EXACT_ORACLE_CAPABILITIES,
            fidelity="pc_exact",
        )

        with mock.patch.object(
            self.script,
            "PersistentSubprocessExactControllerOracle",
            _FakePersistentOracle,
        ):
            oracle, factory = self.script._persistent_backend_factory(
                "/tmp/replay",
                identity,
                timeout_s=4.0,
            )
            first = factory()
            second = factory()
        self.assertIsNot(first, second)
        self.assertIs(first.transport, oracle)
        self.assertIs(second.transport, oracle)
        self.assertEqual(
            oracle.command[-2:],
            ("--artifact-sha256", "a" * 64),
        )
        oracle.close()
        self.assertTrue(oracle.closed)

    def test_writer_failure_still_closes_persistent_oracle(self):
        oracle = SimpleNamespace(closed=False)

        def close():
            oracle.closed = True

        oracle.close = close
        with mock.patch.object(
            self.script,
            "write_assimilation_run",
            side_effect=RuntimeError("writer failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "writer failed"):
                self.script._write_run_with_oracle_cleanup(
                    oracle=oracle,
                    config={},
                    config_sha256="a" * 64,
                    prepared={},
                    output_root=Path("/tmp"),
                    run_id="failure",
                    source_commit="test",
                    closed_loop_dependencies=object(),
                )
        self.assertTrue(oracle.closed)


if __name__ == "__main__":
    unittest.main()
