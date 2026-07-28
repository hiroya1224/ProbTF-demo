import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from grape_param_estim.alternative_backends import (
    EXACT_ORACLE_PROTOCOL,
    ExactOracleConformanceFixture,
    ExactOracleFixtureProvenance,
    ExactOracleIdentity,
    ExactOracleReplayOutput,
    evaluate_exact_oracle_conformance,
    required_conformance_channels,
)
from grape_param_estim.controller import (
    ControllerBackendIdentity,
    evaluate_exact_closed_loop_gate,
)
from grape_param_estim.controller.contracts import (
    PC_EXACT_REQUIRED_CAPABILITIES,
)
from grape_param_estim.data import (
    InitialStatePosterior,
    with_disturbance_samples,
)
from grape_param_estim.episode import stable_hash
from grape_param_estim.forward import (
    ClosedLoopForwardModel,
    CommandSample,
    OpenLoopForwardModel,
    RecordedCommandSeries,
)
from grape_param_estim.forward.cache import RolloutCache as ForwardRolloutCache
from grape_param_estim.forward.closed_loop import (
    _closed_loop_scheduler,
)
from grape_param_estim.inference import (
    IndependentBoundedPrior,
    LikelihoodComponents,
    PlantPosterior,
    PriorDimension,
    RolloutCache,
    episode_excitation_report,
    local_identifiability,
)
from grape_param_estim.output import (
    PlantAssimilationArtifactWriter,
    PlantRunProvenance,
    plain_data,
)
from grape_param_estim.output.artifacts import REQUIRED_ARTIFACTS
from grape_param_estim.output.manifest import verify_run_manifest
from grape_param_estim.plant import (
    ActuatorBackend,
    ActuatorCalibrationIdentity,
    ActuatorParameters,
    EffectiveRigidBodyPlantBackend,
    EpisodeNuisance,
    FirstOrderActuatorBackend,
    ObservationBackend,
    PlantBackend,
    PlantHypothesis,
    PlantParameters,
    RealizedWrench,
    RigidBodyObservationBackend,
    RigidBodyPlantBackend,
)
from grape_param_estim.plant.parameters import (
    ACTUATOR_PARAMETER_NAMES,
    CALIBRATED_RIGID_BODY_MODEL_ID,
    EFFECTIVE_CLOSED_LOOP_MODEL_ID,
    EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES,
)
from grape_param_estim.plant_assimilation import (
    _identifiability,
    validate_posterior,
)
from grape_param_estim.validation import FailureEvent


def _state(position=(0.0, 0.0, 1.0), omega=(0.0, 0.0, 0.0)):
    return np.asarray(
        tuple(position)
        + (0.0, 0.0, 0.0)
        + (0.0, 0.0, 0.0, 1.0)
        + tuple(omega),
        dtype=float,
    )


def _effective_hypothesis():
    return PlantHypothesis(
        model_id=EFFECTIVE_CLOSED_LOOP_MODEL_ID,
        plant_parameters=np.asarray(
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        actuator_parameters=np.asarray(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
        ),
        disturbance_parameters=np.zeros(0),
        plant_parameter_names=EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES,
        actuator_parameter_names=ACTUATOR_PARAMETER_NAMES,
    )


def _physical_values():
    return np.asarray(
        [2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.1, 0.0, 1.2]
    )


def _physical_hypothesis():
    return PlantHypothesis(
        model_id=CALIBRATED_RIGID_BODY_MODEL_ID,
        plant_parameters=_physical_values(),
        actuator_parameters=np.asarray(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
        ),
        disturbance_parameters=np.zeros(0),
    )


def _nuisance(position=(0.0, 0.0, 1.0), **changes):
    values = {
        "initial_plant_state": _state(position),
        "initial_actuator_state": np.zeros(0),
        "disturbance_parameters": np.zeros(6),
        "sensor_bias": np.zeros(6),
    }
    values.update(changes)
    return EpisodeNuisance(**values)


def _grids(integration, likelihood=None, controller=()):
    return SimpleNamespace(
        plant_integration_grid=np.asarray(integration, dtype=float),
        likelihood_grid=np.asarray(
            integration if likelihood is None else likelihood, dtype=float
        ),
        controller_tick_grid=np.asarray(controller, dtype=float),
    )


def _passing_exact_closed_loop_gate():
    identity = ControllerBackendIdentity(
        backend_id="gimbalrotor_controller_cpp/v2",
        fidelity="pc_exact",
        is_exact=True,
        capabilities=PC_EXACT_REQUIRED_CAPABILITIES,
        implementation_language="c++",
        source_commit="2786cc3e",
        artifact_sha256="a" * 64,
        protocol=EXACT_ORACLE_PROTOCOL,
    )
    oracle_identity = ExactOracleIdentity(
        protocol=EXACT_ORACLE_PROTOCOL,
        backend_id=identity.backend_id,
        implementation_language=identity.implementation_language,
        source_commit=identity.source_commit,
        artifact_sha256=identity.artifact_sha256,
        capabilities=identity.capabilities,
        fidelity=identity.fidelity,
    )
    payload = {"factual_replay": "runtime-test"}
    continuous = {
        channel: (
            np.asarray([[1.0]])
            if channel == "command_timestamp"
            else np.zeros((1, 1))
        )
        for channel in required_conformance_channels(
            oracle_identity.fidelity
        )
    }
    events = np.zeros(1, dtype=int)
    provenance = ExactOracleFixtureProvenance.create(
        source_bag_sha256="f" * 64,
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
        fidelity=oracle_identity.fidelity,
    )

    class Oracle:
        is_exact = True
        identity = oracle_identity

        def replay(self, request):
            return ExactOracleReplayOutput(
                identity=self.identity,
                continuous=continuous,
                events=events,
            )

    evidence = evaluate_exact_oracle_conformance(
        Oracle(), payload, fixture
    )
    report = evaluate_exact_closed_loop_gate(identity, evidence)
    if not report.passed:
        raise AssertionError("test exact-controller gate must pass")
    return report


def _write_artifact_bundle(directory, likelihood_components=None):
    posterior = PlantPosterior.from_arrays(
        (_effective_hypothesis(), _effective_hypothesis()),
        weights=np.asarray([0.7, 0.3]),
        log_likelihood=np.asarray([-1.0, -2.0]),
        model_id=(
            "open_loop_plant_identification/"
            + EFFECTIVE_CLOSED_LOOP_MODEL_ID
        ),
        prior_id="prior/v1",
        likelihood_id="likelihood/v1",
        controller_snapshot_id="1" * 64,
    )
    provenance = PlantRunProvenance(
        source_commit="deadbeef",
        source_bag_sha256=("2" * 64,),
        normalized_episode_sha256=("3" * 64,),
        controller_snapshot_sha256="1" * 64,
        controller_artifact_sha256="4" * 64,
        plant_backend_id="open_loop_effective_forward_v1",
        plant_backend_sha256="5" * 64,
        plant_geometry_profile_id="test_geometry/v1",
        plant_geometry_sha256="8" * 64,
        prior_id="prior/v1",
        likelihood_id="likelihood/v1",
        seed=7,
        config_sha256="6" * 64,
        fixture_sha256="7" * 64,
    )
    identifiability = local_identifiability(
        np.zeros((1, posterior.raw_parameters.shape[1])),
        posterior.raw_parameter_names,
        EFFECTIVE_CLOSED_LOOP_MODEL_ID,
        structural_gauge_dimension=1,
    )
    likelihood = LikelihoodComponents(
        episode_id="failure-04",
        pose=-1.0,
        orientation=0.0,
        velocity=0.0,
        imu=0.0,
        angular_velocity=0.0,
        command=0.0,
        failure_event=0.0,
        saturation_mode_event=0.0,
        scored_sample_count=2,
        censored_sample_count=0,
        total=-1.0,
        diagnostics=MappingProxyType({"source": "synthetic"}),
    )
    components = (
        (likelihood,)
        if likelihood_components is None
        else likelihood_components
    )
    destination = PlantAssimilationArtifactWriter(directory).write(
        run_id="runtime-test",
        posterior=posterior,
        provenance=provenance,
        controller_snapshot=MappingProxyType(
            {"snapshot_id": "1" * 64}
        ),
        controller_replay_audit={"passed": False},
        factual_replay_report={"passed": False},
        identifiability_report=identifiability,
        likelihood_components=components,
        posterior_predictive={
            "weights": posterior.weights,
            "failure_time": np.asarray([np.nan, 2.0]),
        },
        failure_validation={
            "event": FailureEvent(
                "ground_contact",
                2.0,
                "synthetic/v1",
                metadata=MappingProxyType({"sample": 1}),
            )
        },
        success_validation={"passed": True},
        interpretation="effective_plant_posterior",
    )
    return Path(destination), posterior, provenance


def _rewrite_manifest_with_valid_content_hash(destination, manifest):
    content = dict(manifest)
    content.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = stable_hash(content)
    (Path(destination) / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class SerializationAndIdentifiabilityTests(unittest.TestCase):
    def test_plain_data_handles_mapping_proxies_and_nonfinite_floats(self):
        parameters = PlantParameters.effective(
            _effective_hypothesis().plant_parameters,
            nested=MappingProxyType(
                {
                    "positive": float("inf"),
                    "negative": np.float64(-np.inf),
                    "missing": np.asarray([np.nan]),
                }
            ),
        )
        payload = plain_data(
            {
                "parameters": parameters,
                "failure": FailureEvent(
                    "ground_contact",
                    2.0,
                    "synthetic/v1",
                    metadata=MappingProxyType({"index": np.int64(4)}),
                ),
            }
        )
        self.assertEqual(
            payload["parameters"]["metadata"]["nested"]["positive"],
            "Infinity",
        )
        self.assertEqual(
            payload["parameters"]["metadata"]["nested"]["negative"],
            "-Infinity",
        )
        self.assertEqual(
            payload["parameters"]["metadata"]["nested"]["missing"],
            ["NaN"],
        )
        self.assertEqual(payload["failure"]["metadata"]["index"], 4)
        json.dumps(payload, allow_nan=False)

    def test_declared_gauge_is_reported_even_for_full_rank_jacobian(self):
        report = local_identifiability(
            np.eye(3),
            ("scale", "authority", "drag"),
            EFFECTIVE_CLOSED_LOOP_MODEL_ID,
            structural_gauge_dimension=1,
        )
        self.assertEqual(report.jacobian_rank, 2)
        self.assertEqual(report.structural_gauge_dimension, 1)
        self.assertEqual(report.excitation_nullity, 0)
        self.assertEqual(report.null_directions.shape, (1, 3))
        self.assertEqual(report.direction_coefficients.shape, (2, 3))
        self.assertFalse(report.direction_coefficients.flags.writeable)

    def test_episode_excitation_reports_direction_coefficients_and_all_samples(
        self,
    ):
        # Each nuisance realization excites a different raw parameter.  The
        # square-root weighting makes the stacked matrix a weighted local
        # sensitivity/Fisher factor for the episode.
        weighted_jacobian = np.asarray(
            [
                [np.sqrt(0.25) * 4.0, 0.0, 0.0],
                [0.0, np.sqrt(0.75) * 2.0, 0.0],
            ]
        )
        episode = episode_excitation_report(
            weighted_jacobian,
            ("mass_scale", "roll_authority", "drag"),
            episode_id="failure-maneuver-04",
            nuisance_sample_ids=("state-a", "state-b"),
            nuisance_sample_weights=(1.0, 3.0),
        )
        self.assertEqual(episode.episode_id, "failure-maneuver-04")
        self.assertEqual(
            episode.nuisance_sample_ids, ("state-a", "state-b")
        )
        self.assertEqual(episode.nuisance_sample_count, 2)
        np.testing.assert_allclose(
            episode.nuisance_sample_weights, [0.25, 0.75]
        )
        self.assertEqual(episode.jacobian_row_count, 2)
        self.assertEqual(episode.jacobian_rank, 2)
        np.testing.assert_allclose(
            episode.direction_coefficients,
            np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        )
        self.assertEqual(
            episode.direction_labels,
            ("sv_direction_0", "sv_direction_1"),
        )
        self.assertFalse(episode.direction_coefficients.flags.writeable)

        report = local_identifiability(
            weighted_jacobian,
            episode.parameter_names,
            EFFECTIVE_CLOSED_LOOP_MODEL_ID,
            episode_excitation=(episode,),
        )
        self.assertEqual(
            tuple(item.episode_id for item in report.episode_excitation),
            ("failure-maneuver-04",),
        )
        payload = plain_data(report)
        self.assertEqual(
            payload["episode_excitation"][0]["nuisance_sample_ids"],
            ["state-a", "state-b"],
        )
        self.assertEqual(
            payload["episode_excitation"][0]["direction_coefficients"],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        json.dumps(payload, allow_nan=False)

    def test_identifiability_aggregates_every_inference_episode_and_nuisance(
        self,
    ):
        hypothesis = _effective_hypothesis()
        center = np.concatenate(
            (
                hypothesis.plant_parameters,
                hypothesis.actuator_parameters[:5],
            )
        )
        names = (
            EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES
            + ACTUATOR_PARAMETER_NAMES[:5]
        )
        prior = IndependentBoundedPrior(
            tuple(
                PriorDimension(
                    name,
                    "bounded_uniform",
                    center[index] - 2.0,
                    center[index] + 2.0,
                )
                for index, name in enumerate(names)
            )
        )
        posterior = SimpleNamespace(
            particles=(hypothesis,),
            weights=np.asarray([1.0]),
            raw_parameters=np.asarray([hypothesis.vector]),
        )
        timestamps = np.asarray([0.0, 1.0])

        def observations(episode_id, role="inference_failure"):
            return SimpleNamespace(
                episode_id=episode_id,
                role=role,
                timestamps=timestamps,
            )

        prepared = {
            # Deliberately reverse insertion order: the artifact order must be
            # stable by episode ID rather than depend on mapping construction.
            "failure-b": SimpleNamespace(
                observations=observations("failure-b"),
                nuisance_samples=(
                    _nuisance(state_sample_id="b-state-0", weight=2.0),
                    _nuisance(state_sample_id="b-state-1", weight=2.0),
                ),
            ),
            "validation": SimpleNamespace(
                observations=observations(
                    "validation-success", role="validation_success"
                ),
                nuisance_samples=(
                    _nuisance(state_sample_id="must-not-run"),
                ),
            ),
            "failure-a": SimpleNamespace(
                observations=observations("failure-a"),
                nuisance_samples=(
                    _nuisance(state_sample_id="a-state-0", weight=1.0),
                    _nuisance(state_sample_id="a-state-1", weight=3.0),
                ),
            ),
        }
        excited_parameter = {
            ("failure-a", "a-state-0"): 0,
            ("failure-a", "a-state-1"): 1,
            ("failure-b", "b-state-0"): 2,
            ("failure-b", "b-state-1"): 3,
        }
        calls = []

        def rollout(raw, observed, nuisance):
            key = (observed.episode_id, nuisance.state_sample_id)
            calls.append(key)
            parameter = excited_parameter[key]
            positions = np.zeros((timestamps.size, 3))
            positions[:, 0] = (
                np.asarray(raw)[parameter] * np.asarray([1.0, 2.0])
            )
            return SimpleNamespace(
                integration_timestamps=timestamps,
                positions=positions,
            )

        report = _identifiability(
            posterior,
            prior,
            prepared,
            rollout,
        )
        self.assertEqual(
            tuple(item.episode_id for item in report.episode_excitation),
            ("failure-a", "failure-b"),
        )
        self.assertEqual(report.jacobian_rank, 4)
        for episode in report.episode_excitation:
            self.assertEqual(episode.nuisance_sample_count, 2)
            self.assertEqual(episode.jacobian_rank, 2)
            self.assertEqual(
                episode.jacobian_row_count,
                2 * timestamps.size * 3,
            )
        self.assertEqual(
            report.episode_excitation[0].nuisance_sample_ids,
            ("a-state-0", "a-state-1"),
        )
        np.testing.assert_allclose(
            report.episode_excitation[0].nuisance_sample_weights,
            [0.25, 0.75],
        )
        self.assertEqual(
            report.episode_excitation[1].nuisance_sample_ids,
            ("b-state-0", "b-state-1"),
        )
        np.testing.assert_allclose(
            report.episode_excitation[1].nuisance_sample_weights,
            [0.5, 0.5],
        )
        self.assertNotIn(
            ("validation-success", "must-not-run"),
            calls,
        )
        for key in excited_parameter:
            # One nominal rollout plus one perturbation per raw parameter.
            self.assertEqual(calls.count(key), 1 + len(names))

        # SVD vectors may rotate inside equal-singular-value blocks, so test
        # their projector: collectively they span exactly the four excited
        # raw parameter axes.
        projector = (
            report.direction_coefficients.T
            @ report.direction_coefficients
        )
        np.testing.assert_allclose(
            np.diag(projector),
            np.concatenate((np.ones(4), np.zeros(len(names) - 4))),
            atol=1.0e-10,
        )

    def test_artifact_bundle_serializes_frozen_metadata_and_backend_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            destination, posterior, provenance = _write_artifact_bundle(
                directory
            )
            manifest = verify_run_manifest(destination)
            self.assertEqual(
                {item.name for item in destination.iterdir()},
                set(REQUIRED_ARTIFACTS),
            )
            self.assertEqual(
                set(manifest["files"]),
                set(REQUIRED_ARTIFACTS) - {"run_manifest.json"},
            )
            self.assertEqual(
                manifest["provenance"]["plant_backend_sha256"], "5" * 64
            )
            expected_provenance = plain_data(provenance)
            expected_provenance["model_id"] = posterior.model_id
            for name in (
                item
                for item in REQUIRED_ARTIFACTS
                if item.endswith(".json")
            ):
                with self.subTest(json_artifact=name):
                    payload = json.loads(
                        (destination / name).read_text(encoding="utf-8")
                    )
                    self.assertEqual(
                        payload["artifact_provenance"],
                        expected_provenance,
                    )

            required_provenance_fields = (
                "source_commit",
                "source_bag_sha256",
                "controller_artifact_sha256",
                "model_id",
                "prior_id",
                "likelihood_id",
                "seed",
            )
            for name in (
                "posterior_particles.npz",
                "posterior_predictive.npz",
            ):
                with self.subTest(npz_artifact=name), np.load(
                    str(destination / name), allow_pickle=False
                ) as archive:
                    self.assertEqual(
                        json.loads(
                            str(archive["artifact_provenance_json"].item())
                        ),
                        expected_provenance,
                    )
                    for field in required_provenance_fields:
                        self.assertIn(
                            "artifact_provenance_{}".format(field),
                            archive.files,
                        )

            for name in (
                "posterior_hpd95.csv",
                "likelihood_components.csv",
            ):
                with self.subTest(csv_artifact=name), (
                    destination / name
                ).open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertTrue(rows)
                for row in rows:
                    self.assertEqual(
                        row["artifact_provenance_source_commit"],
                        provenance.source_commit,
                    )
                    self.assertEqual(
                        json.loads(
                            row[
                                "artifact_provenance_source_bag_sha256"
                            ]
                        ),
                        list(provenance.source_bag_sha256),
                    )
                    self.assertEqual(
                        row[
                            "artifact_provenance_controller_artifact_sha256"
                        ],
                        provenance.controller_artifact_sha256,
                    )
                    self.assertEqual(
                        row["artifact_provenance_model_id"],
                        posterior.model_id,
                    )
                    self.assertEqual(
                        row["artifact_provenance_prior_id"],
                        provenance.prior_id,
                    )
                    self.assertEqual(
                        row["artifact_provenance_likelihood_id"],
                        provenance.likelihood_id,
                    )
                    self.assertEqual(
                        int(row["artifact_provenance_seed"]),
                        provenance.seed,
                    )

            report = (Path(destination) / "REPORT.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("5" * 64, report)
            self.assertIn("## Artifact provenance", report)
            for value in (
                provenance.source_commit,
                provenance.source_bag_sha256[0],
                provenance.controller_artifact_sha256,
                posterior.model_id,
                provenance.prior_id,
                provenance.likelihood_id,
                str(provenance.seed),
            ):
                self.assertIn(value, report)
            identifiability_payload = json.loads(
                (Path(destination) / "identifiability_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                identifiability_payload["condition_number"], "Infinity"
            )
            self.assertEqual(
                identifiability_payload["direction_coefficients"], []
            )
            self.assertEqual(
                identifiability_payload["episode_excitation"], []
            )

    def test_manifest_rejects_missing_and_extra_disk_artifacts(self):
        cases = (
            (
                "missing",
                lambda destination: (
                    destination / "REPORT.md"
                ).unlink(),
            ),
            (
                "extra",
                lambda destination: (
                    destination / "unexpected.txt"
                ).write_text("unexpected\n", encoding="utf-8"),
            ),
        )
        for expected, mutation in cases:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as directory:
                    destination, _, _ = _write_artifact_bundle(directory)
                    mutation(destination)
                    with self.assertRaisesRegex(
                        ValueError,
                        "run artifact file set mismatch.*{}".format(
                            expected
                        ),
                    ):
                        verify_run_manifest(destination)

    def test_manifest_rejects_incomplete_and_extra_file_entries(self):
        def remove_required(files):
            files.pop("REPORT.md")

        def add_unexpected(files):
            files["unexpected.txt"] = {
                "sha256": "0" * 64,
                "bytes": 0,
            }

        cases = (
            ("missing", remove_required),
            ("extra", add_unexpected),
        )
        for expected, mutation in cases:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as directory:
                    destination, _, _ = _write_artifact_bundle(directory)
                    manifest_path = destination / "run_manifest.json"
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    mutation(manifest["files"])
                    _rewrite_manifest_with_valid_content_hash(
                        destination, manifest
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "run manifest files mismatch.*{}".format(expected),
                    ):
                        verify_run_manifest(destination)

    def test_manifest_rejects_missing_and_extra_top_level_identities(self):
        cases = (
            (
                "missing posterior_content_sha256",
                lambda manifest: manifest.pop(
                    "posterior_content_sha256"
                ),
            ),
            (
                "extra invented_identity",
                lambda manifest: manifest.update(
                    {"invented_identity": "0" * 64}
                ),
            ),
        )
        for expected, mutation in cases:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as directory:
                    destination, _, _ = _write_artifact_bundle(directory)
                    manifest_path = destination / "run_manifest.json"
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    mutation(manifest)
                    _rewrite_manifest_with_valid_content_hash(
                        destination, manifest
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "run manifest top-level fields mismatch.*{}".format(
                            expected
                        ),
                    ):
                        verify_run_manifest(destination)

    def test_manifest_rejects_false_top_level_identities(self):
        cases = (
            ("provenance", False, "provenance must be an object"),
            (
                "artifact_provenance",
                False,
                "artifact_provenance must be an object",
            ),
            (
                "posterior_particles_sha256",
                False,
                "posterior_particles_sha256 must be a lowercase SHA-256",
            ),
            (
                "posterior_content_sha256",
                False,
                "posterior_content_sha256 must be a lowercase SHA-256",
            ),
        )
        for field, false_value, expected in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as directory:
                    destination, _, _ = _write_artifact_bundle(directory)
                    manifest_path = destination / "run_manifest.json"
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    manifest[field] = false_value
                    _rewrite_manifest_with_valid_content_hash(
                        destination, manifest
                    )
                    with self.assertRaisesRegex(ValueError, expected):
                        verify_run_manifest(destination)

    def test_manifest_cross_checks_top_level_posterior_identities(self):
        cases = (
            (
                "posterior_particles_sha256",
                "posterior_particles_sha256 does not match its files entry",
            ),
            (
                "posterior_content_sha256",
                "posterior_content_sha256 does not match "
                "posterior_particles.npz",
            ),
        )
        for field, expected in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as directory:
                    destination, _, _ = _write_artifact_bundle(directory)
                    manifest_path = destination / "run_manifest.json"
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    manifest[field] = "0" * 64
                    _rewrite_manifest_with_valid_content_hash(
                        destination, manifest
                    )
                    with self.assertRaisesRegex(ValueError, expected):
                        verify_run_manifest(destination)

    def test_manifest_reconstructs_and_cross_checks_provenance(self):
        cases = (
            (
                "provenance",
                lambda manifest: manifest["provenance"].update(
                    {"controller_artifact_sha256": "a" * 64}
                ),
                "artifact_provenance does not match provenance",
            ),
            (
                "artifact_provenance",
                lambda manifest: manifest[
                    "artifact_provenance"
                ].update({"model_id": "forged-model/v1"}),
                "artifact_provenance does not match posterior_particles.npz",
            ),
        )
        for field, mutation, expected in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as directory:
                    destination, _, _ = _write_artifact_bundle(directory)
                    manifest_path = destination / "run_manifest.json"
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    mutation(manifest)
                    _rewrite_manifest_with_valid_content_hash(
                        destination, manifest
                    )
                    with self.assertRaisesRegex(ValueError, expected):
                        verify_run_manifest(destination)

    def test_empty_likelihood_csv_still_contains_literal_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            destination, posterior, provenance = _write_artifact_bundle(
                directory, likelihood_components=()
            )
            with (destination / "likelihood_components.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["episode_id"], "")
            self.assertEqual(rows[0]["total"], "")
            self.assertEqual(
                rows[0]["artifact_provenance_source_commit"],
                provenance.source_commit,
            )
            self.assertEqual(
                rows[0]["artifact_provenance_model_id"],
                posterior.model_id,
            )
            verify_run_manifest(destination)

    def test_artifact_publication_never_replaces_concurrent_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            destination = output_root / "runtime-test"
            original_write_payloads = (
                PlantAssimilationArtifactWriter._write_payloads
            )

            def write_payloads_then_compete(**arguments):
                original_write_payloads(**arguments)
                destination.mkdir()

            with mock.patch.object(
                PlantAssimilationArtifactWriter,
                "_write_payloads",
                side_effect=write_payloads_then_compete,
            ):
                with self.assertRaises(FileExistsError):
                    _write_artifact_bundle(directory)

            self.assertTrue(destination.is_dir())
            self.assertEqual(tuple(destination.iterdir()), ())
            self.assertFalse(
                any(
                    item.name.startswith(".runtime-test.staging.")
                    for item in output_root.iterdir()
                )
            )


class ActuatorAndPlantInvariantTests(unittest.TestCase):
    def test_episode_disturbance_prior_is_deterministic_and_marginalized(self):
        initial = InitialStatePosterior(
            episode_id="failure-1",
            stamp=0.0,
            samples=(_nuisance(),),
            source_trajectory_sha256="a" * 64,
        )
        arguments = {
            "model_id": (
                "effective_constant_acceleration_disturbance_v1"
            ),
            "lower": -np.ones(6),
            "upper": np.ones(6),
            "sample_count": 3,
            "seed": 17,
        }
        first = with_disturbance_samples(initial, **arguments)
        second = with_disturbance_samples(initial, **arguments)
        self.assertEqual(first.content_sha256, second.content_sha256)
        self.assertEqual(len(first.samples), 3)
        self.assertAlmostEqual(
            sum(item.weight for item in first.samples), 1.0
        )
        np.testing.assert_allclose(
            first.samples[0].disturbance_parameters, np.zeros(6)
        )
        self.assertTrue(
            any(
                np.any(item.disturbance_parameters != 0.0)
                for item in first.samples[1:]
            )
        )
        self.assertTrue(
            all(
                item.disturbance_model_id
                == "effective_constant_acceleration_disturbance_v1"
                for item in first.samples
            )
        )
        changed = with_disturbance_samples(
            initial, **dict(arguments, seed=18)
        )
        self.assertNotEqual(
            first.content_sha256, changed.content_sha256
        )

    def test_delay_releases_once_and_lag_continues_toward_target(self):
        backend = FirstOrderActuatorBackend()
        backend.reset(np.zeros(0))
        parameters = ActuatorParameters.first_order(
            motor_time_constant=1.0,
            command_delay=0.2,
        )
        command = CommandSample(0.0, np.ones(4), np.zeros(4))
        first = backend.step(
            command, parameters, 0.0, evaluation_stamp=0.0
        )
        before_delay = backend.step(
            command, parameters, 0.1, evaluation_stamp=0.1
        )
        released = backend.step(
            command, parameters, 0.1, evaluation_stamp=0.2
        )
        continued = backend.step(
            command, parameters, 0.1, evaluation_stamp=0.3
        )
        continued_again = backend.step(
            command, parameters, 0.1, evaluation_stamp=0.4
        )
        np.testing.assert_allclose(first.actuator_state[:4], 0.0)
        np.testing.assert_allclose(before_delay.actuator_state[:4], 0.0)
        np.testing.assert_allclose(
            released.actuator_state[:4], 0.0
        )
        np.testing.assert_allclose(
            continued.actuator_state[:4], 1.0 - np.exp(-0.1)
        )
        np.testing.assert_allclose(
            continued_again.actuator_state[:4], 1.0 - np.exp(-0.2)
        )

    def test_rigid_body_hover_free_fall_and_principal_axis_conservation(self):
        parameters = PlantParameters.calibrated_rigid_body(
            _physical_values()
        )
        backend = RigidBodyPlantBackend()
        backend.reset(_state(omega=(0.0, 0.0, 1.0)))
        hover = backend.step(
            RealizedWrench(
                stamp=0.1,
                force_body=np.asarray([0.0, 0.0, 2.0 * 9.80665]),
                torque_body=np.zeros(3),
                actuator_state=np.zeros(14),
                saturated=False,
                calibrated_wrench=True,
                model_id="calibrated_test",
            ),
            parameters,
            0.1,
        )
        np.testing.assert_allclose(hover.position_world, [0.0, 0.0, 1.0])
        np.testing.assert_allclose(hover.velocity_world, 0.0)
        np.testing.assert_allclose(
            hover.angular_velocity_body, [0.0, 0.0, 1.0]
        )
        np.testing.assert_allclose(hover.angular_acceleration_body, 0.0)

        backend.reset(_state())
        fallen = backend.step(
            RealizedWrench(
                stamp=0.1,
                force_body=np.zeros(3),
                torque_body=np.zeros(3),
                actuator_state=np.zeros(14),
                saturated=False,
                calibrated_wrench=True,
                model_id="calibrated_test",
            ),
            parameters,
            0.1,
        )
        self.assertAlmostEqual(fallen.velocity_world[2], -0.980665)
        self.assertAlmostEqual(
            fallen.position_world[2], 1.0 - 0.5 * 9.80665 * 0.1**2
        )

    def test_plan_protocols_and_forward_cache_compatibility_exist(self):
        self.assertIs(ForwardRolloutCache, RolloutCache)
        self.assertIsInstance(FirstOrderActuatorBackend(), ActuatorBackend)
        self.assertIsInstance(EffectiveRigidBodyPlantBackend(), PlantBackend)
        self.assertIsInstance(
            RigidBodyObservationBackend(), ObservationBackend
        )


class ForwardSchedulingTests(unittest.TestCase):
    def test_open_loop_has_no_controller_backend_path(self):
        calls = []

        def forbidden_controller_factory():
            calls.append(True)
            raise AssertionError("open-loop constructed a controller")

        with self.assertRaisesRegex(
            TypeError, "unexpected keyword argument"
        ):
            OpenLoopForwardModel(
                controller_backend_factory=forbidden_controller_factory
            )
        self.assertEqual(calls, [])

    def test_open_loop_never_applies_a_future_recorded_command(self):
        commands = RecordedCommandSeries(
            timestamps=np.asarray([0.0, 1.0]),
            base_thrust=np.asarray(
                [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
            ),
            gimbal_angle=np.zeros((2, 4)),
        )
        result = OpenLoopForwardModel(
            effective_plant_factory=lambda: EffectiveRigidBodyPlantBackend(
                gravity_m_s2=0.0
            )
        ).run(
            commands,
            _effective_hypothesis(),
            _nuisance(),
            _grids([0.0, 1.0, 2.0]),
        )
        np.testing.assert_allclose(result.positions[1], [0.0, 0.0, 1.0])
        self.assertGreater(result.positions[2, 2], 1.0)
        self.assertEqual(
            tuple(item.stamp for item in result.commands), (0.0, 1.0)
        )
        np.testing.assert_allclose(
            [item.stamp for item in result.plant_states],
            result.integration_timestamps,
        )

    def test_calibrated_forward_path_requires_calibration_identity(self):
        commands = RecordedCommandSeries(
            timestamps=np.asarray([0.0, 1.0]),
            base_thrust=np.zeros((2, 4)),
            gimbal_angle=np.zeros((2, 4)),
        )
        with self.assertRaisesRegex(
            ValueError, "actuator calibration identity"
        ):
            OpenLoopForwardModel().run(
                commands,
                _physical_hypothesis(),
                _nuisance(),
                _grids([0.0, 1.0]),
            )
        identity = ActuatorCalibrationIdentity(
            "a" * 64, "calibrated_first_order_gimbal_actuator_v1"
        )
        result = OpenLoopForwardModel(
            actuator_calibration_identity=identity
        ).run(
            commands,
            _physical_hypothesis(),
            _nuisance(),
            _grids([0.0, 1.0]),
        )
        self.assertTrue(result.realized_wrenches[0].calibrated_wrench)

    def test_episode_disturbance_uses_explicit_effective_units(self):
        commands = RecordedCommandSeries(
            timestamps=np.asarray([0.0, 1.0]),
            base_thrust=np.zeros((2, 4)),
            gimbal_angle=np.zeros((2, 4)),
        )
        nuisance = _nuisance(
            disturbance_parameters=np.asarray(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            disturbance_model_id=(
                "effective_constant_acceleration_disturbance_v1"
            ),
        )
        result = OpenLoopForwardModel().run(
            commands,
            _effective_hypothesis(),
            nuisance,
            _grids([0.0, 1.0]),
        )
        self.assertAlmostEqual(result.positions[-1, 0], 0.5)
        self.assertAlmostEqual(result.velocities[-1, 0], 1.0)
        with self.assertRaisesRegex(
            ValueError, "units/model"
        ):
            OpenLoopForwardModel().run(
                commands,
                _effective_hypothesis(),
                _nuisance(
                    disturbance_parameters=np.asarray(
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    )
                ),
                _grids([0.0, 1.0]),
            )


@dataclass(frozen=True)
class _FixtureInput:
    stamp: float
    dt: float
    position: object
    velocity: object
    orientation: object
    angular_velocity: object
    current_rpy: object
    joint_positions: object = ()
    allocation_geometry: object = ()


def _nominal_controller_geometry():
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
        "thrust_coordinate_rotations": [
            np.eye(3).tolist() for _ in range(4)
        ],
    }


def _rotation_about_x_for_test(angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, cosine, -sine),
            (0.0, sine, cosine),
        )
    )


class _Controller:
    def __init__(self):
        self.inputs = []
        self.reset_calls = 0

    def reset(self, snapshot, state):
        self.reset_calls += 1

    def step(self, item):
        self.inputs.append(item)
        return SimpleNamespace(
            command=SimpleNamespace(
                base_thrust=np.zeros(4),
                gimbal_angle=np.zeros(4),
                generalized_wrench=None,
                saturated=False,
            ),
            events=(),
        )


class _GimbalFeedbackController(_Controller):
    def step(self, item):
        self.inputs.append(item)
        feedback_angle = float(item.joint_positions[0])
        return SimpleNamespace(
            command=SimpleNamespace(
                base_thrust=np.full(4, feedback_angle),
                gimbal_angle=np.ones(4),
                generalized_wrench=None,
                saturated=False,
            ),
            events=(),
        )


class ClosedLoopSchedulingTests(unittest.TestCase):
    def test_duck_typed_gate_report_is_rejected(self):
        report = SimpleNamespace(
            passed=True,
            identity=SimpleNamespace(fidelity="pc_exact"),
        )
        with self.assertRaisesRegex(
            TypeError, "ExactClosedLoopGateReport"
        ):
            ClosedLoopForwardModel(lambda: _Controller(), report)

    def test_scheduler_prioritizes_fresh_plant_then_controller_then_likelihood(
        self,
    ):
        scheduler = _closed_loop_scheduler(
            np.asarray([10.0, 10.1, 10.2]),
            np.asarray([10.0, 10.2]),
            np.asarray([10.0, 10.2]),
        )
        self.assertEqual(
            scheduler.priority,
            ("plant_integration", "controller_tick", "likelihood"),
        )
        self.assertEqual(
            [
                event.grid_name
                for event in scheduler
                if np.isclose(event.time, 10.2)
            ],
            ["plant_integration", "controller_tick", "likelihood"],
        )

    def test_first_tick_runs_before_first_interval_and_feedback_is_fresh(self):
        controller = _Controller()
        report = _passing_exact_closed_loop_gate()
        fixture = SimpleNamespace(
            controller_inputs=(
                _FixtureInput(
                    10.0,
                    0.025,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3),
                    np.zeros(3),
                    np.full(3, 99.0),
                ),
                _FixtureInput(
                    10.2,
                    0.187,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3),
                    np.zeros(3),
                    np.full(3, 99.0),
                ),
            )
        )
        result = ClosedLoopForwardModel(
            lambda: controller, report
        ).run(
            fixture=fixture,
            snapshot=SimpleNamespace(
                nominal_geometry=_nominal_controller_geometry()
            ),
            hypothesis=_effective_hypothesis(),
            nuisance=_nuisance(controller_state=object()),
            grids=_grids(
                [10.0, 10.1, 10.2],
                likelihood=[10.05, 10.15],
                controller=[10.0, 10.2],
            ),
        )
        self.assertEqual(controller.reset_calls, 1)
        self.assertEqual(
            [item.stamp for item in controller.inputs], [10.0, 10.2]
        )
        self.assertAlmostEqual(controller.inputs[0].dt, 0.025)
        # Explicit live-controller dt is not reconstructed from stamp
        # differences: the legacy wrapper timestamps state at tick completion.
        self.assertAlmostEqual(controller.inputs[1].dt, 0.187)
        self.assertLess(controller.inputs[1].position[2], 1.0)
        np.testing.assert_allclose(controller.inputs[0].current_rpy, 0.0)
        np.testing.assert_allclose(controller.inputs[1].current_rpy, 0.0)
        self.assertEqual(
            tuple(item.stamp for item in result.commands), (10.0, 10.2)
        )
        self.assertEqual(len(result.plant_states), 3)
        np.testing.assert_allclose(
            result.likelihood_timestamps, [10.05, 10.15]
        )

    def test_actuator_lag_drives_later_controller_input_and_command(self):
        report = _passing_exact_closed_loop_gate()
        recorded_geometry = _nominal_controller_geometry()
        recorded_geometry["thrust_coordinate_rotations"] = [
            np.asarray(
                (
                    (0.0, -1.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                )
            ).tolist()
            for _ in range(4)
        ]
        fixture = SimpleNamespace(
            controller_inputs=tuple(
                _FixtureInput(
                    stamp,
                    0.1,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3),
                    np.zeros(3),
                    np.zeros(3),
                    joint_positions=(9.0, 9.0, 9.0, 9.0),
                    allocation_geometry=recorded_geometry,
                )
                for stamp in (0.0, 0.1, 0.2)
            )
        )
        snapshot = SimpleNamespace(
            nominal_geometry=_nominal_controller_geometry()
        )
        controllers = []

        def rollout(gimbal_time_constant):
            controller = _GimbalFeedbackController()
            controllers.append(controller)
            hypothesis = PlantHypothesis(
                model_id=EFFECTIVE_CLOSED_LOOP_MODEL_ID,
                plant_parameters=_effective_hypothesis().plant_parameters,
                actuator_parameters=np.asarray(
                    [
                        1.0,
                        0.0,
                        0.0,
                        gimbal_time_constant,
                        0.0,
                        0.0,
                        100.0,
                    ]
                ),
                disturbance_parameters=np.zeros(0),
                plant_parameter_names=EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES,
                actuator_parameter_names=ACTUATOR_PARAMETER_NAMES,
            )
            return ClosedLoopForwardModel(
                lambda: controller,
                report,
            ).run(
                fixture=fixture,
                snapshot=snapshot,
                hypothesis=hypothesis,
                nuisance=_nuisance(controller_state=object()),
                grids=_grids(
                    [0.0, 0.1, 0.2],
                    controller=[0.0, 0.1, 0.2],
                ),
            )

        fast = rollout(0.0)
        slow = rollout(1.0)
        fast_angle = float(controllers[0].inputs[1].joint_positions[0])
        slow_angle = float(controllers[1].inputs[1].joint_positions[0])
        self.assertAlmostEqual(fast_angle, 1.0)
        self.assertAlmostEqual(slow_angle, 1.0 - np.exp(-0.1))
        self.assertNotEqual(fast_angle, 9.0)
        self.assertNotEqual(slow_angle, 9.0)
        np.testing.assert_allclose(
            fast.commands[1].base_thrust,
            fast_angle,
        )
        np.testing.assert_allclose(
            slow.commands[1].base_thrust,
            slow_angle,
        )
        self.assertFalse(
            np.array_equal(
                fast.commands[1].base_thrust,
                slow.commands[1].base_thrust,
            )
        )
        fast_rotation = np.asarray(
            controllers[0]
            .inputs[1]
            .allocation_geometry["thrust_coordinate_rotations"][0]
        )
        slow_rotation = np.asarray(
            controllers[1]
            .inputs[1]
            .allocation_geometry["thrust_coordinate_rotations"][0]
        )
        np.testing.assert_allclose(
            fast_rotation,
            _rotation_about_x_for_test(fast_angle),
        )
        np.testing.assert_allclose(
            slow_rotation,
            _rotation_about_x_for_test(slow_angle),
        )
        self.assertFalse(
            np.array_equal(fast_rotation, slow_rotation)
        )


class PosteriorPredictiveOrchestrationTests(unittest.TestCase):
    def test_inference_episode_uses_conditional_nuisance_posterior(self):
        posterior = SimpleNamespace(
            particles=(_effective_hypothesis(),),
            weights=np.asarray([1.0]),
            credible_probability=0.95,
        )
        times = np.asarray([0.0, 1.0])
        observed = SimpleNamespace(
            episode_id="inference-episode",
            role="inference_failure",
            timestamps=times,
            position_world=np.tile(
                np.asarray([0.0, 0.0, 1.0]), (2, 1)
            ),
            failure_type=None,
            failure_time=None,
        )
        nuisance = (
            _nuisance(
                state_sample_id="low-likelihood",
                weight=0.5,
                disturbance_model_id=(
                    "effective_constant_acceleration_disturbance_v1"
                ),
            ),
            _nuisance(
                state_sample_id="high-likelihood",
                weight=0.5,
                disturbance_parameters=np.asarray(
                    [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
                ),
                disturbance_model_id=(
                    "effective_constant_acceleration_disturbance_v1"
                ),
            ),
        )
        prepared = {
            "inference-episode": SimpleNamespace(
                observations=observed,
                nuisance_samples=nuisance,
            ),
            "validation-success": SimpleNamespace(
                observations=SimpleNamespace(
                    episode_id="validation-success",
                    role="validation_success",
                    timestamps=times,
                    position_world=np.tile(
                        np.asarray([0.0, 0.0, 1.0]), (2, 1)
                    ),
                    failure_type=None,
                    failure_time=None,
                ),
                nuisance_samples=(nuisance[0],),
            ),
        }

        def rollout(_raw, _observations, state_sample):
            return SimpleNamespace(
                integration_timestamps=times,
                positions=np.tile(
                    state_sample.initial_plant_state[:3], (2, 1)
                ),
                orientations_xyzw=np.tile(
                    np.asarray([0.0, 0.0, 0.0, 1.0]), (2, 1)
                ),
                events=(),
                state_sample_id=state_sample.state_sample_id,
            )

        class ConditionalLikelihood:
            @staticmethod
            def evaluate(result, _observations):
                score = (
                    np.log(9.0)
                    if result.state_sample_id == "high-likelihood"
                    else 0.0
                )
                return LikelihoodComponents(
                    episode_id="inference-episode",
                    pose=float(score),
                    orientation=0.0,
                    velocity=0.0,
                    imu=0.0,
                    angular_velocity=0.0,
                    command=0.0,
                    failure_event=0.0,
                    saturation_mode_event=0.0,
                    scored_sample_count=2,
                    censored_sample_count=0,
                    total=float(score),
                )

        _, _, predictive, rows = validate_posterior(
            posterior,
            prepared,
            rollout,
            ConditionalLikelihood(),
        )
        np.testing.assert_allclose(
            predictive["inference_episode_weights"], [0.1, 0.9]
        )
        np.testing.assert_allclose(
            [row["conditional_nuisance_weight"] for row in rows],
            [0.1, 0.9],
        )
        self.assertEqual(
            predictive["inference_episode_disturbance_parameters"].shape,
            (2, 6),
        )

    def test_initial_state_samples_are_marginalized_and_saved(self):
        particles = (_effective_hypothesis(), _effective_hypothesis())
        posterior = SimpleNamespace(
            particles=particles,
            weights=np.asarray([0.75, 0.25]),
            credible_probability=0.95,
        )
        times = np.asarray([0.0, 1.0])
        observed = SimpleNamespace(
            episode_id="success-episode",
            role="validation_success",
            timestamps=times,
            position_world=np.asarray(
                [[0.5, 2.0, 1.0], [0.5, 2.0, 1.0]]
            ),
            failure_type=None,
            failure_time=None,
        )
        nuisance = (
            _nuisance(
                (0.0, 0.0, 1.0),
                state_sample_id="state-a",
                weight=0.2,
            ),
            _nuisance(
                (1.0, 0.0, 1.0),
                state_sample_id="state-b",
                weight=0.8,
            ),
        )
        prepared = {
            "success-episode": SimpleNamespace(
                observations=observed,
                nuisance_samples=nuisance,
            )
        }
        calls = []

        def rollout(raw, observations, state_sample):
            calls.append(state_sample.state_sample_id)
            position = np.tile(
                state_sample.initial_plant_state[:3], (times.size, 1)
            )
            return SimpleNamespace(
                integration_timestamps=times,
                positions=position,
                orientations_xyzw=np.tile(
                    np.asarray([0.0, 0.0, 0.0, 1.0]),
                    (times.size, 1),
                ),
                events=(),
            )

        (
            _,
            success,
            predictive,
            _,
        ) = validate_posterior(
            posterior,
            prepared,
            rollout,
            component_likelihood=object(),
            validation_config={
                "minimum_success_coverage": 0.60,
                "maximum_success_failure_probability": 0.01,
            },
        )
        self.assertEqual(
            calls, ["state-a", "state-b", "state-a", "state-b"]
        )
        self.assertEqual(
            predictive["success_episode_position_particles"].shape,
            (4, 2, 3),
        )
        np.testing.assert_allclose(
            predictive["success_episode_weights"],
            [0.15, 0.60, 0.05, 0.20],
        )
        np.testing.assert_array_equal(
            predictive["success_episode_particle_index"], [0, 0, 1, 1]
        )
        np.testing.assert_array_equal(
            predictive["success_episode_nuisance_sample_index"],
            [0, 1, 0, 1],
        )
        np.testing.assert_array_equal(
            predictive["success_episode_failure_indicator"], False
        )
        self.assertTrue(
            np.all(np.isnan(predictive["success_episode_failure_time"]))
        )
        self.assertEqual(
            float(predictive["success_episode_failure_probability"]), 0.0
        )
        self.assertTrue(success["passed"])
        self.assertAlmostEqual(
            success["episodes"][0]["trajectory_coverage"], 2.0 / 3.0
        )
        self.assertEqual(
            success["config"]["minimum_trajectory_coverage"], 0.60
        )

    def test_heldout_failure_reports_censored_trajectory_and_event_separately(
        self,
    ):
        posterior = SimpleNamespace(
            particles=(_effective_hypothesis(),),
            weights=np.asarray([1.0]),
            credible_probability=0.95,
        )
        times = np.arange(4.0)
        failure_observed = SimpleNamespace(
            episode_id="heldout-failure",
            role="validation_failure",
            timestamps=times,
            position_world=np.asarray(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.5],
                    [0.0, 0.0, -0.1],
                    [100.0, 100.0, 100.0],
                ]
            ),
            failure_type="ground_contact",
            failure_time=2.0,
        )
        success_observed = SimpleNamespace(
            episode_id="heldout-success",
            role="validation_success",
            timestamps=times,
            position_world=np.tile(
                np.asarray([0.0, 0.0, 1.0]), (times.size, 1)
            ),
            failure_type=None,
            failure_time=None,
        )
        nuisance = (_nuisance(state_sample_id="state-a"),)
        prepared = {
            "heldout-failure": SimpleNamespace(
                observations=failure_observed,
                nuisance_samples=nuisance,
            ),
            "heldout-success": SimpleNamespace(
                observations=success_observed,
                nuisance_samples=nuisance,
            ),
        }

        def rollout(raw, observations, state_sample):
            if observations.role == "validation_failure":
                positions = np.asarray(
                    [
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.5],
                        [0.0, 0.0, -0.1],
                        [0.0, 0.0, -5.0],
                    ]
                )
            else:
                positions = np.tile(
                    np.asarray([0.0, 0.0, 1.0]),
                    (times.size, 1),
                )
            return SimpleNamespace(
                integration_timestamps=times,
                positions=positions,
                orientations_xyzw=np.tile(
                    np.asarray([0.0, 0.0, 0.0, 1.0]),
                    (times.size, 1),
                ),
                events=(),
            )

        failure, success, _, _ = validate_posterior(
            posterior,
            prepared,
            rollout,
            component_likelihood=object(),
            validation_config={
                "minimum_failure_trajectory_coverage": 1.0,
                "minimum_success_coverage": 1.0,
                "maximum_success_failure_probability": 0.0,
            },
        )
        report = failure["held_out"][0]
        self.assertTrue(report["trajectory"]["passed"])
        self.assertEqual(
            report["trajectory"]["evaluated_time_count"], 3
        )
        self.assertTrue(report["event"]["passed"])
        self.assertEqual(
            report["censoring"]["score_mask"],
            [True, True, True, False],
        )
        self.assertEqual(
            report["censoring"]["censored_sample_count"], 1
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["reasons"], [])
        self.assertTrue(failure["passed"])
        self.assertTrue(success["passed"])
        self.assertEqual(
            failure["config"]["minimum_trajectory_coverage"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
