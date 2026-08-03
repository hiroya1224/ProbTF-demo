import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactStateError,
    ArtifactValidationError,
    CANCELLED_STATUS,
    IncompleteArtifactError,
    read_json,
    request_fingerprint,
    write_json_atomic,
    write_npz_atomic,
)
from grape_param_estim.augmented_parameter_artifact import (
    AUGMENTED_PARAMETER_ESTIMATE_SCHEMA,
    SEQUENTIAL_ENRTS_PATH_SEMANTICS,
    AugmentedParameterArtifactBagInput,
    diagonal_q_artifact_fingerprint,
    load_augmented_parameter_artifact,
    mark_augmented_parameter_artifact_cancelled,
    read_augmented_parameter_manifest,
    write_augmented_parameter_artifact,
)
from grape_param_estim.augmented_parameter_state import (
    MINIMUM_PROCESS_NOISE_MEMBER_COUNT,
)
from grape_param_estim.controller import ControllerConfig, initial_controller_state
from grape_param_estim.diagonal_q_artifact import (
    DiagonalQArtifactBagInput,
    write_diagonal_q_artifact,
)
from grape_param_estim.diagonal_q_em import (
    DiagonalQBagExpectation,
    DiagonalQEmConfig,
    DiagonalQInitialPilot,
    run_diagonal_q_em,
)
from grape_param_estim.multi_bag_augmented_parameter import (
    PreparedAugmentedParameterBag,
    run_multi_bag_augmented_parameter_filter,
)
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.real_rosbag import (
    ControllerGainSnapshot,
    EpisodeProvenance,
    RealFlightEpisode,
)
from grape_param_estim.stochastic_closed_loop import PoseObservationCovariance
from grape_param_estim.strong_constraint import StrongConstraintProblem
from grape_param_estim.synthetic import run_synthetic_experiment
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    GrapeGeometry,
    RigidBodyState,
    VehicleParameters,
)


def _descriptor_sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class AugmentedParameterArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.fingerprint = request_fingerprint({"test": "stage-2-artifact"})
        cls.project_fingerprint = request_fingerprint({"project": "grape"})
        cls.stage_fingerprint = request_fingerprint({"upstream": "q"})
        cls.config_fingerprint = "manual-group:sha256:" + "c" * 64
        cls.problems = {}
        cls.episodes = {}
        cls.nominals = {}
        for index, (bag_id, duration) in enumerate(
            (("bag-a", 0.08), ("bag-b", 0.12))
        ):
            synthetic = run_synthetic_experiment(
                duration=duration,
                time_step=0.04,
                truth_actuators=ActuatorParameters(delay=0.012),
                truth_residual_wrench=lambda _time, _state: np.zeros(6),
                translation_noise=0.002,
                rotation_noise=0.001,
                seed=710 + index,
            )
            configuration = ControllerConfig.grape()
            controller_state = initial_controller_state(
                configuration, trim_hover=True
            )
            parameters = VehicleParameters.nominal()
            initial_actuator = ActuatorState(
                synthetic.nominal.actuator_thrust[0],
                synthetic.nominal.actuator_gimbal_angle[0],
            )
            problem = StrongConstraintProblem(
                references=synthetic.references,
                observations=synthetic.observations,
                nominal_trajectory=synthetic.nominal,
                initial_state_anchor=RigidBodyState(
                    synthetic.nominal.position[0],
                    synthetic.nominal.orientation_xyzw[0],
                    synthetic.nominal.linear_velocity[0],
                    synthetic.nominal.angular_velocity[0],
                ),
                initial_controller_anchor=controller_state,
                controller_configuration=configuration,
                controller_parameters=parameters,
                geometry=GrapeGeometry.grape(),
                actuator_parameters=ActuatorParameters(delay=0.02),
                parameter_chart=VehicleParameterChart(parameters),
                initial_actuator_state=initial_actuator,
            )
            base = 100.0 + 10.0 * index
            local_start = 4.0 + index
            record_times = synthetic.observations.times + base
            provenance = EpisodeProvenance(
                bag_path="/flight/{}.bag".format(bag_id),
                bag_sha256=("a" if index == 0 else "b") * 64,
                bag_size_bytes=1000 + index,
                bag_record_start=base - local_start,
                bag_record_end=record_times[-1] + 2.0,
                time_basis="rosbag_record_time",
                requested_window_start=base,
                requested_window_end=base + duration + 0.01,
                source_available_start=base - 1.0,
                source_available_end=base + duration + 1.0,
                resample_period=0.04,
                selected_flight_state=5,
                flight_transition_record_times=np.asarray(
                    (base - 0.1, base + duration + 0.1)
                ),
                flight_transition_states=np.asarray((5, 4)),
                static_window_start=base - 2.0,
                static_window_end=base - 1.0,
                static_position_samples=20,
                static_position_inliers=19,
                static_orientation_samples=20,
                static_orientation_inliers=18,
                static_position_center=np.zeros(3),
                static_orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
                covariance_outlier_threshold=6.0,
                covariance_eigenvalue_floor=1.0e-12,
                controller_state_anchor_record_time=base - 0.02,
                joint_anchor_record_time=base - 0.02,
                thrust_anchor_record_time=base - 0.02,
                thrust_anchor_kind="recorded_command",
                reference_acceleration_kind="recorded_pid_reference",
                controller_static_source="ControllerConfig.grape",
                controller_source_revision="artifact-test",
                topic_names=("/flight_state", "/cog/odom"),
                topic_types=("std_msgs/UInt8", "nav_msgs/Odometry"),
            )
            snapshot = ControllerGainSnapshot(
                groups=("xy", "z", "roll_pitch", "yaw"),
                record_times=np.asarray(
                    (base - 0.4, base - 0.3, base - 0.2, base - 0.1)
                ),
                gains=np.asarray(
                    ((4.0, 0.1, 2.0), (5.0, 1.0, 2.5),
                     (13.0, 1.0, 20.0), (6.0, 1.0, 2.0))
                ),
                pid_control_flags=np.ones(4, dtype=bool),
                source_kinds=("dynamic_reconfigure_applied",) * 4,
            )
            episode = RealFlightEpisode(
                record_times=record_times,
                window_start_record_time=record_times[0],
                window_end_record_time=record_times[-1],
                window_start_local_time=local_start,
                window_end_local_time=local_start + duration,
                observations=synthetic.observations,
                references=synthetic.references,
                controller_configuration=configuration,
                initial_controller_state=controller_state,
                initial_actuator_state=initial_actuator,
                controller_snapshot=snapshot,
                provenance=provenance,
            )
            cls.problems[bag_id] = problem
            cls.episodes[bag_id] = episode
            cls.nominals[bag_id] = synthetic.nominal

        pilots = tuple(
            DiagonalQInitialPilot(
                bag_id,
                cls.problems[bag_id].observations.times.size,
                np.asarray((0.4, 0.5, 0.6, 0.08, 0.09, 0.10)),
            )
            for bag_id in ("bag-a", "bag-b")
        )

        def expectation_step(_covariance, _context):
            values = []
            for index, bag_id in enumerate(("bag-a", "bag-b")):
                times = cls.problems[bag_id].observations.times
                wrench = np.empty((3, times.size, 6), dtype=float)
                for member in range(3):
                    wrench[member] = (
                        (index + 1) * 0.03
                        + member * 0.01
                        + times[:, None] * np.arange(1.0, 7.0)[None, :]
                    )
                values.append(
                    DiagonalQBagExpectation(
                        bag_id,
                        times,
                        0.18 + index * 0.03,
                        wrench,
                        -10.0 - index,
                    )
                )
            return tuple(values)

        cls.q_result = run_diagonal_q_em(
            pilots,
            expectation_step,
            DiagonalQEmConfig(
                maximum_iterations=2,
                log_q_tolerance=1.0e-12,
                component_floor=np.full(6, 1.0e-9),
            ),
        )
        q_inputs = []
        for bag_id in ("bag-a", "bag-b"):
            episode = cls.episodes[bag_id]
            q_inputs.append(
                DiagonalQArtifactBagInput(
                    bag_id=bag_id,
                    source_path=episode.provenance.bag_path,
                    source_sha256=episode.provenance.bag_sha256,
                    source_size_bytes=episode.provenance.bag_size_bytes,
                    selected_interval_local_seconds=(
                        episode.window_start_local_time,
                        episode.window_end_local_time + 0.01,
                    ),
                    effective_interval_local_seconds=(
                        episode.window_start_local_time,
                        episode.window_end_local_time,
                    ),
                    episode_index=0 if bag_id == "bag-a" else 1,
                    configuration_fingerprint="complete:" + "d" * 64,
                    fixed_model_provenance={"model": "nominal", "bag": bag_id},
                    constant_delay_seconds=0.02,
                    translation_covariance=np.diag((0.01, 0.01, 0.01)),
                    rotation_covariance=np.diag((0.002, 0.002, 0.002)),
                    fixed_r_provenance={"method": "artifact-test"},
                )
            )
        cls.q_root = cls.root / "q"
        write_diagonal_q_artifact(
            cls.q_root,
            run_id="q-run",
            stage_id="diagonal_q",
            request_fingerprint=cls.fingerprint,
            project_fingerprint=cls.project_fingerprint,
            stage_input_fingerprint=cls.stage_fingerprint,
            implementation_provenance={
                "algorithm_version": "q-test-v1",
                "source_revision": "artifact-test",
                "source_dirty": False,
            },
            bag_inputs=q_inputs,
            result=cls.q_result,
            expectations=cls.q_result.final_expectations,
        )
        cls.q_fingerprint = diagonal_q_artifact_fingerprint(cls.q_root)
        r = PoseObservationCovariance.isotropic(0.02, 0.012)
        prepared = tuple(
            PreparedAugmentedParameterBag(
                bag_id,
                cls.problems[bag_id],
                r,
                0.18 + index * 0.03,
                cls.config_fingerprint,
            )
            for index, bag_id in enumerate(("bag-a", "bag-b"))
        )
        cls.result = run_multi_bag_augmented_parameter_filter(
            prepared,
            cls.q_result.covariance,
            ensemble_size=MINIMUM_PROCESS_NOISE_MEMBER_COUNT,
            seed=901,
            run_id="artifact-multi-bag",
        )
        cls.inputs = tuple(
            AugmentedParameterArtifactBagInput(
                bag_id=bag_id,
                episode_index=index,
                episode=cls.episodes[bag_id],
                problem=cls.problems[bag_id],
                nominal_trajectory=cls.nominals[bag_id],
                configuration_fingerprint=cls.config_fingerprint,
                model_provenance={
                    "implementation": "closed-loop-stepper-v1",
                    "bag": bag_id,
                },
            )
            for index, bag_id in enumerate(("bag-a", "bag-b"))
        )
        cls.bundle_root = cls.root / "stage2"
        cls._write(cls.bundle_root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def _write(cls, root):
        return write_augmented_parameter_artifact(
            root,
            run_id="parameter-run",
            stage_id="static_parameters",
            request_fingerprint=cls.fingerprint,
            project_fingerprint=cls.project_fingerprint,
            stage_input_fingerprint=cls.stage_fingerprint,
            implementation_provenance={
                "algorithm_version": "augmented-static-enkf-v1",
                "source_revision": "artifact-test",
                "source_dirty": False,
            },
            upstream_diagonal_q_path=cls.q_root,
            upstream_diagonal_q_fingerprint=cls.q_fingerprint,
            bag_inputs=cls.inputs,
            result=cls.result,
        )

    def _copy(self, directory):
        target = Path(directory) / "copy"
        shutil.copytree(self.bundle_root, target)
        return target

    def _refresh(self, root, manifest, descriptor):
        path = Path(root) / descriptor["path"]
        descriptor["sha256"] = _descriptor_sha(path)
        descriptor["size_bytes"] = path.stat().st_size
        write_json_atomic(Path(root) / "manifest.json", manifest)

    def test_round_trip_preserves_variable_length_real_paths_and_raw_law(self):
        bundle = load_augmented_parameter_artifact(self.bundle_root)
        manifest = bundle.manifest
        self.assertEqual(manifest["schema"], AUGMENTED_PARAMETER_ESTIMATE_SCHEMA)
        self.assertEqual(bundle.bag_ids, ("bag-a", "bag-b"))
        self.assertEqual(
            manifest["path_semantics"]["kind"],
            SEQUENTIAL_ENRTS_PATH_SEMANTICS,
        )
        self.assertFalse(
            manifest["path_semantics"]
            ["earlier_bags_recomputed_with_final_shared_posterior"]
        )
        shared = bundle.shared_posterior
        members = MINIMUM_PROCESS_NOISE_MEMBER_COUNT
        self.assertEqual(shared["final_shared_coordinates"].shape, (members, 19))
        self.assertEqual(shared["mass"].shape, (members,))
        self.assertEqual(shared["inertia"].shape, (members, 3, 3))
        self.assertEqual(shared["constant_delay"].shape, (members,))
        self.assertEqual(bundle.bags["bag-a"]["times"].size, 3)
        self.assertEqual(bundle.bags["bag-b"]["times"].size, 4)
        self.assertEqual(
            bundle.bags["bag-a"]["static_smoothed_coordinates"].shape,
            (members, 3, 19),
        )
        np.testing.assert_array_equal(
            bundle.bags["bag-a"]["static_smoothed_coordinates"][:, -1],
            bundle.bags["bag-b"]["initial_shared_coordinates"],
        )
        np.testing.assert_array_equal(
            bundle.bags["bag-b"]["static_smoothed_coordinates"][:, -1],
            shared["final_shared_coordinates"],
        )
        self.assertEqual(
            manifest["upstream_diagonal_q"]["artifact_fingerprint"],
            self.q_fingerprint,
        )
        self.assertEqual(
            manifest["bags"]["bag-a"]["configuration_fingerprint"],
            self.config_fingerprint,
        )
        self.assertEqual(
            manifest["bags"]["bag-a"]["controller_snapshot"]["groups"],
            ["xy", "z", "roll_pitch", "yaw"],
        )

    def test_payload_digest_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._copy(directory)
            manifest = read_json(root / "manifest.json")
            descriptor = manifest["artifacts"]["bags"]["bag-a"]
            path = root / descriptor["path"]
            payload = bytearray(path.read_bytes())
            payload[-7] ^= 1
            path.write_bytes(payload)
            with self.assertRaisesRegex(ArtifactValidationError, "SHA256"):
                load_augmented_parameter_artifact(root)

    def test_member_swap_with_refreshed_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._copy(directory)
            manifest = read_json(root / "manifest.json")
            descriptor = manifest["artifacts"]["bags"]["bag-b"]
            path = root / descriptor["path"]
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["member_id"] = arrays["member_id"][::-1]
            write_npz_atomic(path, arrays)
            self._refresh(root, manifest, descriptor)
            with self.assertRaisesRegex(ArtifactValidationError, "member IDs"):
                load_augmented_parameter_artifact(root)

    def test_extra_npz_member_is_rejected_even_with_refreshed_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._copy(directory)
            manifest = read_json(root / "manifest.json")
            descriptor = manifest["artifacts"]["shared_posterior"]
            path = root / descriptor["path"]
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["unexpected"] = np.zeros(1)
            write_npz_atomic(path, arrays)
            self._refresh(root, manifest, descriptor)
            with self.assertRaisesRegex(ArtifactValidationError, "ZIP members"):
                load_augmented_parameter_artifact(root)

    def test_decoded_law_tampering_is_rejected_against_raw_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._copy(directory)
            manifest = read_json(root / "manifest.json")
            descriptor = manifest["artifacts"]["shared_posterior"]
            path = root / descriptor["path"]
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["mass"] = arrays["mass"].copy()
            arrays["mass"][0] *= 1.01
            write_npz_atomic(path, arrays)
            self._refresh(root, manifest, descriptor)
            with self.assertRaisesRegex(
                ArtifactValidationError, "differs from raw coordinates"
            ):
                load_augmented_parameter_artifact(root)

    def test_manifest_provenance_and_upstream_q_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._copy(directory)
            manifest = read_json(root / "manifest.json")
            manifest["bags"]["bag-a"]["source_path"] = "/changed.bag"
            write_json_atomic(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ArtifactValidationError, "provenance"):
                load_augmented_parameter_artifact(root)
        with tempfile.TemporaryDirectory() as directory:
            root = self._copy(directory)
            manifest = read_json(root / "manifest.json")
            manifest["upstream_diagonal_q"]["final_stationary_variance"][0] *= 2.0
            write_json_atomic(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ArtifactValidationError, "fixed Q"):
                load_augmented_parameter_artifact(root)

    def test_incomplete_status_wins_before_malformed_nested_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._copy(directory)
            manifest = read_json(root / "manifest.json")
            manifest["status"] = "writing"
            manifest["bags"] = "malformed"
            write_json_atomic(root / "manifest.json", manifest)
            with self.assertRaises(IncompleteArtifactError):
                load_augmented_parameter_artifact(root)

    def test_upstream_fingerprint_and_manual_group_validation_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ArtifactValidationError, "fingerprint"):
                write_augmented_parameter_artifact(
                    Path(directory) / "wrong-upstream",
                    run_id="parameter-run",
                    stage_id="static_parameters",
                    request_fingerprint=self.fingerprint,
                    project_fingerprint=self.project_fingerprint,
                    stage_input_fingerprint=self.stage_fingerprint,
                    implementation_provenance={"algorithm": "test"},
                    upstream_diagonal_q_path=self.q_root,
                    upstream_diagonal_q_fingerprint="sha256:" + "0" * 64,
                    bag_inputs=self.inputs,
                result=self.result,
                )
        with self.assertRaisesRegex(ArtifactValidationError, "manual-group"):
            AugmentedParameterArtifactBagInput(
                bag_id="bag-a",
                episode_index=0,
                episode=self.episodes["bag-a"],
                problem=self.problems["bag-a"],
                nominal_trajectory=self.nominals["bag-a"],
                configuration_fingerprint="manual-group:" + "c" * 64,
                model_provenance={"model": "test"},
            )

    def test_writer_verifies_upstream_q_payload_digest_before_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            copied_q = Path(directory) / "q-copy"
            shutil.copytree(self.q_root, copied_q)
            manifest = read_json(copied_q / "manifest.json")
            descriptor = manifest["artifacts"]["bags"]["bag-a"]
            payload_path = copied_q / descriptor["path"]
            payload = bytearray(payload_path.read_bytes())
            payload[-5] ^= 1
            payload_path.write_bytes(payload)
            with self.assertRaisesRegex(ArtifactValidationError, "SHA256"):
                write_augmented_parameter_artifact(
                    Path(directory) / "stage2",
                    run_id="parameter-run",
                    stage_id="static_parameters",
                    request_fingerprint=self.fingerprint,
                    project_fingerprint=self.project_fingerprint,
                    stage_input_fingerprint=self.stage_fingerprint,
                    implementation_provenance={"algorithm": "test"},
                    upstream_diagonal_q_path=copied_q,
                    upstream_diagonal_q_fingerprint=self.q_fingerprint,
                    bag_inputs=self.inputs,
                    result=self.result,
                )

    def test_writing_bundle_can_be_cancelled_but_complete_bundle_cannot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cancelled"
            with mock.patch(
                "grape_param_estim.augmented_parameter_artifact.write_npz_atomic",
                side_effect=OSError("injected write failure"),
            ):
                with self.assertRaises(OSError):
                    self._write(root)
            mark_augmented_parameter_artifact_cancelled(root, "user stopped")
            manifest = read_augmented_parameter_manifest(root)
            self.assertEqual(manifest["status"], CANCELLED_STATUS)
            self.assertEqual(manifest["cancellation"]["reason"], "user stopped")
            with self.assertRaises(IncompleteArtifactError):
                load_augmented_parameter_artifact(root)
        with self.assertRaises(ArtifactStateError):
            mark_augmented_parameter_artifact_cancelled(
                self.bundle_root, "too late"
            )


if __name__ == "__main__":
    unittest.main()
