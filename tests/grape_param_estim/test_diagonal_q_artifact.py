import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactStateError,
    ArtifactValidationError,
    CANCELLED_STATUS,
    COMPLETE_STATUS,
    IncompleteArtifactError,
    WRITING_STATUS,
    load_npz_strict,
    read_json,
    request_fingerprint,
    write_json_atomic,
    write_npz_atomic,
)
from grape_param_estim.diagonal_q import (
    BODY_WRENCH_COMPONENT_ORDER,
    BODY_WRENCH_FRAME,
    BODY_WRENCH_VARIANCE_UNITS,
    shared_diagonal_q_m_step,
)
from grape_param_estim.diagonal_q_artifact import (
    DIAGONAL_Q_ESTIMATE_SCHEMA,
    DiagonalQArtifactBagInput,
    load_diagonal_q_artifact,
    mark_diagonal_q_artifact_cancelled,
    read_diagonal_q_manifest,
    write_diagonal_q_artifact,
)
from grape_param_estim.diagonal_q_em import (
    DiagonalQBagExpectation,
    DiagonalQEmConfig,
    DiagonalQInitialPilot,
    run_diagonal_q_em,
)


def _sha256(path):
    return "sha256:{}".format(hashlib.sha256(path.read_bytes()).hexdigest())


class DiagonalQArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.first_times = np.asarray((0.0, 0.12, 0.31))
        cls.second_times = np.asarray((0.0, 0.4))
        cls.first_wrench = np.asarray(
            (
                np.arange(18, dtype=float).reshape(3, 6) / 11.0,
                -np.arange(18, dtype=float).reshape(3, 6) / 17.0,
            )
        )
        cls.second_wrench = np.asarray(
            (
                ((0.2,) * 6, (0.5,) * 6),
                ((-0.3,) * 6, (0.1,) * 6),
            )
        )
        cls.pilots = (
            DiagonalQInitialPilot(
                "bag-z",
                3,
                np.asarray((1.0, 1.2, 1.4, 0.2, 0.3, 0.4)),
            ),
            DiagonalQInitialPilot(
                "bag-a",
                2,
                np.asarray((0.8, 0.9, 1.1, 0.4, 0.3, 0.2)),
            ),
        )
        cls.intervals = {
            "bag-a": (200.0, 200.43),
            "bag-z": (100.0, 100.34),
        }
        cls.effective_intervals = {
            "bag-a": (200.0, 200.4),
            "bag-z": (100.0, 100.31),
        }
        cls.request_fingerprint = request_fingerprint(
            {"request": "artifact-test", "revision": 1}
        )
        cls.project_fingerprint = request_fingerprint(
            {"project": "grape", "revision": 3}
        )
        cls.stage_input_fingerprint = request_fingerprint(
            {"stage": "diagonal_q", "upstream": "fixed-model-9"}
        )
        cls.implementation_provenance = {
            "algorithm_version": "diagonal-q-em-v1",
            "source_revision": "artifact-test-revision",
            "source_dirty": False,
        }

    def _bag_inputs(self):
        return (
            DiagonalQArtifactBagInput(
                bag_id="bag-z",
                source_path="/flight-data/bag-z.bag",
                source_sha256="f" * 64,
                source_size_bytes=17001,
                selected_interval_local_seconds=self.intervals["bag-z"],
                effective_interval_local_seconds=(
                    self.effective_intervals["bag-z"]
                ),
                episode_index=2,
                configuration_fingerprint="complete:" + "f" * 64,
                fixed_model_provenance={
                    "fixed_model": "z",
                    "revision": 4,
                },
                constant_delay_seconds=0.015,
                translation_covariance=np.diag((0.01, 0.02, 0.03)),
                rotation_covariance=np.diag((0.04, 0.05, 0.06)),
                fixed_r_provenance={
                    "method": "fixed_pose_pilot",
                    "calibration_id": "pose-r-z",
                },
            ),
            DiagonalQArtifactBagInput(
                bag_id="bag-a",
                source_path="/flight-data/bag-a.bag",
                source_sha256="a" * 64,
                source_size_bytes=12003,
                selected_interval_local_seconds=self.intervals["bag-a"],
                effective_interval_local_seconds=(
                    self.effective_intervals["bag-a"]
                ),
                episode_index=1,
                configuration_fingerprint=(
                    "manual-group:sha256:" + "a" * 64
                ),
                fixed_model_provenance={
                    "fixed_model": "a",
                    "revision": 7,
                },
                constant_delay_seconds=0.005,
                translation_covariance=np.diag((0.011, 0.021, 0.031)),
                rotation_covariance=np.diag((0.041, 0.051, 0.061)),
                fixed_r_provenance={
                    "method": "fixed_pose_pilot",
                    "calibration_id": "pose-r-a",
                },
            ),
        )

    def _result(self):
        def expectation_step(_covariance, iteration):
            return (
                DiagonalQBagExpectation(
                    "bag-z",
                    self.first_times,
                    0.35,
                    self.first_wrench,
                    -20.0 - float(iteration),
                ),
                DiagonalQBagExpectation(
                    "bag-a",
                    self.second_times,
                    0.7,
                    self.second_wrench,
                    -10.0 - float(iteration),
                ),
            )

        return run_diagonal_q_em(
            self.pilots,
            expectation_step,
            DiagonalQEmConfig(
                maximum_iterations=4,
                log_q_tolerance=1.0e-12,
                component_floor=np.asarray(
                    (1.0e-6, 2.0e-6, 3.0e-6, 4.0e-7, 5.0e-7, 6.0e-7)
                ),
            ),
        )

    def _write(self, root):
        result = self._result()
        destination = write_diagonal_q_artifact(
            root,
            run_id="run-q-17",
            stage_id="estimate_diagonal_q",
            request_fingerprint=self.request_fingerprint,
            project_fingerprint=self.project_fingerprint,
            stage_input_fingerprint=self.stage_input_fingerprint,
            implementation_provenance=self.implementation_provenance,
            bag_inputs=self._bag_inputs(),
            result=result,
            expectations=result.final_expectations,
        )
        return result, destination

    def _descriptor_path(self, root, manifest, bag_id=None):
        descriptor = (
            manifest["artifacts"]["em_trace"]
            if bag_id is None
            else manifest["artifacts"]["bags"][bag_id]
        )
        return Path(root) / descriptor["path"], descriptor

    def _refresh_descriptor(self, root, manifest, bag_id=None):
        path, descriptor = self._descriptor_path(root, manifest, bag_id)
        descriptor["sha256"] = _sha256(path)
        descriptor["size_bytes"] = path.stat().st_size
        write_json_atomic(Path(root) / "manifest.json", manifest)

    def test_complete_round_trip_preserves_provenance_trace_and_terminal_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            result, destination = self._write(directory)
            bundle = load_diagonal_q_artifact(destination)
            manifest = bundle.manifest

            self.assertEqual(destination, Path(directory).resolve())
            self.assertEqual(manifest["schema"], DIAGONAL_Q_ESTIMATE_SCHEMA)
            self.assertEqual(manifest["status"], COMPLETE_STATUS)
            self.assertEqual(manifest["run_id"], "run-q-17")
            self.assertEqual(manifest["stage_id"], "estimate_diagonal_q")
            self.assertEqual(
                manifest["request_fingerprint"], self.request_fingerprint
            )
            self.assertEqual(
                manifest["project_fingerprint"], self.project_fingerprint
            )
            self.assertEqual(
                manifest["stage_input_fingerprint"],
                self.stage_input_fingerprint,
            )
            self.assertEqual(
                manifest["implementation"]["provenance"],
                self.implementation_provenance,
            )
            self.assertEqual(bundle.bag_ids, ("bag-a", "bag-z"))
            self.assertEqual(
                manifest["body_wrench"]["frame"], BODY_WRENCH_FRAME
            )
            self.assertEqual(
                manifest["body_wrench"]["component_order"],
                list(BODY_WRENCH_COMPONENT_ORDER),
            )
            self.assertEqual(
                manifest["body_wrench"]["variance_units"],
                list(BODY_WRENCH_VARIANCE_UNITS),
            )
            input_by_id = {
                value.bag_id: value for value in self._bag_inputs()
            }
            loaded_input_by_id = {
                value.bag_id: value for value in bundle.bag_inputs
            }
            for bag_id in bundle.bag_ids:
                metadata = manifest["bags"][bag_id]
                expected_input = input_by_id[bag_id]
                loaded_input = loaded_input_by_id[bag_id]
                self.assertEqual(metadata["time_basis"], "episode_relative_seconds")
                self.assertEqual(
                    metadata["selected_interval_local_seconds"],
                    list(expected_input.selected_interval_local_seconds),
                )
                self.assertEqual(
                    metadata["effective_interval_local_seconds"],
                    list(expected_input.effective_interval_local_seconds),
                )
                self.assertEqual(metadata["source_path"], expected_input.source_path)
                self.assertEqual(
                    metadata["source_sha256"], expected_input.source_sha256
                )
                self.assertEqual(
                    metadata["source_size_bytes"],
                    expected_input.source_size_bytes,
                )
                self.assertTrue(
                    metadata["fixed_observation_covariance"]["fixed"]
                )
                self.assertEqual(
                    metadata["fixed_observation_covariance"]["provenance"],
                    expected_input.fixed_r_provenance,
                )
                self.assertEqual(
                    metadata["fixed_model_provenance"],
                    expected_input.fixed_model_provenance,
                )
                self.assertEqual(
                    metadata["fixed_observation_covariance"]
                    ["covariance_fingerprint"],
                    expected_input.observation_covariance_fingerprint,
                )
                np.testing.assert_array_equal(
                    loaded_input.translation_covariance,
                    expected_input.translation_covariance,
                )
                np.testing.assert_array_equal(
                    loaded_input.rotation_covariance,
                    expected_input.rotation_covariance,
                )
            np.testing.assert_array_equal(
                bundle.covariance.stationary_variance,
                result.covariance.stationary_variance,
            )
            np.testing.assert_array_equal(
                manifest["smoothed_wrench_input_stationary_variance"],
                result.final_expectation_input_covariance.stationary_variance,
            )
            self.assertEqual(
                manifest["em"]["smoothed_wrench_semantics"],
                "terminal_e_step_conditioned_on_final_q",
            )
            terminal_implied = shared_diagonal_q_m_step(
                tuple(
                    value.sufficient_statistics
                    for value in result.final_expectations
                ),
                result.config.component_floor,
            )
            np.testing.assert_array_equal(
                manifest["terminal_implied_raw_stationary_variance"],
                terminal_implied.raw_stationary_variance,
            )
            np.testing.assert_array_equal(
                manifest["terminal_implied_stationary_variance"],
                terminal_implied.covariance.stationary_variance,
            )

            self.assertEqual(
                bundle.trace["iteration"].tolist(),
                [value.iteration for value in result.iterations],
            )
            np.testing.assert_array_equal(
                bundle.trace["raw_stationary_variance"],
                [
                    value.update.raw_stationary_variance
                    for value in result.iterations
                ],
            )
            np.testing.assert_array_equal(
                bundle.trace["floor_applied"],
                [value.update.floor_applied for value in result.iterations],
            )
            for loaded, expected in zip(
                bundle.expectations, result.final_expectations
            ):
                self.assertEqual(loaded.bag_id, expected.bag_id)
                self.assertEqual(
                    loaded.correlation_time, expected.correlation_time
                )
                self.assertEqual(
                    loaded.approx_log_likelihood,
                    expected.approx_log_likelihood,
                )
                np.testing.assert_array_equal(loaded.times, expected.times)
                np.testing.assert_array_equal(
                    loaded.smoothed_wrench, expected.smoothed_wrench
                )
            for loaded, expected in zip(
                bundle.last_em_statistics, result.last_expectations
            ):
                statistics = expected.sufficient_statistics
                self.assertEqual(loaded.bag_id, statistics.bag_id)
                self.assertEqual(loaded.member_count, statistics.member_count)
                np.testing.assert_array_equal(
                    loaded.initial_second_moment,
                    statistics.initial_second_moment,
                )
                np.testing.assert_array_equal(
                    loaded.transition_second_moment,
                    statistics.transition_second_moment,
                )

            descriptors = [manifest["artifacts"]["em_trace"]]
            descriptors.extend(
                manifest["artifacts"]["bags"][bag_id]
                for bag_id in bundle.bag_ids
            )
            self.assertEqual(
                len({value["path"] for value in descriptors}),
                len(descriptors),
            )
            for descriptor in descriptors:
                path = destination / descriptor["path"]
                self.assertEqual(descriptor["sha256"], _sha256(path))
                self.assertEqual(descriptor["size_bytes"], path.stat().st_size)
                arrays = load_npz_strict(path)
                self.assertTrue(
                    all(not value.dtype.hasobject for value in arrays.values())
                )

            bundle.trace["iteration"][0] = 99
            reloaded = load_diagonal_q_artifact(destination)
            self.assertEqual(reloaded.trace["iteration"][0], 1)
            with self.assertRaises(ArtifactStateError):
                mark_diagonal_q_artifact_cancelled(destination, "too_late")
            with self.assertRaises(ArtifactStateError):
                self._write(directory)

    def test_writing_and_cancelled_manifests_are_never_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "grape_param_estim.diagonal_q_artifact.write_npz_atomic",
                side_effect=OSError("injected write failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    self._write(directory)
            manifest = read_diagonal_q_manifest(directory)
            self.assertEqual(manifest["status"], WRITING_STATUS)
            malformed = dict(manifest)
            malformed["bags"] = "intentionally malformed nested payload"
            write_json_atomic(Path(directory) / "manifest.json", malformed)
            with self.assertRaises(IncompleteArtifactError):
                load_diagonal_q_artifact(directory)
            write_json_atomic(Path(directory) / "manifest.json", manifest)

            mark_diagonal_q_artifact_cancelled(directory, "user_requested")
            cancelled = read_diagonal_q_manifest(directory)
            self.assertEqual(cancelled["status"], CANCELLED_STATUS)
            self.assertEqual(cancelled["cancellation_reason"], "user_requested")
            with self.assertRaises(IncompleteArtifactError):
                load_diagonal_q_artifact(directory)
            mark_diagonal_q_artifact_cancelled(directory, "user_requested")
            with self.assertRaises(ArtifactStateError):
                mark_diagonal_q_artifact_cancelled(directory, "timeout")

    def test_digest_path_traversal_and_symlink_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _result, root = self._write(directory)
            manifest = read_json(root / "manifest.json")
            bag_path, _descriptor = self._descriptor_path(
                root, manifest, "bag-a"
            )
            with bag_path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(
                ArtifactValidationError, "size|SHA256"
            ):
                load_diagonal_q_artifact(root)

        for malicious in (
            "../escape.npz",
            "/tmp/escape.npz",
            "bags\\0000.npz",
            "bags/\x00.npz",
            "bags/./0000.npz",
            "C:/escape.npz",
        ):
            with self.subTest(path=malicious):
                with tempfile.TemporaryDirectory() as directory:
                    _result, root = self._write(directory)
                    manifest = read_json(root / "manifest.json")
                    manifest["artifacts"]["bags"]["bag-a"][
                        "path"
                    ] = malicious
                    write_json_atomic(root / "manifest.json", manifest)
                    with self.assertRaisesRegex(
                        ArtifactValidationError, "inside|path"
                    ):
                        load_diagonal_q_artifact(root)

        with tempfile.TemporaryDirectory() as workspace:
            outside = Path(workspace) / "outside"
            outside.mkdir()
            _result, root = self._write(Path(workspace) / "bundle")
            manifest = read_json(root / "manifest.json")
            external = outside / "external.npz"
            external.write_bytes(b"external")
            link = root / "bags" / "escape.npz"
            link.symlink_to(external)
            descriptor = manifest["artifacts"]["bags"]["bag-a"]
            descriptor["path"] = "bags/escape.npz"
            descriptor["sha256"] = _sha256(external)
            descriptor["size_bytes"] = external.stat().st_size
            write_json_atomic(root / "manifest.json", manifest)
            with self.assertRaisesRegex(
                ArtifactValidationError, "outside|symbolic"
            ):
                load_diagonal_q_artifact(root)

        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace) / "bundle"
            root.mkdir()
            outside = Path(workspace) / "outside"
            outside.mkdir()
            (root / "bags").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ArtifactStateError, "outside"):
                self._write(root)
            self.assertEqual(tuple(outside.iterdir()), ())
            self.assertFalse((root / "manifest.json").exists())

        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace) / "bundle"
            root.mkdir()
            _result, valid_root = self._write(Path(workspace) / "source")
            (root / "manifest.json").symlink_to(valid_root / "manifest.json")
            with self.assertRaisesRegex(ArtifactValidationError, "symbolic"):
                load_diagonal_q_artifact(root)

    def test_manifest_missing_extra_and_bad_fingerprint_are_rejected(self):
        mutations = (
            ("extra", lambda value: value.update({"unexpected": 1})),
            ("missing", lambda value: value.pop("run_id")),
            (
                "nested extra",
                lambda value: value["em"].update({"unexpected": 1}),
            ),
            (
                "bad fingerprint",
                lambda value: value.update({"request_fingerprint": "sha256:bad"}),
            ),
            (
                "bag map missing",
                lambda value: value["bags"].pop("bag-a"),
            ),
            (
                "implementation provenance changed",
                lambda value: value["implementation"]["provenance"].update(
                    {"source_revision": "different-revision"}
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                _result, root = self._write(directory)
                manifest = read_json(root / "manifest.json")
                mutate(manifest)
                write_json_atomic(root / "manifest.json", manifest)
                with self.assertRaises(ArtifactValidationError):
                    load_diagonal_q_artifact(root)

    def test_npz_missing_extra_and_object_arrays_are_rejected(self):
        for kind in ("missing", "extra", "object"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                _result, root = self._write(directory)
                manifest = read_json(root / "manifest.json")
                path, _descriptor = self._descriptor_path(
                    root, manifest, "bag-a"
                )
                arrays = load_npz_strict(path)
                if kind == "missing":
                    arrays.pop("times")
                    write_npz_atomic(path, arrays)
                elif kind == "extra":
                    arrays["unexpected"] = np.zeros(1)
                    write_npz_atomic(path, arrays)
                else:
                    arrays["smoothed_wrench"] = np.full(
                        arrays["smoothed_wrench"].shape,
                        object(),
                        dtype=object,
                    )
                    np.savez_compressed(str(path), **arrays)
                self._refresh_descriptor(root, manifest, "bag-a")
                with self.assertRaises(ArtifactValidationError):
                    load_diagonal_q_artifact(root)

    def test_shape_finite_bag_identity_and_cross_file_metadata_are_strict(self):
        mutations = (
            (
                "nan wrench",
                lambda arrays: arrays["smoothed_wrench"].__setitem__(
                    (0, 0, 0), np.nan
                ),
            ),
            (
                "wrong shape",
                lambda arrays: arrays.update(
                    {"smoothed_wrench": arrays["smoothed_wrench"][:, :, :5]}
                ),
            ),
            (
                "wrong dtype",
                lambda arrays: arrays.update(
                    {"times": np.arange(arrays["times"].size, dtype=np.int64)}
                ),
            ),
            (
                "wrong bag",
                lambda arrays: arrays.update({"bag_id": np.asarray(("bag-z",))}),
            ),
            (
                "wrong correlation",
                lambda arrays: arrays.update(
                    {"correlation_time": np.asarray((9.0,))}
                ),
            ),
            (
                "wrong conditioning q",
                lambda arrays: arrays.update(
                    {
                        "smoothed_wrench_input_stationary_variance": (
                            arrays[
                                "smoothed_wrench_input_stationary_variance"
                            ]
                            * 2.0
                        )
                    }
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                _result, root = self._write(directory)
                manifest = read_json(root / "manifest.json")
                path, _descriptor = self._descriptor_path(
                    root, manifest, "bag-a"
                )
                arrays = load_npz_strict(path)
                mutate(arrays)
                write_npz_atomic(path, arrays)
                self._refresh_descriptor(root, manifest, "bag-a")
                with self.assertRaises(ArtifactValidationError):
                    load_diagonal_q_artifact(root)

    def test_trace_and_pilot_cross_checks_reject_digest_consistent_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            _result, root = self._write(directory)
            manifest = read_json(root / "manifest.json")
            path, _descriptor = self._descriptor_path(root, manifest)
            arrays = load_npz_strict(path)
            arrays["maximum_absolute_log_q_change"][0] += 0.25
            write_npz_atomic(path, arrays)
            self._refresh_descriptor(root, manifest)
            with self.assertRaisesRegex(ArtifactValidationError, "log-Q"):
                load_diagonal_q_artifact(root)

        with tempfile.TemporaryDirectory() as directory:
            _result, root = self._write(directory)
            manifest = read_json(root / "manifest.json")
            manifest["bags"]["bag-a"][
                "pilot_stationary_standard_deviation"
            ][0] *= 2.0
            write_json_atomic(root / "manifest.json", manifest)
            with self.assertRaisesRegex(
                ArtifactValidationError, "initial covariance"
            ):
                load_diagonal_q_artifact(root)

        with tempfile.TemporaryDirectory() as directory:
            _result, root = self._write(directory)
            manifest = read_json(root / "manifest.json")
            manifest["smoothed_wrench_input_stationary_variance"][0] *= 2.0
            write_json_atomic(root / "manifest.json", manifest)
            with self.assertRaisesRegex(
                ArtifactValidationError, "conditioned on final Q"
            ):
                load_diagonal_q_artifact(root)

    def test_bag_payload_swap_is_rejected_after_digest_and_size_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            _result, root = self._write(directory)
            manifest = read_json(root / "manifest.json")
            first_path, _first_descriptor = self._descriptor_path(
                root, manifest, "bag-a"
            )
            second_path, _second_descriptor = self._descriptor_path(
                root, manifest, "bag-z"
            )
            first_payload = first_path.read_bytes()
            second_payload = second_path.read_bytes()
            first_path.write_bytes(second_payload)
            second_path.write_bytes(first_payload)
            self._refresh_descriptor(root, manifest, "bag-a")
            self._refresh_descriptor(root, manifest, "bag-z")
            refreshed = read_json(root / "manifest.json")
            for bag_id in ("bag-a", "bag-z"):
                path, descriptor = self._descriptor_path(
                    root, refreshed, bag_id
                )
                self.assertEqual(descriptor["sha256"], _sha256(path))
                self.assertEqual(descriptor["size_bytes"], path.stat().st_size)
            with self.assertRaisesRegex(ArtifactValidationError, "bag_id"):
                load_diagonal_q_artifact(root)

    def test_r_payload_and_last_m_step_statistics_are_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            _result, root = self._write(directory)
            manifest = read_json(root / "manifest.json")
            path, _descriptor = self._descriptor_path(
                root, manifest, "bag-a"
            )
            arrays = load_npz_strict(path)
            arrays["observation_translation_covariance"][0, 0] *= 2.0
            write_npz_atomic(path, arrays)
            self._refresh_descriptor(root, manifest, "bag-a")
            with self.assertRaisesRegex(
                ArtifactValidationError, "observation covariance payload"
            ):
                load_diagonal_q_artifact(root)

        with tempfile.TemporaryDirectory() as directory:
            _result, root = self._write(directory)
            manifest = read_json(root / "manifest.json")
            manifest["bags"]["bag-a"]["fixed_model_provenance"][
                "revision"
            ] = 999
            write_json_atomic(root / "manifest.json", manifest)
            with self.assertRaisesRegex(
                ArtifactValidationError, "fixed model fingerprint"
            ):
                load_diagonal_q_artifact(root)

        with tempfile.TemporaryDirectory() as directory:
            _result, root = self._write(directory)
            manifest = read_json(root / "manifest.json")
            path, _descriptor = self._descriptor_path(
                root, manifest, "bag-a"
            )
            arrays = load_npz_strict(path)
            arrays["last_em_initial_second_moment"][0] *= 1.25
            write_npz_atomic(path, arrays)
            self._refresh_descriptor(root, manifest, "bag-a")
            with self.assertRaisesRegex(
                ArtifactValidationError, "final M-step"
            ):
                load_diagonal_q_artifact(root)

        with tempfile.TemporaryDirectory() as directory:
            _result, root = self._write(directory)
            manifest = read_json(root / "manifest.json")
            manifest["bags"]["bag-a"]["fixed_observation_covariance"][
                "provenance"
            ]["calibration_id"] = "different-calibration"
            write_json_atomic(root / "manifest.json", manifest)
            with self.assertRaisesRegex(
                ArtifactValidationError, "observation covariance payload"
            ):
                load_diagonal_q_artifact(root)

    def test_episode_relative_time_origin_and_duration_are_strict(self):
        for name, mutate in (
            (
                "nonzero origin",
                lambda arrays: arrays["times"].__setitem__(0, 0.01),
            ),
            (
                "wrong duration",
                lambda arrays: arrays["times"].__setitem__(-1, 0.3),
            ),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                _result, root = self._write(directory)
                manifest = read_json(root / "manifest.json")
                path, _descriptor = self._descriptor_path(
                    root, manifest, "bag-a"
                )
                arrays = load_npz_strict(path)
                mutate(arrays)
                write_npz_atomic(path, arrays)
                self._refresh_descriptor(root, manifest, "bag-a")
                with self.assertRaisesRegex(
                    ArtifactValidationError, "zero|duration"
                ):
                    load_diagonal_q_artifact(root)

    def test_bag_input_rejects_non_spd_observation_covariance(self):
        common = {
            "bag_id": "bag-a",
            "source_path": "/flight-data/bag-a.bag",
            "source_sha256": "a" * 64,
            "source_size_bytes": 1,
            "selected_interval_local_seconds": (0.0, 1.0),
            "effective_interval_local_seconds": (0.0, 1.0),
            "episode_index": 0,
            "configuration_fingerprint": "complete:" + "a" * 64,
            "fixed_model_provenance": {"fixed_model": "a"},
            "constant_delay_seconds": 0.0,
            "rotation_covariance": np.eye(3),
            "fixed_r_provenance": {"method": "fixed"},
        }
        for name, covariance in (
            (
                "not symmetric",
                np.asarray(
                    (
                        (1.0, 1.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0),
                    )
                ),
            ),
            ("not positive definite", np.diag((1.0, 0.0, 1.0))),
            ("not finite", np.diag((1.0, np.nan, 1.0))),
        ):
            with self.subTest(name=name), self.assertRaises(
                ArtifactValidationError
            ):
                DiagonalQArtifactBagInput(
                    translation_covariance=covariance, **common
                )
        for field, value in (
            ("constant_delay_seconds", -0.001),
            ("source_path", "bad\x00path"),
            ("configuration_fingerprint", "complete:not-a-digest"),
        ):
            invalid = dict(common)
            invalid[field] = value
            with self.subTest(field=field), self.assertRaises(
                ArtifactValidationError
            ):
                DiagonalQArtifactBagInput(
                    translation_covariance=np.eye(3), **invalid
                )

    def test_bag_input_accepts_manual_configuration_group_fingerprint(self):
        value = DiagonalQArtifactBagInput(
            bag_id="bag-a",
            source_path="/flight-data/bag-a.bag",
            source_sha256="a" * 64,
            source_size_bytes=1,
            selected_interval_local_seconds=(0.0, 1.0),
            effective_interval_local_seconds=(0.0, 1.0),
            episode_index=0,
            configuration_fingerprint="manual-group:sha256:" + "b" * 64,
            fixed_model_provenance={"fixed_model": "a"},
            constant_delay_seconds=0.0,
            translation_covariance=np.eye(3),
            rotation_covariance=np.eye(3),
            fixed_r_provenance={"method": "fixed"},
        )
        self.assertEqual(
            value.configuration_fingerprint,
            "manual-group:sha256:" + "b" * 64,
        )

    def test_atomic_npz_writer_rejects_object_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.npz"
            with self.assertRaisesRegex(
                ArtifactValidationError, "object dtype"
            ):
                write_npz_atomic(
                    path, {"unsafe": np.asarray((object(),), dtype=object)}
                )
            self.assertFalse(path.exists())

    def test_terminal_paths_need_not_imply_the_max_iteration_output_q(self):
        pilot = DiagonalQInitialPilot("bag-a", 2, np.ones(6))

        def expectation_step(covariance, iteration):
            variance = covariance.stationary_variance * 2.0
            scale = np.sqrt(variance)
            wrench = np.stack(
                (
                    np.stack((scale, 0.5 * scale), axis=0),
                    np.stack((-scale, -0.5 * scale), axis=0),
                ),
                axis=0,
            )
            return (
                DiagonalQBagExpectation(
                    "bag-a", (0.0, 1.0), 0.5, wrench, -float(iteration)
                ),
            )

        result = run_diagonal_q_em(
            (pilot,),
            expectation_step,
            DiagonalQEmConfig(1, 1.0e-12, 1.0e-9),
        )
        self.assertFalse(result.converged)
        terminal_implied = shared_diagonal_q_m_step(
            tuple(
                value.sufficient_statistics
                for value in result.final_expectations
            ),
            result.config.component_floor,
        ).covariance
        self.assertFalse(
            np.array_equal(
                terminal_implied.stationary_variance,
                result.covariance.stationary_variance,
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            root = write_diagonal_q_artifact(
                directory,
                run_id="run-maximum",
                stage_id="estimate_diagonal_q",
                request_fingerprint=self.request_fingerprint,
                project_fingerprint=self.project_fingerprint,
                stage_input_fingerprint=self.stage_input_fingerprint,
                implementation_provenance=self.implementation_provenance,
                bag_inputs=(
                    DiagonalQArtifactBagInput(
                        bag_id="bag-a",
                        source_path="/flight-data/bag-a.bag",
                        source_sha256="a" * 64,
                        source_size_bytes=12,
                        selected_interval_local_seconds=(0.0, 1.0),
                        effective_interval_local_seconds=(0.0, 1.0),
                        episode_index=0,
                        configuration_fingerprint="complete:" + "a" * 64,
                        fixed_model_provenance={
                            "fixed_model": "maximum-test"
                        },
                        constant_delay_seconds=0.0,
                        translation_covariance=np.eye(3),
                        rotation_covariance=2.0 * np.eye(3),
                        fixed_r_provenance={"method": "fixed"},
                    ),
                ),
                result=result,
                expectations=result.final_expectations,
            )
            loaded = load_diagonal_q_artifact(root)
            np.testing.assert_array_equal(
                loaded.covariance.stationary_variance,
                result.covariance.stationary_variance,
            )
            np.testing.assert_array_equal(
                loaded.expectations[0].smoothed_wrench,
                result.final_expectations[0].smoothed_wrench,
            )


if __name__ == "__main__":
    unittest.main()
