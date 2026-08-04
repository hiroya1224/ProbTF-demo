import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactValidationError,
    IncompleteArtifactError,
    UnsupportedArtifactSchema,
)
from grape_param_estim.batch_artifact import (
    BATCH_ESTIMATION_RUN_SCHEMA,
    file_sha256,
    load_batch_estimation_run,
    write_batch_estimation_run,
)


class BatchArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "run"
        self.root.mkdir()
        self.bag_ids = ("bag-a", "bag-b")
        self._write_core_run(mcmc=False)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _save(path, arrays):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **arrays)

    def _descriptor(self, relative):
        path = self.root / relative
        return {"path": relative, "sha256": file_sha256(path)}

    def _write_manifest(self, manifest):
        with (self.root / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, sort_keys=True)

    def _read_manifest(self):
        with (self.root / "manifest.json").open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def _refresh_descriptor(self, manifest, name, bag_id=None):
        if bag_id is None:
            relative = manifest["artifacts"][name]["path"]
            manifest["artifacts"][name]["sha256"] = file_sha256(
                self.root / relative
            )
        else:
            relative = manifest["artifacts"][name][bag_id]["path"]
            manifest["artifacts"][name][bag_id]["sha256"] = file_sha256(
                self.root / relative
            )

    def _map_static(self):
        return {
            "parameter_coordinate_map": np.zeros(18),
            "mass": np.asarray((2.4,)),
            "inertia": np.diag((0.2, 0.25, 0.3)),
            "cog": np.zeros(3),
            "force_effectiveness": np.ones(4),
            "torque_effectiveness": np.ones(4),
            "delay": np.asarray((0.006,)),
            "q_diagonal": 1.5 * np.arange(1.0, 7.0),
            "objective_component_names": np.asarray(
                ("observation", "dynamics", "prior")
            ),
            "objective_component_values": np.asarray((10.0, 2.0, 0.5)),
            "prior_objective": np.asarray((0.5,)),
            "likelihood_objective": np.asarray((12.0,)),
            "bag_id": np.asarray(self.bag_ids),
            "bag_objective": np.asarray((5.0, 7.0)),
        }

    @staticmethod
    def _q_em():
        q = np.arange(1.0, 7.0)[None, :]
        return {
            "iteration": np.asarray((0,), dtype=np.int64),
            "input_q": 2.0 * q,
            "target_q": q,
            "accepted_q": 1.5 * q,
            "alpha": np.asarray((0.5,)),
            "log_q_change": np.asarray((0.1,)),
            "map_objective": np.asarray((12.0,)),
            "approximate_marginal_objective": np.asarray((13.0,)),
            "lag": np.asarray((0.006,)),
            "accepted": np.asarray((True,), dtype=bool),
            "reason": np.asarray(("marginal_objective_improved",)),
            "floor_activation": np.zeros((1, 6), dtype=bool),
            "expected_residual_second_moment": 2.0 * q,
            "map_residual_second_moment": q,
            "covariance_correction": q,
        }

    @staticmethod
    def _laplace():
        dimension = 18
        return {
            "reduced_likelihood_hessian": np.eye(dimension),
            "reduced_posterior_hessian": 2.0 * np.eye(dimension),
            "covariance": 0.5 * np.eye(dimension),
            "eigenvalues": 2.0 * np.ones(dimension),
            "eigenvectors": np.eye(dimension),
            "effective_rank": np.asarray((dimension,), dtype=np.int64),
            "exact_ridge_direction": np.eye(dimension)[0],
            "ridge_alignment": np.asarray((1.0,)),
            "condition_number": np.asarray((2.0,)),
            "delay_profile_grid": np.asarray((0.0, 0.005, 0.01)),
            "delay_profile_objective": np.asarray((2.0, 1.0, 3.0)),
            "delay_local_uncertainty": np.asarray((0.001,)),
        }

    def _diagnostics(self, mcmc=False):
        count = len(self.bag_ids)
        result = {
            "bag_id": np.asarray(self.bag_ids),
            "knot_count": np.asarray((10, 11), dtype=np.int64),
            "factor_count": np.asarray((40, 44), dtype=np.int64),
            "residual_dimension": np.asarray((120, 132), dtype=np.int64),
            "jacobian_nnz": np.asarray((900, 990), dtype=np.int64),
            "assembly_seconds": np.full(count, 0.01),
            "factorization_seconds": np.full(count, 0.02),
            "schur_solve_seconds": np.full(count, 0.03),
            "nonlinear_iteration_seconds": np.asarray((0.1, 0.09)),
            "em_iteration_seconds": np.asarray((0.4,)),
            "mcmc_target_seconds": (
                np.asarray((0.3, 0.31)) if mcmc else np.asarray(())
            ),
            "peak_memory_bytes": np.asarray((123456,), dtype=np.int64),
        }
        if mcmc:
            result.update(
                {
                    "mcmc_chain_id": np.asarray(("chain-0", "chain-1")),
                    "mcmc_mode_id": np.asarray(("nominal",)),
                    "mcmc_draws_per_chain": np.asarray((4,), dtype=np.int64),
                    "mcmc_split_rhat": np.ones(19),
                    "mcmc_effective_sample_size": np.full(19, 8.0),
                    "mcmc_integrated_autocorrelation_time": np.ones(19),
                    "mcmc_ridge_coordinate_trace": np.zeros((2, 4)),
                    "mcmc_delay_trace": np.full((2, 4), 0.006),
                    "mcmc_log_posterior_trace": np.zeros((2, 4)),
                    "mcmc_kernel_names": np.asarray(("ridge", "delay")),
                    "mcmc_kernel_attempts": np.asarray((8, 8), dtype=np.int64),
                    "mcmc_kernel_stage_one_accepted": np.asarray(
                        (6, 6), dtype=np.int64
                    ),
                    "mcmc_kernel_stage_two_attempted": np.asarray(
                        (6, 6), dtype=np.int64
                    ),
                    "mcmc_kernel_stage_two_accepted": np.asarray(
                        (4, 4), dtype=np.int64
                    ),
                    "mcmc_kernel_full_target_cache_hits": np.asarray(
                        (1, 1), dtype=np.int64
                    ),
                    "mcmc_kernel_inner_solve_failures": np.asarray(
                        (0, 0), dtype=np.int64
                    ),
                    "mcmc_kernel_inner_iterations": np.asarray(
                        (10, 10), dtype=np.int64
                    ),
                    "mcmc_completed": np.asarray((True,), dtype=bool),
                    "mcmc_converged": np.asarray((True,), dtype=bool),
                    "mcmc_rhat_threshold": np.asarray((1.01,)),
                    "mcmc_minimum_effective_sample_size": np.asarray((4.0,)),
                }
            )
        return result

    @staticmethod
    def _empty_stream(prefix, value_name, dimension, covariance_dimension):
        result = {
            "{}_time".format(prefix): np.asarray(()),
            "{}_record_time".format(prefix): np.asarray(()),
            value_name: np.empty((0, dimension)),
            "{}_valid".format(prefix): np.empty((0,), dtype=bool),
        }
        if covariance_dimension is not None:
            result["{}_covariance".format(prefix)] = np.empty(
                (0, covariance_dimension, covariance_dimension)
            )
            result["{}_covariance_valid".format(prefix)] = np.empty(
                (0,), dtype=bool
            )
        return result

    def _bag(self, bag_id):
        count = 4
        time = np.asarray((0.0, 0.1, 0.2, 0.3))
        quaternion = np.zeros((count, 4))
        quaternion[:, 3] = 1.0
        result = {
            "bag_id": np.asarray((bag_id,)),
            "knot_time": time,
            "knot_record_time": 100.0 + time,
            "reference_time": time,
            "reference_record_time": 100.0 + time,
            "reference_position": np.zeros((count, 3)),
            "reference_linear_velocity": np.zeros((count, 3)),
            "reference_linear_acceleration": np.zeros((count, 3)),
            "reference_rpy": np.zeros((count, 3)),
            "reference_angular_velocity": np.zeros((count, 3)),
            "reference_angular_acceleration": np.zeros((count, 3)),
            "nominal_position": np.zeros((count, 3)),
            "nominal_orientation_xyzw": quaternion,
            "nominal_linear_velocity": np.zeros((count, 3)),
            "nominal_angular_velocity": np.zeros((count, 3)),
            "nominal_controller_integral": np.zeros((count, 6)),
            "nominal_actuator_thrust": np.ones((count, 4)),
            "nominal_actuator_gimbal": np.zeros((count, 4)),
            "map_position": np.zeros((count, 3)),
            "map_orientation_xyzw": quaternion,
            "map_linear_velocity": np.zeros((count, 3)),
            "map_angular_velocity": np.zeros((count, 3)),
            "map_controller_integral": np.zeros((count, 6)),
            "map_actuator_thrust": np.ones((count, 4)),
            "map_actuator_gimbal": np.zeros((count, 4)),
            "map_dynamics_residual": np.zeros((count - 1, 6)),
            "map_dynamics_residual_valid": np.ones(count - 1, dtype=bool),
            "correction_translation": np.zeros((count, 3)),
            "correction_rotation_vector": np.zeros((count, 3)),
            "factor_names": np.asarray(("pose", "dynamics")),
            "factor_residual_history": np.zeros((2, 2)),
            "factor_normalized_residual_history": np.zeros((2, 2)),
            "objective_component_names": np.asarray(("pose", "dynamics")),
            "objective_component_values": np.asarray((1.0, 2.0)),
            "numerical_diagnostic_names": np.asarray(("condition",)),
            "numerical_diagnostic_values": np.asarray((10.0,)),
        }
        for prefix, value_name, dimension, covariance_dimension in (
            ("pose", "pose_position", 3, 6),
            ("velocity", "velocity", 3, 3),
            ("gyro", "gyro", 3, 3),
            ("accelerometer", "accelerometer", 3, 3),
            ("gimbal_observation", "gimbal_observation", 4, 4),
        ):
            result.update(
                self._empty_stream(
                    prefix, value_name, dimension, covariance_dimension
                )
            )
        result["pose_orientation_xyzw"] = np.empty((0, 4))
        for prefix in ("thrust_command", "gimbal_command"):
            result.update(self._empty_stream(prefix, prefix, 4, None))
            result["{}_covariance".format(prefix)] = np.empty((0, 4, 4))
            result["{}_covariance_valid".format(prefix)] = np.empty(
                (0,), dtype=bool
            )
        result.update(
            self._empty_stream(
                "controller_integral",
                "controller_integral_observation",
                6,
                6,
            )
        )
        return result

    @staticmethod
    def _mcmc():
        count = 3
        inertia = np.repeat(np.eye(3)[None, :, :], count, axis=0)
        return {
            "sample_id": np.asarray((101, 107, 109), dtype=np.int64),
            "chain_id": np.asarray(("chain-0", "chain-0", "chain-1")),
            "draw_index": np.asarray((0, 1, 0), dtype=np.int64),
            "parameter_coordinate": np.zeros((count, 18)),
            "mass": np.full(count, 2.4),
            "inertia": inertia,
            "cog": np.zeros((count, 3)),
            "force_effectiveness": np.ones((count, 4)),
            "torque_effectiveness": np.ones((count, 4)),
            "delay": np.full(count, 0.006),
            "log_posterior": np.asarray((-10.0, -9.0, -11.0)),
            "log_likelihood_approximation": np.asarray((-8.0, -7.0, -9.0)),
            "log_determinant_term": np.asarray((1.0, 1.1, 0.9)),
            "accepted_kernel": np.asarray(("ridge", "delay", "ridge")),
            "source_mode_id": np.asarray(("nominal",) * count),
        }

    @staticmethod
    def _trajectory(sample_ids=(101, 109)):
        sample_ids = np.asarray(sample_ids, dtype=np.int64)
        sample_count = sample_ids.size
        knot_count = 4
        quaternion = np.zeros((sample_count, knot_count, 4))
        quaternion[:, :, 3] = 1.0
        return {
            "sample_id": sample_ids,
            "knot_time": np.asarray((0.0, 0.1, 0.2, 0.3)),
            "conditional_position": np.zeros((sample_count, knot_count, 3)),
            "conditional_orientation_xyzw": quaternion,
            "conditional_linear_velocity": np.zeros(
                (sample_count, knot_count, 3)
            ),
            "conditional_angular_velocity": np.zeros(
                (sample_count, knot_count, 3)
            ),
            "conditional_controller_integral": np.zeros(
                (sample_count, knot_count, 6)
            ),
            "conditional_actuator_thrust": np.ones(
                (sample_count, knot_count, 4)
            ),
            "conditional_actuator_gimbal": np.zeros(
                (sample_count, knot_count, 4)
            ),
            "correction_translation": np.zeros((sample_count, knot_count, 3)),
            "correction_rotation_vector": np.zeros(
                (sample_count, knot_count, 3)
            ),
            "dynamics_residual": np.zeros((sample_count, knot_count - 1, 6)),
            "dynamics_residual_valid": np.ones(
                (sample_count, knot_count - 1), dtype=bool
            ),
            "conditional_objective": np.arange(sample_count, dtype=float),
        }

    def _write_core_run(self, mcmc):
        self._save(self.root / "map_static.npz", self._map_static())
        self._save(self.root / "q_em.npz", self._q_em())
        self._save(self.root / "laplace.npz", self._laplace())
        self._save(self.root / "diagnostics.npz", self._diagnostics(mcmc=mcmc))
        for bag_id in self.bag_ids:
            self._save(self.root / "bags" / (bag_id + ".npz"), self._bag(bag_id))

        artifacts = {
            "map_static": self._descriptor("map_static.npz"),
            "q_em": self._descriptor("q_em.npz"),
            "laplace": self._descriptor("laplace.npz"),
            "diagnostics": self._descriptor("diagnostics.npz"),
            "bags": {
                bag_id: self._descriptor("bags/{}.npz".format(bag_id))
                for bag_id in self.bag_ids
            },
        }
        substage_status = {
            "map": {"converged": True, "termination_reason": "gradient_tolerance"},
            "laplace_em": {
                "converged": True,
                "termination_reason": "log_q_tolerance",
            },
            "laplace": {"converged": True, "termination_reason": "completed"},
        }
        if mcmc:
            self._save(self.root / "mcmc_samples.npz", self._mcmc())
            self._save(
                self.root
                / "trajectories"
                / self.bag_ids[0]
                / "selected_samples.npz",
                self._trajectory(),
            )
            artifacts["mcmc_samples"] = self._descriptor("mcmc_samples.npz")
            artifacts["trajectories"] = {
                self.bag_ids[0]: self._descriptor(
                    "trajectories/{}/selected_samples.npz".format(
                        self.bag_ids[0]
                    )
                )
            }
            substage_status["mcmc"] = {
                "converged": True,
                "termination_reason": "diagnostics_satisfied",
            }

        digest = "sha256:" + "a" * 64
        manifest = {
            "schema": BATCH_ESTIMATION_RUN_SCHEMA,
            "status": "complete",
            "run_id": "run-a",
            "estimator_revision": "test-revision",
            "selected_bag_ids": list(self.bag_ids),
            "selected_intervals": {
                "bag-a": [18.0, 24.0],
                "bag-b": [20.0, 25.0],
            },
            "selected_bag_sha256": {
                bag_id: digest for bag_id in self.bag_ids
            },
            "configuration_fingerprint": digest,
            "controller_snapshot_fingerprint": digest,
            "sensor_contracts": {
                bag_id: {"pose": {"topic": "/mocap/pose"}}
                for bag_id in self.bag_ids
            },
            "observation_factors": {
                bag_id: {
                    "pose": {"enabled": True, "disabled_reason": None},
                    "accelerometer": {
                        "enabled": False,
                        "disabled_reason": "sensor origin is not calibrated",
                    },
                }
                for bag_id in self.bag_ids
            },
            "parameter_prior": {"kind": "gaussian", "dimension": 18},
            "delay_prior": {"kind": "uniform", "bounds_seconds": [0.0, 0.02]},
            "q_definition": {
                "definition": "explicit synthetic six-axis discrepancy spectrum",
                "components": ["x", "y", "z", "roll", "pitch", "yaw"],
                "units": ["explicit-unit"] * 6,
            },
            "knot_policy": {"kind": "event_union"},
            "interpolation_policy": {"pose": "SO3_geodesic"},
            "solver_settings": {"kind": "sparse_lm"},
            "em_settings": {"maximum_iterations": 8},
            "mcmc_settings": {"enabled": mcmc},
            "request_fingerprint": digest,
            "substage_status": substage_status,
            "warnings": [],
            "artifacts": artifacts,
        }
        self._write_manifest(manifest)

    def _manifest_metadata(self):
        manifest = self._read_manifest()
        return {
            key: value
            for key, value in manifest.items()
            if key not in {"schema", "status", "artifacts"}
        }

    def test_loads_complete_pickle_free_run(self):
        run = load_batch_estimation_run(self.root)

        self.assertEqual(tuple(run.bags), self.bag_ids)
        self.assertEqual(run.map_static["parameter_coordinate_map"].shape, (18,))
        self.assertIsNone(run.mcmc_samples)
        self.assertFalse(run.map_static["mass"].flags.writeable)

    def test_unknown_and_old_schemas_are_rejected(self):
        for schema in (
            "grape-param-estim/assimilation-run/v1",
            "grape-param-estim/batch-estimation-run/v2",
        ):
            manifest = self._read_manifest()
            manifest["schema"] = schema
            self._write_manifest(manifest)
            with self.assertRaises(UnsupportedArtifactSchema):
                load_batch_estimation_run(self.root)

    def test_incomplete_status_is_rejected(self):
        manifest = self._read_manifest()
        manifest["status"] = "cancelled"
        self._write_manifest(manifest)

        with self.assertRaises(IncompleteArtifactError):
            load_batch_estimation_run(self.root)

    def test_q_definition_and_units_have_no_defaults(self):
        manifest = self._read_manifest()
        del manifest["q_definition"]["units"]
        self._write_manifest(manifest)

        with self.assertRaisesRegex(ArtifactValidationError, "units"):
            load_batch_estimation_run(self.root)

    def test_artifact_paths_are_canonical_and_safe(self):
        manifest = self._read_manifest()
        manifest["artifacts"]["map_static"]["path"] = "../map_static.npz"
        self._write_manifest(manifest)

        with self.assertRaisesRegex(ArtifactValidationError, "map_static.npz"):
            load_batch_estimation_run(self.root)

    def test_file_hash_mismatch_is_rejected(self):
        manifest = self._read_manifest()
        manifest["artifacts"]["laplace"]["sha256"] = (
            "sha256:" + "b" * 64
        )
        self._write_manifest(manifest)

        with self.assertRaisesRegex(ArtifactValidationError, "SHA-256 mismatch"):
            load_batch_estimation_run(self.root)

    def test_object_dtype_is_rejected_without_pickle(self):
        self._save(
            self.root / "q_em.npz",
            {"iteration": np.asarray([{"unsafe": True}], dtype=object)},
        )
        manifest = self._read_manifest()
        self._refresh_descriptor(manifest, "q_em")
        self._write_manifest(manifest)

        with self.assertRaisesRegex(ArtifactValidationError, "without pickle"):
            load_batch_estimation_run(self.root)

    def test_map_static_core_shape_is_checked(self):
        arrays = self._map_static()
        arrays["parameter_coordinate_map"] = np.zeros(19)
        self._save(self.root / "map_static.npz", arrays)
        manifest = self._read_manifest()
        self._refresh_descriptor(manifest, "map_static")
        self._write_manifest(manifest)

        with self.assertRaisesRegex(ArtifactValidationError, "expected \(18,\)"):
            load_batch_estimation_run(self.root)

    def test_bag_core_shape_and_bag_identity_are_checked(self):
        arrays = self._bag("wrong-bag")
        self._save(self.root / "bags" / "bag-a.npz", arrays)
        manifest = self._read_manifest()
        self._refresh_descriptor(manifest, "bags", "bag-a")
        self._write_manifest(manifest)

        with self.assertRaisesRegex(ArtifactValidationError, "does not match"):
            load_batch_estimation_run(self.root)

    def test_disabled_factor_requires_an_explicit_reason(self):
        manifest = self._read_manifest()
        manifest["observation_factors"]["bag-a"]["accelerometer"][
            "disabled_reason"
        ] = ""
        self._write_manifest(manifest)

        with self.assertRaisesRegex(ArtifactValidationError, "disabled_reason"):
            load_batch_estimation_run(self.root)

    def test_mcmc_and_selected_trajectory_ids_are_aligned(self):
        self._write_core_run(mcmc=True)
        run = load_batch_estimation_run(self.root)

        self.assertTrue(
            np.array_equal(run.mcmc_samples["sample_id"], (101, 107, 109))
        )
        self.assertTrue(
            np.array_equal(
                run.trajectories["bag-a"]["sample_id"], (101, 109)
            )
        )

        self._save(
            self.root / "trajectories" / "bag-a" / "selected_samples.npz",
            self._trajectory(sample_ids=(101, 999)),
        )
        manifest = self._read_manifest()
        descriptor = manifest["artifacts"]["trajectories"]["bag-a"]
        descriptor["sha256"] = file_sha256(self.root / descriptor["path"])
        self._write_manifest(manifest)
        with self.assertRaisesRegex(ArtifactValidationError, "999"):
            load_batch_estimation_run(self.root)

    def test_mcmc_particle_weights_are_forbidden(self):
        self._write_core_run(mcmc=True)
        arrays = self._mcmc()
        arrays["weight"] = np.full(3, 1.0 / 3.0)
        self._save(self.root / "mcmc_samples.npz", arrays)
        manifest = self._read_manifest()
        self._refresh_descriptor(manifest, "mcmc_samples")
        self._write_manifest(manifest)

        with self.assertRaisesRegex(ArtifactValidationError, "particle fields"):
            load_batch_estimation_run(self.root)

    def test_writer_atomically_publishes_round_trip_with_sha_descriptors(self):
        destination = Path(self.temporary.name) / "written-run"
        run = write_batch_estimation_run(
            destination,
            manifest_metadata=self._manifest_metadata(),
            map_static=self._map_static(),
            q_em=self._q_em(),
            laplace=self._laplace(),
            diagnostics=self._diagnostics(),
            bags={bag_id: self._bag(bag_id) for bag_id in self.bag_ids},
        )

        self.assertEqual(run.manifest["status"], "complete")
        self.assertEqual(tuple(run.bags), self.bag_ids)
        self.assertFalse(run.map_static["mass"].flags.writeable)
        for descriptor in (
            run.manifest["artifacts"]["map_static"],
            run.manifest["artifacts"]["q_em"],
            run.manifest["artifacts"]["laplace"],
            run.manifest["artifacts"]["diagnostics"],
        ):
            self.assertEqual(
                descriptor["sha256"],
                file_sha256(destination / descriptor["path"]),
            )
        loaded = load_batch_estimation_run(destination)
        self.assertEqual(loaded.manifest, run.manifest)

    def test_writer_round_trips_optional_mcmc_and_trajectory_subset(self):
        self._write_core_run(mcmc=True)
        destination = Path(self.temporary.name) / "written-mcmc-run"
        run = write_batch_estimation_run(
            destination,
            manifest_metadata=self._manifest_metadata(),
            map_static=self._map_static(),
            q_em=self._q_em(),
            laplace=self._laplace(),
            diagnostics=self._diagnostics(mcmc=True),
            bags={bag_id: self._bag(bag_id) for bag_id in self.bag_ids},
            mcmc_samples=self._mcmc(),
            trajectories={self.bag_ids[0]: self._trajectory()},
        )

        self.assertTrue(
            np.array_equal(run.mcmc_samples["sample_id"], (101, 107, 109))
        )
        self.assertEqual(tuple(run.trajectories), ("bag-a",))
        self.assertIn("mcmc_samples", run.manifest["artifacts"])
        self.assertIn("trajectories", run.manifest["artifacts"])

    def test_writer_failure_leaves_authoritative_incomplete_status(self):
        destination = Path(self.temporary.name) / "invalid-run"
        invalid = self._q_em()
        invalid["stale_old_field"] = np.asarray((1.0,))

        with self.assertRaisesRegex(ArtifactValidationError, "unknown keys"):
            write_batch_estimation_run(
                destination,
                manifest_metadata=self._manifest_metadata(),
                map_static=self._map_static(),
                q_em=invalid,
                laplace=self._laplace(),
                diagnostics=self._diagnostics(),
                bags={
                    bag_id: self._bag(bag_id) for bag_id in self.bag_ids
                },
            )

        with (destination / "manifest.json").open(
            "r", encoding="utf-8"
        ) as stream:
            self.assertEqual(json.load(stream)["status"], "writing")
        with self.assertRaises(IncompleteArtifactError):
            load_batch_estimation_run(destination)

    def test_writer_rejects_object_dtype_without_creating_a_run(self):
        destination = Path(self.temporary.name) / "object-run"
        invalid = self._map_static()
        invalid["mass"] = np.asarray(({"unsafe": True},), dtype=object)

        with self.assertRaisesRegex(ArtifactValidationError, "object dtype"):
            write_batch_estimation_run(
                destination,
                manifest_metadata=self._manifest_metadata(),
                map_static=invalid,
                q_em=self._q_em(),
                laplace=self._laplace(),
                diagnostics=self._diagnostics(),
                bags={
                    bag_id: self._bag(bag_id) for bag_id in self.bag_ids
                },
            )
        self.assertFalse(destination.exists())

    def test_exact_v1_rejects_unknown_manifest_and_array_fields(self):
        manifest = self._read_manifest()
        manifest["legacy_stage"] = {"schema": "old"}
        self._write_manifest(manifest)
        with self.assertRaisesRegex(ArtifactValidationError, "unknown keys"):
            load_batch_estimation_run(self.root)

        self._write_core_run(mcmc=False)
        arrays = self._map_static()
        arrays["member_id"] = np.asarray((1,), dtype=np.int64)
        self._save(self.root / "map_static.npz", arrays)
        manifest = self._read_manifest()
        self._refresh_descriptor(manifest, "map_static")
        self._write_manifest(manifest)
        with self.assertRaisesRegex(ArtifactValidationError, "member_id"):
            load_batch_estimation_run(self.root)


if __name__ == "__main__":
    unittest.main()
