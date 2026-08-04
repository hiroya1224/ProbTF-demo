from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from grape_param_estim_gui import artifact_loader
from grape_param_estim_gui.presentation import (
    map_parameter_text,
    sample_parameter_text,
)
from grape_param_estim_gui.project_io import new_project_manifest
from grape_param_estim_gui.state import BagRecord, ProjectStore


def _empty_observation(
    prefix: str, value_name: str, dimension: int, covariance_dimension: int
) -> dict[str, np.ndarray]:
    return {
        "{}_time".format(prefix): np.asarray((), dtype=float),
        "{}_record_time".format(prefix): np.asarray((), dtype=float),
        value_name: np.empty((0, dimension), dtype=float),
        "{}_valid".format(prefix): np.empty((0,), dtype=bool),
        "{}_covariance".format(prefix): np.empty(
            (0, covariance_dimension, covariance_dimension), dtype=float
        ),
        "{}_covariance_valid".format(prefix): np.empty((0,), dtype=bool),
    }


def _bag_arrays() -> dict[str, np.ndarray]:
    count = 3
    time = np.asarray((0.0, 0.1, 0.2))
    quaternion = np.zeros((count, 4))
    quaternion[:, 3] = 1.0
    arrays: dict[str, np.ndarray] = {
        "bag_id": np.asarray(("bag-a",)),
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
        "map_position": np.ones((count, 3)),
        "map_orientation_xyzw": quaternion,
        "map_linear_velocity": np.zeros((count, 3)),
        "map_angular_velocity": np.zeros((count, 3)),
        "map_controller_integral": np.zeros((count, 6)),
        "map_actuator_thrust": np.ones((count, 4)),
        "map_actuator_gimbal": np.zeros((count, 4)),
        "map_dynamics_residual": np.zeros((count - 1, 6)),
        "map_dynamics_residual_valid": np.ones((count - 1,), dtype=bool),
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
        ("thrust_command", "thrust_command", 4, 4),
        ("gimbal_command", "gimbal_command", 4, 4),
        ("gimbal_observation", "gimbal_observation", 4, 4),
        (
            "controller_integral",
            "controller_integral_observation",
            6,
            6,
        ),
    ):
        arrays.update(
            _empty_observation(
                prefix, value_name, dimension, covariance_dimension
            )
        )
    arrays["pose_orientation_xyzw"] = np.empty((0, 4))
    return arrays


def _map_arrays() -> dict[str, np.ndarray]:
    return {
        "parameter_coordinate_map": np.zeros(18),
        "mass": np.asarray((2.4,)),
        "inertia": np.diag((0.2, 0.25, 0.3)),
        "cog": np.zeros(3),
        "force_effectiveness": np.ones(4),
        "torque_effectiveness": np.ones(4),
        "delay": np.asarray((0.006,)),
        "q_diagonal": np.arange(1.0, 7.0),
        "objective_component_names": np.asarray(("pose", "dynamics")),
        "objective_component_values": np.asarray((10.0, 2.0)),
        "prior_objective": np.asarray((0.5,)),
        "likelihood_objective": np.asarray((12.0,)),
        "bag_id": np.asarray(("bag-a",)),
        "bag_objective": np.asarray((12.0,)),
    }


def _q_arrays() -> dict[str, np.ndarray]:
    q = np.arange(1.0, 7.0)[None, :]
    return {
        "iteration": np.asarray((0,), dtype=np.int64),
        "input_q": q,
        "target_q": q,
        "accepted_q": q,
        "alpha": np.asarray((1.0,)),
        "log_q_change": np.asarray((0.0,)),
        "map_objective": np.asarray((12.0,)),
        "approximate_marginal_objective": np.asarray((13.0,)),
        "lag": np.asarray((0.006,)),
        "accepted": np.asarray((True,), dtype=bool),
        "reason": np.asarray(("converged",)),
        "floor_activation": np.zeros((1, 6), dtype=bool),
        "expected_residual_second_moment": 2.0 * q,
        "map_residual_second_moment": q,
        "covariance_correction": q,
    }


def _laplace_arrays(valid_geometry: bool = True) -> dict[str, np.ndarray]:
    delay_grid = np.asarray((0.0, 0.006, 0.012))
    curvature = 1.0e6
    sensitivity = np.zeros(18)
    sensitivity[0] = 2.0
    coordinate = np.outer(delay_grid - 0.006, sensitivity)
    objective = 0.5 * curvature * (delay_grid - 0.006) ** 2
    conditional_covariance = 0.5 * np.eye(18)
    joint_covariance = np.zeros((19, 19))
    joint_covariance[:18, :18] = conditional_covariance + np.outer(
        sensitivity, sensitivity
    ) / curvature
    joint_covariance[:18, -1] = sensitivity / curvature
    joint_covariance[-1, :18] = sensitivity / curvature
    joint_covariance[-1, -1] = 1.0 / curvature
    arrays = {
        "reduced_likelihood_hessian": np.eye(18),
        "reduced_posterior_hessian": 2.0 * np.eye(18),
        "covariance": conditional_covariance,
        "static_covariance_conditioning": np.asarray(
            ("fixed_delay_conditional",)
        ),
        "eigenvalues": 2.0 * np.ones(18),
        "eigenvectors": np.eye(18),
        "effective_rank": np.asarray((18,), dtype=np.int64),
        "exact_ridge_direction": np.eye(18)[0],
        "ridge_alignment": np.asarray((1.0,)),
        "condition_number": np.asarray((2.0,)),
        "delay_profile_available": np.asarray((True,), dtype=bool),
        "delay_profile_grid": delay_grid,
        "delay_profile_objective": objective,
        "delay_profile_approximate_marginal_objective": objective + 0.5,
        "delay_profile_static_coordinate": coordinate,
        "delay_local_uncertainty": np.asarray(
            (0.001 if valid_geometry else 0.003,)
        ),
        "delay_uncertainty_source": np.asarray(
            (
                "profile_curvature"
                if valid_geometry
                else "uniform_delay_prior_fallback",
            )
        ),
        "delay_profile_curvature": np.asarray(
            (curvature if valid_geometry else 0.0,)
        ),
        "delay_profile_curvature_valid": np.asarray(
            (valid_geometry,), dtype=bool
        ),
        "delay_local_geometry_valid": np.asarray(
            (valid_geometry,), dtype=bool
        ),
        "delay_local_geometry_method": np.asarray(
            ("nonuniform_three_point_map_profile_v1",)
        ),
        "delay_local_geometry_reason": np.asarray(
            ("valid" if valid_geometry else "map_delay_at_profile_boundary",)
        ),
        "delay_profile_support_lag": (
            delay_grid if valid_geometry else np.empty((0,), dtype=float)
        ),
        "delay_profile_support_map_objective": (
            objective if valid_geometry else np.empty((0,), dtype=float)
        ),
        "delay_profile_support_static_coordinate": (
            coordinate if valid_geometry else np.empty((0, 18), dtype=float)
        ),
        "delay_profile_gradient": (
            np.asarray((0.0,))
            if valid_geometry
            else np.empty((0,), dtype=float)
        ),
        "delay_static_sensitivity": (
            sensitivity if valid_geometry else np.empty((0,), dtype=float)
        ),
        "parameter_delay_cross_covariance": (
            sensitivity / curvature
            if valid_geometry
            else np.empty((0,), dtype=float)
        ),
        "joint_parameter_delay_information": (
            np.linalg.inv(joint_covariance)
            if valid_geometry
            else np.empty((0, 0), dtype=float)
        ),
        "joint_parameter_delay_covariance": (
            joint_covariance
            if valid_geometry
            else np.empty((0, 0), dtype=float)
        ),
        "mcmc_quadratic_surrogate_method": np.asarray(
            (
                "joint_profile_information_v1"
                if valid_geometry
                else "proposal_only_block_diagonal_fallback_v1",
            )
        ),
    }
    return arrays


def _mcmc_arrays() -> dict[str, np.ndarray]:
    count = 3
    return {
        "sample_id": np.asarray((101, 107, 109), dtype=np.int64),
        "chain_id": np.asarray(("chain-0", "chain-0", "chain-1")),
        "draw_index": np.asarray((0, 1, 0), dtype=np.int64),
        "parameter_coordinate": np.zeros((count, 18)),
        "mass": np.asarray((2.3, 2.4, 2.5)),
        "inertia": np.repeat(np.eye(3)[None, :, :], count, axis=0),
        "cog": np.zeros((count, 3)),
        "force_effectiveness": np.ones((count, 4)),
        "torque_effectiveness": np.ones((count, 4)),
        "delay": np.full((count,), 0.006),
        "log_posterior": np.asarray((-10.0, -9.0, -11.0)),
        "log_likelihood_approximation": np.asarray((-8.0, -7.0, -9.0)),
        "log_determinant_term": np.asarray((1.0, 1.1, 0.9)),
        "accepted_kernel": np.asarray(("ridge", "delay", "ridge")),
        "source_mode_id": np.asarray(("nominal",) * count),
    }


def _diagnostic_arrays(mcmc: bool) -> dict[str, np.ndarray]:
    arrays = {
        "bag_id": np.asarray(("bag-a",)),
        "knot_count": np.asarray((3,), dtype=np.int64),
        "factor_count": np.asarray((10,), dtype=np.int64),
        "residual_dimension": np.asarray((30,), dtype=np.int64),
        "jacobian_nnz": np.asarray((100,), dtype=np.int64),
        "assembly_seconds": np.asarray((0.01,)),
        "factorization_seconds": np.asarray((0.02,)),
        "schur_solve_seconds": np.asarray((0.03,)),
        "nonlinear_iteration_seconds": np.asarray((0.1,)),
        "em_iteration_seconds": np.asarray((0.2,)),
        "mcmc_target_seconds": np.asarray((0.3,)) if mcmc else np.asarray(()),
        "peak_memory_bytes": np.asarray((1234,), dtype=np.int64),
    }
    if mcmc:
        arrays.update(
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
                "mcmc_kernel_attempts": np.asarray((8, 8)),
                "mcmc_kernel_stage_one_accepted": np.asarray((6, 6)),
                "mcmc_kernel_stage_two_attempted": np.asarray((6, 6)),
                "mcmc_kernel_stage_two_accepted": np.asarray((4, 4)),
                "mcmc_kernel_full_target_cache_hits": np.asarray((1, 1)),
                "mcmc_kernel_inner_solve_failures": np.asarray((0, 0)),
                "mcmc_kernel_inner_iterations": np.asarray((10, 10)),
                "mcmc_completed": np.asarray((True,), dtype=bool),
                "mcmc_converged": np.asarray((True,), dtype=bool),
                "mcmc_rhat_threshold": np.asarray((1.01,)),
                "mcmc_minimum_effective_sample_size": np.asarray((4.0,)),
            }
        )
    return arrays


def _trajectory_arrays() -> dict[str, np.ndarray]:
    sample_ids = np.asarray((101, 109), dtype=np.int64)
    sample_count = sample_ids.size
    knot_count = 3
    quaternion = np.zeros((sample_count, knot_count, 4))
    quaternion[:, :, 3] = 1.0
    return {
        "sample_id": sample_ids,
        "knot_time": np.asarray((0.0, 0.1, 0.2)),
        "conditional_position": np.ones((sample_count, knot_count, 3)),
        "conditional_orientation_xyzw": quaternion,
        "conditional_linear_velocity": np.zeros((sample_count, knot_count, 3)),
        "conditional_angular_velocity": np.zeros((sample_count, knot_count, 3)),
        "conditional_controller_integral": np.zeros((sample_count, knot_count, 6)),
        "conditional_actuator_thrust": np.ones((sample_count, knot_count, 4)),
        "conditional_actuator_gimbal": np.zeros((sample_count, knot_count, 4)),
        "correction_translation": np.zeros((sample_count, knot_count, 3)),
        "correction_rotation_vector": np.zeros((sample_count, knot_count, 3)),
        "dynamics_residual": np.zeros((sample_count, knot_count - 1, 6)),
        "dynamics_residual_valid": np.ones(
            (sample_count, knot_count - 1), dtype=bool
        ),
        "conditional_objective": np.asarray((1.0, 2.0)),
    }


def _backend_bundle(
    root: Path, mcmc: bool, valid_geometry: bool = True
) -> SimpleNamespace:
    digest = "sha256:" + "a" * 64
    manifest = {
        "schema": "grape-param-estim/batch-estimation-run/v1",
        "status": "complete",
        "run_id": "run-a",
        "request_fingerprint": digest,
        "warnings": [],
        "selected_bag_ids": ["bag-a"],
        "mcmc_settings": {"delay_scale_seconds": 0.0004},
        "sensor_contracts": {"bag-a": {"pose": {"frame": "world"}}},
        "observation_factors": {
            "bag-a": {
                "pose": {"enabled": True, "disabled_reason": None}
            }
        },
    }
    return SimpleNamespace(
        root=root,
        manifest=manifest,
        map_static=_map_arrays(),
        q_em=_q_arrays(),
        laplace=_laplace_arrays(valid_geometry),
        diagnostics=_diagnostic_arrays(mcmc),
        bags={"bag-a": _bag_arrays()},
        mcmc_samples=_mcmc_arrays() if mcmc else None,
        trajectories={"bag-a": _trajectory_arrays()} if mcmc else {},
    )


class BatchEstimationLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "run"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _load(
        self, mcmc: bool, valid_geometry: bool = True
    ) -> artifact_loader.BatchEstimationRun:
        bundle = _backend_bundle(self.root, mcmc, valid_geometry)
        with mock.patch.object(
            artifact_loader.batch_artifact_io,
            "load_batch_estimation_run",
            return_value=bundle,
        ):
            return artifact_loader.load_batch_estimation_run(self.root)

    def test_map_laplace_q_and_bag_are_typed_without_mcmc(self) -> None:
        run = self._load(mcmc=False)

        self.assertEqual(run.static_map.mass, 2.4)
        self.assertEqual(run.laplace.effective_rank, 18)
        self.assertEqual(run.q_em.reason.tolist(), ["converged"])
        self.assertEqual(run.bags["bag-a"].map_trajectory.position.shape, (3, 3))
        self.assertIsNone(run.mcmc)
        self.assertEqual(run.sample_ids, ())
        self.assertEqual(run.selected_trajectories, {})
        self.assertIsNone(run.diagnostics.mcmc)

    def test_valid_joint_geometry_is_exposed_without_covariance_alias(self) -> None:
        run = self._load(mcmc=False, valid_geometry=True)
        laplace = run.laplace

        self.assertFalse(hasattr(laplace, "covariance"))
        self.assertTrue(laplace.delay_local_geometry_valid)
        self.assertEqual(laplace.delay_local_geometry_reason, "valid")
        self.assertTrue(
            np.allclose(laplace.reduced_prior_information, np.eye(18))
        )
        self.assertEqual(
            laplace.fixed_delay_conditional_static_covariance.shape,
            (18, 18),
        )
        self.assertEqual(laplace.joint_parameter_delay_covariance.shape, (19, 19))
        self.assertEqual(
            laplace.joint_static_parameter_marginal_covariance.shape,
            (18, 18),
        )
        self.assertEqual(laplace.parameter_delay_cross_covariance.shape, (18,))
        self.assertAlmostEqual(
            laplace.joint_delay_marginal_standard_deviation, 0.001
        )
        self.assertIsNone(laplace.fallback_delay_prior_standard_deviation)

    def test_invalid_geometry_cannot_be_misread_as_joint_marginal(self) -> None:
        run = self._load(mcmc=False, valid_geometry=False)
        laplace = run.laplace

        self.assertFalse(laplace.delay_local_geometry_valid)
        self.assertIsNone(laplace.joint_parameter_delay_information)
        self.assertIsNone(laplace.joint_parameter_delay_covariance)
        self.assertIsNone(laplace.joint_static_parameter_marginal_covariance)
        self.assertIsNone(laplace.parameter_delay_cross_covariance)
        self.assertIsNone(laplace.joint_delay_marginal_standard_deviation)
        self.assertAlmostEqual(
            laplace.fallback_delay_prior_standard_deviation, 0.003
        )
        self.assertAlmostEqual(
            laplace.mcmc_delay_proposal_scale_seconds, 0.0004
        )

    def test_mcmc_sample_and_selected_trajectory_share_string_identity(self) -> None:
        run = self._load(mcmc=True)

        self.assertEqual(run.sample_ids, ("101", "107", "109"))
        self.assertEqual(
            run.selected_trajectories["bag-a"].sample_id.tolist(),
            ["101", "109"],
        )
        selected = run.selected_trajectory("bag-a", "109")
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.sample_id, "109")
        self.assertEqual(selected.state.position.shape, (3, 3))
        self.assertIsNone(run.selected_trajectory("bag-a", "107"))
        self.assertTrue(np.allclose(run.mcmc.equal_weights, 1.0 / 3.0))
        self.assertTrue(run.diagnostics.mcmc.converged)

    def test_backend_rejections_are_exposed_as_gui_errors(self) -> None:
        backend = artifact_loader.batch_artifact_io
        errors = (
            backend.UnsupportedArtifactSchema("old schema"),
            backend.IncompleteArtifactError("status writing"),
            backend.ArtifactValidationError("object arrays require pickle"),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__), mock.patch.object(
                backend, "load_batch_estimation_run", side_effect=error
            ):
                with self.assertRaisesRegex(
                    artifact_loader.GuiArtifactError, str(error)
                ):
                    artifact_loader.load_batch_estimation_run(self.root)

    def test_derived_prior_information_must_remain_symmetric(self) -> None:
        bundle = _backend_bundle(self.root, mcmc=False)
        bundle.laplace["reduced_posterior_hessian"] = (
            bundle.laplace["reduced_posterior_hessian"].copy()
        )
        bundle.laplace["reduced_posterior_hessian"][0, 1] = 0.2
        with mock.patch.object(
            artifact_loader.batch_artifact_io,
            "load_batch_estimation_run",
            return_value=bundle,
        ):
            with self.assertRaisesRegex(
                artifact_loader.GuiArtifactError,
                "posterior information.*symmetric",
            ):
                artifact_loader.load_batch_estimation_run(self.root)

    def test_prior_psd_audit_tolerates_roundoff_but_rejects_real_negative_information(self) -> None:
        tiny = _backend_bundle(self.root, mcmc=False)
        tiny.laplace["reduced_posterior_hessian"] = (
            tiny.laplace["reduced_posterior_hessian"].copy()
        )
        tiny.laplace["reduced_posterior_hessian"][0, 0] = 1.0 - 1.0e-12
        with mock.patch.object(
            artifact_loader.batch_artifact_io,
            "load_batch_estimation_run",
            return_value=tiny,
        ):
            run = artifact_loader.load_batch_estimation_run(self.root)
        self.assertAlmostEqual(
            run.laplace.reduced_prior_information[0, 0], -1.0e-12
        )

        negative = _backend_bundle(self.root, mcmc=False)
        negative.laplace["reduced_posterior_hessian"] = (
            negative.laplace["reduced_posterior_hessian"].copy()
        )
        negative.laplace["reduced_posterior_hessian"][0, 0] = 0.9
        with mock.patch.object(
            artifact_loader.batch_artifact_io,
            "load_batch_estimation_run",
            return_value=negative,
        ):
            with self.assertRaisesRegex(
                artifact_loader.GuiArtifactError,
                "positive semidefinite",
            ):
                artifact_loader.load_batch_estimation_run(self.root)

    def test_presentation_uses_map_and_mcmc_sample_vocabulary(self) -> None:
        run = self._load(mcmc=True)

        self.assertIn("MAP | mass 2.4 kg", map_parameter_text(run.static_map))
        text = sample_parameter_text(run.mcmc, "107")
        self.assertIn("MCMC sample 107", text)
        self.assertIn("chain chain-0 draw 1", text)


class SampleStateSynchronizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        manifest = new_project_manifest("sample-sync")
        manifest["bags"] = [
            {
                "bag_id": "bag-a",
                "sha256": "a" * 64,
            }
        ]
        self.store = ProjectStore(
            self.root / "project", manifest
        )
        self.store.add(
            BagRecord(
                bag_id="bag-a",
                path=self.root / "bag-a.bag",
                source_path=self.root / "bag-a.bag",
                sha256="a" * 64,
                included=True,
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(
        self, mcmc: bool, stored_first: str | None = None
    ) -> artifact_loader.BatchEstimationRun:
        bundle = _backend_bundle(self.root / "run", mcmc)
        if stored_first is not None:
            subset = bundle.trajectories["bag-a"]
            sample_ids = subset["sample_id"].tolist()
            index = sample_ids.index(int(stored_first))
            order = np.roll(np.arange(len(sample_ids)), -index)
            for name, value in tuple(subset.items()):
                if name != "knot_time" and value.shape[0] == len(sample_ids):
                    subset[name] = value[order]
        with mock.patch.object(
            artifact_loader.batch_artifact_io,
            "load_batch_estimation_run",
            return_value=bundle,
        ):
            run = artifact_loader.load_batch_estimation_run(bundle.root)
        run.manifest["request_fingerprint"] = self.store.request_fingerprint()
        return run

    def test_selection_is_a_shared_string_sample_id(self) -> None:
        observed: list[object] = []
        self.store.selectedSampleChanged.connect(observed.append)

        self.store.apply_estimation(self._run(mcmc=True))
        self.assertEqual(self.store.selected_sample_id, "101")
        self.assertEqual(self.store.current_record().result.bag_id, "bag-a")

        self.store.set_selected_sample("109")
        self.assertEqual(self.store.selected_sample_id, "109")
        self.store.set_selected_sample("missing")
        self.assertEqual(self.store.selected_sample_id, "109")
        self.assertEqual(observed, ["101", "109"])
        self.assertEqual(self.store.snapshot().selected_sample_id, "109")

    def test_default_prefers_stored_trajectory_but_nonstored_remains_selectable(self) -> None:
        self.store.apply_estimation(
            self._run(mcmc=True, stored_first="109")
        )

        self.assertEqual(self.store.selected_sample_id, "109")
        self.assertIsNotNone(
            self.store.estimation_run.selected_trajectory("bag-a", "109")
        )
        self.store.set_selected_sample("107")
        self.assertEqual(self.store.selected_sample_id, "107")
        self.assertIsNone(
            self.store.estimation_run.selected_trajectory("bag-a", "107")
        )

    def test_map_only_run_has_no_fake_sample_selection(self) -> None:
        self.store.apply_estimation(self._run(mcmc=False))

        self.assertIsNone(self.store.posterior_samples)
        self.assertIsNone(self.store.selected_sample_id)
        self.assertIsNone(self.store.snapshot().selected_sample_id)


try:
    from PySide6.QtWidgets import QApplication
except ImportError:
    QApplication = None


@unittest.skipIf(QApplication is None, "PySide6 is unavailable")
class BatchResultViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert QApplication is not None
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        manifest = new_project_manifest("batch-view")
        manifest["bags"] = [
            {"bag_id": "bag-a", "sha256": "a" * 64}
        ]
        self.store = ProjectStore(self.root / "project", manifest)
        self.store.add(
            BagRecord(
                bag_id="bag-a",
                path=self.root / "bag-a.bag",
                source_path=self.root / "bag-a.bag",
                sha256="a" * 64,
                included=True,
                auto_interval=(0.0, 0.2),
                selected_interval=(0.0, 0.2),
                status="ready",
                view_range=(0.0, 0.2),
            )
        )
        self.widgets: list[object] = []

    def tearDown(self) -> None:
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        self.application.processEvents()
        self.temporary.cleanup()

    def _load_run(
        self, mcmc: bool, valid_geometry: bool = True
    ) -> artifact_loader.BatchEstimationRun:
        bundle = _backend_bundle(
            self.root / "run", mcmc, valid_geometry
        )
        bag = bundle.bags["bag-a"]
        pose_time = np.asarray((0.0, 0.1, 0.2))
        pose_orientation = np.zeros((3, 4))
        pose_orientation[:, 3] = 1.0
        bag.update(
            {
                "pose_time": pose_time,
                "pose_record_time": 100.0 + pose_time,
                "pose_position": np.zeros((3, 3)),
                "pose_orientation_xyzw": pose_orientation,
                "pose_valid": np.ones((3,), dtype=bool),
                "pose_covariance": np.repeat(
                    np.eye(6)[None, :, :], 3, axis=0
                ),
                "pose_covariance_valid": np.ones((3,), dtype=bool),
            }
        )
        observation_valid = np.asarray((True, False, True), dtype=bool)
        for prefix, value_name, dimension, _covariance_dimension in (
            ("velocity", "velocity", 3, 3),
            ("gyro", "gyro", 3, 3),
            ("accelerometer", "accelerometer", 3, 3),
            ("thrust_command", "thrust_command", 4, 4),
            ("gimbal_command", "gimbal_command", 4, 4),
            ("gimbal_observation", "gimbal_observation", 4, 4),
            (
                "controller_integral",
                "controller_integral_observation",
                6,
                6,
            ),
        ):
            bag.update(
                {
                    "{}_time".format(prefix): pose_time,
                    "{}_record_time".format(prefix): 100.0 + pose_time,
                    value_name: np.arange(
                        pose_time.size * dimension, dtype=float
                    ).reshape(pose_time.size, dimension),
                    "{}_valid".format(prefix): observation_valid,
                    "{}_covariance".format(prefix): np.repeat(
                        np.eye(dimension)[None, :, :],
                        pose_time.size,
                        axis=0,
                    ),
                    "{}_covariance_valid".format(prefix): np.ones(
                        pose_time.size, dtype=bool
                    ),
                }
            )
        substages = {
            "map": {"converged": True, "termination_reason": "done"},
            "laplace_em": {
                "converged": True,
                "termination_reason": "done",
            },
            "laplace": {"converged": True, "termination_reason": "done"},
        }
        if mcmc:
            substages["mcmc"] = {
                "converged": True,
                "termination_reason": "done",
            }
        bundle.manifest["substage_status"] = substages
        bundle.manifest["q_definition"] = {
            "definition": (
                "body_wrench/continuous_spectral_density"
            ),
            "units": ["m/s^2"] * 3 + ["rad/s^2"] * 3,
        }
        with mock.patch.object(
            artifact_loader.batch_artifact_io,
            "load_batch_estimation_run",
            return_value=bundle,
        ):
            run = artifact_loader.load_batch_estimation_run(bundle.root)
        run.manifest["request_fingerprint"] = self.store.request_fingerprint()
        self.store.apply_estimation(run)
        return run

    def test_master_shows_map_laplace_em_and_optional_mcmc(self) -> None:
        from grape_param_estim_gui.widgets.master_view import MasterView

        run = self._load_run(mcmc=True)
        view = MasterView(self.store)
        self.widgets.append(view)

        self.assertIn("MCMC sample 101", view.sample_detail.text())
        self.assertIn("Laplace rank: 18/18", view.diagnostic_label.text())
        self.assertIn("3 retained equal-weight samples", view.diagnostic_label.text())
        self.assertEqual(view.posterior_widget.run, run)
        geometry_names = {
            item.name()
            for item in view.posterior_widget.geometry_plot.listDataItems()
        }
        self.assertIn("joint static marginal 1σ", geometry_names)
        self.assertNotIn("fixed-delay conditional static 1σ", geometry_names)
        self.assertEqual(
            {
                item.name()
                for item in view.posterior_widget.information_plot.listDataItems()
            },
            {
                "prior information",
                "likelihood information",
                "posterior information",
            },
        )
        trace = view.mcmc_trace_widget
        self.assertEqual(trace.chain_combo.count(), 3)
        self.assertEqual(len(trace.ridge_plot.listDataItems()), 2)
        self.assertEqual(len(trace.delay_plot.listDataItems()), 2)
        self.assertEqual(
            {
                item.name()
                for item in trace.delay_marginal_plot.listDataItems()
            },
            {"joint Laplace delay marginal", "MCMC delay samples"},
        )
        self.assertIn("stage1 6/8", trace.kernel_label.text())
        self.assertIn("inner failures 0", trace.kernel_label.text())
        trace.chain_combo.setCurrentIndex(1)
        self.assertEqual(len(trace.ridge_plot.listDataItems()), 1)
        self.assertEqual(trace.chain_combo.currentText(), "chain-0")
        self.store.set_selected_sample("109")
        self.assertIn("MCMC sample 109", view.sample_detail.text())

    def test_master_labels_invalid_geometry_as_conditional_not_marginal(self) -> None:
        from grape_param_estim_gui.widgets.master_view import MasterView

        self._load_run(mcmc=True, valid_geometry=False)
        view = MasterView(self.store)
        self.widgets.append(view)

        geometry_names = {
            item.name()
            for item in view.posterior_widget.geometry_plot.listDataItems()
        }
        self.assertIn(
            "fixed-delay conditional static 1σ", geometry_names
        )
        self.assertNotIn("joint static marginal 1σ", geometry_names)
        marginal_names = {
            item.name()
            for item in view.mcmc_trace_widget.delay_marginal_plot.listDataItems()
        }
        self.assertEqual(marginal_names, {"MCMC delay samples"})
        self.assertIn(
            "Joint local Laplace delay marginal unavailable",
            view.mcmc_trace_widget.delay_marginal_status.text(),
        )
        self.assertIn("uniform-prior delay 1σ=0.003 s", view.diagnostic_label.text())
        self.assertIn("MCMC proposal scale=0.0004 s", view.diagnostic_label.text())
        estimate_only = self._load_run(
            mcmc=False, valid_geometry=False
        )
        view.mcmc_trace_widget.set_run(estimate_only)
        self.assertIn(
            "MCMC was not run; configured fallback",
            view.mcmc_trace_widget.delay_marginal_status.text(),
        )

    def test_master_map_only_run_keeps_mcmc_controls_empty(self) -> None:
        from grape_param_estim_gui.widgets.master_view import MasterView

        map_store = ProjectStore(
            self.root / "map-project", new_project_manifest("map-view")
        )
        map_store.manifest["bags"] = [
            {"bag_id": "bag-a", "sha256": "a" * 64}
        ]
        map_store.add(
            BagRecord(
                "bag-a",
                self.root / "bag-a.bag",
                self.root / "bag-a.bag",
                "a" * 64,
                included=True,
            )
        )
        bundle = _backend_bundle(self.root / "map-run", False)
        bundle.manifest["substage_status"] = {
            "map": {"converged": True, "termination_reason": "done"},
            "laplace_em": {"converged": True, "termination_reason": "done"},
            "laplace": {"converged": True, "termination_reason": "done"},
        }
        bundle.manifest["q_definition"] = {
            "definition": (
                "body_wrench/continuous_spectral_density"
            ),
            "units": ["unit"] * 6,
        }
        with mock.patch.object(
            artifact_loader.batch_artifact_io,
            "load_batch_estimation_run",
            return_value=bundle,
        ):
            map_run = artifact_loader.load_batch_estimation_run(bundle.root)
        map_run.manifest["request_fingerprint"] = map_store.request_fingerprint()
        map_store.apply_estimation(map_run)
        map_view = MasterView(map_store)
        self.widgets.append(map_view)
        self.assertTrue(map_view.sample_detail.text().startswith("MAP |"))
        self.assertIn("MAP/Laplace result only", map_view.diagnostic_label.text())
        self.assertEqual(map_view.mcmc_trace_widget.chain_combo.count(), 0)
        self.assertEqual(
            map_view.mcmc_trace_widget.kernel_label.text(),
            "MCMC traces unavailable.",
        )

    def test_bag_browser_uses_selected_conditional_and_dynamics_residual(self) -> None:
        import os

        os.environ["GRAPE_PARAM_ESTIM_DISABLE_3D"] = "1"
        from grape_param_estim_gui.widgets.bag_browser import BagBrowserView

        self._load_run(mcmc=True)
        view = BagBrowserView(self.store)
        self.widgets.append(view)

        self.assertEqual(view.signal_tabs.tabText(2), "Dynamics residual")
        self.assertEqual(view.signal_tabs.tabText(3), "Direct observations")
        for trajectory_plot, correction_plot in zip(
            view.trajectory_panel.plots,
            view.correction_panel.plots,
            strict=True,
        ):
            np.testing.assert_allclose(
                trajectory_plot.getViewBox().viewRange()[1],
                correction_plot.getViewBox().viewRange()[1],
            )
        view.correction_panel.plots[0].setYRange(-4.0, 5.0, padding=0.0)
        self.application.processEvents()
        np.testing.assert_allclose(
            view.trajectory_panel.plots[0].getViewBox().viewRange()[1],
            (-4.0, 5.0),
        )
        view.trajectory_panel.plots[0].setYRange(-2.0, 3.0, padding=0.0)
        self.application.processEvents()
        np.testing.assert_allclose(
            view.correction_panel.plots[0].getViewBox().viewRange()[1],
            (-2.0, 3.0),
        )
        self.assertTrue(
            {
                "reference",
                "observed",
                "nominal",
                "map",
                "selected",
            }.issubset(view.trajectory_panel.series_data)
        )
        self.assertTrue(
            {
                "map",
                "selected",
                "q_upper",
                "q_lower",
                "map_normalized",
                "selected_normalized",
            }.issubset(view.dynamics_panel.series_data)
        )
        trajectory_names = {
            item.name()
            for item in view.trajectory_panel.plots[0].listDataItems()
            if item.name() is not None
        }
        self.assertTrue(
            {
                "Reference trajectory",
                "Observed pose",
                "Nominal trajectory",
                "MAP trajectory",
                "Stored posterior 5–95% band",
                "Selected conditional sample",
            }.issubset(trajectory_names)
        )
        view.trajectory_panel.set_series_visible("observed", False)
        self.assertFalse(
            view.trajectory_panel.series_checkboxes["observed"].isChecked()
        )
        self.assertTrue(
            all(
                "Observed pose"
                not in {
                    item.name()
                    for item in plot.listDataItems()
                    if item.name() is not None
                }
                for plot in view.trajectory_panel.plots
            )
        )
        correction_names = {
            item.name()
            for item in view.correction_panel.plots[0].listDataItems()
            if item.name() is not None
        }
        self.assertTrue(
            {
                "Nominal (zero correction)",
                "MAP correction",
                "Stored posterior 5–95% band",
                "Selected correction sample",
            }.issubset(correction_names)
        )
        view.correction_panel.set_series_visible("nominal", False)
        self.assertNotIn(
            "Nominal (zero correction)",
            {
                item.name()
                for item in view.correction_panel.plots[0].listDataItems()
                if item.name() is not None
            },
        )
        self.assertIn("pose: used", view.inspection_details.text())
        direct = view.direct_observation_panel
        self.assertEqual(
            set(direct.series_data),
            {
                "velocity",
                "gyro",
                "accelerometer",
                "thrust_command",
                "gimbal_command",
                "gimbal_observation",
                "controller_integral_observation",
            },
        )
        self.assertEqual(direct.series_data["velocity"][0].tolist(), [0.0, 0.2])
        self.assertEqual(len(direct.plot.listDataItems()), 3)
        direct.set_component_visible("velocity", 1, False)
        self.assertEqual(
            {
                item.name()
                for item in direct.plot.listDataItems()
                if item.name() is not None
            },
            {"vx", "vz"},
        )
        direct.observation_combo.setCurrentIndex(
            direct.observation_combo.findData("thrust_command")
        )
        self.assertEqual(len(direct.plot.listDataItems()), 4)
        self.assertIn("2/3 valid samples", direct.status_label.text())
        direct.observation_combo.setCurrentIndex(
            direct.observation_combo.findData("velocity")
        )
        self.assertFalse(direct.component_checkboxes[1].isChecked())
        self.assertEqual(len(direct.plot.listDataItems()), 2)
        view.dynamics_panel.dynamics_display_combo.setCurrentIndex(1)
        self.assertIn("±1 after normalization", view.dynamics_panel.status_label.text())
        self.assertTrue(
            {
                "MAP normalized residual",
                "Selected normalized residual",
                "Q reference ±1",
            }.issubset(
                {
                    item.name()
                    for item in view.dynamics_panel.plots[0].listDataItems()
                    if item.name() is not None
                }
            )
        )
        view.dynamics_panel.set_series_visible("q_band", False)
        self.assertTrue(
            all(
                "Q reference ±1"
                not in {
                    item.name()
                    for item in plot.listDataItems()
                    if item.name() is not None
                }
                for plot in view.dynamics_panel.plots
            )
        )
        self.store.set_selected_sample("107")
        self.assertNotIn("selected", view.trajectory_panel.series_data)
        self.assertFalse(
            view.trajectory_panel.series_checkboxes["observed"].isChecked()
        )
        original_record = view.current_record
        flight_record = SimpleNamespace(
            preview=SimpleNamespace(
                time=np.asarray((0.0, 0.1, 0.2)),
                flight_state=np.asarray((0.0, 1.0, 1.0)),
            )
        )
        view.current_record = flight_record
        view._render_flight_state(flight_record)
        self.assertEqual(
            {
                item.name()
                for item in view.flight_state_plot.listDataItems()
                if item.name() is not None
            },
            {"Recorded flight state"},
        )
        view.flight_state_checkbox.setChecked(False)
        self.assertEqual(view.flight_state_plot.listDataItems(), [])
        view.current_record = original_record
        self.store.set_selected_sample("109")
        self.assertIn("selected", view.trajectory_panel.series_data)
        self.assertEqual(view.sample_label.text(), "109")


if __name__ == "__main__":
    unittest.main()
