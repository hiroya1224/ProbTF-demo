from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.geometry import correction_transform_path
from grape_param_estim.strong_constraint import (
    CONTROL_DIMENSION,
    PARAMETER_OFFSET,
    StrongConstraintIEnKS,
    StrongConstraintPrior,
)
from grape_param_estim.strong_constraint_experiments import (
    _problem_from_synthetic,
    run_phase2_experiment,
    save_phase2_experiment,
)


class StrongConstraintIEnKSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        arguments = dict(
            duration=0.4,
            time_step=0.04,
            ensemble_size=38,
            maximum_iterations=1,
            seed=11,
        )
        cls.experiment_a = run_phase2_experiment("A", **arguments)
        cls.experiment_b = run_phase2_experiment("B", **arguments)

    def test_prior_ensemble_has_exact_declared_moments(self):
        prior = StrongConstraintPrior.grape()
        ensemble = prior.ensemble(38, seed=5)
        self.assertEqual(ensemble.shape, (38, CONTROL_DIMENSION))
        np.testing.assert_allclose(
            np.mean(ensemble, axis=0), prior.mean, atol=2.0e-16
        )
        np.testing.assert_allclose(
            np.cov(ensemble, rowvar=False),
            prior.covariance,
            atol=2.0e-16,
        )

    def test_ensemble_regression_recovers_linear_sensitivity(self):
        generator = np.random.RandomState(4)
        sensitivity = generator.normal(size=(17, 8))
        coordinates = generator.normal(size=(8, 25))
        coordinates -= np.mean(coordinates, axis=1, keepdims=True)
        intercept = generator.normal(size=(17, 1))
        residuals = intercept + sensitivity @ coordinates
        recovered = StrongConstraintIEnKS._regression(
            residuals, coordinates
        )
        np.testing.assert_allclose(recovered, sensitivity, atol=2.0e-14)

    def test_full_closed_loop_is_invariant_along_exact_parameter_ridge(self):
        problem = _problem_from_synthetic(self.experiment_a.synthetic)
        first = np.zeros(CONTROL_DIMENSION)
        second = first.copy()
        second[PARAMETER_OFFSET:] = (
            0.19 * problem.parameter_chart.ridge_direction()
        )
        baseline = problem.forecast(first)
        equivalent = problem.forecast(second)
        np.testing.assert_allclose(
            equivalent.position, baseline.position, atol=2.0e-15
        )
        np.testing.assert_allclose(
            equivalent.orientation_xyzw,
            baseline.orientation_xyzw,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            equivalent.linear_velocity,
            baseline.linear_velocity,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            equivalent.angular_velocity,
            baseline.angular_velocity,
            atol=2.0e-14,
        )

    def test_experiment_a_recovers_equivalence_class_and_paths(self):
        result = self.experiment_a
        posterior = result.posterior
        metrics = result.metrics
        self.assertEqual(posterior.ensemble_rank, CONTROL_DIMENSION)
        self.assertEqual(posterior.control_ensemble.shape, (38, 36))
        self.assertEqual(len(posterior.trajectory_ensemble), 38)
        self.assertEqual(
            posterior.correction_translation.shape,
            (38, result.synthetic.truth.times.size, 3),
        )
        self.assertLess(metrics.posterior_pose_rmse, 0.1 * metrics.prior_pose_rmse)
        self.assertLess(
            metrics.posterior_identifiable_parameter_error,
            metrics.prior_identifiable_parameter_error,
        )
        self.assertGreater(metrics.ridge_variance_ratio, 0.80)
        # 17 identifiable directions: the truth equivalence class lies well
        # inside the posterior ellipsoid (95% chi-square is about 27.6).
        self.assertLess(metrics.truth_equivalence_mahalanobis, 27.6)
        self.assertGreater(metrics.truth_pose_component_coverage, 0.90)
        objectives = [value.objective for value in posterior.iterations]
        self.assertTrue(np.all(np.diff(objectives) < 0.0))
        for iteration in posterior.iterations:
            self.assertLess(iteration.accepted_objective, iteration.objective)
        self.assertIn(
            posterior.termination_reason,
            ("step_tolerance", "maximum_iterations"),
        )

        translation, rotation = correction_transform_path(
            result.synthetic.nominal.position,
            result.synthetic.nominal.orientation_xyzw,
            posterior.trajectory_ensemble[7].position,
            posterior.trajectory_ensemble[7].orientation_xyzw,
        )
        np.testing.assert_array_equal(
            translation, posterior.correction_translation[7]
        )
        np.testing.assert_array_equal(
            rotation, posterior.correction_rotation_vector[7]
        )

    def test_experiment_b_smooths_unobserved_velocity_and_omega(self):
        metrics = self.experiment_b.metrics
        observations = self.experiment_b.synthetic.observations
        self.assertFalse(hasattr(observations, "linear_velocity"))
        self.assertFalse(hasattr(observations, "angular_velocity"))
        self.assertLess(
            metrics.posterior_velocity_rmse,
            0.20 * metrics.prior_velocity_rmse,
        )
        self.assertLess(
            metrics.posterior_omega_rmse,
            0.30 * metrics.prior_omega_rmse,
        )
        self.assertLess(metrics.posterior_pose_rmse, metrics.prior_pose_rmse)
        self.assertGreater(metrics.ridge_variance_ratio, 0.80)
        self.assertGreater(metrics.truth_pose_component_coverage, 0.75)

    def test_phase2_artifact_keeps_raw_member_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = save_phase2_experiment(
                str(Path(directory) / "phase2.npz"), self.experiment_a
            )
            with np.load(str(destination), allow_pickle=False) as artifact:
                self.assertEqual(
                    str(artifact["schema"][0]),
                    "grape-weak-constraint/phase2",
                )
                self.assertEqual(
                    artifact["posterior_control_ensemble"].shape,
                    (38, CONTROL_DIMENSION),
                )
                self.assertEqual(
                    artifact["posterior_position"].shape[0], 38
                )
                self.assertEqual(
                    artifact["correction_translation"].shape[0], 38
                )
                self.assertEqual(
                    artifact["posterior_commanded_thrust"].shape[0], 38
                )
                self.assertEqual(
                    artifact["posterior_body_wrench"].shape[0], 38
                )
                self.assertEqual(artifact["prior_covariance"].shape, (36, 36))
                self.assertNotIn("weights", artifact.files)
                self.assertNotIn("particles", artifact.files)


if __name__ == "__main__":
    unittest.main()
