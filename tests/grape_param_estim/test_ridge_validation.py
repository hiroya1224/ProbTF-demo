from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.ridge_validation import (
    save_ridge_validation,
    validate_phase2_ridge,
    validate_weak_zero_realization_ridge,
)
from grape_param_estim.strong_constraint import PARAMETER_OFFSET
from grape_param_estim.strong_constraint_experiments import (
    run_phase2_experiment,
)


class RidgeValidationTests(unittest.TestCase):
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
        cls.report_a = validate_phase2_ridge(cls.experiment_a)
        cls.report_b = validate_phase2_ridge(cls.experiment_b)
        cls.weak_report = validate_weak_zero_realization_ridge(
            cls.experiment_a,
            maximum_iterations=1,
            seed=11,
        )

    def test_exact_ridge_rollouts_and_pose_likelihood_are_invariant(self):
        for report in (self.report_a, self.report_b):
            self.assertLess(
                np.max(report.rollout_position_max_error), 3.0e-12
            )
            self.assertLess(
                np.max(report.rollout_rotation_max_error), 3.0e-12
            )
            self.assertLess(report.pose_cost_range, 1.0e-7)

    def test_raw_lambda_law_remains_the_proper_prior_law(self):
        for report in (self.report_a, self.report_b):
            self.assertLess(report.posterior_lambda_mean_zscore, 0.35)
            self.assertGreater(report.posterior_lambda_variance_ratio, 0.80)
            self.assertLess(report.posterior_lambda_variance_ratio, 1.20)
            self.assertLess(report.lambda_wasserstein_ratio, 0.05)
            self.assertAlmostEqual(
                np.mean(report.prior_lambda_samples),
                report.theoretical_lambda_mean,
                delta=2.0e-14,
            )
            self.assertAlmostEqual(
                np.var(report.prior_lambda_samples, ddof=1),
                report.theoretical_lambda_variance,
                delta=2.0e-14,
            )

    def test_identifiable_quotient_contains_synthetic_truth(self):
        for report in (self.report_a, self.report_b):
            self.assertLess(report.quotient_truth_mahalanobis, 27.6)
            np.testing.assert_allclose(
                report.posterior_parameter_samples
                - np.outer(
                    report.posterior_lambda_samples, report.direction
                ),
                report.posterior_quotient_samples,
                atol=2.0e-16,
            )

    def test_prior_whitened_covariance_does_not_inform_exact_ridge(self):
        for report in (self.report_a, self.report_b):
            self.assertLess(
                report.prior_whitened_information_leak, 0.10
            )

    def test_true_correction_path_is_covered_by_raw_member_law(self):
        self.assertGreater(self.report_a.path_component_coverage, 0.90)
        self.assertGreater(self.report_b.path_component_coverage, 0.75)
        for report, experiment in (
            (self.report_a, self.experiment_a),
            (self.report_b, self.experiment_b),
        ):
            member_count = experiment.posterior.control_ensemble.shape[0]
            sample_count = experiment.synthetic.truth.times.size
            self.assertEqual(
                report.path_residual_samples.shape,
                (member_count, sample_count, 6),
            )
            self.assertEqual(
                report.correction_translation_samples.shape,
                (member_count, sample_count, 3),
            )
            self.assertEqual(
                report.posterior_parameter_samples.shape,
                (member_count, 18),
            )

    def test_report_is_reproducible_and_retains_raw_samples(self):
        repeated = validate_phase2_ridge(self.experiment_a)
        for name in (
            "rollout_pose_cost",
            "prior_parameter_samples",
            "posterior_parameter_samples",
            "prior_lambda_samples",
            "posterior_lambda_samples",
            "prior_quotient_samples",
            "posterior_quotient_samples",
            "correction_translation_samples",
            "correction_rotation_vector_samples",
            "path_residual_samples",
        ):
            np.testing.assert_array_equal(
                getattr(repeated, name), getattr(self.report_a, name)
            )
        np.testing.assert_array_equal(
            self.report_a.prior_parameter_samples,
            self.experiment_a.posterior.prior_control_ensemble[
                :, PARAMETER_OFFSET:
            ],
        )
        np.testing.assert_array_equal(
            self.report_a.posterior_parameter_samples,
            self.experiment_a.posterior.parameter_ensemble.coordinates,
        )

    def test_invalid_ridge_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "containing zero"):
            validate_phase2_ridge(self.experiment_a, (-0.2, 0.1, 0.3))

    def test_ienks_q_preserves_zero_realization_ridge_law(self):
        report = self.weak_report
        self.assertLess(report.posterior_lambda_mean_zscore, 0.35)
        self.assertGreater(report.posterior_lambda_variance_ratio, 0.80)
        self.assertLess(report.posterior_lambda_variance_ratio, 1.20)
        self.assertLess(report.lambda_wasserstein_ratio, 0.05)
        self.assertLess(report.prior_whitened_information_leak, 0.10)
        self.assertGreater(report.static_ridge_variance_ratio, 0.80)
        self.assertLess(report.static_ridge_variance_ratio, 1.20)
        self.assertLess(report.maximum_center_residual_wrench, 0.005)
        self.assertFalse(report.particle_correction_required)

    def test_weak_nonzero_residual_symmetry_scales_static_and_q_together(self):
        report = self.weak_report
        self.assertLess(
            np.max(report.augmented_position_max_error), 3.0e-12
        )
        self.assertLess(
            np.max(report.augmented_rotation_max_error), 3.0e-12
        )
        self.assertLess(
            np.max(report.augmented_pose_residual_max_error), 1.0e-8
        )
        self.assertGreater(
            np.max(np.abs(report.posterior_residual_wrench_samples)), 0.0
        )
        self.assertEqual(
            report.posterior_innovation_samples.shape[0],
            report.posterior_parameter_samples.shape[0],
        )

    def test_pickle_free_artifact_retains_each_raw_law(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = save_ridge_validation(
                str(Path(directory) / "ridge.npz"),
                (self.report_a, self.report_b),
                self.weak_report,
            )
            with np.load(str(destination), allow_pickle=False) as artifact:
                self.assertEqual(
                    str(artifact["schema"][0]),
                    "grape-weak-constraint/phase4-ridge",
                )
                self.assertEqual(
                    artifact["experiment_label"].tolist(), ["A", "B"]
                )
                np.testing.assert_array_equal(
                    artifact["report_1_posterior_parameter_samples"],
                    self.report_b.posterior_parameter_samples,
                )
                np.testing.assert_array_equal(
                    artifact["report_0_path_residual_samples"],
                    self.report_a.path_residual_samples,
                )
                np.testing.assert_array_equal(
                    artifact["weak_posterior_innovation_samples"],
                    self.weak_report.posterior_innovation_samples,
                )
                self.assertFalse(
                    artifact["weak_particle_correction_required"][0]
                )
                self.assertNotIn("weights", artifact.files)


if __name__ == "__main__":
    unittest.main()
