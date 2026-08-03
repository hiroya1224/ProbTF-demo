from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.ensemble_convergence import (
    deterministic_sliced_wasserstein_1,
    empirical_wasserstein_1,
    run_ensemble_size_convergence,
    save_ensemble_convergence,
)
from grape_param_estim.strong_constraint import (
    CONTROL_DIMENSION,
    PARAMETER_OFFSET,
    StrongConstraintPrior,
)
from grape_param_estim.strong_constraint_experiments import (
    _problem_from_synthetic,
)


class EnsembleSizeConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Four real assimilations: strong at two M and the short full-block
        # weak problem at two M>D resolutions.  One accepted IEnKS iteration
        # keeps this convergence regression suitable for CI.
        cls.report = run_ensemble_size_convergence()

    def test_sliced_w1_uses_raw_laws_not_only_gaussian_moments(self):
        first = np.asarray((-1.0, -1.0, 1.0, 1.0))[:, None]
        second = np.asarray(
            (-np.sqrt(2.0), 0.0, 0.0, np.sqrt(2.0))
        )[:, None]
        np.testing.assert_allclose(
            np.mean(first, axis=0), np.mean(second, axis=0)
        )
        np.testing.assert_allclose(
            np.cov(first, rowvar=False), np.cov(second, rowvar=False)
        )
        # A single-Gaussian summary cannot distinguish these laws, whereas
        # the canonical empirical-member diagnostic does.
        self.assertGreater(
            deterministic_sliced_wasserstein_1(first, second), 0.50
        )
        self.assertEqual(
            deterministic_sliced_wasserstein_1(first, first), 0.0
        )
        self.assertAlmostEqual(
            empirical_wasserstein_1((0.0, 2.0), (1.0,)), 1.0
        )

    def test_two_sizes_really_assimilate_each_constraint_model(self):
        report = self.report
        interval_count = report.synthetic.truth.times.size - 1
        self.assertEqual(
            report.weak_control_dimension,
            CONTROL_DIMENSION + 6 * interval_count,
        )
        self.assertEqual(
            [law.ensemble_size for law in report.strong_laws],
            [38, 46],
        )
        self.assertEqual(
            [law.ensemble_size for law in report.weak_laws],
            [report.weak_control_dimension + 2,
             report.weak_control_dimension + 10],
        )
        for law in report.strong_laws + report.weak_laws:
            self.assertGreater(law.ensemble_size, law.control_dimension)
            self.assertEqual(law.iteration_count, 1)
            self.assertTrue(law.assimilation_reduced_objective)
            self.assertLess(
                law.final_accepted_objective, law.initial_objective
            )

    def test_raw_quotient_and_full_path_members_are_retained(self):
        report = self.report
        sample_count = report.synthetic.truth.times.size
        prior = StrongConstraintPrior.grape()
        parameter_covariance = prior.covariance[
            PARAMETER_OFFSET:, PARAMETER_OFFSET:
        ]
        factor = np.linalg.cholesky(parameter_covariance)
        ridge = _problem_from_synthetic(
            report.synthetic
        ).parameter_chart.ridge_direction()
        whitened_ridge = np.linalg.solve(factor, ridge)
        for law in report.strong_laws + report.weak_laws:
            member_count = law.ensemble_size
            self.assertEqual(
                law.parameter_coordinates.shape, (member_count, 18)
            )
            self.assertEqual(
                law.identifiable_quotient.shape, (member_count, 18)
            )
            self.assertEqual(
                law.whitened_identifiable_quotient.shape,
                (member_count, 18),
            )
            self.assertEqual(
                law.correction_translation.shape,
                (member_count, sample_count, 3),
            )
            self.assertEqual(
                law.correction_rotation_vector.shape,
                (member_count, sample_count, 3),
            )
            self.assertEqual(
                law.whitened_correction_path.shape,
                (member_count, 6 * sample_count),
            )
            # The exact common-scale ridge has been quotiented in the prior
            # metric, without sorting or collapsing the member rows.
            np.testing.assert_allclose(
                law.whitened_identifiable_quotient @ whitened_ridge,
                0.0,
                atol=2.0e-13,
            )

    def test_endpoint_raw_laws_converge_below_declared_uncertainty_scale(self):
        # Parameter coordinates are in prior-sigma units and path coordinates
        # in pose-observation-sigma units.  These bounds therefore require an
        # M increase to move the raw law by less than one declared uncertainty
        # scale; they are not fitted physical-parameter tolerances.
        for comparison in (
            self.report.strong_endpoint_comparison,
            self.report.weak_endpoint_comparison,
        ):
            self.assertLess(comparison.identifiable_sliced_w1, 0.50)
            self.assertLess(comparison.path_sliced_w1, 0.75)
            self.assertLess(comparison.identifiable_mean_shift, 0.50)
            self.assertLess(comparison.path_mean_shift, 0.75)
            self.assertLess(comparison.ridge_variance_ratio_change, 0.02)
            self.assertLessEqual(comparison.path_coverage_change, 0.10)
            self.assertTrue(comparison.ridge_conclusion_stable)
            self.assertTrue(comparison.pose_mean_conclusion_stable)
            self.assertTrue(comparison.coverage_conclusion_stable)

    def test_weak_vs_strong_conclusion_is_stable_at_both_sizes(self):
        report = self.report
        self.assertTrue(report.strong_size_conclusions_stable)
        self.assertTrue(report.weak_size_conclusions_stable)
        self.assertTrue(report.weak_strong_conclusion_stable)
        for law in report.strong_laws + report.weak_laws:
            self.assertTrue(law.ridge_preserved)
        self.assertEqual(
            [value.signature for value in report.weak_strong_conclusions],
            [(True, True, True), (True, True, True)],
        )
        # The qualitative result being held stable is explicit: weak-Q has
        # lower identifiable bias/path mean error and better truth coverage.
        for conclusion in report.weak_strong_conclusions:
            self.assertTrue(conclusion.weak_has_lower_identifiable_bias)
            self.assertTrue(conclusion.weak_has_lower_path_mean_error)
            self.assertTrue(conclusion.weak_has_higher_path_coverage)

    def test_pickle_free_artifact_retains_raw_resolution_laws(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = save_ensemble_convergence(
                str(Path(directory) / "convergence.npz"), self.report
            )
            with np.load(str(destination), allow_pickle=False) as artifact:
                self.assertEqual(
                    str(artifact["schema"][0]),
                    "grape-param-estim/ensemble-convergence/v1",
                )
                np.testing.assert_array_equal(
                    artifact["weak_1_parameter_coordinates"],
                    self.report.weak_laws[1].parameter_coordinates,
                )
                np.testing.assert_array_equal(
                    artifact["strong_0_whitened_correction_path"],
                    self.report.strong_laws[0].whitened_correction_path,
                )
                self.assertTrue(
                    artifact["weak_strong_conclusion_stable"][0]
                )
                self.assertNotIn("weights", artifact.files)


if __name__ == "__main__":
    unittest.main()
