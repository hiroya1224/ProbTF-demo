import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.geometry import correction_transform_path
from grape_param_estim.model_error import GaussMarkovWrenchProcess
from grape_param_estim.strong_constraint import CONTROL_DIMENSION
from grape_param_estim.strong_constraint_experiments import (
    _problem_from_synthetic,
)
from grape_param_estim.weak_constraint import WeakConstraintProblem
from grape_param_estim.weak_constraint_experiments import (
    oracle_effective_residual_wrench,
    replay_nominal_actuators_on_truth,
    run_phase3_experiment,
    save_phase3_experiment,
)


class WeakConstraintIEnKSQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_phase3_experiment(
            duration=0.3,
            time_step=0.04,
            ensemble_size=88,
            maximum_iterations=2,
            seed=7,
        )

    def test_prior_spans_every_independent_model_error_block(self):
        posterior = self.result.weak_posterior
        prior = posterior.prior_control_ensemble
        interval_count = self.result.synthetic.truth.times.size - 1
        expected_dimension = CONTROL_DIMENSION + 6 * interval_count
        self.assertEqual(prior.shape, (88, expected_dimension))
        self.assertEqual(posterior.ensemble_rank, expected_dimension)
        np.testing.assert_allclose(np.mean(prior, axis=0), 0.0, atol=5.0e-15)
        covariance = np.cov(prior, rowvar=False)
        innovation_covariance = covariance[
            CONTROL_DIMENSION:, CONTROL_DIMENSION:
        ]
        np.testing.assert_allclose(
            innovation_covariance,
            np.eye(6 * interval_count),
            atol=2.0e-11,
        )
        np.testing.assert_allclose(
            covariance[:CONTROL_DIMENSION, CONTROL_DIMENSION:],
            0.0,
            atol=2.0e-12,
        )

    def test_zero_q_blocks_reduce_exactly_to_strong_constraint(self):
        strong_problem = _problem_from_synthetic(self.result.synthetic)
        process = GaussMarkovWrenchProcess(
            times=self.result.synthetic.observations.times[:-1],
            stationary_standard_deviation=np.ones(6),
            correlation_time=0.4,
        )
        weak_problem = WeakConstraintProblem(strong_problem, process)
        strong_control = np.zeros(CONTROL_DIMENSION)
        weak_control = np.zeros(weak_problem.control_dimension)
        strong = strong_problem.forecast(strong_control)
        weak = weak_problem.forecast(weak_control)
        np.testing.assert_array_equal(weak.position, strong.position)
        np.testing.assert_array_equal(
            weak.orientation_xyzw, strong.orientation_xyzw
        )
        np.testing.assert_array_equal(
            weak.commanded_thrust, strong.commanded_thrust
        )

    def test_experiment_c_separates_static_and_time_varying_error(self):
        metrics = self.result.metrics
        # The matched run has the same static truth, observation-noise
        # realization and short window.  Its bias is therefore the control
        # for finite-window/non-identifiability effects, leaving the excess
        # strong bias attributable to model error.
        self.assertGreater(
            metrics.strong_static_bias,
            1.50 * metrics.matched_strong_static_bias,
        )
        self.assertLess(
            abs(
                metrics.weak_static_bias
                - metrics.matched_strong_static_bias
            ),
            0.25
            * abs(
                metrics.strong_static_bias
                - metrics.matched_strong_static_bias
            ),
        )
        self.assertLess(metrics.weak_static_bias, 0.75 * metrics.strong_static_bias)
        self.assertLess(
            metrics.weak_pose_rmse, 1.05 * metrics.strong_pose_rmse
        )
        self.assertLess(
            metrics.weak_rotation_rmse,
            0.80 * metrics.strong_rotation_rmse,
        )
        self.assertLess(
            metrics.weak_velocity_rmse, 0.90 * metrics.strong_velocity_rmse
        )
        self.assertLess(metrics.weak_omega_rmse, 0.70 * metrics.strong_omega_rmse)
        self.assertGreater(metrics.weak_path_coverage, metrics.strong_path_coverage)
        self.assertGreater(metrics.weak_path_coverage, 0.85)
        self.assertGreater(metrics.residual_acceleration_r_squared, 0.30)
        self.assertGreater(metrics.residual_component_coverage, 0.75)

    def test_matched_control_reuses_static_truth_and_noise_realization(self):
        mismatched = self.result.synthetic
        matched = self.result.matched_synthetic
        np.testing.assert_array_equal(
            mismatched.truth.times, matched.truth.times
        )
        np.testing.assert_allclose(
            mismatched.observations.position - mismatched.truth.position,
            matched.observations.position - matched.truth.position,
            atol=2.0e-16,
        )
        np.testing.assert_allclose(
            matched.truth_parameters.linear_drag, np.zeros(3), atol=0.0
        )
        np.testing.assert_allclose(
            matched.truth_parameters.angular_drag, np.zeros(3), atol=0.0
        )
        coordinates = _problem_from_synthetic(
            matched
        ).parameter_chart.encode(matched.truth_parameters)
        np.testing.assert_allclose(
            coordinates,
            self.result.truth_static_coordinates,
            atol=2.0e-15,
        )

    def test_oracle_uses_independent_causal_nominal_replay(self):
        synthetic = copy.deepcopy(self.result.synthetic)
        expected = oracle_effective_residual_wrench(synthetic)
        replay = replay_nominal_actuators_on_truth(synthetic)
        # At k=0 both controllers see no actuator feedback.  Thereafter the
        # nominal replay feeds back its own actuator, and must differ from the
        # lagged truth actuator path used to generate the episode.
        np.testing.assert_allclose(
            replay.commanded_thrust[0],
            synthetic.truth.commanded_thrust[0],
            atol=1.0e-13,
        )
        self.assertGreater(
            np.max(
                np.abs(
                    replay.actuator_gimbal_angle
                    - synthetic.truth.actuator_gimbal_angle
                )
            ),
            0.01,
        )
        # Regression guard: corrupting recorded truth commands cannot change
        # the counterfactual base wrench.  The old implementation failed this
        # because it copied truth.commanded_* directly into the weak model.
        synthetic.truth.commanded_thrust[:] = 26.0
        synthetic.truth.commanded_gimbal_angle[:] = 1.0
        actual = oracle_effective_residual_wrench(synthetic)
        np.testing.assert_array_equal(actual, expected)

    def test_static_residual_trajectory_and_path_members_stay_aligned(self):
        posterior = self.result.weak_posterior
        member = 13
        decoded = posterior.residual_wrench_ensemble[member]
        np.testing.assert_array_equal(
            decoded,
            self.result.wrench_process.decode(
                posterior.innovation_ensemble[member]
            ),
        )
        translation, rotation = correction_transform_path(
            self.result.synthetic.nominal.position,
            self.result.synthetic.nominal.orientation_xyzw,
            posterior.trajectory_ensemble[member].position,
            posterior.trajectory_ensemble[member].orientation_xyzw,
        )
        np.testing.assert_array_equal(
            translation, posterior.correction_translation[member]
        )
        np.testing.assert_array_equal(
            rotation, posterior.correction_rotation_vector[member]
        )
        self.assertEqual(
            self.result.wrench_process.innovation_dimension, decoded.size
        )

    def test_phase3_artifact_is_pickle_free_and_member_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = save_phase3_experiment(
                str(Path(directory) / "phase3.npz"), self.result
            )
            with np.load(str(destination), allow_pickle=False) as artifact:
                self.assertEqual(
                    str(artifact["schema"][0]),
                    "grape-weak-constraint/phase3",
                )
                self.assertEqual(
                    artifact["weak_control_ensemble"].shape[0], 88
                )
                self.assertEqual(
                    artifact["weak_residual_wrench_ensemble"].shape[0], 88
                )
                self.assertEqual(
                    artifact["weak_correction_translation"].shape[0], 88
                )
                member = 13
                translation, rotation = correction_transform_path(
                    artifact["nominal_position"],
                    artifact["nominal_orientation_xyzw"],
                    artifact["weak_position"][member],
                    artifact["weak_orientation_xyzw"][member],
                )
                np.testing.assert_array_equal(
                    translation,
                    artifact["weak_correction_translation"][member],
                )
                np.testing.assert_array_equal(
                    rotation,
                    artifact["weak_correction_rotation_vector"][member],
                )
                for field in (
                    "position",
                    "orientation_xyzw",
                    "linear_velocity",
                    "angular_velocity",
                    "controller_integral",
                    "commanded_thrust",
                    "commanded_gimbal_angle",
                    "actuator_thrust",
                    "actuator_gimbal_angle",
                    "body_wrench",
                ):
                    self.assertIn("nominal_{}".format(field), artifact.files)
                    self.assertIn("truth_{}".format(field), artifact.files)
                self.assertNotIn("weights", artifact.files)
                for field in (
                    "matched_truth_position",
                    "matched_observations_position",
                    "matched_strong_parameter_coordinates",
                    "matched_strong_position",
                    "matched_strong_static_bias",
                    "counterfactual_commanded_thrust",
                    "counterfactual_actuator_gimbal_angle",
                    "counterfactual_interval_midpoint_thrust",
                ):
                    self.assertIn(field, artifact.files)


if __name__ == "__main__":
    unittest.main()
