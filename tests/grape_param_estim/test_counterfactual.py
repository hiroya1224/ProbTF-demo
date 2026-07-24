import unittest

import numpy as np

from grape_param_estim.controller_replay import (
    ControllerParameters,
    PidLimits,
)
from grape_param_estim.counterfactual import (
    EXTRAPOLATIVE,
    SUPPORTED,
    UNSUPPORTED,
    ClosedLoopCounterfactualEvaluator,
    CounterfactualCandidate,
    CounterfactualConfig,
    InitialStateSample,
    SupportReference,
    TargetTrajectory,
    TargetTube,
    classify_support,
    connected_candidate_regions,
    evaluate_target_tube,
)
from grape_param_estim.effective_response import (
    EffectiveResponseParameters,
    EffectiveResponsePosterior,
    IdentifiabilityReport,
    ResponseState,
)


def controller(delay_compensation=0.0, p_gain=3.0):
    return ControllerParameters(
        p_gain=np.full(6, p_gain),
        i_gain=np.zeros(6),
        d_gain=np.full(6, 1.2),
        limits=PidLimits(
            output=50.0,
            p_term=50.0,
            i_term=10.0,
            d_term=50.0,
            p_error=10.0,
            i_state=10.0,
            d_error=10.0,
        ),
        controller_mass=1.0,
        controller_inertia_diagonal=np.ones(3),
        allocation_scale=np.ones(6),
        thrust_scale=1.0,
        delay_compensation_s=delay_compensation,
    )


def response_posterior():
    samples = []
    for delay in (0.12, 0.15, 0.18):
        samples.append(
            EffectiveResponseParameters(
                effectiveness=np.eye(6),
                time_constant_s=np.full(6, 0.05),
                delay_s=np.full(6, delay),
                damping=np.full(6, 0.05),
                bias=np.zeros(6),
            )
        )
    identifiability = IdentifiabilityReport(
        design_rank=48,
        parameter_count=48,
        condition_number=2.0,
        per_axis_design_rank=np.full(6, 8),
        per_axis_condition_number=np.full(6, 2.0),
        singular_values=np.ones(48),
        null_directions=np.empty((0, 48)),
        posterior_maximum_absolute_correlation=0.2,
        identifiable=True,
        scope="conditional_linear_coefficients_given_delay_and_lag",
    )
    return EffectiveResponsePosterior(
        samples=tuple(samples),
        weights=np.full(3, 1.0 / 3.0),
        grid_delay_s=np.array([0.12, 0.15, 0.18]),
        grid_time_constant_s=np.full(3, 0.05),
        grid_weights=np.full(3, 1.0 / 3.0),
        identifiability=identifiability,
        log_evidence=0.0,
        approximation="test_posterior",
        source_sample_ids=(0,),
    )


def target_trajectory():
    times = np.arange(0.0, 5.0 + 1.0e-9, 0.05)
    position = np.zeros((times.size, 6))
    velocity = np.zeros_like(position)
    acceleration = np.zeros_like(position)
    frequency = 1.2
    position[:, 0] = 0.4 * np.sin(frequency * times)
    velocity[:, 0] = 0.4 * frequency * np.cos(frequency * times)
    acceleration[:, 0] = -0.4 * frequency ** 2 * np.sin(frequency * times)
    return TargetTrajectory(times, position, velocity, acceleration)


def support_reference(candidates):
    vectors = np.asarray([item.vector() for item in candidates])
    # Multiple factual settings keep importance ESS meaningful.
    observed = np.vstack((vectors, vectors + 1.0e-3, vectors - 1.0e-3))
    state_action = np.zeros((8, 18))
    return SupportReference(
        observed_candidate_vectors=observed,
        observed_state_action_points=state_action,
        candidate_scale=np.full(vectors.shape[1], 0.5),
        state_action_scale=np.full(18, 5.0),
        supported_distance=3.0,
        unsupported_distance=8.0,
        minimum_importance_ess=2.0,
        maximum_predictive_std=2.0,
    )


class CounterfactualTests(unittest.TestCase):
    def test_target_tube_checks_tracking_saturation_ground_and_tilt(self):
        times = np.arange(0.0, 0.51, 0.1)
        zeros = np.zeros((times.size, 6))
        target = TargetTrajectory(times, zeros, zeros, zeros)
        actual = zeros.copy()
        actual[:, 2] = 0.4
        actual[3, 0] = 0.5
        actual[4, 3] = 0.7
        saturation = np.zeros_like(actual, dtype=bool)
        saturation[1:4, 0] = True
        result = evaluate_target_tube(
            target,
            TargetTube(
                position_tolerance=np.full(6, 0.2),
                velocity_tolerance=np.full(6, 0.3),
                maximum_continuous_saturation_s=0.15,
                minimum_height_m=0.5,
                maximum_tilt_rad=0.5,
            ),
            actual,
            zeros,
            saturation,
        )
        self.assertFalse(result.success)
        self.assertIn("position_tube", result.violations)
        self.assertIn("saturation_duration", result.violations)
        self.assertIn("ground_or_contact", result.violations)
        self.assertIn("tilt_safety", result.violations)

    def test_short_pointwise_excursion_can_use_explicit_duration_allowance(self):
        times = np.arange(0.0, 0.51, 0.1)
        zeros = np.zeros((times.size, 6))
        target = TargetTrajectory(times, zeros, zeros, zeros)
        actual = zeros.copy()
        actual[2, 0] = 0.25
        result = evaluate_target_tube(
            target,
            TargetTube(
                position_tolerance=np.full(6, 0.2),
                velocity_tolerance=np.ones(6),
                allowed_outside_duration_s=0.11,
            ),
            actual,
            zeros,
            np.zeros_like(actual, dtype=bool),
        )
        self.assertTrue(result.success)
        self.assertIn("position_tube", result.diagnostic_exceedances)
        self.assertNotIn("position_tube", result.violations)

    def test_support_labels_use_parameters_state_action_ess_and_uncertainty(self):
        reference = SupportReference(
            observed_candidate_vectors=np.array(
                [[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0]]
            ),
            observed_state_action_points=np.array(
                [[0.0, 0.0], [0.2, 0.0], [-0.2, 0.0]]
            ),
            candidate_scale=np.ones(2),
            state_action_scale=np.ones(2),
            supported_distance=1.0,
            unsupported_distance=4.0,
            minimum_importance_ess=1.0,
            maximum_predictive_std=1.0,
        )
        supported = classify_support(
            np.array([0.0, 0.0]),
            np.array([[0.1, 0.0], [0.0, 0.1]]),
            0.1,
            reference,
        )
        self.assertEqual(supported.label, SUPPORTED)
        extrapolative = classify_support(
            np.array([2.0, 0.0]),
            np.array([[0.0, 0.0]]),
            0.1,
            reference,
        )
        self.assertEqual(extrapolative.label, EXTRAPOLATIVE)
        unsupported = classify_support(
            np.array([10.0, 0.0]),
            np.array([[10.0, 0.0]]),
            2.0,
            reference,
        )
        self.assertEqual(unsupported.label, UNSUPPORTED)
        self.assertIn("candidate_parameter_distance", unsupported.reasons)
        self.assertIn("state_action_distance", unsupported.reasons)
        self.assertIn("posterior_predictive_uncertainty", unsupported.reasons)
        nonfinite = classify_support(
            np.array([0.0, 0.0]),
            np.array([[0.0, 0.0]]),
            float("nan"),
            reference,
        )
        self.assertEqual(nonfinite.label, UNSUPPORTED)

    def test_closed_loop_recomputes_commands_and_delay_compensation_changes_rollout(self):
        target = target_trajectory()
        baseline = CounterfactualCandidate("baseline", controller(0.0))
        compensated = CounterfactualCandidate("compensated", controller(0.15))
        reference = support_reference((baseline, compensated))
        evaluator = ClosedLoopCounterfactualEvaluator(reference)
        initial = InitialStateSample(
            sample_id=7,
            state=ResponseState(
                target.position[0],
                target.velocity[0],
                np.zeros(6),
            ),
            weight=1.0,
        )
        tube = TargetTube(
            position_tolerance=np.full(6, 0.25),
            velocity_tolerance=np.full(6, 0.5),
            evaluation_start_offset_s=0.5,
            maximum_continuous_saturation_s=0.5,
            allowed_outside_duration_s=1.5,
        )
        config = CounterfactualConfig(
            process_noise_sigma=np.full(6, 0.002),
            process_noise_replicates=3,
            seed=22,
            source_bag_hashes=("a" * 64,),
            normalized_dataset_hashes=("b" * 64,),
        )
        first = evaluator.evaluate(
            baseline,
            target,
            tube,
            response_posterior(),
            [initial],
            config,
        )
        second = evaluator.evaluate(
            compensated,
            target,
            tube,
            response_posterior(),
            [initial],
            config,
        )
        repeated = evaluator.evaluate(
            baseline,
            target,
            tube,
            response_posterior(),
            [initial],
            config,
        )
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(first.run_id, repeated.run_id)
        self.assertEqual(first.support.label, SUPPORTED)
        self.assertEqual(second.support.label, SUPPORTED)
        self.assertTrue(first.recommendable)
        self.assertGreater(first.effective_rollout_sample_size, 1.0)
        self.assertGreaterEqual(first.credible_upper, first.credible_lower)
        self.assertGreaterEqual(first.success_probability, 0.0)
        self.assertLessEqual(first.success_probability, 1.0)
        baseline_position = first.rollouts[0].position[:, 0]
        compensated_position = second.rollouts[0].position[:, 0]
        self.assertFalse(np.allclose(baseline_position, compensated_position))
        self.assertFalse(
            np.allclose(
                first.rollouts[0].command,
                second.rollouts[0].command,
            )
        )
        np.testing.assert_allclose(
            first.rollouts[0].position,
            repeated.rollouts[0].position,
        )
        regions = connected_candidate_regions([first, second], gamma=0.0)
        self.assertEqual(regions, (("baseline", "compensated"),))

    def test_online_prefix_provenance_requires_cutoff(self):
        with self.assertRaisesRegex(ValueError, "prefix_cutoff"):
            CounterfactualConfig(
                process_noise_sigma=np.zeros(6),
                analysis_mode="online_prefix",
            )
        valid = CounterfactualConfig(
            process_noise_sigma=np.zeros(6),
            analysis_mode="online_prefix",
            prefix_cutoff=2.0,
            inference_data_end_time=2.0,
        )
        self.assertEqual(valid.prefix_cutoff, 2.0)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            CounterfactualConfig(
                process_noise_sigma=np.zeros(6),
                analysis_mode="online_prefix",
                prefix_cutoff=2.0,
                inference_data_end_time=2.1,
            )

    def test_online_prefix_evaluator_rejects_unstamped_or_future_initial_state(self):
        target = target_trajectory()
        # Rebase target so it begins at the causal cutoff.
        shifted = TargetTrajectory(
            target.timestamps + 2.0,
            target.position,
            target.velocity,
            target.acceleration,
        )
        candidate = CounterfactualCandidate("causal", controller())
        evaluator = ClosedLoopCounterfactualEvaluator(
            support_reference((candidate,))
        )
        unstamped = InitialStateSample(
            1,
            ResponseState(
                shifted.position[0], shifted.velocity[0], np.zeros(6)
            ),
            1.0,
        )
        with self.assertRaisesRegex(ValueError, "stamped at prefix_cutoff"):
            evaluator.evaluate(
                candidate,
                shifted,
                TargetTube(np.ones(6), np.ones(6)),
                response_posterior(),
                [unstamped],
                CounterfactualConfig(
                    process_noise_sigma=np.zeros(6),
                    analysis_mode="online_prefix",
                    prefix_cutoff=2.0,
                    inference_data_end_time=2.0,
                ),
            )

    def test_rollout_arrays_are_immutable(self):
        target = target_trajectory()
        candidate = CounterfactualCandidate("immutable", controller())
        evaluator = ClosedLoopCounterfactualEvaluator(
            support_reference((candidate,))
        )
        result = evaluator.evaluate(
            candidate,
            target,
            TargetTube(
                np.ones(6),
                np.ones(6),
                allowed_outside_duration_s=5.0,
            ),
            response_posterior(),
            [
                InitialStateSample(
                    0,
                    ResponseState(
                        target.position[0],
                        target.velocity[0],
                        np.zeros(6),
                    ),
                    1.0,
                )
            ],
            CounterfactualConfig(process_noise_sigma=np.zeros(6)),
        )
        with self.assertRaises(ValueError):
            result.rollouts[0].position[0, 0] = 99.0


if __name__ == "__main__":
    unittest.main()
