import unittest

import numpy as np

from grape_param_estim.effective_response import (
    EffectiveResponseFitConfig,
    EffectiveResponseParameters,
    LowDimensionalEffectiveResponse,
    ResponseState,
    TrajectoryTransitionBatch,
    fit_effective_response,
    fit_hierarchical_effective_response,
    trajectory_mixture_log_likelihood,
)


def commands_for(times):
    return np.column_stack(
        (
            np.sin(0.7 * times) + 0.25 * np.sin(2.1 * times),
            np.cos(0.9 * times) + 0.15 * np.sin(2.6 * times),
            np.sin(1.1 * times + 0.4),
            np.cos(1.3 * times + 0.7),
            np.sin(1.7 * times + 1.1),
            np.cos(2.0 * times + 0.2),
        )
    )


def truth_parameters(scale=1.0):
    effectiveness = np.diag([0.85, 0.95, 1.10, 0.75, 0.90, 1.05]) * scale
    effectiveness[3, 4] = 0.12 * scale
    effectiveness[4, 3] = -0.08 * scale
    return EffectiveResponseParameters(
        effectiveness=effectiveness,
        time_constant_s=np.full(6, 0.08),
        delay_s=np.full(6, 0.04),
        damping=np.array([0.10, 0.12, 0.08, 0.18, 0.16, 0.14]),
        bias=np.array([0.02, -0.01, 0.03, 0.01, -0.02, 0.0]),
    )


def simulate_batch(episode_id="episode-a", scale=1.0, sample_id=0, weight=1.0):
    times = np.arange(0.0, 10.0 + 1.0e-9, 0.02)
    commands = commands_for(times)
    model = LowDimensionalEffectiveResponse()
    state = ResponseState(np.zeros(6), np.zeros(6), commands[0])
    positions = np.empty_like(commands)
    velocities = np.empty_like(commands)
    positions[0] = state.generalized_position
    velocities[0] = state.generalized_velocity
    parameters = truth_parameters(scale)
    for index in range(times.size - 1):
        result = model.transition(
            state,
            times[: index + 1],
            commands[: index + 1],
            times[index],
            times[index + 1] - times[index],
            parameters,
        )
        state = result.state
        positions[index + 1] = state.generalized_position
        velocities[index + 1] = state.generalized_velocity
    return TrajectoryTransitionBatch(
        timestamps=times,
        generalized_position=positions,
        generalized_velocity=velocities,
        commands=commands,
        episode_id=episode_id,
        trajectory_sample_id=sample_id,
        trajectory_weight=weight,
        state_source="synthetic_known_truth",
    )


class EffectiveResponseTests(unittest.TestCase):
    def test_transition_and_heavy_tail_likelihood_are_finite(self):
        model = LowDimensionalEffectiveResponse()
        state = ResponseState(np.zeros(6), np.zeros(6), np.zeros(6))
        times = np.array([0.0, 0.1])
        commands = np.vstack((np.ones(6), np.ones(6)))
        result = model.transition(
            state, times, commands, 0.1, 0.1, truth_parameters()
        )
        self.assertTrue(np.all(np.isfinite(result.generalized_acceleration)))
        outlier = ResponseState(
            result.state.generalized_position + 10.0,
            result.state.generalized_velocity - 5.0,
            result.state.actuator_state,
        )
        log_likelihood = model.transition_log_likelihood(
            result.state,
            outlier,
            position_sigma=np.full(6, 0.02),
            velocity_sigma=np.full(6, 0.05),
            degrees_of_freedom=5.0,
        )
        self.assertTrue(np.isfinite(log_likelihood))

    def test_fit_recovers_effectiveness_delay_and_lag(self):
        batch = simulate_batch()
        config = EffectiveResponseFitConfig(
            delay_grid_s=[0.0, 0.04, 0.08],
            time_constant_grid_s=[0.04, 0.08, 0.16],
            ridge=1.0e-5,
            residual_sigma_floor=1.0e-5,
            posterior_sample_count=160,
            seed=8,
        )
        posterior = fit_effective_response([batch], config)
        mean = posterior.mean_parameters()
        expected = truth_parameters()
        self.assertAlmostEqual(float(mean.delay_s[0]), 0.04, delta=0.025)
        self.assertAlmostEqual(float(mean.time_constant_s[0]), 0.08, delta=0.05)
        np.testing.assert_allclose(
            np.diag(mean.effectiveness),
            np.diag(expected.effectiveness),
            atol=0.08,
            rtol=0.0,
        )
        self.assertAlmostEqual(mean.effectiveness[3, 4], 0.12, delta=0.06)
        self.assertIn(batch.trajectory_sample_id, posterior.source_sample_ids)
        lower, upper = posterior.vector_credible_interval()
        self.assertEqual(lower.shape, expected.as_vector().shape)
        self.assertTrue(np.all(lower <= upper))

    def test_multiple_trajectory_samples_are_marginalized_by_weight(self):
        first = simulate_batch(sample_id=3, weight=0.8)
        second = simulate_batch(sample_id=9, weight=0.2)
        config = EffectiveResponseFitConfig(
            delay_grid_s=[0.04],
            time_constant_grid_s=[0.08],
            posterior_sample_count=40,
            seed=4,
        )
        posterior = fit_effective_response([first, second], config)
        self.assertEqual(posterior.source_sample_ids, (3, 9))
        self.assertEqual(posterior.weights.size, 40)
        self.assertIn(
            "episode_likelihood_is_logsumexp_over_mutually_exclusive_trajectories",
            posterior.fit_diagnostics,
        )
        self.assertIn("trajectory_mixture_marginal", posterior.approximation)

    def test_splitting_one_trajectory_weight_does_not_create_information(self):
        original = simulate_batch(sample_id=1, weight=1.0)
        duplicates = [
            TrajectoryTransitionBatch(
                timestamps=original.timestamps,
                generalized_position=original.generalized_position,
                generalized_velocity=original.generalized_velocity,
                commands=original.commands,
                episode_id=original.episode_id,
                trajectory_sample_id=index + 10,
                trajectory_weight=1.0 / 8.0,
                state_source=original.state_source,
            )
            for index in range(8)
        ]
        config = EffectiveResponseFitConfig(
            delay_grid_s=[0.04],
            time_constant_grid_s=[0.08],
            posterior_sample_count=64,
            seed=12,
        )
        single = fit_effective_response([original], config)
        repeated = fit_effective_response(duplicates, config)
        np.testing.assert_allclose(
            [item.as_vector() for item in single.samples],
            [item.as_vector() for item in repeated.samples],
            atol=1.0e-10,
            rtol=0.0,
        )
        self.assertAlmostEqual(single.log_evidence, repeated.log_evidence, places=8)
        self.assertAlmostEqual(
            trajectory_mixture_log_likelihood([original], truth_parameters()),
            trajectory_mixture_log_likelihood(duplicates, truth_parameters()),
            places=8,
        )

    def test_integrated_position_is_part_of_the_common_parameter_likelihood(self):
        original = simulate_batch(sample_id=1)
        inconsistent_position = np.array(
            original.generalized_position, copy=True
        )
        elapsed = original.timestamps - original.timestamps[0]
        inconsistent_position[:, 0] += 0.02 * elapsed * elapsed
        inconsistent = TrajectoryTransitionBatch(
            timestamps=original.timestamps,
            generalized_position=inconsistent_position,
            generalized_velocity=original.generalized_velocity,
            commands=original.commands,
            episode_id=original.episode_id,
            trajectory_sample_id=2,
            state_source="synthetic_known_truth",
        )
        config = EffectiveResponseFitConfig(
            delay_grid_s=[0.04],
            time_constant_grid_s=[0.08],
            position_sigma=0.005,
            velocity_sigma=0.05,
            posterior_sample_count=32,
            seed=3,
        )
        consistent_fit = fit_effective_response([original], config)
        inconsistent_fit = fit_effective_response([inconsistent], config)
        self.assertLess(
            inconsistent_fit.log_evidence, consistent_fit.log_evidence
        )
        self.assertGreater(
            abs(
                inconsistent_fit.mean_parameters().bias[0]
                - consistent_fit.mean_parameters().bias[0]
            ),
            1.0e-3,
        )

    def test_rank_deficiency_is_reported_for_unexciting_hover(self):
        times = np.arange(0.0, 3.0, 0.05)
        zeros = np.zeros((times.size, 6))
        constant = np.ones((times.size, 6))
        batch = TrajectoryTransitionBatch(
            times,
            zeros,
            zeros,
            constant,
            episode_id="hover",
            state_source="synthetic_known_truth",
        )
        posterior = fit_effective_response(
            [batch],
            EffectiveResponseFitConfig(
                delay_grid_s=[0.0],
                time_constant_grid_s=[0.1],
                posterior_sample_count=24,
            ),
        )
        self.assertFalse(posterior.identifiability.identifiable)
        self.assertLess(
            posterior.identifiability.design_rank,
            posterior.identifiability.parameter_count,
        )
        self.assertTrue(np.isinf(posterior.identifiability.condition_number))

    def test_raw_mocap_derivative_cannot_enter_main_likelihood(self):
        times = np.arange(4, dtype=float)
        values = np.zeros((4, 6))
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            TrajectoryTransitionBatch(
                times,
                values,
                values,
                values,
                episode_id="bad",
                raw_mocap_derivative=True,
            )

    def test_hierarchical_fit_keeps_episode_random_effects(self):
        first = simulate_batch("episode-a", scale=0.9)
        second = simulate_batch("episode-b", scale=1.1)
        result = fit_hierarchical_effective_response(
            [first, second],
            EffectiveResponseFitConfig(
                delay_grid_s=[0.04],
                time_constant_grid_s=[0.08],
                posterior_sample_count=32,
                seed=2,
            ),
            shrinkage_observations=50.0,
        )
        self.assertEqual(
            set(result.episode_parameter_means), {"episode-a", "episode-b"}
        )
        self.assertEqual(result.episode_random_effect_covariance.shape, (60, 60))
        first_gain = result.episode_parameter_means["episode-a"].effectiveness[0, 0]
        second_gain = result.episode_parameter_means["episode-b"].effectiveness[0, 0]
        self.assertLess(first_gain, second_gain)


if __name__ == "__main__":
    unittest.main()
