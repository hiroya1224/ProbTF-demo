"""Small end-to-end Bayesian counterfactual vertical slice.

This test intentionally connects the public contracts instead of replacing
any stage with a hand-made posterior:

synthetic mocap/IMU -> trajectory posterior -> effective-response fit ->
tempered SMC -> closed-loop counterfactual -> support classification.
"""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from grape_param_estim.controller_replay import (
    ControllerParameters,
    PidLimits,
)
from grape_param_estim.counterfactual import (
    SUPPORTED,
    UNSUPPORTED,
    ClosedLoopCounterfactualEvaluator,
    CounterfactualCandidate,
    CounterfactualConfig,
    InitialStateSample,
    SupportReference,
    TargetTrajectory,
    TargetTube,
)
from grape_param_estim.effective_response import (
    EffectiveResponseFitConfig,
    EffectiveResponseParameters,
    EffectiveResponsePosterior,
    LowDimensionalEffectiveResponse,
    ResponseState,
    TrajectoryTransitionBatch,
    delayed_command,
    fit_effective_response,
)
from grape_param_estim.inference import (
    BoundedLogitTransform,
    BoxUniformPrior,
    TemperedResampleMoveSmc,
    TemperedSmcConfig,
    marginalize_trajectory_log_likelihood,
)
from grape_param_estim.state_smoother import (
    SmootherConfig,
    TrajectoryObservations,
    smooth_trajectory,
)


def _commands(timestamps):
    times = np.asarray(timestamps, dtype=float)
    return 0.25 * np.column_stack(
        (
            np.sin(1.1 * times) + 0.3 * np.sin(2.7 * times),
            np.cos(0.9 * times) + 0.2 * np.sin(2.1 * times),
            np.sin(1.5 * times + 0.3),
            np.cos(1.3 * times + 0.2),
            np.sin(1.7 * times + 0.6),
            np.cos(1.9 * times + 0.4),
        )
    )


def _truth_response():
    effectiveness = np.diag([0.8, 0.9, 1.0, 0.7, 0.85, 0.95])
    effectiveness[3, 4] = 0.08
    effectiveness[4, 3] = -0.06
    return EffectiveResponseParameters(
        effectiveness=effectiveness,
        time_constant_s=np.full(6, 0.08),
        delay_s=np.full(6, 0.05),
        damping=np.array([0.10, 0.12, 0.08, 0.16, 0.14, 0.12]),
        bias=np.zeros(6),
    )


def _synthetic_sensor_episode(future_corruption_after=None):
    """Generate mutually consistent low-angle mocap and IMU evidence."""

    rng = np.random.default_rng(109)
    times = np.arange(0.0, 3.0 + 1.0e-9, 0.05)
    commands = _commands(times)
    response = LowDimensionalEffectiveResponse()
    state = ResponseState(np.zeros(6), np.zeros(6), commands[0])
    positions = np.empty_like(commands)
    velocities = np.empty_like(commands)
    accelerations = np.empty_like(commands)
    positions[0] = state.generalized_position
    velocities[0] = state.generalized_velocity
    for index in range(times.size - 1):
        transition = response.transition(
            state,
            times[: index + 1],
            commands[: index + 1],
            times[index],
            times[index + 1] - times[index],
            _truth_response(),
        )
        accelerations[index] = transition.generalized_acceleration
        state = transition.state
        positions[index + 1] = state.generalized_position
        velocities[index + 1] = state.generalized_velocity
    accelerations[-1] = accelerations[-2]

    quaternion = Rotation.from_rotvec(positions[:, 3:]).as_quat()
    gravity = np.array([0.0, 0.0, -9.80665])
    accelerometer = np.empty((times.size, 3))
    for index in range(times.size):
        accelerometer[index] = Rotation.from_quat(
            quaternion[index]
        ).inv().apply(accelerations[index, :3] - gravity)
    accelerometer += rng.normal(0.0, 0.006, accelerometer.shape)
    gyro = velocities[:, 3:] + rng.normal(0.0, 0.001, (times.size, 3))

    mocap_indices = np.arange(0, times.size, 2)
    mocap_times = times[mocap_indices]
    mocap_position = positions[mocap_indices, :3] + rng.normal(
        0.0, 0.002, (mocap_indices.size, 3)
    )
    mocap_quaternion = (
        Rotation.from_quat(quaternion[mocap_indices])
        * Rotation.from_rotvec(
            rng.normal(0.0, np.deg2rad(0.1), (mocap_indices.size, 3))
        )
    ).as_quat()
    mocap_valid = ~((mocap_times > 1.1) & (mocap_times < 1.45))

    if future_corruption_after is not None:
        cutoff = float(future_corruption_after)
        mocap_position[mocap_times > cutoff] += 100.0
        accelerometer[times > cutoff, 0] += 50.0
        gyro[times > cutoff, 2] += 20.0

    observations = TrajectoryObservations(
        mocap_times=mocap_times,
        mocap_positions_world=mocap_position,
        mocap_quaternions_xyzw=mocap_quaternion,
        imu_times=times,
        accelerometer_body=accelerometer,
        gyro_body=gyro,
        mocap_valid_mask=mocap_valid,
    )
    return observations, times, commands


def _smoother_config(sample_count=4):
    return SmootherConfig(
        mocap_position_sigma=0.004,
        mocap_orientation_sigma=np.deg2rad(0.25),
        accelerometer_noise_sigma=0.03,
        gyro_noise_sigma=np.deg2rad(0.15),
        accelerometer_bias_random_walk_sigma=0.005,
        gyro_bias_random_walk_sigma=np.deg2rad(0.01),
        mocap_nis_gate=1000.0,
        trajectory_sample_count=sample_count,
        seed=17,
    )


def _trajectory_batches(posterior):
    commands = _commands(posterior.timestamps)
    batches = []
    for sample_index, sample_id in enumerate(posterior.sample_ids):
        rotation_vector = Rotation.from_quat(
            posterior.sample_quaternion_xyzw[sample_index]
        ).as_rotvec()
        generalized_position = np.column_stack(
            (
                posterior.sample_position_world[sample_index],
                rotation_vector,
            )
        )
        generalized_velocity = np.column_stack(
            (
                posterior.sample_velocity_world[sample_index],
                posterior.sample_angular_velocity_body[sample_index],
            )
        )
        batches.append(
            TrajectoryTransitionBatch(
                timestamps=posterior.timestamps,
                generalized_position=generalized_position,
                generalized_velocity=generalized_velocity,
                commands=commands,
                episode_id="synthetic-flight",
                trajectory_sample_id=int(sample_id),
                trajectory_weight=float(
                    posterior.sample_weights[sample_index]
                ),
                state_source="trajectory_posterior",
            )
        )
    return tuple(batches)


def _smc_scale_likelihood(fitted_mean, batches):
    """Marginalized likelihood for a scale on fitted effectiveness."""

    trajectory_weights = np.asarray(
        [item.trajectory_weight for item in batches], dtype=float
    )
    trajectory_weights /= np.sum(trajectory_weights)
    base_accelerations = []
    offsets = []
    observed_accelerations = []
    for batch in batches:
        actuator = np.array(batch.commands[0], copy=True)
        base = []
        offset = []
        for index, delta in enumerate(np.diff(batch.timestamps)):
            delayed = delayed_command(
                batch.timestamps[: index + 1],
                batch.commands[: index + 1],
                batch.timestamps[index],
                fitted_mean.delay_s,
            )
            decay = np.exp(-float(delta) / fitted_mean.time_constant_s)
            actuator = decay * actuator + (1.0 - decay) * delayed
            base.append(fitted_mean.effectiveness @ actuator)
            offset.append(
                -fitted_mean.damping
                * batch.generalized_velocity[index]
                + fitted_mean.bias
            )
        base_accelerations.append(np.asarray(base))
        offsets.append(np.asarray(offset))
        observed_accelerations.append(
            np.diff(batch.generalized_velocity, axis=0)
            / np.diff(batch.timestamps)[:, None]
        )

    def log_likelihood(particles):
        values = np.asarray(particles, dtype=float)
        conditional = np.empty((values.shape[0], len(batches)))
        for sample_index in range(len(batches)):
            prediction = (
                values[:, 0, None, None]
                * base_accelerations[sample_index][None, :, :]
                + offsets[sample_index][None, :, :]
            )
            residual = (
                observed_accelerations[sample_index][None, :, :]
                - prediction
            ) / 0.25
            conditional[:, sample_index] = -0.5 * np.sum(
                residual * residual, axis=(1, 2)
            )
        return marginalize_trajectory_log_likelihood(
            conditional, trajectory_weights
        )

    return log_likelihood


def _controller(p_gain=1.5):
    return ControllerParameters(
        p_gain=np.full(6, p_gain),
        i_gain=np.zeros(6),
        d_gain=np.full(6, 0.5),
        limits=PidLimits(
            output=20.0,
            p_term=20.0,
            i_term=5.0,
            d_term=20.0,
            p_error=20.0,
            i_state=5.0,
            d_error=20.0,
        ),
        controller_mass=1.0,
        controller_inertia_diagonal=np.ones(3),
        allocation_scale=np.ones(6),
        thrust_scale=1.0,
    )


def _response_posterior_from_smc(fit, smc, count=6):
    fitted_mean = fit.mean_parameters()
    rng = np.random.default_rng(51)
    selected = rng.choice(
        smc.particles.shape[0],
        size=int(count),
        replace=True,
        p=smc.weights,
    )
    samples = tuple(
        EffectiveResponseParameters(
            effectiveness=(
                fitted_mean.effectiveness
                * float(smc.particles[index, 0])
            ),
            time_constant_s=fitted_mean.time_constant_s,
            delay_s=fitted_mean.delay_s,
            damping=fitted_mean.damping,
            bias=fitted_mean.bias,
        )
        for index in selected
    )
    return EffectiveResponsePosterior(
        samples=samples,
        weights=np.full(len(samples), 1.0 / len(samples)),
        grid_delay_s=fit.grid_delay_s,
        grid_time_constant_s=fit.grid_time_constant_s,
        grid_weights=fit.grid_weights,
        identifiability=fit.identifiability,
        log_evidence=float(smc.log_evidence),
        approximation=(
            "effective_response_fit_plus_tempered_smc_scale;"
            "trajectory_likelihood_marginalized"
        ),
        source_sample_ids=fit.source_sample_ids,
    )


class BayesianPipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        observations, _, _ = _synthetic_sensor_episode()
        cls.trajectory = smooth_trajectory(
            observations, _smoother_config(sample_count=4)
        )
        cls.batches = _trajectory_batches(cls.trajectory)
        cls.fit = fit_effective_response(
            cls.batches,
            EffectiveResponseFitConfig(
                delay_grid_s=[0.0, 0.05],
                time_constant_grid_s=[0.05, 0.10],
                prior_scale=5.0,
                residual_sigma_floor=0.02,
                posterior_sample_count=16,
                seed=23,
            ),
        )
        prior = BoxUniformPrior([0.5], [1.5])
        transform = BoundedLogitTransform(prior.lower, prior.upper)
        cls.smc = TemperedResampleMoveSmc(
            prior,
            transform,
            TemperedSmcConfig(
                particle_count=128,
                target_ess_fraction=0.7,
                resample_ess_fraction=0.4,
                mcmc_steps=1,
                proposal_scale=0.6,
                seed=29,
            ),
        ).run(_smc_scale_likelihood(cls.fit.mean_parameters(), cls.batches))
        cls.response_posterior = _response_posterior_from_smc(
            cls.fit, cls.smc
        )

    def _initial_samples(self):
        output = []
        for index in range(2):
            position = np.concatenate(
                (
                    self.trajectory.sample_position_world[index, 0],
                    Rotation.from_quat(
                        self.trajectory.sample_quaternion_xyzw[index, 0]
                    ).as_rotvec(),
                )
            )
            velocity = np.concatenate(
                (
                    self.trajectory.sample_velocity_world[index, 0],
                    self.trajectory.sample_angular_velocity_body[index, 0],
                )
            )
            output.append(
                InitialStateSample(
                    sample_id=int(self.trajectory.sample_ids[index]),
                    state=ResponseState(position, velocity, np.zeros(6)),
                    weight=0.5,
                )
            )
        return output

    def _target(self):
        timestamps = np.arange(0.0, 0.8 + 1.0e-9, 0.05)
        zeros = np.zeros((timestamps.size, 6))
        return TargetTrajectory(timestamps, zeros, zeros, zeros)

    def _support_reference(self, candidate):
        vector = candidate.vector()
        observed_candidates = np.vstack(
            (vector, vector + 1.0e-4, vector - 1.0e-4)
        )
        observed_points = np.vstack(
            [
                np.column_stack(
                    (
                        batch.generalized_position,
                        batch.generalized_velocity,
                        batch.commands,
                    )
                )
                for batch in self.batches
            ]
        )
        return SupportReference(
            observed_candidate_vectors=observed_candidates,
            observed_state_action_points=observed_points,
            candidate_scale=np.maximum(np.abs(vector) * 0.2, 0.2),
            state_action_scale=np.maximum(np.ptp(observed_points, axis=0), 1.0),
            supported_distance=10.0,
            unsupported_distance=20.0,
            minimum_importance_ess=1.0,
            maximum_predictive_std=100.0,
        )

    def test_sensor_to_supported_counterfactual_pipeline(self):
        self.assertTrue(self.trajectory.is_smoothed)
        self.assertEqual(self.trajectory.sample_count, 4)
        self.assertEqual(
            self.fit.source_sample_ids,
            tuple(int(item) for item in self.trajectory.sample_ids),
        )
        self.assertAlmostEqual(
            self.smc.stages[-1].inverse_temperature, 1.0
        )
        self.assertTrue(np.all(np.isfinite(self.smc.particles)))
        self.assertAlmostEqual(float(np.sum(self.smc.weights)), 1.0)
        self.assertIn(
            "trajectory_likelihood_marginalized",
            self.response_posterior.approximation,
        )

        candidate = CounterfactualCandidate("within-support", _controller())
        evaluator = ClosedLoopCounterfactualEvaluator(
            self._support_reference(candidate)
        )
        result = evaluator.evaluate(
            candidate,
            self._target(),
            TargetTube(
                position_tolerance=np.full(6, 1000.0),
                velocity_tolerance=np.full(6, 1000.0),
                allowed_outside_duration_s=0.8,
            ),
            self.response_posterior,
            self._initial_samples(),
            CounterfactualConfig(
                process_noise_sigma=np.zeros(6),
                seed=31,
                source_bag_hashes=("a" * 64,),
                normalized_dataset_hashes=("b" * 64,),
                source_commit="synthetic-test",
            ),
        )
        self.assertEqual(result.support.label, SUPPORTED)
        self.assertTrue(result.recommendable)
        self.assertEqual(result.success_probability, 1.0)
        self.assertEqual(len(result.rollouts), 12)
        self.assertEqual(
            result.provenance["controller_backend"],
            "python_vector_pid_surrogate/v1",
        )

    def test_future_prefix_evidence_is_ignored_and_misdeclared_cutoff_is_rejected(self):
        cutoff = 1.5
        baseline, _, _ = _synthetic_sensor_episode()
        corrupted, _, _ = _synthetic_sensor_episode(
            future_corruption_after=cutoff
        )
        first = smooth_trajectory(
            baseline,
            _smoother_config(sample_count=2),
            online_prefix=True,
            cutoff=cutoff,
        )
        second = smooth_trajectory(
            corrupted,
            _smoother_config(sample_count=2),
            online_prefix=True,
            cutoff=cutoff,
        )
        self.assertFalse(first.is_smoothed)
        self.assertLessEqual(float(first.timestamps[-1]), cutoff)
        np.testing.assert_array_equal(first.timestamps, second.timestamps)
        np.testing.assert_array_equal(
            first.sample_position_world, second.sample_position_world
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            CounterfactualConfig(
                process_noise_sigma=np.zeros(6),
                analysis_mode="online_prefix",
                prefix_cutoff=cutoff,
                inference_data_end_time=cutoff + 0.01,
            )

    def test_raw_mocap_derivative_cannot_be_added_as_second_likelihood(self):
        observations, _, _ = _synthetic_sensor_episode()
        raw_position = np.column_stack(
            (
                observations.mocap_positions_world,
                Rotation.from_quat(
                    observations.mocap_quaternions_xyzw
                ).as_rotvec(),
            )
        )
        raw_derivative = np.zeros_like(raw_position)
        raw_derivative[:, :3] = np.gradient(
            observations.mocap_positions_world,
            observations.mocap_times,
            axis=0,
        )
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            TrajectoryTransitionBatch(
                timestamps=observations.mocap_times,
                generalized_position=raw_position,
                generalized_velocity=raw_derivative,
                commands=_commands(observations.mocap_times),
                episode_id="invalid-double-likelihood",
                trajectory_sample_id=999,
                trajectory_weight=1.0,
                state_source="trajectory_posterior",
                raw_mocap_derivative=True,
            )

    def test_high_probability_support_extrapolation_is_not_recommendable(self):
        factual = CounterfactualCandidate("factual", _controller())
        far = CounterfactualCandidate("far-outside-support", _controller(50.0))
        evaluator = ClosedLoopCounterfactualEvaluator(
            self._support_reference(factual)
        )
        result = evaluator.evaluate(
            far,
            self._target(),
            TargetTube(
                position_tolerance=np.full(6, 1.0e6),
                velocity_tolerance=np.full(6, 1.0e6),
                allowed_outside_duration_s=0.8,
            ),
            self.response_posterior,
            self._initial_samples(),
            CounterfactualConfig(
                process_noise_sigma=np.zeros(6),
                seed=37,
            ),
        )
        self.assertEqual(result.success_probability, 1.0)
        self.assertEqual(result.support.label, UNSUPPORTED)
        self.assertIn(
            "candidate_parameter_distance", result.support.reasons
        )
        self.assertFalse(result.recommendable)


if __name__ == "__main__":
    unittest.main()
