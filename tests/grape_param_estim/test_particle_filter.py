import unittest

import numpy as np

from grape_param_estim.dynamics import physical_parameter_mask, predict_wrench
from grape_param_estim.particle_filter import (
    ObservationBatch,
    ParameterBounds,
    ParticleFilterConfig,
    StaticParameterParticleFilter,
    effective_sample_size,
    systematic_resample,
)


TRUTH = np.array(
    [
        2.351597590812377,
        0.015275288306,
        0.001069474264,
        -0.047551249361,
        0.065000061483,
        -0.000000727899,
        0.000019015080,
        0.064952656340,
        0.000000059167,
        0.128992110664,
    ]
)
LOWER = np.array([0.5, -0.2, -0.2, -0.2, 0.01, -0.06, -0.06, 0.01, -0.06, 0.01])
UPPER = np.array([5.0, 0.2, 0.2, 0.15, 0.25, 0.06, 0.06, 0.25, 0.06, 0.25])


def exciting_dataset(count=100, seed=19):
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, 18.0, count)
    specific = np.column_stack(
        (
            2.0 * np.sin(0.73 * times) + 0.7 * np.cos(1.9 * times),
            1.7 * np.cos(0.91 * times) + 0.5 * np.sin(2.1 * times),
            9.80665 + 2.2 * np.sin(1.13 * times) + 0.4 * np.cos(2.4 * times),
        )
    )
    omega = np.column_stack(
        (
            1.5 * np.sin(0.63 * times) + 0.4 * np.sin(1.7 * times),
            1.4 * np.cos(0.83 * times) + 0.3 * np.cos(1.9 * times),
            1.7 * np.sin(1.27 * times) + 0.2 * np.cos(2.3 * times),
        )
    )
    alpha = np.gradient(omega, times, axis=0)
    wrench = np.array(
        [predict_wrench(TRUTH, specific[i], omega[i], alpha[i]) for i in range(count)]
    )
    sigma = np.array([0.30, 0.30, 0.30, 0.010, 0.010, 0.010])
    wrench += rng.normal(0.0, sigma, wrench.shape)
    return specific, omega, alpha, wrench, sigma


class ParticleFilterTests(unittest.TestCase):
    def test_systematic_resampling_and_ess(self):
        weights = np.array([0.7, 0.1, 0.1, 0.1])
        indices = systematic_resample(weights, np.random.default_rng(4))
        self.assertEqual(indices.shape, (4,))
        self.assertTrue(np.all((indices >= 0) & (indices < 4)))
        self.assertAlmostEqual(effective_sample_size(np.ones(4)), 4.0)
        self.assertLess(effective_sample_size(weights), 2.0)

    def test_bounded_initial_particles_are_physical_and_not_truth_centered(self):
        particle_filter = StaticParameterParticleFilter(
            ParameterBounds(LOWER, UPPER),
            ParticleFilterConfig(particle_count=256, mcmc_steps=0, seed=3),
        )
        self.assertTrue(np.all(physical_parameter_mask(particle_filter.particles)))
        initial_mean = np.mean(particle_filter.particles, axis=0)
        self.assertGreater(abs(initial_mean[0] - TRUTH[0]), 0.2)
        self.assertGreater(np.linalg.norm(initial_mean[1:4] - TRUTH[1:4]), 0.02)

    def test_joint_parameters_recover_from_exciting_data(self):
        specific, omega, alpha, wrench, sigma = exciting_dataset()
        particle_filter = StaticParameterParticleFilter(
            ParameterBounds(LOWER, UPPER),
            ParticleFilterConfig(
                particle_count=1024,
                resample_ess_fraction=0.4,
                tempering_ess_fraction=0.6,
                mcmc_steps=2,
                local_move_scale=0.65,
                prior_move_probability=0.02,
                seed=7,
            ),
        )
        saw_resampling = False
        for start in range(0, len(specific), 5):
            update = particle_filter.update(
                ObservationBatch(
                    specific[start : start + 5],
                    omega[start : start + 5],
                    alpha[start : start + 5],
                    wrench[start : start + 5],
                    sigma[:3],
                    sigma[3:],
                )
            )
            saw_resampling |= update.resampled
        estimate = particle_filter.posterior_summary().mean
        self.assertTrue(saw_resampling)
        self.assertLess(abs(estimate[0] - TRUTH[0]) / TRUTH[0], 0.05)
        self.assertLess(np.linalg.norm(estimate[1:4] - TRUTH[1:4]), 0.015)
        self.assertLess(
            np.linalg.norm(estimate[4:] - TRUTH[4:]) / np.linalg.norm(TRUTH[4:]),
            0.15,
        )

    def test_hover_only_reports_rank_deficiency(self):
        particle_filter = StaticParameterParticleFilter(
            ParameterBounds(LOWER, UPPER),
            ParticleFilterConfig(particle_count=128, mcmc_steps=0, seed=5),
        )
        specific = np.tile([0.0, 0.0, 9.80665], (5, 1))
        zero = np.zeros((5, 3))
        wrench = np.array([predict_wrench(TRUTH, specific[0], zero[0], zero[0])] * 5)
        particle_filter.update(
            ObservationBatch(specific, zero, zero, wrench, np.full(3, 0.5), np.full(3, 0.05))
        )
        rank, condition = particle_filter.excitation_metrics(
            particle_filter.posterior_summary().mean
        )
        self.assertLess(rank, 10)
        self.assertTrue(np.isinf(condition))


if __name__ == "__main__":
    unittest.main()
