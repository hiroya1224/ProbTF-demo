import unittest

import numpy as np

from grape_param_estim.joint_assimilation import (
    JointBagProblem,
    JointIEnKSConfig,
    JointWeakConstraintIEnKSQ,
    JointWeakConstraintPrior,
    JointWeakConstraintProblem,
)
from grape_param_estim.model_error import KnotGaussMarkovWrenchProcess
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.strong_constraint_experiments import (
    _problem_from_synthetic,
)
from grape_param_estim.synthetic import run_synthetic_experiment
from grape_param_estim.system import ActuatorParameters, VehicleParameters
from grape_param_estim.weak_constraint import WeakConstraintProblem


class JointDelayIdentifiabilityTest(unittest.TestCase):
    """Known sub-sample delay remains identified when Q is also estimated."""

    truth_delay = 0.017
    time_step = 0.04
    prior_delay_standard_deviation = 0.012

    @classmethod
    def setUpClass(cls):
        nominal = VehicleParameters.nominal()
        chart = VehicleParameterChart(nominal)
        truth_coordinates = np.zeros(18)
        # Excite the exact mass/effectiveness ridge while estimating the full
        # physical chart and delay jointly.  Recovery is consequently judged
        # from raw delay members, never from one arbitrary physical point.
        truth_coordinates[0] = 0.06
        truth_coordinates[10:14] = np.asarray(
            (-0.04, -0.03, -0.05, -0.02)
        )
        truth_coordinates[14:18] = np.asarray(
            (0.02, -0.015, 0.025, -0.01)
        )
        cls.chart = chart
        cls.synthetic = run_synthetic_experiment(
            duration=1.0,
            time_step=cls.time_step,
            truth_parameters=chart.decode(truth_coordinates),
            truth_actuators=ActuatorParameters(delay=cls.truth_delay),
            truth_residual_wrench=lambda _time, _state: np.zeros(6),
            translation_noise=0.0015,
            rotation_noise=np.deg2rad(0.08),
            seed=2026,
        )
        strong = _problem_from_synthetic(cls.synthetic)
        cls.results = {}
        for label, stationary_deviation in (
            (
                "near_zero_q",
                np.asarray((1.0e-5,) * 3 + (1.0e-6,) * 3),
            ),
            (
                "flexible_q",
                np.asarray((0.08,) * 3 + (0.005,) * 3),
            ),
        ):
            process = KnotGaussMarkovWrenchProcess(
                integration_times=strong.observations.times,
                knot_indices=np.asarray(
                    (0, strong.observations.times.size - 1),
                    dtype=np.int64,
                ),
                stationary_standard_deviation=stationary_deviation,
                correlation_time=0.4,
            )
            problem = JointWeakConstraintProblem(
                (
                    JointBagProblem(
                        "known_delay",
                        WeakConstraintProblem(strong, process),
                        "synthetic-same-hardware",
                    ),
                )
            )
            prior = JointWeakConstraintPrior.grape(
                problem,
                delay_mean=0.028,
                delay_standard_deviation=(
                    cls.prior_delay_standard_deviation
                ),
            )
            posterior = JointWeakConstraintIEnKSQ(
                JointIEnKSConfig(
                    ensemble_size=24,
                    maximum_iterations=2,
                    minimum_line_search_step=1.0 / 16.0,
                    seed=77,
                )
            ).fit(problem, prior)
            cls.results[label] = (problem, prior, posterior)

    def test_truth_delay_is_strictly_sub_sample(self):
        self.assertGreater(self.truth_delay, 0.0)
        self.assertLess(self.truth_delay, self.time_step)
        self.assertFalse(
            np.isclose(
                self.truth_delay / self.time_step,
                round(self.truth_delay / self.time_step),
            )
        )
        self.assertEqual(
            self.synthetic.truth_actuator_parameters.delay,
            self.truth_delay,
        )
        self.assertEqual(
            self.synthetic.nominal_actuator_parameters.delay,
            0.0,
        )

    def test_truth_is_covered_and_delay_variance_shrinks_for_both_q_laws(self):
        for label, (_problem, prior, posterior) in self.results.items():
            with self.subTest(q_law=label):
                delay = posterior.shared_parameter_ensemble.constant_delay
                lower, upper = np.percentile(delay, (2.5, 97.5))
                self.assertLessEqual(lower, self.truth_delay)
                self.assertGreaterEqual(upper, self.truth_delay)
                prior_variance = (
                    self.prior_delay_standard_deviation**2
                )
                posterior_variance = float(
                    np.var(
                        posterior.shared_parameter_ensemble
                        .constant_delay_coordinate,
                        ddof=1,
                    )
                )
                self.assertLess(posterior_variance, prior_variance)
                self.assertEqual(
                    prior.standard_deviation[
                        _problem.layout.shared_delay_index
                    ],
                    self.prior_delay_standard_deviation,
                )

    def test_flexible_q_does_not_absorb_the_delay(self):
        near_delay = self.results["near_zero_q"][2].shared_parameter_ensemble
        flexible_delay = self.results["flexible_q"][2].shared_parameter_ensemble
        flexible_lower, _flexible_upper = np.percentile(
            flexible_delay.constant_delay, (2.5, 97.5)
        )
        # A 95% lower bound above the physical zero-delay boundary is a direct
        # non-absorption criterion; it does not use a tuned error tolerance.
        self.assertGreater(flexible_lower, 0.0)
        # The Q-law sensitivity must remain smaller than the declared prior
        # delay scale, rather than an ad-hoc numerical threshold.
        self.assertLess(
            abs(
                np.median(flexible_delay.constant_delay)
                - np.median(near_delay.constant_delay)
            ),
            self.prior_delay_standard_deviation,
        )

    def test_full_mass_effectiveness_ridge_is_estimated_simultaneously(self):
        expected_direction = np.concatenate(
            (self.chart.ridge_direction(), np.asarray((0.0,)))
        )
        expected_direction /= np.linalg.norm(expected_direction)
        for label, (_problem, _prior, posterior) in self.results.items():
            with self.subTest(q_law=label):
                shared = posterior.shared_parameter_ensemble
                self.assertEqual(
                    shared.physical_parameter_coordinates.shape,
                    (24, 18),
                )
                self.assertEqual(shared.coordinates.shape, (24, 19))
                np.testing.assert_allclose(
                    posterior.ridge.expected_direction,
                    expected_direction,
                )
                self.assertEqual(
                    posterior.ridge.expected_direction[-1], 0.0
                )


if __name__ == "__main__":
    unittest.main()
