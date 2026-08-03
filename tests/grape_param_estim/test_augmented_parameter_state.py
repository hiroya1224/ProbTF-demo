import unittest
from types import SimpleNamespace

import numpy as np

from grape_param_estim.augmented_parameter_state import (
    AUGMENTED_FILTER_DIMENSION,
    ESTIMATED_UNKNOWN_DIMENSION,
    LOCAL_INITIAL_DIMENSION,
    MINIMUM_FULL_RANK_MEMBER_COUNT,
    MINIMUM_PROCESS_NOISE_MEMBER_COUNT,
    SHARED_STATIC_DIMENSION,
    AugmentedParameterPrior,
    decode_shared_static_coordinates,
    draw_augmented_initial_ensemble,
)
from grape_param_estim.diagonal_q import BodyWrenchDiagonalCovariance
from grape_param_estim.controller import (
    ControllerConfig,
    initial_controller_state,
)
from grape_param_estim.real_assimilation import build_real_strong_problem
from grape_param_estim.synthetic import run_synthetic_experiment
from grape_param_estim.system import ActuatorParameters, ActuatorState
from grape_param_estim.timing import BoundedDelayChart


class AugmentedParameterStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        synthetic = run_synthetic_experiment(
            duration=0.40,
            time_step=0.04,
            truth_actuators=ActuatorParameters(),
            truth_residual_wrench=lambda _time, _state: np.zeros(6),
            translation_noise=0.001,
            rotation_noise=0.001,
            seed=12,
        )
        configuration = ControllerConfig.grape()
        episode = SimpleNamespace(
            observations=synthetic.observations,
            references=synthetic.references,
            controller_configuration=configuration,
            initial_controller_state=initial_controller_state(
                configuration, trim_hover=True
            ),
            initial_actuator_state=ActuatorState(
                synthetic.nominal.actuator_thrust[0],
                synthetic.nominal.actuator_gimbal_angle[0],
            ),
        )
        cls.problem = build_real_strong_problem(episode)[0]

    def test_dimensions_are_constant_and_have_the_documented_breakdown(self):
        self.assertEqual(SHARED_STATIC_DIMENSION, 19)
        self.assertEqual(LOCAL_INITIAL_DIMENSION, 26)
        self.assertEqual(ESTIMATED_UNKNOWN_DIMENSION, 45)
        self.assertEqual(AUGMENTED_FILTER_DIMENSION, 51)
        self.assertEqual(MINIMUM_FULL_RANK_MEMBER_COUNT, 52)
        self.assertEqual(MINIMUM_PROCESS_NOISE_MEMBER_COUNT, 58)

    def test_default_prior_has_exact_shared_and_local_blocks(self):
        prior = AugmentedParameterPrior.grape(0.03, 0.01)
        delay_chart = BoundedDelayChart()
        self.assertEqual(prior.shared_mean.shape, (19,))
        self.assertEqual(prior.local_mean.shape, (26,))
        self.assertEqual(prior.shared_mean[-1], delay_chart.encode(0.03))
        self.assertEqual(
            prior.shared_covariance[-1, -1],
            delay_chart.coordinate_standard_deviation(0.03, 0.01) ** 2,
        )
        np.testing.assert_array_equal(
            np.diag(prior.local_covariance)[18:],
            np.asarray((0.30,) * 4 + (0.03,) * 4) ** 2,
        )

    def test_exact_initial_unknowns_and_wrench_are_sample_uncorrelated(self):
        covariance = BodyWrenchDiagonalCovariance(
            np.asarray((1.0, 2.0, 3.0, 0.1, 0.2, 0.3))
        )
        ensemble = draw_augmented_initial_ensemble(
            self.problem, covariance, 64, seed=31
        )
        unknowns = ensemble.estimated_unknown_coordinates
        wrench = np.asarray(
            [state.residual_wrench for state in ensemble.filter_states]
        )
        prior = AugmentedParameterPrior.grape()
        np.testing.assert_allclose(
            np.mean(ensemble.shared_coordinates, axis=0),
            prior.shared_mean,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            np.cov(ensemble.shared_coordinates, rowvar=False),
            prior.shared_covariance,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            np.cov(ensemble.local_coordinates, rowvar=False),
            prior.local_covariance,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            (unknowns - unknowns.mean(0)).T
            @ (wrench - wrench.mean(0))
            / 63.0,
            0.0,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            np.cov(wrench, rowvar=False), covariance.matrix, atol=2.0e-15
        )

    def test_static_decode_uses_one_to_one_bounded_delay_coordinate(self):
        delay_chart = BoundedDelayChart()
        coordinates = np.zeros(19)
        coordinates[-1] = delay_chart.encode(0.017)
        parameters, delay = decode_shared_static_coordinates(
            self.problem, coordinates, delay_chart
        )
        self.assertAlmostEqual(delay, 0.017)
        self.assertAlmostEqual(parameters.mass, 2.3515975908123767)
        coordinates[-1] *= -1.0
        _parameters, reflected = decode_shared_static_coordinates(
            self.problem, coordinates, delay_chart
        )
        self.assertNotAlmostEqual(reflected, delay)

    def test_member_floor_and_input_types_are_enforced(self):
        covariance = BodyWrenchDiagonalCovariance(np.ones(6))
        with self.assertRaises(ValueError):
            draw_augmented_initial_ensemble(
                self.problem,
                covariance,
                MINIMUM_FULL_RANK_MEMBER_COUNT - 1,
                1,
            )
        with self.assertRaises(TypeError):
            draw_augmented_initial_ensemble(object(), covariance, 64, 1)
        with self.assertRaises(TypeError):
            draw_augmented_initial_ensemble(self.problem, object(), 64, 1)


if __name__ == "__main__":
    unittest.main()
