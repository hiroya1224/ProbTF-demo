from dataclasses import replace
import unittest

import numpy as np

from grape_param_estim.augmented_parameter_state import (
    LOCAL_INITIAL_DIMENSION,
    MINIMUM_PROCESS_NOISE_MEMBER_COUNT,
    SHARED_STATIC_DIMENSION,
    AugmentedParameterPrior,
)
from grape_param_estim.controller import (
    ControllerConfig,
    initial_controller_state,
)
from grape_param_estim.diagonal_q import BodyWrenchDiagonalCovariance
from grape_param_estim.multi_bag_augmented_parameter import (
    MultiBagAugmentedParameterResult,
    PreparedAugmentedParameterBag,
    run_multi_bag_augmented_parameter_filter,
)
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.progress import CancellationToken, ProgressCancelled
from grape_param_estim.stochastic_closed_loop import (
    PoseObservationCovariance,
)
from grape_param_estim.strong_constraint import StrongConstraintProblem
from grape_param_estim.synthetic import run_synthetic_experiment
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    GrapeGeometry,
    RigidBodyState,
    VehicleParameters,
)


class MultiBagAugmentedParameterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        synthetic = run_synthetic_experiment(
            duration=0.08,
            time_step=0.04,
            truth_actuators=ActuatorParameters(delay=0.012),
            truth_residual_wrench=lambda _time, _state: np.zeros(6),
            translation_noise=0.003,
            rotation_noise=0.002,
            seed=211,
        )
        configuration = ControllerConfig.grape()
        controller_state = initial_controller_state(
            configuration, trim_hover=True
        )
        parameters = VehicleParameters.nominal()
        initial_actuator = ActuatorState(
            synthetic.nominal.actuator_thrust[0],
            synthetic.nominal.actuator_gimbal_angle[0],
        )
        problem = StrongConstraintProblem(
            references=synthetic.references,
            observations=synthetic.observations,
            nominal_trajectory=synthetic.nominal,
            initial_state_anchor=RigidBodyState(
                synthetic.nominal.position[0],
                synthetic.nominal.orientation_xyzw[0],
                synthetic.nominal.linear_velocity[0],
                synthetic.nominal.angular_velocity[0],
            ),
            initial_controller_anchor=controller_state,
            controller_configuration=configuration,
            controller_parameters=parameters,
            geometry=GrapeGeometry.grape(),
            actuator_parameters=ActuatorParameters(delay=0.02),
            parameter_chart=VehicleParameterChart(parameters),
            initial_actuator_state=initial_actuator,
        )
        observation_covariance = PoseObservationCovariance.isotropic(
            0.02, 0.012
        )
        cls.first = PreparedAugmentedParameterBag(
            "bag-a",
            problem,
            observation_covariance,
            0.16,
            "same-airframe",
        )
        cls.second = PreparedAugmentedParameterBag(
            "bag-b",
            problem,
            observation_covariance,
            0.21,
            "same-airframe",
        )
        cls.q = BodyWrenchDiagonalCovariance(
            np.asarray((0.04, 0.05, 0.06, 0.004, 0.005, 0.006))
        )
        cls.progress_events = []
        cls.forward = run_multi_bag_augmented_parameter_filter(
            (cls.second, cls.first),
            cls.q,
            ensemble_size=MINIMUM_PROCESS_NOISE_MEMBER_COUNT,
            seed=702,
            progress_callback=cls.progress_events.append,
            run_id="multi-bag-forward",
        )
        cls.reverse_input = run_multi_bag_augmented_parameter_filter(
            (cls.first, cls.second),
            cls.q,
            ensemble_size=MINIMUM_PROCESS_NOISE_MEMBER_COUNT,
            seed=702,
            run_id="multi-bag-reverse-input",
        )

    def test_order_dimensions_and_exact_member_aligned_carry(self):
        result = self.forward
        members = MINIMUM_PROCESS_NOISE_MEMBER_COUNT
        self.assertEqual(result.bag_ids, ("bag-a", "bag-b"))
        self.assertEqual(result.maximum_delay, 0.2)
        np.testing.assert_array_equal(result.member_id, np.arange(members))
        self.assertEqual(
            result.initial_shared_ensemble.shape,
            (members, SHARED_STATIC_DIMENSION),
        )
        self.assertEqual(
            result.final_shared_posterior.shape,
            (members, SHARED_STATIC_DIMENSION),
        )
        first = result.bag("bag-a")
        second = result.bag("bag-b")
        self.assertEqual(first.correlation_time, 0.16)
        self.assertEqual(second.correlation_time, 0.21)
        np.testing.assert_array_equal(
            first.wrench_covariance.stationary_variance,
            self.q.stationary_variance,
        )
        np.testing.assert_array_equal(
            first.observation_covariance.matrix,
            self.first.observation_covariance.matrix,
        )
        np.testing.assert_array_equal(
            first.filter_result.final_static_ensemble,
            second.initial_ensemble.shared_coordinates,
        )
        np.testing.assert_array_equal(
            first.initial_ensemble.member_id,
            second.initial_ensemble.member_id,
        )
        for value in result.bags:
            self.assertEqual(
                value.initial_ensemble.local_coordinates.shape,
                (members, LOCAL_INITIAL_DIMENSION),
            )
            self.assertEqual(
                value.filter_result.final_static_ensemble.shape,
                (members, SHARED_STATIC_DIMENSION),
            )
            self.assertEqual(
                value.filter_result.smoothed_wrench_ensemble.shape,
                (members, 3, 6),
            )
        completed = [value.completed_units for value in self.progress_events]
        self.assertEqual(completed, sorted(completed))
        self.assertEqual(self.progress_events[0].completed_units, 0)
        self.assertEqual(
            self.progress_events[-1].completed_units,
            self.progress_events[-1].total_units,
        )

    def test_each_bag_local_and_wrench_prior_is_exact_and_orthogonal(self):
        prior = AugmentedParameterPrior.grape()
        members = MINIMUM_PROCESS_NOISE_MEMBER_COUNT
        for value in self.forward.bags:
            initial = value.initial_ensemble
            shared = initial.shared_coordinates
            local = initial.local_coordinates
            wrench = np.asarray(
                [state.residual_wrench for state in initial.filter_states]
            )
            np.testing.assert_allclose(
                np.mean(local, axis=0), prior.local_mean, atol=3.0e-15
            )
            np.testing.assert_allclose(
                np.cov(local, rowvar=False),
                prior.local_covariance,
                rtol=4.0e-14,
                atol=3.0e-15,
            )
            shared_anomaly = shared - np.mean(shared, axis=0)
            local_anomaly = local - np.mean(local, axis=0)
            np.testing.assert_allclose(
                shared_anomaly.T @ local_anomaly / (members - 1.0),
                0.0,
                atol=4.0e-15,
            )
            unknowns = np.concatenate((shared, local), axis=1)
            unknown_anomaly = unknowns - np.mean(unknowns, axis=0)
            np.testing.assert_allclose(
                unknown_anomaly.T @ wrench / (members - 1.0),
                0.0,
                atol=5.0e-15,
            )
            np.testing.assert_allclose(
                np.mean(wrench, axis=0), 0.0, atol=2.0e-15
            )
            np.testing.assert_allclose(
                np.cov(wrench, rowvar=False),
                self.q.matrix,
                rtol=4.0e-14,
                atol=2.0e-15,
            )
        self.assertFalse(
            np.array_equal(
                self.forward.bag("bag-a").initial_ensemble.local_coordinates,
                self.forward.bag("bag-b").initial_ensemble.local_coordinates,
            )
        )

    def test_seeded_result_is_reproducible_and_input_order_independent(self):
        self.assertEqual(self.forward.bag_ids, self.reverse_input.bag_ids)
        for bag_id in self.forward.bag_ids:
            first = self.forward.bag(bag_id)
            second = self.reverse_input.bag(bag_id)
            np.testing.assert_array_equal(
                first.initial_ensemble.shared_coordinates,
                second.initial_ensemble.shared_coordinates,
            )
            np.testing.assert_array_equal(
                first.initial_ensemble.local_coordinates,
                second.initial_ensemble.local_coordinates,
            )
            np.testing.assert_array_equal(
                first.filter_result.final_static_ensemble,
                second.filter_result.final_static_ensemble,
            )
            np.testing.assert_array_equal(
                first.filter_result.smoothed_wrench_ensemble,
                second.filter_result.smoothed_wrench_ensemble,
            )

    def test_accessors_return_copies_and_result_validation_is_strict(self):
        posterior = self.forward.final_shared_posterior
        expected = posterior.copy()
        posterior[0, 0] += 100.0
        np.testing.assert_array_equal(
            self.forward.final_shared_posterior, expected
        )
        with self.assertRaises(KeyError):
            self.forward.bag("missing")
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            MultiBagAugmentedParameterResult(
                self.q, tuple(reversed(self.forward.bags))
            )

    def test_invalid_bags_configuration_chart_and_controls_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            run_multi_bag_augmented_parameter_filter((), self.q)
        with self.assertRaises(TypeError):
            run_multi_bag_augmented_parameter_filter((object(),), self.q)
        with self.assertRaisesRegex(ValueError, "unique"):
            run_multi_bag_augmented_parameter_filter(
                (self.first, self.first), self.q
            )
        mismatch = replace(
            self.second, configuration_fingerprint="different-airframe"
        )
        with self.assertRaisesRegex(ValueError, "configuration fingerprint"):
            run_multi_bag_augmented_parameter_filter(
                (self.first, mismatch), self.q
            )

        other_parameters = replace(VehicleParameters.nominal(), mass=2.7)
        other_problem = replace(
            self.second.problem,
            parameter_chart=VehicleParameterChart(other_parameters),
        )
        other_chart = replace(self.second, problem=other_problem)
        with self.assertRaisesRegex(ValueError, "physical parameter chart"):
            run_multi_bag_augmented_parameter_filter(
                (self.first, other_chart), self.q
            )
        for size in (False, 57, 58.5):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "at least 58"):
                    run_multi_bag_augmented_parameter_filter(
                        (self.first,), self.q, ensemble_size=size
                    )
        with self.assertRaisesRegex(ValueError, "seed"):
            run_multi_bag_augmented_parameter_filter(
                (self.first,), self.q, seed=-1
            )
        with self.assertRaises(TypeError):
            run_multi_bag_augmented_parameter_filter(
                (self.first,), object()
            )

        cancellation = CancellationToken()
        cancellation.cancel("unit_test")
        with self.assertRaises(ProgressCancelled):
            run_multi_bag_augmented_parameter_filter(
                (self.first,),
                self.q,
                cancellation_token=cancellation,
            )


if __name__ == "__main__":
    unittest.main()
