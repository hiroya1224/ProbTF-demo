import unittest

import numpy as np

from grape_param_estim.controller import (
    ControllerConfig,
    GrapeController,
    initial_controller_state,
)
from grape_param_estim.diagonal_q import (
    BodyWrenchDiagonalCovariance,
    OuTransitionFactors,
)
from grape_param_estim.dynamics import FullSixDofPlant
from grape_param_estim.ensemble_state_smoother import exact_gaussian_ensemble
from grape_param_estim.filter_state import (
    GrapeFilterState,
    GrapeFilterStateChart,
)
from grape_param_estim.geometry import (
    euler_xyz_to_matrix,
    matrix_to_quaternion,
)
from grape_param_estim.progress import (
    CancellationToken,
    ProgressCancelled,
)
from grape_param_estim.stochastic_closed_loop import (
    PoseObservationCovariance,
    ou_wrench_transition,
    run_stochastic_closed_loop_e_step,
)
from grape_param_estim.synthetic import full_six_dof_reference
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    GrapeGeometry,
    RigidBodyState,
    VehicleParameters,
)


class StochasticClosedLoopEStepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.asarray((0.0, 0.015, 0.043))
        cls.references = full_six_dof_reference(cls.times)
        cls.parameters = VehicleParameters.nominal()
        cls.geometry = GrapeGeometry.grape()
        cls.configuration = ControllerConfig.grape()
        cls.actuator_parameters = ActuatorParameters(
            thrust_time_constant=0.03,
            gimbal_time_constant=0.04,
            delay=0.021,
        )
        cls.q_covariance = BodyWrenchDiagonalCovariance(
            np.asarray((0.09, 0.16, 0.25, 0.01, 0.0144, 0.0196))
        )
        cls.observation_covariance = PoseObservationCovariance.isotropic(
            0.03, 0.02
        )
        first = cls.references[0]
        cls.initial_rigid = RigidBodyState(
            first.position,
            matrix_to_quaternion(euler_xyz_to_matrix(first.rpy)),
            first.linear_velocity,
            first.angular_velocity,
        )
        cls.initial_controller = initial_controller_state(
            cls.configuration, trim_hover=True
        )
        cls.observed_position = np.asarray(
            [value.position for value in cls.references]
        )
        cls.observed_orientation = np.asarray(
            [
                matrix_to_quaternion(euler_xyz_to_matrix(value.rpy))
                for value in cls.references
            ]
        )
        wrench = exact_gaussian_ensemble(
            np.zeros(6), cls.q_covariance.matrix, 40, 319
        )
        # Deliberately invalid actuator snapshots exercise analysis clipping.
        actuator = ActuatorState(
            np.asarray((-4.0, 5.0, 31.0, 7.0)),
            np.asarray((-4.0, -0.1, 0.2, 4.0)),
        )
        cls.initial_ensemble = tuple(
            GrapeFilterState(
                rigid=cls.initial_rigid,
                controller=cls.initial_controller,
                actuator=actuator,
                residual_wrench=value,
            )
            for value in wrench
        )

    def _controller(self):
        return GrapeController(
            self.configuration, self.parameters, self.geometry
        )

    def _run(self, seed=47, **overrides):
        arguments = {
            "times": self.times,
            "references": self.references,
            "observed_position": self.observed_position,
            "observed_orientation_xyzw": self.observed_orientation,
            "initial_state_ensemble": self.initial_ensemble,
            "controller": self._controller(),
            "plant": FullSixDofPlant(self.parameters, self.geometry),
            "actuator_parameters": self.actuator_parameters,
            "wrench_covariance": self.q_covariance,
            "correlation_time": 0.13,
            "observation_covariance": self.observation_covariance,
            "seed": seed,
            "progress_run_id": "synthetic-q-only",
            "bag_id": "synthetic",
        }
        arguments.update(overrides)
        return run_stochastic_closed_loop_e_step(**arguments)

    def test_small_synthetic_run_is_aligned_and_clips_analysis_actuators(self):
        events = []
        result = self._run(progress_callback=events.append)
        self.assertEqual(result.times.shape, (3,))
        self.assertEqual(result.smoothed_wrench_ensemble.shape, (40, 3, 6))
        self.assertEqual(result.filter_log_likelihood_by_time.shape, (3,))
        self.assertEqual(result.filter_nis.shape, (3,))
        self.assertEqual(len(result.forecast_state_ensembles), 3)
        self.assertEqual(len(result.analysis_state_ensembles), 3)
        self.assertEqual(len(result.smoothed_state_ensembles), 3)
        self.assertEqual(len(result.smoothing_gains), 2)
        self.assertAlmostEqual(
            result.filter_log_likelihood,
            np.sum(result.filter_log_likelihood_by_time),
        )
        self.assertAlmostEqual(result.filter_nis[0], 0.0, places=24)

        limits = self.actuator_parameters
        for ensemble in result.analysis_state_ensembles:
            for state in ensemble:
                self.assertTrue(
                    np.all(state.actuator.thrust >= limits.minimum_thrust)
                )
                self.assertTrue(
                    np.all(state.actuator.thrust <= limits.maximum_thrust)
                )
                self.assertTrue(
                    np.all(
                        np.abs(state.actuator.gimbal_angle)
                        <= limits.maximum_gimbal_angle
                    )
                )
        for issue_times in result.command_issue_times:
            np.testing.assert_array_equal(issue_times, self.times[:-1])
        completed = [event.completed_units for event in events]
        self.assertEqual(completed, sorted(completed))
        self.assertEqual(events[0].completed_units, 0)
        self.assertEqual(events[-1].completed_units, events[-1].total_units)

    def test_process_noise_is_seeded_exact_and_analysis_orthogonal(self):
        first = self._run(seed=2026)
        repeated = self._run(seed=2026)
        changed = self._run(seed=2027)
        np.testing.assert_array_equal(
            first.smoothed_wrench_ensemble,
            repeated.smoothed_wrench_ensemble,
        )
        np.testing.assert_array_equal(
            first.filter_log_likelihood_by_time,
            repeated.filter_log_likelihood_by_time,
        )
        self.assertFalse(
            np.array_equal(
                first.smoothed_wrench_ensemble,
                changed.smoothed_wrench_ensemble,
            )
        )

        factors = OuTransitionFactors(self.times, 0.13)
        expected_variance = factors.innovation_variance(self.q_covariance)
        for index in range(self.times.size - 1):
            analysis = first.analysis_state_ensembles[index]
            forecast = first.forecast_state_ensembles[index + 1]
            current = np.asarray(
                [state.residual_wrench for state in analysis]
            )
            following = np.asarray(
                [state.residual_wrench for state in forecast]
            )
            noise = following - factors.rho[index] * current
            np.testing.assert_allclose(np.mean(noise, axis=0), 0.0, atol=1e-15)
            np.testing.assert_allclose(
                np.cov(noise, rowvar=False),
                np.diag(expected_variance[index]),
                rtol=2e-14,
                atol=2e-16,
            )
            chart = GrapeFilterStateChart.from_ensemble(analysis)
            coordinates = chart.encode_ensemble(analysis)
            anomalies = coordinates - np.mean(coordinates, axis=0)
            np.testing.assert_allclose(
                anomalies.T @ noise / (noise.shape[0] - 1.0),
                0.0,
                atol=2e-15,
            )

    def test_zero_innovation_uses_unequal_dt_ou_and_trapezoidal_wrench(self):
        factors = OuTransitionFactors(self.times, 0.13)
        current = np.arange(12, dtype=float).reshape(2, 6) - 3.0
        for rho in factors.rho:
            following, interval = ou_wrench_transition(
                current, rho, np.zeros_like(current)
            )
            np.testing.assert_array_equal(following, rho * current)
            np.testing.assert_array_equal(
                interval, 0.5 * (current + following)
            )
            current = following
        self.assertNotEqual(factors.rho[0], factors.rho[1])

    def test_cancellation_and_strict_validation(self):
        cancellation = CancellationToken()
        cancellation.cancel("unit_test")
        with self.assertRaises(ProgressCancelled):
            self._run(cancellation_token=cancellation)
        with self.assertRaisesRegex(ValueError, "references"):
            self._run(references=self.references[:-1])
        with self.assertRaisesRegex(ValueError, "observed_position"):
            self._run(observed_position=np.zeros((2, 3)))
        invalid_orientation = self.observed_orientation.copy()
        invalid_orientation[1] = 0.0
        with self.assertRaisesRegex(ValueError, "quaternion"):
            self._run(observed_orientation_xyzw=invalid_orientation)
        with self.assertRaisesRegex(ValueError, "seed"):
            self._run(seed=-1)
        with self.assertRaisesRegex(ValueError, "positive definite"):
            PoseObservationCovariance(np.eye(3), np.zeros((3, 3)))
        with self.assertRaisesRegex(ValueError, "member-first"):
            ou_wrench_transition(np.zeros((2, 6)), 0.9, np.zeros((2, 5)))


if __name__ == "__main__":
    unittest.main()
