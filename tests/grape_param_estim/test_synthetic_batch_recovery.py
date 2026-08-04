import unittest

import numpy as np
from scipy.optimize import least_squares

from grape_param_estim.batch.factors.imu import (
    evaluate_accelerometer_factor,
    evaluate_gyro_factor,
)
from grape_param_estim.batch.factors.kinematics import (
    evaluate_orientation_kinematic_factor,
    evaluate_position_kinematic_factor,
)
from grape_param_estim.batch.factors.pose import (
    evaluate_pose_observation_factors,
)
from grape_param_estim.batch.factors.velocity import (
    evaluate_world_sensor_velocity_factor,
)
from grape_param_estim.batch.lag_profile import (
    LagObjectiveResult,
    LagProfileSettings,
    optimize_lag_profile,
)
from grape_param_estim.batch.laplace_em import (
    QInnerEvaluation,
    compute_diagonal_q_target,
    damped_diagonal_q_update,
)
from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.ridge import analyze_reduced_hessian
from grape_param_estim.batch.state import BatchState
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.geometry import (
    so3_exp,
    so3_geodesic_interpolation_with_right_jacobians,
)
from grape_param_estim.batch.evidence import (
    compute_delay_static_laplace_geometry,
)
from grape_param_estim.synthetic_batch import (
    generate_known_q_laplace_moments,
    generate_perfect_model_batch_trajectory,
    simulate_delayed_zoh_first_order,
)


_KNOT_KINDS = (
    VariableKind.POSITION,
    VariableKind.ORIENTATION_TANGENT,
    VariableKind.LINEAR_VELOCITY,
    VariableKind.ANGULAR_VELOCITY,
    VariableKind.CONTROLLER_INTEGRAL,
    VariableKind.ACTUATOR_THRUST,
    VariableKind.GIMBAL_ANGLE,
)


def _marker_state(marker):
    keys = [VariableKey(VariableKind.STATIC_PARAMETERS)]
    keys.extend(
        VariableKey(kind, bag_id="synthetic", knot_index=0)
        for kind in _KNOT_KINDS
    )
    layout = VariableLayout(tuple(keys))
    values = {}
    for key in layout.variable_keys:
        values[key] = (
            np.eye(3)
            if key.kind is VariableKind.ORIENTATION_TANGENT
            else np.zeros(key.dimension)
        )
    values[VariableKey(VariableKind.STATIC_PARAMETERS)][0] = float(marker)
    return BatchState(layout, values)


class PerfectModelSyntheticRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajectory = generate_perfect_model_batch_trajectory(
            interval_count=30,
            seed=917,
        )

    def test_full_trajectory_satisfies_kinematics_and_six_axis_dynamics(self):
        trajectory = self.trajectory
        dynamics = []
        for index, dt in enumerate(trajectory.time_step):
            dynamics.append(
                trajectory.dynamics_evaluation(
                    index, trajectory.truth_parameter_coordinates
                ).residual
            )
            position = evaluate_position_kinematic_factor(
                "synthetic-perfect",
                index,
                trajectory.position[index],
                trajectory.position[index + 1],
                trajectory.linear_velocity[index],
                trajectory.linear_velocity[index + 1],
                dt,
                np.eye(3),
            )
            orientation = evaluate_orientation_kinematic_factor(
                "synthetic-perfect",
                index,
                trajectory.rotation[index],
                trajectory.rotation[index + 1],
                trajectory.angular_velocity[index],
                trajectory.angular_velocity[index + 1],
                dt,
                np.eye(3),
            )
            self.assertLess(np.linalg.norm(position.residual, ord=np.inf), 2.0e-14)
            self.assertLess(
                np.linalg.norm(orientation.residual, ord=np.inf),
                3.0e-13,
            )
        stacked = np.asarray(dynamics)
        self.assertLess(np.linalg.norm(stacked, ord=np.inf), 2.0e-10)
        self.assertTrue(np.all(np.ptp(stacked, axis=0) < 4.0e-10))
        self.assertGreater(np.ptp(trajectory.time_step), 0.01)

    def test_18d_static_chart_recovers_truth_modulo_exact_common_scale(self):
        trajectory = self.trajectory
        truth = trajectory.truth_parameter_coordinates
        ridge = trajectory.parameter_chart.ridge_direction()
        ridge /= np.linalg.norm(ridge)
        random = np.random.RandomState(44)
        perturbation = 0.07 * random.normal(size=18)
        perturbation -= ridge * float(ridge @ perturbation)
        initial = truth + perturbation

        def evaluate(coordinates):
            return trajectory.body_wrench_residual_and_jacobian(
                coordinates
            )

        def evaluate_with_ridge_gauge(coordinates):
            residual, jacobian = evaluate(coordinates)
            gauge = float(ridge @ (coordinates - truth))
            return (
                np.concatenate((residual, np.asarray((gauge,)))),
                np.vstack((jacobian, ridge)),
            )

        result = least_squares(
            lambda value: evaluate_with_ridge_gauge(value)[0],
            initial,
            jac=lambda value: evaluate_with_ridge_gauge(value)[1],
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=80,
        )
        error = result.x - truth
        identified_error = error - ridge * float(ridge @ error)
        self.assertTrue(result.success, msg=result.message)
        self.assertLess(np.linalg.norm(result.fun, ord=np.inf), 3.0e-9)
        self.assertLess(np.linalg.norm(identified_error, ord=np.inf), 2.0e-7)

        residual, jacobian = evaluate(truth)
        analysis = analyze_reduced_hessian(
            jacobian.T @ jacobian,
            relative_rank_tolerance=1.0e-9,
        )
        self.assertEqual(analysis.effective_rank, 17)
        numerical_ridge = analysis.ridge_directions[0].vector
        self.assertGreater(abs(float(numerical_ridge @ ridge)), 1.0 - 1.0e-10)
        self.assertLess(np.linalg.norm(residual, ord=np.inf), 2.0e-10)

        baseline = float(residual @ residual)
        for scale in (-0.7, -0.25, 0.3, 0.65):
            shifted, shifted_jacobian = evaluate(truth + scale * ridge)
            self.assertAlmostEqual(
                float(shifted @ shifted), baseline, delta=2.0e-18
            )
            self.assertLess(
                np.linalg.norm(shifted_jacobian @ ridge, ord=np.inf),
                3.0e-11,
            )


class KnownQSyntheticRecoveryTests(unittest.TestCase):
    def test_laplace_em_recovers_six_components_across_scale_and_bags(self):
        cases = (
            (
                "isotropic-small",
                np.asarray((0.015, 0.015, 0.015, 0.001, 0.001, 0.001)),
            ),
            (
                "anisotropic",
                np.asarray((0.012, 0.045, 0.19, 0.0004, 0.0025, 0.014)),
            ),
            (
                "large-spectral-density",
                np.asarray((3.0, 1.1, 0.42, 0.09, 0.025, 0.006)),
            ),
        )
        random = np.random.RandomState(81)
        bag_steps = (
            random.uniform(0.008, 0.025, size=900),
            random.uniform(0.031, 0.074, size=1100),
        )
        for case_index, (name, truth) in enumerate(cases):
            with self.subTest(name=name):
                synthetic = generate_known_q_laplace_moments(
                    truth,
                    bag_steps,
                    observation_noise_ratio=0.8,
                    seed=700 + case_index,
                )
                target = compute_diagonal_q_target(
                    synthetic.definition,
                    synthetic.moments,
                    synthetic.time_step,
                    floor=np.full(6, 1.0e-12),
                )
                weights = synthetic.definition.interval_weights(
                    synthetic.time_step
                )
                generated_second_moment = np.mean(
                    weights[:, None] * synthetic.latent_residual**2,
                    axis=0,
                )
                np.testing.assert_allclose(
                    generated_second_moment,
                    truth,
                    rtol=0.09,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    target.target,
                    truth,
                    rtol=0.09,
                    atol=0.0,
                )
                map_relative_error = np.linalg.norm(
                    target.map_second_moment / truth - 1.0
                )
                corrected_relative_error = np.linalg.norm(
                    target.target / truth - 1.0
                )
                self.assertGreater(map_relative_error, 0.75)
                self.assertLess(corrected_relative_error, 0.23)
                self.assertTrue(np.all(target.covariance_correction > 0.0))
                self.assertEqual(set(synthetic.bag_index.tolist()), {0, 1})

                for bag_index in (0, 1):
                    selected = synthetic.bag_index == bag_index
                    bag_target = compute_diagonal_q_target(
                        synthetic.definition,
                        type(synthetic.moments)(
                            synthetic.moments.map_residual[selected],
                            synthetic.moments.covariance_correction[selected],
                        ),
                        synthetic.time_step[selected],
                        floor=np.full(6, 1.0e-12),
                    )
                    np.testing.assert_allclose(
                        bag_target.target,
                        truth,
                        rtol=0.14,
                        atol=0.0,
                    )

                initial = QInnerEvaluation(
                    q=truth * np.asarray((1.8, 0.6, 1.5, 0.7, 1.3, 0.55)),
                    successful=True,
                    map_objective=4.0,
                    approximate_marginal_objective=8.0,
                    lag=0.02,
                    failure_reason="",
                )

                def evaluator(candidate, _warm_start):
                    return QInnerEvaluation(
                        q=candidate,
                        successful=True,
                        map_objective=3.0,
                        approximate_marginal_objective=7.0,
                        lag=0.02,
                        failure_reason="",
                    )

                update = damped_diagonal_q_update(initial, target, evaluator)
                self.assertTrue(update.accepted)
                np.testing.assert_allclose(
                    update.accepted_q,
                    target.target,
                    rtol=8.0e-16,
                    atol=0.0,
                )


class ContinuousLagSyntheticRecoveryTests(unittest.TestCase):
    @staticmethod
    def _profile(command_values, truth_delay, initial_marker=0.0):
        event_times = np.asarray((0.0, 0.073, 0.151, 0.238, 0.319, 0.447, 0.561))
        observation_times = np.linspace(0.0, 0.78, 241)
        time_constant = 0.031
        observed = simulate_delayed_zoh_first_order(
            observation_times,
            event_times,
            command_values,
            delay=truth_delay,
            time_constant=time_constant,
        )

        def evaluator(lag, _warm_start):
            predicted = simulate_delayed_zoh_first_order(
                observation_times,
                event_times,
                command_values,
                delay=lag,
                time_constant=time_constant,
            )
            residual = (observed - predicted) / 0.004
            return LagObjectiveResult(
                objective=0.5 * float(np.sum(residual * residual)),
                converged=True,
                state=_marker_state(lag),
                inner_iterations=1,
                termination_reason="synthetic_exact_zoh",
            )

        result = optimize_lag_profile(
            evaluator,
            LagProfileSettings(
                0.0,
                0.14,
                coarse_grid_points=9,
                refinement_tolerance=2.0e-6,
                maximum_refinement_evaluations=36,
            ),
            initial_warm_start=_marker_state(initial_marker),
        )
        return result

    def test_zero_and_subsample_delay_with_multiple_zoh_switches(self):
        commands = np.asarray(
            (
                (0.1, -0.2),
                (1.0, 0.3),
                (-0.4, 1.1),
                (0.7, -0.8),
                (-1.1, 0.5),
                (0.2, 1.3),
                (0.9, -0.1),
            )
        )
        zero = self._profile(commands, 0.0)
        self.assertEqual(zero.best_lag, 0.0)
        truth = 0.0873
        first = self._profile(commands, truth, initial_marker=-3.0)
        repeated = self._profile(commands, truth, initial_marker=4.0)
        self.assertAlmostEqual(first.best_lag, truth, delta=4.0e-6)
        self.assertAlmostEqual(repeated.best_lag, truth, delta=4.0e-6)
        self.assertAlmostEqual(first.best_lag, repeated.best_lag, delta=1.0e-12)
        self.assertGreater(truth, np.min(np.diff((0.0, 0.073, 0.151, 0.238))))
        self.assertGreater(
            sum(point.warm_start_lag is not None for point in first.points),
            5,
        )
        best_coordinate = next(
            point.static_coordinate
            for point in first.points
            if point.lag == first.best_lag
        )
        delay_static_geometry = compute_delay_static_laplace_geometry(
            (first,),
            (0.0, 0.14),
            np.eye(18),
            first.best_lag,
            best_coordinate,
            2.0e-6,
        )
        self.assertIsNotNone(delay_static_geometry.curvature)
        self.assertLess(
            delay_static_geometry.standard_deviation_seconds, 0.01
        )

    def test_low_excitation_reports_uniform_prior_uncertainty(self):
        constant = np.broadcast_to(np.asarray((0.4, -0.2)), (7, 2)).copy()
        profile = self._profile(constant, 0.0873)
        self.assertEqual(profile.best_lag, 0.0)
        self.assertTrue(
            all(point.objective == 0.0 for point in profile.points)
        )
        best_coordinate = next(
            point.static_coordinate
            for point in profile.points
            if point.lag == profile.best_lag
        )
        delay_static_geometry = compute_delay_static_laplace_geometry(
            (profile,),
            (0.0, 0.14),
            np.eye(18),
            profile.best_lag,
            best_coordinate,
            2.0e-6,
        )
        self.assertIsNone(delay_static_geometry.curvature)
        self.assertEqual(
            delay_static_geometry.source, "uniform_delay_prior_fallback"
        )
        self.assertAlmostEqual(
            delay_static_geometry.standard_deviation_seconds,
            0.14 / np.sqrt(12.0),
        )


class AsynchronousSensorSyntheticRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajectory = generate_perfect_model_batch_trajectory(
            interval_count=20,
            seed=102,
        )
        cls.alphas = (0.19, 0.37, 0.58, 0.83)
        cls.truth_sensor_position = np.asarray((0.15, -0.08, 0.12))
        cls.sensor_position = cls.truth_sensor_position.copy()
        cls.sensor_to_body = so3_exp((0.08, -0.04, 0.06))
        cls.truth_body_to_gyro = so3_exp((-0.11, 0.07, 0.04))
        cls.body_to_gyro = cls.truth_body_to_gyro.copy()
        cls.body_to_accelerometer = so3_exp((0.06, 0.09, -0.13))
        cls.truth_cog = cls.trajectory.parameter_chart.decode(
            cls.trajectory.truth_parameter_coordinates
        ).cog_offset
        cls.truth_gyro_bias = np.asarray((0.021, -0.014, 0.009))
        cls.truth_accelerometer_bias = np.asarray((0.11, -0.06, 0.075))
        cls.cog_jacobian = np.zeros((3, 18))
        cls.cog_jacobian[:, 7:10] = np.eye(3)

    def _sensor_system(self, configuration, estimate):
        cog = estimate[:3]
        gyro_bias = estimate[3:6]
        accelerometer_bias = estimate[6:9]
        residuals = []
        jacobians = []
        for index, alpha in enumerate(self.alphas):
            trajectory = self.trajectory
            rotation, _, _ = so3_geodesic_interpolation_with_right_jacobians(
                trajectory.rotation[index],
                trajectory.rotation[index + 1],
                alpha,
            )
            position = (
                (1.0 - alpha) * trajectory.position[index]
                + alpha * trajectory.position[index + 1]
            )
            velocity = (
                (1.0 - alpha) * trajectory.linear_velocity[index]
                + alpha * trajectory.linear_velocity[index + 1]
            )
            omega = (
                (1.0 - alpha) * trajectory.angular_velocity[index]
                + alpha * trajectory.angular_velocity[index + 1]
            )
            truth_lever = self.truth_sensor_position - self.truth_cog
            observed_position = position + rotation @ truth_lever
            observed_rotation = rotation @ self.sensor_to_body
            pose_position, pose_orientation = evaluate_pose_observation_factors(
                "sensor-synthetic",
                index,
                alpha,
                trajectory.position[index],
                trajectory.position[index + 1],
                trajectory.rotation[index],
                trajectory.rotation[index + 1],
                observed_position,
                observed_rotation,
                self.sensor_position,
                self.sensor_to_body,
                cog,
                self.cog_jacobian,
                np.eye(3),
                np.eye(3),
            )
            self.assertLess(np.linalg.norm(pose_orientation.residual), 2.0e-13)
            residuals.append(pose_position.residual)
            row = np.zeros((3, 9))
            row[:, :3] = pose_position.jacobian_blocks[-1].value[:, 7:10]
            jacobians.append(row)

            if "velocity" in configuration:
                observed_velocity = velocity + rotation @ np.cross(
                    omega, truth_lever
                )
                factor = evaluate_world_sensor_velocity_factor(
                    "sensor-synthetic",
                    index,
                    alpha,
                    trajectory.linear_velocity[index],
                    trajectory.linear_velocity[index + 1],
                    trajectory.rotation[index],
                    trajectory.rotation[index + 1],
                    trajectory.angular_velocity[index],
                    trajectory.angular_velocity[index + 1],
                    observed_velocity,
                    self.sensor_position,
                    cog,
                    self.cog_jacobian,
                    np.eye(3),
                )
                residuals.append(factor.residual)
                row = np.zeros((3, 9))
                row[:, :3] = factor.jacobian_blocks[-1].value[:, 7:10]
                jacobians.append(row)

            if "gyro" in configuration:
                observed_gyro = (
                    self.truth_body_to_gyro @ omega + self.truth_gyro_bias
                )
                factor = evaluate_gyro_factor(
                    "sensor-synthetic",
                    index,
                    alpha,
                    trajectory.angular_velocity[index],
                    trajectory.angular_velocity[index + 1],
                    gyro_bias,
                    observed_gyro,
                    self.body_to_gyro,
                    np.eye(3),
                )
                residuals.append(factor.residual)
                row = np.zeros((3, 9))
                row[:, 3:6] = factor.jacobian_blocks[-1].value
                jacobians.append(row)

            if "accelerometer" in configuration:
                dt = trajectory.time_step[index]
                acceleration_world = (
                    (
                        trajectory.linear_velocity[index + 1]
                        - trajectory.linear_velocity[index]
                    )
                    / dt
                    - np.asarray((0.0, 0.0, -9.80665))
                )
                angular_acceleration = (
                    trajectory.angular_velocity[index + 1]
                    - trajectory.angular_velocity[index]
                ) / dt
                modeled_body = (
                    rotation.T @ acceleration_world
                    + np.cross(angular_acceleration, truth_lever)
                    + np.cross(omega, np.cross(omega, truth_lever))
                )
                observed_accelerometer = (
                    self.body_to_accelerometer @ modeled_body
                    + self.truth_accelerometer_bias
                )
                factor = evaluate_accelerometer_factor(
                    "sensor-synthetic",
                    index,
                    alpha,
                    trajectory.rotation[index],
                    trajectory.rotation[index + 1],
                    trajectory.linear_velocity[index],
                    trajectory.linear_velocity[index + 1],
                    trajectory.angular_velocity[index],
                    trajectory.angular_velocity[index + 1],
                    accelerometer_bias,
                    observed_accelerometer,
                    self.body_to_accelerometer,
                    self.sensor_position,
                    cog,
                    self.cog_jacobian,
                    np.asarray((0.0, 0.0, -9.80665)),
                    dt,
                    np.eye(3),
                )
                residuals.append(factor.residual)
                row = np.zeros((3, 9))
                row[:, :3] = factor.jacobian_blocks[-1].value[:, 7:10]
                row[:, 6:9] = factor.jacobian_blocks[-2].value
                jacobians.append(row)
        return np.concatenate(residuals), np.vstack(jacobians)

    def test_pose_velocity_imu_bias_lever_frame_and_async_recovery(self):
        configurations = (
            ("pose-only", ()),
            ("pose-velocity", ("velocity",)),
            ("pose-gyro", ("gyro",)),
            (
                "pose-gyro-accelerometer",
                ("velocity", "gyro", "accelerometer"),
            ),
        )
        truth = np.concatenate(
            (
                self.truth_cog,
                self.truth_gyro_bias,
                self.truth_accelerometer_bias,
            )
        )
        for name, configuration in configurations:
            with self.subTest(name=name):
                active = 3
                if "gyro" in configuration:
                    active = 6
                if "accelerometer" in configuration:
                    active = 9
                initial = np.zeros(9)
                residual, jacobian = self._sensor_system(
                    configuration, initial
                )
                delta, _, rank, _ = np.linalg.lstsq(
                    jacobian[:, :active], -residual, rcond=None
                )
                self.assertEqual(rank, active)
                recovered = initial.copy()
                recovered[:active] += delta
                final_residual, _ = self._sensor_system(
                    configuration, recovered
                )
                np.testing.assert_allclose(
                    recovered[:active], truth[:active], atol=3.0e-12
                )
                self.assertLess(
                    np.linalg.norm(final_residual, ord=np.inf),
                    3.0e-11,
                )

        self.assertTrue(all(0.0 < alpha < 1.0 for alpha in self.alphas))
        correct, _ = self._sensor_system(
            ("velocity", "gyro", "accelerometer"), truth
        )
        self.assertLess(np.linalg.norm(correct), 1.0e-10)
        original_position = self.sensor_position
        original_gyro_rotation = self.body_to_gyro
        try:
            type(self).sensor_position = self.truth_cog.copy()
            type(self).body_to_gyro = np.eye(3)
            wrong, _ = self._sensor_system(
                ("velocity", "gyro", "accelerometer"), truth
            )
        finally:
            type(self).sensor_position = original_position
            type(self).body_to_gyro = original_gyro_rotation
        self.assertGreater(np.linalg.norm(wrong), 0.1)


if __name__ == "__main__":
    unittest.main()
