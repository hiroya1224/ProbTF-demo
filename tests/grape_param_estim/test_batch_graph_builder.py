from dataclasses import replace
import unittest

import numpy as np

from grape_param_estim.batch.covariance import ArrowheadLaplaceFactorization
from grape_param_estim.batch.dynamics_moments import (
    compute_expected_dynamics_moments,
    evaluate_prepared_dynamics_intervals,
)
from grape_param_estim.batch.factors.dynamics_factor import (
    BODY_WRENCH_QUANTITY,
    SPECIFIC_ACCELERATION_QUANTITY,
)
from grape_param_estim.batch.graph_builder import (
    AccelerometerFactorContract,
    GaussianCovariance,
    MeasurementBracket,
    OrientationGaussianPrior,
    PreparedAccelerometerMeasurement,
    PreparedActuatorInterval,
    PreparedBagGraphData,
    PreparedBagPriors,
    PreparedBatchGraphData,
    PreparedCommandSegment,
    PreparedControllerIntegralMeasurement,
    PreparedControllerInterval,
    PreparedDynamicsConfiguration,
    PreparedDynamicsIntervalStatus,
    PreparedFactorCovariances,
    PreparedGimbalMeasurement,
    PreparedGyroMeasurement,
    PreparedKnotPrior,
    PreparedKnotState,
    PreparedPoseMeasurement,
    PreparedSensorExtrinsics,
    PreparedVelocityMeasurement,
    VectorGaussianPrior,
    build_fixed_batch_problem,
    build_initial_batch_state,
)
from grape_param_estim.batch.laplace_em import (
    DiagonalQDefinition,
    QIntervalModel,
)
from grape_param_estim.batch.state import StateScaling
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.controller import ControllerConfig, GrapeController
from grape_param_estim.geometry import so3_exp
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.system import (
    ActuatorCommand,
    ActuatorParameters,
    GrapeGeometry,
    ReferenceState,
    VehicleParameters,
)


def _covariance(dimension, scale=1.0):
    return GaussianCovariance(scale * np.eye(dimension))


def _vector_prior(mean, scale=1.0):
    value = np.asarray(mean, dtype=float)
    return VectorGaussianPrior(value, _covariance(value.size, scale))


def _orientation_prior(rotation, scale=1.0):
    return OrientationGaussianPrior(rotation, _covariance(3, scale))


def _q_definition(quantity):
    units = (
        ("N", "N", "N", "Nm", "Nm", "Nm")
        if quantity == BODY_WRENCH_QUANTITY
        else ("m/s^2",) * 3 + ("rad/s^2",) * 3
    )
    return DiagonalQDefinition(
        residual_quantity=quantity,
        component_names=("x", "y", "z", "roll", "pitch", "yaw"),
        component_units=units,
        interval_model=QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY,
    )


class BatchGraphBuilderTests(unittest.TestCase):
    def _prepared(self, quantity=SPECIFIC_ACCELERATION_QUANTITY):
        nominal = replace(
            VehicleParameters.nominal(),
            cog_offset=np.asarray((0.012, -0.008, 0.016)),
            linear_drag=np.asarray((0.08, 0.11, 0.14)),
            angular_drag=np.asarray((0.012, 0.016, 0.019)),
        )
        chart = VehicleParameterChart(nominal)
        coordinates = np.asarray(
            (
                0.03,
                0.01,
                -0.008,
                0.012,
                0.004,
                -0.003,
                0.005,
                0.002,
                -0.001,
                0.003,
                0.015,
                -0.012,
                0.009,
                -0.006,
                0.008,
                -0.007,
                0.006,
                -0.005,
            )
        )
        geometry = GrapeGeometry.grape()
        controller = GrapeController(
            ControllerConfig.grape(), nominal, geometry
        )
        knot0 = PreparedKnotState(
            time=10.0,
            position=np.asarray((0.12, -0.07, 1.04)),
            rotation=so3_exp((0.04, -0.03, 0.09)),
            linear_velocity=np.asarray((0.05, -0.03, 0.02)),
            angular_velocity=np.asarray((0.03, -0.02, 0.04)),
            controller_integral=np.asarray(
                (0.01, -0.015, 0.02, 0.005, -0.004, 0.006)
            ),
            actuator_thrust=np.asarray((6.0, 6.2, 5.9, 6.1)),
            gimbal_angle=np.asarray((0.025, -0.03, 0.02, -0.015)),
        )
        knot1 = PreparedKnotState(
            time=10.02,
            position=np.asarray((0.1212, -0.0704, 1.0405)),
            rotation=knot0.rotation @ so3_exp((0.0007, -0.0004, 0.0009)),
            linear_velocity=np.asarray((0.061, -0.019, 0.026)),
            angular_velocity=np.asarray((0.036, -0.018, 0.046)),
            controller_integral=np.asarray(
                (0.011, -0.014, 0.021, 0.006, -0.003, 0.0065)
            ),
            actuator_thrust=np.asarray((6.05, 6.16, 5.96, 6.08)),
            gimbal_angle=np.asarray((0.026, -0.028, 0.019, -0.017)),
        )
        reference = ReferenceState(
            position=np.asarray((0.16, -0.09, 1.08)),
            linear_velocity=np.asarray((0.02, 0.0, -0.01)),
            linear_acceleration=np.asarray((0.04, -0.03, 0.02)),
            rpy=np.asarray((0.06, -0.02, 0.11)),
            angular_velocity=np.asarray((0.01, -0.02, 0.03)),
            angular_acceleration=np.asarray((0.01, 0.0, -0.01)),
        )
        command = ActuatorCommand(
            thrust=np.asarray((6.1, 6.0, 6.2, 5.95)),
            gimbal_angle=np.asarray((0.02, -0.025, 0.018, -0.014)),
            virtual_force=np.zeros(8),
            desired_acceleration=np.zeros(6),
        )
        covariances = PreparedFactorCovariances(
            position_observation=_covariance(3, 0.08),
            orientation_observation=_covariance(3, 0.06),
            velocity_observation=_covariance(3, 0.12),
            gyro_observation=_covariance(3, 0.09),
            accelerometer_observation=None,
            issued_thrust_observation=_covariance(4, 0.2),
            issued_gimbal_observation=_covariance(4, 0.1),
            actual_gimbal_observation=_covariance(4, 0.05),
            controller_integral_observation=_covariance(6, 0.11),
            controller_integral_transition=_covariance(6, 0.15),
            actuator_thrust_transition=_covariance(4, 0.18),
            actuator_gimbal_transition=_covariance(4, 0.12),
            position_kinematic=_covariance(3, 0.07),
            orientation_kinematic=_covariance(3, 0.07),
        )
        knot_prior = PreparedKnotPrior(
            position=_vector_prior(knot0.position, 0.5),
            rotation=_orientation_prior(knot0.rotation, 0.5),
            linear_velocity=_vector_prior(knot0.linear_velocity, 0.5),
            angular_velocity=_vector_prior(knot0.angular_velocity, 0.5),
            controller_integral=_vector_prior(
                knot0.controller_integral, 0.5
            ),
            actuator_thrust=_vector_prior(knot0.actuator_thrust, 0.5),
            gimbal_angle=_vector_prior(knot0.gimbal_angle, 0.5),
        )
        bracket = MeasurementBracket(0, 0.37)
        bag = PreparedBagGraphData(
            bag_id="bag-a",
            knots=(knot0, knot1),
            initial_gyro_bias=np.asarray((0.002, -0.001, 0.003)),
            initial_accelerometer_bias=None,
            priors=PreparedBagPriors(
                gyro_bias=_vector_prior((0.0, 0.0, 0.0), 0.4),
                accelerometer_bias=None,
                initial_knot=knot_prior,
            ),
            controller=controller,
            controller_intervals=(
                PreparedControllerInterval(
                    left_knot_index=0,
                    reference=reference,
                    roll_pitch_integration_active=True,
                    issued_thrust_observation=np.asarray(
                        (6.0, 6.1, 5.95, 6.05)
                    ),
                    issued_gimbal_observation=np.asarray(
                        (0.021, -0.024, 0.017, -0.015)
                    ),
                ),
            ),
            actuator_parameters=ActuatorParameters(
                thrust_time_constant=0.06,
                gimbal_time_constant=0.04,
                minimum_thrust=1.5,
                maximum_thrust=27.0,
                maximum_gimbal_angle=1.0,
                maximum_gimbal_rate=5.0,
            ),
            actuator_intervals=(
                PreparedActuatorInterval(
                    0,
                    (
                        PreparedCommandSegment(command, 0.007),
                        PreparedCommandSegment(command, 0.013),
                    ),
                ),
            ),
            dynamics_interval_statuses=(
                PreparedDynamicsIntervalStatus(0, True, ""),
            ),
            pose_measurements=(
                PreparedPoseMeasurement(
                    bracket,
                    position=np.asarray((0.113, -0.071, 1.094)),
                    rotation=knot0.rotation @ so3_exp((0.0, 0.0, 0.001)),
                ),
            ),
            velocity_measurements=(
                PreparedVelocityMeasurement(
                    bracket, np.asarray((0.057, -0.025, 0.021))
                ),
            ),
            gyro_measurements=(
                PreparedGyroMeasurement(
                    bracket, np.asarray((0.034, -0.019, 0.045))
                ),
            ),
            accelerometer_measurements=(),
            controller_integral_measurements=(
                PreparedControllerIntegralMeasurement(
                    bracket,
                    np.asarray(
                        (0.011, -0.014, 0.021, 0.006, -0.003, 0.006)
                    ),
                ),
            ),
            actual_gimbal_measurements=(
                PreparedGimbalMeasurement(
                    bracket, np.asarray((0.026, -0.029, 0.019, -0.016))
                ),
            ),
            sensor_extrinsics=PreparedSensorExtrinsics(
                pose_sensor_position_in_body=np.asarray(
                    (-0.0173, -0.0011, 0.0571)
                ),
                pose_sensor_to_body_rotation=np.eye(3),
                velocity_sensor_position_in_body=np.asarray(
                    (-0.0173, -0.0011, 0.0571)
                ),
                body_to_gyro_sensor_rotation=np.eye(3),
                accelerometer_sensor_position_in_body=np.asarray(
                    (-0.0173, -0.0011, 0.0571)
                ),
                body_to_accelerometer_sensor_rotation=np.eye(3),
            ),
            covariances=covariances,
            accelerometer=AccelerometerFactorContract(
                enabled=False,
                disabled_reason="physical_imu_origin_not_separately_calibrated",
            ),
        )
        return PreparedBatchGraphData(
            parameter_chart=chart,
            initial_parameter_coordinates=coordinates,
            static_parameter_prior=_vector_prior(np.zeros(18), 0.8),
            geometry=geometry,
            dynamics=PreparedDynamicsConfiguration(
                q_definition=_q_definition(quantity),
                q=np.asarray((1.8, 1.9, 2.0, 0.8, 0.9, 1.0)),
                gravity_world=np.asarray((0.0, 0.0, -9.80665)),
            ),
            fixed_delay=0.0085,
            scaling=StateScaling.unit(),
            bags=(bag,),
        )

    def test_builds_complete_problem_in_declared_factor_order(self):
        prepared = self._prepared()
        problem = build_fixed_batch_problem(prepared)
        state = build_initial_batch_state(prepared)
        self.assertEqual(problem.layout, state.layout)
        self.assertEqual(problem.layout.total_dimension, 18 + 3 + 2 * 26)
        self.assertNotIn(
            VariableKind.ACCELEROMETER_BIAS,
            tuple(key.kind for key in problem.layout.variable_keys),
        )
        factors = problem.evaluate_factors(state)
        self.assertEqual(
            tuple(factor.residual.size for factor in factors),
            (
                18,
                3,
                3,
                3,
                3,
                3,
                6,
                4,
                4,
                3,
                3,
                3,
                3,
                6,
                6,
                4,
                4,
                8,
                4,
                3,
                3,
                6,
            ),
        )
        linearization = problem.linearize(state)
        self.assertTrue(np.isfinite(linearization.sparse.objective))
        self.assertEqual(linearization.sparse.jacobian.shape[1], 73)

    def test_whole_objective_analytic_directional_derivative(self):
        prepared = self._prepared()
        problem = build_fixed_batch_problem(prepared)
        state = build_initial_batch_state(prepared)
        linearization = problem.linearize(state).sparse
        generator = np.random.RandomState(7319)
        direction = generator.normal(size=problem.layout.total_dimension)
        direction /= np.linalg.norm(direction)
        analytic = float(linearization.gradient @ direction)
        step = 2.0e-7
        plus = problem.linearize(state.retract(step * direction)).sparse.objective
        minus = problem.linearize(state.retract(-step * direction)).sparse.objective
        numerical = (plus - minus) / (2.0 * step)
        self.assertAlmostEqual(
            analytic,
            numerical,
            delta=2.0e-5 * max(1.0, abs(analytic), abs(numerical)),
        )

    def test_q_quantity_has_two_explicit_paths_and_no_default(self):
        specific = self._prepared(SPECIFIC_ACCELERATION_QUANTITY)
        body = self._prepared(BODY_WRENCH_QUANTITY)
        specific_factor = build_fixed_batch_problem(specific).evaluate_factors(
            build_initial_batch_state(specific)
        )[-1]
        body_factor = build_fixed_batch_problem(body).evaluate_factors(
            build_initial_batch_state(body)
        )[-1]
        self.assertFalse(
            np.allclose(specific_factor.residual, body_factor.residual)
        )
        unsupported = DiagonalQDefinition(
            residual_quantity="implicit_default_forbidden",
            component_names=("x", "y", "z", "r", "p", "y2"),
            component_units=("u",) * 6,
            interval_model=QIntervalModel.FIXED_INTERVAL_COVARIANCE,
        )
        with self.assertRaisesRegex(ValueError, "explicitly"):
            PreparedDynamicsConfiguration(
                q_definition=unsupported,
                q=np.ones(6),
                gravity_world=np.asarray((0.0, 0.0, -9.80665)),
            )

    def test_accelerometer_contract_and_prebracketing_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "canonical reason"):
            AccelerometerFactorContract(False, "")
        with self.assertRaisesRegex(ValueError, "disabled reason"):
            AccelerometerFactorContract(True, "requested")
        self.assertTrue(AccelerometerFactorContract(True, None).enabled)
        with self.assertRaisesRegex(ValueError, "interpolation_fraction"):
            MeasurementBracket(0, 1.01)
        prepared = self._prepared()
        bag = prepared.bags[0]
        bad_pose = PreparedPoseMeasurement(
            MeasurementBracket(1, 0.0),
            np.zeros(3),
            np.eye(3),
        )
        with self.assertRaisesRegex(ValueError, "outside knot support"):
            replace(bag, pose_measurements=(bad_pose,))

    def test_enabled_accelerometer_adds_bias_prior_and_analytic_factor(self):
        prepared = self._prepared()
        bag = prepared.bags[0]
        bias = np.asarray((0.08, -0.04, 0.11))
        enabled_bag = replace(
            bag,
            initial_accelerometer_bias=bias,
            priors=replace(
                bag.priors,
                accelerometer_bias=_vector_prior(bias, 0.03),
            ),
            accelerometer_measurements=(
                PreparedAccelerometerMeasurement(
                    MeasurementBracket(0, 0.37),
                    np.asarray((0.3, -0.2, 9.7)),
                ),
            ),
            sensor_extrinsics=replace(
                bag.sensor_extrinsics,
                accelerometer_sensor_position_in_body=np.asarray(
                    (0.12, -0.06, 0.09)
                ),
                body_to_accelerometer_sensor_rotation=so3_exp(
                    (-0.12, 0.08, 0.05)
                ),
            ),
            covariances=replace(
                bag.covariances,
                accelerometer_observation=_covariance(3, 0.04),
            ),
            accelerometer=AccelerometerFactorContract(True, None),
        )
        enabled = replace(prepared, bags=(enabled_bag,))
        problem = build_fixed_batch_problem(enabled)
        state = build_initial_batch_state(enabled)
        bias_key = next(
            key
            for key in problem.layout.variable_keys
            if key.kind is VariableKind.ACCELEROMETER_BIAS
        )
        np.testing.assert_array_equal(state.value(bias_key), bias)
        factors = problem.evaluate_factors(state)
        accelerometer_factors = tuple(
            factor
            for factor in factors
            if any(
                block.variable_key == bias_key
                for block in factor.jacobian_blocks
            )
        )
        self.assertEqual(len(accelerometer_factors), 2)
        self.assertEqual(
            tuple(factor.residual.size for factor in accelerometer_factors),
            (3, 3),
        )
        self.assertEqual(
            tuple(
                index
                for index, factor in enumerate(factors)
                if any(
                    factor is selected
                    for selected in accelerometer_factors
                )
            ),
            (2, 14),
        )
        self.assertTrue(np.isfinite(problem.linearize(state).sparse.objective))

    def test_covariance_contract_is_spd_and_immutable(self):
        covariance = GaussianCovariance(
            np.asarray(((2.0, 0.3), (0.3, 1.4)))
        )
        whitening = covariance.square_root_information
        np.testing.assert_allclose(
            whitening.T @ whitening,
            np.linalg.inv(covariance.value),
            atol=1.0e-14,
        )
        with self.assertRaises(ValueError):
            covariance.value[0, 0] = 1.0
        with self.assertRaises(ValueError):
            whitening[0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "positive definite"):
            GaussianCovariance(np.asarray(((1.0, 2.0), (2.0, 1.0))))

    def test_disabled_observation_has_no_invented_covariance(self):
        prepared = self._prepared()
        bag = prepared.bags[0]
        without_velocity = replace(
            bag,
            velocity_measurements=(),
            covariances=replace(
                bag.covariances, velocity_observation=None
            ),
        )
        problem = build_fixed_batch_problem(
            replace(prepared, bags=(without_velocity,))
        )
        self.assertTrue(
            np.isfinite(
                problem.linearize(
                    build_initial_batch_state(
                        replace(prepared, bags=(without_velocity,))
                    )
                ).sparse.objective
            )
        )
        with self.assertRaisesRegex(ValueError, "present exactly"):
            replace(bag, velocity_measurements=())
        with self.assertRaisesRegex(ValueError, "present exactly"):
            replace(
                bag,
                covariances=replace(
                    bag.covariances, gyro_observation=None
                ),
            )

    def test_dynamics_laplace_moments_match_selected_dense_oracle(self):
        prepared = self._prepared()
        problem = build_fixed_batch_problem(prepared)
        state = build_initial_batch_state(prepared)
        sparse = problem.linearize(state).sparse
        factorization = ArrowheadLaplaceFactorization(sparse)
        intervals = evaluate_prepared_dynamics_intervals(prepared, state)
        result = compute_expected_dynamics_moments(
            intervals, factorization
        )

        self.assertEqual(intervals.valid_interval_count, 1)
        self.assertEqual(intervals.excluded_interval_count, 0)
        np.testing.assert_allclose(result.time_step, (0.02,), atol=1.0e-14)
        interval = intervals.intervals[0]
        dense_jacobian = np.zeros(
            (6, problem.layout.total_dimension), dtype=float
        )
        for block in interval.jacobian_blocks:
            dense_jacobian[
                :, problem.layout.column_slice(block.variable_key)
            ] = block.value
        expected = np.diag(
            dense_jacobian
            @ np.linalg.solve(sparse.hessian.toarray(), dense_jacobian.T)
        )
        np.testing.assert_allclose(
            result.moments.map_residual[0],
            interval.residual,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            result.moments.covariance_correction[0],
            expected,
            rtol=3.0e-11,
            atol=3.0e-11,
        )
        self.assertTrue(np.any(expected > 0.0))

    def test_invalid_dynamics_interval_is_audited_and_excluded(self):
        prepared = self._prepared()
        bag = replace(
            prepared.bags[0],
            dynamics_interval_statuses=(
                PreparedDynamicsIntervalStatus(
                    0, False, "mocap_dropout"
                ),
            ),
        )
        selected = replace(prepared, bags=(bag,))
        problem = build_fixed_batch_problem(selected)
        state = build_initial_batch_state(selected)
        factors = problem.evaluate_factors(state)
        reference_factors = build_fixed_batch_problem(
            prepared
        ).evaluate_factors(build_initial_batch_state(prepared))
        self.assertEqual(len(factors), len(reference_factors) - 1)
        self.assertEqual(factors[-1].residual.size, 3)
        intervals = evaluate_prepared_dynamics_intervals(selected, state)
        self.assertEqual(intervals.valid_interval_count, 0)
        self.assertEqual(intervals.excluded_interval_count, 1)
        self.assertEqual(
            intervals.excluded_intervals[0].reason, "mocap_dropout"
        )
        self.assertAlmostEqual(
            intervals.excluded_intervals[0].time_step, 0.02
        )

        reference = self._prepared()
        reference_problem = build_fixed_batch_problem(reference)
        reference_state = build_initial_batch_state(reference)
        factorization = ArrowheadLaplaceFactorization(
            reference_problem.linearize(reference_state).sparse
        )
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            compute_expected_dynamics_moments(intervals, factorization)

    def test_dynamics_status_is_strict_and_variable_dt_is_preserved(self):
        prepared = self._prepared()
        bag = prepared.bags[0]
        with self.assertRaisesRegex(ValueError, "cover every knot interval"):
            replace(bag, dynamics_interval_statuses=())
        with self.assertRaisesRegex(TypeError, "must be bool"):
            PreparedDynamicsIntervalStatus(0, np.bool_(True), "")
        with self.assertRaisesRegex(ValueError, "canonical reason"):
            PreparedDynamicsIntervalStatus(0, False, "")
        with self.assertRaisesRegex(ValueError, "cannot have"):
            PreparedDynamicsIntervalStatus(0, True, "not_invalid")

        knot2 = replace(bag.knots[1], time=10.05)
        controller2 = replace(
            bag.controller_intervals[0], left_knot_index=1
        )
        command = bag.actuator_intervals[0].delayed_command_segments[0].command
        actuator2 = PreparedActuatorInterval(
            1, (PreparedCommandSegment(command, 0.03),)
        )
        extended_bag = replace(
            bag,
            knots=bag.knots + (knot2,),
            controller_intervals=bag.controller_intervals + (controller2,),
            actuator_intervals=bag.actuator_intervals + (actuator2,),
            dynamics_interval_statuses=(
                PreparedDynamicsIntervalStatus(0, True, ""),
                PreparedDynamicsIntervalStatus(1, True, ""),
            ),
        )
        extended = replace(prepared, bags=(extended_bag,))
        intervals = evaluate_prepared_dynamics_intervals(
            extended, build_initial_batch_state(extended)
        )
        np.testing.assert_allclose(
            intervals.time_step, (0.02, 0.03), atol=1.0e-14
        )

    def test_laplace_bridge_rejects_a_different_bag_layout(self):
        prepared = self._prepared()
        problem = build_fixed_batch_problem(prepared)
        state = build_initial_batch_state(prepared)
        factorization = ArrowheadLaplaceFactorization(
            problem.linearize(state).sparse
        )
        other_bag = replace(prepared.bags[0], bag_id="bag-b")
        other = replace(prepared, bags=(other_bag,))
        other_intervals = evaluate_prepared_dynamics_intervals(
            other, build_initial_batch_state(other)
        )
        with self.assertRaisesRegex(ValueError, "layout does not match"):
            compute_expected_dynamics_moments(
                other_intervals, factorization
            )


if __name__ == "__main__":
    unittest.main()
