import unittest

import numpy as np

from grape_param_estim.controller import (
    ControllerConfig,
    GrapeController,
    initial_controller_state,
)
from grape_param_estim.real_calibration import (
    calibrate_model_error_from_closed_loop_pose,
    calibrate_model_error_from_pose,
    pose_derived_initial_state,
    select_ou_knot_resolution,
)
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    ClosedLoopTrajectory,
    GrapeGeometry,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


def _trajectory(times, position, velocity, omega=None):
    count = times.size
    return ClosedLoopTrajectory(
        times=times,
        position=position,
        orientation_xyzw=np.tile((0.0, 0.0, 0.0, 1.0), (count, 1)),
        linear_velocity=velocity,
        angular_velocity=(np.zeros((count, 3)) if omega is None else omega),
        controller_integral=np.zeros((count, 6)),
        commanded_thrust=np.ones((count, 4)),
        commanded_gimbal_angle=np.zeros((count, 4)),
        actuator_thrust=np.ones((count, 4)),
        actuator_gimbal_angle=np.zeros((count, 4)),
        body_wrench=np.zeros((count, 6)),
    )


class RealCalibrationTest(unittest.TestCase):
    def test_pose_only_pilot_recovers_constant_force_scale(self):
        times = np.linspace(0.0, 8.0, 81)
        acceleration = np.asarray((0.35, -0.20, 0.10))
        observed_position = 0.5 * times[:, None] ** 2 * acceleration
        nominal = _trajectory(
            times,
            np.zeros((times.size, 3)),
            np.zeros((times.size, 3)),
        )
        result = calibrate_model_error_from_pose(
            times,
            observed_position,
            np.tile((0.0, 0.0, 0.0, 1.0), (times.size, 1)),
            nominal,
            VehicleParameters.nominal(),
        )
        expected = VehicleParameters.nominal().mass * acceleration
        np.testing.assert_allclose(
            result.pilot_location[:3], expected, rtol=2.0e-5
        )
        self.assertTrue(np.all(result.stationary_standard_deviation[:3] > 0.0))
        self.assertTrue(np.all(result.stationary_standard_deviation[3:] > 0.0))
        self.assertGreater(result.correlation_time, 0.0)
        self.assertEqual(result.proxy_wrench.shape, (times.size, 6))

    def test_closed_loop_pose_calibration_is_zero_for_nominal_hover(self):
        times = np.linspace(0.0, 2.0, 21)
        position = np.zeros((times.size, 3))
        orientation = np.tile((0.0, 0.0, 0.0, 1.0), (times.size, 1))
        references = tuple(
            ReferenceState(
                position=np.zeros(3),
                linear_velocity=np.zeros(3),
                linear_acceleration=np.zeros(3),
                rpy=np.zeros(3),
                angular_velocity=np.zeros(3),
                angular_acceleration=np.zeros(3),
            )
            for _time in times
        )
        configuration = ControllerConfig.grape()
        controller_state = initial_controller_state(
            configuration, trim_hover=True
        )
        parameters = VehicleParameters.nominal()
        geometry = GrapeGeometry.grape()
        command, _next = GrapeController(
            configuration, parameters, geometry
        ).step(
            RigidBodyState(
                np.zeros(3),
                np.asarray((0.0, 0.0, 0.0, 1.0)),
                np.zeros(3),
                np.zeros(3),
            ),
            references[0],
            controller_state,
            times[1] - times[0],
        )
        calibration = calibrate_model_error_from_closed_loop_pose(
            times,
            position,
            orientation,
            references,
            configuration,
            controller_state,
            ActuatorState(command.thrust, command.gimbal_angle),
            ActuatorParameters(),
            parameters,
            geometry,
        )
        np.testing.assert_allclose(
            calibration.proxy_wrench[calibration.valid_mask],
            0.0,
            atol=2.0e-12,
        )
        self.assertEqual(
            calibration.method,
            "pose-only-counterfactual-closed-loop-wrench/v1",
        )

    def test_required_knots_obey_ou_bridge_bound(self):
        times = np.linspace(0.0, 10.0, 101)
        result = select_ou_knot_resolution(times, correlation_time=2.0)
        self.assertEqual(result.knot_indices[0], 0)
        self.assertEqual(result.knot_indices[-1], times.size - 1)
        self.assertTrue(result.resolution_sufficient)
        self.assertLessEqual(
            result.achieved_maximum_gap,
            result.maximum_bridge_gap * (1.0 + 1.0e-12),
        )

    def test_initial_latent_anchor_uses_only_pose_derivatives(self):
        times = np.linspace(0.0, 3.0, 31)
        velocity = np.asarray((0.4, -0.2, 0.1))
        initial_position = np.asarray((1.0, 2.0, 3.0))
        position = initial_position + times[:, None] * velocity
        orientation = np.tile((0.0, 0.0, 0.0, 1.0), (times.size, 1))
        state = pose_derived_initial_state(times, position, orientation)
        np.testing.assert_allclose(state.position, initial_position)
        np.testing.assert_allclose(state.linear_velocity, velocity, atol=1.0e-13)
        np.testing.assert_array_equal(state.angular_velocity, 0.0)

    def test_budget_limit_is_explicitly_insufficient(self):
        times = np.linspace(0.0, 20.0, 201)
        result = select_ou_knot_resolution(
            times, correlation_time=0.5, maximum_knots=5
        )
        self.assertEqual(result.knot_indices.size, 5)
        self.assertGreater(result.required_knot_count, 5)
        self.assertFalse(result.resolution_sufficient)
        self.assertGreater(
            result.achieved_maximum_gap, result.maximum_bridge_gap
        )

    def test_invalid_maximum_knots_is_rejected(self):
        with self.assertRaises(ValueError):
            select_ou_knot_resolution(
                np.asarray((0.0, 1.0)), 1.0, maximum_knots=1
            )


if __name__ == "__main__":
    unittest.main()
