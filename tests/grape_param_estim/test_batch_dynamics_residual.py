from dataclasses import replace
import unittest

import numpy as np

from grape_param_estim.batch.factors.dynamics import (
    DynamicsResidualEvaluation,
    DynamicsResidualJacobian,
    evaluate_raw_dynamics_residual,
)
from grape_param_estim.dynamics import actuator_wrench
from grape_param_estim.geometry import (
    so3_exp,
    so3_geodesic_midpoint,
)
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.system import (
    GRAVITY,
    ActuatorState,
    GrapeGeometry,
    VehicleParameters,
)


class BatchDynamicsResidualTests(unittest.TestCase):
    def setUp(self):
        nominal = VehicleParameters.nominal()
        nominal = replace(
            nominal,
            cog_offset=np.asarray((0.018, -0.013, 0.022)),
            linear_drag=np.asarray((0.21, 0.17, 0.29)),
            angular_drag=np.asarray((0.018, 0.024, 0.031)),
        )
        self.chart = VehicleParameterChart(nominal)
        self.coordinates = np.asarray(
            (
                0.08,
                0.03,
                -0.02,
                0.04,
                0.012,
                -0.009,
                0.014,
                0.006,
                -0.004,
                0.008,
                0.05,
                -0.03,
                0.02,
                -0.04,
                0.03,
                -0.02,
                0.04,
                -0.01,
            ),
            dtype=float,
        )
        self.geometry = replace(
            GrapeGeometry.grape(),
            thrust_offset=0.019,
            moment_force_rate=-0.027,
        )
        self.values = {
            "rotation_left": so3_exp((0.24, -0.11, 0.19)),
            "rotation_right": (
                so3_exp((0.24, -0.11, 0.19))
                @ so3_exp((0.31, -0.17, 0.13))
            ),
            "linear_velocity_left": np.asarray((0.38, -0.24, 0.17)),
            "linear_velocity_right": np.asarray((0.46, -0.19, 0.11)),
            "angular_velocity_left": np.asarray((0.42, -0.31, 0.27)),
            "angular_velocity_right": np.asarray((0.37, -0.22, 0.34)),
            "actuator_thrust_left": np.asarray((6.2, 5.7, 6.8, 5.9)),
            "actuator_thrust_right": np.asarray((6.5, 5.4, 7.1, 6.1)),
            "gimbal_angle_left": np.asarray((0.16, -0.21, 0.13, -0.18)),
            "gimbal_angle_right": np.asarray((0.19, -0.17, 0.09, -0.23)),
            "time_step": 0.037,
            "parameter_chart": self.chart,
            "parameter_coordinates": self.coordinates,
            "geometry": self.geometry,
        }

    def _evaluate(self, **changes):
        arguments = dict(self.values)
        arguments.update(changes)
        return evaluate_raw_dynamics_residual(**arguments)

    def test_matches_physical_required_minus_actuator_and_fixed_drag(self):
        result = self._evaluate()
        parameters = self.chart.decode(self.coordinates)
        rotation = so3_geodesic_midpoint(
            self.values["rotation_left"], self.values["rotation_right"]
        )
        velocity_midpoint = 0.5 * (
            self.values["linear_velocity_left"]
            + self.values["linear_velocity_right"]
        )
        omega_midpoint = 0.5 * (
            self.values["angular_velocity_left"]
            + self.values["angular_velocity_right"]
        )
        angular_acceleration = (
            self.values["angular_velocity_right"]
            - self.values["angular_velocity_left"]
        ) / self.values["time_step"]
        acceleration_minus_gravity = (
            (
                self.values["linear_velocity_right"]
                - self.values["linear_velocity_left"]
            )
            / self.values["time_step"]
            - np.asarray((0.0, 0.0, -GRAVITY))
        )
        required = np.concatenate(
            (
                parameters.mass * rotation.T @ acceleration_minus_gravity,
                parameters.inertia @ angular_acceleration
                + np.cross(
                    omega_midpoint,
                    parameters.inertia @ omega_midpoint,
                ),
            )
        )
        actuator_state = ActuatorState(
            thrust=0.5
            * (
                self.values["actuator_thrust_left"]
                + self.values["actuator_thrust_right"]
            ),
            gimbal_angle=0.5
            * (
                self.values["gimbal_angle_left"]
                + self.values["gimbal_angle_right"]
            ),
        )
        modeled = actuator_wrench(
            actuator_state, parameters, self.geometry
        )
        modeled[:3] -= parameters.linear_drag * (
            rotation.T @ velocity_midpoint
        )
        modeled[3:] -= parameters.angular_drag * omega_midpoint

        self.assertIsInstance(result, DynamicsResidualEvaluation)
        self.assertIsInstance(result.jacobian, DynamicsResidualJacobian)
        np.testing.assert_allclose(result.required_wrench, required, atol=1e-13)
        np.testing.assert_allclose(result.modeled_wrench, modeled, atol=1e-13)
        np.testing.assert_allclose(result.residual, required - modeled, atol=1e-13)

    def test_every_analytic_block_matches_test_only_central_difference(self):
        result = self._evaluate()
        blocks = (
            ("rotation_left", result.jacobian.rotation_left, 3),
            ("rotation_right", result.jacobian.rotation_right, 3),
            (
                "linear_velocity_left",
                result.jacobian.linear_velocity_left,
                3,
            ),
            (
                "linear_velocity_right",
                result.jacobian.linear_velocity_right,
                3,
            ),
            (
                "angular_velocity_left",
                result.jacobian.angular_velocity_left,
                3,
            ),
            (
                "angular_velocity_right",
                result.jacobian.angular_velocity_right,
                3,
            ),
            (
                "actuator_thrust_left",
                result.jacobian.actuator_thrust_left,
                4,
            ),
            (
                "actuator_thrust_right",
                result.jacobian.actuator_thrust_right,
                4,
            ),
            ("gimbal_angle_left", result.jacobian.gimbal_angle_left, 4),
            ("gimbal_angle_right", result.jacobian.gimbal_angle_right, 4),
            (
                "parameter_coordinates",
                result.jacobian.static_parameters,
                18,
            ),
        )
        step = 1.0e-7
        for name, analytic, size in blocks:
            with self.subTest(name=name):
                numerical = np.empty((6, size), dtype=float)
                baseline = np.asarray(self.values[name], dtype=float)
                for coordinate in range(size):
                    direction = np.zeros(size, dtype=float)
                    direction[coordinate] = step
                    if name.startswith("rotation_"):
                        plus = baseline @ so3_exp(direction)
                        minus = baseline @ so3_exp(-direction)
                    else:
                        plus = baseline + direction
                        minus = baseline - direction
                    numerical[:, coordinate] = (
                        self._evaluate(**{name: plus}).residual
                        - self._evaluate(**{name: minus}).residual
                    ) / (2.0 * step)
                np.testing.assert_allclose(
                    analytic,
                    numerical,
                    rtol=2.0e-5,
                    atol=3.0e-7,
                )

    def test_variable_time_step_is_used_without_resampling(self):
        short = self._evaluate(time_step=0.031)
        long = self._evaluate(time_step=0.059)
        self.assertFalse(np.allclose(short.residual, long.residual))
        self.assertFalse(
            np.allclose(
                short.jacobian.linear_velocity_left,
                long.jacobian.linear_velocity_left,
            )
        )
        self.assertFalse(
            np.allclose(
                short.jacobian.angular_velocity_right,
                long.jacobian.angular_velocity_right,
            )
        )

    def test_midpoint_branch_is_diagnosed_and_outputs_are_immutable(self):
        axis = np.asarray((0.4, -0.2, 0.7), dtype=float)
        axis /= np.linalg.norm(axis)
        left = so3_exp((0.1, -0.2, 0.05))
        right = left @ so3_exp((np.pi - 1.0e-7) * axis)
        result = self._evaluate(rotation_left=left, rotation_right=right)
        self.assertTrue(
            result.branch_diagnostics["rotation_midpoint_log_near_pi"][0]
        )
        with self.assertRaises(ValueError):
            result.residual[0] = 0.0
        with self.assertRaises(ValueError):
            result.jacobian.static_parameters[0, 0] = 0.0
        with self.assertRaises(ValueError):
            result.branch_diagnostics[
                "rotation_midpoint_log_near_pi"
            ][0] = False
        with self.assertRaises(TypeError):
            result.branch_diagnostics["new"] = np.asarray((False,))

    def test_rejects_invalid_physical_inputs(self):
        with self.assertRaisesRegex(ValueError, "time_step"):
            self._evaluate(time_step=0.0)
        with self.assertRaisesRegex(ValueError, "rotation_left"):
            self._evaluate(rotation_left=np.eye(2))
        with self.assertRaisesRegex(ValueError, "18 finite values"):
            self._evaluate(parameter_coordinates=np.zeros(17))
        with self.assertRaisesRegex(ValueError, "gravity_world"):
            self._evaluate(gravity_world=(0.0, np.nan, 0.0))
        with self.assertRaisesRegex(TypeError, "parameter_chart"):
            self._evaluate(parameter_chart=object())


if __name__ == "__main__":
    unittest.main()
