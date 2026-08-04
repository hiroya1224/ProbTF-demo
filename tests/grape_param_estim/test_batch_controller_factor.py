import unittest

import numpy as np

from grape_param_estim.batch.factors.controller import (
    evaluate_controller_integral_observation_factor,
    evaluate_controller_step_factors,
)
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.controller import ControllerConfig, GrapeController
from grape_param_estim.geometry import euler_xyz_to_matrix, so3_exp
from grape_param_estim.system import (
    ControllerState,
    GrapeGeometry,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


class BatchControllerFactorTests(unittest.TestCase):
    def test_asynchronous_integral_proxy_has_exact_linear_blocks(self):
        left = np.arange(6, dtype=float) * 0.1
        right = left + 0.4
        observed = left + 0.31
        whitening = np.diag(np.linspace(0.7, 1.2, 6))
        factor = evaluate_controller_integral_observation_factor(
            bag_id="bag-a",
            left_knot_index=4,
            interpolation_fraction=0.25,
            integral_left=left,
            integral_right=right,
            observed_integral=observed,
            square_root_information=whitening,
        )

        np.testing.assert_allclose(
            factor.residual,
            whitening @ (observed - 0.75 * left - 0.25 * right),
        )
        np.testing.assert_allclose(
            factor.jacobian_blocks[0].value, -0.75 * whitening
        )
        np.testing.assert_allclose(
            factor.jacobian_blocks[1].value, -0.25 * whitening
        )
        self.assertEqual(
            factor.jacobian_blocks[1].variable_key.knot_index, 5
        )

    def setUp(self):
        self.controller = GrapeController(
            ControllerConfig.grape(),
            VehicleParameters.nominal(),
            GrapeGeometry.grape(),
        )
        self.position = np.asarray((0.12, -0.08, 1.03))
        self.rotation = euler_xyz_to_matrix((0.04, -0.03, 0.12))
        self.velocity = np.asarray((0.03, -0.02, 0.01))
        self.omega = np.asarray((0.02, -0.01, 0.03))
        self.integral0 = np.asarray((0.01, -0.02, 0.03, 0.0, 0.01, -0.01))
        self.gimbal = np.asarray((0.03, -0.04, 0.02, -0.01))
        self.reference = ReferenceState(
            position=np.asarray((0.18, -0.12, 1.08)),
            linear_velocity=np.asarray((0.02, 0.01, -0.03)),
            linear_acceleration=np.asarray((0.12, -0.08, 0.04)),
            rpy=np.asarray((0.08, -0.01, 0.15)),
            angular_velocity=np.asarray((0.02, -0.03, 0.04)),
            angular_acceleration=np.asarray((0.01, -0.02, 0.015)),
        )
        state = RigidBodyState(
            self.position,
            np.asarray((0.0218476, -0.013777, 0.0602467, 0.997848)),
            self.velocity,
            self.omega,
        )
        forward = self.controller.step_with_jacobian(
            state,
            self.reference,
            ControllerState(self.integral0, True),
            0.02,
            self.gimbal,
        )
        self.integral1 = (
            forward.next_state.integral_error
            + np.asarray((0.001, -0.002, 0.003, -0.001, 0.002, -0.003))
        )
        self.observed_thrust = (
            forward.command.thrust + np.asarray((0.01, -0.02, 0.03, -0.04))
        )
        self.observed_gimbal = (
            forward.command.gimbal_angle
            + np.asarray((0.002, -0.003, 0.004, -0.005))
        )
        self.integral_whitening = np.diag((1.2, 1.1, 0.9, 1.3, 0.8, 1.4))
        self.command_whitening = np.diag((2.0, 1.8, 2.2, 1.6))

    def _evaluate(self, **changes):
        values = {
            "position": self.position,
            "rotation": self.rotation,
            "world_velocity": self.velocity,
            "body_omega": self.omega,
            "integral_left": self.integral0,
            "integral_right": self.integral1,
            "current_gimbal_angle": self.gimbal,
        }
        values.update(changes)
        return evaluate_controller_step_factors(
            controller=self.controller,
            bag_id="bag-a",
            knot_index=7,
            reference=self.reference,
            time_step=0.02,
            roll_pitch_integration_active=True,
            integral_square_root_information=self.integral_whitening,
            issued_thrust_observation=self.observed_thrust,
            issued_thrust_square_root_information=self.command_whitening,
            issued_gimbal_observation=self.observed_gimbal,
            issued_gimbal_square_root_information=self.command_whitening,
            **values
        )

    def test_returns_separate_covariance_factors_and_fixed_schedule(self):
        result = self._evaluate()
        self.assertEqual(len(result.factors), 3)
        self.assertEqual(result.integral_transition.residual.shape, (6,))
        self.assertEqual(result.issued_thrust.residual.shape, (4,))
        self.assertEqual(result.issued_gimbal_angle.residual.shape, (4,))
        self.assertTrue(result.next_roll_pitch_integration_active)
        self.assertIn(
            "pid_output_sum_saturated",
            result.integral_transition.active_set,
        )
        self.assertIn(
            "roll_pitch_integration_active",
            result.integral_transition.active_set,
        )

    def test_analytic_blocks_match_test_only_central_differences(self):
        result = self._evaluate()
        names = (
            "position",
            "rotation",
            "world_velocity",
            "body_omega",
            "integral_left",
            "current_gimbal_angle",
            "integral_right",
        )
        sizes = (3, 3, 3, 3, 6, 4, 6)
        step = 1.0e-7
        for factor in result.factors:
            factor_names = names if factor.residual.size == 6 else names[:-1]
            factor_sizes = sizes if factor.residual.size == 6 else sizes[:-1]
            for block, name, size in zip(
                factor.jacobian_blocks,
                factor_names,
                factor_sizes,
            ):
                numerical = np.empty((factor.residual.size, size), dtype=float)
                for coordinate in range(size):
                    plus = getattr(self, {
                        "world_velocity": "velocity",
                        "body_omega": "omega",
                        "integral_left": "integral0",
                        "integral_right": "integral1",
                        "current_gimbal_angle": "gimbal",
                    }.get(name, name)).copy()
                    minus = plus.copy()
                    if name == "rotation":
                        direction = np.zeros(3)
                        direction[coordinate] = step
                        plus = self.rotation @ so3_exp(direction)
                        minus = self.rotation @ so3_exp(-direction)
                    else:
                        plus[coordinate] += step
                        minus[coordinate] -= step
                    numerical[:, coordinate] = (
                        getattr(self._evaluate(**{name: plus}), {
                            6: "integral_transition",
                        }.get(factor.residual.size, (
                            "issued_thrust"
                            if factor is result.issued_thrust
                            else "issued_gimbal_angle"
                        ))).residual
                        - getattr(self._evaluate(**{name: minus}), {
                            6: "integral_transition",
                        }.get(factor.residual.size, (
                            "issued_thrust"
                            if factor is result.issued_thrust
                            else "issued_gimbal_angle"
                        ))).residual
                    ) / (2.0 * step)
                np.testing.assert_allclose(
                    block.value,
                    numerical,
                    rtol=2.0e-6,
                    atol=2.0e-7,
                )

        self.assertEqual(
            result.integral_transition.jacobian_blocks[-1].variable_key.kind,
            VariableKind.CONTROLLER_INTEGRAL,
        )
        self.assertEqual(
            result.integral_transition.jacobian_blocks[-1]
            .variable_key.knot_index,
            8,
        )

    def test_optional_observations_must_have_matching_covariance(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            evaluate_controller_step_factors(
                controller=self.controller,
                bag_id="bag-a",
                knot_index=7,
                position=self.position,
                rotation=self.rotation,
                world_velocity=self.velocity,
                body_omega=self.omega,
                integral_left=self.integral0,
                integral_right=self.integral1,
                current_gimbal_angle=self.gimbal,
                reference=self.reference,
                time_step=0.02,
                roll_pitch_integration_active=True,
                integral_square_root_information=self.integral_whitening,
                issued_thrust_observation=self.observed_thrust,
            )


if __name__ == "__main__":
    unittest.main()
