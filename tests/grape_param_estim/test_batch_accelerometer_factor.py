import unittest

import numpy as np

from grape_param_estim.batch.factors.imu import (
    evaluate_accelerometer_factor,
)
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.geometry import so3_exp


class BatchAccelerometerFactorTests(unittest.TestCase):
    def setUp(self):
        self.bag_id = "accelerometer-bag"
        self.left_index = 2
        self.alpha = 0.37
        self.dt = 0.043
        self.rotation_left = so3_exp((0.21, -0.14, 0.08))
        self.rotation_right = self.rotation_left @ so3_exp(
            (0.018, -0.011, 0.024)
        )
        self.velocity_left = np.asarray((0.42, -0.31, 0.17))
        self.velocity_right = np.asarray((0.45, -0.26, 0.21))
        self.omega_left = np.asarray((0.34, -0.27, 0.41))
        self.omega_right = np.asarray((0.39, -0.19, 0.46))
        self.bias = np.asarray((0.08, -0.05, 0.12))
        self.observation = np.asarray((1.7, -0.9, 9.2))
        self.body_to_sensor = so3_exp((-0.17, 0.12, 0.09))
        self.sensor_position = np.asarray((0.16, -0.08, 0.11))
        self.cog_offset = np.asarray((0.025, -0.014, 0.019))
        self.cog_jacobian = np.zeros((3, 18), dtype=float)
        self.cog_jacobian[:, 7:10] = np.asarray(
            ((1.0, 0.2, -0.1), (-0.3, 0.9, 0.15), (0.1, -0.2, 1.1))
        )
        self.gravity = np.asarray((0.0, 0.0, -9.80665))
        covariance = np.asarray(
            ((0.08, 0.01, 0.0), (0.01, 0.06, 0.005), (0.0, 0.005, 0.1))
        )
        self.whitening = np.linalg.inv(np.linalg.cholesky(covariance))

    def _evaluate(self, **updates):
        values = {
            "rotation_left": self.rotation_left,
            "rotation_right": self.rotation_right,
            "linear_velocity_left": self.velocity_left,
            "linear_velocity_right": self.velocity_right,
            "angular_velocity_left": self.omega_left,
            "angular_velocity_right": self.omega_right,
            "accelerometer_bias_sensor": self.bias,
            "body_to_sensor_rotation": self.body_to_sensor,
            "cog_offset_in_body": self.cog_offset,
            "time_step": self.dt,
        }
        values.update(updates)
        return evaluate_accelerometer_factor(
            bag_id=self.bag_id,
            left_knot_index=self.left_index,
            interpolation_fraction=self.alpha,
            observed_specific_force_sensor=self.observation,
            sensor_position_in_body=self.sensor_position,
            cog_offset_chart_jacobian=self.cog_jacobian,
            gravity_world=self.gravity,
            square_root_information=self.whitening,
            **values
        )

    def _finite_difference(self, key, direction, epsilon=1.0e-7):
        if key.kind is VariableKind.ORIENTATION_TANGENT:
            field = (
                "rotation_left"
                if key.knot_index == self.left_index
                else "rotation_right"
            )
            base = getattr(self, field)
            plus = self._evaluate(**{field: base @ so3_exp(epsilon * direction)})
            minus = self._evaluate(
                **{field: base @ so3_exp(-epsilon * direction)}
            )
        elif key.kind is VariableKind.LINEAR_VELOCITY:
            field = (
                "linear_velocity_left"
                if key.knot_index == self.left_index
                else "linear_velocity_right"
            )
            velocity_name = (
                "velocity_left"
                if key.knot_index == self.left_index
                else "velocity_right"
            )
            base = getattr(self, velocity_name)
            plus = self._evaluate(**{field: base + epsilon * direction})
            minus = self._evaluate(**{field: base - epsilon * direction})
        elif key.kind is VariableKind.ANGULAR_VELOCITY:
            field = (
                "angular_velocity_left"
                if key.knot_index == self.left_index
                else "angular_velocity_right"
            )
            omega_name = (
                "omega_left"
                if key.knot_index == self.left_index
                else "omega_right"
            )
            base = getattr(self, omega_name)
            plus = self._evaluate(**{field: base + epsilon * direction})
            minus = self._evaluate(**{field: base - epsilon * direction})
        elif key.kind is VariableKind.ACCELEROMETER_BIAS:
            plus = self._evaluate(
                accelerometer_bias_sensor=self.bias + epsilon * direction
            )
            minus = self._evaluate(
                accelerometer_bias_sensor=self.bias - epsilon * direction
            )
        elif key.kind is VariableKind.STATIC_PARAMETERS:
            displacement = self.cog_jacobian @ direction
            plus = self._evaluate(
                cog_offset_in_body=self.cog_offset + epsilon * displacement
            )
            minus = self._evaluate(
                cog_offset_in_body=self.cog_offset - epsilon * displacement
            )
        else:
            raise AssertionError("unexpected factor key")
        return (plus.residual - minus.residual) / (2.0 * epsilon)

    def test_analytic_jacobians_match_test_only_finite_difference_oracle(self):
        factor = self._evaluate()
        blocks = {block.variable_key: block.value for block in factor.jacobian_blocks}
        expected_keys = {
            VariableKey(
                VariableKind.ORIENTATION_TANGENT,
                self.bag_id,
                self.left_index,
            ),
            VariableKey(
                VariableKind.ORIENTATION_TANGENT,
                self.bag_id,
                self.left_index + 1,
            ),
            VariableKey(
                VariableKind.LINEAR_VELOCITY,
                self.bag_id,
                self.left_index,
            ),
            VariableKey(
                VariableKind.LINEAR_VELOCITY,
                self.bag_id,
                self.left_index + 1,
            ),
            VariableKey(
                VariableKind.ANGULAR_VELOCITY,
                self.bag_id,
                self.left_index,
            ),
            VariableKey(
                VariableKind.ANGULAR_VELOCITY,
                self.bag_id,
                self.left_index + 1,
            ),
            VariableKey(VariableKind.ACCELEROMETER_BIAS, self.bag_id),
            VariableKey(VariableKind.STATIC_PARAMETERS),
        }
        self.assertEqual(set(blocks), expected_keys)
        generator = np.random.RandomState(71)
        for key in sorted(
            expected_keys,
            key=lambda value: (
                value.kind.value,
                -1 if value.knot_index is None else value.knot_index,
            ),
        ):
            direction = generator.normal(size=key.dimension)
            direction /= np.linalg.norm(direction)
            analytic = blocks[key] @ direction
            oracle = self._finite_difference(key, direction)
            np.testing.assert_allclose(
                analytic,
                oracle,
                rtol=3.0e-6,
                atol=3.0e-6,
                err_msg=str(key),
            )

    def test_variable_time_step_and_explicit_sensor_rotation_change_model(self):
        baseline = self._evaluate().residual
        changed_dt = self._evaluate(time_step=0.061).residual
        identity_sensor_frame = self._evaluate(
            body_to_sensor_rotation=np.eye(3)
        ).residual
        self.assertGreater(np.linalg.norm(baseline - changed_dt), 1.0e-3)
        self.assertGreater(
            np.linalg.norm(baseline - identity_sensor_frame),
            1.0e-3,
        )


if __name__ == "__main__":
    unittest.main()
