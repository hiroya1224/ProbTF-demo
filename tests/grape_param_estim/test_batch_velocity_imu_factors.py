import unittest

import numpy as np

from grape_param_estim.batch.factors.imu import evaluate_gyro_factor
from grape_param_estim.batch.factors.velocity import (
    evaluate_world_sensor_velocity_factor,
)
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.geometry import so3_exp


class WorldSensorVelocityFactorTests(unittest.TestCase):
    def setUp(self):
        self.velocity0 = np.asarray((0.4, -0.2, 0.1))
        self.velocity1 = np.asarray((0.45, -0.16, 0.08))
        self.rotation0 = so3_exp((0.25, -0.14, 0.32))
        self.rotation1 = self.rotation0 @ so3_exp((0.07, -0.05, 0.04))
        self.omega0 = np.asarray((0.6, -0.3, 0.2))
        self.omega1 = np.asarray((0.7, -0.25, 0.16))
        self.observation = np.asarray((0.44, -0.18, 0.12))
        self.sensor_position = np.asarray((-0.0173, -0.0011, 0.057061))
        self.cog = np.asarray((-0.002, 0.001, 0.008))
        self.cog_jacobian = np.zeros((3, 18), dtype=float)
        self.cog_jacobian[:, 7:10] = np.eye(3)
        self.whitening = np.asarray(
            ((1.8, 0.1, 0.0), (0.0, 1.2, -0.1), (0.0, 0.0, 0.9))
        )
        self.alpha = 0.42

    def _evaluate(
        self,
        velocity0=None,
        velocity1=None,
        rotation0=None,
        rotation1=None,
        omega0=None,
        omega1=None,
        cog=None,
    ):
        return evaluate_world_sensor_velocity_factor(
            "bag-a",
            6,
            self.alpha,
            self.velocity0 if velocity0 is None else velocity0,
            self.velocity1 if velocity1 is None else velocity1,
            self.rotation0 if rotation0 is None else rotation0,
            self.rotation1 if rotation1 is None else rotation1,
            self.omega0 if omega0 is None else omega0,
            self.omega1 if omega1 is None else omega1,
            self.observation,
            self.sensor_position,
            self.cog if cog is None else cog,
            self.cog_jacobian,
            self.whitening,
        )

    def test_analytic_blocks_match_central_difference(self):
        factor = self._evaluate()
        expected_kinds = (
            VariableKind.LINEAR_VELOCITY,
            VariableKind.LINEAR_VELOCITY,
            VariableKind.ORIENTATION_TANGENT,
            VariableKind.ORIENTATION_TANGENT,
            VariableKind.ANGULAR_VELOCITY,
            VariableKind.ANGULAR_VELOCITY,
            VariableKind.STATIC_PARAMETERS,
        )
        self.assertEqual(
            tuple(block.variable_key.kind for block in factor.jacobian_blocks),
            expected_kinds,
        )
        step = 1.0e-7
        for block_index, block in enumerate(factor.jacobian_blocks):
            numerical = np.empty_like(block.value)
            for coordinate in range(block.value.shape[1]):
                if block_index < 2:
                    values = [self.velocity0.copy(), self.velocity1.copy()]
                    plus_values = [value.copy() for value in values]
                    minus_values = [value.copy() for value in values]
                    plus_values[block_index][coordinate] += step
                    minus_values[block_index][coordinate] -= step
                    plus = self._evaluate(
                        velocity0=plus_values[0], velocity1=plus_values[1]
                    ).residual
                    minus = self._evaluate(
                        velocity0=minus_values[0], velocity1=minus_values[1]
                    ).residual
                elif block_index < 4:
                    index = block_index - 2
                    direction = np.zeros(3, dtype=float)
                    direction[coordinate] = step
                    plus_values = [self.rotation0.copy(), self.rotation1.copy()]
                    minus_values = [self.rotation0.copy(), self.rotation1.copy()]
                    plus_values[index] = plus_values[index] @ so3_exp(direction)
                    minus_values[index] = minus_values[index] @ so3_exp(-direction)
                    plus = self._evaluate(
                        rotation0=plus_values[0], rotation1=plus_values[1]
                    ).residual
                    minus = self._evaluate(
                        rotation0=minus_values[0], rotation1=minus_values[1]
                    ).residual
                elif block_index < 6:
                    index = block_index - 4
                    values = [self.omega0.copy(), self.omega1.copy()]
                    plus_values = [value.copy() for value in values]
                    minus_values = [value.copy() for value in values]
                    plus_values[index][coordinate] += step
                    minus_values[index][coordinate] -= step
                    plus = self._evaluate(
                        omega0=plus_values[0], omega1=plus_values[1]
                    ).residual
                    minus = self._evaluate(
                        omega0=minus_values[0], omega1=minus_values[1]
                    ).residual
                else:
                    direction = self.cog_jacobian[:, coordinate] * step
                    plus = self._evaluate(cog=self.cog + direction).residual
                    minus = self._evaluate(cog=self.cog - direction).residual
                numerical[:, coordinate] = (plus - minus) / (2.0 * step)
            np.testing.assert_allclose(
                block.value, numerical, rtol=1.0e-7, atol=1.0e-8
            )


class GyroFactorTests(unittest.TestCase):
    def test_residual_blocks_and_frame_transform(self):
        omega0 = np.asarray((0.4, -0.2, 0.3))
        omega1 = np.asarray((0.5, -0.1, 0.2))
        bias = np.asarray((0.01, -0.02, 0.005))
        transform = so3_exp((0.03, -0.02, 0.01))
        whitening = np.diag((2.0, 1.5, 0.8))
        alpha = 0.35
        observation = np.asarray((0.43, -0.15, 0.28))
        factor = evaluate_gyro_factor(
            "bag-a",
            2,
            alpha,
            omega0,
            omega1,
            bias,
            observation,
            transform,
            whitening,
        )
        expected = whitening @ (
            observation
            - transform @ ((1.0 - alpha) * omega0 + alpha * omega1)
            - bias
        )
        np.testing.assert_allclose(factor.residual, expected, atol=1.0e-15)
        np.testing.assert_allclose(
            factor.jacobian_blocks[0].value,
            -(1.0 - alpha) * whitening @ transform,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            factor.jacobian_blocks[1].value,
            -alpha * whitening @ transform,
            atol=1.0e-15,
        )
        np.testing.assert_array_equal(
            factor.jacobian_blocks[2].value, -whitening
        )
        self.assertEqual(
            factor.jacobian_blocks[2].variable_key.kind,
            VariableKind.GYRO_BIAS,
        )

    def test_invalid_fraction_and_transform_are_rejected(self):
        arguments = (
            "bag-a",
            0,
            0.5,
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.eye(3),
            np.eye(3),
        )
        with self.assertRaisesRegex(ValueError, "interpolation_fraction"):
            evaluate_gyro_factor(*arguments[:2], -0.1, *arguments[3:])
        invalid = np.eye(3)
        invalid[0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            evaluate_gyro_factor(*arguments[:7], invalid, arguments[8])


if __name__ == "__main__":
    unittest.main()
