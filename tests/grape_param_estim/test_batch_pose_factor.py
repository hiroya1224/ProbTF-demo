import unittest

import numpy as np

from grape_param_estim.batch.factors.pose import (
    evaluate_pose_observation_factors,
)
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.geometry import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    so3_exp,
)


class PoseObservationFactorTests(unittest.TestCase):
    def setUp(self):
        self.bag_id = "flight-a"
        self.knot_index = 4
        self.alpha = 0.37
        self.position0 = np.asarray((0.2, -0.3, 0.9))
        self.position1 = np.asarray((0.24, -0.27, 0.93))
        self.rotation0 = so3_exp((0.21, -0.16, 0.31))
        self.rotation1 = self.rotation0 @ so3_exp((0.08, -0.04, 0.07))
        self.observed_position = np.asarray((0.25, -0.28, 0.98))
        self.observed_rotation = self.rotation0 @ so3_exp((0.05, 0.01, 0.04))
        self.sensor_position = np.asarray((-0.0173, -0.0011, 0.057061))
        self.body_to_sensor = so3_exp((0.015, -0.02, 0.01))
        self.cog = np.asarray((-0.002, 0.001, 0.008))
        self.cog_jacobian = np.zeros((3, 18), dtype=float)
        self.cog_jacobian[:, 7:10] = np.eye(3)
        self.position_whitening = np.asarray(
            ((2.0, 0.1, 0.0), (0.0, 1.5, -0.1), (0.0, 0.0, 0.8))
        )
        self.orientation_whitening = np.asarray(
            ((1.4, 0.0, 0.1), (0.0, 1.1, -0.1), (0.0, 0.0, 0.9))
        )

    def _evaluate(
        self,
        position0=None,
        position1=None,
        rotation0=None,
        rotation1=None,
        cog=None,
        observed_rotation=None,
    ):
        return evaluate_pose_observation_factors(
            self.bag_id,
            self.knot_index,
            self.alpha,
            self.position0 if position0 is None else position0,
            self.position1 if position1 is None else position1,
            self.rotation0 if rotation0 is None else rotation0,
            self.rotation1 if rotation1 is None else rotation1,
            self.observed_position,
            (
                self.observed_rotation
                if observed_rotation is None
                else observed_rotation
            ),
            self.sensor_position,
            self.body_to_sensor,
            self.cog if cog is None else cog,
            self.cog_jacobian,
            self.position_whitening,
            self.orientation_whitening,
        )

    def test_factor_keys_keep_position_orientation_and_static_blocks_sparse(self):
        position_factor, orientation_factor = self._evaluate()
        self.assertEqual(
            [block.variable_key.kind for block in position_factor.jacobian_blocks],
            [
                VariableKind.POSITION,
                VariableKind.POSITION,
                VariableKind.ORIENTATION_TANGENT,
                VariableKind.ORIENTATION_TANGENT,
                VariableKind.STATIC_PARAMETERS,
            ],
        )
        self.assertEqual(
            [
                block.variable_key.kind
                for block in orientation_factor.jacobian_blocks
            ],
            [
                VariableKind.ORIENTATION_TANGENT,
                VariableKind.ORIENTATION_TANGENT,
            ],
        )
        self.assertEqual(position_factor.jacobian_blocks[-1].value.shape, (3, 18))
        self.assertFalse(
            orientation_factor.active_set["rotation_log_near_pi"][0]
        )

    def test_position_factor_analytic_blocks_match_central_difference(self):
        factor, _ = self._evaluate()
        step = 1.0e-7
        for block_index, block in enumerate(factor.jacobian_blocks):
            numerical = np.empty_like(block.value)
            for coordinate in range(block.value.shape[1]):
                if block_index < 2:
                    plus_positions = [self.position0.copy(), self.position1.copy()]
                    minus_positions = [self.position0.copy(), self.position1.copy()]
                    plus_positions[block_index][coordinate] += step
                    minus_positions[block_index][coordinate] -= step
                    plus = self._evaluate(
                        position0=plus_positions[0], position1=plus_positions[1]
                    )[0].residual
                    minus = self._evaluate(
                        position0=minus_positions[0], position1=minus_positions[1]
                    )[0].residual
                elif block_index < 4:
                    rotation_index = block_index - 2
                    direction = np.zeros(3, dtype=float)
                    direction[coordinate] = step
                    plus_rotations = [self.rotation0.copy(), self.rotation1.copy()]
                    minus_rotations = [self.rotation0.copy(), self.rotation1.copy()]
                    plus_rotations[rotation_index] = (
                        plus_rotations[rotation_index] @ so3_exp(direction)
                    )
                    minus_rotations[rotation_index] = (
                        minus_rotations[rotation_index] @ so3_exp(-direction)
                    )
                    plus = self._evaluate(
                        rotation0=plus_rotations[0], rotation1=plus_rotations[1]
                    )[0].residual
                    minus = self._evaluate(
                        rotation0=minus_rotations[0], rotation1=minus_rotations[1]
                    )[0].residual
                else:
                    chart_direction = self.cog_jacobian[:, coordinate] * step
                    plus = self._evaluate(cog=self.cog + chart_direction)[0].residual
                    minus = self._evaluate(cog=self.cog - chart_direction)[0].residual
                numerical[:, coordinate] = (plus - minus) / (2.0 * step)
            np.testing.assert_allclose(
                block.value, numerical, rtol=8.0e-8, atol=8.0e-9
            )

    def test_orientation_factor_analytic_blocks_match_central_difference(self):
        _, factor = self._evaluate()
        step = 1.0e-7
        for block_index, block in enumerate(factor.jacobian_blocks):
            numerical = np.empty((3, 3), dtype=float)
            for coordinate in range(3):
                direction = np.zeros(3, dtype=float)
                direction[coordinate] = step
                plus_rotations = [self.rotation0.copy(), self.rotation1.copy()]
                minus_rotations = [self.rotation0.copy(), self.rotation1.copy()]
                plus_rotations[block_index] = (
                    plus_rotations[block_index] @ so3_exp(direction)
                )
                minus_rotations[block_index] = (
                    minus_rotations[block_index] @ so3_exp(-direction)
                )
                plus = self._evaluate(
                    rotation0=plus_rotations[0], rotation1=plus_rotations[1]
                )[1].residual
                minus = self._evaluate(
                    rotation0=minus_rotations[0], rotation1=minus_rotations[1]
                )[1].residual
                numerical[:, coordinate] = (plus - minus) / (2.0 * step)
            np.testing.assert_allclose(
                block.value, numerical, rtol=8.0e-8, atol=8.0e-9
            )

    def test_quaternion_sign_does_not_enter_rotation_factor(self):
        _, baseline = self._evaluate()
        quaternion = matrix_to_quaternion(self.observed_rotation)
        positive_matrix = quaternion_to_matrix(quaternion)
        negative_matrix = quaternion_to_matrix(-quaternion)
        _, positive = self._evaluate(observed_rotation=positive_matrix)
        _, negative = self._evaluate(observed_rotation=negative_matrix)
        np.testing.assert_array_equal(positive_matrix, negative_matrix)
        np.testing.assert_array_equal(positive.residual, negative.residual)
        np.testing.assert_allclose(positive.residual, baseline.residual, atol=1.0e-15)

    def test_invalid_fraction_rotation_chart_and_covariance_are_rejected(self):
        arguments = (
            self.bag_id,
            self.knot_index,
            self.alpha,
            self.position0,
            self.position1,
            self.rotation0,
            self.rotation1,
            self.observed_position,
            self.observed_rotation,
            self.sensor_position,
            self.body_to_sensor,
            self.cog,
            self.cog_jacobian,
            self.position_whitening,
            self.orientation_whitening,
        )
        with self.assertRaisesRegex(ValueError, "interpolation_fraction"):
            evaluate_pose_observation_factors(
                *arguments[:2], 1.1, *arguments[3:]
            )
        invalid_rotation = self.rotation0.copy()
        invalid_rotation[0, 0] *= 2.0
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            evaluate_pose_observation_factors(
                *arguments[:5], invalid_rotation, *arguments[6:]
            )
        with self.assertRaisesRegex(ValueError, "3 by 18"):
            evaluate_pose_observation_factors(
                *arguments[:12], np.zeros((3, 17)), *arguments[13:]
            )
        with self.assertRaisesRegex(ValueError, "full rank"):
            evaluate_pose_observation_factors(
                *arguments[:13], np.zeros((3, 3)), arguments[14]
            )


if __name__ == "__main__":
    unittest.main()
