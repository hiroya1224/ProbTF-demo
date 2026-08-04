import unittest

import numpy as np

from grape_param_estim.batch.factors.kinematics import (
    evaluate_orientation_kinematic_factor,
    evaluate_position_kinematic_factor,
)
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.geometry import so3_exp


def _right_difference(base, plus, minus, step):
    return (plus - minus) / (2.0 * step)


class PositionKinematicFactorTests(unittest.TestCase):
    def test_residual_and_analytic_blocks_match_central_difference(self):
        position0 = np.asarray((0.2, -0.4, 1.1))
        position1 = np.asarray((0.23, -0.36, 1.08))
        velocity0 = np.asarray((0.5, 0.3, -0.2))
        velocity1 = np.asarray((0.4, 0.2, -0.1))
        whitening = np.asarray(
            ((2.0, 0.1, 0.0), (0.0, 1.4, -0.2), (0.0, 0.0, 0.8))
        )
        dt = 0.04

        evaluation = evaluate_position_kinematic_factor(
            "bag-a",
            8,
            position0,
            position1,
            velocity0,
            velocity1,
            dt,
            whitening,
        )
        expected = whitening @ (
            position1 - position0 - 0.5 * dt * (velocity0 + velocity1)
        )
        np.testing.assert_allclose(evaluation.residual, expected, atol=1.0e-15)
        self.assertEqual(
            [block.variable_key.kind for block in evaluation.jacobian_blocks],
            [
                VariableKind.POSITION,
                VariableKind.POSITION,
                VariableKind.LINEAR_VELOCITY,
                VariableKind.LINEAR_VELOCITY,
            ],
        )

        values = [position0, position1, velocity0, velocity1]
        step = 1.0e-7
        for value_index, block in enumerate(evaluation.jacobian_blocks):
            numerical = np.empty((3, 3), dtype=float)
            for coordinate in range(3):
                plus_values = [value.copy() for value in values]
                minus_values = [value.copy() for value in values]
                plus_values[value_index][coordinate] += step
                minus_values[value_index][coordinate] -= step
                plus = evaluate_position_kinematic_factor(
                    "bag-a", 8, *plus_values, dt, whitening
                ).residual
                minus = evaluate_position_kinematic_factor(
                    "bag-a", 8, *minus_values, dt, whitening
                ).residual
                numerical[:, coordinate] = _right_difference(
                    evaluation.residual, plus, minus, step
                )
            np.testing.assert_allclose(
                block.value, numerical, rtol=2.0e-9, atol=2.0e-9
            )


class OrientationKinematicFactorTests(unittest.TestCase):
    def test_zero_defect_for_constant_midpoint_angular_velocity(self):
        rotation0 = so3_exp((0.2, -0.1, 0.3))
        omega = np.asarray((0.4, -0.2, 0.1))
        dt = 0.025
        rotation1 = rotation0 @ so3_exp(dt * omega)
        evaluation = evaluate_orientation_kinematic_factor(
            "bag-a", 3, rotation0, rotation1, omega, omega, dt, np.eye(3)
        )
        np.testing.assert_allclose(evaluation.residual, 0.0, atol=2.0e-15)
        self.assertFalse(evaluation.active_set["rotation_log_near_pi"][0])

    def test_analytic_blocks_match_right_tangent_central_difference(self):
        rotation0 = so3_exp((0.31, -0.27, 0.14))
        rotation1 = rotation0 @ so3_exp((0.13, -0.08, 0.06))
        omega0 = np.asarray((0.7, -0.3, 0.2))
        omega1 = np.asarray((0.6, -0.2, 0.25))
        whitening = np.asarray(
            ((1.7, -0.1, 0.2), (0.0, 1.2, 0.1), (0.0, 0.0, 0.9))
        )
        dt = 0.03
        evaluation = evaluate_orientation_kinematic_factor(
            "bag-b",
            12,
            rotation0,
            rotation1,
            omega0,
            omega1,
            dt,
            whitening,
        )
        step = 1.0e-7
        for block_index, block in enumerate(evaluation.jacobian_blocks):
            numerical = np.empty((3, 3), dtype=float)
            for coordinate in range(3):
                direction = np.zeros(3, dtype=float)
                direction[coordinate] = step
                plus_rotations = [rotation0.copy(), rotation1.copy()]
                minus_rotations = [rotation0.copy(), rotation1.copy()]
                plus_omegas = [omega0.copy(), omega1.copy()]
                minus_omegas = [omega0.copy(), omega1.copy()]
                if block_index < 2:
                    plus_rotations[block_index] = (
                        plus_rotations[block_index] @ so3_exp(direction)
                    )
                    minus_rotations[block_index] = (
                        minus_rotations[block_index] @ so3_exp(-direction)
                    )
                else:
                    omega_index = block_index - 2
                    plus_omegas[omega_index][coordinate] += step
                    minus_omegas[omega_index][coordinate] -= step
                plus = evaluate_orientation_kinematic_factor(
                    "bag-b",
                    12,
                    plus_rotations[0],
                    plus_rotations[1],
                    plus_omegas[0],
                    plus_omegas[1],
                    dt,
                    whitening,
                ).residual
                minus = evaluate_orientation_kinematic_factor(
                    "bag-b",
                    12,
                    minus_rotations[0],
                    minus_rotations[1],
                    minus_omegas[0],
                    minus_omegas[1],
                    dt,
                    whitening,
                ).residual
                numerical[:, coordinate] = (plus - minus) / (2.0 * step)
            np.testing.assert_allclose(
                block.value, numerical, rtol=5.0e-8, atol=5.0e-9
            )

    def test_near_pi_branch_is_explicit_and_finite(self):
        axis = np.asarray((0.3, -0.4, 0.5), dtype=float)
        axis /= np.linalg.norm(axis)
        evaluation = evaluate_orientation_kinematic_factor(
            "bag-a",
            0,
            np.eye(3),
            so3_exp((np.pi - 2.0e-6) * axis),
            np.zeros(3),
            np.zeros(3),
            0.02,
            np.eye(3),
        )
        self.assertTrue(evaluation.active_set["rotation_log_near_pi"][0])
        self.assertTrue(np.all(np.isfinite(evaluation.residual)))
        for block in evaluation.jacobian_blocks:
            self.assertTrue(np.all(np.isfinite(block.value)))

    def test_invalid_rotation_timestep_and_whitening_are_rejected(self):
        invalid_rotation = np.eye(3)
        invalid_rotation[0, 0] = 2.0
        arguments = (
            "bag-a",
            0,
            np.eye(3),
            np.eye(3),
            np.zeros(3),
            np.zeros(3),
            0.02,
            np.eye(3),
        )
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            evaluate_orientation_kinematic_factor(
                *arguments[:2], invalid_rotation, *arguments[3:]
            )
        with self.assertRaisesRegex(ValueError, "time_step"):
            evaluate_orientation_kinematic_factor(
                *arguments[:6], 0.0, arguments[7]
            )
        with self.assertRaisesRegex(ValueError, "full rank"):
            evaluate_orientation_kinematic_factor(
                *arguments[:7], np.zeros((3, 3))
            )


if __name__ == "__main__":
    unittest.main()
