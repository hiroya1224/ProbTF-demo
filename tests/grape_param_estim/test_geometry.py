import unittest

import numpy as np

from grape_param_estim.geometry import (
    correction_transform_path,
    euler_xyz_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    right_tangent_rotation_action_jacobian,
    rotation_matrix_from_vector,
    rotation_vector_from_matrix,
    so3_exp,
    so3_geodesic_midpoint,
    so3_geodesic_midpoint_with_right_jacobians,
    so3_left_jacobian,
    so3_left_jacobian_inverse,
    so3_log,
    so3_right_jacobian,
    so3_right_jacobian_inverse,
)


def _vee(matrix):
    return 0.5 * np.asarray(
        (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ),
        dtype=float,
    )


def _right_tangent_matrix_derivative(base, plus, minus, step):
    matrix_derivative = (plus - minus) / (2.0 * step)
    return _vee(base.T @ matrix_derivative)


def _left_tangent_matrix_derivative(base, plus, minus, step):
    matrix_derivative = (plus - minus) / (2.0 * step)
    return _vee(matrix_derivative @ base.T)


class GeometryTests(unittest.TestCase):
    def test_rotation_log_exp_and_quaternion_round_trip(self):
        for vector in (
            np.asarray((1.0e-10, -2.0e-10, 3.0e-10)),
            np.asarray((0.2, -0.3, 0.4)),
            np.asarray((np.pi - 1.0e-6, 0.0, 0.0)),
        ):
            with self.subTest(vector=vector):
                rotation = rotation_matrix_from_vector(vector)
                recovered = rotation_vector_from_matrix(rotation)
                np.testing.assert_allclose(
                    rotation_matrix_from_vector(recovered),
                    rotation,
                    atol=1.0e-9,
                )
                np.testing.assert_allclose(
                    quaternion_to_matrix(matrix_to_quaternion(rotation)),
                    rotation,
                    atol=1.0e-12,
                )

    def test_exp_and_log_wrappers_share_the_so3_source_of_truth(self):
        vector = np.asarray((0.31, -0.27, 0.18))
        rotation = so3_exp(vector)
        np.testing.assert_array_equal(
            rotation_matrix_from_vector(vector),
            rotation,
        )
        np.testing.assert_array_equal(
            rotation_vector_from_matrix(rotation),
            so3_log(rotation),
        )

    def test_exp_jacobians_match_central_differences(self):
        rng = np.random.RandomState(45817)
        vectors = [
            rng.normal(size=3) * rng.uniform(0.2, 0.8)
            for _ in range(6)
        ]
        vectors.extend(
            (
                np.asarray((1.0e-11, -2.0e-11, 3.0e-11)),
                (np.pi - 2.0e-4)
                * np.asarray((0.3, -0.4, 0.5))
                / np.linalg.norm((0.3, -0.4, 0.5)),
            )
        )
        step = 1.0e-7
        for vector in vectors:
            with self.subTest(vector=vector):
                rotation = so3_exp(vector)
                numerical_left = np.empty((3, 3), dtype=float)
                numerical_right = np.empty((3, 3), dtype=float)
                for coordinate in range(3):
                    direction = np.zeros(3, dtype=float)
                    direction[coordinate] = step
                    plus = so3_exp(vector + direction)
                    minus = so3_exp(vector - direction)
                    numerical_left[:, coordinate] = (
                        _left_tangent_matrix_derivative(
                            rotation, plus, minus, step
                        )
                    )
                    numerical_right[:, coordinate] = (
                        _right_tangent_matrix_derivative(
                            rotation, plus, minus, step
                        )
                    )
                np.testing.assert_allclose(
                    so3_left_jacobian(vector),
                    numerical_left,
                    rtol=2.0e-7,
                    atol=2.0e-8,
                )
                np.testing.assert_allclose(
                    so3_right_jacobian(vector),
                    numerical_right,
                    rtol=2.0e-7,
                    atol=2.0e-8,
                )

    def test_log_jacobian_inverses_match_central_differences(self):
        rng = np.random.RandomState(7013)
        vectors = [rng.normal(size=3) * 0.55 for _ in range(5)]
        axis = np.asarray((0.2, 0.7, -0.3), dtype=float)
        axis /= np.linalg.norm(axis)
        vectors.extend(
            (
                np.asarray((2.0e-11, -3.0e-11, 1.0e-11)),
                (np.pi - 5.0e-4) * axis,
            )
        )
        step = 1.0e-6
        for vector in vectors:
            with self.subTest(vector=vector):
                rotation = so3_exp(vector)
                numerical_left = np.empty((3, 3), dtype=float)
                numerical_right = np.empty((3, 3), dtype=float)
                for coordinate in range(3):
                    direction = np.zeros(3, dtype=float)
                    direction[coordinate] = step
                    numerical_left[:, coordinate] = (
                        so3_log(so3_exp(direction) @ rotation)
                        - so3_log(so3_exp(-direction) @ rotation)
                    ) / (2.0 * step)
                    numerical_right[:, coordinate] = (
                        so3_log(rotation @ so3_exp(direction))
                        - so3_log(rotation @ so3_exp(-direction))
                    ) / (2.0 * step)
                np.testing.assert_allclose(
                    so3_left_jacobian_inverse(vector),
                    numerical_left,
                    rtol=2.0e-6,
                    atol=2.0e-7,
                )
                np.testing.assert_allclose(
                    so3_right_jacobian_inverse(vector),
                    numerical_right,
                    rtol=2.0e-6,
                    atol=2.0e-7,
                )
                np.testing.assert_allclose(
                    so3_left_jacobian(vector)
                    @ so3_left_jacobian_inverse(vector),
                    np.eye(3),
                    atol=2.0e-12,
                )
                np.testing.assert_allclose(
                    so3_right_jacobian(vector)
                    @ so3_right_jacobian_inverse(vector),
                    np.eye(3),
                    atol=2.0e-12,
                )

    def test_right_tangent_rotation_action_jacobian(self):
        rotation = so3_exp((0.35, -0.21, 0.44))
        vector = np.asarray((0.7, -1.2, 0.3))
        step = 1.0e-7
        numerical = np.empty((3, 3), dtype=float)
        for coordinate in range(3):
            direction = np.zeros(3, dtype=float)
            direction[coordinate] = step
            numerical[:, coordinate] = (
                rotation @ so3_exp(direction) @ vector
                - rotation @ so3_exp(-direction) @ vector
            ) / (2.0 * step)
        np.testing.assert_allclose(
            right_tangent_rotation_action_jacobian(rotation, vector),
            numerical,
            rtol=2.0e-8,
            atol=2.0e-9,
        )

    def test_geodesic_midpoint_endpoint_jacobians(self):
        left = so3_exp((0.25, -0.12, 0.31))
        axis = np.asarray((-0.4, 0.2, 0.7), dtype=float)
        axis /= np.linalg.norm(axis)
        relative_vectors = (
            np.asarray((1.0e-10, -2.0e-10, 3.0e-10)),
            np.asarray((0.8, -0.5, 0.35)),
            (np.pi - 5.0e-4) * axis,
        )
        step = 1.0e-7
        for relative_vector in relative_vectors:
            with self.subTest(relative_vector=relative_vector):
                right = left @ so3_exp(relative_vector)
                midpoint, left_block, right_block = (
                    so3_geodesic_midpoint_with_right_jacobians(left, right)
                )
                np.testing.assert_allclose(
                    midpoint,
                    so3_geodesic_midpoint(left, right),
                    atol=1.0e-14,
                )
                numerical_left = np.empty((3, 3), dtype=float)
                numerical_right = np.empty((3, 3), dtype=float)
                for coordinate in range(3):
                    direction = np.zeros(3, dtype=float)
                    direction[coordinate] = step
                    left_plus = so3_geodesic_midpoint(
                        left @ so3_exp(direction), right
                    )
                    left_minus = so3_geodesic_midpoint(
                        left @ so3_exp(-direction), right
                    )
                    numerical_left[:, coordinate] = (
                        _right_tangent_matrix_derivative(
                            midpoint, left_plus, left_minus, step
                        )
                    )
                    right_plus = so3_geodesic_midpoint(
                        left, right @ so3_exp(direction)
                    )
                    right_minus = so3_geodesic_midpoint(
                        left, right @ so3_exp(-direction)
                    )
                    numerical_right[:, coordinate] = (
                        _right_tangent_matrix_derivative(
                            midpoint, right_plus, right_minus, step
                        )
                    )
                np.testing.assert_allclose(
                    left_block,
                    numerical_left,
                    rtol=8.0e-6,
                    atol=8.0e-7,
                )
                np.testing.assert_allclose(
                    right_block,
                    numerical_right,
                    rtol=8.0e-6,
                    atol=8.0e-7,
                )

    def test_near_pi_operations_remain_finite(self):
        axis = np.asarray((0.37, -0.51, 0.78), dtype=float)
        axis /= np.linalg.norm(axis)
        for angle in (np.pi - 1.0e-10, np.pi):
            with self.subTest(angle=angle):
                vector = angle * axis
                rotation = so3_exp(vector)
                recovered = so3_log(rotation)
                midpoint, left_block, right_block = (
                    so3_geodesic_midpoint_with_right_jacobians(
                        np.eye(3), rotation
                    )
                )
                values = (
                    rotation,
                    recovered,
                    so3_left_jacobian(vector),
                    so3_right_jacobian(vector),
                    so3_left_jacobian_inverse(vector),
                    so3_right_jacobian_inverse(vector),
                    midpoint,
                    left_block,
                    right_block,
                )
                for value in values:
                    self.assertTrue(np.all(np.isfinite(value)))
                np.testing.assert_allclose(
                    so3_exp(recovered), rotation, atol=2.0e-9
                )

    def test_so3_primitives_validate_finite_shapes(self):
        vector_functions = (
            so3_exp,
            so3_left_jacobian,
            so3_right_jacobian,
            so3_left_jacobian_inverse,
            so3_right_jacobian_inverse,
        )
        for function in vector_functions:
            with self.subTest(function=function.__name__, failure="shape"):
                with self.assertRaises(ValueError):
                    function((0.0, 0.0))
            with self.subTest(function=function.__name__, failure="finite"):
                with self.assertRaises(ValueError):
                    function((0.0, np.nan, 0.0))

        with self.assertRaises(ValueError):
            so3_log(np.eye(2))
        invalid_rotation = np.eye(3)
        invalid_rotation[0, 0] = np.inf
        with self.assertRaises(ValueError):
            so3_log(invalid_rotation)
        with self.assertRaises(ValueError):
            right_tangent_rotation_action_jacobian(np.eye(2), np.ones(3))
        with self.assertRaises(ValueError):
            right_tangent_rotation_action_jacobian(np.eye(3), np.ones(2))
        with self.assertRaises(ValueError):
            so3_geodesic_midpoint(np.eye(2), np.eye(3))
        with self.assertRaises(ValueError):
            so3_geodesic_midpoint_with_right_jacobians(
                np.eye(3), invalid_rotation
            )

    def test_correction_path_reconstructs_candidate_pose(self):
        count = 20
        nominal_position = np.column_stack(
            (
                np.linspace(0.0, 1.0, count),
                np.linspace(0.2, -0.1, count),
                np.linspace(1.0, 1.2, count),
            )
        )
        candidate_position = nominal_position + np.asarray((0.1, -0.2, 0.3))
        nominal_orientation = np.asarray(
            [
                matrix_to_quaternion(
                    euler_xyz_to_matrix((0.01 * i, -0.005 * i, 0.02 * i))
                )
                for i in range(count)
            ]
        )
        candidate_orientation = np.asarray(
            [
                matrix_to_quaternion(
                    quaternion_to_matrix(nominal_orientation[i])
                    @ rotation_matrix_from_vector((0.02, -0.01, 0.03))
                )
                for i in range(count)
            ]
        )
        translation, rotation_vector = correction_transform_path(
            nominal_position,
            nominal_orientation,
            candidate_position,
            candidate_orientation,
        )
        for index in range(count):
            nominal_rotation = quaternion_to_matrix(
                nominal_orientation[index]
            )
            recovered_position = (
                nominal_position[index]
                + nominal_rotation @ translation[index]
            )
            recovered_rotation = nominal_rotation @ (
                rotation_matrix_from_vector(rotation_vector[index])
            )
            np.testing.assert_allclose(
                recovered_position, candidate_position[index], atol=1.0e-12
            )
            np.testing.assert_allclose(
                recovered_rotation,
                quaternion_to_matrix(candidate_orientation[index]),
                atol=1.0e-10,
            )


if __name__ == "__main__":
    unittest.main()
