import unittest

import numpy as np

from grape_param_estim.geometry import (
    correction_transform_path,
    euler_xyz_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_matrix_from_vector,
    rotation_vector_from_matrix,
)


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
