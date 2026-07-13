import unittest

import numpy as np

from deflecomp_core.observation.bingham import BinghamUtils
from probtf.geometry import quat_left_matrix, quat_right_matrix


class SharedGeometryTest(unittest.TestCase):
    def test_deflecomp_uses_shared_quaternion_matrices_without_rescaling(self):
        quaternion = np.array([2.0, -0.5, 0.25, 1.5], dtype=float)

        np.testing.assert_allclose(
            BinghamUtils._lmat(quaternion),
            quat_left_matrix(quaternion, normalize_input=False),
        )
        np.testing.assert_allclose(
            BinghamUtils._rmat(quaternion),
            quat_right_matrix(quaternion, normalize_input=False),
        )

    def test_deflecomp_tangent_matrix_matches_left_matrix_columns(self):
        quaternion = np.array([0.8, -0.1, 0.3, 0.5], dtype=float)
        expected = quat_left_matrix(quaternion, normalize_input=False)[:, 1:]

        np.testing.assert_allclose(BinghamUtils.qmat_from_quat_wxyz(quaternion), expected)

    def test_deflecomp_spatial_tangent_matches_right_matrix_columns(self):
        quaternion = np.array([0.8, -0.1, 0.3, 0.5], dtype=float)
        expected = quat_right_matrix(quaternion, normalize_input=False)[:, 1:]

        np.testing.assert_allclose(
            BinghamUtils.spatial_qmat_from_quat_wxyz(quaternion),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
