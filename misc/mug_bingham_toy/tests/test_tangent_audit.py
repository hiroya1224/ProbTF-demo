"""Meaningful checks for the passive chapter-4 tangent observer."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figure_geometry as geometry
from tangent_audit_math import audit_direction, direction_tangent_covariance


class TangentObserverTests(unittest.TestCase):
    def test_diagonal_X_and_Z_match_chapter4_closed_form(self):
        matrix = np.diag([3., 1., -1., -3.])
        x = direction_tangent_covariance(matrix, [1., 0., 0.])
        z = direction_tangent_covariance(matrix, [0., 0., 1.])
        np.testing.assert_array_equal(x["basis_input"], [[0., 0.], [1., 0.], [0., 1.]])
        np.testing.assert_array_equal(z["basis_input"], [[1., 0.], [0., 1.], [0., 0.]])
        np.testing.assert_allclose(x["leading_precision_2x2"], np.diag([3., 2.]), atol=0)
        np.testing.assert_allclose(z["leading_precision_2x2"], np.diag([2., 1.]), atol=0)
        np.testing.assert_allclose(x["covariance_2x2_rad2"], np.diag([.3, 3/7]), atol=0)
        np.testing.assert_allclose(z["covariance_2x2_rad2"], np.diag([3/7, .75]), atol=0)
        for item in (x, z):
            self.assertEqual(item["status"], "defined_le_proxy")
            np.testing.assert_allclose(item["principal_sigma_deg"],
                np.degrees(np.sqrt(item["principal_variance_rad2"])), atol=0)
            json.dumps(item, allow_nan=False)

    def test_precision_matches_independent_fiber_leading_exponent_hessian(self):
        matrix = np.array([[3., .2, -.3, .1], [.2, 1., -.2, .4],
                           [-.3, -.2, -1., .3], [.1, .4, .3, -3.]])
        result = direction_tangent_covariance(matrix, [0., 0., 1.])
        mode, basis = np.asarray(result["surrogate_mode"]), np.asarray(result["basis_output"])

        def energy(coordinates):
            length = np.linalg.norm(coordinates)
            point = mode*np.cos(length)+(basis @ coordinates)*np.sinc(length/np.pi)
            projection = .5*(np.eye(4)-geometry.quaternion_left_matrix(np.r_[0., point])
                              @ geometry.quaternion_right_matrix([0., 0., 0., 1.]))
            _, vectors = np.linalg.eigh(projection)
            fiber = vectors[:, -2:]
            return np.linalg.eigvalsh(fiber.T @ matrix @ fiber)[-1]

        step = 2e-4
        estimated = np.empty((2, 2))
        for i in range(2):
            delta = np.eye(2)[i]*step
            estimated[i, i] = -(energy(delta)-2*energy(np.zeros(2))+energy(-delta))/step**2
        dx, dy = np.eye(2)*step
        estimated[0, 1] = estimated[1, 0] = -(energy(dx+dy)-energy(dx-dy)
                                                -energy(-dx+dy)+energy(-dx-dy))/(4*step**2)
        np.testing.assert_allclose(estimated, result["leading_precision_2x2"], rtol=0., atol=2e-7)

    def test_camera_frame_trace_gauge_and_scaling_law(self):
        matrix = np.array([[3., .2, -.3, .1], [.2, 1., -.2, .4],
                           [-.3, -.2, -1., .3], [.1, .4, .3, -3.]])
        q = np.array([.4, -.2, .6, .3]); q /= np.linalg.norm(q)
        rotation = geometry.quaternion_matrix(q)
        world = geometry.frame_change_bingham(matrix, rotation)
        for v in ([1., 0., 0.], [0., 0., 1.], [1., 2., -3.]):
            first = direction_tangent_covariance(matrix, v)
            second = direction_tangent_covariance(world, v)
            shifted = direction_tangent_covariance(matrix+11*np.eye(4), v)
            scaled = direction_tangent_covariance(7*matrix, v)
            covariance = np.asarray(first["covariance_3x3_rad2"])
            np.testing.assert_allclose(second["covariance_3x3_rad2"], rotation @ covariance @ rotation.T, atol=3e-15)
            np.testing.assert_allclose(shifted["covariance_3x3_rad2"], covariance, atol=3e-15)
            np.testing.assert_allclose(scaled["finite_precision_2x2"],
                7*(np.asarray(first["finite_precision_2x2"])-np.eye(2)/3)+np.eye(2)/3, atol=1e-14)

    def test_uniform_ridge_and_spherical_LE_covariance_are_undefined(self):
        for matrix in (np.zeros((4, 4)), np.diag([3., -3., -3., 3.]),
                       np.diag([1., 1., 1., -3.])):
            result = direction_tangent_covariance(matrix)
            self.assertFalse(result["covariance_defined"])
            self.assertEqual(result["status"], "structurally_undefined")
            self.assertIsNone(result["covariance_2x2_rad2"])
            self.assertIsNone(result["principal_sigma_deg"])
            json.dumps(result, allow_nan=False)

    def test_strictly_positive_weak_gap_is_not_cut_off(self):
        matrix = np.diag([1., -1., -1., 1.-1e-12])
        result = audit_direction(matrix)
        self.assertEqual(result["structural_case"], "unique_mode_nonpolar")
        self.assertTrue(result["covariance_defined"])
        self.assertGreater(max(result["high_concentration"]["covariance_eigenvalues_2d"]), 1e12)
        self.assertFalse(result["pseudoinverse_cutoff"])

    def test_aligned_polar_exception_is_labelled_as_LE_only(self):
        result = direction_tangent_covariance(np.diag([3., 3., -3., -3.]))
        self.assertEqual(result["structural_case"], "polar_aligned_LE_only_nonGaussian_exact_limit")
        self.assertTrue(result["covariance_defined"])
        np.testing.assert_allclose(result["leading_precision_2x2"], np.eye(2)*3., atol=0)

    def test_matches_existing_core_tangent_covariance_without_calling_sampler(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ros/core/probtf_core/src"))
        from probtf.spherical_law.tangent import induced_vector_moments_tangent
        matrix = np.array([[3., .2, -.3, .1], [.2, 1., -.2, .4],
                           [-.3, -.2, -1., .3], [.1, .4, .3, -3.]])
        original = matrix.copy()
        for v in ([1., 0., 0.], [0., 0., 1.]):
            result = direction_tangent_covariance(matrix, v)
            existing = induced_vector_moments_tangent(v, matrix)
            expected = np.degrees(np.sqrt(np.linalg.eigvalsh(existing.sigma_mat)))
            np.testing.assert_allclose(result["principal_sigma_deg"], expected, rtol=0., atol=3e-13)
        np.testing.assert_array_equal(matrix, original)


if __name__ == "__main__":
    unittest.main()
