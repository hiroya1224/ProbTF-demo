"""Check the display geometry against independent physical rotations/densities."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figure_geometry as geometry


class QuaternionFrameTests(unittest.TestCase):
    def test_hamilton_product_and_antipodal_rotation(self):
        p = np.array([0.4, -0.2, 0.5, 0.7])
        p /= np.linalg.norm(p)
        q = np.array([-0.3, 0.1, 0.8, -0.4])
        q /= np.linalg.norm(q)
        product = geometry.quaternion_left_matrix(p) @ q
        reference = (Rotation.from_quat(p[[1, 2, 3, 0]])
                     * Rotation.from_quat(q[[1, 2, 3, 0]])).as_matrix()
        np.testing.assert_allclose(geometry.quaternion_matrix(product), reference, atol=8e-16)
        np.testing.assert_allclose(geometry.quaternion_matrix(-product), reference, atol=8e-16)
        left = geometry.quaternion_left_matrix(p)
        np.testing.assert_allclose(left.T @ left, np.eye(4), atol=3e-16)
        np.testing.assert_allclose(geometry.quaternion_left_matrix(2*p), 2*left, atol=0)

    def test_frame_change_preserves_density_mode_and_axis_tips(self):
        raw = np.array([[1.0, 0.2, 0.3, -0.1], [0.2, -4.0, 0.5, 0.2],
                        [0.3, 0.5, -7.0, -0.4], [-0.1, 0.2, -0.4, -10.0]])
        matrix = raw - np.trace(raw)*np.eye(4)/4
        qwc = np.array([0.7, -0.2, 0.6, -0.1])
        qwc /= np.linalg.norm(qwc)
        rotation_wc = Rotation.from_quat(qwc[[1, 2, 3, 0]]).as_matrix()
        world = geometry.frame_change_bingham(matrix, rotation_wc)
        left = geometry.quaternion_left_matrix(qwc)
        np.testing.assert_allclose(world, left @ matrix @ left.T, atol=4e-15)
        self.assertAlmostEqual(np.trace(world), 0.0, places=13)
        np.testing.assert_allclose(np.linalg.eigvalsh(world), np.linalg.eigvalsh(matrix), atol=5e-15)
        quaternions, _ = geometry.half_height_quaternions(matrix)
        transformed = quaternions @ left.T
        density_camera = np.einsum("ni,ij,nj->n", quaternions, matrix, quaternions)
        density_world = np.einsum("ni,ij,nj->n", transformed, world, transformed)
        np.testing.assert_allclose(density_world, density_camera, atol=8e-15)
        np.testing.assert_allclose(geometry.quaternion_matrix(transformed),
                                   rotation_wc @ geometry.quaternion_matrix(quaternions), atol=2e-15)
        mode_world = geometry.mode_quaternion(world)
        expected_mode = left @ geometry.mode_quaternion(matrix)
        self.assertAlmostEqual(abs(mode_world @ expected_mode), 1.0, places=14)

    def test_camera_up_world_poses_and_panel_handle_directions(self):
        poses = geometry.camera_up_world_poses()
        np.testing.assert_array_equal(poses["angles_deg"], [0, 60, 90, 120, 180])
        world = poses["R_world_camera"]
        np.testing.assert_allclose(world.T @ world, np.eye(3), atol=2e-16)
        self.assertAlmostEqual(np.linalg.det(world), 1.0, places=14)
        np.testing.assert_allclose(poses["R_world_object"][:, :, 2],
                                   np.tile([0., 0., 1.], (5, 1)), atol=0)
        np.testing.assert_allclose(geometry.quaternion_matrix(poses["q_camera_object_wxyz"]),
                                   world.T @ poses["R_world_object"], atol=5e-16)
        left = geometry.quaternion_left_matrix(poses["q_world_camera_wxyz"])
        composed = poses["q_camera_object_wxyz"] @ left.T
        np.testing.assert_allclose(geometry.quaternion_matrix(composed), poses["R_world_object"], atol=9e-16)
        handle_camera = poses["R_camera_object"][:, :, 0]
        self.assertLess(handle_camera[0, 0], -0.99)
        self.assertGreater(handle_camera[2, 2], 0.98)
        self.assertGreater(handle_camera[-1, 0], 0.99)


class HalfHeightTests(unittest.TestCase):
    def assert_on_half_height(self, matrix, quaternions, atol=2e-14):
        self.assertEqual(quaternions.ndim, 2)
        self.assertEqual(quaternions.shape[1], 4)
        self.assertTrue(np.isfinite(quaternions).all())
        np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-15)
        peak = np.linalg.eigvalsh(matrix)[-1]
        density = np.einsum("ni,ij,nj->n", quaternions, matrix, quaternions)
        np.testing.assert_allclose(density-peak, -np.log(2), atol=atol)
        antipodal = np.einsum("ni,ij,nj->n", -quaternions, matrix, -quaternions)
        np.testing.assert_array_equal(density, antipodal)

    def test_concentrated_points_are_exact_reference_construction(self):
        matrix = np.array([[3., .2, -.3, .7], [.2, -1., .1, .2],
                           [-.3, .1, -4., -.3], [.7, .2, -.3, -8.]])
        original = matrix.copy()
        quaternions, diagnostics = geometry.half_height_quaternions(matrix)
        eigenvalues, basis = np.linalg.eigh(matrix)
        gaps = eigenvalues[-1] - eigenvalues[:3]
        index = np.arange(960) + 0.5
        z = 1 - 2*index/len(index)
        phase = np.pi*(3-np.sqrt(5))*index
        radius = np.sqrt(1-z*z)
        directions = np.column_stack((radius*np.cos(phase), radius*np.sin(phase), z))
        transverse = directions*np.sqrt(np.log(2)/gaps)
        axial = np.sqrt(1-np.sum(transverse**2, axis=1))
        expected = transverse @ basis[:, :3].T + axial[:, None]*basis[:, -1]
        np.testing.assert_array_equal(quaternions, expected)
        self.assertEqual(len(quaternions), 960)
        self.assertEqual(diagnostics["method"], "principal_ellipsoid_fibonacci")
        self.assertEqual(diagnostics["non_crossing_direction_count"], 0)
        self.assert_on_half_height(matrix, quaternions)
        np.testing.assert_array_equal(matrix, original)
        json.dumps(diagnostics, allow_nan=False)

    def test_wide_repeated_modes_preserve_true_level_and_omissions(self):
        matrix = np.diag([-5., -4., 0., 0.])
        quaternions, diagnostics = geometry.half_height_quaternions(matrix)
        self.assertGreater(len(quaternions), 0)
        self.assertLess(len(quaternions), 960)
        self.assertEqual(diagnostics["method"], "great_circle_fibonacci_crossings")
        self.assertGreater(diagnostics["non_crossing_direction_count"], 0)
        self.assertEqual(diagnostics["crossing_direction_count"]
                         + diagnostics["non_crossing_direction_count"], 960)
        self.assert_on_half_height(matrix, quaternions)
        json.dumps(diagnostics, allow_nan=False)

    def test_uniform_and_everywhere_above_half_height_are_empty(self):
        for matrix in (np.zeros((4, 4)), np.eye(4)*3, np.diag([-.2, -.1, 0., .1])):
            quaternions, diagnostics = geometry.half_height_quaternions(matrix)
            self.assertEqual(quaternions.shape, (0, 4))
            self.assertFalse(diagnostics["level_set_exists"])
            self.assertEqual(diagnostics["non_crossing_direction_count"], 960)
            self.assertIsNone(diagnostics["max_log_density_drop_residual"])
            json.dumps(diagnostics, allow_nan=False)

    def test_narrow_crossing_domain_and_critical_eigenspaces(self):
        log2 = np.log(2)
        # No Fibonacci direction hits this very small domain; the geometric
        # principal-direction fallback still returns true contour points.
        matrix = np.diag([-log2*(1+1e-12), 0., 0., 0.])
        quaternions, diagnostics = geometry.half_height_quaternions(matrix)
        self.assertEqual(diagnostics["crossing_direction_count"], 0)
        self.assertEqual(diagnostics["principal_crossing_anchor_count"], 2)
        self.assert_on_half_height(matrix, quaternions)
        for matrix, expected in ((np.diag([-log2, 0., 0., 0.]), 1),
                                 (np.diag([-log2, -log2, 0., 0.]), 960),
                                 (np.diag([-log2, -log2, -log2, 0.]), 960)):
            quaternions, diagnostics = geometry.half_height_quaternions(matrix)
            self.assertEqual(len(quaternions), expected)
            self.assertEqual(diagnostics["method"], "critical_minimum_eigenspace")
            self.assert_on_half_height(matrix, quaternions)
            json.dumps(diagnostics, allow_nan=False)

    def test_one_ulp_above_critical_level_avoids_subtractive_roundoff_nan(self):
        gap = np.nextafter(np.log(2), np.inf)
        matrix = np.diag([-gap, -gap, -gap, 0.])
        quaternions, diagnostics = geometry.half_height_quaternions(matrix)
        self.assertEqual(len(quaternions), 960)
        self.assertGreater(diagnostics["axial_roundoff_correction_count"], 0)
        self.assert_on_half_height(matrix, quaternions)
        json.dumps(diagnostics, allow_nan=False)


class HalfWidthTests(unittest.TestCase):
    def test_width_matches_independent_physical_angle_root_and_reference(self):
        basis, _ = np.linalg.qr(np.array([[1., 2., 3., 4.], [4., 1., -2., 3.],
                                        [2., -3., 1., 4.], [3., 4., -1., 2.]]))
        matrix = basis @ np.diag([-14., -7., -3., 2.]) @ basis.T
        mode = geometry.mode_quaternion(matrix)
        axis = np.array([.3, -.2, .8])
        axis /= np.linalg.norm(axis)
        measured = geometry.yaw_half_width(matrix, mode, axis)
        mode_rotation = Rotation.from_quat(mode[[1, 2, 3, 0]])
        peak = mode @ matrix @ mode

        def independent(theta):
            physical = Rotation.from_rotvec(theta*axis) * mode_rotation
            q = physical.as_quat()[[3, 0, 1, 2]]
            return q @ matrix @ q - peak + np.log(2)

        expected = np.degrees(brentq(independent, 0., np.pi, xtol=5e-15))
        self.assertAlmostEqual(measured["width_deg"], expected, places=11)
        reference = np.degrees(2*np.arcsin(np.sqrt(np.log(2)/measured["gap"])))
        self.assertAlmostEqual(measured["width_deg"], reference, places=11)
        self.assertLess(abs(measured["log_density_drop_residual"]), 2e-14)
        self.assertAlmostEqual(geometry.yaw_half_width(matrix, -mode, axis)["width_deg"],
                               measured["width_deg"], places=12)

    def test_width_and_axis_are_invariant_under_frame_change(self):
        matrix = np.diag([2., -3., -5., -11.])
        qwc = np.array([.2, .7, -.3, .4])
        qwc /= np.linalg.norm(qwc)
        rotation_wc = geometry.quaternion_matrix(qwc)
        world = geometry.frame_change_bingham(matrix, rotation_wc)
        axis = np.array([0., 0., 1.])
        first = geometry.yaw_half_width(matrix, axis=axis)
        second = geometry.yaw_half_width(world, axis=rotation_wc @ axis)
        self.assertAlmostEqual(first["width_deg"], second["width_deg"], places=12)

    def test_wide_uniform_and_exact_half_turn_are_finite(self):
        for matrix in (np.zeros((4, 4)), np.diag([0., -.2, -.3, -.4])):
            result = geometry.yaw_half_width(matrix)
            self.assertIsNone(result["width_deg"])
            self.assertFalse(result["crosses_half_height_within_180_deg"])
            json.dumps(result, allow_nan=False)
        matrix = np.diag([0., -1., -1., -np.log(2)])
        result = geometry.yaw_half_width(matrix)
        self.assertEqual(result["width_deg"], 180.0)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
