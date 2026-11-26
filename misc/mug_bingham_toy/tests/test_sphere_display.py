"""Check display-only latitude/longitude topology and unchanged density values."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_geometry import axis_direction_log_density
import sphere_display as display


class LatitudeLongitudeDisplayTests(unittest.TestCase):
    def test_closed_sphere_has_z_poles_xy_equator_and_no_duplicate_seam(self):
        rows, columns = 12, 24
        mesh = display.latitude_longitude_mesh(rows, columns)
        vertices, faces = mesh["vertices"], mesh["faces"]
        self.assertEqual(len(vertices), 2+(rows-1)*columns)
        self.assertEqual(len(faces), 2*(rows-1)*columns)
        np.testing.assert_array_equal(vertices[[0, -1]], [[0., 0., 1.], [0., 0., -1.]])
        np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1., atol=3e-16)
        self.assertEqual(len(np.unique(vertices, axis=0)), len(vertices))
        equator = vertices[1+(rows//2-1)*columns:1+rows//2*columns]
        np.testing.assert_allclose(equator[:, 2], 0., atol=1e-15)
        edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, np.full(len(unique), 2))
        self.assertEqual(len(vertices)-len(unique)+len(faces), 2)
        triangles = vertices[faces]
        outward = np.einsum("ni,ni->n", np.cross(triangles[:, 1]-triangles[:, 0],
                                                 triangles[:, 2]-triangles[:, 0]), triangles.sum(axis=1))
        self.assertTrue((outward > 0).all())
        self.assertAlmostEqual(mesh["areas"].sum(), 4*np.pi, places=13)
        self.assertTrue(mesh["front_faces"].any())
        self.assertTrue((~mesh["front_faces"]).any())

    def test_visible_wires_are_latitudes_and_longitudes_about_camera_z(self):
        lines = display.latitude_longitude_graticule(return_xyz=True)
        self.assertEqual(len(lines), 5+12)
        for line, degrees in zip(lines[:5], (-60, -30, 0, 30, 60)):
            np.testing.assert_allclose(line[:, 2], np.sin(np.radians(degrees)), atol=0)
            np.testing.assert_allclose(np.linalg.norm(line[:, :2], axis=1),
                                       np.cos(np.radians(degrees)), atol=4e-16)
        for line, degrees in zip(lines[5:], range(0, 360, 30)):
            np.testing.assert_array_equal(line[[0, -1]], [[0., 0., 1.], [0., 0., -1.]])
            longitude = np.radians(degrees)
            np.testing.assert_allclose(line[:, 0]*np.sin(longitude)-line[:, 1]*np.cos(longitude), 0., atol=2e-16)
        projected = display.latitude_longitude_graticule()
        self.assertEqual(len(projected), 3+12)
        for line, degrees in zip(projected[:3], (-60, -30, 0)):
            np.testing.assert_allclose(np.linalg.norm(line, axis=1), np.cos(np.radians(degrees)), atol=4e-16)
        for line in projected[3:]:
            np.testing.assert_array_equal(line[0], [0., 0.])
            self.assertAlmostEqual(np.linalg.norm(line[-1]), 1., places=15)
            self.assertTrue((np.diff(np.linalg.norm(line, axis=1)) >= 0).all())

    def test_analytic_density_is_unchanged_and_color_only_omits_common_normalizer(self):
        A = np.array([[2., .3, -.2, .1], [.3, -1., .7, -.4],
                      [-.2, .7, -3., .5], [.1, -.4, .5, 2.]])
        original = A.copy()
        mesh = display.latitude_longitude_density_mesh(A, 8, 16, log_normalizer=4.)
        np.testing.assert_array_equal(A, original)
        np.testing.assert_array_equal(mesh["vertex_log_density"], axis_direction_log_density(A, mesh["vertices"], 4.))
        np.testing.assert_array_equal(mesh["face_log_density"], axis_direction_log_density(A, mesh["centers"], 4.))
        relative = display.latitude_longitude_density_mesh(A, 8, 16)
        np.testing.assert_allclose(relative["face_rgb"], mesh["face_rgb"], atol=7e-16)
        self.assertFalse(mesh["diagnostics"]["matrix_modified"])
        self.assertFalse(mesh["diagnostics"]["monte_carlo"])
        self.assertFalse(mesh["vertices"].flags.writeable)


if __name__ == "__main__":
    unittest.main()
