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

    def test_wire_depth_is_preserved_until_hemisphere_assignment(self):
        segments = display.latitude_longitude_graticule_segments()
        self.assertEqual(len(segments["far"])+len(segments["near"]), 17*180)
        self.assertTrue((segments["far"][:, :, 2].mean(axis=1) > 0.).all())
        self.assertTrue((segments["near"][:, :, 2].mean(axis=1) <= 0.).all())
        # The XY equator is retained as a visible near-side outline.
        equator = segments["near"][np.all(segments["near"][:, :, 2] == 0., axis=1)]
        self.assertEqual(len(equator), 180)
        np.testing.assert_allclose(np.linalg.norm(equator[:, :, :2], axis=2), 1., atol=4e-16)

    def test_painter_keeps_raw_colors_and_places_far_wire_and_ray_behind_near_faces(self):
        from matplotlib.figure import Figure

        figure = Figure()
        ax = figure.subplots()
        A = np.diag([3., 1., -1., -3.])
        mesh = display.latitude_longitude_density_mesh(A, 8, 16)
        original_rgb = mesh["face_rgb"].copy()
        artists = display.draw_latitude_longitude_sphere(ax, mesh, np.eye(3), ("red", "green", "blue"))
        np.testing.assert_array_equal(mesh["face_rgb"], original_rgb)
        raster = display.camera_sphere_density_raster(A)
        self.assertEqual(artists["raster_metadata"]["pixel_size"], 320)
        np.testing.assert_array_equal(artists["raster_metadata"]["color_log_offsets"], raster["color_log_offsets"])
        self.assertEqual(artists["raster_metadata"]["extent"], [-1., 1., 1., -1.])
        for side, far, fade in (("far", True, .25), ("near", False, 1.)):
            np.testing.assert_array_equal(artists[side+"_density"].get_array(), raster[side+"_rgba"])
            self.assertIsNotNone(artists[side+"_density"].get_clip_path())
            self.assertEqual(artists[side+"_wire"].get_alpha(), display.GRID_ALPHA*fade)
            np.testing.assert_array_equal(artists[side+"_wire"].get_linewidths(), [display.GRID_LINEWIDTH])
            support = artists[side+"_support"]
            self.assertEqual(support.radius, 1.)
            np.testing.assert_array_equal(support.center, [0., 0.])
            np.testing.assert_array_equal(support.get_facecolor(), [.89, .89, .89, .14*fade])
        self.assertEqual(len(ax.patches), 2)
        self.assertEqual(len(ax.collections), 2)
        self.assertEqual(len(ax.images), 2)
        self.assertLess(artists["far_wire"].get_zorder(), artists["near_support"].get_zorder())
        self.assertLess(artists["far_wire"].get_zorder(), artists["near_density"].get_zorder())
        self.assertGreater(artists["near_wire"].get_zorder(), artists["near_density"].get_zorder())
        self.assertEqual(len(artists["far_axes"]), 1)
        self.assertEqual(artists["far_axes"][0].get_alpha(), 1.)
        self.assertGreater(artists["far_axes"][0].get_zorder(), artists["near_density"].get_zorder())
        self.assertEqual(artists["far_axes"][0].get_linewidth(), 1.8)
        self.assertTrue(all(line.get_alpha() == 1. for line in artists["near_axes"]))

    def test_pixel_raster_is_the_same_analytic_law_with_one_shared_color_scale(self):
        A = np.array([[2., .3, -.2, .1], [.3, -1., .7, -.4],
                      [-.2, .7, -3., .5], [.1, -.4, .5, 2.]])
        original = A.copy()
        packet = display.camera_sphere_density_raster(A, pixel_size=32, log_normalizer=4.)
        np.testing.assert_array_equal(A, original)
        logs = []
        for side, fade in (("far", .25), ("near", 1.)):
            directions = packet[side+"_directions"]
            np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1., atol=2e-16)
            expected = axis_direction_log_density(A, directions, 4.)
            logs.append(expected)
            np.testing.assert_array_equal(packet[side+"_log_density"], expected)
            rgba = packet[side+"_rgba"]
            rgb = np.exp(expected-packet["color_log_offsets"])
            np.testing.assert_array_equal(rgba[packet["inside"], :3], rgb)
            np.testing.assert_array_equal(rgba[packet["inside"], 3], .9*rgb.max(axis=1)*fade)
            self.assertTrue((rgba[~packet["inside"], 3] == 0.).all())
        np.testing.assert_array_equal(packet["color_log_offsets"], np.concatenate(logs).max(axis=0))
        mesh = display.latitude_longitude_density_mesh(A, 8, 16)
        self.assertFalse(np.shares_memory(mesh["A_camera"], A))
        np.testing.assert_array_equal(mesh["A_camera"], A)

    def test_flat_field_has_no_mesh_resolution_or_limb_overpainting(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from unittest.mock import patch

        inside, _ = display._orthographic_pixel_directions(320)
        constant = {"pixel_size": 320, "color_log_offsets": np.zeros(3)}
        for side, fade in (("far", .25), ("near", 1.)):
            rgba = np.zeros((320, 320, 4))
            rgba[inside, :3] = .2
            rgba[inside, 3] = .9*.2*fade
            constant[side+"_rgba"] = rgba
        images = []
        for resolution in (12, 48, 96):
            figure = Figure(figsize=(2.56, 2.56), dpi=100, facecolor="white")
            FigureCanvasAgg(figure)
            ax = figure.add_axes([0, 0, 1, 1])
            ax.set(xlim=(-1.1, 1.1), ylim=(1.1, -1.1), aspect="equal")
            ax.axis("off")
            mesh = display.latitude_longitude_density_mesh(np.diag([3., 1., -1., -3.]), resolution, 2*resolution)
            with patch.object(display, "camera_sphere_density_raster", return_value=constant):
                artists = display.draw_latitude_longitude_sphere(ax, mesh, None, ("red", "green", "blue"))
            artists["far_wire"].set_visible(False)
            artists["near_wire"].set_visible(False)
            figure.canvas.draw()
            images.append(np.asarray(figure.canvas.buffer_rgba()).copy())
        np.testing.assert_array_equal(images[0], images[1])
        np.testing.assert_array_equal(images[0], images[2])
        y, x = np.mgrid[:256, :256]
        radius = np.sqrt(((x+.5-128)*2.2/256)**2+((y+.5-128)*2.2/256)**2)
        center = images[0][radius < .7, :3].mean(axis=1)/255.
        limb = images[0][(radius > .95) & (radius < .98), :3].mean(axis=1)/255.
        self.assertLess(abs(float(np.median(center))-float(np.median(limb))), 1/255.)
        self.assertGreater(float(limb.min()), .80)


if __name__ == "__main__":
    unittest.main()
