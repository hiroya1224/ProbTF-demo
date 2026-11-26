"""Native sample identity and passive camera/depth display checks."""
import json
import colorsys
from pathlib import Path
import random
import sys
import unittest

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sample_display as display


class NativeSampleTests(unittest.TestCase):
    def test_wrapper_matches_existing_native_core_output_exactly(self):
        matrix = np.array([[3., .2, -.3, .1], [.2, 1., -.2, .4],
                           [-.3, -.2, -1., .3], [.1, .4, .3, -3.]])
        packet = display.sample_camera_axes(matrix, count=1000, seed=20261127)
        from probtf import _sampling
        direct = np.empty((1000, 4), dtype=np.float64)
        b, proposals = _sampling.sample_bingham(np.ascontiguousarray(matrix), direct, 20261127)
        np.testing.assert_array_equal(packet["quaternions"], direct)
        self.assertEqual(packet["optimal_b"], b)
        self.assertEqual(packet["proposal_count"], proposals)

    def test_shared_unit_quaternions_reproducibility_and_accepted_prefix(self):
        matrix = np.array([[3., .2, -.3, .1], [.2, 1., -.2, .4],
                           [-.3, -.2, -1., .3], [.1, .4, .3, -3.]])
        original = matrix.copy()
        first = display.sample_camera_axes(matrix, seed=73)
        repeated = display.sample_camera_axes(matrix, seed=73)
        longer = display.sample_camera_axes(matrix, count=6000, seed=73)
        self.assertEqual(first["quaternions"].shape, (4096, 4))
        np.testing.assert_array_equal(first["quaternions"], repeated["quaternions"])
        np.testing.assert_array_equal(first["quaternions"], longer["quaternions"][:4096])
        np.testing.assert_allclose(np.linalg.norm(first["quaternions"], axis=1), 1., atol=4e-16)
        independent = Rotation.from_quat(first["quaternions"][:, [1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(first["tips"], independent, atol=8e-16)
        np.testing.assert_allclose(np.swapaxes(first["tips"], 1, 2) @ first["tips"],
                                   np.tile(np.eye(3), (4096, 1, 1)), atol=2e-15)
        self.assertGreaterEqual(first["proposal_count"], 4096)
        self.assertGreater(first["optimal_b"], 0.)
        self.assertLessEqual(first["optimal_b"], 4.)
        self.assertEqual(first["metadata"]["colored_axis_tip_count"], 12288)
        np.testing.assert_array_equal(matrix, original)
        json.dumps(first["metadata"], allow_nan=False)

    def test_scalar_matrix_shift_and_antipodal_sign_preserve_the_display_law(self):
        matrix = np.diag([3., 1., -1., -3.])
        first = display.sample_camera_axes(matrix, count=1000, seed=91)
        shifted = display.sample_camera_axes(matrix+7*np.eye(4), count=1000, seed=91)
        np.testing.assert_array_equal(first["quaternions"], shifted["quaternions"])
        np.testing.assert_array_equal(first["tips"], shifted["tips"])
        np.testing.assert_array_equal(display.camera_axes_from_quaternions(-first["quaternions"]), first["tips"])
        self.assertEqual(first["proposal_count"], shifted["proposal_count"])

    def test_visual_sampler_does_not_consume_global_training_rng_streams(self):
        numpy_before, python_before = np.random.get_state(), random.getstate()
        display.sample_camera_axes(np.zeros((4, 4)), count=1000, seed=15)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertEqual(python_before, random.getstate())


class SamplePainterTests(unittest.TestCase):
    def test_uniform_without_unique_mode_draws_no_arbitrary_axes(self):
        packet = display.sample_camera_axes(np.zeros((4, 4)), count=1000, seed=12)
        figure = Figure(figsize=(3., 3.))
        FigureCanvasAgg(figure)
        axes = figure.add_subplot()
        artists = display.draw_sample_sphere(axes, packet, rotation=None)
        self.assertEqual(artists["mode_axes"], [])
        self.assertEqual(artists["display_metadata"]["mode_axis_count"], 0)
        self.assertEqual(len(axes.lines), 0)

    def test_opaque_rear_points_keep_hue_raise_lightness_and_preserve_samples(self):
        packet = display.sample_camera_axes(np.zeros((4, 4)), seed=9)
        tips_before = packet["tips"].copy()
        q_before = packet["quaternions"].copy()
        figure = Figure(figsize=(3., 3.))
        canvas = FigureCanvasAgg(figure)
        axes = figure.add_subplot()
        # Mode X points directly to the rear; its ray must remain opaque.
        rotation = Rotation.from_euler("y", -75, degrees=True).as_matrix()
        artists = display.draw_sample_sphere(axes, packet, rotation=rotation)
        canvas.draw()
        from matplotlib.colors import to_rgb
        for axis in range(3):
            near, far = artists["near_points"][axis], artists["far_points"][axis]
            self.assertEqual(near.get_alpha(), 1.)
            self.assertEqual(far.get_alpha(), 1.)
            front_rgb = np.asarray(to_rgb(display.AXIS_COLORS[axis]))
            front_hue, lightness, _ = colorsys.rgb_to_hls(*front_rgb)
            target = min(1., 1.5*lightness)
            fraction = (target-lightness)/(1.-lightness)
            rear_rgb = (1.-fraction)*front_rgb+fraction
            np.testing.assert_array_equal(near.get_facecolors()[0, :3], front_rgb)
            np.testing.assert_array_equal(far.get_facecolors()[0, :3], rear_rgb)
            rear_hue, rear_lightness, _ = colorsys.rgb_to_hls(*far.get_facecolors()[0, :3])
            self.assertAlmostEqual(rear_hue, front_hue, places=14)
            self.assertAlmostEqual(rear_lightness, target, places=14)
            record = artists["display_metadata"]["point_color_mapping"]["axes"]["xyz"[axis]]
            self.assertEqual(record["white_fraction"], fraction)
            np.testing.assert_array_equal(record["rear_rgb"], rear_rgb)
            self.assertEqual(len(near.get_offsets())+len(far.get_offsets()), 4096)
            self.assertEqual(len(far.get_offsets()), np.count_nonzero(packet["tips"][:, 2, axis] > 0.))
        self.assertAlmostEqual(artists["far_wire"].get_alpha()/artists["near_wire"].get_alpha(), 1/5)
        self.assertLess(artists["far_support"].get_zorder(), artists["far_points"][0].get_zorder())
        self.assertLess(artists["near_support"].get_zorder(), artists["far_points"][0].get_zorder())
        self.assertEqual(artists["display_metadata"]["point_color_mapping"]["rear_lightness_ratio"], 1.5)
        for ray in artists["mode_axes"]:
            self.assertEqual(ray.get_alpha(), 1.)
            self.assertEqual(ray.get_linewidth(), 1.8)
            self.assertGreater(ray.get_zorder(), artists["near_points"][0].get_zorder())
        self.assertGreater(axes.get_ylim()[0], axes.get_ylim()[1])
        self.assertEqual(artists["display_metadata"]["rear_point_count_per_axis"],
                         [int(np.count_nonzero(packet["tips"][:, 2, k] > 0.)) for k in range(3)])
        np.testing.assert_array_equal(packet["tips"], tips_before)
        np.testing.assert_array_equal(packet["quaternions"], q_before)
        json.dumps(artists["display_metadata"], allow_nan=False)


if __name__ == "__main__":
    unittest.main()
