"""Geometry, label, source identity, and pixel consistency checks."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render


class GeometryTests(unittest.TestCase):
    def test_retained_source_and_normalized_colored_mesh(self):
        metadata = json.loads((render.ASSETS / "mesh_metadata.json").read_text())
        original = render.ASSETS / metadata["original_stl"]
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),
                         "b594ed7482a3e1148de7407d6b690c03a9a0b44a3d5d1d06b62e4e5c9bc243eb")
        native = trimesh.load(original, force="mesh")
        colored = trimesh.load(render.ASSETS / metadata["colored_mesh"], force="mesh")
        expected = (native.vertices - metadata["center_native"]) * metadata["scale_native_to_object"]
        # PLY persists vertex coordinates as float32; compare as unordered sets.
        np.testing.assert_allclose(np.sort(expected, axis=0),
                                   np.sort(colored.vertices, axis=0), atol=3e-8)
        np.testing.assert_allclose(colored.bounds.mean(axis=0), 0., atol=3e-8)
        self.assertAlmostEqual(colored.extents.max(), 1., places=7)
        self.assertTrue(np.all(colored.visual.vertex_colors == [188, 21, 67, 255]))

    def test_hamilton_quaternion_matches_independent_scipy_rotation(self):
        q = render.uniform_quaternions(100, 1234)
        reference = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(render.quaternion_matrix(q), reference, atol=1e-14)
        np.testing.assert_allclose(render.quaternion_matrix(-q), reference, atol=1e-14)
        recovered = Rotation.from_matrix(reference).as_quat()[:, [3, 0, 1, 2]]
        np.testing.assert_allclose(np.abs(np.sum(recovered*q, axis=1)), 1., atol=1e-14)

    def test_split_rng_reproduction_and_distinctness(self):
        q = render.uniform_quaternions(10000, 1234)
        np.testing.assert_array_equal(q[:10], render.uniform_quaternions(10, 1234))
        self.assertFalse(np.array_equal(q[:10], render.uniform_quaternions(10, 1235)))
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1., atol=3e-16)
        # Uniform S3 has second moment I4/4; this rejects biased Euler sampling.
        np.testing.assert_allclose(q.T@q/len(q), np.eye(4)/4., atol=.009)


class RenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = render.MugRenderer(width=128, height=128)
        cls.mesh = trimesh.load(render.ASSETS / "mug_centered_colored.ply", force="mesh")

    @classmethod
    def tearDownClass(cls):
        cls.renderer.close()

    def test_actual_camera_composition_and_image_projection(self):
        for q in render.uniform_quaternions(5, 1234):
            world_pose = self.renderer.world_object_pose(q)
            # This is the transform actually passed to the OpenGL object node.
            camera_rotation = render.R_CAMERA_WORLD @ world_pose[:3, :3]
            np.testing.assert_allclose(camera_rotation, render.quaternion_matrix(q), atol=1e-14)
            vertices_camera = self.mesh.vertices @ camera_rotation.T + [0., 0., 2.5]
            uv = vertices_camera[:, :2] / vertices_camera[:, 2, None]
            uv = uv * self.renderer.focal + self.renderer.K[:2, 2]
            rgb, depth = self.renderer.render(q)
            ys, xs = np.nonzero(depth > 0)
            self.assertGreater(len(xs), 2000)
            self.assertGreaterEqual(xs.min(), 4)
            self.assertLessEqual(xs.max(), 123)
            self.assertGreaterEqual(ys.min(), 4)
            self.assertLessEqual(ys.max(), 123)
            # Raster silhouette and independently projected vertices share the
            # optical x/y axes, focal length, and quaternion convention.
            # At most one pixel plus MSAA coverage rounding at triangle edges.
            np.testing.assert_allclose([xs.min(), ys.min()], uv.min(axis=0), atol=1.25)
            np.testing.assert_allclose([xs.max(), ys.max()], uv.max(axis=0), atol=1.25)
            self.assertTrue(np.all(rgb[0] == 255))
            self.assertTrue(np.all(rgb[:, 0] == 255))

    def test_antipodal_labels_render_identically(self):
        q = render.uniform_quaternions(1, 678)[0]
        a, da = self.renderer.render(q)
        b, db = self.renderer.render(-q)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(da, db)

    def test_saved_images_and_labels_are_the_same_pose(self):
        with tempfile.TemporaryDirectory(prefix="mug-pose-check-") as temporary:
            output = Path(temporary)
            render.render_split(self.renderer, output, "test", 3, 1236)
            saved = np.load(output / "test/poses.npz")
            np.testing.assert_array_equal(saved["q_wxyz"], render.uniform_quaternions(3, 1236))
            np.testing.assert_array_equal(saved["index"], np.arange(3, dtype=np.int64))
            for i, q in enumerate(saved["q_wxyz"]):
                expected, _ = self.renderer.render(q)
                actual = np.array(Image.open(output / f"test/images/{i:06d}.png"))
                np.testing.assert_array_equal(actual, expected)


class ResolutionTests(unittest.TestCase):
    def test_default_360p_intrinsics_and_pixel_dimensions(self):
        renderer = render.MugRenderer()
        try:
            self.assertEqual((renderer.width, renderer.height), (640, 360))
            np.testing.assert_array_equal(renderer.K,
                [[590.625, 0., 319.5], [0., 590.625, 179.5], [0., 0., 1.]])
            rgb, depth = renderer.render(render.uniform_quaternions(1, 1234)[0])
            self.assertEqual(rgb.shape, (360, 640, 3))
            self.assertEqual(depth.shape, (360, 640))
            ys, xs = np.nonzero(depth > 0)
            self.assertGreater(xs.min(), 150)
            self.assertLess(xs.max(), 490)
            self.assertGreater(ys.min(), 10)
            self.assertLess(ys.max(), 350)
        finally:
            renderer.close()


if __name__ == "__main__":
    unittest.main()
