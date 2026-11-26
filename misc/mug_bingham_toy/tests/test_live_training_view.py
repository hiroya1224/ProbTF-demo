"""Verify passive observer parsing and provenance without running a network."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import live_training_view as viewer


def epoch_record(epoch, image_hash="preview", variance=None):
    if variance is None:
        variance = [1e-50, 2e-3]
    component = {"principal_variance_rad2": variance,
                 "principal_sigma_deg": None if variance is None else [1., 2.]}
    return {"epoch": epoch, "train_nll_mean": 2., "validation": {"nll_mean": 3.},
            "tangent_xz": {"epoch": epoch, "inputs_sha256": image_hash,
                           "angles_deg": list(viewer.ANGLES),
                           "A_camera": np.tile(np.diag([3., 1., -1., -3.]), (5, 1, 1)).tolist(),
                           "X": [component.copy() for _ in range(5)],
                           "Z": [component.copy() for _ in range(5)]}}


class LiveTrainingViewTests(unittest.TestCase):
    def test_partial_append_is_not_drawn_as_a_complete_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"metrics.jsonl"
            first = json.dumps(epoch_record(1))
            second = json.dumps(epoch_record(2))
            path.write_text(first+"\n"+second[:80])
            rows, errors = viewer.read_metric_records(path)
            self.assertEqual([row["epoch"] for row in rows], [1])
            self.assertEqual(errors, [])
            with path.open("a") as stream:
                stream.write(second[80:]+"\n")
            rows, errors = viewer.read_metric_records(path)
            self.assertEqual([row["epoch"] for row in rows], [1, 2])
            self.assertEqual(errors, [])

    def test_covariances_have_no_floor_and_undefined_values_stay_missing(self):
        rows = [epoch_record(1), epoch_record(2)]
        rows[1]["tangent_xz"]["X"][2] = {"principal_variance_rad2": None, "status": "undefined"}
        epochs, values = viewer.variance_history(rows, "X", 90)
        np.testing.assert_array_equal(epochs, [1, 2])
        self.assertEqual(values[0, 0], 1e-50)
        self.assertTrue(np.isnan(values[1]).all())
        self.assertEqual(viewer.sigma_label(None), "— / —")
        failure = {"covariance_defined": False, "principal_variance_rad2": [-1., 2.]}
        self.assertTrue(np.isnan(viewer.principal_values(failure, "principal_variance_rad2")).all())

    def test_record_rejects_epoch_or_matrix_frame_packet_mismatch(self):
        row = epoch_record(1)
        row["tangent_xz"]["epoch"] = 2
        with self.assertRaisesRegex(ValueError, "same epoch"):
            viewer.measurement(row)
        row = epoch_record(1)
        row["tangent_xz"]["A_camera"][0][0][1] = 10
        with self.assertRaisesRegex(ValueError, "symmetric"):
            viewer.measurement(row)

    def test_preview_keeps_the_same_bytes_and_pixels_after_disk_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"inputs.npz"
            pixels = np.full((5, 3, 4, 3), 255, dtype=np.uint8)
            np.savez(path, rgb=pixels, angles_deg=viewer.ANGLES)
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            frozen = viewer.load_preview(path)
            self.assertEqual(frozen.sha256, expected)
            pixels[:] = 0
            np.savez(path, rgb=pixels, angles_deg=viewer.ANGLES)
            self.assertTrue((frozen.images == 255).all())
            self.assertNotEqual(viewer.load_preview(path).sha256, frozen.sha256)
            self.assertFalse(frozen.images.flags.writeable)

    def test_viewer_does_not_pair_a_prediction_with_different_rgb(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root/"inputs.npz"
            np.savez(inputs, rgb=np.full((5, 360, 640, 3), 255, np.uint8), angles_deg=viewer.ANGLES)
            (root/"metrics.jsonl").write_text(json.dumps(epoch_record(1, image_hash="different"))+"\n")
            window = viewer.TrainingViewer(root, inputs, subdivisions=2)
            self.assertIn("hash", window.status.get_text())
            self.assertIsNone(window._last_plot_key)
            self.assertTrue(all(len(ax.collections) == 0 for ax in window.sphere_axes))
            plt.close(window.figure)

    def test_saved_epoch_updates_real_density_mesh_and_variance_lines_without_inference(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root/"inputs.npz"
            np.savez(inputs, rgb=np.full((5, 360, 640, 3), 255, np.uint8), angles_deg=viewer.ANGLES)
            window = viewer.TrainingViewer(root, inputs, subdivisions=2)
            record = epoch_record(1, image_hash=window.preview.sha256)
            (root/"metrics.jsonl").write_text(json.dumps(record)+"\n")
            window.poll()
            window.figure.canvas.draw()
            self.assertIn("epoch 1", window.status.get_text())
            self.assertTrue(all(len(ax.collections) == 2 for ax in window.sphere_axes))
            from sphere_display import latitude_longitude_graticule, GRID_COLOR
            for ax in window.sphere_axes:
                grid = [line for line in ax.lines if line.get_color() == GRID_COLOR]
                self.assertEqual(len(grid), len(latitude_longitude_graticule()))
                self.assertTrue(all(line.get_alpha() == .30 for line in grid))
                from sphere_display import latitude_longitude_density_mesh
                mesh = latitude_longitude_density_mesh(np.diag([3., 1., -1., -3.]), 12, 24)
                order = np.argsort(mesh["face_depth"])[::-1]
                fade = np.where(mesh["face_depth"][order] > 0., .25, 1.)
                np.testing.assert_array_equal(ax.collections[0].get_facecolors()[:, 3], .14*fade)
                rgba = ax.collections[1].get_facecolors()
                np.testing.assert_array_equal(rgba[:, :3], mesh["face_rgb"][order])
                np.testing.assert_allclose(rgba[:, 3], .9*mesh["face_rgb"][order].max(axis=1)*fade, atol=0)
                rays = [line for line in ax.lines if line.get_color() != GRID_COLOR]
                self.assertEqual([line.get_alpha() for line in rays], [1., 1., .25])
            self.assertEqual(window.history_lines["X"][0].get_ydata()[0], 1e-50)
            self.assertEqual(window.history_axes["X"].get_yscale(), "log")
            self.assertNotIn("torch", viewer.__dict__)
            (root/"test_metrics.json").write_text("{}\n")
            window.poll()
            self.assertIn("完了", window.status.get_text())
            plt.close(window.figure)


if __name__ == "__main__":
    unittest.main()
