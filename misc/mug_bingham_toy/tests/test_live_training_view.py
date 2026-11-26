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
        sigma_epochs, sigma = viewer.standard_deviation_history(rows, "X", 90)
        np.testing.assert_array_equal(sigma_epochs, epochs)
        self.assertEqual(sigma[0, 0], np.rad2deg(np.sqrt(1e-50)))
        self.assertTrue(np.isnan(sigma[1]).all())
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

    def test_saved_epoch_displays_shared_quaternion_samples_and_saved_standard_deviations(self):
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
            self.assertTrue(all(len(ax.images) == 0 for ax in window.sphere_axes))
            self.assertEqual(len(window.sample_packets), 5)
            for index, (packet, artists) in enumerate(zip(window.sample_packets, window.sample_artists)):
                np.testing.assert_array_equal(packet["A_camera"], record["tangent_xz"]["A_camera"][index])
                self.assertEqual(packet["seed"], window.sample_seed+index)
                self.assertEqual(packet["quaternions"].shape, (1000, 4))
                for axis in range(3):
                    near = packet["tips"][:, 2, axis] <= 0.
                    far = ~near
                    near_points = artists["near_points"][axis]
                    far_points = artists["far_points"][axis]
                    near_tips, far_tips = packet["tips"][near, :, axis], packet["tips"][far, :, axis]
                    near_tips = near_tips[np.argsort(near_tips[:, 2])[::-1]]
                    far_tips = far_tips[np.argsort(far_tips[:, 2])[::-1]]
                    np.testing.assert_array_equal(near_points.get_offsets(), near_tips[:, :2])
                    np.testing.assert_array_equal(far_points.get_offsets(), far_tips[:, :2])
                    self.assertEqual(len(near_points.get_offsets())+len(far_points.get_offsets()), 1000)
                    self.assertEqual(near_points.get_alpha(), .7)
                    self.assertEqual(far_points.get_alpha(), .35)
                self.assertTrue(all(line.get_alpha() == 1. and line.get_linewidth() == 1.8
                                    for line in artists["mode_axes"]))
            originals = [packet["quaternions"].copy() for packet in window.sample_packets]
            random_state = np.random.get_state()
            window._draw_spheres(record["tangent_xz"])
            for original, packet in zip(originals, window.sample_packets):
                np.testing.assert_array_equal(original, packet["quaternions"])
            after_state = np.random.get_state()
            self.assertEqual(random_state[0], after_state[0])
            np.testing.assert_array_equal(random_state[1], after_state[1])
            self.assertEqual(random_state[2:], after_state[2:])
            self.assertEqual(window.history_lines["X"][0].get_ydata()[0], np.rad2deg(np.sqrt(1e-50)))
            self.assertEqual(window.history_axes["X"].get_ylabel(), "接空間の主標準偏差 [deg]")
            self.assertEqual(window.history_axes["X"].get_yscale(), "log")
            self.assertNotIn("torch", viewer.__dict__)
            (root/"test_metrics.json").write_text("{}\n")
            window.poll()
            self.assertIn("完了", window.status.get_text())
            plt.close(window.figure)


if __name__ == "__main__":
    unittest.main()
