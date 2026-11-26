"""Preservation, interrupted generation and validated resume on small fixtures."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import expand_dataset as expansion


def make_source(root):
    root.mkdir()
    renderer = {"image_size": [640, 360], "image_blur_sigma_pixels": 0.,
                "source_code_sha256": expansion.sha(Path(expansion.render.__file__)), "backend": "egl"}
    (root / "render_metadata.json").write_text(json.dumps(renderer)+"\n")
    for split, count, seed in (("train", 2, 1234), ("val", 1, 1235), ("test", 1, 1236)):
        folder = root / split
        (folder / "images").mkdir(parents=True)
        digest = hashlib.sha256()
        for index in range(count):
            path = folder / "images" / f"{index:06d}.png"
            Image.new("RGB", (640, 360), (seed % 255, index, 67)).save(path)
            digest.update(path.read_bytes())
        np.savez_compressed(folder / "poses.npz", q_wxyz=expansion.uniform_quaternions(count, seed),
                            index=np.arange(count, dtype=np.int64),
                            t_camera=np.tile([0., 0., 2.5], (count, 1)), seed=np.array(seed, dtype=np.int64))
        metadata = {"split": split, "count": count, "seed": seed,
                    "poses_sha256": expansion.sha(folder / "poses.npz"),
                    "images_concat_sha256": digest.hexdigest()}
        (folder / "metadata.json").write_text(json.dumps(metadata)+"\n")
    return renderer


def file_hashes(root):
    return {str(path.relative_to(root)): expansion.sha(path) for path in root.rglob("*") if path.is_file()}


class FakeRenderer:
    def __init__(self, metadata, fail_at=None):
        self.recorded_metadata, self.fail_at = metadata, fail_at
        self.calls, self.closed = [], False

    def metadata(self):
        return self.recorded_metadata

    def render(self, quaternion):
        self.calls.append(quaternion.copy())
        if len(self.calls) == self.fail_at:
            raise RuntimeError("interrupted test renderer")
        return np.full((360, 640, 3), [188, 21, 67], dtype=np.uint8), None

    def close(self):
        self.closed = True


class DatasetExpansionTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch.object(expansion, name, value) for name, value in
                        (("ORIGINAL_TRAIN_COUNT", 2), ("TARGET_TRAIN_COUNT", 4), ("HELDOUT_COUNT", 1))]
        for current in self.patches:
            current.start()
        self.addCleanup(lambda: [current.stop() for current in reversed(self.patches)])

    def test_original_bytes_and_heldout_splits_are_preserved_and_only_extra_poses_render(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "original", root / "expanded"
            renderer = FakeRenderer(make_source(source))
            before = file_hashes(source)
            with patch.object(expansion, "MugRenderer", return_value=renderer):
                metadata = expansion.expand_dataset(source, output)
            self.assertEqual(file_hashes(source), before)
            self.assertEqual(len(renderer.calls), 2)
            np.testing.assert_array_equal(renderer.calls, expansion.uniform_quaternions(4, 1234)[2:])
            with np.load(output / "train/poses.npz") as labels:
                np.testing.assert_array_equal(labels["q_wxyz"], expansion.uniform_quaternions(4, 1234))
                np.testing.assert_array_equal(labels["index"], np.arange(4))
            for split, count in (("train", 2), ("val", 1), ("test", 1)):
                for index in range(count):
                    path = Path(split) / "images" / f"{index:06d}.png"
                    self.assertEqual((source / path).read_bytes(), (output / path).read_bytes())
                    self.assertNotEqual((source / path).stat().st_ino, (output / path).stat().st_ino)
            for split in ("val", "test"):
                for name in ("metadata.json", "poses.npz"):
                    self.assertEqual((source / split / name).read_bytes(), (output / split / name).read_bytes())
            self.assertEqual((source / "render_metadata.json").read_bytes(), (output / "render_metadata.json").read_bytes())
            self.assertEqual(metadata["images_concat_sha256"], expansion.image_digest(output / "train", 4, (640, 360)))
            with patch.object(expansion, "MugRenderer") as factory:
                expansion.expand_dataset(source, output)
                factory.assert_not_called()

    def test_resume_skips_only_hash_validated_images_and_completion_waits_for_every_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "original", root / "expanded"
            metadata = make_source(source)
            first = FakeRenderer(metadata, fail_at=2)
            with patch.object(expansion, "MugRenderer", return_value=first):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    expansion.expand_dataset(source, output)
            self.assertTrue(first.closed)
            self.assertFalse((output / "train/metadata.json").exists())
            second = FakeRenderer(metadata, fail_at=1)
            with patch.object(expansion, "MugRenderer", return_value=second):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    expansion.expand_dataset(source, output)
            np.testing.assert_array_equal(second.calls, expansion.uniform_quaternions(4, 1234)[3:])
            (output / "train/images/000002.png").write_bytes(b"interrupted PNG")
            final = FakeRenderer(metadata)
            with patch.object(expansion, "MugRenderer", return_value=final):
                expansion.expand_dataset(source, output)
            np.testing.assert_array_equal(final.calls, expansion.uniform_quaternions(4, 1234)[2:])
            self.assertTrue((output / "train/metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
