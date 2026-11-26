"""Small CPU checks of continuation state, counts and dataset lineage."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extend_training as extension


class ExtensionTests(unittest.TestCase):
    def test_lr_is_explicit_and_global_counts_retain_the_smaller_parent_epochs(self):
        with self.assertRaises(SystemExit):
            extension.parse_args([])
        args = extension.parse_args(["--learning-rate", "1e-5"])
        self.assertEqual(args.learning_rate, 1e-5)
        self.assertEqual(args.monitor_inputs, args.output / "inputs.npz")
        self.assertEqual(extension.progress_at_epoch(50)["total_training_images_seen"], 500000)
        self.assertEqual(extension.progress_at_epoch(51)["total_training_images_seen"], 520000)
        final = extension.progress_at_epoch(100)
        self.assertEqual(final["additional_epochs_completed"], 50)
        self.assertEqual(final["total_training_images_seen"], 1500000)
        self.assertEqual(final["optimizer_step"], 93750)

    def test_cosine_restart_and_resume_preserve_all_adam_moments_and_steps(self):
        torch.manual_seed(19)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, foreach=False)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(2, 3)).square().sum().backward()
            optimizer.step()
        optimizer.param_groups[0]["initial_lr"] = 1e-4
        optimizer.param_groups[0]["lr"] = 0.
        parent = deepcopy(optimizer.state_dict())
        carried = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
        carried.load_state_dict(parent)
        scheduler = extension.restart_cosine(carried, 1e-5)
        self.assertEqual(scheduler.base_lrs, [1e-5])
        self.assertEqual(carried.param_groups[0]["lr"], 1e-5)
        self.assertEqual(carried.param_groups[0]["initial_lr"], 1e-5)
        self.assertEqual(scheduler.T_max, 50)
        for key, values in parent["state"].items():
            for name, tensor in values.items():
                torch.testing.assert_close(carried.state_dict()["state"][key][name], tensor, rtol=0, atol=0)
        for _ in range(17):
            carried.step()
            scheduler.step()
        saved_optimizer, saved_scheduler = deepcopy(carried.state_dict()), deepcopy(scheduler.state_dict())
        expected_lr = carried.param_groups[0]["lr"]
        resumed = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
        resumed.load_state_dict(saved_optimizer)
        resumed_scheduler = extension.resume_cosine(resumed, saved_scheduler)
        self.assertEqual(resumed.param_groups[0]["lr"], expected_lr)
        self.assertEqual(resumed_scheduler.state_dict(), saved_scheduler)
        for key, values in saved_optimizer["state"].items():
            for name, tensor in values.items():
                torch.testing.assert_close(resumed.state_dict()["state"][key][name], tensor, rtol=0, atol=0)
        for _ in range(33):
            resumed.step()
            resumed_scheduler.step()
        self.assertAlmostEqual(resumed.param_groups[0]["lr"], 0., places=18)

    def test_parent_output_and_input_paths_cannot_be_modified_in_place(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_run, old_data = root / "old-run", root / "old-data"
            extension.assert_distinct_paths(old_run, old_data, root / "new-run", root / "new-data")
            for output, data in ((old_run, root / "new-data"), (old_run / "nested", root / "new-data"),
                                 (root / "new-run", old_data), (old_data / "nested", root / "new-data")):
                with self.assertRaises(ValueError):
                    extension.assert_distinct_paths(old_run, old_data, output, data)

    def test_data_lineage_requires_exact_prefix_and_unchanged_validation_test_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = root / "old", root / "new"
            tiny_splits = {"train": (4, 1234), "val": (1, 1235), "test": (1, 1236)}
            metadata = {"image_size": [640, 360], "image_blur_sigma_pixels": 0., "mesh": {"center_native": [40, 40, 45]}}
            for dataset in (old, new):
                dataset.mkdir()
                (dataset / "render_metadata.json").write_text(json.dumps(metadata))
                for split, (new_count, seed) in tiny_splits.items():
                    count = 2 if split == "train" and dataset == old else new_count
                    folder = dataset / split
                    (folder / "images").mkdir(parents=True)
                    q = extension.render.uniform_quaternions(count, seed)
                    np.savez_compressed(folder / "poses.npz", q_wxyz=q, index=np.arange(count),
                                        t_camera=np.tile([0., 0., 2.5], (count, 1)), seed=np.array(seed))
                    digest = hashlib.sha256()
                    for index in range(count):
                        contents = f"{split} unchanged image {index}".encode()
                        (folder / "images" / f"{index:06d}.png").write_bytes(contents)
                        digest.update(contents)
                    (folder / "metadata.json").write_text(json.dumps({"count": count, "seed": seed,
                        "poses_sha256": extension.base.sha256(folder / "poses.npz"),
                        "images_concat_sha256": digest.hexdigest()}))
            proof = {"datasets": {"train": {"actual_images_concat_sha256": extension.png_hash(old / "train", 2)}}}
            with patch.multiple(extension, PARENT_TRAIN_COUNT=2, TRAIN_COUNT=4, SPLITS=tiny_splits):
                lineage = extension.validate_data_lineage(old, new, proof)
                self.assertEqual(lineage["train_prefix_count"], 2)
                (new / "train/images/000000.png").write_bytes(b"changed old image")
                with self.assertRaisesRegex(ValueError, "training PNG bytes"):
                    extension.validate_data_lineage(old, new, proof)
                (new / "train/images/000000.png").write_bytes(b"train unchanged image 0")
                (new / "test/images/000000.png").write_bytes(b"changed test image")
                with self.assertRaisesRegex(ValueError, "test: images differ"):
                    extension.validate_data_lineage(old, new, proof)


if __name__ == "__main__":
    unittest.main()
