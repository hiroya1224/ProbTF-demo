"""CPU checks of the actual full network and deterministic input plumbing."""
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bingham import BinghamNLL
from model import VGG16Bingham, parameter_counts
from train import (PoseDataset, index_sha256, normalize_images, parse_args,
                   restore_rng, rng_state, seed_everything)


class ModelTrainingTests(unittest.TestCase):
    def test_full_vgg16_nine_output_and_two_end_to_end_updates(self):
        seed_everything(20261126, 4)
        model = VGG16Bingham()
        self.assertEqual(parameter_counts(model),
                         {"total": 134297417, "trainable": 134297417, "frozen": 0})
        self.assertEqual(sum(isinstance(layer, torch.nn.Conv2d) for layer in model.features), 13)
        self.assertEqual(model.classifier[0].in_features, 25088)
        self.assertEqual(model.classifier[0].out_features, 4096)
        self.assertEqual(model.classifier[3].out_features, 4096)
        self.assertEqual(model.classifier[6].out_features, 9)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertAlmostEqual(float(model.classifier[-1].weight.std()), 1e-4, delta=2e-6)
        self.assertTrue(torch.equal(model.classifier[-1].bias, torch.zeros(9)))
        # SGD avoids allocating the ~1.1 GB Adam moments in this CPU check.
        # Production training uses AdamW over these same full-network tensors.
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        criterion = BinghamNLL()
        pixels = torch.randint(0, 256, (1, 32, 32, 3), dtype=torch.uint8)
        target = torch.tensor([[.5, -.5, .5, -.5]], dtype=torch.float64)
        images = normalize_images(pixels, torch.device("cpu"))
        initial_conv = model.features[0].weight.detach().clone()
        initial_head = model.classifier[-1].weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            parameters = model(images)
            self.assertEqual(parameters.shape, (1, 9))
            self.assertEqual(parameters.dtype, torch.float32)
            loss = criterion(parameters, target)
            self.assertEqual(loss.dtype, torch.float64)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(model.features[0].weight.grad.abs().sum()), 0)
            optimizer.step()
        self.assertFalse(torch.equal(initial_conv, model.features[0].weight))
        self.assertFalse(torch.equal(initial_head, model.classifier[-1].weight))

    def test_dataset_preserves_whole_rgb_bytes_labels_and_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "images").mkdir()
            pixels = np.arange(32*40*3, dtype=np.uint8).reshape(32, 40, 3)
            for index in range(3):
                Image.fromarray(pixels + index, "RGB").save(folder / "images" / f"{index:06d}.png")
            labels = np.array([[1., 0, 0, 0], [-1., 0, 0, 0], [0., 1., 0, 0]])
            np.savez_compressed(folder / "poses.npz", q_wxyz=labels, index=np.arange(3),
                                t_camera=np.tile([0., 0., 2.5], (3, 1)), seed=np.array(1234))
            lazy = PoseDataset(folder, preload=False, image_size=(40, 32),
                               expected_count=3, expected_seed=1234)
            cached = PoseDataset(folder, preload=True, image_size=(40, 32))
            for index in range(3):
                image, target, saved_index = lazy[index]
                self.assertEqual(image.shape, (32, 40, 3))
                self.assertTrue(torch.equal(image, cached[index][0]))
                np.testing.assert_array_equal(target, labels[index])
                self.assertEqual(saved_index, index)
            self.assertEqual(lazy.images_sha256, cached.images_sha256)
            self.assertEqual(index_sha256(lazy.index), hashlib.sha256(np.arange(3, dtype="<i8").tobytes()).hexdigest())
            normalized = normalize_images(torch.tensor([[[[0, 127, 255]]]], dtype=torch.uint8), torch.device("cpu"))
            torch.testing.assert_close(normalized[0, :, 0, 0], torch.tensor([-1., 127/255*2-1, 1.]))
            np.testing.assert_array_equal(lazy[0][0], pixels)
            with self.assertRaises(ValueError):
                PoseDataset(folder, preload=False, image_size=(20, 16))

    def test_rng_and_parser_preserve_resume_settings(self):
        seed_everything(20261126, 4)
        shuffle = torch.Generator().manual_seed(12)
        workers = torch.Generator().manual_seed(13)
        saved = rng_state(shuffle, workers)
        draws = (random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=shuffle))
        restore_rng(saved, shuffle, workers)
        self.assertEqual(draws[0], random.random())
        self.assertEqual(draws[1], np.random.rand())
        self.assertTrue(torch.equal(draws[2], torch.rand(3)))
        self.assertTrue(torch.equal(draws[3], torch.rand(3, generator=shuffle)))
        args = parse_args([])
        self.assertEqual((args.epochs, args.batch_size, args.seed, args.workers), (50, 8, 20261126, 4))
        self.assertFalse(args.preload)
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text(json.dumps({"batch_size": 16, "workers": 0, "device": "cpu"}))
            args = parse_args(["--config", str(config), "--batch-size", "8"])
            self.assertEqual((args.batch_size, args.workers, args.device), (8, 0, "cpu"))


if __name__ == "__main__":
    unittest.main()
