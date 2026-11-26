"""CPU checks of the actual full network and deterministic input plumbing."""
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bingham import BinghamNLL
from model import VGG16Bingham, parameter_counts
from train import (PoseDataset, TangentObserver, index_sha256, normalize_images, parse_args,
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
        self.assertIsNone(args.monitor_inputs)
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text(json.dumps({"batch_size": 16, "workers": 0, "device": "cpu"}))
            args = parse_args(["--config", str(config), "--batch-size", "8"])
            self.assertEqual((args.batch_size, args.workers, args.device), (8, 0, "cpu"))

    def test_tangent_observer_reuses_raw_matrices_and_preserves_training_state(self):
        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(1, 9)
                self.dropout = torch.nn.Dropout(.5)
                self.calls = []

            def forward(self, images):
                self.calls.append((torch.is_grad_enabled(), self.training))
                # Deliberately consume all three streams: the observer must
                # isolate even a future forward/helper that uses randomness.
                random.random()
                np.random.rand()
                torch.rand(1)
                return self.head(self.dropout(images.mean((1, 2, 3))[:, None]))

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.npz"
            pixels = np.zeros((5, 360, 640, 3), dtype=np.uint8)
            pixels[:] = np.arange(5, dtype=np.uint8)[:, None, None, None] * 40
            np.savez_compressed(path, rgb=pixels, angles_deg=[0, 60, 90, 120, 180],
                                q_camera_object_wxyz=np.tile([1., 0., 0., 0.], (5, 1)))
            observer = TangentObserver(path)
            frozen_hash = observer.input_sha256
            # Replacing the source packet after construction must not alter
            # the frozen RGB or the hash accompanying the measured matrices.
            path.write_bytes(b"later unrelated preview replacement")
            self.assertEqual(observer.input_sha256, frozen_hash)
            np.testing.assert_array_equal(observer.pixels, pixels)
            seed_everything(20261126, 4)
            model = SmallModel()
            model.train()
            model.dropout.eval()
            for parameter in model.parameters():
                parameter.grad = torch.full_like(parameter, .25)
            modes = [module.training for module in model.modules()]
            weights = [parameter.detach().clone() for parameter in model.parameters()]
            gradients = [parameter.grad.clone() for parameter in model.parameters()]
            torch_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
            seen = []

            def covariance(matrix, v):
                seen.append((matrix.copy(), tuple(v)))
                return {"status": "structurally_undefined", "covariance_defined": False,
                        "principal_sigma_deg": None, "covariance_2x2_rad2": None}

            helper = types.SimpleNamespace(direction_tangent_covariance=covariance)
            with patch.dict(sys.modules, {"tangent_audit_math": helper}):
                result = observer.evaluate(model, torch.device("cpu"), epoch=3, optimizer_step=1875)
            self.assertEqual(result["status"], "observed")
            self.assertEqual((result["epoch"], result["optimizer_step"]), (3, 1875))
            self.assertEqual(model.calls, [(False, False)])
            self.assertEqual([module.training for module in model.modules()], modes)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
            self.assertEqual(random.getstate(), python_rng)
            after = np.random.get_state()
            self.assertEqual(after[0], numpy_rng[0])
            np.testing.assert_array_equal(after[1], numpy_rng[1])
            self.assertEqual(after[2:], numpy_rng[2:])
            for parameter, weight, gradient in zip(model.parameters(), weights, gradients):
                self.assertTrue(torch.equal(parameter, weight))
                self.assertTrue(torch.equal(parameter.grad, gradient))
            from bingham import symmetric_matrix
            expected = symmetric_matrix(torch.tensor(result["parameters_9"], dtype=torch.float64)).numpy()
            np.testing.assert_array_equal(result["A_camera"], expected)
            self.assertEqual(len(seen), 10)
            for index in range(5):
                np.testing.assert_array_equal(seen[index][0], expected[index])
                np.testing.assert_array_equal(seen[index + 5][0], expected[index])
                self.assertEqual(seen[index][1], (1., 0., 0.))
                self.assertEqual(seen[index + 5][1], (0., 0., 1.))
                self.assertIsNone(result["X"][index]["principal_sigma_deg"])
            json.dumps(result, allow_nan=False)

    def test_tangent_observer_failure_is_passive_and_restores_rng(self):
        class FailingModel(torch.nn.Module):
            def forward(self, images):
                torch.rand(1)
                random.random()
                np.random.rand()
                raise RuntimeError("observer-only forward failure")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.npz"
            np.savez_compressed(path, rgb=np.zeros((5, 360, 640, 3), dtype=np.uint8),
                                angles_deg=[0, 60, 90, 120, 180])
            observer = TangentObserver(path)
            model = FailingModel().train()
            helper = types.SimpleNamespace(direction_tangent_covariance=lambda matrix, v: {})
            seed_everything(42, 4)
            torch_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
            with patch.dict(sys.modules, {"tangent_audit_math": helper}):
                result = observer.evaluate(model, torch.device("cpu"), epoch=1, optimizer_step=625)
            self.assertEqual(result["status"], "observer_error")
            self.assertIn("observer-only forward failure", result["error"])
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
            self.assertEqual(random.getstate(), python_rng)
            np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])


if __name__ == "__main__":
    unittest.main()
