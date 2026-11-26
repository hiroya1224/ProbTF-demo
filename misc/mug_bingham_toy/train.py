#!/usr/bin/env python3
"""Train the isolated, full VGG16 mug Bingham toy experiment from scratch.

Example:
  python train.py --data data --output runs/vgg16_360p --device cuda

The complete 640x360 RGB image is used: no resize, crop, augmentation, or
ImageNet normalization. Pixels become (uint8 / 255 - 0.5) / 0.5. All 10,000
training images are used exactly once per epoch. Validation chooses best.pt;
the independent test split is evaluated only after every configured epoch.
Large local checkpoints and generated images should remain Git-ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

try:
    from .bingham import BinghamNLL, modal_quaternion, symmetric_matrix
    from .model import VGG16Bingham, parameter_counts
except ImportError:
    from bingham import BinghamNLL, modal_quaternion, symmetric_matrix
    from model import VGG16Bingham, parameter_counts

HERE = Path(__file__).resolve().parent
IMAGE_SIZE = (640, 360)
SPLITS = {"train": (10000, 1234), "val": (1000, 1235), "test": (1000, 1236)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def write_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class PoseDataset(Dataset):
    """Saved RGB bytes and exact wxyz labels, without image augmentation.

    Preloading stores only uint8 arrays, about 6.9 GB for the training split.
    With Linux fork workers those read-only pages are shared. Pixel conversion
    and normalization happen after transfer to the training device.
    """

    def __init__(self, folder: Path, *, preload: bool = True,
                 image_size: tuple[int, int] = IMAGE_SIZE,
                 expected_count: int | None = None, expected_seed: int | None = None):
        self.folder = Path(folder)
        self.image_size = tuple(image_size)
        with np.load(self.folder / "poses.npz", allow_pickle=False) as labels:
            self.q_wxyz = np.asarray(labels["q_wxyz"], dtype=np.float64).copy()
            self.index = np.asarray(labels["index"], dtype=np.int64).copy()
            self.seed = int(labels["seed"])
            translation = np.asarray(labels["t_camera"], dtype=np.float64)
        count = len(self.index)
        if expected_count is not None and count != expected_count:
            raise ValueError(f"{folder}: expected {expected_count} images, found {count}")
        if expected_seed is not None and self.seed != expected_seed:
            raise ValueError(f"{folder}: expected pose seed {expected_seed}, found {self.seed}")
        if self.q_wxyz.shape != (count, 4) or translation.shape != (count, 3):
            raise ValueError(f"{folder}: invalid quaternion/translation array shapes")
        if not np.array_equal(self.index, np.arange(count, dtype=np.int64)):
            raise ValueError(f"{folder}: indices must be the original contiguous render indices")
        if not np.isfinite(self.q_wxyz).all() or not np.allclose(
                np.linalg.norm(self.q_wxyz, axis=1), 1, rtol=0, atol=1e-12):
            raise ValueError(f"{folder}: saved wxyz quaternions must already be finite and unit")
        if not np.array_equal(translation, np.tile([0., 0., 2.5], (count, 1))):
            raise ValueError(f"{folder}: fixed camera/object translations changed")
        self.paths = [self.folder / "images" / f"{i:06d}.png" for i in self.index]
        width, height = self.image_size
        self.images = np.empty((count, height, width, 3), dtype=np.uint8) if preload else None
        digest = hashlib.sha256()
        for row, path in enumerate(self.paths):
            raw = path.read_bytes()
            digest.update(raw)
            with Image.open(io.BytesIO(raw)) as image:
                if image.mode != "RGB" or image.size != self.image_size:
                    raise ValueError(f"{path}: expected RGB {self.image_size}, got {image.mode} {image.size}")
                if preload:
                    self.images[row] = np.asarray(image, dtype=np.uint8)
        self.images_sha256 = digest.hexdigest()
        metadata_path = self.folder / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("images_concat_sha256") != self.images_sha256:
                raise ValueError(f"{folder}: actual image bytes do not match render metadata")
            if metadata.get("poses_sha256") != sha256(self.folder / "poses.npz"):
                raise ValueError(f"{folder}: actual labels do not match render metadata")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, position):
        if self.images is None:
            with Image.open(self.paths[position]) as image:
                pixels = np.array(image, dtype=np.uint8, copy=True)
        else:
            # No in-place operation is applied to this shared uint8 storage.
            pixels = self.images[position]
        return (torch.from_numpy(pixels), torch.from_numpy(self.q_wxyz[position]),
                int(self.index[position]))

    def provenance(self):
        return {"count": len(self), "seed": self.seed,
                "actual_images_concat_sha256": self.images_sha256,
                "poses_sha256": sha256(self.folder / "poses.npz"),
                "sorted_indices_sha256": index_sha256(self.index),
                "preloaded_uint8_bytes": 0 if self.images is None else self.images.nbytes}


def normalize_images(pixels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert whole uint8 NHWC RGB to normalized float32 NCHW on device."""
    if pixels.dtype != torch.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError("Expected a uint8 RGB batch with shape [B,H,W,3]")
    images = pixels.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32,
                                         memory_format=torch.channels_last,
                                         non_blocking=device.type == "cuda")
    return images.div_(255).sub_(0.5).div_(0.5)


def seed_everything(seed: int, cpu_threads: int):
    torch.set_num_threads(cpu_threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # AdaptiveAvgPool2d's CUDA backward can use atomic additions. We preserve
    # the standard architecture and record this finite reproducibility limit.


def rng_state(shuffle_generator, worker_generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "shuffle": shuffle_generator.get_state(), "workers": worker_generator.get_state()}


def restore_rng(state, shuffle_generator, worker_generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    shuffle_generator.set_state(state["shuffle"])
    worker_generator.set_state(state["workers"])


def make_loader(dataset, batch_size, workers, device, *, shuffle_generator=None,
                worker_generator=None):
    options = dict(batch_size=batch_size, num_workers=workers, drop_last=False,
                   pin_memory=device.type == "cuda", generator=worker_generator)
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    if shuffle_generator is not None:
        options["sampler"] = RandomSampler(dataset, generator=shuffle_generator)
    return DataLoader(dataset, **options)


@torch.no_grad()
def pose_diagnostics(parameters, target):
    mode = modal_quaternion(parameters)
    target = target.to(device=parameters.device, dtype=torch.float64)
    dot = (mode * target).sum(-1).abs().clamp(0, 1)
    error = torch.rad2deg(2 * torch.acos(dot))
    spectrum = torch.linalg.eigvalsh(symmetric_matrix(parameters.to(dtype=torch.float64)))
    return mode, error, spectrum[:, -1] - spectrum[:, -2]


def evaluate(model, loader, criterion, device, *, retain_predictions=False):
    """All validation/test items are evaluated; no quality gate filters them."""
    model.eval()
    nlls, errors, gaps = [], [], []
    predictions = {key: [] for key in ("parameters_9", "q_wxyz", "index", "mode_q_wxyz")}
    with torch.no_grad():
        for pixels, target, indices in loader:
            parameters = model(normalize_images(pixels, device))
            nll = criterion(parameters, target, reduce=False)
            mode, error, gap = pose_diagnostics(parameters, target)
            nlls.append(nll.cpu().numpy())
            errors.append(error.cpu().numpy())
            gaps.append(gap.cpu().numpy())
            if retain_predictions:
                predictions["parameters_9"].append(parameters.cpu().numpy())
                predictions["q_wxyz"].append(target.numpy())
                predictions["index"].append(indices.numpy())
                predictions["mode_q_wxyz"].append(mode.cpu().numpy())
    nlls, errors, gaps = (np.concatenate(values) for values in (nlls, errors, gaps))
    if not np.isfinite(nlls).all() or not np.isfinite(errors).all() or not np.isfinite(gaps).all():
        raise FloatingPointError("Nonfinite evaluation result; no examples were discarded")
    metrics = {"count": int(len(nlls)), "nll_mean": float(nlls.mean()),
               "mode_error_deg_mean": float(errors.mean()),
               "mode_error_deg_median": float(np.median(errors)),
               "mode_error_deg_p90": float(np.quantile(errors, .9)),
               "top_eigen_gap_mean": float(gaps.mean())}
    if retain_predictions:
        arrays = {key: np.concatenate(values) for key, values in predictions.items()}
        arrays.update(nll=nlls, mode_error_deg=errors, top_eigen_gap=gaps)
        return metrics, arrays
    return metrics


def save_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, *, map_location="cpu"):
    # Checkpoints are local artifacts created by this script, including Python
    # and numpy RNG states. Do not load an untrusted third-party checkpoint.
    return torch.load(path, map_location=map_location, weights_only=False)


def input_manifest(folder):
    paths = [folder / "render_metadata.json"]
    paths += [folder / split / name for split in SPLITS
              for name in ("poses.npz", "metadata.json")]
    return {str(path.relative_to(folder)): sha256(path) for path in paths}


def environment(device):
    result = {"python": platform.python_version(), "numpy": np.__version__,
              "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "device": str(device),
              "cpu_threads": torch.get_num_threads(),
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "cuda_bitwise_reproducibility": "Not guaranteed: standard adaptive-pooling backward can use CUDA atomic additions"}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu=properties.name, gpu_memory_bytes=properties.total_memory,
                      gpu_compute_capability=[properties.major, properties.minor],
                      amp_bfloat16_supported=torch.cuda.is_bf16_supported())
    return result


def current_sources():
    return {name: sha256(HERE / name) for name in ("train.py", "model.py", "bingham.py", "render.py")}


def canonical_config(args):
    return {"data": str(args.data.resolve()), "output": str(args.output.resolve()),
            "epochs": args.epochs, "batch_size": args.batch_size, "device": args.device,
            "seed": args.seed, "workers": args.workers, "cpu_threads": args.cpu_threads,
            "preload": args.preload, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "quadrature_nodes": args.quadrature_nodes,
            "image_size": list(IMAGE_SIZE), "normalization": "(uint8/255 - 0.5)/0.5",
            "augmentation": None, "architecture": "standard VGG16 configuration D, full 4096/4096 classifier, 9 outputs",
            "pretrained": False, "last_linear_init_std": 1e-4,
            "optimizer": "AdamW, foreach=False", "scheduler": "CosineAnnealingLR, eta_min=0",
            "split_sizes_seeds": {key: list(value) for key, value in SPLITS.items()},
            "checkpoint_selection": "lowest complete-validation mean Bingham NLL",
            "training": "all training indices once per epoch; no early stopping, gradient clipping or sample filtering"}


def available_ram_bytes():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def run_training(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this process")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((args.output / name).exists() for name in ("last.pt", "metrics.jsonl", "test_metrics.json")):
        raise FileExistsError("Output already contains a run; choose a new folder or use --resume last.pt")
    config = canonical_config(args)
    manifest = input_manifest(args.data)
    sources = current_sources()
    render_metadata = json.loads((args.data / "render_metadata.json").read_text())
    if render_metadata.get("image_size") != list(IMAGE_SIZE):
        raise ValueError("Dataset must contain whole 640x360 renders; resize is not performed")
    ram = available_ram_bytes()
    cache_bytes = sum(count for count, _ in SPLITS.values()) * IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3
    if args.preload and ram is not None and ram < cache_bytes + 4 * 1024**3:
        raise MemoryError("Insufficient available RAM for uint8 cache plus training; use --no-preload")
    datasets = {}
    for split in ("train", "val"):
        count, seed = SPLITS[split]
        print(f"Verifying/loading {split} ({count} whole RGB images)", flush=True)
        datasets[split] = PoseDataset(args.data / split, preload=args.preload,
                                      expected_count=count, expected_seed=seed)
    model = VGG16Bingham(amp_bfloat16=device.type == "cuda" and torch.cuda.is_bf16_supported())
    model.to(device=device, memory_format=torch.channels_last)
    counts = parameter_counts(model)
    if counts["total"] != 134297417 or counts["frozen"]:
        raise AssertionError("Full VGG16 parameter count/trainability changed")
    criterion = BinghamNLL(device=device, nodes=args.quadrature_nodes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    shuffle_generator = torch.Generator().manual_seed(args.seed + 1)
    worker_generator = torch.Generator().manual_seed(args.seed + 2)
    start_epoch, best_val, best_epoch, total_seen = 0, math.inf, 0, 0
    provenance = {"config": config, "source_sha256": sources, "input_sha256": manifest,
                  "environment": environment(device), "model_parameter_counts": counts,
                  "model_implementation": "Self-contained faithful standard VGG16 definition; no torchvision dependency",
                  "amp_feature_dtype": "bfloat16" if model.amp_bfloat16 else "float32",
                  "final_head_dtype": "float32", "nll_dtype": "float64",
                  "available_ram_bytes_at_start": ram,
                  "datasets": {key: dataset.provenance() for key, dataset in datasets.items()},
                  "test_access": "Labels/metadata hashed before training; test images/predictions evaluated only after all epochs"}
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if checkpoint["config"] != config:
            raise ValueError("Resume configuration differs from saved configuration")
        if checkpoint["source_sha256"] != sources or checkpoint["input_sha256"] != manifest:
            raise ValueError("Resume source code or input hashes differ from saved run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, best_val = checkpoint["epoch"], checkpoint["best_val_nll"]
        best_epoch, total_seen = checkpoint["best_epoch"], checkpoint["total_training_images_seen"]
        restore_rng(checkpoint["rng"], shuffle_generator, worker_generator)
        del checkpoint
        print(f"Resuming after epoch {start_epoch}; best epoch {best_epoch}", flush=True)
    else:
        try:
            provenance["git_head_at_start"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
        except (OSError, subprocess.SubprocessError):
            provenance["git_head_at_start"] = None
        write_json(args.output / "config.json", config)
        write_json(args.output / "provenance.json", provenance)
    train_loader = make_loader(datasets["train"], args.batch_size, args.workers, device,
                               shuffle_generator=shuffle_generator, worker_generator=worker_generator)
    val_loader = make_loader(datasets["val"], args.batch_size, args.workers, device,
                             worker_generator=worker_generator)
    for epoch_index in range(start_epoch, args.epochs):
        started = time.perf_counter()
        model.train()
        seen_indices, train_nll_sum = [], 0.0
        epoch_lr = optimizer.param_groups[0]["lr"]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for batch_number, (pixels, target, indices) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            parameters = model(normalize_images(pixels, device))
            loss = criterion(parameters, target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training NLL; batch was not skipped")
            loss.backward()
            optimizer.step()
            train_nll_sum += float(loss.detach()) * len(indices)
            seen_indices.append(indices.numpy())
            if batch_number % 100 == 0 or batch_number == len(train_loader):
                print(json.dumps({"progress": "training", "epoch": epoch_index + 1,
                                  "batch": batch_number, "batches": len(train_loader),
                                  "images_seen": sum(len(batch) for batch in seen_indices),
                                  "latest_batch_nll": float(loss.detach()),
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
        seen_indices = np.concatenate(seen_indices)
        if not np.array_equal(np.sort(seen_indices), datasets["train"].index):
            raise AssertionError("An epoch omitted or duplicated a training index")
        total_seen += len(seen_indices)
        validation = evaluate(model, val_loader, criterion, device)
        epoch = epoch_index + 1
        improved = validation["nll_mean"] < best_val
        if improved:
            best_val, best_epoch = validation["nll_mean"], epoch
        scheduler.step()
        coverage = {"seen": int(len(seen_indices)), "unique": int(len(np.unique(seen_indices))),
                    "sorted_indices_sha256": index_sha256(np.sort(seen_indices)),
                    "shuffled_order_sha256": index_sha256(seen_indices)}
        metrics = {"epoch": epoch, "learning_rate": epoch_lr,
                   "train_nll_mean": train_nll_sum / len(seen_indices), "validation": validation,
                   "coverage": coverage, "total_training_images_seen": total_seen,
                   "best_epoch": best_epoch, "best_val_nll": best_val,
                   "seconds_train_and_validation": time.perf_counter() - started,
                   "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
        common = {"model": model.state_dict(), "epoch": epoch, "config": config,
                  "source_sha256": sources, "input_sha256": manifest,
                  "best_val_nll": best_val, "best_epoch": best_epoch,
                  "total_training_images_seen": total_seen, "coverage": coverage,
                  "model_parameter_counts": counts}
        if improved:
            save_checkpoint(args.output / "best.pt", common)
        save_checkpoint(args.output / "last.pt", dict(common, optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), rng=rng_state(shuffle_generator, worker_generator)))
        metrics["seconds_including_checkpoint"] = time.perf_counter() - started
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        print(json.dumps(metrics, allow_nan=False), flush=True)
    if (args.output / "test_metrics.json").exists():
        print("This completed run already has its final test evaluation; not evaluating test again.", flush=True)
        return
    if total_seen != args.epochs * SPLITS["train"][0]:
        raise AssertionError("Final test cannot run before every configured training epoch")
    best = load_checkpoint(args.output / "best.pt")
    model.load_state_dict(best["model"])
    del best
    count, seed = SPLITS["test"]
    test_dataset = PoseDataset(args.data / "test", preload=args.preload,
                               expected_count=count, expected_seed=seed)
    if input_manifest(args.data) != manifest:
        raise ValueError("Input files changed during training")
    test_loader = make_loader(test_dataset, args.batch_size, args.workers, device,
                              worker_generator=worker_generator)
    test_metrics, arrays = evaluate(model, test_loader, criterion, device, retain_predictions=True)
    if not np.array_equal(arrays["index"], test_dataset.index):
        raise AssertionError("Saved test predictions do not match original image indices")
    np.savez_compressed(args.output / "test_predictions.npz", **arrays)
    test_metrics.update(checkpoint="best.pt", best_epoch=best_epoch, selection="validation NLL only",
                        epochs_completed=args.epochs, total_training_images_seen=total_seen,
                        dataset=test_dataset.provenance(),
                        predictions_sha256=sha256(args.output / "test_predictions.npz"))
    write_json(args.output / "test_metrics.json", test_metrics)
    print(json.dumps({"final_test": test_metrics}, allow_nan=False), flush=True)


def run_benchmark(args):
    """Full-model synthetic timing only: never creates training checkpoints."""
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this process")
    args.output.mkdir(parents=True, exist_ok=True)
    model = VGG16Bingham(amp_bfloat16=device.type == "cuda" and torch.cuda.is_bf16_supported())
    model.to(device=device, memory_format=torch.channels_last)
    criterion = BinghamNLL(device=device, nodes=args.quadrature_nodes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, foreach=False)
    generator = torch.Generator().manual_seed(args.seed)
    pixels = torch.randint(0, 256, (args.batch_size, IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
                           dtype=torch.uint8, generator=generator)
    target = torch.randn((args.batch_size, 4), generator=generator, dtype=torch.float64)
    target = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations, losses = [], []
    for step in range(args.benchmark_steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(normalize_images(pixels, device)), target)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
        losses.append(float(loss.detach()))
        print(f"Synthetic benchmark step {step+1}/{args.benchmark_steps}: {durations[-1]:.3f}s, NLL {losses[-1]:.6g}", flush=True)
    steady = durations[1:] if len(durations) > 1 else durations
    report = {"kind": "synthetic throughput/memory benchmark; not a training result",
              "steps": args.benchmark_steps, "batch_size": args.batch_size, "image_size": list(IMAGE_SIZE),
              "step_seconds": durations, "step_nll": losses,
              "steady_seconds_per_step": float(np.mean(steady)),
              "steady_training_images_per_second": args.batch_size / float(np.mean(steady)),
              "estimated_training_seconds_per_10000_images": 10000 / args.batch_size * float(np.mean(steady)),
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
              "model_parameter_counts": parameter_counts(model), "environment": environment(device),
              "amp_feature_dtype": "bfloat16" if model.amp_bfloat16 else "float32",
              "source_sha256": current_sources(), "seed": args.seed}
    write_json(args.output / "benchmark.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="optional JSON mapping of CLI option names to values")
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=HERE / "runs" / "vgg16_360p")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=False,
                        help="cache decoded uint8 images in RAM (off by default; requires about 8.3 GB)")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--quadrature-nodes", type=int, default=200)
    parser.add_argument("--resume", type=Path, help="local last.pt; all saved configuration and input/source hashes must match")
    parser.add_argument("--benchmark-steps", type=int, default=0,
                        help="run this many synthetic full-resolution optimizer steps, save timing only")
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    preliminary, _ = pre_parser.parse_known_args(argv)
    if preliminary.config:
        defaults = json.loads(preliminary.config.read_text())
        allowed = {action.dest for action in parser._actions} - {"help", "config"}
        if not isinstance(defaults, dict) or set(defaults) - allowed:
            parser.error("Config must map recognized CLI option names (using underscores) to values")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.data, args.output = Path(args.data), Path(args.output)
    if args.resume:
        args.resume = Path(args.resume)
    for name in ("epochs", "batch_size", "cpu_threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.benchmark_steps < 0:
        parser.error("--workers and --benchmark-steps cannot be negative")
    if args.quadrature_nodes < 32 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Require quadrature nodes >=32, learning rate >0, and weight decay >=0")
    if args.resume and args.benchmark_steps:
        parser.error("--resume cannot be combined with --benchmark-steps")
    return args


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed, args.cpu_threads)
    if args.benchmark_steps:
        run_benchmark(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
