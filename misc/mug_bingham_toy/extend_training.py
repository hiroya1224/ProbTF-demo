#!/usr/bin/env python3
"""Continue the completed 10k/50-epoch mug run on 20k images to epoch 100.

The parent run remains read-only. Model tensors, AdamW moments and all saved
RNG streams come from its last.pt. Only the explicitly supplied learning rate
and a new 50-epoch cosine schedule are restarted. Network, loss, saved-image
normalization, data sampler, evaluation and passive observer are reused from
the frozen train.py implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

try:
    from . import render, train as base
except ImportError:
    import render
    import train as base

HERE = Path(__file__).resolve().parent
PARENT_EPOCHS = 50
FINAL_EPOCH = 100
PARENT_TRAIN_COUNT = 10000
TRAIN_COUNT = 20000
PARENT_SEEN = PARENT_EPOCHS * PARENT_TRAIN_COUNT
BATCH_SIZE = 16
SPLITS = {"train": (TRAIN_COUNT, 1234), "val": (1000, 1235), "test": (1000, 1236)}
DEFAULT_PARENT = HERE / "runs/vgg16_10000_cylinder_center_no_blur_20261127"
DEFAULT_DATA = HERE / "data/cylinder_center_no_blur_20000_20261127"
DEFAULT_OUTPUT = HERE / "runs/vgg16_20000_cylinder_center_no_blur_20261127"


def progress_at_epoch(epoch):
    """Global exposure/optimizer counts retain the first 50 smaller epochs."""
    if not PARENT_EPOCHS <= epoch <= FINAL_EPOCH:
        raise ValueError("Extension epoch must be between 50 and 100")
    additional = epoch - PARENT_EPOCHS
    return {"epoch": epoch, "additional_epochs_completed": additional,
            "total_training_images_seen": PARENT_SEEN + additional * TRAIN_COUNT,
            "optimizer_step": PARENT_EPOCHS * math.ceil(PARENT_TRAIN_COUNT / BATCH_SIZE)
                              + additional * math.ceil(TRAIN_COUNT / BATCH_SIZE)}


def restart_cosine(optimizer, learning_rate):
    """Reset only LR metadata/scheduler; AdamW moments and step tensors survive."""
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("An explicit positive finite extension learning rate is required")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        group["initial_lr"] = learning_rate
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINAL_EPOCH - PARENT_EPOCHS, eta_min=0.)


def resume_cosine(optimizer, state):
    """Restore extension scheduler state while retaining loaded current LRs."""
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINAL_EPOCH - PARENT_EPOCHS, eta_min=0.)
    scheduler.load_state_dict(state)
    return scheduler


def png_hash(folder, count):
    digest = hashlib.sha256()
    for index in range(count):
        digest.update((Path(folder) / "images" / f"{index:06d}.png").read_bytes())
    return digest.hexdigest()


def labels_at(folder):
    with np.load(Path(folder) / "poses.npz", allow_pickle=False) as labels:
        return {key: labels[key].copy() for key in ("q_wxyz", "index", "t_camera", "seed")}


def validate_data_lineage(parent_data, data, parent_provenance):
    """Verify unchanged rendering, old training prefix and validation/test bytes."""
    parent_data, data = Path(parent_data), Path(data)
    old_render = json.loads((parent_data / "render_metadata.json").read_text())
    new_render = json.loads((data / "render_metadata.json").read_text())
    if any(new_render.get(key) != value for key, value in old_render.items()):
        raise ValueError("20k dataset rendering differs from the completed no-blur parent")
    if new_render.get("image_size") != list(base.IMAGE_SIZE) or new_render.get("image_blur_sigma_pixels") != 0:
        raise ValueError("Extension requires the same whole 640x360 no-blur images")
    old_labels, new_labels = labels_at(parent_data / "train"), labels_at(data / "train")
    train_metadata = json.loads((data / "train/metadata.json").read_text())
    if (train_metadata.get("count"), train_metadata.get("seed")) != (TRAIN_COUNT, SPLITS["train"][1]):
        raise ValueError("20k train metadata count/seed differs from the approved dataset")
    expected_q = render.uniform_quaternions(TRAIN_COUNT, SPLITS["train"][1])
    if not np.array_equal(new_labels["q_wxyz"], expected_q):
        raise ValueError("20k train quaternions differ from the fixed seed-1234 full-S3 law")
    for key in ("q_wxyz", "index", "t_camera"):
        if not np.array_equal(new_labels[key][:PARENT_TRAIN_COUNT], old_labels[key]):
            raise ValueError(f"20k train {key} does not preserve the original 10k prefix")
    old_prefix = png_hash(parent_data / "train", PARENT_TRAIN_COUNT)
    new_prefix = png_hash(data / "train", PARENT_TRAIN_COUNT)
    recorded_prefix = parent_provenance["datasets"]["train"]["actual_images_concat_sha256"]
    if old_prefix != recorded_prefix or new_prefix != recorded_prefix:
        raise ValueError("The original training PNG bytes changed or were not retained as the 20k prefix")
    result = {"train_prefix_count": PARENT_TRAIN_COUNT,
              "train_prefix_png_concat_sha256": new_prefix,
              "train_labels_exact_seed": SPLITS["train"][1], "same_render_metadata": True,
              "validation_test": {}}
    for split in ("val", "test"):
        count, seed = SPLITS[split]
        old_meta = json.loads((parent_data / split / "metadata.json").read_text())
        new_meta = json.loads((data / split / "metadata.json").read_text())
        if (new_meta.get("count"), new_meta.get("seed")) != (count, seed):
            raise ValueError(f"{split}: count/seed differs from the completed parent")
        label_hash = base.sha256(data / split / "poses.npz")
        if label_hash != base.sha256(parent_data / split / "poses.npz") or label_hash != old_meta["poses_sha256"]:
            raise ValueError(f"{split}: labels differ from the completed parent bytes")
        old_png, new_png = png_hash(parent_data / split, count), png_hash(data / split, count)
        if old_png != old_meta["images_concat_sha256"] or new_png != old_png or new_meta.get("images_concat_sha256") != old_png:
            raise ValueError(f"{split}: images differ from the completed parent bytes")
        result["validation_test"][split] = {"count": count, "seed": seed,
                                             "poses_sha256": label_hash,
                                             "images_concat_sha256": new_png}
    return result


def inspect_parent(parent_run):
    """Validate completion and snapshot parent artifacts without changing them."""
    parent_run = Path(parent_run)
    config = json.loads((parent_run / "config.json").read_text())
    provenance = json.loads((parent_run / "provenance.json").read_text())
    rows = [json.loads(line) for line in (parent_run / "metrics.jsonl").read_text().splitlines()]
    test = json.loads((parent_run / "test_metrics.json").read_text())
    if [row["epoch"] for row in rows] != list(range(1, PARENT_EPOCHS + 1)):
        raise ValueError("Parent must have naturally completed every epoch 1..50")
    if rows[-1]["total_training_images_seen"] != PARENT_SEEN or test["total_training_images_seen"] != PARENT_SEEN:
        raise ValueError("Parent training exposure must be exactly 500,000 images")
    if config["epochs"] != PARENT_EPOCHS or config["batch_size"] != BATCH_SIZE or config["weight_decay"] != 1e-4:
        raise ValueError("Parent epoch/batch/weight-decay configuration differs")
    if config["split_sizes_seeds"] != {key: [count, seed] for key, (count, seed) in base.SPLITS.items()}:
        raise ValueError("Parent split sizes/seeds differ from the frozen experiment")
    if test["epochs_completed"] != PARENT_EPOCHS or test["selection"] != "validation NLL only":
        raise ValueError("Parent test record does not describe a completed 50-epoch run")
    current_sources = base.current_sources(config.get("monitor_inputs"))
    if provenance["source_sha256"] != current_sources:
        raise ValueError("Frozen parent training/model/loss/render/observer sources changed")
    if provenance["input_sha256"] != base.input_manifest(Path(config["data"])):
        raise ValueError("Parent input manifest changed after its completed training")
    filenames = ("last.pt", "best.pt", "config.json", "provenance.json", "metrics.jsonl", "test_metrics.json")
    snapshot = {"run": str(parent_run.resolve()), "epoch": PARENT_EPOCHS,
                "total_training_images_seen": PARENT_SEEN,
                "best_epoch": rows[-1]["best_epoch"], "best_val_nll": rows[-1]["best_val_nll"],
                "source_sha256": current_sources, "input_sha256": provenance["input_sha256"],
                "files": {name: {"sha256": base.sha256(parent_run / name),
                                  "bytes": (parent_run / name).stat().st_size} for name in filenames}}
    return config, provenance, snapshot


def extension_config(args, parent_config, parent_snapshot):
    config = dict(parent_config)
    config.update(data=str(args.data.resolve()), output=str(args.output.resolve()), epochs=FINAL_EPOCH,
                  learning_rate=args.learning_rate, device=args.device, workers=args.workers,
                  cpu_threads=args.cpu_threads, preload=False,
                  monitor_inputs=str(args.monitor_inputs.resolve()), monitor_inputs_sha256=base.sha256(args.monitor_inputs),
                  split_sizes_seeds={key: list(value) for key, value in SPLITS.items()},
                  parent_run=str(args.parent_run.resolve()), parent_epoch=PARENT_EPOCHS,
                  additional_epochs=FINAL_EPOCH - PARENT_EPOCHS,
                  total_training_images_seen_at_start=PARENT_SEEN,
                  final_total_training_images_seen=progress_at_epoch(FINAL_EPOCH)["total_training_images_seen"],
                  scheduler="CosineAnnealingLR: new approved LR at epoch51, T_max=50, eta_min=0",
                  optimizer_continuation="parent last.pt AdamW moments and step tensors; only lr/initial_lr reset",
                  parent_last_sha256=parent_snapshot["files"]["last.pt"]["sha256"],
                  parent_best_sha256=parent_snapshot["files"]["best.pt"]["sha256"])
    return config


def assert_distinct_paths(parent_run, parent_data, output, data):
    parent_run, parent_data, output, data = map(lambda path: Path(path).resolve(), (parent_run, parent_data, output, data))
    if output == parent_run or parent_run in output.parents:
        raise ValueError("The completed parent run is read-only; choose a separate output directory")
    if (data == parent_data or parent_data in data.parents or output == data
            or output == parent_data or parent_data in output.parents):
        raise ValueError("Extension data and output must be separate from the completed parent data")


def run_extension(args):
    parent_config, parent_provenance, parent_snapshot = inspect_parent(args.parent_run)
    assert_distinct_paths(args.parent_run, parent_config["data"], args.output, args.data)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this process")
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = ("best.pt", "last.pt", "config.json", "provenance.json", "metrics.jsonl", "test_metrics.json")
    if not args.resume and any((args.output / name).exists() for name in artifacts):
        raise FileExistsError("Extension output already contains training; use its --resume last.pt or another folder")
    config = extension_config(args, parent_config, parent_snapshot)
    sources = dict(base.current_sources(args.monitor_inputs), **{"extend_training.py": base.sha256(Path(__file__))})
    manifest = base.input_manifest(args.data)
    lineage = validate_data_lineage(Path(parent_config["data"]), args.data, parent_provenance)
    observer = base.TangentObserver(args.monitor_inputs)
    if observer.input_sha256 != config["monitor_inputs_sha256"]:
        raise ValueError("Monitor packet changed while loading the extension")
    base.seed_everything(parent_config["seed"], args.cpu_threads)
    datasets = {}
    for split in ("train", "val"):
        count, seed = SPLITS[split]
        print(f"Verifying extension {split}: {count} whole RGB images", flush=True)
        datasets[split] = base.PoseDataset(args.data / split, preload=False,
                                          expected_count=count, expected_seed=seed)
    model = base.VGG16Bingham(amp_bfloat16=device.type == "cuda" and torch.cuda.is_bf16_supported())
    model.to(device=device, memory_format=torch.channels_last)
    counts = base.parameter_counts(model)
    if counts != {"total": 134297417, "trainable": 134297417, "frozen": 0}:
        raise AssertionError("Full VGG16 parameter count/trainability changed")
    criterion = base.BinghamNLL(device=device, nodes=parent_config["quadrature_nodes"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4, foreach=False)
    checkpoint = base.load_checkpoint(args.resume or args.parent_run / "last.pt")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    shuffle_generator, worker_generator = torch.Generator(), torch.Generator()
    if args.resume:
        if checkpoint["config"] != config or checkpoint["source_sha256"] != sources or checkpoint["input_sha256"] != manifest:
            raise ValueError("Extension resume configuration/source/input hashes differ")
        if checkpoint["parent_snapshot"] != parent_snapshot:
            raise ValueError("Read-only parent artifacts changed since the extension started")
        scheduler = resume_cosine(optimizer, checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
    else:
        if checkpoint["epoch"] != PARENT_EPOCHS or checkpoint["total_training_images_seen"] != PARENT_SEEN:
            raise ValueError("Continuation must start from parent epoch50 last.pt")
        if checkpoint["config"] != parent_config or checkpoint["source_sha256"] != parent_snapshot["source_sha256"]:
            raise ValueError("Parent last checkpoint metadata differs from its recorded configuration/sources")
        if checkpoint["input_sha256"] != parent_snapshot["input_sha256"]:
            raise ValueError("Parent last checkpoint input manifest differs")
        scheduler = restart_cosine(optimizer, args.learning_rate)
        start_epoch = PARENT_EPOCHS
        # This is a read-only copy of the parent's weights-only selected
        # checkpoint. Its validation NLL remains the initial best criterion.
        shutil.copyfile(args.parent_run / "best.pt", args.output / "best.pt")
        base.write_json(args.output / "config.json", config)
        base.write_json(args.output / "provenance.json", {
            "config": config, "source_sha256": sources, "input_sha256": manifest,
            "parent_snapshot": parent_snapshot, "data_lineage": lineage,
            "model_parameter_counts": counts, "environment": base.environment(device),
            "amp_feature_dtype": "bfloat16" if model.amp_bfloat16 else "float32",
            "final_head_dtype": "float32", "nll_dtype": "float64",
            "datasets": {key: dataset.provenance() for key, dataset in datasets.items()},
            "test_access": "Existing validation/test bytes verified unchanged; extension test inference only after epoch100",
            "rng_continuation": "parent Python/numpy/Torch/CUDA/shuffle/worker RNG restored after model and optimizer initialization"})
    expected = progress_at_epoch(start_epoch)
    total_seen = checkpoint["total_training_images_seen"]
    if total_seen != expected["total_training_images_seen"]:
        raise ValueError("Saved extension exposure count is inconsistent with global epoch")
    best_val, best_epoch = checkpoint["best_val_nll"], checkpoint["best_epoch"]
    if not args.resume and (best_val != parent_snapshot["best_val_nll"] or best_epoch != parent_snapshot["best_epoch"]):
        raise ValueError("Parent best criterion differs from completed epoch50 metadata")
    base.restore_rng(checkpoint["rng"], shuffle_generator, worker_generator)
    del checkpoint
    train_loader = base.make_loader(datasets["train"], BATCH_SIZE, args.workers, device,
                                    shuffle_generator=shuffle_generator, worker_generator=worker_generator)
    val_loader = base.make_loader(datasets["val"], BATCH_SIZE, args.workers, device,
                                  worker_generator=worker_generator)
    print(json.dumps({"continuation_from_epoch": start_epoch, "to_epoch": FINAL_EPOCH,
                      "next_learning_rate": optimizer.param_groups[0]["lr"],
                      "best_epoch": best_epoch, "best_val_nll": best_val,
                      "total_training_images_seen": total_seen}), flush=True)
    for epoch in range(start_epoch + 1, FINAL_EPOCH + 1):
        started = time.perf_counter()
        model.train()
        seen, nll_sum = [], 0.
        epoch_lr = optimizer.param_groups[0]["lr"]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for batch_number, (pixels, target, indices) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(base.normalize_images(pixels, device)), target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite extension NLL; batch was not skipped")
            loss.backward()
            optimizer.step()
            nll_sum += float(loss.detach()) * len(indices)
            seen.append(indices.numpy())
            if batch_number % 100 == 0 or batch_number == len(train_loader):
                print(json.dumps({"progress": "training", "epoch": epoch, "batch": batch_number,
                                  "batches": len(train_loader), "images_seen": sum(map(len, seen)),
                                  "latest_batch_nll": float(loss.detach()),
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
        seen = np.concatenate(seen)
        if not np.array_equal(np.sort(seen), datasets["train"].index):
            raise AssertionError("An extension epoch omitted or duplicated a training index")
        total_seen += len(seen)
        progress = progress_at_epoch(epoch)
        if total_seen != progress["total_training_images_seen"]:
            raise AssertionError("Extension cumulative exposure count changed")
        validation = base.evaluate(model, val_loader, criterion, device)
        improved = validation["nll_mean"] < best_val
        if improved:
            best_val, best_epoch = validation["nll_mean"], epoch
        scheduler.step()
        coverage = {"seen": len(seen), "unique": len(np.unique(seen)),
                    "sorted_indices_sha256": base.index_sha256(np.sort(seen)),
                    "shuffled_order_sha256": base.index_sha256(seen)}
        metrics = dict(progress, learning_rate=epoch_lr, train_nll_mean=nll_sum / len(seen),
                       validation=validation, coverage=coverage, best_epoch=best_epoch, best_val_nll=best_val,
                       seconds_train_and_validation=time.perf_counter() - started,
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
        metrics["tangent_xz"] = observer.evaluate(model, device, epoch=epoch,
                                                 optimizer_step=progress["optimizer_step"])
        common = {"model": model.state_dict(), "epoch": epoch, "config": config,
                  "source_sha256": sources, "input_sha256": manifest, "parent_snapshot": parent_snapshot,
                  "best_val_nll": best_val, "best_epoch": best_epoch,
                  "total_training_images_seen": total_seen, "optimizer_step": progress["optimizer_step"],
                  "coverage": coverage, "model_parameter_counts": counts}
        if improved:
            base.save_checkpoint(args.output / "best.pt", common)
        base.save_checkpoint(args.output / "last.pt", dict(common, optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(), rng=base.rng_state(shuffle_generator, worker_generator)))
        metrics["seconds_including_checkpoint"] = time.perf_counter() - started
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        print(json.dumps(metrics, allow_nan=False), flush=True)
    if (args.output / "test_metrics.json").exists():
        print("Extension final test already exists; not evaluating it again", flush=True)
        return
    if total_seen != progress_at_epoch(FINAL_EPOCH)["total_training_images_seen"]:
        raise AssertionError("Extension test cannot run before global epoch100")
    best = base.load_checkpoint(args.output / "best.pt")
    model.load_state_dict(best["model"])
    del best
    count, seed = SPLITS["test"]
    test_dataset = base.PoseDataset(args.data / "test", preload=False, expected_count=count, expected_seed=seed)
    if base.input_manifest(args.data) != manifest:
        raise ValueError("Extension input metadata changed during training")
    test_loader = base.make_loader(test_dataset, BATCH_SIZE, args.workers, device, worker_generator=worker_generator)
    test_metrics, arrays = base.evaluate(model, test_loader, criterion, device, retain_predictions=True)
    if not np.array_equal(arrays["index"], test_dataset.index):
        raise AssertionError("Extension test predictions differ from original indices")
    np.savez_compressed(args.output / "test_predictions.npz", **arrays)
    test_metrics.update(checkpoint="best.pt", best_epoch=best_epoch, selection="validation NLL only",
                        epochs_completed=FINAL_EPOCH, additional_epochs_completed=FINAL_EPOCH - PARENT_EPOCHS,
                        total_training_images_seen=total_seen, parent_run=str(args.parent_run.resolve()),
                        dataset=test_dataset.provenance(), predictions_sha256=base.sha256(args.output / "test_predictions.npz"))
    base.write_json(args.output / "test_metrics.json", test_metrics)
    print(json.dumps({"final_test": test_metrics}, allow_nan=False), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--learning-rate", type=float, required=True,
                        help="explicit user-approved LR for epoch51; cosine reaches zero after epoch100")
    parser.add_argument("--monitor-inputs", type=Path, help="defaults to the NEW output directory's inputs.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", type=Path, help="resume this separate extension's own last.pt")
    args = parser.parse_args(argv)
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.workers < 0 or args.cpu_threads < 1:
        parser.error("Require positive finite learning rate, workers>=0 and cpu-threads>=1")
    if args.monitor_inputs is None:
        args.monitor_inputs = args.output / "inputs.npz"
    return args


if __name__ == "__main__":
    run_extension(parse_args())
