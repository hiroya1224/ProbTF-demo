#!/usr/bin/env python3
"""Preserve the 10k dataset and render only its next 10k PCG64 poses.

The original dataset is read-only. The new train metadata is published
atomically only after every image and the complete 20k label packet validate.
An internal hash journal permits interrupted generation to resume safely.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

import render
from render import MugRenderer, uniform_quaternions


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "data/cylinder_center_no_blur_20261127"
DEFAULT_OUTPUT = HERE / "data/cylinder_center_no_blur_20000_20261127"
ORIGINAL_TRAIN_COUNT = 10000
TARGET_TRAIN_COUNT = 20000
HELDOUT_COUNT = 1000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def copy_exact(source, destination):
    """Copy bytes and file metadata without introducing a shared hard link."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    if (destination.exists() and destination.read_bytes() == raw
            and (source.stat().st_dev, source.stat().st_ino)
            != (destination.stat().st_dev, destination.stat().st_ino)):
        return raw
    temporary = destination.with_name(destination.name+".copy.tmp")
    shutil.copy2(source, temporary)
    if temporary.read_bytes() != raw:
        raise ValueError(f"Source changed during copy: {source}")
    temporary.replace(destination)
    return raw


def validate_png(raw, image_size):
    with Image.open(BytesIO(raw)) as image:
        if image.mode != "RGB" or image.size != tuple(image_size):
            raise ValueError("Expected unchanged RGB image dimensions")
        image.load()


def image_digest(folder, count, image_size):
    digest = hashlib.sha256()
    for index in range(count):
        raw = (folder / "images" / f"{index:06d}.png").read_bytes()
        validate_png(raw, image_size)
        digest.update(raw)
    return digest.hexdigest()


def expand_dataset(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use a separate dataset directory; the original must stay untouched")
    started = time.monotonic()
    renderer_metadata = json.loads((source / "render_metadata.json").read_text())
    if (renderer_metadata["image_size"] != [640, 360]
            or renderer_metadata.get("image_blur_sigma_pixels") != 0.
            or renderer_metadata["source_code_sha256"] != sha(Path(render.__file__))):
        raise ValueError("The frozen 640x360 no-blur renderer does not match the parent dataset")
    counts = {"train": ORIGINAL_TRAIN_COUNT, "val": HELDOUT_COUNT, "test": HELDOUT_COUNT}
    seeds = {"train": 1234, "val": 1235, "test": 1236}
    parent, fingerprint = {}, {"render_metadata.json": sha(source / "render_metadata.json")}
    full_q = uniform_quaternions(TARGET_TRAIN_COUNT, seeds["train"])
    for split, count in counts.items():
        folder = source / split
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["count"] != count or metadata["seed"] != seeds[split]:
            raise ValueError(f"Parent {split} count/seed changed")
        if metadata["poses_sha256"] != sha(folder / "poses.npz"):
            raise ValueError(f"Parent {split} label hash changed")
        with np.load(folder / "poses.npz", allow_pickle=False) as labels:
            expected_q = full_q[:count] if split == "train" else uniform_quaternions(count, seeds[split])
            if (not np.array_equal(labels["q_wxyz"], expected_q)
                    or not np.array_equal(labels["index"], np.arange(count, dtype=np.int64))
                    or not np.array_equal(labels["t_camera"], np.tile([0., 0., 2.5], (count, 1)))
                    or int(labels["seed"]) != seeds[split]):
                raise ValueError(f"Parent {split} poses do not match their frozen PCG64 sequence")
        parent[split] = metadata
        for name in ("metadata.json", "poses.npz"):
            fingerprint[f"{split}/{name}"] = sha(folder / name)
    parent_fingerprint = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    identity = {"parent_fingerprint": parent_fingerprint, "target_count": TARGET_TRAIN_COUNT,
                "original_count": ORIGINAL_TRAIN_COUNT, "seed": 1234,
                "full_q_float64_sha256": hashlib.sha256(full_q.astype("<f8").tobytes()).hexdigest(),
                "renderer_source_sha256": sha(Path(render.__file__)),
                "expansion_source_sha256": sha(Path(__file__))}
    output.mkdir(parents=True, exist_ok=True)
    completed_path = output / "train" / "metadata.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text())
        if (completed.get("expansion", {}).get("parent_fingerprint") != parent_fingerprint
                or completed.get("count") != TARGET_TRAIN_COUNT
                or completed.get("poses_sha256") != sha(output / "train" / "poses.npz")
                or completed.get("images_concat_sha256")
                != image_digest(output / "train", TARGET_TRAIN_COUNT, renderer_metadata["image_size"])):
            raise ValueError("Existing completed expansion does not validate")
        for split in ("val", "test"):
            if (sha(output / split / "metadata.json") != fingerprint[f"{split}/metadata.json"]
                    or sha(output / split / "poses.npz") != fingerprint[f"{split}/poses.npz"]
                    or image_digest(output / split, counts[split], renderer_metadata["image_size"])
                    != parent[split]["images_concat_sha256"]):
                raise ValueError(f"Completed {split} differs from the original split")
        if sha(output / "render_metadata.json") != fingerprint["render_metadata.json"]:
            raise ValueError("Completed renderer metadata differs from the original bytes")
        print(f"Expanded dataset already complete and verified: {output}", flush=True)
        return completed
    journal = output / "train" / ".expansion_progress.jsonl"
    if journal.exists():
        complete = [line for line in journal.read_bytes().splitlines(keepends=True) if line.endswith(b"\n")]
        if not complete or json.loads(complete[0]) != {"identity": identity}:
            raise ValueError("Expansion journal belongs to another source or configuration")
        recorded = [json.loads(line) for line in complete[1:]]
    else:
        if (output / "train" / "metadata.json").exists():
            raise FileExistsError("Existing training metadata has no matching expansion journal")
        recorded = []
    # Verify and copy every original byte. Held-out labels/metadata stay exact.
    for split, count in counts.items():
        digest = hashlib.sha256()
        for index in range(count):
            name = Path("images") / f"{index:06d}.png"
            raw = copy_exact(source / split / name, output / split / name)
            validate_png(raw, renderer_metadata["image_size"])
            digest.update(raw)
        if digest.hexdigest() != parent[split]["images_concat_sha256"]:
            raise ValueError(f"Parent {split} image bytes no longer match their saved hash")
        if split != "train":
            for name in ("poses.npz", "metadata.json"):
                copy_exact(source / split / name, output / split / name)
        print(f"Preserved {split}: {count} original images", flush=True)
    # Skip an extra image only when its completed journal entry, bytes and PNG
    # decoding all validate. An unjournaled or interrupted image is regenerated.
    valid = {}
    for entry in recorded:
        index = entry["index"]
        if not ORIGINAL_TRAIN_COUNT <= index < TARGET_TRAIN_COUNT:
            raise ValueError("Expansion journal contains an out-of-range index")
        path = output / "train" / "images" / f"{index:06d}.png"
        try:
            raw = path.read_bytes()
            validate_png(raw, renderer_metadata["image_size"])
            if hashlib.sha256(raw).hexdigest() == entry["sha256"]:
                valid[index] = entry
        except (OSError, ValueError):
            pass
    temporary = journal.with_name(journal.name+".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps({"identity": identity}, sort_keys=True)+"\n")
        for index in sorted(valid):
            stream.write(json.dumps(valid[index], sort_keys=True)+"\n")
    temporary.replace(journal)
    missing = [index for index in range(ORIGINAL_TRAIN_COUNT, TARGET_TRAIN_COUNT) if index not in valid]
    if missing:
        for key, field in (("EGL_DEVICE_ID", "egl_device_id"), ("PYOPENGL_PLATFORM", "backend")):
            if renderer_metadata.get(field) is not None:
                os.environ.setdefault(key, str(renderer_metadata[field]))
        renderer = MugRenderer(backend=renderer_metadata["backend"])
        try:
            actual_metadata = renderer.metadata()
            if actual_metadata != renderer_metadata:
                different = sorted(key for key in set(actual_metadata) | set(renderer_metadata)
                                   if actual_metadata.get(key) != renderer_metadata.get(key))
                raise ValueError("Renderer differs from frozen parent metadata: "+", ".join(different))
            with journal.open("a") as stream:
                for index in missing:
                    rgb, _ = renderer.render(full_q[index])
                    if rgb.dtype != np.uint8 or rgb.shape != (360, 640, 3):
                        raise ValueError("Renderer did not return the unchanged RGB image format")
                    path = output / "train" / "images" / f"{index:06d}.png"
                    temporary = path.with_name(path.name+".tmp")
                    Image.fromarray(rgb, "RGB").save(temporary, format="PNG", compress_level=3)
                    temporary.replace(path)
                    entry = {"index": index, "sha256": sha(path)}
                    stream.write(json.dumps(entry, sort_keys=True)+"\n")
                    stream.flush()
                    if (index+1) % 1000 == 0 or index == TARGET_TRAIN_COUNT-1:
                        print(f"Expanded train: {index+1}/{TARGET_TRAIN_COUNT}", flush=True)
        finally:
            renderer.close()
    train = output / "train"
    temporary = train / "poses.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, q_wxyz=full_q, index=np.arange(TARGET_TRAIN_COUNT, dtype=np.int64),
                            t_camera=np.tile([0., 0., 2.5], (TARGET_TRAIN_COUNT, 1)),
                            seed=np.array(1234, dtype=np.int64))
    temporary.replace(train / "poses.npz")
    copy_exact(source / "render_metadata.json", output / "render_metadata.json")
    digest = image_digest(train, TARGET_TRAIN_COUNT, renderer_metadata["image_size"])
    metadata = dict(parent["train"], count=TARGET_TRAIN_COUNT, poses_sha256=sha(train / "poses.npz"),
                    images_concat_sha256=digest, elapsed_seconds=time.monotonic()-started,
                    elapsed_scope="this expansion invocation, including copied-byte validation",
                    expansion={**identity, "parent_dataset": str(source), "parent_file_sha256": fingerprint,
                               "preserved_train_index_range": [0, ORIGINAL_TRAIN_COUNT-1],
                               "newly_rendered_index_range": [ORIGINAL_TRAIN_COUNT, TARGET_TRAIN_COUNT-1],
                               "heldout_splits_copied_unchanged": ["val", "test"],
                               "render_metadata_copied_unchanged": True})
    atomic_json(train / "metadata.json", metadata)
    print(f"Expanded dataset complete: {output}; train={TARGET_TRAIN_COUNT}, val/test unchanged", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", "--source-data", dest="source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    expand_dataset(args.source, args.output)


if __name__ == "__main__":
    main()
