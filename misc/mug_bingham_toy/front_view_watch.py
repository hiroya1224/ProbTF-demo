#!/usr/bin/env python3
"""Refresh the true 0-degree display from newly saved best checkpoints.

This observer runs existing CPU inference and plotting in a separate process.
It does not train, modify checkpoints, render substitute inputs, or alter the
training process. Inference and plotting failures remain visible exceptions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

import toyexample


def checkpoint_signature(path: Path):
    """Notice atomic checkpoint replacement even when its filename is fixed."""
    try:
        state = path.stat()
    except FileNotFoundError:
        return None
    return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns


def training_tail_ready(run: Path) -> bool:
    """Yield the final refresh to the training launcher's existing tail."""
    if (run / "test_metrics.json").exists():
        return True
    configuration = run / "config.json"
    epochs = int(json.loads(configuration.read_text())["epochs"]) if configuration.exists() else 50
    try:
        lines = (run / "metrics.jsonl").read_bytes().splitlines(keepends=True)
    except FileNotFoundError:
        return False
    complete = [line for line in lines if line.endswith(b"\n") and line.strip()]
    return bool(complete and int(json.loads(complete[-1])["epoch"]) >= epochs)


def watch(run: Path, output: Path, *, interval: float = 2., samples: int = 4096,
          sample_seed: int = 20261127):
    run, output = Path(run).resolve(), Path(output).resolve()
    with np.load(output / "inputs.npz", allow_pickle=False) as inputs:
        if float(inputs["elevation_degrees"]) != 0.:
            raise ValueError("The front-view observer requires actual rendered inputs at elevation 0 degrees")
    arguments = argparse.Namespace(run=run, output=output, device="cpu",
                                   samples=samples, sample_seed=sample_seed)
    checkpoint = run / "best.pt"
    processed = None
    waiting_announced = False
    while True:
        if training_tail_ready(run):
            print("Training finalization started; its existing launcher tail owns the final front-view refresh", flush=True)
            return
        signature = checkpoint_signature(checkpoint)
        if signature is None:
            if not waiting_announced:
                print(f"Waiting for checkpoint: {checkpoint}", flush=True)
                waiting_announced = True
        elif processed is None or signature != processed:
            print("Refreshing 0-degree images from best.pt on CPU (training in progress)", flush=True)
            toyexample.infer(arguments)
            if training_tail_ready(run):
                print("Training finalization started during inference; leaving final plotting to the launcher tail", flush=True)
                return
            # Rendering uses the system interpreter, where the existing
            # native ProbTF sampler is built. Torch inference uses its venv.
            # Both read exactly the same saved prediction packet.
            subprocess.run(['/usr/bin/python3',str(Path(toyexample.__file__).resolve()),
                'plot','--output',str(output),'--samples',str(samples),
                '--sample-seed',str(sample_seed)],check=True)
            if checkpoint_signature(checkpoint) != signature:
                print("best.pt changed during refresh; reading the new checkpoint immediately", flush=True)
                continue
            processed = signature
            print(f"Updated front-view display: {output}", flush=True)
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=toyexample.DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=toyexample.DEFAULT_OUTPUT)
    parser.add_argument("--interval", type=float, default=2.)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--sample-seed", type=int, default=20261127)
    args = parser.parse_args()
    if not np.isfinite(args.interval) or args.interval <= 0.:
        parser.error("--interval must be finite and positive")
    if args.samples < 1:
        parser.error("--samples must be positive")
    if not 0 <= args.sample_seed < 1 << 64:
        parser.error("--sample-seed must be an unsigned 64-bit integer")
    watch(args.run, args.output, interval=args.interval, samples=args.samples, sample_seed=args.sample_seed)


if __name__ == "__main__":
    main()
