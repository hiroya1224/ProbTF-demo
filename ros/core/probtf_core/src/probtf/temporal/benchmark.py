"""Reproducible benchmark metadata and small statistical helpers."""

from dataclasses import dataclass
import gc
import importlib.metadata
import os
import platform
import sys
import time
import tracemalloc

import numpy as np


@dataclass(frozen=True)
class BenchmarkSummary:
    repetitions: int
    p50_seconds: float
    p95_seconds: float
    peak_bytes: int


def _cpu_model():
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as stream:
            for line in stream:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment_manifest(random_seeds=(), packages=("numpy", "scipy")):
    """Capture hardware/runtime inputs required to interpret a benchmark."""

    versions = {}
    for package in packages:
        try:
            versions[str(package)] = importlib.metadata.version(str(package))
        except importlib.metadata.PackageNotFoundError:
            versions[str(package)] = "not-installed"
    return {
        "cpu_count": os.cpu_count(),
        "cpu_model": _cpu_model(),
        "machine": platform.machine(),
        "numpy_float": np.dtype(float).str,
        "packages": versions,
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "random_seeds": [int(seed) for seed in random_seeds],
    }


def benchmark_callable(function, repetitions=30, warmups=5):
    """Return p50/p95 latency and peak traced Python memory."""

    repetitions = int(repetitions)
    warmups = int(warmups)
    if repetitions < 1 or warmups < 0:
        raise ValueError("repetitions must be positive and warmups non-negative.")
    for _ in range(warmups):
        function()
    gc.collect()
    tracemalloc.start()
    durations = []
    try:
        for _ in range(repetitions):
            started = time.perf_counter()
            function()
            durations.append(time.perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return BenchmarkSummary(
        repetitions=repetitions,
        p50_seconds=float(np.quantile(durations, 0.50)),
        p95_seconds=float(np.quantile(durations, 0.95)),
        peak_bytes=int(peak),
    )


def bootstrap_mean_confidence_interval(
    values,
    *,
    confidence=0.95,
    resamples=4000,
    seed=0,
):
    """Percentile bootstrap interval for an episode/seed-level mean."""

    samples = np.asarray(values, dtype=float).reshape(-1)
    if not len(samples) or not np.all(np.isfinite(samples)):
        raise ValueError("values must contain at least one finite value.")
    confidence = float(confidence)
    resamples = int(resamples)
    if not 0.0 < confidence < 1.0 or resamples < 1:
        raise ValueError("confidence must be within (0,1) and resamples positive.")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(samples), size=(resamples, len(samples)))
    means = np.mean(samples[indices], axis=1)
    tail = 0.5 * (1.0 - confidence)
    return (
        float(np.mean(samples)),
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )


def energy_distance_samples(left, right, max_samples=1024):
    """Multivariate energy distance using deterministic prefix subsampling."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("left and right must be two-dimensional with equal width.")
    if not len(left) or not len(right):
        raise ValueError("left and right must not be empty.")
    count = int(max_samples)
    if count < 1:
        raise ValueError("max_samples must be positive.")
    left = left[:count]
    right = right[:count]

    def mean_distance(first, second):
        difference = first[:, None, :] - second[None, :, :]
        return float(np.mean(np.linalg.norm(difference, axis=2)))

    return max(
        0.0,
        2.0 * mean_distance(left, right)
        - mean_distance(left, left)
        - mean_distance(right, right),
    )


__all__ = [
    "BenchmarkSummary",
    "benchmark_callable",
    "bootstrap_mean_confidence_interval",
    "energy_distance_samples",
    "environment_manifest",
]
