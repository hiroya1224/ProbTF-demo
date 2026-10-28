"""Local linear stability fractions on the entire fixed plant sample set.

The denominator is always N. Non-finite candidate radii do not count as
stable, and are reported separately. No successful-controller baseline is
accepted by this calculation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SampleStability:
    sample_count: int
    stable_fraction: np.ndarray
    stable_count: np.ndarray
    valid_count: np.ndarray
    unresolved_count: np.ndarray
    alpha_classification: np.ndarray


def sample_stability(spectral_radius: np.ndarray, evaluated: np.ndarray,
                     alpha: float = 0.95) -> SampleStability:
    radii = np.asarray(spectral_radius, dtype=float)
    done = np.asarray(evaluated, dtype=bool)
    if radii.ndim != done.ndim + 1 or radii.shape[:-1] != done.shape or radii.shape[-1] == 0:
        raise ValueError('plant sample count or gain-grid shape differs')
    if not (0 < float(alpha) <= 1):
        raise ValueError('alpha must lie in (0, 1]')
    count = radii.shape[-1]
    finite = np.isfinite(radii)
    stable = finite & (radii < 1.0)
    valid_count = np.count_nonzero(finite, axis=-1).astype(np.int32)
    stable_count = np.count_nonzero(stable, axis=-1).astype(np.int32)
    unresolved = count - valid_count
    fraction = np.where(done, stable_count.astype(float) / count, np.nan)
    stable_count = np.where(done, stable_count, -1).astype(np.int32)
    valid_count = np.where(done, valid_count, -1).astype(np.int32)
    unresolved = np.where(done, unresolved, -1).astype(np.int32)
    classification = np.zeros(done.shape, dtype=np.uint8)
    classification[done & (fraction < alpha)] = 2
    classification[done & (fraction >= alpha)] = 3
    return SampleStability(count, fraction, stable_count, valid_count,
                           unresolved, classification)
