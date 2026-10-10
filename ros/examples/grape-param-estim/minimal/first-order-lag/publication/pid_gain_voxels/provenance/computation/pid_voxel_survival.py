"""The conditional-survival criterion used by the existing PID contours.

Baseline samples with rho(success) >= 1 or an unresolved baseline are excluded
once, before interventions. An unresolved intervention in the retained set
counts as a break, exactly as ExactGroupGainEvaluator.survival_fraction does.
Colors are RdYlGn on the fixed [0, 1] survival scale; alpha is the boundary.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConditionalSurvival:
    baseline_stable_mask: np.ndarray
    baseline_stable_count: int
    survival_fraction: np.ndarray
    caused_break_count: np.ndarray
    conditional_unresolved_count: np.ndarray
    alpha_classification: np.ndarray


def conditional_survival(spectral_radius: np.ndarray, baseline_radius: np.ndarray,
                         evaluated: np.ndarray, alpha: float = 0.95) -> ConditionalSurvival:
    radii = np.asarray(spectral_radius, dtype=float)
    baseline = np.asarray(baseline_radius, dtype=float)
    done = np.asarray(evaluated, dtype=bool)
    if baseline.ndim != 1 or radii.shape != done.shape + baseline.shape:
        raise ValueError('plant sample order/count or gain-grid shape differs')
    if not (0 < float(alpha) < 1):
        raise ValueError('alpha must lie in (0, 1)')
    mask = np.isfinite(baseline) & (baseline < 1.0)
    count = int(np.count_nonzero(mask))
    if count == 0:
        raise ValueError('all-success controller has no stable sample; conditioning is undefined')
    retained = radii[..., mask]
    finite = np.isfinite(retained)
    stable = finite & (retained < 1.0)
    breaks = np.count_nonzero(~stable, axis=-1).astype(np.int32)
    unresolved = np.count_nonzero(~finite, axis=-1).astype(np.int32)
    fraction = 1.0 - breaks.astype(float) / count
    fraction = np.where(done, fraction, np.nan)
    breaks = np.where(done, breaks, -1).astype(np.int32)
    unresolved = np.where(done, unresolved, -1).astype(np.int32)
    classification = np.zeros(done.shape, dtype=np.uint8)
    classification[done & (fraction < alpha)] = 2
    classification[done & (fraction >= alpha)] = 3
    return ConditionalSurvival(mask.copy(), count, fraction, breaks, unresolved, classification)
