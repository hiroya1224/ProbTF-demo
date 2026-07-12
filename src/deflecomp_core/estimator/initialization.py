from typing import Tuple

import numpy as np


def initial_log_kp_std(kp_lim: Tuple[float, float]) -> float:
    kp_min, kp_max = (float(value) for value in kp_lim)
    if kp_min <= 0.0 or kp_max <= kp_min:
        raise ValueError(f"kp_min/kp_max must satisfy 0 < kp_min < kp_max, got {kp_lim}")
    return (np.log(kp_max) - np.log(kp_min)) / 4.0


def initial_log_kp_state(size: int, kp_lim: Tuple[float, float]) -> np.ndarray:
    kp_min, kp_max = (float(value) for value in kp_lim)
    if kp_min <= 0.0 or kp_max <= kp_min:
        raise ValueError(f"kp_min/kp_max must satisfy 0 < kp_min < kp_max, got {kp_lim}")
    return np.full(int(size), 0.5 * (np.log(kp_min) + np.log(kp_max)), dtype=float)

