import numpy as np


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec.copy()
    return vec / norm
