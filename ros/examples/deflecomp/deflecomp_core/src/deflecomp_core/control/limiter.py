import numpy as np


def clip_vector(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(values, dtype=float), np.asarray(lower, dtype=float)), np.asarray(upper, dtype=float))
