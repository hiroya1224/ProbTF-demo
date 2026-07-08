import numpy as np


def wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    return (theta + np.pi) % (2.0 * np.pi) - np.pi
