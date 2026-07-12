import threading
from bisect import bisect_left
from typing import List, Optional

import numpy as np


class ImuBuffer:
    """Thread-safe timestamped unit-vector buffer with linear interpolation."""

    def __init__(self, maxlen: int = 1000) -> None:
        self.t_list: List[float] = []
        self.g_list: List[np.ndarray] = []
        self.maxlen = int(maxlen)
        self.lock = threading.RLock()

    def push(self, timestamp: float, direction: np.ndarray) -> None:
        unit_direction = np.asarray(direction, dtype=float)
        unit_direction = unit_direction / (np.linalg.norm(unit_direction) + 1e-12)
        with self.lock:
            index = bisect_left(self.t_list, timestamp)
            if index < len(self.t_list) and abs(self.t_list[index] - timestamp) < 1e-12:
                self.t_list[index] = timestamp
                self.g_list[index] = unit_direction
            else:
                self.t_list.insert(index, timestamp)
                self.g_list.insert(index, unit_direction)
            while len(self.t_list) > self.maxlen:
                self.t_list.pop(0)
                self.g_list.pop(0)

    def interpolate(self, timestamp: float) -> Optional[np.ndarray]:
        with self.lock:
            if not self.t_list:
                return None
            if timestamp <= self.t_list[0]:
                return self.g_list[0].copy()
            if timestamp >= self.t_list[-1]:
                return self.g_list[-1].copy()

            index = bisect_left(self.t_list, timestamp)
            t0 = self.t_list[index - 1]
            t1 = self.t_list[index]
            g0 = self.g_list[index - 1]
            g1 = self.g_list[index]
            if t1 - t0 <= 1e-12:
                return g1.copy()
            alpha = (timestamp - t0) / (t1 - t0)
            direction = (1.0 - alpha) * g0 + alpha * g1
            return direction / (np.linalg.norm(direction) + 1e-12)

