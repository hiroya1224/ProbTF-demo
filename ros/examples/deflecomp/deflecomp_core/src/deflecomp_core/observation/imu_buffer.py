import threading
from bisect import bisect_left
from typing import List, Optional, Tuple

import numpy as np


def imu_sample_is_quasi_static(
    linear_acceleration: np.ndarray,
    angular_velocity: np.ndarray,
    gravity_norm: float = 9.81,
    acceleration_tolerance: float = 0.75,
    max_angular_speed: float = 0.20,
) -> bool:
    """Return whether an IMU sample is safe to interpret as gravity direction.

    The stiffness observation model is quasi-static: it has no term for link
    linear/angular acceleration.  Dynamic samples must therefore be rejected
    instead of being normalized and mistaken for a precise gravity vector.
    """

    acceleration = np.asarray(linear_acceleration, dtype=float).reshape(-1)
    angular_speed = np.asarray(angular_velocity, dtype=float).reshape(-1)
    if acceleration.size != 3 or angular_speed.size != 3:
        return False
    if not np.all(np.isfinite(acceleration)) or not np.all(np.isfinite(angular_speed)):
        return False
    acceleration_norm = float(np.linalg.norm(acceleration))
    if acceleration_norm < 1e-9:
        return False
    if abs(acceleration_norm - float(gravity_norm)) > max(0.0, float(acceleration_tolerance)):
        return False
    return float(np.linalg.norm(angular_speed)) <= max(0.0, float(max_angular_speed))


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

    def clear(self) -> None:
        with self.lock:
            self.t_list.clear()
            self.g_list.clear()

    def interpolate_with_support_stamp(
        self,
        timestamp: float,
        max_age: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Return an interpolated direction and its newest supporting sample stamp.

        The support stamp is deliberately distinct from the query timestamp.
        In particular, endpoint hold returns the actual endpoint stamp.  A
        consumer can therefore reject repeated reuse of one stopped/slow IMU
        sample instead of re-stamping it as a new independent observation on
        every control cycle.
        """
        max_distance = None if max_age is None else max(0.0, float(max_age))
        with self.lock:
            if not self.t_list:
                return None
            if timestamp <= self.t_list[0]:
                if max_distance is not None and timestamp < self.t_list[0]:
                    # Causal estimator updates must not assign the first future
                    # sample to a command sent before that sample existed.
                    return None
                if max_distance is not None and self.t_list[0] - timestamp > max_distance:
                    return None
                return self.g_list[0].copy(), float(self.t_list[0])
            if timestamp >= self.t_list[-1]:
                if max_distance is not None and timestamp - self.t_list[-1] > max_distance:
                    return None
                return self.g_list[-1].copy(), float(self.t_list[-1])

            index = bisect_left(self.t_list, timestamp)
            t0 = self.t_list[index - 1]
            t1 = self.t_list[index]
            if max_distance is not None and (
                timestamp - t0 > max_distance or t1 - timestamp > max_distance
            ):
                return None
            g0 = self.g_list[index - 1]
            g1 = self.g_list[index]
            if t1 - t0 <= 1e-12:
                return g1.copy(), float(t1)
            alpha = (timestamp - t0) / (t1 - t0)
            direction = (1.0 - alpha) * g0 + alpha * g1
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            return direction, float(t1)

    def interpolate(self, timestamp: float, max_age: Optional[float] = None) -> Optional[np.ndarray]:
        sample = self.interpolate_with_support_stamp(timestamp, max_age=max_age)
        return None if sample is None else sample[0]
