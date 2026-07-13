import threading
from bisect import bisect_left, bisect_right
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

    def latest_timestamp(self) -> Optional[float]:
        """Return the newest sample timestamp without exposing buffer storage."""
        with self.lock:
            if not self.t_list:
                return None
            return float(self.t_list[-1])

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


class TimedVectorHistory:
    """Timestamped, piecewise-constant vector history.

    This is used to match a delayed observation to the command that could
    causally have produced it.  ``settled_value_at`` additionally requires the
    vector to have stayed within a configured tolerance for the entire dwell
    window.  Comparing the whole window to its final value (rather than only
    adjacent samples) also detects a slow ramp whose per-cycle increments are
    individually small.
    """

    def __init__(self, maxlen: int = 4000) -> None:
        self.t_list: List[float] = []
        self.v_list: List[np.ndarray] = []
        self.maxlen = max(1, int(maxlen))
        self.lock = threading.RLock()

    def push(self, timestamp: float, value: np.ndarray) -> None:
        stamp = float(timestamp)
        vector = np.asarray(value, dtype=float).reshape(-1).copy()
        if not np.isfinite(stamp) or not np.all(np.isfinite(vector)):
            raise ValueError("TimedVectorHistory requires finite timestamps and values")
        with self.lock:
            if self.v_list and vector.shape != self.v_list[0].shape:
                raise ValueError("TimedVectorHistory vector size cannot change")
            index = bisect_left(self.t_list, stamp)
            if index < len(self.t_list) and abs(self.t_list[index] - stamp) < 1e-12:
                self.t_list[index] = stamp
                self.v_list[index] = vector
            else:
                self.t_list.insert(index, stamp)
                self.v_list.insert(index, vector)
            while len(self.t_list) > self.maxlen:
                self.t_list.pop(0)
                self.v_list.pop(0)

    def clear(self) -> None:
        with self.lock:
            self.t_list.clear()
            self.v_list.clear()

    def value_at(
        self,
        observation_stamp: float,
        apply_delay: float = 0.0,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Return the latest value effective at ``observation_stamp``.

        The returned stamp is the assumed application time, i.e. publication
        stamp plus ``apply_delay``.
        """
        query_publish_stamp = float(observation_stamp) - max(0.0, float(apply_delay))
        with self.lock:
            index = bisect_right(self.t_list, query_publish_stamp + 1.0e-12) - 1
            if index < 0:
                return None
            return (
                self.v_list[index].copy(),
                float(self.t_list[index] + max(0.0, float(apply_delay))),
            )

    def settled_value_at(
        self,
        observation_stamp: float,
        dwell_time: float,
        tolerance: float,
        apply_delay: float = 0.0,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Return a causal value only if it was stable for a full dwell window.

        Stability is measured using the infinity norm against the value active
        at the end of the window.  A record at or before the start of the
        window is required, so a newly-created history can never be declared
        settled prematurely.
        """
        delay = max(0.0, float(apply_delay))
        dwell = max(0.0, float(dwell_time))
        tol = max(0.0, float(tolerance))
        query_publish_stamp = float(observation_stamp) - delay
        start_publish_stamp = query_publish_stamp - dwell
        with self.lock:
            end_index = bisect_right(self.t_list, query_publish_stamp + 1.0e-12) - 1
            start_index = bisect_right(self.t_list, start_publish_stamp + 1.0e-12) - 1
            if end_index < 0 or start_index < 0:
                return None
            candidate = self.v_list[end_index]
            for vector in self.v_list[start_index : end_index + 1]:
                if float(np.max(np.abs(vector - candidate), initial=0.0)) > tol:
                    return None
            return candidate.copy(), float(self.t_list[end_index] + delay)
