"""Raw, asynchronous two-IMU input to native ProbTF, without ROS imports."""

from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from relative_coords_from_paired_imu.imu_extended_leastsq import ExtendedLeastSquaresImuRelativePoseEstimator
from relative_coords_from_paired_imu.imu_kinematics import ImuKinematics, _covariance, _vector


def imu_measurement_covariance(values, fallback_variance):
    """Interpret sensor_msgs/Imu covariance conventions without ROS types.

    All-zero covariance means unknown, NOT a noiseless measurement.  A -1
    first entry means the measurement is unavailable and must not be fused.
    Other supplied full covariances are retained, including off-diagonals.
    """

    covariance = np.asarray(values, dtype=float).reshape(3, 3)
    if covariance[0, 0] == -1.0:
        raise ValueError("Required gyro/acceleration measurement is unavailable (covariance[0] = -1).")
    if np.all(covariance == 0.0):
        variance = float(fallback_variance)
        if not np.isfinite(variance) or variance <= 0.0:
            raise ValueError("Unknown-covariance fallback variance must be positive and finite.")
        return np.eye(3) * variance
    return _covariance(covariance, "measurement covariance")


@dataclass(frozen=True)
class _SgWindow:
    start: float
    end: float
    center: float
    scale: float
    coefficients: np.ndarray
    fit_weights: np.ndarray
    gyro_covariances: np.ndarray
    force_covariances: np.ndarray

    @classmethod
    def fit(cls, samples, polynomial_order):
        center = samples[(len(samples) - 1) // 2][0]
        times = np.array([sample[0] - center for sample in samples])
        scale = float(np.max(np.abs(times)))
        design = np.vander(times / scale, N=polynomial_order + 1, increasing=True)
        weights = np.linalg.pinv(design)
        values = np.column_stack((
            np.stack([sample[1] for sample in samples]),
            np.stack([sample[2] for sample in samples]),
        ))
        return cls(
            samples[0][0], samples[-1][0], center, scale, weights @ values, weights,
            np.stack([sample[3] for sample in samples]),
            np.stack([sample[4] for sample in samples]),
        )

    def evaluate(self, frame_id, stamp):
        if not self.start <= stamp <= self.end:
            raise ValueError("SG evaluation must not extrapolate outside its sample window.")
        # The SAME polynomial gives the interpolated value and derivative.
        # A positive evaluation-time shift uses +(t - center), for both.
        offset = (stamp - self.center) / self.scale
        degree = self.coefficients.shape[0] - 1
        basis = offset ** np.arange(degree + 1)
        derivative_basis = np.zeros(degree + 1)
        derivative_basis[1:] = np.arange(1, degree + 1) * offset ** np.arange(degree) / self.scale
        value = basis @ self.coefficients
        derivative = derivative_basis @ self.coefficients
        value_weights = basis @ self.fit_weights
        derivative_weights = derivative_basis @ self.fit_weights
        return ImuKinematics(
            frame_id=frame_id,
            angular_velocity=value[:3],
            angular_acceleration=derivative[:3],
            specific_force=value[3:],
            angular_velocity_covariance=np.einsum("i,ijk->jk", value_weights ** 2, self.gyro_covariances),
            angular_acceleration_covariance=np.einsum("i,ijk->jk", derivative_weights ** 2, self.gyro_covariances),
            specific_force_covariance=np.einsum("i,ijk->jk", value_weights ** 2, self.force_covariances),
            stamp=stamp,
        )


class SynchronizedImuPairPreprocessor:
    """Source SG window selection, made independent of cross-sensor delivery.

    Each sensor's acquisition timestamps must increase. Earlier/later arrival
    between sensors and arbitrary batch partitions are supported: retain raw
    input until BOTH acquisition frontiers have reached it, then process events
    in timestamp order. Equal-time child input precedes parent input.

    As in the source, each parent event evaluates the midpoint of the rolling
    windows' overlap. For EACH sensor choose the nearest-centered covering
    fit from its last ten SG fits. Never gate on both sensors having new raw
    input since the previous evaluation; interpolation already defines both
    observations at the common time.

    ``update_all`` returns every newly ready pair. The single-result ``update``
    retains extra ready pairs, accessible with ``drain``; it never drops them.
    Raw input awaiting the other sensor is not silently evicted or rate capped.
    """

    def __init__(self, parent_frame_id, child_frame_id, window_size=11, polynomial_order=7,
                 fit_history_size=10):
        self.frames = tuple(str(frame).strip().lstrip("/") for frame in (parent_frame_id, child_frame_id))
        if not all(self.frames) or self.frames[0] == self.frames[1]:
            raise ValueError("Two distinct IMU frame identifiers are required.")
        self.window_size = int(window_size)
        self.polynomial_order = int(polynomial_order)
        if self.window_size != window_size or self.window_size < 3:
            raise ValueError("window_size must be an integer of at least three.")
        if self.polynomial_order != polynomial_order or not 1 <= self.polynomial_order < self.window_size:
            raise ValueError("polynomial_order must be an integer in [1, window_size).")
        self.fit_history_size = int(fit_history_size)
        if self.fit_history_size != fit_history_size or self.fit_history_size < 1:
            raise ValueError("fit_history_size must be a positive integer.")
        # A logical selector and a full selector for the same base name
        # would both match one physical sensor on the multiplexed topic.
        for index, selector in enumerate(self.frames):
            other = self.frames[1 - index]
            if "__" not in selector and other.split("__", 1)[0] == selector:
                raise ValueError("The two selectors overlap; select two different IMUs.")
        self.reset()

    def reset(self):
        self.samples = {frame: deque(maxlen=self.window_size) for frame in self.frames}
        self.fit_history = {frame: deque(maxlen=self.fit_history_size) for frame in self.frames}
        self.pending = {frame: deque() for frame in self.frames}
        self.input_stamps = {}
        self._ready_pairs = deque()
        self.bound_frames = {}
        self.last_frontier = None
        self._last_generated_stamp = None
        self.last_stamp = None
        self.last_support = None

    def _select(self, raw_frame_id):
        raw_frame_id = str(raw_frame_id).strip().lstrip("/")
        for selector in self.frames:
            if raw_frame_id == selector or (
                "__" not in selector and raw_frame_id.split("__", 1)[0] == selector
            ):
                previous = self.bound_frames.get(selector)
                if previous is not None and previous != raw_frame_id:
                    raise ValueError(
                        "IMU selector {} is ambiguous: {} and {}. Use the full frame_id including MAC.".format(
                            selector, previous, raw_frame_id
                        )
                    )
                return selector, raw_frame_id
        return None, raw_frame_id

    def select_frame(self, frame_id):
        """Return the matching configured frame, or None for other IMUs."""
        return self._select(frame_id)[0]

    @staticmethod
    def _evaluate(frame_id, samples, stamp, polynomial_order):
        return _SgWindow.fit(samples, polynomial_order).evaluate(frame_id, stamp)

    def add_sample(self, frame_id, stamp, angular_velocity, specific_force,
                   angular_velocity_covariance, specific_force_covariance):
        """Ingest one asynchronous sample; queue every newly ready evaluation."""
        selector, raw_frame = self._select(frame_id)
        if selector is None:
            return
        stamp = float(stamp)
        if not np.isfinite(stamp) or stamp < 0.0:
            raise ValueError("Acquisition timestamps must be finite and non-negative.")
        if selector in self.input_stamps and stamp <= self.input_stamps[selector]:
            raise ValueError("IMU {} acquisition timestamps must be strictly increasing.".format(selector))
        sample = (
            stamp,
            _vector(angular_velocity, "angular_velocity").copy(),
            _vector(specific_force, "specific_force").copy(),
            _covariance(angular_velocity_covariance, "angular_velocity_covariance").copy(),
            _covariance(specific_force_covariance, "specific_force_covariance").copy(),
        )
        self.pending[selector].append(sample)
        self.input_stamps[selector] = stamp
        self.bound_frames[selector] = raw_frame
        if len(self.input_stamps) != 2:
            return
        frontier = min(self.input_stamps.values())
        while True:
            # The child wins exact timestamp ties, so a parent observation
            # sees the child's simultaneous sample even after batched delivery.
            candidates = [
                (self.pending[frame][0][0], priority, frame)
                for priority, frame in enumerate(reversed(self.frames))
                if self.pending[frame] and self.pending[frame][0][0] <= frontier
            ]
            if not candidates:
                break
            _, _, frame = min(candidates)
            self._process_acquisition(frame, self.pending[frame].popleft())
        self.last_frontier = frontier

    def _process_acquisition(self, frame, sample):
        buffer = self.samples[frame]
        buffer.append(sample)
        if len(buffer) == self.window_size:
            self.fit_history[frame].append(_SgWindow.fit(tuple(buffer), self.polynomial_order))
        if frame != self.frames[0] or any(not self.fit_history[key] for key in self.frames):
            return
        lower = max(self.samples[frame][0][0] for frame in self.frames)
        upper = min(self.samples[frame][-1][0] for frame in self.frames)
        if lower > upper:
            return
        common_stamp = lower + (upper - lower) * 0.5
        if self._last_generated_stamp is not None and common_stamp <= self._last_generated_stamp:
            return
        windows = {
            frame: min(
                (fit for fit in self.fit_history[frame] if fit.start <= common_stamp <= fit.end),
                key=lambda fit: abs(fit.center - common_stamp),
            ) for frame in self.frames
        }
        observations = tuple(windows[frame].evaluate(frame, common_stamp) for frame in self.frames)
        support = {
            frame: (self.bound_frames[frame], windows[frame].start, windows[frame].end)
            for frame in self.frames
        }
        self._ready_pairs.append((observations, support))
        self._last_generated_stamp = common_stamp

    def pop_pair(self):
        """Return the oldest ready pair, updating its matching provenance."""
        if not self._ready_pairs:
            return None
        pair, self.last_support = self._ready_pairs.popleft()
        self.last_stamp = pair[0].stamp
        return pair

    def drain(self):
        while self._ready_pairs:
            yield self.pop_pair()

    def update(self, *args, **kwargs):
        """Single-result compatibility API; extra ready pairs stay queued."""
        self.add_sample(*args, **kwargs)
        return self.pop_pair()

    def update_all(self, *args, **kwargs):
        self.add_sample(*args, **kwargs)
        return tuple(self.drain())


class ImuRelativePosePipeline:
    """Feed asynchronous raw IMUs; update_all returns all ready ProbTF records."""

    def __init__(self, parent_frame_id, child_frame_id, window_size=11,
                 polynomial_order=7, fit_history_size=10, **estimator_options):
        self.preprocessor = SynchronizedImuPairPreprocessor(
            parent_frame_id, child_frame_id, window_size, polynomial_order, fit_history_size
        )
        self.estimator = ExtendedLeastSquaresImuRelativePoseEstimator(
            parent_frame_id, child_frame_id, **estimator_options
        )
        self.last_pair = None
        self._ready_records = deque()

    def reset(self):
        self.preprocessor.reset()
        self.estimator.reset()
        self.last_pair = None
        self._ready_records.clear()

    def _estimate_pair(self, pair):
        record = self.estimator.update(*pair)
        support = self.preprocessor.last_support
        source_ids = (self.estimator.source_id,) + tuple(support[frame][0] for frame in self.preprocessor.frames)
        detail = record.provenance.detail + " Polynomial input windows: " + "; ".join(
            "{} [{:.9f}, {:.9f}]".format(*support[frame]) for frame in self.preprocessor.frames
        )
        provenance = replace(record.provenance, source_ids=source_ids, detail=detail)
        components = tuple(
            replace(component, provenance=replace(component.provenance, source_ids=source_ids))
            for component in record.distribution.components
        )
        # Only provenance changes; retain exactly the estimator's laws.
        return replace(record, provenance=provenance,
                       distribution=replace(record.distribution, components=components))

    def _process_input(self, *args, **kwargs):
        self.preprocessor.add_sample(*args, **kwargs)
        for pair in self.preprocessor.drain():
            self._ready_records.append((self._estimate_pair(pair), pair))

    def pop_record(self):
        if not self._ready_records:
            return None
        record, self.last_pair = self._ready_records.popleft()
        return record

    def drain(self):
        """Return all remaining records in acquisition-time order."""
        while self._ready_records:
            yield self.pop_record()

    def update(self, *args, **kwargs):
        """Single-result compatibility API; call drain() after finite input."""
        self._process_input(*args, **kwargs)
        return self.pop_record()

    def update_all(self, *args, **kwargs):
        """Process one input and return EVERY newly ready posterior record."""
        self._process_input(*args, **kwargs)
        return tuple(self.drain())
