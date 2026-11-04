"""Raw, asynchronous two-IMU input to native ProbTF, without ROS imports."""

from collections import deque
from dataclasses import replace

import numpy as np

from probtf_estimators.imu_extended_leastsq import ExtendedLeastSquaresImuRelativePoseEstimator
from probtf_estimators.imu_kinematics import ImuKinematics, _covariance, _vector


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


class SynchronizedImuPairPreprocessor:
    """Fit the two sensor windows at ONE time inside their overlap.

    This ports the source's overlapping polynomial-window approach. Times
    are centered and scaled before fitting, so ROS epoch stamps and the
    original degree-7, 11-sample fit do not make the Vandermonde ill-scaled.
    Emit when BOTH sensor windows have new input, at a new common acquisition
    frontier. Do not count one held sensor window repeatedly as new paired
    information. There is no timer, fixed output rate, or extrapolation.

    Covariances use the actual linear fit weights and each raw sample's
    full covariance. Independence of raw sample noise and subsequent
    recursive updates is an approximation; overlapping fits are correlated.
    """

    def __init__(self, parent_frame_id, child_frame_id, window_size=11, polynomial_order=7):
        self.frames = tuple(str(frame).strip().lstrip("/") for frame in (parent_frame_id, child_frame_id))
        if not all(self.frames) or self.frames[0] == self.frames[1]:
            raise ValueError("Two distinct IMU frame identifiers are required.")
        self.window_size = int(window_size)
        self.polynomial_order = int(polynomial_order)
        if self.window_size != window_size or self.window_size < 3:
            raise ValueError("window_size must be an integer of at least three.")
        if self.polynomial_order != polynomial_order or not 1 <= self.polynomial_order < self.window_size:
            raise ValueError("polynomial_order must be an integer in [1, window_size).")
        # A logical selector and a full selector for the same base name
        # would both match one physical sensor on the multiplexed topic.
        for index, selector in enumerate(self.frames):
            other = self.frames[1 - index]
            if "__" not in selector and other.split("__", 1)[0] == selector:
                raise ValueError("The two selectors overlap; select two different IMUs.")
        self.reset()

    def reset(self):
        self.samples = {frame: deque(maxlen=self.window_size) for frame in self.frames}
        self.bound_frames = {}
        self.last_frontier = None
        self.last_input_stamps = None
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
        times = np.array([sample[0] - stamp for sample in samples])
        scale = float(np.max(np.abs(times)))
        design = np.vander(times / scale, N=polynomial_order + 1, increasing=True)
        fit = np.linalg.pinv(design)
        value_weights = fit[0]
        derivative_weights = fit[1] / scale
        gyros = np.stack([sample[1] for sample in samples])
        forces = np.stack([sample[2] for sample in samples])
        gyro_covariances = np.stack([sample[3] for sample in samples])
        force_covariances = np.stack([sample[4] for sample in samples])
        return ImuKinematics(
            frame_id=frame_id,
            angular_velocity=value_weights @ gyros,
            angular_acceleration=derivative_weights @ gyros,
            specific_force=value_weights @ forces,
            angular_velocity_covariance=np.einsum("i,ijk->jk", value_weights ** 2, gyro_covariances),
            angular_acceleration_covariance=np.einsum("i,ijk->jk", derivative_weights ** 2, gyro_covariances),
            specific_force_covariance=np.einsum("i,ijk->jk", value_weights ** 2, force_covariances),
            stamp=stamp,
        )

    def update(self, frame_id, stamp, angular_velocity, specific_force,
               angular_velocity_covariance, specific_force_covariance):
        selector, raw_frame = self._select(frame_id)
        if selector is None:
            return None
        stamp = float(stamp)
        if not np.isfinite(stamp) or stamp < 0.0:
            raise ValueError("Acquisition timestamps must be finite and non-negative.")
        buffer = self.samples[selector]
        if buffer and stamp <= buffer[-1][0]:
            raise ValueError("IMU {} acquisition timestamps must be strictly increasing.".format(selector))
        sample = (
            stamp,
            _vector(angular_velocity, "angular_velocity").copy(),
            _vector(specific_force, "specific_force").copy(),
            _covariance(angular_velocity_covariance, "angular_velocity_covariance").copy(),
            _covariance(specific_force_covariance, "specific_force_covariance").copy(),
        )
        buffer.append(sample)
        self.bound_frames[selector] = raw_frame
        if any(len(self.samples[frame]) < self.window_size for frame in self.frames):
            return None
        latest_inputs = tuple(self.samples[frame][-1][0] for frame in self.frames)
        if self.last_input_stamps is not None and any(
            latest <= previous for latest, previous in zip(latest_inputs, self.last_input_stamps)
        ):
            return None
        lower = max(self.samples[frame][0][0] for frame in self.frames)
        upper = min(self.samples[frame][-1][0] for frame in self.frames)
        if lower > upper or (self.last_frontier is not None and upper <= self.last_frontier):
            return None
        common_stamp = lower + (upper - lower) * 0.5
        if self.last_stamp is not None and common_stamp <= self.last_stamp:
            return None
        pair = tuple(
            self._evaluate(frame, tuple(self.samples[frame]), common_stamp, self.polynomial_order)
            for frame in self.frames
        )
        self.last_frontier = upper
        self.last_input_stamps = latest_inputs
        self.last_stamp = common_stamp
        self.last_support = {
            frame: (self.bound_frames[frame], self.samples[frame][0][0], self.samples[frame][-1][0])
            for frame in self.frames
        }
        return pair


class ImuRelativePosePipeline:
    """Feed one multiplexed raw IMU stream; receive evolving ProbTF records."""

    def __init__(self, parent_frame_id, child_frame_id, window_size=11,
                 polynomial_order=7, **estimator_options):
        self.preprocessor = SynchronizedImuPairPreprocessor(
            parent_frame_id, child_frame_id, window_size, polynomial_order
        )
        self.estimator = ExtendedLeastSquaresImuRelativePoseEstimator(
            parent_frame_id, child_frame_id, **estimator_options
        )
        self.last_pair = None

    def reset(self):
        self.preprocessor.reset()
        self.estimator.reset()
        self.last_pair = None

    def update(self, frame_id, stamp, angular_velocity, specific_force,
               angular_velocity_covariance, specific_force_covariance):
        pair = self.preprocessor.update(
            frame_id, stamp, angular_velocity, specific_force,
            angular_velocity_covariance, specific_force_covariance,
        )
        if pair is None:
            return None
        record = self.estimator.update(*pair)
        self.last_pair = pair
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
