"""Offline kinematic derivatives from timestamped motion-capture poses.

The estimator is intentionally ROS independent.  Input quaternions use the
``geometry_msgs`` field order ``[x, y, z, w]`` and describe the body frame in
the world frame.  Positions and quaternion components are smoothed on a
uniform time grid with a Savitzky--Golay polynomial before derivatives are
formed.  Angular velocity comes from body-frame increments of the smoothed
rotation, rather than from component-wise quaternion differentiation.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.spatial.transform import Rotation, Slerp


def _immutable_array(values: np.ndarray, shape_suffix: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim < len(shape_suffix) or array.shape[-len(shape_suffix) :] != shape_suffix:
        raise ValueError("{} has an unexpected shape".format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    output = np.array(array, dtype=float, copy=True)
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class KinematicsConfig:
    """Configuration for :func:`estimate_kinematics`.

    ``position_sigma`` is in meters and ``orientation_sigma`` is a small-angle
    tangent standard deviation in radians.  Both describe one raw mocap sample
    and are propagated through the linear Savitzky--Golay derivative
    coefficients to provide a useful first-order noise scale.
    """

    window_length: int = 15
    polynomial_order: int = 3
    position_sigma: float = 0.01
    orientation_sigma: float = float(np.deg2rad(1.0))
    gravity_world: Tuple[float, float, float] = (0.0, 0.0, -9.80665)

    def __post_init__(self) -> None:
        window = int(self.window_length)
        order = int(self.polynomial_order)
        if window != self.window_length or window < 5 or window % 2 != 1:
            raise ValueError("window_length must be an odd integer of at least 5")
        if order != self.polynomial_order or order < 2 or order >= window:
            raise ValueError(
                "polynomial_order must be an integer in [2, window_length)"
            )
        position_sigma = float(self.position_sigma)
        orientation_sigma = float(self.orientation_sigma)
        if not np.isfinite(position_sigma) or position_sigma < 0.0:
            raise ValueError("position_sigma must be finite and non-negative")
        if not np.isfinite(orientation_sigma) or orientation_sigma < 0.0:
            raise ValueError("orientation_sigma must be finite and non-negative")
        gravity = np.asarray(self.gravity_world, dtype=float)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity_world must be a finite three-vector")
        object.__setattr__(self, "window_length", window)
        object.__setattr__(self, "polynomial_order", order)
        object.__setattr__(self, "position_sigma", position_sigma)
        object.__setattr__(self, "orientation_sigma", orientation_sigma)
        object.__setattr__(self, "gravity_world", tuple(float(value) for value in gravity))


@dataclass(frozen=True)
class DerivativeNoiseEstimate:
    """Isotropic derivative-noise approximation induced by mocap noise."""

    linear_acceleration_std: float
    angular_velocity_std: float
    angular_acceleration_std: float

    def __post_init__(self) -> None:
        for name in (
            "linear_acceleration_std",
            "angular_velocity_std",
            "angular_acceleration_std",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("{} must be finite and non-negative".format(name))
            object.__setattr__(self, name, value)

    @staticmethod
    def _isotropic_covariance(standard_deviation: float) -> np.ndarray:
        covariance = np.eye(3, dtype=float) * float(standard_deviation) ** 2
        covariance.setflags(write=False)
        return covariance

    @property
    def linear_acceleration_covariance(self) -> np.ndarray:
        return self._isotropic_covariance(self.linear_acceleration_std)

    @property
    def specific_acceleration_covariance(self) -> np.ndarray:
        # An orthogonal frame change preserves isotropic covariance.
        return self.linear_acceleration_covariance

    @property
    def angular_velocity_covariance(self) -> np.ndarray:
        return self._isotropic_covariance(self.angular_velocity_std)

    @property
    def angular_acceleration_covariance(self) -> np.ndarray:
        return self._isotropic_covariance(self.angular_acceleration_std)


@dataclass(frozen=True)
class KinematicsEstimate:
    """Smoothed pose derivatives aligned with the original timestamps.

    Values outside ``valid_mask`` are set to NaN.  This makes accidental use
    of the Savitzky--Golay edge extrapolation fail loudly while retaining a
    same-length result for rosbag annotation.
    """

    timestamps: np.ndarray
    position_world_smoothed: np.ndarray
    quaternion_xyzw_smoothed: np.ndarray
    linear_acceleration_world: np.ndarray
    angular_velocity_body: np.ndarray
    angular_acceleration_body: np.ndarray
    specific_acceleration_body: np.ndarray
    valid_mask: np.ndarray
    edge_margin: int
    noise: DerivativeNoiseEstimate

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=float)
        if timestamps.ndim != 1 or not np.all(np.isfinite(timestamps)):
            raise ValueError("timestamps must be a finite one-dimensional array")
        count = timestamps.size
        if count and not np.all(np.diff(timestamps) > 0.0):
            raise ValueError("timestamps must be strictly increasing")
        arrays = {
            "position_world_smoothed": self.position_world_smoothed,
            "quaternion_xyzw_smoothed": self.quaternion_xyzw_smoothed,
            "linear_acceleration_world": self.linear_acceleration_world,
            "angular_velocity_body": self.angular_velocity_body,
            "angular_acceleration_body": self.angular_acceleration_body,
            "specific_acceleration_body": self.specific_acceleration_body,
        }
        suffixes = {
            "position_world_smoothed": (count, 3),
            "quaternion_xyzw_smoothed": (count, 4),
            "linear_acceleration_world": (count, 3),
            "angular_velocity_body": (count, 3),
            "angular_acceleration_body": (count, 3),
            "specific_acceleration_body": (count, 3),
        }
        # Derived arrays deliberately contain NaNs at invalid edges, so only
        # shape is enforced here.  Valid samples are checked below.
        frozen = {}
        for name, values in arrays.items():
            array = np.asarray(values, dtype=float)
            if array.shape != suffixes[name]:
                raise ValueError("{} must have shape {}".format(name, suffixes[name]))
            copy = np.array(array, dtype=float, copy=True)
            copy.setflags(write=False)
            frozen[name] = copy
        valid = np.asarray(self.valid_mask, dtype=bool)
        if valid.shape != (count,):
            raise ValueError("valid_mask must have shape (N,)")
        if np.any(valid):
            for name, values in frozen.items():
                if not np.all(np.isfinite(values[valid])):
                    raise ValueError("{} contains non-finite valid samples".format(name))
        quaternion_norms = np.linalg.norm(frozen["quaternion_xyzw_smoothed"][valid], axis=1)
        if quaternion_norms.size and not np.allclose(
            quaternion_norms, 1.0, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError("valid smoothed quaternions must be unit length")
        edge_margin = int(self.edge_margin)
        if edge_margin < 0:
            raise ValueError("edge_margin must be non-negative")
        if not isinstance(self.noise, DerivativeNoiseEstimate):
            raise TypeError("noise must be DerivativeNoiseEstimate")

        timestamp_copy = np.array(timestamps, copy=True)
        timestamp_copy.setflags(write=False)
        valid_copy = np.array(valid, copy=True)
        valid_copy.setflags(write=False)
        object.__setattr__(self, "timestamps", timestamp_copy)
        object.__setattr__(self, "valid_mask", valid_copy)
        object.__setattr__(self, "edge_margin", edge_margin)
        for name, values in frozen.items():
            object.__setattr__(self, name, values)

    @property
    def acceleration_world(self) -> np.ndarray:
        """Short alias for ``linear_acceleration_world``."""

        return self.linear_acceleration_world

    @property
    def omega_body(self) -> np.ndarray:
        """Short alias for ``angular_velocity_body``."""

        return self.angular_velocity_body

    @property
    def alpha_body(self) -> np.ndarray:
        """Short alias for ``angular_acceleration_body``."""

        return self.angular_acceleration_body


def _timestamps(values: Sequence[float], minimum_count: int) -> np.ndarray:
    try:
        timestamps = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamps must contain numeric values") from exc
    if timestamps.ndim != 1 or timestamps.size < minimum_count:
        raise ValueError(
            "timestamps must be one-dimensional with at least {} samples".format(
                minimum_count
            )
        )
    if not np.all(np.isfinite(timestamps)) or not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("timestamps must be finite and strictly increasing")
    return timestamps


def _sample_matrix(values: np.ndarray, count: int, width: int, name: str) -> np.ndarray:
    try:
        samples = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must contain numeric values".format(name)) from exc
    if samples.shape != (count, width):
        raise ValueError("{} must have shape ({}, {})".format(name, count, width))
    if not np.all(np.isfinite(samples)):
        raise ValueError("{} must contain only finite values".format(name))
    return samples


def _continuous_unit_quaternions(quaternions_xyzw: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions_xyzw, axis=1)
    if np.any(norms <= 1.0e-12):
        raise ValueError("quaternions_xyzw must not contain zero quaternions")
    output = np.asarray(quaternions_xyzw, dtype=float) / norms[:, None]
    output = np.array(output, copy=True)
    for index in range(1, len(output)):
        if float(np.dot(output[index - 1], output[index])) < 0.0:
            output[index] *= -1.0
    return output


def _uniform_grid(timestamps: np.ndarray) -> Tuple[np.ndarray, float]:
    grid = np.linspace(float(timestamps[0]), float(timestamps[-1]), timestamps.size)
    delta = float(grid[1] - grid[0])
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("timestamps do not span a positive finite duration")
    return grid, delta


def _interpolate_columns(
    source_times: np.ndarray,
    values: np.ndarray,
    destination_times: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(destination_times, source_times, values[:, column])
            for column in range(values.shape[1])
        ]
    )


def _interpolate_smooth_vectors(
    source_times: np.ndarray,
    values: np.ndarray,
    destination_times: np.ndarray,
) -> np.ndarray:
    """Interpolate a smooth vector signal without piecewise-linear corners."""

    return np.asarray(
        CubicSpline(source_times, values, axis=0)(destination_times),
        dtype=float,
    )


def _body_rotation_derivatives(
    timestamps: np.ndarray,
    quaternions_xyzw: np.ndarray,
    window_length: int,
    polynomial_order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rotations = Rotation.from_quat(quaternions_xyzw)
    matrices = rotations.as_matrix()
    relative_matrices = np.einsum(
        "nij,njk->nik",
        np.swapaxes(matrices[:-1], 1, 2),
        matrices[1:],
    )
    interval_rotvec_body = Rotation.from_matrix(relative_matrices).as_rotvec()
    interval_dt = np.diff(timestamps)
    interval_omega_body = interval_rotvec_body / interval_dt[:, None]

    # R_WB maps body coordinates to world coordinates.  A relative rotation
    # vector is parallel to its own increment axis, so expressing it through
    # the interval-start rotation is also its midpoint world representation.
    interval_omega_world = np.einsum(
        "nij,nj->ni", matrices[:-1], interval_omega_body
    )
    midpoint_times = 0.5 * (timestamps[:-1] + timestamps[1:])
    omega_world_raw = _interpolate_columns(midpoint_times, interval_omega_world, timestamps)

    # A plain finite difference of noisy angular velocity amplifies the
    # 1-degree mocap tangent noise by roughly 1/dt a second time.  Differentiate
    # a centered local polynomial instead.  This is the rotational analogue of
    # the position SG derivative above and is essential to avoid an
    # errors-in-variables attenuation bias in the inertia estimate.
    uniform_times, uniform_delta = _uniform_grid(timestamps)
    omega_world_uniform = _interpolate_columns(
        timestamps, omega_world_raw, uniform_times
    )
    omega_world_uniform = savgol_filter(
        omega_world_uniform,
        window_length,
        polynomial_order,
        deriv=0,
        axis=0,
        mode="interp",
    )
    alpha_world_uniform = savgol_filter(
        omega_world_uniform,
        window_length,
        polynomial_order,
        deriv=1,
        delta=uniform_delta,
        axis=0,
        mode="interp",
    )
    omega_world = _interpolate_columns(uniform_times, omega_world_uniform, timestamps)
    alpha_world = _interpolate_columns(uniform_times, alpha_world_uniform, timestamps)
    omega_body = np.einsum("nji,nj->ni", matrices, omega_world)
    alpha_body = np.einsum("nji,nj->ni", matrices, alpha_world)
    return omega_body, alpha_body


def derivative_noise_estimate(
    sample_period: float,
    config: KinematicsConfig = KinematicsConfig(),
) -> DerivativeNoiseEstimate:
    """Propagate raw isotropic mocap noise through SG derivative weights.

    This is a first-order independent-sample approximation.  It intentionally
    does not claim to model temporal mocap correlation or quaternion
    renormalization; those effects should be calibrated from residual data.
    """

    delta = float(sample_period)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("sample_period must be finite and positive")
    if not isinstance(config, KinematicsConfig):
        raise TypeError("config must be KinematicsConfig")
    velocity_weights = savgol_coeffs(
        config.window_length,
        config.polynomial_order,
        deriv=1,
        delta=delta,
        use="dot",
    )
    acceleration_weights = savgol_coeffs(
        config.window_length,
        config.polynomial_order,
        deriv=2,
        delta=delta,
        use="dot",
    )
    return DerivativeNoiseEstimate(
        linear_acceleration_std=float(
            config.position_sigma * np.linalg.norm(acceleration_weights)
        ),
        angular_velocity_std=float(
            config.orientation_sigma * np.linalg.norm(velocity_weights)
        ),
        angular_acceleration_std=float(
            config.orientation_sigma * np.linalg.norm(acceleration_weights)
        ),
    )


def estimate_kinematics(
    timestamps: Sequence[float],
    positions_world: np.ndarray,
    quaternions_xyzw: np.ndarray,
    config: KinematicsConfig = KinematicsConfig(),
) -> KinematicsEstimate:
    """Estimate translational and body angular kinematics from mocap poses.

    Irregular timestamps are handled by interpolating the pose to a uniform
    grid before Savitzky--Golay smoothing, then evaluating the smoothed pose
    and translational derivative back at the original timestamps.  Rotation
    interpolation uses SLERP.  A margin of ``window_length // 2 + 2`` samples
    is marked invalid to exclude both SG edge extrapolation and the centered
    rotation/derivative stencil.
    """

    if not isinstance(config, KinematicsConfig):
        raise TypeError("config must be KinematicsConfig")
    times = _timestamps(timestamps, config.window_length)
    count = times.size
    positions = _sample_matrix(positions_world, count, 3, "positions_world")
    quaternions = _continuous_unit_quaternions(
        _sample_matrix(quaternions_xyzw, count, 4, "quaternions_xyzw")
    )

    uniform_times, uniform_delta = _uniform_grid(times)
    positions_uniform = _interpolate_smooth_vectors(times, positions, uniform_times)
    quaternions_uniform = Slerp(times, Rotation.from_quat(quaternions))(
        uniform_times
    ).as_quat()
    quaternions_uniform = _continuous_unit_quaternions(quaternions_uniform)

    smooth_positions_uniform = savgol_filter(
        positions_uniform,
        config.window_length,
        config.polynomial_order,
        deriv=0,
        axis=0,
        mode="interp",
    )
    acceleration_uniform = savgol_filter(
        positions_uniform,
        config.window_length,
        config.polynomial_order,
        deriv=2,
        delta=uniform_delta,
        axis=0,
        mode="interp",
    )
    smooth_quaternions_uniform = savgol_filter(
        quaternions_uniform,
        config.window_length,
        config.polynomial_order,
        deriv=0,
        axis=0,
        mode="interp",
    )
    smooth_quaternions_uniform = _continuous_unit_quaternions(
        smooth_quaternions_uniform
    )

    smooth_positions = _interpolate_smooth_vectors(
        uniform_times, smooth_positions_uniform, times
    )
    acceleration_world = _interpolate_smooth_vectors(
        uniform_times, acceleration_uniform, times
    )
    smooth_quaternions = Slerp(
        uniform_times,
        Rotation.from_quat(smooth_quaternions_uniform),
    )(times).as_quat()
    smooth_quaternions = _continuous_unit_quaternions(smooth_quaternions)

    omega_body, alpha_body = _body_rotation_derivatives(
        times,
        smooth_quaternions,
        config.window_length,
        config.polynomial_order,
    )
    matrices = Rotation.from_quat(smooth_quaternions).as_matrix()
    gravity = np.asarray(config.gravity_world, dtype=float)
    specific_world = acceleration_world - gravity[None, :]
    specific_body = np.einsum("nji,nj->ni", matrices, specific_world)

    edge_margin = config.window_length // 2 + 2
    valid = np.zeros(count, dtype=bool)
    if count > 2 * edge_margin:
        valid[edge_margin : count - edge_margin] = True
    if not np.any(valid):
        raise ValueError(
            "not enough samples remain after the derivative edge margin; "
            "provide more samples or a shorter window"
        )

    def mask_edges(values: np.ndarray) -> np.ndarray:
        output = np.array(values, dtype=float, copy=True)
        output[~valid] = np.nan
        return output

    return KinematicsEstimate(
        timestamps=times,
        position_world_smoothed=mask_edges(smooth_positions),
        quaternion_xyzw_smoothed=mask_edges(smooth_quaternions),
        linear_acceleration_world=mask_edges(acceleration_world),
        angular_velocity_body=mask_edges(omega_body),
        angular_acceleration_body=mask_edges(alpha_body),
        specific_acceleration_body=mask_edges(specific_body),
        valid_mask=valid,
        edge_margin=edge_margin,
        noise=derivative_noise_estimate(uniform_delta, config),
    )


__all__ = [
    "DerivativeNoiseEstimate",
    "KinematicsConfig",
    "KinematicsEstimate",
    "derivative_noise_estimate",
    "estimate_kinematics",
]
