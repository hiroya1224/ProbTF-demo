"""Offline body angular velocity/acceleration from irregular-clock mocap.

The local polynomial is fitted to sign-continuous quaternion components.
Both derivatives of the quaternion normalization are included.  This is a
centered, finite-window SG approximation, not a derivative of Euler angles
and not a causal estimate for the held-out sparse-gyro predictor.
"""
from __future__ import annotations

import numpy as np


def _timestamps(values, name):
    time = np.asarray(values, dtype=float)
    if (time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0.0)):
        raise ValueError(name + " must be finite and strictly increasing")
    return time


def continuous_quaternions(quaternions_xyzw):
    """Normalize and choose continuous representatives of the same rotations."""
    quaternion = np.asarray(quaternions_xyzw, dtype=float).copy()
    if (quaternion.ndim != 2 or quaternion.shape[1] != 4
            or not np.isfinite(quaternion).all()):
        raise ValueError("quaternions must be finite Nx4 xyzw")
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("zero quaternion")
    quaternion /= norms[:, None]
    for index in range(1, len(quaternion)):
        if quaternion[index] @ quaternion[index - 1] < 0.0:
            quaternion[index] *= -1.0
    return quaternion


def differentiate_quaternions(time, quaternions_xyzw, query_times,
                              window_seconds=.32, degree=3):
    """Return body omega and dot-omega from a centered quaternion SG fit.

    Quaternion convention is xyzw, Hamilton product, body-to-world rotation.
    Thus omega = 2 vec(conj(q) * qdot).  The vector part of the derivative is
    alpha = 2 vec(conj(q) * qddot), since conj(qdot) * qdot is real.

    Source timestamps are used directly; no uniform resampling, time shift,
    gyro fit or parameter-model fit is performed here.  Rank and condition
    diagnostics are recorded; finite ill-conditioning is not a rejection
    criterion.  Full centered-window source support is required explicitly.
    """
    stamps = _timestamps(time, "time")
    quaternion = continuous_quaternions(quaternions_xyzw)
    query = np.atleast_1d(np.asarray(query_times, dtype=float))
    width = float(window_seconds)
    order = int(degree)
    if (len(quaternion) != len(stamps) or query.ndim != 1
            or not np.isfinite(query).all() or not np.isfinite(width)
            or width <= 0.0 or order != degree or order < 2):
        raise ValueError("invalid second-derivative local-polynomial inputs")
    half = width / 2.0
    if np.any(query - half < stamps[0]) or np.any(query + half > stamps[-1]):
        raise ValueError("queries must have full centered-window source support")

    fitted = np.empty((len(query), 4))
    derivative = np.empty_like(fitted)
    second_derivative = np.empty_like(fitted)
    counts = np.empty(len(query), dtype=int)
    residual = np.empty(len(query))
    condition = np.empty(len(query))
    ranks = np.empty(len(query), dtype=int)
    first = np.empty(len(query), dtype=int)
    last = np.empty(len(query), dtype=int)
    for index, selected in enumerate(query):
        left = np.searchsorted(stamps, selected - half, side="left")
        right = np.searchsorted(stamps, selected + half, side="right")
        offsets = (stamps[left:right] - selected) / half
        if len(offsets) < order + 1:
            raise ValueError("centered window has too few source measurements")
        design = np.polynomial.polynomial.polyvander(offsets, order)
        coefficients, _, rank, singular = np.linalg.lstsq(
            design, quaternion[left:right], rcond=None)
        raw = coefficients[0]
        raw_dot = coefficients[1] / half
        raw_ddot = 2.0 * coefficients[2] / half**2
        radius = np.linalg.norm(raw)
        if not np.isfinite(radius) or radius == 0.0:
            raise FloatingPointError("SG quaternion fit has zero/nonfinite norm")
        radius_dot = (raw @ raw_dot) / radius
        radius_ddot = (raw_dot @ raw_dot + raw @ raw_ddot
                       - radius_dot**2) / radius
        q = raw / radius
        q_dot = (raw_dot - q * radius_dot) / radius
        q_ddot = (raw_ddot - 2.0 * q_dot * radius_dot
                  - q * radius_ddot) / radius
        fitted[index] = q
        derivative[index] = q_dot
        second_derivative[index] = q_ddot
        counts[index] = len(offsets)
        residual[index] = np.sqrt(np.mean(
            (design @ coefficients - quaternion[left:right])**2))
        condition[index] = singular[0] / singular[-1]
        ranks[index] = rank
        first[index], last[index] = left, right - 1

    vector, scalar = fitted[:, :3], fitted[:, 3:]
    vector_dot, scalar_dot = derivative[:, :3], derivative[:, 3:]
    vector_ddot, scalar_ddot = second_derivative[:, :3], second_derivative[:, 3:]
    omega = 2.0 * (scalar * vector_dot - scalar_dot * vector
                   - np.cross(vector, vector_dot))
    alpha = 2.0 * (scalar * vector_ddot - scalar_ddot * vector
                   - np.cross(vector, vector_ddot))
    if not all(np.isfinite(value).all() for value in
               (fitted, derivative, second_derivative, omega, alpha)):
        raise FloatingPointError("nonfinite SG quaternion derivatives")
    return dict(
        time=query.copy(), quaternion_xyzw=fitted,
        quaternion_derivative_xyzw=derivative,
        quaternion_second_derivative_xyzw=second_derivative,
        omega_body_rad_s=omega, alpha_body_rad_s2=alpha,
        source_points_per_fit=counts, quaternion_fit_rms=residual,
        design_condition=condition, design_rank=ranks,
        source_fit_first_row=first, source_fit_last_row=last,
        source_fit_first_time=stamps[first], source_fit_last_time=stamps[last],
        quaternion_tangency_residual=np.einsum("ni,ni->n", fitted, derivative),
        quaternion_second_normalization_residual=(
            np.einsum("ni,ni->n", fitted, second_derivative)
            + np.einsum("ni,ni->n", derivative, derivative)),
        window_seconds=width, degree=order,
        method=("centered irregular-clock quaternion-component SG; first and "
                "second derivatives of normalization; body omega and alpha "
                "from conjugate(q) Hamilton quaternion derivatives"),
        quaternion_order="xyzw", angular_velocity_frame="body",
        angular_acceleration_frame="body", causal_filter=False,
        uniform_time_resampling=False, fitted_time_shift_seconds=0.0)


def prepare_kinematics(data, query_times, window_seconds=.32, degree=3):
    """Convenience adapter for the archived mocap/gyro SG input dictionary."""
    result = differentiate_quaternions(
        data["mocap_time"], data["mocap_quaternion_xyzw"], query_times,
        window_seconds=window_seconds, degree=degree)
    for source_key in ("mocap_source_index", "mocap_stamp_ns", "mocap_record_ns"):
        if source_key in data:
            values = np.asarray(data[source_key])
            if len(values) != len(data["mocap_time"]):
                raise ValueError(source_key + " has inconsistent source length")
            result["source_fit_first_" + source_key] = values[result["source_fit_first_row"]]
            result["source_fit_last_" + source_key] = values[result["source_fit_last_row"]]
    return result
