"""Numerically stable :math:`SO(3)`/:math:`SE(3)` exponential coordinates.

The transform convention matches :class:`DeterministicTransform`: a pose maps
child coordinates into parent coordinates.  A body increment is composed on
the right,

``T(t + dt) = T(t) Exp(xi_body * dt)``.

Twists use ``[rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]`` ordering.  ``rho``
is the translational part of the SE(3) logarithm, not an independently
extrapolated Cartesian velocity.
"""

import math

import numpy as np

from probtf.geometry.quaternion import quat_conj, quat_mul, quat_normalize, quat_to_rotmat
from probtf.geometry.rotation import skew
from probtf.geometry.transform import DeterministicTransform


_SMALL_ANGLE = 1.0e-6


def rotation_vector_to_quaternion(rotation_vector):
    """Return the ``wxyz`` quaternion exponential of a rotation vector."""

    vector = np.asarray(rotation_vector, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation_vector must be a finite vector with shape (3,).")
    angle = float(np.linalg.norm(vector))
    if angle < _SMALL_ANGLE:
        # sin(theta / 2) / theta, expanded through theta**4.
        scale = 0.5 - angle * angle / 48.0 + angle ** 4 / 3840.0
        return quat_normalize(
            np.concatenate(([1.0 - angle * angle / 8.0], scale * vector))
        )
    half = 0.5 * angle
    return np.concatenate(([math.cos(half)], math.sin(half) * vector / angle))


def quaternion_to_rotation_vector(quaternion_wxyz):
    """Return the shortest deterministic rotation vector for a quaternion."""

    quaternion = quat_normalize(quaternion_wxyz)
    # q and -q describe the same rotation.  Canonicalization keeps the result
    # continuous away from pi and deterministic at pi.
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    elif abs(quaternion[0]) < 1.0e-15:
        pivot = 1 + int(np.argmax(np.abs(quaternion[1:])))
        if quaternion[pivot] < 0.0:
            quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[1:]))
    if vector_norm < _SMALL_ANGLE:
        return 2.0 * quaternion[1:]
    angle = 2.0 * math.atan2(vector_norm, max(0.0, float(quaternion[0])))
    return angle * quaternion[1:] / vector_norm


def _left_jacobian_so3(rotation_vector):
    vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    omega = skew(vector)
    if angle < _SMALL_ANGLE:
        return np.eye(3) + 0.5 * omega + (1.0 / 6.0) * (omega @ omega)
    angle2 = angle * angle
    return (
        np.eye(3)
        + ((1.0 - math.cos(angle)) / angle2) * omega
        + ((angle - math.sin(angle)) / (angle2 * angle)) * (omega @ omega)
    )


def _left_jacobian_so3_inverse(rotation_vector):
    vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    omega = skew(vector)
    if angle < _SMALL_ANGLE:
        return np.eye(3) - 0.5 * omega + (1.0 / 12.0) * (omega @ omega)
    half = 0.5 * angle
    cot_half = math.cos(half) / math.sin(half)
    coefficient = (1.0 - half * cot_half) / (angle * angle)
    return np.eye(3) - 0.5 * omega + coefficient * (omega @ omega)


def se3_exp(twist):
    """Map one body exponential-coordinate increment to a transform."""

    value = np.asarray(twist, dtype=float)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("twist must be a finite vector with shape (6,).")
    rotation_vector = value[3:]
    translation = _left_jacobian_so3(rotation_vector) @ value[:3]
    return DeterministicTransform(
        translation,
        rotation_vector_to_quaternion(rotation_vector),
    )


def se3_log(transform):
    """Return body exponential coordinates for a deterministic transform."""

    if not isinstance(transform, DeterministicTransform):
        raise TypeError("transform must be a DeterministicTransform.")
    rotation_vector = quaternion_to_rotation_vector(transform.rotation_wxyz)
    translation_vector = _left_jacobian_so3_inverse(rotation_vector) @ transform.translation
    return np.concatenate((translation_vector, rotation_vector))


def compose_transforms(left, right):
    """Return the matrix product ``left * right``."""

    if not isinstance(left, DeterministicTransform) or not isinstance(
        right, DeterministicTransform
    ):
        raise TypeError("left and right must be DeterministicTransform objects.")
    return DeterministicTransform(
        left.translation + quat_to_rotmat(left.rotation_wxyz) @ right.translation,
        quat_mul(left.rotation_wxyz, right.rotation_wxyz),
    )


def relative_transform(left, right):
    """Return ``inverse(left) * right``."""

    if not isinstance(left, DeterministicTransform) or not isinstance(
        right, DeterministicTransform
    ):
        raise TypeError("left and right must be DeterministicTransform objects.")
    inverse_rotation = quat_conj(left.rotation_wxyz)
    inverse_matrix = quat_to_rotmat(inverse_rotation)
    return DeterministicTransform(
        inverse_matrix @ (right.translation - left.translation),
        quat_mul(inverse_rotation, right.rotation_wxyz),
    )


def interpolate_transform(left, right, fraction):
    """Interpolate on SE(3), matching both endpoints exactly."""

    alpha = float(fraction)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("fraction must be finite and within [0, 1].")
    if alpha == 0.0:
        return left
    if alpha == 1.0:
        return right
    relative = relative_transform(left, right)
    return compose_transforms(left, se3_exp(alpha * se3_log(relative)))


def body_twist_between(left, right, duration):
    """Return the constant body twist taking ``left`` to ``right``."""

    elapsed = float(duration)
    if not np.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError("duration must be finite and positive.")
    return se3_log(relative_transform(left, right)) / elapsed


def integrate_linear_body_twist(
    endpoint_body_twist,
    body_acceleration,
    duration,
    *,
    substeps=64,
):
    """Integrate ``T_dot = T hat(xi_0 + a t)`` with midpoint Lie steps.

    A single ``Exp(xi_0 h + a h**2 / 2)`` is only exact when the twists
    commute.  This deterministic reference integrator preserves the
    time-ordering needed by a body-frame constant-acceleration model.
    """

    twist = np.asarray(endpoint_body_twist, dtype=float)
    acceleration = np.asarray(body_acceleration, dtype=float)
    elapsed = float(duration)
    count = int(substeps)
    if (
        twist.shape != (6,)
        or acceleration.shape != (6,)
        or not np.all(np.isfinite(twist))
        or not np.all(np.isfinite(acceleration))
    ):
        raise ValueError(
            "endpoint_body_twist and body_acceleration must be finite vectors "
            "with shape (6,)."
        )
    if not np.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("duration must be finite and non-negative.")
    if count < 1 or count != substeps:
        raise ValueError("substeps must be a positive integer.")
    if elapsed == 0.0:
        return DeterministicTransform.identity()
    if np.max(np.abs(acceleration)) == 0.0:
        return se3_exp(twist * elapsed)

    step = elapsed / count
    result = DeterministicTransform.identity()
    for index in range(count):
        midpoint = (index + 0.5) * step
        result = compose_transforms(
            result,
            se3_exp((twist + acceleration * midpoint) * step),
        )
    return result


def infer_endpoint_body_twist(
    left,
    right,
    duration,
    body_acceleration,
    *,
    substeps=64,
    maximum_iterations=12,
    tolerance=1.0e-10,
):
    """Recover the final body twist of a constant-acceleration pose segment.

    The logarithm of the endpoint pose difference is not the arithmetic mean
    twist when velocity and acceleration do not commute.  This shooting solve
    uses the same time-ordered integrator as prediction and fails closed if
    the requested segment is outside its numerical support.
    """

    if not isinstance(left, DeterministicTransform) or not isinstance(
        right, DeterministicTransform
    ):
        raise TypeError("left and right must be DeterministicTransform objects.")
    elapsed = float(duration)
    acceleration = np.asarray(body_acceleration, dtype=float)
    iterations = int(maximum_iterations)
    if not np.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError("duration must be finite and positive.")
    if acceleration.shape != (6,) or not np.all(np.isfinite(acceleration)):
        raise ValueError("body_acceleration must be a finite vector with shape (6,).")
    if iterations < 1 or iterations != maximum_iterations:
        raise ValueError("maximum_iterations must be a positive integer.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    average = body_twist_between(left, right, elapsed)
    if np.max(np.abs(acceleration)) == 0.0:
        return average

    target = relative_transform(left, right)
    start_twist = average - 0.5 * acceleration * elapsed

    def residual(value):
        integrated = integrate_linear_body_twist(
            value,
            acceleration,
            elapsed,
            substeps=substeps,
        )
        return se3_log(relative_transform(integrated, target))

    epsilon = 1.0e-6
    error = residual(start_twist)
    for _ in range(iterations):
        if float(np.linalg.norm(error)) <= tolerance:
            break
        jacobian = np.empty((6, 6), dtype=float)
        for column in range(6):
            delta = np.zeros(6, dtype=float)
            delta[column] = epsilon
            jacobian[:, column] = (
                residual(start_twist + delta) - residual(start_twist - delta)
            ) / (2.0 * epsilon)
        update, _, _, _ = np.linalg.lstsq(jacobian, -error, rcond=1.0e-12)
        start_twist = start_twist + update
        error = residual(start_twist)
    if float(np.linalg.norm(error)) > max(1.0e-8, 100.0 * tolerance):
        raise ValueError(
            "Constant-acceleration endpoint-twist solve did not converge."
        )
    return start_twist + acceleration * elapsed


__all__ = [
    "body_twist_between",
    "compose_transforms",
    "integrate_linear_body_twist",
    "infer_endpoint_body_twist",
    "interpolate_transform",
    "quaternion_to_rotation_vector",
    "relative_transform",
    "rotation_vector_to_quaternion",
    "se3_exp",
    "se3_log",
]
