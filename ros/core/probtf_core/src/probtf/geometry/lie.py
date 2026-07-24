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


_SMALL_ANGLE = 1.0e-8


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


__all__ = [
    "body_twist_between",
    "compose_transforms",
    "interpolate_transform",
    "quaternion_to_rotation_vector",
    "relative_transform",
    "rotation_vector_to_quaternion",
    "se3_exp",
    "se3_log",
]
