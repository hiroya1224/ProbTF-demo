"""Deterministic rigid transforms with TF-compatible direction semantics."""

from dataclasses import dataclass

import numpy as np

from probtf.geometry.quaternion import quat_conj, quat_mul, quat_normalize, quat_to_rotmat


def _immutable_vector(values, size, name):
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError("{} must be a finite vector with shape ({},).".format(name, size))
    output = vector.copy()
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class DeterministicTransform:
    """Map child/source coordinates to parent/target coordinates.

    ``apply(z)`` evaluates ``R(q) z + translation``.  Quaternion storage is
    ``wxyz``.
    """

    translation: np.ndarray
    rotation_wxyz: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "translation", _immutable_vector(self.translation, 3, "translation"))
        quaternion = quat_normalize(self.rotation_wxyz)
        quaternion.setflags(write=False)
        object.__setattr__(self, "rotation_wxyz", quaternion)

    @classmethod
    def identity(cls):
        return cls(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    def apply(self, point):
        value = np.asarray(point, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("point must be a finite vector with shape (3,).")
        return quat_to_rotmat(self.rotation_wxyz) @ value + self.translation

    def inverse(self):
        inverse_quaternion = quat_conj(self.rotation_wxyz)
        inverse_rotation = quat_to_rotmat(inverse_quaternion)
        return DeterministicTransform(-inverse_rotation @ self.translation, inverse_quaternion)

    def then(self, next_transform):
        """Compose transforms in application order: ``next(self(point))``."""

        if not isinstance(next_transform, DeterministicTransform):
            raise TypeError("next_transform must be a DeterministicTransform.")
        next_rotation = quat_to_rotmat(next_transform.rotation_wxyz)
        return DeterministicTransform(
            next_rotation @ self.translation + next_transform.translation,
            quat_mul(next_transform.rotation_wxyz, self.rotation_wxyz),
        )

