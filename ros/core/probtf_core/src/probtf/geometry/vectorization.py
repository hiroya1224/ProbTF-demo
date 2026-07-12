"""Fixed matrix packing and column-major ``vec(R)`` conventions."""

import numpy as np

from probtf.geometry.quaternion import quat_to_rotmat
from probtf.geometry.rotation import skew


SYMMETRIC_4_UPPER_INDICES = (
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
    (2, 3),
    (3, 3),
)
SYMMETRIC_3_UPPER_INDICES = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 1),
    (1, 2),
    (2, 2),
)


def vectorize_rotation(rotation):
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix.")
    return matrix.reshape(9, order="F").copy()


def rotation_vector_from_quaternion(quat_wxyz):
    return vectorize_rotation(quat_to_rotmat(quat_wxyz))


def rotation_action_matrix(vector):
    """Return ``H`` such that ``H @ vec(R) == R @ vector``."""

    value = np.asarray(vector, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("vector must be a finite vector with shape (3,).")
    output = np.zeros((3, 9), dtype=float)
    for row in range(3):
        for column in range(3):
            output[row, row + 3 * column] = value[column]
    return output


def pack_symmetric_upper(matrix):
    value = np.asarray(matrix, dtype=float)
    if value.shape == (4, 4):
        indices = SYMMETRIC_4_UPPER_INDICES
    elif value.shape == (3, 3):
        indices = SYMMETRIC_3_UPPER_INDICES
    else:
        raise ValueError("matrix must have shape (3, 3) or (4, 4).")
    if not np.all(np.isfinite(value)) or not np.allclose(value, value.T, atol=1e-10):
        raise ValueError("matrix must be finite and symmetric.")
    return np.array([value[row, column] for row, column in indices], dtype=float)


def unpack_symmetric_upper(values, size):
    if size == 4:
        indices = SYMMETRIC_4_UPPER_INDICES
    elif size == 3:
        indices = SYMMETRIC_3_UPPER_INDICES
    else:
        raise ValueError("size must be 3 or 4.")
    packed = np.asarray(values, dtype=float)
    if packed.shape != (len(indices),) or not np.all(np.isfinite(packed)):
        raise ValueError("packed symmetric matrix has an invalid shape or value.")
    matrix = np.zeros((size, size), dtype=float)
    for value, (row, column) in zip(packed, indices):
        matrix[row, column] = value
        matrix[column, row] = value
    return matrix


def right_perturbation_vec_rotation_jacobian(reference_quaternion_wxyz):
    """Return the column-major right-perturbation Jacobian ``D`` (9 x 3).

    For ``R(u) = R_ref exp([u]x)``, column ``i`` is
    ``vec(R_ref [e_i]x)``.  Its Moore-Penrose inverse is ``D.T / 2``.
    """

    rotation = quat_to_rotmat(reference_quaternion_wxyz)
    basis = np.eye(3, dtype=float)
    return np.column_stack(
        [vectorize_rotation(rotation @ skew(basis[:, index])) for index in range(3)]
    )

