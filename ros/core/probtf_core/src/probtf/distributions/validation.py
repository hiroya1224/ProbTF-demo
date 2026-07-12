"""Validation primitives shared by immutable Prob-TF distribution objects."""

import numpy as np


MATRIX_TOLERANCE = 1e-10
PSD_TOLERANCE = 1e-10
QUATERNION_TOLERANCE = 1e-8


class DistributionValidationError(ValueError):
    pass


def immutable_array(values, shape, name):
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise DistributionValidationError("{} must have shape {}.".format(name, shape))
    if not np.all(np.isfinite(array)):
        raise DistributionValidationError("{} must contain only finite values.".format(name))
    output = array.copy()
    output.setflags(write=False)
    return output


def immutable_symmetric_matrix(values, size, name, positive_semidefinite=False):
    matrix = immutable_array(values, (size, size), name)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=MATRIX_TOLERANCE):
        raise DistributionValidationError("{} must be symmetric.".format(name))
    output = 0.5 * (matrix + matrix.T)
    if positive_semidefinite and float(np.linalg.eigvalsh(output)[0]) < -PSD_TOLERANCE:
        raise DistributionValidationError("{} must be positive semidefinite.".format(name))
    output.setflags(write=False)
    return output


def immutable_unit_quaternion(values, name="quaternion"):
    quaternion = immutable_array(values, (4,), name)
    norm = float(np.linalg.norm(quaternion))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=QUATERNION_TOLERANCE):
        raise DistributionValidationError("{} must have unit norm.".format(name))
    output = quaternion / norm
    output.setflags(write=False)
    return output


def identifier(value, name):
    result = str(value).strip()
    if not result:
        raise DistributionValidationError("{} must not be empty.".format(name))
    return result


def frame_id(value, name):
    result = identifier(value, name)
    return result[1:] if result.startswith("/") else result

