"""Whitened Euclidean and SO(3) Gaussian prior factors."""

import numpy as np

from grape_param_estim.batch.factor import FactorEvaluation, JacobianBlock
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.geometry import (
    so3_log,
    so3_right_jacobian_inverse,
)


_LOG_BRANCH_WARNING_DISTANCE = 1.0e-5


def _square_whitening(value: np.ndarray, dimension: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (dimension, dimension) or not np.all(
        np.isfinite(result)
    ):
        raise ValueError(
            "square_root_information must be a finite {} by {} matrix".format(
                dimension,
                dimension,
            )
        )
    if np.linalg.matrix_rank(result) != dimension:
        raise ValueError("square_root_information must have full rank")
    return result


def _proper_rotation(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite 3 by 3 matrix".format(name))
    if not np.allclose(result.T @ result, np.eye(3), rtol=0.0, atol=1.0e-9):
        raise ValueError("{} must be orthonormal".format(name))
    if not np.isclose(np.linalg.det(result), 1.0, rtol=0.0, atol=1.0e-9):
        raise ValueError("{} must have determinant one".format(name))
    return result


def evaluate_vector_prior_factor(
    variable_key: VariableKey,
    value: np.ndarray,
    mean: np.ndarray,
    square_root_information: np.ndarray,
) -> FactorEvaluation:
    """Evaluate a Gaussian prior for one Euclidean variable block."""

    if not isinstance(variable_key, VariableKey):
        raise TypeError("variable_key must be a VariableKey")
    if variable_key.kind is VariableKind.ORIENTATION_TANGENT:
        raise ValueError("orientation priors must use the SO(3) prior factor")
    dimension = variable_key.dimension
    current = np.asarray(value, dtype=float)
    expected = np.asarray(mean, dtype=float)
    if (
        current.shape != (dimension,)
        or expected.shape != (dimension,)
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(expected))
    ):
        raise ValueError(
            "value and mean must contain {} finite values".format(dimension)
        )
    whitening = _square_whitening(square_root_information, dimension)
    residual = whitening @ (current - expected)
    return FactorEvaluation(
        residual=residual,
        jacobian_blocks=(JacobianBlock(variable_key, whitening),),
        squared_error=float(residual @ residual),
        active_set={},
    )


def evaluate_orientation_prior_factor(
    variable_key: VariableKey,
    rotation: np.ndarray,
    mean_rotation: np.ndarray,
    square_root_information: np.ndarray,
) -> FactorEvaluation:
    """Evaluate a right-tangent Gaussian prior on one SO(3) state."""

    if not isinstance(variable_key, VariableKey):
        raise TypeError("variable_key must be a VariableKey")
    if variable_key.kind is not VariableKind.ORIENTATION_TANGENT:
        raise ValueError("SO(3) prior requires an orientation variable key")
    current = _proper_rotation(rotation, "rotation")
    expected = _proper_rotation(mean_rotation, "mean_rotation")
    whitening = _square_whitening(square_root_information, 3)
    raw_residual = so3_log(expected.T @ current)
    residual = whitening @ raw_residual
    jacobian = whitening @ so3_right_jacobian_inverse(raw_residual)
    distance_to_branch = np.pi - float(np.linalg.norm(raw_residual))
    return FactorEvaluation(
        residual=residual,
        jacobian_blocks=(JacobianBlock(variable_key, jacobian),),
        squared_error=float(residual @ residual),
        active_set={
            "rotation_log_near_pi": np.asarray(
                (distance_to_branch <= _LOG_BRANCH_WARNING_DISTANCE,),
                dtype=bool,
            )
        },
    )


__all__ = [
    "evaluate_orientation_prior_factor",
    "evaluate_vector_prior_factor",
]
