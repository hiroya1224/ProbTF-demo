"""Rigid-body inertial parameterization and wrench prediction.

The public parameter vector is ordered as

``[mass, cog_x, cog_y, cog_z, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]``.

The center of gravity and the inertia tensor are both expressed in the body
(``fc``) frame.  The inertia entries describe :math:`J_C`, the inertia about
the center of mass, rather than the inertia about the body-frame origin.
Functions accept one vector with shape ``(10,)`` or a batch with shape
``(..., 10)``.  Kinematic inputs use ordinary NumPy broadcasting over those
leading dimensions.
"""

from typing import Tuple

import numpy as np


PARAMETER_NAMES: Tuple[str, ...] = (
    "mass",
    "cog_x",
    "cog_y",
    "cog_z",
    "inertia_xx",
    "inertia_xy",
    "inertia_xz",
    "inertia_yy",
    "inertia_yz",
    "inertia_zz",
)

PARAMETER_COUNT = len(PARAMETER_NAMES)
_DEFAULT_EIGENVALUE_TOLERANCE = 1.0e-12
_DEFAULT_TRIANGLE_TOLERANCE = 1.0e-12


def _parameter_array(parameters: np.ndarray) -> np.ndarray:
    """Return a finite floating-point parameter array with a checked shape."""

    try:
        values = np.asarray(parameters, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("parameters must contain numeric values") from exc
    if values.ndim < 1 or values.shape[-1] != PARAMETER_COUNT:
        raise ValueError(
            "parameters must have shape (10,) or (..., 10) in PARAMETER_NAMES order"
        )
    return values


def _finite_vector3(values: np.ndarray, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must contain numeric values".format(name)) from exc
    if vector.ndim < 1 or vector.shape[-1] != 3:
        raise ValueError("{} must have shape (3,) or (..., 3)".format(name))
    if not np.all(np.isfinite(vector)):
        raise ValueError("{} must contain only finite values".format(name))
    return vector


def parameters_to_inertia(parameters: np.ndarray) -> np.ndarray:
    """Construct the symmetric center-of-mass inertia tensor ``J_C``.

    Parameters
    ----------
    parameters:
        One inertial parameter vector with shape ``(10,)`` or a vectorized
        batch with shape ``(..., 10)``.

    Returns
    -------
    numpy.ndarray
        The symmetric inertia tensor(s), with shape ``(..., 3, 3)``.  This
        conversion checks shape and finiteness but deliberately does not
        require physical validity; use :func:`physical_parameter_mask` for
        particle rejection.
    """

    values = _parameter_array(parameters)
    if not np.all(np.isfinite(values)):
        raise ValueError("parameters must contain only finite values")
    inertia = np.empty(values.shape[:-1] + (3, 3), dtype=float)
    inertia[..., 0, 0] = values[..., 4]
    inertia[..., 0, 1] = values[..., 5]
    inertia[..., 0, 2] = values[..., 6]
    inertia[..., 1, 0] = values[..., 5]
    inertia[..., 1, 1] = values[..., 7]
    inertia[..., 1, 2] = values[..., 8]
    inertia[..., 2, 0] = values[..., 6]
    inertia[..., 2, 1] = values[..., 8]
    inertia[..., 2, 2] = values[..., 9]
    return inertia


def inertia_to_parameters(
    mass: np.ndarray,
    center_of_mass: np.ndarray,
    inertia_com: np.ndarray,
) -> np.ndarray:
    """Pack mass, center of mass, and ``J_C`` into ``PARAMETER_NAMES`` order.

    Leading dimensions follow NumPy broadcasting rules.  The inertia input
    must be symmetric; physical validity is checked separately by
    :func:`physical_parameter_mask` or :func:`validate_physical_parameters`.
    """

    try:
        mass_values = np.asarray(mass, dtype=float)
        center_values = np.asarray(center_of_mass, dtype=float)
        inertia_values = np.asarray(inertia_com, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("inertial properties must contain numeric values") from exc
    if center_values.ndim < 1 or center_values.shape[-1] != 3:
        raise ValueError("center_of_mass must have shape (3,) or (..., 3)")
    if inertia_values.ndim < 2 or inertia_values.shape[-2:] != (3, 3):
        raise ValueError("inertia_com must have shape (3, 3) or (..., 3, 3)")
    if not (
        np.all(np.isfinite(mass_values))
        and np.all(np.isfinite(center_values))
        and np.all(np.isfinite(inertia_values))
    ):
        raise ValueError("inertial properties must contain only finite values")
    if not np.allclose(inertia_values, np.swapaxes(inertia_values, -1, -2), rtol=0.0, atol=1e-12):
        raise ValueError("inertia_com must be symmetric")

    leading_shape = np.broadcast_shapes(
        mass_values.shape,
        center_values.shape[:-1],
        inertia_values.shape[:-2],
    )
    mass_values = np.broadcast_to(mass_values, leading_shape)
    center_values = np.broadcast_to(center_values, leading_shape + (3,))
    inertia_values = np.broadcast_to(inertia_values, leading_shape + (3, 3))
    output = np.empty(leading_shape + (PARAMETER_COUNT,), dtype=float)
    output[..., 0] = mass_values
    output[..., 1:4] = center_values
    output[..., 4] = inertia_values[..., 0, 0]
    output[..., 5] = inertia_values[..., 0, 1]
    output[..., 6] = inertia_values[..., 0, 2]
    output[..., 7] = inertia_values[..., 1, 1]
    output[..., 8] = inertia_values[..., 1, 2]
    output[..., 9] = inertia_values[..., 2, 2]
    return output


def physical_parameter_mask(
    parameters: np.ndarray,
    eigenvalue_tolerance: float = _DEFAULT_EIGENVALUE_TOLERANCE,
    triangle_tolerance: float = _DEFAULT_TRIANGLE_TOLERANCE,
) -> np.ndarray:
    """Return whether each inertial parameter vector is physically valid.

    A valid vector has finite entries, strictly positive mass, a positive
    definite center-of-mass inertia tensor, and strict principal-moment
    triangle inequalities.  Tolerances are relative to
    ``max(max(principal_moments), 1)`` and make boundary particles invalid.

    A scalar ``numpy.bool_`` is returned for one ``(10,)`` vector; otherwise
    the result has the parameter array's leading shape.
    """

    values = _parameter_array(parameters)
    eigenvalue_tolerance = float(eigenvalue_tolerance)
    triangle_tolerance = float(triangle_tolerance)
    if (
        not np.isfinite(eigenvalue_tolerance)
        or eigenvalue_tolerance < 0.0
        or not np.isfinite(triangle_tolerance)
        or triangle_tolerance < 0.0
    ):
        raise ValueError("validity tolerances must be finite and non-negative")

    finite = np.all(np.isfinite(values), axis=-1)
    positive_mass = values[..., 0] > 0.0

    # eigvalsh cannot be called safely on NaN/Inf particles.  Replacing only
    # for decomposition preserves their false result through ``finite``.
    safe_values = np.where(np.isfinite(values), values, 0.0)
    inertia = parameters_to_inertia(safe_values)
    principal = np.linalg.eigvalsh(inertia)
    scale = np.maximum(np.max(np.abs(principal), axis=-1), 1.0)
    positive_definite = principal[..., 0] > eigenvalue_tolerance * scale

    # For sorted positive moments, checking the largest against the other two
    # is sufficient; retaining the vector form makes the definition explicit.
    triangle_slack = np.sum(principal, axis=-1) - 2.0 * np.max(principal, axis=-1)
    triangle_valid = triangle_slack > triangle_tolerance * scale
    return finite & positive_mass & positive_definite & triangle_valid


def validate_physical_parameters(parameters: np.ndarray) -> np.ndarray:
    """Validate every parameter vector and return it as a float array.

    ``ValueError`` includes the invalid leading indices so callers can locate
    bad particles without silently evaluating a non-physical model.
    """

    values = _parameter_array(parameters)
    valid = np.asarray(physical_parameter_mask(values))
    if not np.all(valid):
        if valid.ndim == 0:
            detail = "the parameter vector"
        else:
            invalid_indices = np.argwhere(~valid)
            shown = [tuple(int(item) for item in index) for index in invalid_indices[:8]]
            suffix = "" if len(invalid_indices) <= 8 else " (additional indices omitted)"
            detail = "particle indices {}{}".format(shown, suffix)
        raise ValueError(
            "non-physical inertial parameters at {}: require finite values, mass > 0, "
            "SPD J_C, and strict principal-moment triangle inequalities".format(detail)
        )
    return values


def parameters_to_origin_inertia(parameters: np.ndarray) -> np.ndarray:
    """Shift center-of-mass inertia ``J_C`` to the body-frame origin ``J_O``."""

    values = validate_physical_parameters(parameters)
    mass = values[..., 0]
    center = values[..., 1:4]
    inertia_com = parameters_to_inertia(values)
    center_squared = np.einsum("...i,...i->...", center, center)
    parallel_axis = (
        center_squared[..., None, None] * np.eye(3, dtype=float)
        - np.einsum("...i,...j->...ij", center, center)
    )
    return inertia_com + mass[..., None, None] * parallel_axis


def predict_wrench(
    parameters: np.ndarray,
    specific_acceleration: np.ndarray,
    angular_velocity: np.ndarray,
    angular_acceleration: np.ndarray,
) -> np.ndarray:
    """Predict the body-frame inertial wrench for one or many particles.

    The returned vector is ordered ``[force_x, force_y, force_z, torque_x,
    torque_y, torque_z]``.  With ``h = m c`` and
    ``J_O = J_C + m ((c.T c) I - c c.T)``, the implemented equations are

    ``F = m s + alpha x h + omega x (omega x h)``

    ``tau = J_O alpha + omega x (J_O omega) + h x s``.

    Parameters and all kinematic vectors broadcast over their leading
    dimensions.  In particular ``parameters.shape == (N, 10)`` with three
    ``(3,)`` kinematic inputs returns ``(N, 6)``.
    """

    values = validate_physical_parameters(parameters)
    specific = _finite_vector3(specific_acceleration, "specific_acceleration")
    omega = _finite_vector3(angular_velocity, "angular_velocity")
    alpha = _finite_vector3(angular_acceleration, "angular_acceleration")

    try:
        leading_shape = np.broadcast_shapes(
            values.shape[:-1],
            specific.shape[:-1],
            omega.shape[:-1],
            alpha.shape[:-1],
        )
    except ValueError as exc:
        raise ValueError(
            "parameter and kinematic leading dimensions are not broadcast-compatible"
        ) from exc

    values = np.broadcast_to(values, leading_shape + (PARAMETER_COUNT,))
    specific = np.broadcast_to(specific, leading_shape + (3,))
    omega = np.broadcast_to(omega, leading_shape + (3,))
    alpha = np.broadcast_to(alpha, leading_shape + (3,))

    mass = values[..., 0]
    center = values[..., 1:4]
    first_moment = mass[..., None] * center
    inertia_origin = parameters_to_origin_inertia(values)

    force = (
        mass[..., None] * specific
        + np.cross(alpha, first_moment)
        + np.cross(omega, np.cross(omega, first_moment))
    )
    inertia_alpha = np.einsum("...ij,...j->...i", inertia_origin, alpha)
    inertia_omega = np.einsum("...ij,...j->...i", inertia_origin, omega)
    torque = (
        inertia_alpha
        + np.cross(omega, inertia_omega)
        + np.cross(first_moment, specific)
    )
    wrench = np.concatenate((force, torque), axis=-1)
    if not np.all(np.isfinite(wrench)):
        raise FloatingPointError("wrench prediction produced non-finite values")
    return wrench


# Descriptive compatibility alias for callers that prefer the longer name.
predict_actuator_wrench = predict_wrench


__all__ = [
    "PARAMETER_COUNT",
    "PARAMETER_NAMES",
    "inertia_to_parameters",
    "parameters_to_inertia",
    "parameters_to_origin_inertia",
    "physical_parameter_mask",
    "predict_actuator_wrench",
    "predict_wrench",
    "validate_physical_parameters",
]
