"""Unconstrained coordinates for positive Grape vehicle parameters.

The chart is centred at a nominal :class:`VehicleParameters` instance.  Its
18 coordinates are, in order,

``log mass`` (1), ``relative inertia`` (6), ``CoG offset`` (3),
``log force effectiveness`` (4), and ``log torque effectiveness`` (4).
The six inertia entries are ``S_xx, S_yy, S_zz, S_xy, S_xz, S_yz``.

The inertia coordinates describe a symmetric matrix ``S`` relative to the
nominal Cholesky factor ``L0``::

    inertia = L0 @ exp(S) @ L0.T

This makes every finite coordinate vector map to positive mass/effectiveness
and a symmetric positive-definite inertia without hard parameter bounds.
Linear and angular drag are fixed at their nominal values; they are not part
of this static-parameter chart.
"""

from typing import Sequence

import numpy as np

from grape_param_estim.system import VehicleParameters


PARAMETER_DIMENSION = 18

# Diagonal entries come first so adding ``scale * ridge_direction()`` adds a
# scalar multiple of the identity to the relative log-inertia matrix.
_SYMMETRIC_COMPONENTS = (
    (0, 0),
    (1, 1),
    (2, 2),
    (0, 1),
    (0, 2),
    (1, 2),
)


def _finite_matrix(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite {} array".format(name, shape))
    return result


def _symmetric_matrix_exponential(matrix: np.ndarray) -> np.ndarray:
    """Return the exponential of a finite real symmetric 3-by-3 matrix."""

    value = _finite_matrix(matrix, (3, 3), "symmetric matrix")
    if not np.allclose(value, value.T, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("matrix exponential input must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (value + value.T))
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            exponential = np.exp(eigenvalues)
        except FloatingPointError as error:
            raise ValueError("matrix exponential is not representable") from error
    if np.any(exponential <= 0.0) or not np.all(np.isfinite(exponential)):
        raise ValueError("matrix exponential is not representable")
    result = (eigenvectors * exponential) @ eigenvectors.T
    return 0.5 * (result + result.T)


def _symmetric_matrix_logarithm(matrix: np.ndarray) -> np.ndarray:
    """Return the principal log of a finite symmetric positive-definite matrix."""

    value = _finite_matrix(matrix, (3, 3), "positive-definite matrix")
    if not np.allclose(value, value.T, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("matrix logarithm input must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (value + value.T))
    if np.any(eigenvalues <= 0.0):
        raise ValueError("matrix logarithm input must be positive definite")
    logarithm = np.log(eigenvalues)
    result = (eigenvectors * logarithm) @ eigenvectors.T
    return 0.5 * (result + result.T)


def _unpack_symmetric(coordinates: Sequence[float]) -> np.ndarray:
    values = np.asarray(coordinates, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("inertia coordinates must contain six finite values")
    result = np.zeros((3, 3), dtype=float)
    for coordinate, (row, column) in zip(values, _SYMMETRIC_COMPONENTS):
        result[row, column] = coordinate
        result[column, row] = coordinate
    return result


def _pack_symmetric(matrix: np.ndarray) -> np.ndarray:
    value = _finite_matrix(matrix, (3, 3), "symmetric matrix")
    if not np.allclose(value, value.T, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("matrix must be symmetric")
    return np.asarray(
        [value[row, column] for row, column in _SYMMETRIC_COMPONENTS],
        dtype=float,
    )


def _positive_exponential(values: np.ndarray, name: str) -> np.ndarray:
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            result = np.exp(values)
        except FloatingPointError as error:
            raise ValueError(
                "{} coordinates are not representable".format(name)
            ) from error
    if np.any(result <= 0.0) or not np.all(np.isfinite(result)):
        raise ValueError("{} coordinates are not representable".format(name))
    return result


class VehicleParameterChart:
    """A bijective 18-D chart around fixed nominal vehicle parameters.

    Args:
        nominal: The controller/plant parameter snapshot defining chart zero.

    ``decode`` accepts one finite vector with shape ``(18,)``.  ``encode`` is
    its inverse for parameters whose drag values equal the nominal fixed drag.
    Returned parameter objects and coordinate arrays do not share mutable
    NumPy storage with the chart.
    """

    def __init__(self, nominal: VehicleParameters):
        if not isinstance(nominal, VehicleParameters):
            raise TypeError("nominal must be a VehicleParameters instance")
        self._mass = float(nominal.mass)
        self._inertia = np.asarray(nominal.inertia, dtype=float).copy()
        self._cholesky = np.linalg.cholesky(self._inertia)
        self._cog_offset = np.asarray(nominal.cog_offset, dtype=float).copy()
        self._force_effectiveness = np.asarray(
            nominal.force_effectiveness, dtype=float
        ).copy()
        self._torque_effectiveness = np.asarray(
            nominal.torque_effectiveness, dtype=float
        ).copy()
        self._linear_drag = np.asarray(nominal.linear_drag, dtype=float).copy()
        self._angular_drag = np.asarray(nominal.angular_drag, dtype=float).copy()

    def decode(self, coordinates: Sequence[float]) -> VehicleParameters:
        """Map one unconstrained coordinate vector to physical parameters."""

        values = np.asarray(coordinates, dtype=float)
        if values.shape != (PARAMETER_DIMENSION,) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "coordinates must contain {} finite values".format(
                    PARAMETER_DIMENSION
                )
            )

        positive_scales = _positive_exponential(
            np.concatenate((values[0:1], values[10:18])),
            "positive parameter",
        )
        relative_log_inertia = _unpack_symmetric(values[1:7])
        relative_inertia = _symmetric_matrix_exponential(
            relative_log_inertia
        )
        inertia = self._cholesky @ relative_inertia @ self._cholesky.T
        inertia = 0.5 * (inertia + inertia.T)
        return VehicleParameters(
            mass=self._mass * positive_scales[0],
            inertia=inertia,
            cog_offset=self._cog_offset + values[7:10],
            force_effectiveness=(
                self._force_effectiveness * positive_scales[1:5]
            ),
            torque_effectiveness=(
                self._torque_effectiveness * positive_scales[5:9]
            ),
            linear_drag=self._linear_drag,
            angular_drag=self._angular_drag,
        )

    def encode(self, parameters: VehicleParameters) -> np.ndarray:
        """Map represented physical parameters back to chart coordinates.

        Drag is intentionally fixed in this chart.  Passing parameters with a
        different linear or angular drag raises ``ValueError`` instead of
        silently discarding an unrepresented physical difference.
        """

        if not isinstance(parameters, VehicleParameters):
            raise TypeError("parameters must be a VehicleParameters instance")
        if not np.allclose(
            parameters.linear_drag,
            self._linear_drag,
            rtol=1.0e-12,
            atol=1.0e-15,
        ) or not np.allclose(
            parameters.angular_drag,
            self._angular_drag,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError("linear and angular drag are fixed by the chart")

        # L0^-1 I L0^-T, evaluated with solves instead of explicit inverses.
        left_whitened = np.linalg.solve(
            self._cholesky, np.asarray(parameters.inertia, dtype=float)
        )
        relative_inertia = np.linalg.solve(
            self._cholesky, left_whitened.T
        ).T
        relative_inertia = 0.5 * (
            relative_inertia + relative_inertia.T
        )
        relative_log_inertia = _symmetric_matrix_logarithm(
            relative_inertia
        )

        result = np.empty(PARAMETER_DIMENSION, dtype=float)
        result[0] = np.log(parameters.mass / self._mass)
        result[1:7] = _pack_symmetric(relative_log_inertia)
        result[7:10] = parameters.cog_offset - self._cog_offset
        result[10:14] = np.log(
            parameters.force_effectiveness / self._force_effectiveness
        )
        result[14:18] = np.log(
            parameters.torque_effectiveness / self._torque_effectiveness
        )
        if not np.all(np.isfinite(result)):
            raise ValueError("parameters cannot be represented by this chart")
        return result

    def ridge_direction(self) -> np.ndarray:
        """Return the exact common-scale dynamics ridge in chart coordinates.

        Adding ``s * direction`` multiplies mass, the full inertia tensor, and
        every rotor force effectiveness by ``exp(s)`` while leaving CoG and
        reaction-torque effectiveness unchanged.  With fixed zero drag this
        scales the complete body wrench and rigid-body inertia together, so
        the six-DoF acceleration forecast is exactly unchanged.
        """

        direction = np.zeros(PARAMETER_DIMENSION, dtype=float)
        direction[0] = 1.0
        direction[1:4] = 1.0
        direction[10:14] = 1.0
        return direction


__all__ = ["PARAMETER_DIMENSION", "VehicleParameterChart"]
