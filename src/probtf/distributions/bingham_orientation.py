"""JMAA-normalized Bingham orientation storage."""

from dataclasses import dataclass

import numpy as np

from probtf.distributions.status import OrientationKind
from probtf.distributions.validation import (
    DistributionValidationError,
    immutable_symmetric_matrix,
    immutable_unit_quaternion,
)


SHAPE_TOLERANCE = 1e-8


def trace_zero_matrix(parameter_matrix):
    matrix = immutable_symmetric_matrix(parameter_matrix, 4, "parameter_matrix")
    output = matrix - 0.25 * float(np.trace(matrix)) * np.eye(4, dtype=float)
    return 0.5 * (output + output.T)


def bingham_shape_magnitude(trace_zero_parameter):
    """Return the JMAA magnitude ``kappa = lambda_1 + lambda_2``."""

    matrix = trace_zero_matrix(trace_zero_parameter)
    eigenvalues = np.linalg.eigvalsh(matrix)
    return float(eigenvalues[-1] + eigenvalues[-2])


def normalize_bingham_shape(parameter_matrix):
    matrix = trace_zero_matrix(parameter_matrix)
    magnitude = bingham_shape_magnitude(matrix)
    if magnitude <= SHAPE_TOLERANCE:
        if np.linalg.norm(matrix) <= SHAPE_TOLERANCE:
            return np.zeros((4, 4), dtype=float), 0.0
        raise DistributionValidationError("Bingham shape magnitude must be positive.")
    shape = matrix / magnitude
    return 0.5 * (shape + shape.T), magnitude


def dirac_shape_from_mode(quat_wxyz):
    quaternion = immutable_unit_quaternion(quat_wxyz, "quat_wxyz")
    seed = np.outer(quaternion, quaternion) - 0.25 * np.eye(4, dtype=float)
    shape, magnitude = normalize_bingham_shape(seed)
    if not np.isclose(magnitude, 0.5, rtol=0.0, atol=SHAPE_TOLERANCE):
        raise AssertionError("Unexpected Dirac shape magnitude.")
    return shape


def _mode_from_shape(shape):
    _, eigenvectors = np.linalg.eigh(shape)
    mode = eigenvectors[:, -1]
    pivot = int(np.argmax(np.abs(mode)))
    if mode[pivot] < 0.0:
        mode = -mode
    mode.setflags(write=False)
    return mode


@dataclass(frozen=True)
class BinghamOrientation:
    """Orientation law in trace-zero JMAA shape/scale form.

    The shape magnitude is ``lambda_1 + lambda_2 == 1``.  A finite law has
    ``A = shape_matrix / inverse_concentration``.  The Dirac limit uses zero
    inverse concentration without calling a finite Bingham normalizer.  The
    uniform law stores a zero shape and positive infinity inverse
    concentration.
    """

    kind: OrientationKind
    inverse_concentration: float
    shape_matrix: np.ndarray
    reference_quaternion_wxyz: np.ndarray

    def __post_init__(self):
        if not isinstance(self.kind, OrientationKind):
            raise TypeError("kind must be an OrientationKind.")
        shape = immutable_symmetric_matrix(self.shape_matrix, 4, "shape_matrix")
        if not np.isclose(np.trace(shape), 0.0, rtol=0.0, atol=SHAPE_TOLERANCE):
            raise DistributionValidationError("shape_matrix must have zero trace.")
        reference = immutable_unit_quaternion(
            self.reference_quaternion_wxyz,
            "reference_quaternion_wxyz",
        )
        inverse = float(self.inverse_concentration)

        if self.kind is OrientationKind.FINITE_BINGHAM:
            if not np.isfinite(inverse) or inverse <= 0.0:
                raise DistributionValidationError(
                    "A finite Bingham orientation requires positive finite inverse concentration."
                )
            magnitude = bingham_shape_magnitude(shape)
            if not np.isclose(magnitude, 1.0, rtol=0.0, atol=SHAPE_TOLERANCE):
                raise DistributionValidationError(
                    "shape_matrix must use JMAA normalization (lambda_1 + lambda_2 == 1)."
                )
        elif self.kind is OrientationKind.DIRAC:
            if inverse != 0.0:
                raise DistributionValidationError("A Dirac orientation requires zero inverse concentration.")
            expected = dirac_shape_from_mode(reference)
            if not np.allclose(shape, expected, rtol=0.0, atol=SHAPE_TOLERANCE):
                raise DistributionValidationError(
                    "A Dirac shape must be the normalized trace-zero shape of its mode."
                )
        else:
            if inverse != np.inf:
                raise DistributionValidationError(
                    "A uniform orientation requires infinite inverse concentration."
                )
            if not np.allclose(shape, np.zeros((4, 4)), rtol=0.0, atol=SHAPE_TOLERANCE):
                raise DistributionValidationError("A uniform orientation requires a zero shape.")

        object.__setattr__(self, "inverse_concentration", inverse)
        object.__setattr__(self, "shape_matrix", shape)
        object.__setattr__(self, "reference_quaternion_wxyz", reference)

    @classmethod
    def from_parameter_matrix(cls, parameter_matrix, reference_quaternion_wxyz=None):
        shape, magnitude = normalize_bingham_shape(parameter_matrix)
        if magnitude == 0.0:
            reference = (
                np.array([1.0, 0.0, 0.0, 0.0])
                if reference_quaternion_wxyz is None
                else reference_quaternion_wxyz
            )
            return cls.uniform(reference)
        reference = _mode_from_shape(shape) if reference_quaternion_wxyz is None else reference_quaternion_wxyz
        return cls(OrientationKind.FINITE_BINGHAM, 1.0 / magnitude, shape, reference)

    @classmethod
    def dirac(cls, quat_wxyz):
        return cls(OrientationKind.DIRAC, 0.0, dirac_shape_from_mode(quat_wxyz), quat_wxyz)

    @classmethod
    def uniform(cls, reference_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0)):
        return cls(
            OrientationKind.UNIFORM,
            np.inf,
            np.zeros((4, 4), dtype=float),
            reference_quaternion_wxyz,
        )

    @property
    def mode_wxyz(self):
        if self.kind is OrientationKind.UNIFORM:
            return self.reference_quaternion_wxyz.copy()
        if self.kind is OrientationKind.DIRAC:
            return self.reference_quaternion_wxyz.copy()
        return _mode_from_shape(self.shape_matrix).copy()

    def parameter_matrix(self):
        if self.kind is OrientationKind.DIRAC:
            raise DistributionValidationError(
                "A Dirac orientation has no finite Bingham parameter matrix."
            )
        if self.kind is OrientationKind.UNIFORM:
            return np.zeros((4, 4), dtype=float)
        return self.shape_matrix / self.inverse_concentration

    def backend_parameter_matrix(self):
        """Return a max-eigenvalue-zero finite-backend gauge."""

        parameter = self.parameter_matrix()
        largest = float(np.linalg.eigvalsh(parameter)[-1])
        return parameter - largest * np.eye(4, dtype=float)

