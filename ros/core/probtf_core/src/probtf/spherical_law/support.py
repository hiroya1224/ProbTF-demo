"""Linear algebra on the actual support of a Gaussian proxy.

The tolerances below classify floating point roundoff; they are never variances
and are never added to a covariance.  In particular a tangent-plane normal
remains deterministic.
"""

from dataclasses import dataclass

import numpy as np


SUPPORT_RTOL = 1.0e-12
SUPPORT_ATOL = 1.0e-14


def readonly(value):
    result = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("A Gaussian proxy must contain finite numbers.")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CovarianceSupport:
    inverse: np.ndarray
    basis: np.ndarray
    eigenvalues: np.ndarray
    tolerance: float
    roundoff_eigenvalues: tuple

    @property
    def rank(self):
        return self.eigenvalues.size

    @property
    def log_pdet(self):
        return float(np.sum(np.log(self.eigenvalues)))


def covariance_support(matrix, name="covariance"):
    """Return a support inverse, rejecting materially negative eigenvalues.

    Eigenvalues within ``atol + rtol * spectral_radius`` of zero are treated as
    numerical zero and recorded.  The supplied matrix is not regularized or
    reconstructed with floored eigenvalues.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("{} must be square.".format(name))
    if not np.all(np.isfinite(matrix)):
        raise ValueError("{} must be finite.".format(name))
    radius = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    symmetry_tolerance = SUPPORT_ATOL + SUPPORT_RTOL * radius
    if np.max(np.abs(matrix - matrix.T), initial=0.0) > symmetry_tolerance:
        raise ValueError("{} must be symmetric.".format(name))
    # Coordinates whose *entire* row/column is explicitly zero are Dirac
    # coordinates, not small random coordinates. Diagonalizing a larger
    # matrix containing these rows can return 1e-20 eigenvector entries there
    # and create spurious covariance during support-space reconstruction.
    # Remove only exact structural zeros, then embed the support with exact
    # zeros. No numerical threshold, covariance floor or uncertainty is added.
    active = np.any(matrix != 0.0, axis=0) | np.any(matrix != 0.0, axis=1)
    reduced = matrix[np.ix_(active, active)]
    eigenvalues, reduced_vectors = np.linalg.eigh(0.5 * (reduced + reduced.T))
    eigenvectors = np.zeros((matrix.shape[0], eigenvalues.size))
    eigenvectors[active] = reduced_vectors
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    tolerance = SUPPORT_ATOL + SUPPORT_RTOL * scale
    if np.any(eigenvalues < -tolerance):
        raise ValueError("{} has a significant negative eigenvalue: {}.".format(
            name, float(np.min(eigenvalues))))
    retained = eigenvalues > tolerance
    values = eigenvalues[retained]
    basis = eigenvectors[:, retained]
    inverse = (basis / values) @ basis.T if values.size else np.zeros_like(matrix)
    return CovarianceSupport(
        readonly(inverse), readonly(basis), readonly(values), tolerance,
        tuple(float(value) for value in eigenvalues[~retained] if value != 0.0),
    )


def support_inverse(matrix, name="covariance"):
    support = covariance_support(matrix, name)
    return support.inverse, support.basis, support.eigenvalues


def require_in_support(vector, support, name="mean difference"):
    vector = np.asarray(vector, dtype=float)
    residual = vector - support.basis @ (support.basis.T @ vector)
    tolerance = 1.0e-10 * max(1.0, float(np.linalg.norm(vector)))
    if np.linalg.norm(residual) > tolerance:
        raise ValueError("{} lies outside the reference affine support.".format(name))


def require_same_support(reference, candidate, name="posterior"):
    """Require equivalent covariance ranges, not merely the same rank."""
    reference_projector = reference.basis @ reference.basis.T
    candidate_projector = candidate.basis @ candidate.basis.T
    if (reference.rank != candidate.rank or not np.allclose(
            reference_projector, candidate_projector, rtol=0.0, atol=1.0e-9)):
        raise ValueError("{} has a different covariance support.".format(name))
