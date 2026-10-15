"""Deterministic integration of the full antipodally symmetric S3 law.

Four bounded gnomonic charts partition RP3 by its largest coordinate (ties
have measure zero). Their exact surface Jacobian is retained; concentration
only rescales chart coordinates, without a tangent-distribution approximation.
The resulting nodes are quadrature values, not a replacement orientation law.
"""
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp

from probtf.bingham import bingham_log_normalizer
from probtf.distributions import BinghamOrientation, OrientationKind
from probtf.geometry import quat_left_matrix


METHOD = "rp3_four_gnomonic_tensor_gauss_legendre_v1"


def _order(value):
    if isinstance(value, bool) or int(value) != value or value < 2:
        raise ValueError("quadrature order must be an integer at least two")
    return int(value)


@lru_cache(maxsize=12)
def _tensor_rule(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    coordinates = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
    product_weights = np.prod(np.stack(np.meshgrid(weights, weights, weights, indexing="ij"), axis=-1), axis=-1).reshape(-1)
    coordinates.setflags(write=False)
    product_weights.setflags(write=False)
    return coordinates, product_weights


def _equivariant_eigenframe(parameter, reference):
    """Resolve eigenspaces using a frame transported with the reference.

    This avoids arbitrary eigenspace rotations changing a finite cubature rule
    after deterministic left composition. Sign changes do not change the rule.
    """
    values, vectors = np.linalg.eigh(parameter)
    anchors = quat_left_matrix(reference)
    frame = np.empty((4, 4))
    tolerance = 1e-10 * max(1., float(np.max(np.abs(values))))
    begin = 0
    while begin < 4:
        end = begin + 1
        while end < 4 and abs(values[end] - values[begin]) <= tolerance:
            end += 1
        projection = vectors[:, begin:end] @ vectors[:, begin:end].T
        selected = []
        for anchor in anchors.T:
            candidate = projection @ anchor
            for previous in selected:
                candidate -= previous * np.dot(previous, candidate)
            norm = np.linalg.norm(candidate)
            if norm > 1e-8:
                selected.append(candidate / norm)
            if len(selected) == end - begin:
                break
        if len(selected) != end - begin:
            raise FloatingPointError("could not resolve the Bingham eigenspace frame")
        frame[:, begin:end] = np.column_stack(selected)
        begin = end
    return values, frame


def _orientation_rule(orientation, order, logarithmic=False):
    if not isinstance(orientation, BinghamOrientation):
        raise TypeError("orientation must be BinghamOrientation")
    order = _order(order)
    if orientation.kind is OrientationKind.DIRAC:
        return orientation.reference_quaternion_wxyz[None, :].copy(), np.full(1, 0. if logarithmic else 1.), 0.
    parameter = orientation.backend_parameter_matrix()
    eigenvalues, frame = _equivariant_eigenframe(parameter, orientation.reference_quaternion_wxyz)
    eigenvalues = eigenvalues - eigenvalues[-1]
    coordinates, product_weights = _tensor_rule(order)
    quaternion_parts, weight_parts = [], []
    for largest in range(4):
        others = np.array([index for index in range(4) if index != largest])
        scale = 1. / np.sqrt(np.maximum(1., eigenvalues[largest] - eigenvalues[others]))
        span = np.arctan(1. / scale)
        tangent = np.tan(span * coordinates)
        chart = tangent * scale
        denominator = 1. + np.sum(chart ** 2, axis=1)
        local = np.empty((len(chart), 4))
        local[:, others] = chart
        local[:, largest] = 1.
        local /= np.sqrt(denominator[:, None])
        quaternion_parts.append(local @ frame.T)
        # Both antipodal representatives have identical integrands.
        log_jacobian = (np.log(2.) - 2. * np.log(denominator)
                        + np.log(span * scale * (1. + tangent ** 2)).sum(axis=1))
        weight_parts.append(np.log(product_weights) + log_jacobian
                            + np.sum(local ** 2 * eigenvalues, axis=1))
    quaternions = np.concatenate(quaternion_parts)
    log_weights = np.concatenate(weight_parts)
    log_mass = float(logsumexp(log_weights))
    weights = log_weights - log_mass if logarithmic else np.exp(log_weights - log_mass)
    return quaternions, weights, log_mass


def orientation_quadrature(orientation, order=9):
    """Return deterministic unit quaternions and normalized integration weights.

    ``order`` is the one-dimensional Gauss--Legendre order: finite and uniform
    laws use ``4*order**3`` nodes. No random generator or fitted Gaussian is used.
    """
    quaternions, weights, _log_mass = _orientation_rule(orientation, order)
    quaternions.setflags(write=False)
    weights.setflags(write=False)
    return quaternions, weights


def orientation_quadrature_diagnostics(orientation, order=9):
    """Report a mass consistency check; it is not a density error bound."""
    quaternions, _weights, log_mass = _orientation_rule(orientation, order)
    result = dict(method=METHOD, order=_order(order), node_count=len(quaternions),
                  orientation_kind=orientation.kind.value, error_bound=None)
    if orientation.kind is OrientationKind.DIRAC:
        result.update(method="exact_dirac", relative_normalizer_error=0.)
    else:
        expected = bingham_log_normalizer(orientation.backend_parameter_matrix(), 480)
        result.update(relative_normalizer_error=float(abs(np.expm1(log_mass - expected))),
                      log_numerical_normalizer=log_mass, log_reference_normalizer=expected)
    return result


def quaternion_rotation_matrices(quaternions):
    """Vectorized active rotations in internal wxyz convention."""
    quaternions = np.asarray(quaternions, dtype=float)
    w, x, y, z = quaternions.T
    rotations = np.empty((len(quaternions), 3, 3))
    rotations[:, 0, 0] = w*w + x*x - y*y - z*z
    rotations[:, 0, 1] = 2*(x*y - w*z)
    rotations[:, 0, 2] = 2*(x*z + w*y)
    rotations[:, 1, 0] = 2*(x*y + w*z)
    rotations[:, 1, 1] = w*w - x*x + y*y - z*z
    rotations[:, 1, 2] = 2*(y*z - w*x)
    rotations[:, 2, 0] = 2*(x*z - w*y)
    rotations[:, 2, 1] = 2*(y*z + w*x)
    rotations[:, 2, 2] = w*w - x*x - y*y + z*z
    return rotations
