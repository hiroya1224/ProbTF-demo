"""Bingham distribution moments for unit quaternions in ``wxyz`` order.

A Bingham parameter matrix is defined only up to an additive scalar multiple
of the identity.  This module uses the canonical gauge whose largest
eigenvalue is zero.  Quaternion second moments have shape ``(4, 4)`` and
fourth moments use the index order ``E[q_i q_j q_k q_l]``.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from probtf._vendor import bingham_normconst
from probtf.geometry import quat_to_rotmat


_MATRIX_TOLERANCE = 1e-10
_MOMENT_TOLERANCE = 1e-6


def _symmetric_matrix(values, size, name, tolerance=_MATRIX_TOLERANCE):
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (size, size):
        raise ValueError("%s must have shape (%d, %d)." % (name, size, size))
    if not np.all(np.isfinite(matrix)):
        raise ValueError("%s must contain only finite values." % name)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        raise ValueError("%s must be symmetric." % name)
    return 0.5 * (matrix + matrix.T)


def _integration_steps(value):
    steps = int(value)
    if steps < 1 or steps != value:
        raise ValueError("integration_steps must be a positive integer.")
    return steps


def validate_bingham_parameter(parameter_matrix):
    """Validate and return a symmetric 4x4 Bingham parameter matrix.

    The matrix acts on quaternions in ``[w, x, y, z]`` order.  Validation does
    not alter its gauge; use :func:`canonical_bingham_parameter` when storing
    or transporting the parameter.
    """

    return _symmetric_matrix(parameter_matrix, 4, "parameter_matrix")


def canonical_bingham_parameter(parameter_matrix):
    """Return the max-eigenvalue-zero representative of a Bingham parameter."""

    matrix = validate_bingham_parameter(parameter_matrix)
    largest_eigenvalue = float(np.linalg.eigvalsh(matrix)[-1])
    canonical = matrix - largest_eigenvalue * np.eye(4, dtype=float)
    return 0.5 * (canonical + canonical.T)


def bingham_log_normalizer(parameter_matrix, integration_steps=120):
    """Return ``log integral exp(q.T A q) dq`` on the unit 3-sphere.

    Unlike moment queries, the normalizer changes under a scalar gauge shift;
    this function therefore restores the largest eigenvalue after evaluating
    the canonical max-eigenvalue-zero parameter.
    """

    parameter = validate_bingham_parameter(parameter_matrix)
    largest_eigenvalue = float(np.linalg.eigvalsh(parameter)[-1])
    canonical = parameter - largest_eigenvalue * np.eye(4, dtype=float)
    eigenvalues = np.linalg.eigvalsh(canonical)
    constant, _, _ = _normalizer_derivatives(eigenvalues, integration_steps)
    return float(np.log(constant) + largest_eigenvalue)


def bingham_mode(parameter_matrix):
    """Return one unit mode quaternion in ``wxyz`` order.

    A Bingham distribution is antipodally symmetric, so both the returned
    quaternion and its negative represent the same mode.  The sign is made
    deterministic by making the largest-magnitude component nonnegative.
    """

    parameter = canonical_bingham_parameter(parameter_matrix)
    _, eigenvectors = np.linalg.eigh(parameter)
    mode = eigenvectors[:, -1]
    pivot = int(np.argmax(np.abs(mode)))
    if mode[pivot] < 0.0:
        mode = -mode
    return mode


def _normalizer_derivatives(eigenvalues, integration_steps, need_second_derivative=False):
    steps = _integration_steps(integration_steps)
    values = np.asarray(eigenvalues, dtype=float).reshape(4)
    values = values - float(np.max(values))

    constant = float(
        np.asarray(bingham_normconst.calc_constant(values, N=steps), dtype=float).reshape(-1)[0]
    )
    derivative = np.asarray(
        bingham_normconst.calc_Dconstant(values, N=steps),
        dtype=float,
    ).reshape(-1, 4)[0]
    if not np.isfinite(constant) or constant <= 0.0 or not np.all(np.isfinite(derivative)):
        raise FloatingPointError("Bingham normalizer evaluation failed.")

    if not need_second_derivative:
        return constant, derivative, None

    second_derivative = np.asarray(
        bingham_normconst.calc_DDconstant(values, N=steps, Hesse=True),
        dtype=float,
    ).reshape(-1, 4, 4)[0]
    if not np.all(np.isfinite(second_derivative)):
        raise FloatingPointError("Bingham normalizer Hessian evaluation failed.")
    return constant, derivative, second_derivative


def _eigendecomposition(parameter_matrix):
    return np.linalg.eigh(canonical_bingham_parameter(parameter_matrix))


def bingham_second_moment(parameter_matrix, integration_steps=120):
    """Return ``E[q q.T]`` for a quaternion Bingham distribution."""

    eigenvalues, eigenvectors = _eigendecomposition(parameter_matrix)
    constant, derivative, _ = _normalizer_derivatives(eigenvalues, integration_steps)
    diagonal_moments = derivative / constant
    diagonal_moments /= float(np.sum(diagonal_moments))
    moment = eigenvectors @ np.diag(diagonal_moments) @ eigenvectors.T
    return 0.5 * (moment + moment.T)


def bingham_fourth_moment(parameter_matrix, integration_steps=120):
    """Return ``E[q_i q_j q_k q_l]`` as a ``(4, 4, 4, 4)`` tensor."""

    eigenvalues, eigenvectors = _eigendecomposition(parameter_matrix)
    constant, _, second_derivative = _normalizer_derivatives(
        eigenvalues,
        integration_steps,
        need_second_derivative=True,
    )
    diagonalized = np.zeros((4, 4, 4, 4), dtype=float)
    normalized_derivative = second_derivative / constant

    for index in range(4):
        diagonalized[index, index, index, index] = normalized_derivative[index, index]

    for row in range(4):
        for column in range(row + 1, 4):
            value = normalized_derivative[row, column]
            for indices in (
                (row, row, column, column),
                (row, column, row, column),
                (row, column, column, row),
                (column, row, row, column),
                (column, row, column, row),
                (column, column, row, row),
            ):
                diagonalized[indices] = value

    moment = np.einsum(
        "ai,bj,ck,dl,ijkl->abcd",
        eigenvectors,
        eigenvectors,
        eigenvectors,
        eigenvectors,
        diagonalized,
    )
    unit_norm = float(np.einsum("iikk->", moment))
    if not np.isfinite(unit_norm) or unit_norm <= 0.0:
        raise FloatingPointError("Bingham fourth moment normalization failed.")
    return moment / unit_norm


def _validated_second_moment(second_moment):
    moment = _symmetric_matrix(second_moment, 4, "second_moment")
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    if float(eigenvalues[0]) < -_MOMENT_TOLERANCE:
        raise ValueError("second_moment must be positive semidefinite.")
    trace = float(np.trace(moment))
    if not np.isclose(trace, 1.0, rtol=0.0, atol=1e-3):
        raise ValueError("A unit-quaternion second moment must have trace one.")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    eigenvalues /= float(np.sum(eigenvalues))
    normalized = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return 0.5 * (normalized + normalized.T)


def _parameter_vector_to_eigenvalues(parameters):
    gaps = np.exp(np.asarray(parameters, dtype=float).reshape(3))
    eigenvalue_2 = -gaps[2]
    eigenvalue_1 = eigenvalue_2 - gaps[1]
    eigenvalue_0 = eigenvalue_1 - gaps[0]
    return np.array([eigenvalue_0, eigenvalue_1, eigenvalue_2, 0.0], dtype=float), gaps


def _eigenvalues_to_parameter_vector(eigenvalues):
    values = np.sort(np.asarray(eigenvalues, dtype=float).reshape(4))
    values -= values[-1]
    gaps = np.array(
        [
            max(values[1] - values[0], 1e-8),
            max(values[2] - values[1], 1e-8),
            max(-values[2], 1e-8),
        ],
        dtype=float,
    )
    return np.log(gaps)


def _moment_eigenvalues_and_jacobian(eigenvalues, integration_steps):
    constant, derivative, second_derivative = _normalizer_derivatives(
        eigenvalues,
        integration_steps,
        need_second_derivative=True,
    )
    raw_moments = derivative / constant
    raw_jacobian = second_derivative / constant - np.outer(derivative, derivative) / (constant ** 2)

    total = float(np.sum(raw_moments))
    total_jacobian = np.sum(raw_jacobian, axis=0)
    moments = raw_moments / total
    jacobian = raw_jacobian / total - np.outer(raw_moments, total_jacobian) / (total ** 2)
    return moments, jacobian


def match_bingham_to_second_moment(
    second_moment,
    integration_steps=120,
    initial_eigenvalues=None,
    max_iterations=80,
):
    """Fit a canonical Bingham parameter by matching a quaternion second moment."""

    target = _validated_second_moment(second_moment)
    target_eigenvalues, target_eigenvectors = np.linalg.eigh(target)
    order = np.argsort(target_eigenvalues)
    target_eigenvalues = target_eigenvalues[order]
    target_eigenvectors = target_eigenvectors[:, order]

    if initial_eigenvalues is None:
        initial_eigenvalues = np.array([-30.0, -15.0, -5.0, 0.0], dtype=float)
    initial_parameters = _eigenvalues_to_parameter_vector(initial_eigenvalues)

    def objective(parameters):
        candidate_eigenvalues, gaps = _parameter_vector_to_eigenvalues(parameters)
        candidate_moments, moment_jacobian = _moment_eigenvalues_and_jacobian(
            candidate_eigenvalues,
            integration_steps,
        )
        error = candidate_moments - target_eigenvalues
        loss = 0.5 * float(error @ error)

        eigenvalue_jacobian = np.array(
            [
                [-gaps[0], -gaps[1], -gaps[2]],
                [0.0, -gaps[1], -gaps[2]],
                [0.0, 0.0, -gaps[2]],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        gradient = eigenvalue_jacobian.T @ (moment_jacobian.T @ error)
        return loss, gradient

    iterations = int(max_iterations)
    if iterations < 1 or iterations != max_iterations:
        raise ValueError("max_iterations must be a positive integer.")
    optimization = minimize(
        lambda parameters: objective(parameters)[0],
        initial_parameters,
        jac=lambda parameters: objective(parameters)[1],
        method="L-BFGS-B",
        bounds=[(-20.0, 12.0)] * 3,
        options={"maxiter": iterations, "ftol": 1e-14, "gtol": 1e-10},
    )
    if not np.all(np.isfinite(optimization.x)):
        raise RuntimeError("Bingham moment matching produced non-finite parameters.")

    fitted_eigenvalues, _ = _parameter_vector_to_eigenvalues(optimization.x)
    parameter = target_eigenvectors @ np.diag(fitted_eigenvalues) @ target_eigenvectors.T
    return canonical_bingham_parameter(parameter)


_QUATERNION_PRODUCT_FORMS = np.array(
    [
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]],
        [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]],
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]],
    ],
    dtype=float,
)


def quaternion_product_second_moment(second_moment_left, second_moment_right):
    """Return the second moment of the product of independent quaternions.

    The result corresponds to ``q_left * q_right`` using the Hamilton product.
    """

    left = _validated_second_moment(second_moment_left)
    right = _validated_second_moment(second_moment_right)
    output = np.einsum(
        "ik,jl,aij,bkl->ab",
        left,
        right,
        _QUATERNION_PRODUCT_FORMS,
        _QUATERNION_PRODUCT_FORMS,
    )
    return _validated_second_moment(0.5 * (output + output.T))


def _rotation_quadratic_forms():
    forms = []
    forms.append(np.diag([1.0, 1.0, -1.0, -1.0]))

    form = np.zeros((4, 4), dtype=float)
    form[0, 3] = form[3, 0] = -1.0
    form[1, 2] = form[2, 1] = 1.0
    forms.append(form)

    form = np.zeros((4, 4), dtype=float)
    form[0, 2] = form[2, 0] = 1.0
    form[1, 3] = form[3, 1] = 1.0
    forms.append(form)

    form = np.zeros((4, 4), dtype=float)
    form[0, 3] = form[3, 0] = 1.0
    form[1, 2] = form[2, 1] = 1.0
    forms.append(form)
    forms.append(np.diag([1.0, -1.0, 1.0, -1.0]))

    form = np.zeros((4, 4), dtype=float)
    form[0, 1] = form[1, 0] = -1.0
    form[2, 3] = form[3, 2] = 1.0
    forms.append(form)

    form = np.zeros((4, 4), dtype=float)
    form[0, 2] = form[2, 0] = -1.0
    form[1, 3] = form[3, 1] = 1.0
    forms.append(form)

    form = np.zeros((4, 4), dtype=float)
    form[0, 1] = form[1, 0] = 1.0
    form[2, 3] = form[3, 2] = 1.0
    forms.append(form)
    forms.append(np.diag([1.0, -1.0, -1.0, 1.0]))
    return tuple(forms)


_ROTATION_ENTRY_FORMS = _rotation_quadratic_forms()
_COLUMN_MAJOR_MATRIX_INDICES = (
    (0, 0),
    (1, 0),
    (2, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (0, 2),
    (1, 2),
    (2, 2),
)


def rotation_first_moment(second_moment):
    """Convert ``E[q q.T]`` to the first rotation-matrix moment ``E[R]``."""

    quaternion_moment = _validated_second_moment(second_moment)
    mean_rotation = np.empty((3, 3), dtype=float)
    for flat_index, form in enumerate(_ROTATION_ENTRY_FORMS):
        row, column = divmod(flat_index, 3)
        mean_rotation[row, column] = float(np.sum(form * quaternion_moment))
    return mean_rotation


def rotation_kronecker_moment(fourth_moment):
    """Convert ``E[q^4]`` to ``E[R kron R]`` for column-major vectorization."""

    quaternion_moment = np.asarray(fourth_moment, dtype=float)
    if quaternion_moment.shape != (4, 4, 4, 4):
        raise ValueError("fourth_moment must have shape (4, 4, 4, 4).")
    if not np.all(np.isfinite(quaternion_moment)):
        raise ValueError("fourth_moment must contain only finite values.")

    kronecker_moment = np.empty((9, 9), dtype=float)
    for row_index, (output_row, output_column) in enumerate(_COLUMN_MAJOR_MATRIX_INDICES):
        for column_index, (input_row, input_column) in enumerate(_COLUMN_MAJOR_MATRIX_INDICES):
            form_a = _ROTATION_ENTRY_FORMS[output_row * 3 + input_row]
            form_b = _ROTATION_ENTRY_FORMS[output_column * 3 + input_column]
            kronecker_moment[row_index, column_index] = float(
                np.einsum("ab,cd,abcd->", form_a, form_b, quaternion_moment)
            )
    return kronecker_moment


def rotation_vector_second_moment(fourth_moment):
    """Return ``E[vec(R) vec(R).T]`` for column-major ``vec(R)``.

    This is distinct from :func:`rotation_kronecker_moment`, whose index
    arrangement is the linear operator used for ``E[R S R.T]``.
    """

    quaternion_moment = np.asarray(fourth_moment, dtype=float)
    if quaternion_moment.shape != (4, 4, 4, 4):
        raise ValueError("fourth_moment must have shape (4, 4, 4, 4).")
    if not np.all(np.isfinite(quaternion_moment)):
        raise ValueError("fourth_moment must contain only finite values.")
    output = np.empty((9, 9), dtype=float)
    for row_index, (row_a, column_a) in enumerate(_COLUMN_MAJOR_MATRIX_INDICES):
        form_a = _ROTATION_ENTRY_FORMS[row_a * 3 + column_a]
        for column_index, (row_b, column_b) in enumerate(_COLUMN_MAJOR_MATRIX_INDICES):
            form_b = _ROTATION_ENTRY_FORMS[row_b * 3 + column_b]
            output[row_index, column_index] = float(
                np.einsum("ab,cd,abcd->", form_a, form_b, quaternion_moment)
            )
    return 0.5 * (output + output.T)


@dataclass(frozen=True)
class RotationMoment:
    """First and Kronecker moments of a random rotation matrix."""

    mean_rot: np.ndarray
    kron_rot: np.ndarray

    def __post_init__(self):
        mean_rotation = np.asarray(self.mean_rot, dtype=float)
        kronecker_rotation = np.asarray(self.kron_rot, dtype=float)
        if mean_rotation.shape != (3, 3) or not np.all(np.isfinite(mean_rotation)):
            raise ValueError("mean_rot must be a finite 3x3 matrix.")
        if kronecker_rotation.shape != (9, 9) or not np.all(np.isfinite(kronecker_rotation)):
            raise ValueError("kron_rot must be a finite 9x9 matrix.")
        object.__setattr__(self, "mean_rot", mean_rotation.copy())
        object.__setattr__(self, "kron_rot", kronecker_rotation.copy())

    @property
    def first(self):
        return self.mean_rot

    @property
    def kronecker(self):
        return self.kron_rot

    def apply_second(self, matrix):
        values = np.asarray(matrix, dtype=float)
        if values.shape != (3, 3) or not np.all(np.isfinite(values)):
            raise ValueError("matrix must be a finite 3x3 matrix.")
        vector = values.reshape(9, order="F")
        return (self.kron_rot @ vector).reshape((3, 3), order="F")

    def compose(self, other):
        if not isinstance(other, RotationMoment):
            raise TypeError("other must be a RotationMoment.")
        return RotationMoment(
            self.mean_rot @ other.mean_rot,
            self.kron_rot @ other.kron_rot,
        )


def identity_rotation_moment():
    return RotationMoment(np.eye(3, dtype=float), np.eye(9, dtype=float))


def deterministic_rotation_moment_from_quaternion(quaternion_wxyz):
    rotation = quat_to_rotmat(quaternion_wxyz)
    return RotationMoment(rotation, np.kron(rotation, rotation))


def rotation_moment_from_bingham(parameter_matrix, integration_steps=120):
    """Return the first and Kronecker rotation moments of a Bingham law."""

    second_moment = bingham_second_moment(parameter_matrix, integration_steps=integration_steps)
    fourth_moment = bingham_fourth_moment(parameter_matrix, integration_steps=integration_steps)
    return RotationMoment(
        rotation_first_moment(second_moment),
        rotation_kronecker_moment(fourth_moment),
    )


# Compatibility names used by the original ProbTF prototype.
compute_mean_rot_from_c2 = rotation_first_moment
compute_kron_rot_from_c4 = rotation_kronecker_moment


__all__ = [
    "RotationMoment",
    "bingham_fourth_moment",
    "bingham_log_normalizer",
    "bingham_mode",
    "bingham_second_moment",
    "canonical_bingham_parameter",
    "compute_kron_rot_from_c4",
    "compute_mean_rot_from_c2",
    "deterministic_rotation_moment_from_quaternion",
    "identity_rotation_moment",
    "match_bingham_to_second_moment",
    "quaternion_product_second_moment",
    "rotation_first_moment",
    "rotation_kronecker_moment",
    "rotation_moment_from_bingham",
    "validate_bingham_parameter",
]
