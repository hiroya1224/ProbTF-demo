"""Bingham-derived, globally first-order correlated transform queries.

The local coordinates are *derived from the supplied Bingham laws*: parent
translation followed by a right rotation vector, in six-coordinate blocks.
They are not a separately assumed SE(3) Gaussian model.  Every output is a
specified Gaussian proxy; in particular its directional mean contraction is
not used to construct a rotation matrix.
"""

from dataclasses import dataclass, field, replace
import hashlib

import numpy as np

from probtf.geometry import (
    quat_left_matrix, quat_mul, quat_normalize, quat_to_rotmat,
    rotation_vector_to_quaternion, rotmat_to_quat, skew,
)
from probtf.spherical_law.support import covariance_support, require_in_support, support_inverse


# An explicitly declared applicability bound, not a covariance adjustment or
# an asymptotic error guarantee.  Broad inputs are rejected, never shrunk.
MAX_FIRST_ORDER_ROTATION_VARIANCE = 0.1  # radian squared


def _array(value, shape=None):
    result = np.array(value, dtype=float, copy=True)
    if shape is not None and result.shape != shape:
        raise ValueError("Expected shape {}, got {}.".format(shape, result.shape))
    if not np.all(np.isfinite(result)):
        raise ValueError("Distribution parameters must be finite.")
    result.setflags(write=False)
    return result


def bingham_rotation_coordinates(matrix):
    """Return modal quaternion and C_A=2(lambda I-N^T A N)^-1.

    The factor two is the quaternion half-angle conversion; there is no
    normal jitter, eigenvalue floor, quadrature, or sampling here.
    """
    matrix = _array(matrix, (4, 4))
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12):
        raise ValueError("Bingham parameter must be symmetric.")
    # B(A+cI)=B(A); remove the scalar offset before testing concentration or
    # taking modal gaps so the computation respects that gauge invariance.
    matrix = matrix - np.trace(matrix)*0.25*np.eye(4)
    values, vectors = np.linalg.eigh(matrix)
    scale = max(1.0, float(np.max(np.abs(values))))
    if values[-1] - values[-2] <= 1e-12 * scale:
        raise ValueError("A single-pose proxy needs a simple maximum Bingham eigenvalue.")
    quaternion = quat_normalize(vectors[:, -1])
    if quaternion[np.argmax(np.abs(quaternion))] < 0.0:
        quaternion = -quaternion
    tangent = quat_left_matrix(quaternion)[:, 1:]
    gap = values[-1] * np.eye(3) - tangent.T @ matrix @ tangent
    inverse, basis, _ = support_inverse(gap, name="Bingham modal gap")
    if basis.shape[1] != 3:
        raise ValueError("A continuous unobserved orientation is not a single-pose Bingham.")
    covariance = 2.0 * inverse
    if np.max(np.linalg.eigvalsh(covariance)) > MAX_FIRST_ORDER_ROTATION_VARIANCE:
        raise ValueError("Bingham rotation exceeds the declared first-order concentration domain (0.1 rad^2).")
    return _array(quaternion), _array(covariance)


def bingham_from_rotation_coordinates(quaternion, covariance):
    """First-order reexpression, A=-2 N Omega^-1 N^T (maximum zero)."""
    quaternion = quat_normalize(quaternion)
    inverse, basis, _ = support_inverse(covariance, name="rotation covariance")
    if basis.shape[1] != 3:
        raise ValueError("A continuous Bingham reexpression needs full rotation support.")
    tangent = quat_left_matrix(quaternion)[:, 1:]
    parameter = -2.0 * tangent @ inverse @ tangent.T
    # This quadratic form is symmetric analytically. Remove only the skew
    # introduced by matrix-product roundoff; r^T A r is unchanged by it.
    return _array(0.5 * (parameter + parameter.T))


@dataclass(frozen=True)
class BinghamTransform:
    """A directly supplied local transform law or an explicitly fixed edge."""

    edge_id: str
    translation: np.ndarray
    bingham: object = None
    translation_covariance: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    rotation_wxyz: object = None

    def __post_init__(self):
        if not self.edge_id:
            raise ValueError("An edge ID is required.")
        object.__setattr__(self, "translation", _array(self.translation, (3,)))
        covariance = _array(self.translation_covariance, (3, 3))
        support_inverse(covariance, name="translation covariance")
        object.__setattr__(self, "translation_covariance", covariance)
        if self.bingham is not None:
            if self.rotation_wxyz is not None:
                raise ValueError("Supply a Bingham law or a fixed rotation, not both.")
            matrix = _array(self.bingham, (4, 4))
            quaternion, _ = bingham_rotation_coordinates(matrix)
            object.__setattr__(self, "bingham", matrix)
        else:
            if self.rotation_wxyz is None:
                raise ValueError("An explicitly fixed edge needs its known rotation.")
            quaternion = quat_normalize(self.rotation_wxyz)
        object.__setattr__(self, "rotation_wxyz", _array(quaternion, (4,)))


@dataclass(frozen=True)
class JointTransformState:
    """Atomic immutable joint proxy, including every cross covariance block."""

    transforms: tuple
    covariance: np.ndarray
    version: int = 0
    stamp: float = 0.0
    command: tuple = ()
    component_id: str = "body"
    mean: object = None
    state_id: str = field(init=False)

    def __post_init__(self):
        transforms = tuple(self.transforms)
        if len({edge.edge_id for edge in transforms}) != len(transforms):
            raise ValueError("Physical transform variables need unique edge IDs.")
        n = 6 * len(transforms)
        covariance = _array(self.covariance, (n, n))
        support_inverse(covariance, name="joint body covariance")
        mean = np.zeros(n) if self.mean is None else np.asarray(self.mean, dtype=float)
        if mean.shape != (n,) or np.any(mean != 0.0):
            raise ValueError("Reexpress the joint state before querying a nonzero chart mean.")
        for i, edge in enumerate(transforms):
            translation_block = covariance[6*i:6*i+3, 6*i:6*i+3]
            if not np.allclose(translation_block, edge.translation_covariance, rtol=1e-8, atol=1e-12):
                raise ValueError("The translation law and joint marginal disagree.")
            block = covariance[6*i+3:6*i+6, 6*i+3:6*i+6]
            if edge.bingham is None and np.any(block != 0.0):
                raise ValueError("A known fixed rotation cannot carry a random rotation block.")
            if edge.bingham is not None:
                _, expected = bingham_rotation_coordinates(edge.bingham)
                if not np.allclose(block, expected, rtol=1e-8, atol=1e-12):
                    raise ValueError("The Bingham and joint rotation marginal disagree.")
        object.__setattr__(self, "transforms", transforms)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "mean", _array(mean))
        object.__setattr__(self, "command", tuple(float(v) for v in self.command))
        identity = hashlib.sha256()
        identity.update(covariance.tobytes())
        identity.update(repr((self.version, self.stamp, self.command, self.component_id)).encode("utf-8"))
        for edge in transforms:
            identity.update(edge.edge_id.encode("utf-8"))
            identity.update(edge.translation.tobytes())
            identity.update(edge.rotation_wxyz.tobytes())
            if edge.bingham is not None:
                identity.update(edge.bingham.tobytes())
        object.__setattr__(self, "state_id", identity.hexdigest())

    @classmethod
    def from_independent(cls, transforms, **metadata):
        """Use only when the input explicitly declares a product prior."""
        transforms = tuple(transforms)
        covariance = np.zeros((6*len(transforms), 6*len(transforms)))
        for i, edge in enumerate(transforms):
            covariance[6*i:6*i+3, 6*i:6*i+3] = edge.translation_covariance
            if edge.bingham is not None:
                covariance[6*i+3:6*i+6, 6*i+3:6*i+6] = bingham_rotation_coordinates(edge.bingham)[1]
        return cls(transforms, covariance, **metadata)

    @property
    def edge_ids(self):
        return tuple(edge.edge_id for edge in self.transforms)


def conditional_total_covariance(covariance, jacobian, split=6):
    """Evaluate the conditional within/between terms on the real support.

    The sum below, rather than a separate J P J^T evaluation, is the actual
    covariance returned to queries.  The direct expression is a check.
    """
    covariance = np.asarray(covariance, dtype=float)
    jacobian = np.asarray(jacobian, dtype=float)
    n = covariance.shape[0]
    split = min(max(int(split), 0), n)
    p11 = covariance[:split, :split]
    p21 = covariance[split:, :split]
    inverse, _, _ = support_inverse(p11, name="conditioning covariance")
    regression = p21 @ inverse
    conditional = covariance[split:, split:] - regression @ p11 @ regression.T
    conditional = 0.5 * (conditional + conditional.T)
    support_inverse(conditional, name="conditional covariance")
    j1, j2 = jacobian[:, :split], jacobian[:, split:]
    within = j2 @ conditional @ j2.T
    conditional_sensitivity = j1 + j2 @ regression
    between = conditional_sensitivity @ p11 @ conditional_sensitivity.T
    output = within + between
    direct = jacobian @ covariance @ jacobian.T
    if not np.allclose(output, direct, rtol=1e-8, atol=1e-12):
        raise ValueError("Conditional total covariance disagrees with J P J^T.")
    return _array(output), _array(within), _array(between)


@dataclass(frozen=True)
class PointQuery:
    mean: np.ndarray
    covariance: np.ndarray
    jacobian: np.ndarray
    cross_covariance: np.ndarray
    within: np.ndarray
    between: np.ndarray
    query_id: str
    frame: str
    stamp: float
    command: tuple
    version: int
    component_id: str
    mode: np.ndarray
    correction: np.ndarray
    rotation_wxyz: object = None
    approximation_order: int = 1
    state_id: str = ""

    def __post_init__(self):
        for name in ("mean", "covariance", "jacobian", "cross_covariance", "within", "between", "mode", "correction"):
            object.__setattr__(self, name, _array(getattr(self, name)))
        if self.rotation_wxyz is not None:
            object.__setattr__(self, "rotation_wxyz", _array(self.rotation_wxyz))

    @property
    def H(self):
        return self.jacobian

    @property
    def c(self):
        # JointTransformState always publishes a reexpressed, zero-centered chart.
        return self.mean


def _path(state, path):
    """Resolve variables once, cancelling adjacent inverse occurrences exactly."""
    indices = {edge.edge_id: i for i, edge in enumerate(state.transforms)}
    result = []
    for value in path:
        edge_id, inverse = (value, False) if isinstance(value, str) else value
        selected = (indices[edge_id], bool(inverse))
        if result and result[-1][0] == selected[0] and result[-1][1] != selected[1]:
            result.pop()
        else:
            result.append(selected)
    return tuple(result)


_GENERATORS = tuple(skew(v) for v in np.eye(3))
_SECOND_GENERATORS = np.array([
    [0.5 * (a @ b + b @ a) for b in _GENERATORS]
    for a in _GENERATORS
])


def _rotation_derivatives(state, index, inverse, with_second=True):
    edge = state.transforms[index]
    rotation = quat_to_rotmat(edge.rotation_wxyz)
    n = state.mean.size
    first = np.zeros((n, 3, 3))
    second = np.zeros((n, n, 3, 3)) if with_second else None
    if edge.bingham is not None:
        for a, generator_a in enumerate(_GENERATORS):
            ia = 6*index+3+a
            first[ia] = rotation @ generator_a
            if with_second:
                for b, generator_b in enumerate(_GENERATORS):
                    ib = 6*index+3+b
                    second[ia, ib] = 0.5 * rotation @ (generator_a @ generator_b + generator_b @ generator_a)
    if inverse:
        return (rotation.T, first.transpose(0, 2, 1),
                second.transpose(0, 1, 3, 2) if with_second else None)
    return rotation, first, second


def _point_jet(state, path, point, direction=False):
    """Analytic action derivatives plus separate order-one proxy mean.

    Each rotation acts on the *mode* of its downstream point when forming
    its own radial contraction.  Mean corrections are carried separately,
    so multiplying corrections can never introduce a P_i P_j term.
    The product rule evaluates the mixed-derivative contractions with the
    original full joint covariance, including repeated variables.
    """
    value = np.asarray(point, dtype=float).copy()
    jacobian = np.zeros((3, state.mean.size))
    correction = np.zeros(3)
    for index, inverse in reversed(path):
        edge = state.transforms[index]
        rotation, derivatives, _ = _rotation_derivatives(state, index, inverse,
                                                         with_second=False)
        input_value = value.copy()
        input_jacobian = jacobian.copy()
        if inverse and not direction:
            input_value -= edge.translation
            input_jacobian[:, 6*index:6*index+3] -= np.eye(3)
        output_mode = rotation @ input_value
        # Cov(Rv) from the same Bingham-derived common rotation block.
        own_jacobian = np.einsum("aij,j->ia", derivatives, input_value)
        own_covariance = own_jacobian @ state.covariance @ own_jacobian.T
        radius2 = float(input_value @ input_value)
        radial = np.zeros(3) if radius2 == 0.0 else -0.5 * np.trace(own_covariance) / radius2 * output_mode
        # Cross term of dR/dz_a and d(input)/dz_b.  This includes rotation /
        # translation and shared/repeated variables without independent copies.
        mixed = np.einsum("aij,jb,ab->i", derivatives, input_jacobian, state.covariance)
        correction = rotation @ correction + radial + mixed
        jacobian = rotation @ input_jacobian + own_jacobian
        value = output_mode
        if not inverse and not direction:
            value = value + edge.translation
            jacobian[:, 6*index:6*index+3] += np.eye(3)
    return value, correction, jacobian


def _query(state, value, correction, jacobian, query_id, frame, rotation_wxyz=None):
    covariance, within, between = conditional_total_covariance(state.covariance, jacobian)
    return PointQuery(value+correction, covariance, jacobian, state.covariance @ jacobian.T,
                      within, between, str(query_id), str(frame), state.stamp,
                      state.command, state.version, state.component_id, value,
                      correction, rotation_wxyz, state_id=state.state_id)


def point_query(state, path, point=(0.0, 0.0, 0.0), query_id="point", frame="base_link"):
    """Query an origin/fixed axis endpoint on the original correlated path."""
    value, correction, jacobian = _point_jet(state, _path(state, path), point)
    return _query(state, value, correction, jacobian, query_id, frame)


def direction_query(state, path, direction, query_id="direction", frame="base_link"):
    """Rotation-only action (e.g. gravity), retaining the same joint body P."""
    value, correction, jacobian = _point_jet(state, _path(state, path), direction, True)
    return _query(state, value, correction, jacobian, query_id, frame)


def linear_query(query, matrix, query_id=None):
    """An explicit linear query of a frozen output, e.g. fixed S² tangent axes."""
    matrix = np.asarray(matrix, dtype=float)
    return replace(query, mean=matrix @ query.mean, covariance=matrix @ query.covariance @ matrix.T,
                   jacobian=matrix @ query.jacobian, cross_covariance=query.cross_covariance @ matrix.T,
                   within=matrix @ query.within @ matrix.T, between=matrix @ query.between @ matrix.T,
                   mode=matrix @ query.mode, correction=matrix @ query.correction,
                   query_id=query.query_id if query_id is None else str(query_id))


def pose_query(state, path, query_id="pose", frame="base_link"):
    """Translation and a frozen modal right-rotation chart for a sensor.

    The intrinsic orientation query is not a Gaussian pose substituted for
    the endpoint queries used by visualization.
    """
    path = _path(state, path)
    value, correction, jacobian = _point_jet(state, path, (0.0, 0.0, 0.0))
    n = state.mean.size
    rotation = np.eye(3)
    first = np.zeros((n, 3, 3))
    # Only Q=sum_ab P_ab d2R/dz_a dz_b enters the orientation mean.
    # Contract the product rule as each edge is composed instead of carrying
    # the full n-by-n matrix Hessian and contracting it at the end.  The
    # first derivatives remain explicit, including shared/repeated variables.
    contracted_second = np.zeros((3, 3))
    for index, inverse in path:
        r, d, _ = _rotation_derivatives(state, index, inverse,
                                         with_second=False)
        contracted_second = contracted_second @ r
        if state.transforms[index].bingham is not None:
            coordinates = slice(6*index+3, 6*index+6)
            # dR for this edge is nonzero only on its three rotation
            # coordinates.  Both mixed product-rule terms retain every
            # covariance row, including body/marker and repeated-path blocks.
            cross = (state.covariance[:, coordinates]
                     + state.covariance[coordinates, :].T)
            weighted_first = np.tensordot(cross.T, first, axes=(1, 0))
            contracted_second += np.einsum("bij,bjk->ik", weighted_first,
                                           d[coordinates])
            local_second = np.einsum("ab,abij->ij",
                                    state.covariance[coordinates, coordinates],
                                    _SECOND_GENERATORS)
            # The derivative of an inverse rotation is the transpose of
            # the forward derivative, including its second derivative.
            local_second = local_second.T @ r if inverse else r @ local_second
            contracted_second += rotation @ local_second
        first = first @ r + rotation @ d
        rotation = rotation @ r
    def vee(matrix):
        return np.array([matrix[2, 1]-matrix[1, 2], matrix[0, 2]-matrix[2, 0], matrix[1, 0]-matrix[0, 1]]) * 0.5
    orientation_jacobian = np.column_stack([vee(rotation.T @ derivative) for derivative in first])
    # In log coordinates the symmetric quadratic matrix term has zero vee;
    # the remaining contraction includes actual shared-variable mixed terms.
    orientation_correction = 0.5 * vee(rotation.T @ contracted_second)
    return _query(state, np.concatenate((value, np.zeros(3))),
                  np.concatenate((correction, orientation_correction)),
                  np.vstack((jacobian, orientation_jacobian)), query_id, frame,
                  rotmat_to_quat(rotation))


def query_cross_covariance(state, left, right):
    if any(query.state_id != state.state_id for query in (left, right)):
        raise ValueError("Cross-output covariance requires the same joint state, component, command, and time.")
    return _array(left.jacobian @ state.covariance @ right.jacobian.T)


def _right_jacobian(vector):
    theta = float(np.linalg.norm(vector))
    cross = skew(vector)
    if theta < 1e-6:
        return np.eye(3) - 0.5*cross + (1.0/6.0)*cross@cross
    return np.eye(3) - (1.0-np.cos(theta))/theta**2*cross + (theta-np.sin(theta))/theta**3*cross@cross


def reexpress_state(state, mean, covariance, version=None, component_id=None):
    """Publish a simultaneous posterior and all its transported cross blocks."""
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    n = state.mean.size
    if mean.shape != (n,) or covariance.shape != (n, n):
        raise ValueError("Posterior dimensions differ from the frozen body state.")
    require_in_support(mean-state.mean, covariance_support(state.covariance),
                       "posterior body mean shift")
    support_inverse(covariance, name="posterior body covariance")
    chart = np.eye(n)
    for i, edge in enumerate(state.transforms):
        if edge.bingham is not None:
            chart[6*i+3:6*i+6, 6*i+3:6*i+6] = _right_jacobian(mean[6*i+3:6*i+6])
    transported = chart @ covariance @ chart.T
    transforms = []
    for i, edge in enumerate(state.transforms):
        rotation_mean = mean[6*i+3:6*i+6]
        quaternion = quat_mul(edge.rotation_wxyz, rotation_vector_to_quaternion(rotation_mean))
        rotation_covariance = transported[6*i+3:6*i+6, 6*i+3:6*i+6]
        transforms.append(BinghamTransform(
            edge.edge_id, edge.translation + mean[6*i:6*i+3],
            None if edge.bingham is None else bingham_from_rotation_coordinates(quaternion, rotation_covariance),
            transported[6*i:6*i+3, 6*i:6*i+3],
            quaternion if edge.bingham is None else None,
        ))
    return JointTransformState(tuple(transforms), transported,
                               state.version+1 if version is None else int(version),
                               state.stamp, state.command,
                               state.component_id if component_id is None else str(component_id))
