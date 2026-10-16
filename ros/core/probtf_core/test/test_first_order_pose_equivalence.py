"""The contracted implementation must equal the pre-optimization dense jets.

The oracle below is the original analytic algorithm, kept independent of the
optimized derivative helpers. It explicitly forms every mixed derivative
before covariance contraction. These checks protect the approximation and
shared-variable semantics while removing unnecessary intermediate tensors.
"""

import numpy as np
import pytest

from probtf.geometry import axis_angle_to_quat, quat_to_rotmat, rotmat_to_quat, skew
import probtf.spherical_law.first_order as core


_GENERATORS = tuple(skew(value) for value in np.eye(3))


def _dense_rotation_derivatives(state, index, inverse):
    edge = state.transforms[index]
    rotation = quat_to_rotmat(edge.rotation_wxyz)
    n = state.mean.size
    first = np.zeros((n, 3, 3))
    second = np.zeros((n, n, 3, 3))
    if edge.bingham is not None:
        for a, generator_a in enumerate(_GENERATORS):
            ia = 6*index+3+a
            first[ia] = rotation @ generator_a
            for b, generator_b in enumerate(_GENERATORS):
                ib = 6*index+3+b
                second[ia, ib] = 0.5*rotation @ (
                    generator_a @ generator_b + generator_b @ generator_a)
    if inverse:
        return rotation.T, first.transpose(0, 2, 1), second.transpose(0, 1, 3, 2)
    return rotation, first, second


def _dense_point_jet(state, path, point, direction=False):
    value = np.asarray(point, dtype=float).copy()
    jacobian = np.zeros((3, state.mean.size))
    correction = np.zeros(3)
    for index, inverse in reversed(path):
        edge = state.transforms[index]
        rotation, derivatives, _ = _dense_rotation_derivatives(state, index, inverse)
        input_value = value.copy()
        input_jacobian = jacobian.copy()
        if inverse and not direction:
            input_value -= edge.translation
            input_jacobian[:, 6*index:6*index+3] -= np.eye(3)
        output_mode = rotation @ input_value
        own_jacobian = np.einsum("aij,j->ia", derivatives, input_value)
        own_covariance = own_jacobian @ state.covariance @ own_jacobian.T
        radius2 = float(input_value @ input_value)
        radial = (np.zeros(3) if radius2 == 0.0 else
                  -0.5*np.trace(own_covariance)/radius2*output_mode)
        mixed = np.einsum("aij,jb,ab->i", derivatives, input_jacobian, state.covariance)
        correction = rotation @ correction + radial + mixed
        jacobian = rotation @ input_jacobian + own_jacobian
        value = output_mode
        if not inverse and not direction:
            value = value + edge.translation
            jacobian[:, 6*index:6*index+3] += np.eye(3)
    return value, correction, jacobian


def _dense_pose_query(state, path):
    path = core._path(state, path)
    value, correction, jacobian = _dense_point_jet(state, path, (0., 0., 0.))
    n = state.mean.size
    rotation = np.eye(3)
    first = np.zeros((n, 3, 3))
    second = np.zeros((n, n, 3, 3))
    for index, inverse in path:
        r, d, dd = _dense_rotation_derivatives(state, index, inverse)
        new_second = (np.einsum("abij,jk->abik", second, r)
                      + np.einsum("aij,bjk->abik", first, d)
                      + np.einsum("bij,ajk->abik", first, d)
                      + np.einsum("ij,abjk->abik", rotation, dd))
        first = np.einsum("aij,jk->aik", first, r) + np.einsum("ij,ajk->aik", rotation, d)
        second = new_second
        rotation = rotation @ r
    def vee(matrix):
        return .5*np.array([matrix[2, 1]-matrix[1, 2], matrix[0, 2]-matrix[2, 0],
                            matrix[1, 0]-matrix[0, 1]])
    orientation_jacobian = np.column_stack([vee(rotation.T @ derivative) for derivative in first])
    orientation_correction = .5*vee(
        rotation.T @ np.einsum("ab,abij->ij", state.covariance, second))
    return core._query(state, np.concatenate((value, np.zeros(3))),
                       np.concatenate((correction, orientation_correction)),
                       np.vstack((jacobian, orientation_jacobian)), "equivalence", "root",
                       rotmat_to_quat(rotation))


@pytest.fixture(scope="module")
def correlated_state():
    # Six body edges, four exact attachments, and one uncertain marker: the
    # same 66-coordinate size as the real moving example. Dense deterministic
    # factors produce translation/rotation, interlink and marker/body blocks.
    names = ["link_{}".format(i+1) for i in range(6)]
    names += ["tool", "rod", "tip", "camera", "marker"]
    active = [6*i+j for i in list(range(6))+[10] for j in range(6)]
    rows = np.arange(1., len(active)+1.)[:, None]
    columns = np.arange(1., len(active)+1.)[None, :]
    factor = np.sin(rows*columns*.17)/np.sqrt(len(active))
    covariance = np.zeros((66, 66))
    covariance[np.ix_(active, active)] = .001*(factor @ factor.T + np.eye(len(active)))
    transforms = []
    for i, name in enumerate(names):
        translation = [.07*(i+1), .03*np.cos(i), -.02*np.sin(i)]
        quaternion = axis_angle_to_quat([1.+i*.1, .5-i*.03, -.7], .12*(i+1))
        if i in list(range(6))+[10]:
            matrix = core.bingham_from_rotation_coordinates(
                quaternion, covariance[6*i+3:6*i+6, 6*i+3:6*i+6])
            transforms.append(core.BinghamTransform(name, translation, matrix,
                covariance[6*i:6*i+3, 6*i:6*i+3]))
        else:
            transforms.append(core.BinghamTransform(name, translation,
                                                    rotation_wxyz=quaternion))
    return core.JointTransformState(tuple(transforms), covariance,
        version=17, stamp=23.75, command=(.1, -.2, .3, .2, -.1, .4), component_id="correlated")


BODY = tuple("link_{}".format(i+1) for i in range(6))
HAND = BODY + ("tool", "rod", "tip")
PATHS = [
    HAND,
    (("marker", True),) + HAND,
    tuple((name, True) for name in reversed(HAND)) + ("marker",),
    (("link_2", False), ("link_3", False), ("link_2", True),
     ("link_5", False), ("link_3", True)),
    (("marker", False), ("marker", True)),
    (),
]


def _assert_query_equivalent(actual, expected):
    for name in ("mean", "covariance", "jacobian", "cross_covariance", "within", "between",
                 "mode", "correction"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                   rtol=2e-12, atol=3e-14, err_msg=name)
    for name in ("query_id", "frame", "stamp", "command", "version", "component_id",
                 "state_id", "approximation_order"):
        assert getattr(actual, name) == getattr(expected, name)
    if actual.rotation_wxyz is not None:
        np.testing.assert_allclose(actual.rotation_wxyz, expected.rotation_wxyz, atol=1e-14)


@pytest.mark.parametrize("path", PATHS)
def test_pose_query_matches_original_dense_jet_with_shared_variables(correlated_state, path):
    state = correlated_state
    expected = _dense_pose_query(state, path)
    actual = core.pose_query(state, path, query_id="equivalence", frame="root")
    _assert_query_equivalent(actual, expected)


@pytest.mark.parametrize("path", PATHS[:4])
@pytest.mark.parametrize("direction", [False, True])
def test_point_and_direction_keep_original_mean_cross_contractions(correlated_state, path, direction):
    state = correlated_state
    value, correction, jacobian = _dense_point_jet(
        state, core._path(state, path), [.13, -.21, .34], direction)
    expected = core._query(state, value, correction, jacobian, "equivalence", "root")
    function = core.direction_query if direction else core.point_query
    actual = function(state, path, [.13, -.21, .34], query_id="equivalence", frame="root")
    _assert_query_equivalent(actual, expected)


def test_oracle_exercises_nonzero_shared_mixed_orientation_correction(correlated_state):
    state = correlated_state
    result = _dense_pose_query(state, PATHS[1])
    assert np.linalg.norm(state.covariance[:36, 60:]) > 1e-5
    assert np.linalg.norm(result.correction[3:]) > 1e-5
    assert np.linalg.norm(result.covariance[:3, 3:]) > 1e-5
