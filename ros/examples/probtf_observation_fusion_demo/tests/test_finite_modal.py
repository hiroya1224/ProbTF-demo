"""Finite geometric values/derivatives against independent deterministic paths."""
from dataclasses import FrozenInstanceError
import pickle

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from probtf.geometry import quat_to_rotmat
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
    pose_query as core_pose_query,
)
from probtf.spherical_law.support import covariance_support
from probtf_observation_fusion_demo import finite_modal as module


def source():
    scales = np.array([.002, .003, .004, .06, .07, .08]*2)
    direction = np.cos(np.arange(12)+.2)
    P = np.zeros((18, 18))
    P[:12, :12] = (np.eye(12)+.1*np.outer(direction, direction))*np.outer(scales, scales)
    edges = []
    for i, name in enumerate(('a', 'b')):
        scipy_q = Rotation.from_euler('xyz', [.2+i*.1, -.4, .25]).as_quat()
        q = scipy_q[[3, 0, 1, 2]]
        edges.append(BinghamTransform(name, [.2, .1*i, -.1],
            bingham_from_rotation_coordinates(q, P[6*i+3:6*i+6, 6*i+3:6*i+6]),
            P[6*i:6*i+3, 6*i:6*i+3]))
    edges.append(BinghamTransform('fixed', [.12, -.03, .04],
                                  rotation_wxyz=[1., 0., 0., 0.]))
    return JointTransformState(tuple(edges), P, version=8, stamp=1.25,
                               command=(.2, -.3), component_id='frozen-original')


PATHS = [
    ('a', 'b', 'fixed'),
    (('fixed', True), ('b', True), ('a', True)),
    (('a', False), ('b', True), ('a', False), ('fixed', False)),
    (('a', False), ('b', False), ('b', True), ('a', True)),
    (),
]


def independent_matrix(state, path, delta):
    """SciPy modal rotations and explicit inverse homogeneous matrices."""
    value = np.eye(4)
    for item in path:
        name, inverse = (item, False) if isinstance(item, str) else item
        index = state.edge_ids.index(name)
        edge = state.transforms[index]
        local = np.eye(4)
        local[:3, :3] = (Rotation.from_quat(edge.rotation_wxyz[[1, 2, 3, 0]]).as_matrix()
                            @ Rotation.from_rotvec(delta[6*index+3:6*index+6]).as_matrix())
        local[:3, 3] = edge.translation+delta[6*index:6*index+3]
        value = value @ (np.linalg.inv(local) if inverse else local)
    return value


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('scale', [0., 1e-9, .4, 2.7])
def test_finite_value_and_analytic_Jacobian_match_independent_matrix_backend(monkeypatch, path, scale):
    state = source()
    delta = scale*np.sin(np.arange(18)+.4)
    original_P = state.covariance.copy()
    original_id = state.state_id
    monkeypatch.setenv('PROBTF_BODY_BACKEND', 'native')
    actual = module.finite_modal_path(state, path, delta)
    monkeypatch.setenv('PROBTF_BODY_BACKEND', 'python_reference')
    reference = module.finite_modal_path(state, path, delta)
    independent = independent_matrix(state, path, delta)
    np.testing.assert_allclose(actual.translation, independent[:3, 3], atol=3e-14)
    np.testing.assert_allclose(quat_to_rotmat(actual.rotation_wxyz), independent[:3, :3], atol=3e-14)
    np.testing.assert_allclose(actual.J_t, reference.J_t, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(actual.J_right_rot, reference.J_right_rot, atol=2e-13, rtol=2e-12)
    assert actual.backend == 'native' and reference.backend == 'python_reference'
    assert actual.source_state_id == actual.reference_state_id == original_id
    assert (actual.source_stamp, actual.source_command, actual.source_version,
            actual.source_component_id) == (state.stamp, state.command, state.version, state.component_id)
    np.testing.assert_array_equal(state.covariance, original_P)
    assert state.state_id == original_id
    np.testing.assert_array_equal(actual.delta, delta)


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('scale', [0., .4, 2.7])
def test_all_finite_delta_derivative_columns_match_central_differences(monkeypatch, path, scale):
    monkeypatch.setenv('PROBTF_BODY_BACKEND', 'native')
    state = source()
    delta = scale*np.sin(np.arange(18)+.4)
    value = module.finite_modal_path(state, path, delta)
    base_rotation = quat_to_rotmat(value.rotation_wxyz)
    Jt = np.zeros_like(value.J_t)
    Jw = np.zeros_like(value.J_right_rot)
    step = 2e-6
    for j in range(delta.size):
        plus, minus = delta.copy(), delta.copy()
        plus[j] += step
        minus[j] -= step
        p = independent_matrix(state, path, plus)
        m = independent_matrix(state, path, minus)
        Jt[:, j] = (p[:3, 3]-m[:3, 3])/(2*step)
        right = base_rotation.T @ ((p[:3, :3]-m[:3, :3])/(2*step))
        Jw[:, j] = .5*np.array([right[2, 1]-right[1, 2],
                                right[0, 2]-right[2, 0], right[1, 0]-right[0, 1]])
    np.testing.assert_allclose(value.J_t, Jt, atol=3e-9, rtol=3e-8)
    np.testing.assert_allclose(value.J_right_rot, Jw, atol=3e-9, rtol=3e-8)


def test_zero_geometry_matches_modal_core_query_on_full_prior_support(monkeypatch):
    monkeypatch.setenv('PROBTF_BODY_BACKEND', 'native')
    state = source()
    support = covariance_support(state.covariance)
    L = support.basis*np.sqrt(support.eigenvalues)
    assert np.all(L[12:] == 0.)
    value = module.finite_modal_path(state, state.edge_ids, np.zeros(18))
    core = core_pose_query(state, state.edge_ids)
    np.testing.assert_allclose(value.translation, core.mode[:3], atol=2e-15)
    np.testing.assert_allclose(value.J_t @ L, core.jacobian[:3] @ L, atol=2e-15)
    np.testing.assert_allclose(value.J_right_rot @ L, core.jacobian[3:] @ L, atol=2e-15)
    assert np.linalg.norm(value.translation-core.mean[:3]) > 1e-5
    # Fixed Dirac coordinates remain zero under the solver's support map,
    # even though the full geometry extension has their formal derivatives.
    assert np.linalg.norm(value.J_right_rot[:, 15:18]) > 0.
    assert np.all(L[15:18] == 0.)


def test_repeated_transform_inverse_cancels_values_and_all_shared_derivatives(monkeypatch):
    monkeypatch.setenv('PROBTF_BODY_BACKEND', 'native')
    state = source()
    value = module.finite_modal_path(state, PATHS[3], np.linspace(-2., 2., 18))
    np.testing.assert_array_equal(value.translation, np.zeros(3))
    np.testing.assert_array_equal(value.J_t, np.zeros((3, 18)))
    np.testing.assert_array_equal(value.J_right_rot, np.zeros((3, 18)))
    np.testing.assert_allclose(quat_to_rotmat(value.rotation_wxyz), np.eye(3), atol=0.)


def test_empty_source_is_exact_identity_and_output_stays_readonly_after_pickle(monkeypatch):
    monkeypatch.setenv('PROBTF_BODY_BACKEND', 'native')
    state = JointTransformState((), np.empty((0, 0)), stamp=2., command=())
    value = module.finite_modal_path(state, (), np.zeros(0))
    assert value.J_t.shape == value.J_right_rot.shape == (3, 0)
    for item in (value, pickle.loads(pickle.dumps(value))):
        np.testing.assert_array_equal(item.translation, np.zeros(3))
        assert item.source_state_id == state.state_id
        for array in (item.translation, item.rotation_wxyz, item.J_t, item.J_right_rot, item.delta):
            assert not array.flags.writeable
        with pytest.raises(FrozenInstanceError):
            item.source_stamp = 9.


@pytest.mark.parametrize('backend', ['native', 'python_reference'])
@pytest.mark.parametrize('bad', ['shape', 'nan', 'inf'])
def test_actual_invalid_delta_has_explicit_failure(monkeypatch, backend, bad):
    monkeypatch.setenv('PROBTF_BODY_BACKEND', backend)
    delta = np.zeros(17 if bad == 'shape' else 18)
    if bad == 'nan':
        delta[0] = np.nan
    if bad == 'inf':
        delta[0] = np.inf
    with pytest.raises(ValueError, match='dimension and be finite'):
        module.finite_modal_path(source(), ('a',), delta)


def test_native_failure_never_silently_switches_to_reference(monkeypatch):
    def unavailable():
        raise RuntimeError('native unavailable fixture')
    monkeypatch.setattr(module, '_finite_library', unavailable)
    monkeypatch.setenv('PROBTF_BODY_BACKEND', 'native')
    with pytest.raises(RuntimeError, match='native unavailable'):
        module.finite_modal_path(source(), ('a',), np.zeros(18))
    monkeypatch.setenv('PROBTF_BODY_BACKEND', 'python_reference')
    assert module.finite_modal_path(source(), ('a',), np.zeros(18)).backend == 'python_reference'
