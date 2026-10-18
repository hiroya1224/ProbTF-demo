"""Supported Schur algebra stays finite without regularizing the source law."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
    _point_jet, _path,
)
from probtf.spherical_law.support import support_inverse
from probtf_observation_fusion_demo import body_native as native


def ill_conditioned_source():
    # An explicit PSD block construction: first covariance A, regression B,
    # and independent residual C. No random draws or eigenvalue adjustment.
    rotation = Rotation.from_euler('xyz', [.3, -.5, .7]).as_matrix()
    first = np.zeros((6, 6))
    first[3:, 3:] = rotation @ np.diag([4.e-6, 3.e-5, .044]) @ rotation.T
    regression = np.zeros((30, 6))
    i, j = np.arange(30)[:, None]+1, np.arange(3)[None, :]+1
    regression[:, 3:] = np.sin(i*j*.37)+np.cos((i+1)*(j+2)*.19)
    conditional = np.zeros((30, 30))
    conditional[np.arange(8), np.arange(8)] = 3.e-9
    covariance = np.block([[first, first @ regression.T],
        [regression @ first, regression @ first @ regression.T+conditional]])
    edges = []
    for index in range(6):
        part = slice(6*index, 6*index+3)
        angle = slice(6*index+3, 6*index+6)
        quaternion = np.array([1., 0., 0., 0.])
        edges.append(BinghamTransform('link_'+str(index),
            np.array([.12+.01*index, .03*np.sin(index), .02*np.cos(index)]),
            bingham_from_rotation_coordinates(quaternion, covariance[angle, angle]),
            covariance[part, part]))
    return JointTransformState(tuple(edges), covariance, version=21, stamp=12.5,
        command=(.1, -.2, .3, -.4, .5, -.6), component_id='deterministic_PSD_Schur_regression')


@pytest.mark.parametrize('kind', ('point', 'direction', 'pose'))
def test_ill_conditioned_supported_schur_preserves_full_covariance_cross_and_moments(monkeypatch, kind):
    monkeypatch.setenv('PROBTF_BODY_BACKEND', 'native')
    state = ill_conditioned_source()
    original = state.covariance.tobytes()
    path, point = state.edge_ids, np.array([.07, -.02, .03])
    if kind == 'point':
        query = native.point_query(state, path, point)
    elif kind == 'direction':
        query = native.direction_query(state, path, point)
    else:
        point = np.zeros(3)
        query = native.pose_query(state, path)
    # The untouched reference jet supplies the same mode, O(P) correction
    # and full Jacobian without invoking its subtractive covariance audit.
    mode, correction, jacobian = _point_jet(state, _path(state, path), point,
                                            direction=(kind == 'direction'))
    np.testing.assert_allclose(query.mode[:3], mode, atol=2.e-13, rtol=1.e-12)
    np.testing.assert_allclose(query.correction[:3], correction, atol=2.e-13, rtol=1.e-12)
    np.testing.assert_allclose(query.jacobian[:3], jacobian, atol=2.e-13, rtol=1.e-12)
    direct = query.jacobian @ state.covariance @ query.jacobian.T
    np.testing.assert_allclose(query.covariance, direct, atol=4.e-13, rtol=2.e-10)
    np.testing.assert_array_equal(query.covariance, query.within+query.between)
    np.testing.assert_allclose(query.cross_covariance,
        state.covariance @ query.jacobian.T, atol=2.e-13, rtol=1.e-12)
    assert np.all(np.isfinite(query.covariance))
    assert state.covariance.tobytes() == original
    assert query.state_id == state.state_id
    # This source exposes the old operation's cancellation: constructing
    # and multiplying back its explicit inverse produces a large spurious
    # negative residual eigenvalue despite the declared PSD construction.
    first, cross = state.covariance[:6, :6], state.covariance[6:, :6]
    regression = cross @ support_inverse(first)[0]
    subtractive = state.covariance[6:, 6:]-regression @ first @ regression.T
    assert np.linalg.eigvalsh(.5*subtractive+.5*subtractive.T)[0] < -1.e-14


@pytest.mark.parametrize('invalid', ('nonfinite', 'negative_conditioning_covariance'))
def test_required_native_input_operations_still_report_actual_failures(monkeypatch, invalid):
    monkeypatch.setenv('PROBTF_BODY_BACKEND', 'native')
    state = ill_conditioned_source()
    original_inputs = native._inputs
    def inputs(source):
        translation, quaternion, random, covariance = original_inputs(source)
        covariance = covariance.copy()
        if invalid == 'nonfinite':
            covariance[3, 3] = np.nan
        else:
            covariance[3, 3] = -1.
        return translation, quaternion, random, covariance
    monkeypatch.setattr(native, '_inputs', inputs)
    with pytest.raises(ValueError, match=('finite' if invalid == 'nonfinite' else 'negative eigenvalue')):
        native.point_query(state, state.edge_ids)
