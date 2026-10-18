"""Same-law C++ body arithmetic versus untouched Python core references."""
from dataclasses import replace
import pickle

import numpy as np
import pytest
from probtf.geometry import rotation_vector_to_quaternion, quat_to_rotmat
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
    reexpress_state as core_reexpress, point_query as core_point,
    pose_query as core_pose, direction_query as core_direction,
)
from probtf_observation_fusion_demo import body_native as native


def source():
    scales=np.array([.002,.003,.0025,.08,.07,.09]*2)
    v=np.sin(np.arange(12)+.4)
    covariance=np.zeros((18,18));covariance[:12,:12]=(np.eye(12)+.08*np.outer(v,v))*np.outer(scales,scales)
    edges=[]
    for i in range(2):
        q=rotation_vector_to_quaternion([.1+.2*i,-.3+.1*i,.2])
        edges.append(BinghamTransform('link_'+str(i),np.array([.2,.04*i,.1]),
            bingham_from_rotation_coordinates(q,covariance[6*i+3:6*i+6,6*i+3:6*i+6]),
            covariance[6*i:6*i+3,6*i:6*i+3]))
    edges.append(BinghamTransform('tool',np.array([.13,0,.02]),
        rotation_wxyz=rotation_vector_to_quaternion([.2,0.,0.])))
    return JointTransformState(tuple(edges),covariance,version=9,stamp=1.25,
        command=(.2,-.3),component_id='full-cross-source')


def compare_query(actual,reference):
    for name in ('mean','mode','correction','covariance','within','between','jacobian','cross_covariance'):
        np.testing.assert_allclose(getattr(actual,name),getattr(reference,name),atol=2e-12,rtol=3e-10)
        assert not getattr(actual,name).flags.writeable
    assert actual.state_id==reference.state_id
    assert (actual.version,actual.stamp,actual.command,actual.component_id)==(reference.version,reference.stamp,reference.command,reference.component_id)
    if reference.rotation_wxyz is not None:
        np.testing.assert_allclose(quat_to_rotmat(actual.rotation_wxyz),quat_to_rotmat(reference.rotation_wxyz),atol=2e-12)


@pytest.mark.parametrize('path',[
    ('link_0','link_1','tool'),
    (('tool',True),('link_1',True),('link_0',True)),
    (('link_0',False),('link_1',False),('link_1',True),('tool',False)),
    (('link_0',False),('link_1',True),('link_0',False)),
    (),
])
@pytest.mark.parametrize('kind',['point','pose','direction'])
def test_correlated_forward_inverse_repeated_variable_queries(monkeypatch,path,kind):
    monkeypatch.setenv('PROBTF_BODY_BACKEND','native');state=source()
    if kind=='point':a=native.point_query(state,path,[.07,-.05,.12]);b=core_point(state,path,[.07,-.05,.12])
    elif kind=='direction':a=native.direction_query(state,path,[.2,-.3,-.7]);b=core_direction(state,path,[.2,-.3,-.7])
    else:a=native.pose_query(state,path);b=core_pose(state,path)
    compare_query(a,b)


def test_reexpression_preserves_full_cross_and_existing_immutable_types(monkeypatch):
    monkeypatch.setenv('PROBTF_BODY_BACKEND','native');state=source();mean=np.zeros(18);mean[:12]=.02*np.sin(np.arange(12));P=.7*state.covariance
    a=native.reexpress_state(state,mean,P,version=27,component_id='posterior');b=core_reexpress(state,mean,P,version=27,component_id='posterior')
    assert type(a) is JointTransformState
    assert all(type(e) is BinghamTransform for e in a.transforms)
    assert a.version==27 and a.component_id=='posterior'
    np.testing.assert_allclose(a.covariance,b.covariance,atol=1e-15,rtol=1e-12)
    for x,y in zip(a.transforms,b.transforms):
        np.testing.assert_array_equal(x.translation,y.translation)
        np.testing.assert_allclose(x.bingham,y.bingham,rtol=1e-10,atol=1e-10) if x.bingham is not None else None
        np.testing.assert_allclose(quat_to_rotmat(x.rotation_wxyz),quat_to_rotmat(y.rotation_wxyz),atol=1e-12)
    # Re-run untouched core validators over the returned objects.
    rebuilt=JointTransformState(a.transforms,a.covariance,a.version,a.stamp,a.command,a.component_id)
    assert rebuilt.state_id==a.state_id
    assert pickle.loads(pickle.dumps(a)).state_id==a.state_id
    compare_query(native.pose_query(a,a.edge_ids),core_pose(a,a.edge_ids))
    np.testing.assert_array_equal(state.mean,np.zeros(18))
    assert not a.covariance.flags.writeable
    assert abs(a.covariance[1,9])>0.


@pytest.mark.parametrize('problem',['nonfinite','negative','unsupported','rotation_rank','too_broad'])
def test_native_reexpression_retains_actual_core_domain_failures(monkeypatch,problem):
    monkeypatch.setenv('PROBTF_BODY_BACKEND','native');s=source();m=np.zeros(18);P=s.covariance.copy()
    if problem=='nonfinite':m[0]=np.nan
    elif problem=='negative':P[0,0]=-1.
    elif problem=='unsupported':m[-1]=.2
    elif problem=='rotation_rank':P[3:6,:]=0.;P[:,3:6]=0.
    elif problem=='too_broad':P[:12,:12]=np.eye(12)*.2
    with pytest.raises(ValueError):core_reexpress(s,m,P)
    with pytest.raises(ValueError):native.reexpress_state(s,m,P)


def test_explicit_python_comparison_backend_and_unknown_backend(monkeypatch):
    s=source();monkeypatch.setenv('PROBTF_BODY_BACKEND','python_reference')
    a=native.point_query(s,s.edge_ids);b=core_point(s,s.edge_ids)
    np.testing.assert_array_equal(a.mean,b.mean);np.testing.assert_array_equal(a.covariance,b.covariance)
    monkeypatch.setenv('PROBTF_BODY_BACKEND','invalid')
    with pytest.raises(ValueError):native.point_query(s,s.edge_ids)


def test_singular_translation_support_keeps_all_cross_and_fixed_edge(monkeypatch):
    monkeypatch.setenv('PROBTF_BODY_BACKEND', 'native')
    # One translation variable is shared by both edges, including the edge
    # whose rotation is exactly known. The random rotation has full support.
    loading = np.zeros((12, 4))
    loading[:3, 0] = [.002, -.001, .003]
    loading[6:9, 0] = [-.004, .002, .001]
    loading[3:6, 0] = [.003, -.002, .001]
    loading[3:6, 1:] = np.diag([.04, .05, .06])
    P = loading @ loading.T
    q = rotation_vector_to_quaternion([.2, -.1, .3])
    edges = (
        BinghamTransform('uncertain', [.1, .2, .3],
            bingham_from_rotation_coordinates(q, P[3:6, 3:6]), P[:3, :3]),
        BinghamTransform('known_rotation', [.3, .1, -.2],
            translation_covariance=P[6:9, 6:9], rotation_wxyz=q),
    )
    state = JointTransformState(edges, P, stamp=.3, command=(.2,),
                                component_id='singular-shared-variable')
    assert np.linalg.matrix_rank(P) == 4
    mean = loading @ np.array([.2, -.1, .3, -.2])
    native_state = native.reexpress_state(state, mean, .6*P)
    reference = core_reexpress(state, mean, .6*P)
    np.testing.assert_allclose(native_state.covariance, reference.covariance,
                               rtol=1e-11, atol=1e-15)
    assert native_state.transforms[1].bingham is None
    assert np.count_nonzero(native_state.covariance[9:, :]) == 0
    assert abs(native_state.covariance[0, 6]) > 0
    for path in (state.edge_ids, (('known_rotation', True), ('uncertain', True))):
        compare_query(native.pose_query(state, path), core_pose(state, path))
        compare_query(native.point_query(state, path, [.1, .2, -.3]),
                      core_point(state, path, [.1, .2, -.3]))
    # A shift orthogonal to the same affine support must still fail.
    bad_mean = mean.copy()
    bad_mean[7] += .01
    with pytest.raises(ValueError):
        core_reexpress(state, bad_mean, P)
    with pytest.raises(ValueError):
        native.reexpress_state(state, bad_mean, P)


@pytest.mark.parametrize('empty', [False, True])
def test_fully_deterministic_and_empty_laws_preserve_immutable_identity(monkeypatch, empty):
    monkeypatch.setenv('PROBTF_BODY_BACKEND', 'native')
    edges = () if empty else (BinghamTransform(
        'known', [.2, -.3, .1], rotation_wxyz=[1., 0., 0., 0.]),)
    state = JointTransformState(edges, np.zeros((6*len(edges), 6*len(edges))),
                                stamp=1.3, command=(.1,), component_id='known')
    n = state.mean.size
    actual = native.reexpress_state(state, np.zeros(n), np.zeros((n, n)), version='17')
    reference = core_reexpress(state, np.zeros(n), np.zeros((n, n)), version='17')
    assert actual.version == 17
    assert actual.state_id == reference.state_id
    assert JointTransformState(actual.transforms, actual.covariance, actual.version,
        actual.stamp, actual.command, actual.component_id).state_id == actual.state_id
    compare_query(native.point_query(actual, actual.edge_ids, [.1, .2, .3]),
                  core_point(actual, actual.edge_ids, [.1, .2, .3]))
    if empty:
        with pytest.raises(ValueError):
            core_pose(actual, actual.edge_ids)
        with pytest.raises(ValueError):
            native.pose_query(actual, actual.edge_ids)
    else:
        compare_query(native.pose_query(actual, actual.edge_ids), core_pose(actual, actual.edge_ids))
