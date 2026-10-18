"""Coordinate representation is distinct from a Bayesian prior support."""
from dataclasses import replace
import pickle

import numpy as np
import pytest

from probtf.geometry import quat_mul, rotation_vector_to_quaternion, quat_to_rotmat
from probtf.spherical_law.first_order import JointTransformState, _right_jacobian
from probtf_observation_fusion_demo import body_native as module
from probtf_observation_fusion_demo.immutable_body_cache import ReexpressionCache
from probtf_observation_fusion_demo.async_workers import _freeze_arrays
from test_body_native import source


def reference_geometry():
    state=source()
    # A positive but very small translation axis is absent from the old
    # prior's numerical affine support. It remains an ordinary coordinate
    # of the chart and is NOT a known attachment/Dirac constraint.
    scaling=np.ones(18);scaling[0]=1e-6
    P=state.covariance*np.outer(scaling,scaling)
    transforms=list(state.transforms)
    transforms[0]=replace(transforms[0],translation_covariance=P[:3,:3],rotation_wxyz=None)
    return replace(state,transforms=tuple(transforms),covariance=P)


def foreign_law(reference):
    m=np.zeros(18);m[:12]=.04*np.sin(np.arange(12)+.3);m[0]=.2
    # This is a DIFFERENT law, not a posterior constrained to the reference
    # prior's support. Its actual full covariance/cross is supplied intact.
    P=source().covariance*.8
    return m,P


@pytest.mark.parametrize('backend',['native','python_reference'])
def test_other_law_outside_reference_support_materializes_without_changing_given_moments(monkeypatch,backend):
    monkeypatch.setenv('PROBTF_BODY_BACKEND',backend)
    reference=reference_geometry();m,P=foreign_law(reference)
    before=(reference.state_id,reference.covariance.tobytes(),m.copy(),P.copy())
    with pytest.raises(ValueError,match='support'):
        module.reexpress_state(reference,m,P)
    result=module.materialize_state_in_chart(reference,m,P,version=37,component_id='other-law')
    G=np.eye(18)
    for index,edge in enumerate(reference.transforms):
        if edge.bingham is not None:
            block=slice(6*index+3,6*index+6);G[block,block]=_right_jacobian(m[block])
    np.testing.assert_allclose(result.covariance,G@P@G.T,atol=2e-17,rtol=2e-13)
    for index,(old,new) in enumerate(zip(reference.transforms,result.transforms)):
        np.testing.assert_array_equal(new.translation,old.translation+m[6*index:6*index+3])
        q=quat_mul(old.rotation_wxyz,rotation_vector_to_quaternion(m[6*index+3:6*index+6]))
        np.testing.assert_allclose(quat_to_rotmat(new.rotation_wxyz),quat_to_rotmat(q),atol=2e-14)
    assert type(result) is JointTransformState
    assert (result.version,result.stamp,result.command,result.component_id)==(
        37,reference.stamp,reference.command,'other-law')
    assert result.covariance[0,0]>reference.covariance[0,0]*1e10
    assert result.covariance[3,9]!=0.
    assert before[:2]==(reference.state_id,reference.covariance.tobytes())
    np.testing.assert_array_equal(m,before[2]);np.testing.assert_array_equal(P,before[3])
    # Existing core types use the worker's ordinary post-pickle readonly
    # restoration; the new coordinate operation does not replace their IPC
    # contract or add a competing candidate-state class.
    restored=_freeze_arrays(pickle.loads(pickle.dumps(result)))
    for item in (result,restored):
        assert item.state_id==result.state_id
        assert not item.covariance.flags.writeable
        for edge in item.transforms:
            assert not edge.translation.flags.writeable
            assert not edge.rotation_wxyz.flags.writeable
    # Existing core validation of the resulting law itself still succeeds.
    rebuilt=JointTransformState(result.transforms,result.covariance,result.version,
        result.stamp,result.command,result.component_id)
    assert rebuilt.state_id==result.state_id


def test_native_and_explicit_reference_have_the_same_geometry_covariance_and_queries(monkeypatch):
    reference=reference_geometry();m,P=foreign_law(reference)
    results=[]
    for backend in ('native','python_reference'):
        monkeypatch.setenv('PROBTF_BODY_BACKEND',backend)
        result=module.materialize_state_in_chart(reference,m,P)
        results.append((result,module.pose_query(result,(('tool',True),('link_1',True),('link_0',True)))))
    for name in ('mean','mode','correction','covariance','cross_covariance','jacobian'):
        np.testing.assert_allclose(getattr(results[0][1],name),getattr(results[1][1],name),
                                   atol=2e-13,rtol=3e-10)
    np.testing.assert_allclose(results[0][0].covariance,results[1][0].covariance,atol=3e-17)


@pytest.mark.parametrize('backend',['native','python_reference'])
def test_existing_strict_path_is_identical_when_prior_support_condition_is_satisfied(monkeypatch,backend):
    monkeypatch.setenv('PROBTF_BODY_BACKEND',backend)
    state=source();mean=np.zeros(18);mean[:12]=.03*np.sin(np.arange(12)+.4)
    strict=module.reexpress_state(state,mean,state.covariance*.7,version=30,component_id='same-law')
    chart=module.materialize_state_in_chart(state,mean,state.covariance*.7,version=30,component_id='same-law')
    assert strict.state_id==chart.state_id
    np.testing.assert_array_equal(strict.covariance,chart.covariance)


@pytest.mark.parametrize('backend',['native','python_reference'])
def test_cache_never_reuses_materialized_other_law_as_an_accepted_strict_posterior(monkeypatch,backend):
    monkeypatch.setenv('PROBTF_BODY_BACKEND',backend)
    cache=ReexpressionCache();state=reference_geometry();mean,P=foreign_law(state)
    actual=cache.materialize(state,mean,P)
    assert cache.materialize(state,mean,P) is actual
    with pytest.raises(ValueError,match='support'):
        cache.reexpress(state,mean,P)
    assert cache.statistics()['hits']==1
    assert cache.statistics()['entries']==1
    # Conversely a valid strict cached state cannot bypass the separate
    # common-chart cache key/operation or alter its numeric output.
    valid=source();mean=np.zeros(18)
    strict=cache.reexpress(valid,mean,valid.covariance)
    chart=cache.materialize(valid,mean,valid.covariance)
    assert chart is not strict and chart.state_id==strict.state_id


@pytest.mark.parametrize('backend',['native','python_reference'])
@pytest.mark.parametrize('failure',['mean_nonfinite','covariance_nonfinite','indefinite','fixed_rotation','singular_rotation'])
def test_new_api_preserves_actual_covariance_and_fixed_rotation_failures(monkeypatch,backend,failure):
    monkeypatch.setenv('PROBTF_BODY_BACKEND',backend)
    state=reference_geometry();m,P=foreign_law(state)
    if failure=='mean_nonfinite':m[0]=np.nan
    elif failure=='covariance_nonfinite':P[0,0]=np.inf
    elif failure=='indefinite':P[0,0]=-1.
    elif failure=='fixed_rotation':P[15:18,15:18]=np.eye(3)*.1
    else:P[3:6,:]=0.;P[:,3:6]=0.
    with pytest.raises((ValueError,np.linalg.LinAlgError)):
        module.materialize_state_in_chart(state,m,P)


def test_native_extension_error_never_silently_falls_back_to_reference(monkeypatch):
    monkeypatch.setenv('PROBTF_BODY_BACKEND','native')
    def absent():raise RuntimeError('common-chart native extension unavailable')
    monkeypatch.setattr(module,'_materialization_library',absent)
    state=reference_geometry();m,P=foreign_law(state)
    with pytest.raises(RuntimeError,match='unavailable'):
        module.materialize_state_in_chart(state,m,P)
