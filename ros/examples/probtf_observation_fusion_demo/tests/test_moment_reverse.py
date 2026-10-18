"""Same core moments and analytic reverse gradients, not FD optimization."""
from dataclasses import FrozenInstanceError
import pickle

import numpy as np
import pytest

from probtf.geometry import (rotation_vector_to_quaternion, quat_to_rotmat,
                             rotmat_to_quat, quaternion_to_rotation_vector,
                             tangent_basis)
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
    reexpress_state, point_query, pose_query, direction_query,
)
from probtf_observation_fusion_demo import moment_reverse as module
from probtf_observation_fusion_demo.sensors import _rotation_log_right_jacobian_inverse


def source():
    scales = np.array([.005, .004, .003, .07, .08, .06]*2)
    shared = np.sin(np.arange(12)+.3)
    P = np.zeros((18,18))
    P[:12,:12] = (np.eye(12)+.13*np.outer(shared,shared))*np.outer(scales,scales)
    transforms=[]
    for i in range(2):
        q=rotation_vector_to_quaternion([.13+i*.2,-.26,.31-i*.1])
        transforms.append(BinghamTransform('edge'+str(i),[.2,.04+i*.08,-.13],
            bingham_from_rotation_coordinates(q,P[6*i+3:6*i+6,6*i+3:6*i+6]),
            P[6*i:6*i+3,6*i:6*i+3]))
    transforms.append(BinghamTransform('fixed',[.1,-.02,.03],
        rotation_wxyz=rotation_vector_to_quaternion([.1,.2,-.3])))
    return JointTransformState(tuple(transforms),P,version=3,stamp=4.5,
                               command=(.25,-.3),component_id='original-prior')


PATHS=[('edge0','edge1','fixed'),
       (('fixed',True),('edge1',True),('edge0',True)),
       (('edge0',False),('edge1',True),('edge0',False)), ()]
NATIVE=rotation_vector_to_quaternion([.23,-.11,.44])
POINT=np.array([.2,-.3,.7])
BASIS=tangent_basis(np.array([.2,-.3,.7])/np.linalg.norm([.2,-.3,.7])).T


def core_moments(state,path,m,P,kind,projection=None):
    trial=reexpress_state(state,m,P)
    if kind=='pose':
        query=pose_query(trial,path)
        phi=quaternion_to_rotation_vector(rotmat_to_quat(
            quat_to_rotmat(NATIVE).T@quat_to_rotmat(query.rotation_wxyz)))
        D=np.eye(6);D[3:,3:]=_rotation_log_right_jacobian_inverse(phi)
        mu=D@query.mean+np.r_[np.zeros(3),phi]
        S=D@query.covariance@D.T
    elif kind=='point':
        query=point_query(trial,path,POINT);mu,S=query.mean,query.covariance
    else:
        query=direction_query(trial,path,POINT);mu,S=query.mean,query.covariance
    if projection is not None:mu,S=projection@mu,projection@S@projection.T
    return mu,S


@pytest.mark.parametrize('path',PATHS)
@pytest.mark.parametrize('kind',['pose','point','direction'])
@pytest.mark.parametrize('shift',[0.,.18])
def test_values_match_existing_reexpression_and_core_O_P_full_cross(path,kind,shift):
    state=source();m=np.zeros(18);m[:12]=shift*np.sin(np.arange(12)+.4)
    P=.7*state.covariance
    projection=BASIS if kind=='direction' else None
    actual=module.evaluate_moment_query(state,path,m,P,kind=kind,
        point_or_direction=POINT,reference_rotation_wxyz=NATIVE,projection_matrix=projection)
    mu,S=core_moments(state,path,m,P,kind,projection)
    np.testing.assert_allclose(actual.mean,mu,atol=2e-14,rtol=2e-12)
    np.testing.assert_allclose(actual.covariance,S,atol=3e-15,rtol=3e-11)
    assert actual.source_state_id==state.state_id
    assert (actual.source_stamp,actual.source_command,actual.source_version,
            actual.source_component_id)==(4.5,(.25,-.3),3,'original-prior')
    assert actual.expected_negative_log_likelihood is None
    assert actual.diagnostics['O_P_mean_correction']
    assert not actual.diagnostics['numerical_parameter_differences']


@pytest.mark.parametrize('kind',['pose','point','direction'])
@pytest.mark.parametrize('path',PATHS[:3])
def test_all_supported_mean_and_symmetric_covariance_gradients_match_core_FD(kind,path):
    state=source();m=np.zeros(18);m[:12]=.16*np.sin(np.arange(12)+.4)
    P=.65*state.covariance
    projection=BASIS if kind=='direction' else None
    dim=2 if kind=='direction' else (6 if kind=='pose' else 3)
    a=np.cos(np.arange(dim)+.6)
    b=np.sin(np.arange(dim*dim).reshape(dim,dim)+.3);b=.5*(b+b.T)
    evaluator=module.MomentQueryEvaluator(state,path,kind=kind,point_or_direction=POINT,
        reference_rotation_wxyz=NATIVE,projection_matrix=projection)
    actual=evaluator.evaluate(m,P,adjoint_mean=a,adjoint_covariance=b)
    def objective(mm,PP):
        mu,S=core_moments(state,path,mm,PP,kind,projection)
        return float(a@mu+np.sum(b*S))
    step=2e-6
    for j in range(12):
        d=np.zeros(18);d[j]=step
        fd=(objective(m+d,P)-objective(m-d,P))/(2*step)
        assert actual.gradient_mean[j]==pytest.approx(fd,abs=3e-9,rel=1e-7)
    # Every covariance entry on the non-Dirac joint support is varied.
    # The FD step is scaled so the covariance remains positive definite.
    scales=np.sqrt(np.diag(P))
    for i in range(12):
        for j in range(i+1):
            d=np.zeros((18,18));h=step*scales[i]*scales[j]
            d[i,j]=h;d[j,i]=h
            fd=(objective(m,P+d)-objective(m,P-d))/(2*h)
            analytic=actual.gradient_covariance[i,j]*(1 if i==j else 2)
            assert analytic==pytest.approx(fd,abs=2e-5,rel=2e-5)
    np.testing.assert_array_equal(actual.gradient_covariance,actual.gradient_covariance.T)


@pytest.mark.parametrize('kind',['pose','direction'])
def test_one_call_expected_likelihood_matches_explicit_adjoint_and_finite_joint_directions(kind):
    state=source();m=np.zeros(18);m[:12]=.12*np.sin(np.arange(12)+.4)
    P=.8*state.covariance
    projection=BASIS if kind=='direction' else None
    dim=2 if kind=='direction' else 6
    y=.2*np.cos(np.arange(dim));W=np.diag(np.arange(dim)+1.)
    evaluator=module.MomentQueryEvaluator(state,PATHS[1],kind=kind,
        point_or_direction=POINT,reference_rotation_wxyz=NATIVE,projection_matrix=projection)
    actual=evaluator.evaluate(m,P,likelihood_mean=y,likelihood_precision=W,
                              likelihood_log_normalizer=3.25)
    explicit=evaluator.evaluate(m,P,adjoint_mean=W@(actual.mean-y),adjoint_covariance=.5*W)
    np.testing.assert_allclose(actual.gradient_mean,explicit.gradient_mean,atol=2e-14)
    np.testing.assert_allclose(actual.gradient_covariance,explicit.gradient_covariance,atol=3e-14)
    expected=.5*((actual.mean-y)@W@(actual.mean-y)+np.trace(W@actual.covariance))
    assert actual.cost==pytest.approx(expected,abs=3e-15)
    assert actual.expected_negative_log_likelihood==pytest.approx(expected+3.25)
    assert sum(actual.expected_negative_log_likelihood_terms)==pytest.approx(expected+3.25)
    # Full-covariance/mean parameter changes together through the ORIGINAL
    # chart, including the derivative of covariance transport G(m).
    dm=np.zeros(18);dm[:12]=np.cos(np.arange(12)+.7)
    root=np.sqrt(np.diag(P));v=np.sin(np.arange(18)+.3)*root
    dP=np.outer(v,v)+np.diag(root**2)*.2
    h=2e-6
    def f(mm,PP):return evaluator.evaluate(mm,PP,likelihood_mean=y,likelihood_precision=W).cost
    fd=(f(m+h*dm,P+h*dP)-f(m-h*dm,P-h*dP))/(2*h)
    analytic=actual.gradient_mean@dm+np.sum(actual.gradient_covariance*dP)
    assert analytic==pytest.approx(fd,abs=2e-8,rel=2e-7)


def test_empty_dirac_query_is_exact_and_readonly_survives_spawn_pickle():
    state=JointTransformState((),np.empty((0,0)),stamp=2.,command=())
    actual=module.evaluate_moment_query(state,(),np.zeros(0),np.empty((0,0)),
        kind='point',point_or_direction=POINT,likelihood_mean=np.zeros(3),likelihood_precision=np.eye(3))
    np.testing.assert_array_equal(actual.mean,POINT)
    np.testing.assert_array_equal(actual.covariance,np.zeros((3,3)))
    for value in (actual,pickle.loads(pickle.dumps(actual))):
        for a in (value.mean,value.covariance,value.gradient_mean,value.gradient_covariance):
            assert not a.flags.writeable
        with pytest.raises(FrozenInstanceError):value.source_stamp=7.


def test_linear_translation_Gaussian_expected_likelihood_gradient_is_analytic():
    P=np.zeros((6,6));P[:3,:3]=np.diag([.01,.02,.03])
    state=JointTransformState((BinghamTransform('t',[.1,.2,.3],
        translation_covariance=P[:3,:3],rotation_wxyz=[1.,0.,0.,0.]),),P)
    m=np.array([.2,-.1,.05,0.,0.,0.]);y=np.array([.2,.3,.4]);W=np.diag([2.,3.,4.])
    actual=module.evaluate_moment_query(state,('t',),m,P,kind='point',
                                      likelihood_mean=y,likelihood_precision=W)
    np.testing.assert_allclose(actual.mean,np.array([.1,.2,.3])+m[:3],atol=0.)
    np.testing.assert_allclose(actual.gradient_mean[:3],W@(actual.mean-y),atol=1e-15)
    np.testing.assert_allclose(actual.gradient_covariance[:3,:3],.5*W,atol=0.)


def test_no_silent_fallback_and_true_invalid_inputs(monkeypatch):
    state=source();m=np.zeros(18)
    def fail():raise RuntimeError('native moment library unavailable')
    monkeypatch.setattr(module,'_library',fail)
    with pytest.raises(RuntimeError,match='unavailable'):
        module.evaluate_moment_query(state,PATHS[0],m,state.covariance)
    with pytest.raises(ValueError,match='finite'):
        module.evaluate_moment_query(state,PATHS[0],np.full(18,np.nan),state.covariance)
    with pytest.raises(ValueError,match='supplied together'):
        module.evaluate_moment_query(state,PATHS[0],m,state.covariance,likelihood_mean=np.zeros(6))


@pytest.mark.parametrize('shift',[0.,1e-8,.6])
def test_native_fixed_chart_at_current_modal_zero_and_large_original_mean(shift):
    state=source();m=np.zeros(18);m[:12]=shift*np.sin(np.arange(12)+.4)
    P=.7*state.covariance
    trial=reexpress_state(state,m,P);query=pose_query(trial,PATHS[0])
    evaluator=module.MomentQueryEvaluator(state,PATHS[0],kind='pose',
                                       reference_rotation_wxyz=query.rotation_wxyz)
    a=np.cos(np.arange(6)+.3);B=np.eye(6)*.4
    actual=evaluator.evaluate(m,P,adjoint_mean=a,adjoint_covariance=B)
    np.testing.assert_allclose(actual.mean,query.mean,atol=3e-14)
    np.testing.assert_allclose(actual.covariance,query.covariance,atol=2e-15)
    d=np.zeros(18);d[:12]=np.cos(np.arange(12)+.7)
    h=1e-7
    plus=evaluator.evaluate(m+h*d,P,adjoint_mean=a,adjoint_covariance=B).cost
    minus=evaluator.evaluate(m-h*d,P,adjoint_mean=a,adjoint_covariance=B).cost
    assert actual.gradient_mean@d==pytest.approx((plus-minus)/(2*h),abs=3e-8,rel=2e-7)


def test_native_chart_context_and_original_source_cannot_be_mutated_by_caller():
    state=source();direction=POINT.copy();projection=BASIS.copy()
    evaluator=module.MomentQueryEvaluator(state,PATHS[1],kind='direction',
        point_or_direction=direction,projection_matrix=projection)
    before=evaluator.evaluate(np.zeros(18),state.covariance)
    direction[:]=0.;projection[:]=0.
    after=evaluator.evaluate(np.zeros(18),state.covariance)
    np.testing.assert_array_equal(before.mean,after.mean)
    np.testing.assert_array_equal(before.covariance,after.covariance)
    with pytest.raises(AttributeError,match='immutable'):evaluator.state=state
    with pytest.raises(TypeError,match='immutable'):after.diagnostics['source']='spoof'


def test_singular_full_cross_source_chains_reverse_to_original_support():
    loading=np.zeros((12,4));loading[:3,0]=[.003,-.002,.001]
    loading[6:9,0]=[-.001,.004,-.002]
    loading[3:6,0]=[.002,-.003,.001]
    loading[3:6,1:]=np.diag([.04,.05,.06])
    P=loading@loading.T
    q=rotation_vector_to_quaternion([.2,-.1,.3])
    state=JointTransformState((
        BinghamTransform('a',[.1,.2,.3],bingham_from_rotation_coordinates(q,P[3:6,3:6]),P[:3,:3]),
        BinghamTransform('fixed_rotation',[.3,.1,-.2],translation_covariance=P[6:9,6:9],rotation_wxyz=q)),P)
    m=loading@np.array([.2,-.1,.3,-.2])
    evaluator=module.MomentQueryEvaluator(state,(('fixed_rotation',True),('a',True)),
        kind='direction',point_or_direction=POINT,projection_matrix=BASIS)
    actual=evaluator.evaluate(m,.6*P,likelihood_mean=np.zeros(2),likelihood_precision=np.eye(2)*3.)
    dm=loading@np.array([-.1,.3,-.2,.4]);dP=loading@np.diag([.3,-.2,.1,-.1])@loading.T
    h=2e-6
    plus=evaluator.evaluate(m+h*dm,.6*P+h*dP,likelihood_mean=np.zeros(2),likelihood_precision=np.eye(2)*3.).cost
    minus=evaluator.evaluate(m-h*dm,.6*P-h*dP,likelihood_mean=np.zeros(2),likelihood_precision=np.eye(2)*3.).cost
    derivative=actual.gradient_mean@dm+np.sum(actual.gradient_covariance*dP)
    assert derivative==pytest.approx((plus-minus)/(2*h),abs=2e-9,rel=2e-7)
