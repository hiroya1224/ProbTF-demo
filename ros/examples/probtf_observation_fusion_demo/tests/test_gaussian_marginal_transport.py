"""An explicit joint Gaussian coupling, without changing either marginal."""
import numpy as np
import pytest

from probtf_observation_fusion_demo.gaussian_marginal_transport import covariance_transport


def test_scalar_expansion_preserves_joint_PSD_where_old_Schmidt_cross_does_not():
    prior=np.array([[1.]])
    posterior=np.array([[2.]])
    cross=np.array([[.8]])
    environment=np.array([[1.]])
    old=posterior@np.linalg.inv(prior)@cross
    assert np.linalg.det(np.block([[posterior,old],[old.T,environment]]))<0.
    A,record=covariance_transport(prior,posterior)
    np.testing.assert_allclose(A,[[np.sqrt(2.)]],atol=2e-15)
    new=A@cross
    np.testing.assert_allclose(A@prior@A.T,posterior,atol=1e-15)
    assert np.linalg.eigvalsh(np.block([[posterior,new],[new.T,environment]])).min()>0.
    np.testing.assert_allclose(environment-new.T@np.linalg.inv(posterior)@new,[[.36]],atol=1e-15)
    assert not record['covariance_clip_applied']
    assert not record['process_noise_added']
    assert not record['round_trip_diagnostic_is_stop_gate']


@pytest.mark.parametrize('scale',[.2,1.,3.])
def test_noncommuting_full_cross_covariance_and_both_marginals_are_preserved(scale):
    L=np.array([[.7,0.,0.],[.2,.9,0.],[-.1,.3,.6]])
    M=np.array([[.4,.1,.2],[-.1,.7,.2],[.15,-.2,.5]])*scale
    prior=L@L.T;posterior=M@M.T
    cross=np.array([[.08,-.1],[.2,.06],[-.04,.1]])
    conditional=np.array([[.9,.2],[.2,.7]])
    environment=conditional+cross.T@np.linalg.solve(prior,cross)
    before=(prior.copy(),posterior.copy(),cross.copy(),environment.copy())
    A,record=covariance_transport(prior,posterior)
    new=A@cross
    np.testing.assert_allclose(A,A.T,atol=0.)
    np.testing.assert_allclose(A@prior@A.T,posterior,atol=3e-15,rtol=3e-14)
    np.testing.assert_allclose(environment-new.T@np.linalg.solve(posterior,new),conditional,atol=5e-15)
    assert np.linalg.eigvalsh(np.block([[posterior,new],[new.T,environment]])).min()>0.
    for old,current in zip(before,(prior,posterior,cross,environment)):
        np.testing.assert_array_equal(old,current)
    assert not A.flags.writeable
    assert record['round_trip_relative_Frobenius_error']<1e-13


def test_singular_original_support_exact_Dirac_and_unchanged_identity():
    U=np.array([[1.,0.],[0.,1.],[0.,0.],[0.,0.]])
    prior=U@np.array([[.2,.04],[.04,.3]])@U.T
    posterior=U@np.array([[.3,-.06],[-.06,.1]])@U.T
    A,record=covariance_transport(prior,posterior)
    np.testing.assert_allclose(A@prior@A.T,posterior,atol=5e-16)
    np.testing.assert_array_equal(A[2:,:],np.zeros((2,4)))
    np.testing.assert_array_equal(A[:,2:],np.zeros((4,2)))
    assert record['original_support_rank']==2
    exact,identity_record=covariance_transport(prior,prior)
    np.testing.assert_array_equal(exact,np.eye(4))
    np.testing.assert_array_equal(exact@prior@exact.T,prior)
    assert identity_record['round_trip_maximum_absolute_error']==0.


def test_semidefinite_target_uses_a_real_nonnegative_Gram_without_floor():
    A,record=covariance_transport(np.eye(3),np.diag([.2,0.,.5]))
    np.testing.assert_allclose(A,np.diag(np.sqrt([.2,0.,.5])),atol=2e-15)
    np.testing.assert_allclose(A@A.T,np.diag([.2,0.,.5]),atol=2e-16)
    assert record['posterior_factorization'].startswith('nonnegative spectral')
    assert not record['negative_eigenvalue_floor_applied']


def test_positive_tiny_target_variance_is_not_floored_or_dropped():
    target=np.diag([1e-30,.2])
    A,record=covariance_transport(np.eye(2),target)
    np.testing.assert_allclose(A@A.T,target,atol=0.,rtol=5e-15)
    assert A[0,0]>0.
    assert record['posterior_factorization'].startswith('Cholesky')


@pytest.mark.parametrize('n',[0,3])
def test_Dirac_identity_is_exact(n):
    P=np.zeros((n,n));A,record=covariance_transport(P,P)
    np.testing.assert_array_equal(A,np.eye(n))
    np.testing.assert_array_equal(A@P@A.T,P)
    assert record['original_support_rank']==0


def test_non_axis_singular_support_preserves_supported_environment_cross():
    u=np.array([1.,2.,-1.]);u/=np.linalg.norm(u)
    P=.4*np.outer(u,u);Q=1.1*np.outer(u,u)
    # A strictly positive reduced 1x1 target uses Cholesky, even though the
    # full ambient matrices have arbitrary null-space eigenvector roundoff.
    A,record=covariance_transport(P,Q)
    np.testing.assert_allclose(A@P@A.T,Q,atol=5e-16)
    C=u[:,None]*.1
    np.testing.assert_allclose((A@C).ravel(),np.sqrt(1.1/.4)*C.ravel(),atol=5e-16)
    assert record['original_support_rank']==1


def test_BFS_keep_reset_mixture_preserves_PSD_fixed_goal_and_body_between_covariance():
    P=np.array([[1.,.2],[.2,.7]])
    B=np.array([[1.6,-.1],[-.1,.5]])
    C=np.array([[.2],[.1]]);E=np.array([[.5]])
    A,unused=covariance_transport(P,B)
    keep_cross=A@C
    keep=np.block([[B,keep_cross],[keep_cross.T,E]])
    reset_body=np.array([[.8,.1],[.1,.9]])
    reset=np.block([[reset_body,np.zeros((2,1))],[np.zeros((1,2)),E]])
    k=.3;mean_keep=np.array([.2,-.1,0.]);mean_reset=np.array([-.1,.4,0.])
    delta=mean_keep-mean_reset
    mixed=k*keep+(1.-k)*reset+k*(1.-k)*np.outer(delta,delta)
    np.testing.assert_array_equal(mixed[2:,2:],E)
    np.testing.assert_allclose(mixed[:2,2:],k*keep_cross,atol=0.)
    assert np.linalg.eigvalsh(mixed).min()>0.


@pytest.mark.parametrize('bad',['negative','outside_Dirac','nonfinite','asymmetric','shape'])
def test_actual_covariance_or_support_failure_is_explicit(bad):
    prior=np.diag([1.,0.]);posterior=np.diag([.5,0.])
    if bad=='negative':posterior[0,0]=-.1
    elif bad=='outside_Dirac':posterior[1,1]=1e-30
    elif bad=='nonfinite':posterior[0,0]=np.nan
    elif bad=='asymmetric':posterior[0,1]=.1
    else:posterior=np.eye(3)
    with pytest.raises(ValueError):covariance_transport(prior,posterior)
