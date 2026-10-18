"""Gaussian marginal optimal-transport coupling for a fixed-goal cut.

This changes a coupling, never either supplied marginal covariance.  On the
original support P=U D U.T the positive symmetric Gaussian transport is

    A = U D^(-1/2) (D^(1/2) U.T Q U D^(1/2))^(1/2) D^(-1/2) U.T.

It satisfies A P A.T=Q on the declared prior support. Applying A to the
body's side of an existing joint law sends its cross C to A C and preserves
the other variable's marginal. Joint congruence proves PSD also when Q is
not a contraction of P. The conditional Schur residual is unchanged when A
is invertible on that support; it may increase after a support reduction.
This is an explicit fixed-goal coupling
policy, not a unique consequence of knowing the two marginals or an extra
Bayesian observation.  The old Schmidt rule remains a separate policy.
"""
import numpy as np

from probtf.spherical_law.support import covariance_support, require_in_support


def _finite_symmetric(value, name):
    value=np.array(value,dtype=float,copy=True)
    if (value.ndim!=2 or value.shape[0]!=value.shape[1]
            or not np.all(np.isfinite(value))):
        raise ValueError(name+' must be finite and square.')
    # Use the unchanged core support convention for covariance symmetry and
    # actual significant-negative-eigenvalue validation. Do not reconstruct
    # or overwrite the supplied matrix with an eigenvalue-floored version.
    support=covariance_support(value,name)
    return value,support


def covariance_transport(prior_cov, posterior_cov):
    """Return an unregularized OT map and non-gating round-trip diagnostics.

    The posterior must lie inside the ORIGINAL prior support. Structural
    Dirac rows/columns remain exactly zero in both input covariances. A
    positive-definite posterior on that support uses Cholesky; an actually
    semidefinite posterior may use a nonnegative spectral Gram factor.
    Negative eigenvalues are never replaced by zero or their absolute value.
    If no real unregularized factor can be formed, the arithmetic failure is
    explicit rather than changing a marginal or manufacturing an identity.
    """
    prior,support=_finite_symmetric(prior_cov,'Gaussian transport prior covariance')
    posterior,unused_support=_finite_symmetric(posterior_cov,'Gaussian transport posterior covariance')
    if posterior.shape!=prior.shape:
        raise ValueError('Gaussian transport covariance dimensions must agree.')
    n=prior.shape[0]
    structural_zero=~(np.any(prior!=0.,axis=0)|np.any(prior!=0.,axis=1))
    if (np.any(posterior[structural_zero,:]!=0.)
            or np.any(posterior[:,structural_zero]!=0.)):
        raise ValueError('Gaussian transport posterior changes an exact prior Dirac coordinate.')
    for column in posterior.T:
        require_in_support(column,support,'Gaussian transport posterior covariance')
    rank=support.basis.shape[1]
    posterior_rank=rank
    factorization='identity: exactly unchanged marginal'
    if np.array_equal(prior,posterior):
        # The otherwise unspecified map outside a singular support can be
        # identity. This exact fast path also preserves existing supported
        # cross arrays without an unnecessary eigensystem round trip.
        A=np.eye(n)
    elif rank==0:
        # There are no retained coordinates under the EXISTING core's
        # numerical support convention. Exact structural Dirac entries are
        # zero by the check above; any nonzero subthreshold stored entries
        # remain untouched in the inputs and visible in round-trip error.
        A=np.zeros_like(prior)
        factorization='rank-zero declared core numerical support'
    else:
        U=support.basis
        root=np.sqrt(support.eigenvalues)
        inverse_root=1./root
        projected=U.T@posterior@U
        projected=.5*projected+.5*projected.T
        try:
            gram=np.linalg.cholesky(projected)
            factorization='Cholesky posterior Gram on the original support'
        except np.linalg.LinAlgError:
            eigenvalues,eigenvectors=np.linalg.eigh(projected)
            if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues<0.):
                minimum=float(eigenvalues.min()) if eigenvalues.size else 0.
                raise ValueError('Gaussian transport posterior has no real unregularized '
                                 'support Gram factor; minimum eigenvalue='+repr(minimum))
            gram=eigenvectors*np.sqrt(eigenvalues)
            posterior_rank=int(np.count_nonzero(eigenvalues>0.))
            factorization='nonnegative spectral posterior Gram on the original support'
        # If M=D^(1/2)L then M M.T is the middle PSD matrix. SVD's singular
        # values are nonnegative by definition; no abs/floor of a negative
        # covariance eigenvalue is used to manufacture its square root.
        M=root[:,None]*gram
        vectors,singular_values,unused_right=np.linalg.svd(M,full_matrices=False)
        middle_root=(vectors*singular_values)@vectors.T
        reduced=inverse_root[:,None]*middle_root*inverse_root[None,:]
        A=U@reduced@U.T
        A=.5*A+.5*A.T
        # U embeds exact structural zero coordinates as exact zeros. Set
        # their algebraically zero map entries explicitly to preserve that
        # invariant after any BLAS implementation's signed-zero convention.
        A[structural_zero,:]=0.;A[:,structural_zero]=0.
    if not np.all(np.isfinite(A)):
        raise ValueError('Gaussian marginal transport is nonfinite.')
    actual=A@prior@A.T
    residual=actual-posterior
    residual_norm=float(np.linalg.norm(residual,ord='fro'))
    posterior_norm=float(np.linalg.norm(posterior,ord='fro'))
    diagnostics=dict(
        method='symmetric Gaussian marginal optimal transport on the original support',
        original_support_rank=rank,dimension=n,
        posterior_rank_on_original_support=posterior_rank,
        conditional_Schur_residual_preserved=posterior_rank==rank,
        numerical_support_convention='unchanged core covariance_support; raw-matrix round-trip discrepancy is separately reported',
        posterior_factorization=factorization,
        fixed_goal_cut_policy=True,
        transport_formula='A=P^(-1/2)(P^(1/2) Q P^(1/2))^(1/2)P^(-1/2) on the original support',
        target_marginal_covariance_changed=False,
        source_marginal_covariance_changed=False,
        covariance_regularization_added=False,covariance_clip_applied=False,
        negative_eigenvalue_floor_applied=False,
        negative_eigenvalue_absolute_value_applied=False,
        process_noise_added=False,cross_covariance_zeroed=False,
        exact_Dirac_coordinates_preserved=True,
        source_and_target_covariance_unchanged=True,
        round_trip_maximum_absolute_error=float(np.max(np.abs(residual),initial=0.)),
        round_trip_Frobenius_error=residual_norm,
        round_trip_relative_Frobenius_error=(residual_norm/posterior_norm
            if posterior_norm!=0. else (0. if residual_norm==0. else None)),
        round_trip_diagnostic_is_stop_gate=False,
        joint_PSD_argument='joint congruence by diag(A,I) preserves PSD and the environment marginal; A P A.T=Q on the declared support; conditional Schur residual is equal only without support loss',
        monte_carlo_used=False)
    A.setflags(write=False)
    return A,diagnostics
