"""Array-only linear Gaussian conditioning shared by dependency adapters."""

import numpy as np


def condition_gaussian_linear(mean, covariance, jacobian, residual, noise_covariance=None,
                              *, isotropic_noise_precision=None):
    """Condition ``residual + H delta = 0`` using the existing Joseph update.

    ``mean[..., d]``, ``covariance[..., d, d]``, ``jacobian[..., m, d]``,
    ``residual[..., m]`` and ``noise_covariance[..., m, m]`` may have
    broadcastable leading batch dimensions. The returned posterior mean,
    posterior covariance, innovation *covariance* and gain have the common
    leading dimensions. The residual is evaluated at the supplied prior
    mean, so the posterior mean is ``mean - gain @ residual``.

    This is precisely the numerical update used by GaussianLatentStore, with
    all covariance cross blocks retained. No distribution projection,
    eigenvalue clipping, regularization, jitter or pseudoinverse is applied.
    Prior covariance may be singular; observation noise must be positive
    definite. Invalid dimensions/nonfinite arrays/covariances fail before any
    caller-owned state can be modified. Inputs are never mutated.

    Alternatively, ``isotropic_noise_precision=precisions`` evaluates the SAME
    fixed, unbatched prior/observation for noise ``I / precisions[...]``. One
    prior Cholesky and SVD serve the entire precision family. Its output has
    the precision array's leading dimensions. The ordinary covariance branch
    is unchanged; specifying both noise representations is an error.
    Its Joseph arithmetic may retain a roundoff residual of an extremely wide
    prior (for example1e30) after a nearly unit gain; the optional factor-form
    family avoids that subtraction when applicable. This is a numerical
    limitation, not a reason to change a caller's declared prior or noise.
    """
    if isotropic_noise_precision is not None:
        if noise_covariance is not None:
            raise ValueError('Specify noise covariance OR isotropic noise precision, not both.')
        return _condition_isotropic_precision_family(mean,covariance,jacobian,residual,
                                                      isotropic_noise_precision)
    names = ('mean', 'covariance', 'jacobian', 'residual', 'noise_covariance')
    arrays = tuple(np.asarray(value, dtype=float) for value in
                   (mean, covariance, jacobian, residual, noise_covariance))
    mean, covariance, jacobian, residual, noise = arrays
    if (mean.ndim < 1 or mean.shape[-1] < 1 or residual.ndim < 1
            or residual.shape[-1] < 1 or covariance.ndim < 2
            or jacobian.ndim < 2 or noise.ndim < 2):
        raise ValueError('Nonempty Gaussian vectors and matrix core dimensions are required.')
    d, m = mean.shape[-1], residual.shape[-1]
    if covariance.shape[-2:] != (d, d) or jacobian.shape[-2:] != (m, d) or noise.shape[-2:] != (m, m):
        raise ValueError('Gaussian conditioning core dimensions do not match.')
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError('Gaussian conditioning inputs must be finite.')
    # np.broadcast_shapes is not available in the supported NumPy versions.
    # These scalar broadcast views determine shape without allocating an
    # array proportional to any caller's batch size.
    core_dimensions = (1, 2, 2, 1, 2)
    try:
        batch = np.broadcast(*(np.broadcast_to(0., value.shape[:-core])
            for value, core in zip(arrays, core_dimensions))).shape
    except ValueError as error:
        raise ValueError('Gaussian conditioning batch dimensions do not broadcast.') from error
    mean = np.broadcast_to(mean, batch+(d,))
    covariance = np.broadcast_to(covariance, batch+(d, d))
    jacobian = np.broadcast_to(jacobian, batch+(m, d))
    residual = np.broadcast_to(residual, batch+(m,))
    noise = np.broadcast_to(noise, batch+(m, m))
    if (not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-10, atol=1e-12)
            or not np.allclose(noise, noise.swapaxes(-1, -2), rtol=1e-10, atol=1e-12)):
        raise ValueError('Gaussian covariance inputs must be symmetric.')
    scale = np.maximum(1., np.linalg.norm(covariance, ord=np.inf, axis=(-2, -1)))
    if np.any(np.linalg.eigvalsh(covariance)[..., 0] < -1e-10*scale):
        raise ValueError('Prior covariance must be positive semidefinite.')
    # Validate the declared proper observation noise without changing it.
    try:
        np.linalg.cholesky(noise)
    except np.linalg.LinAlgError as error:
        raise ValueError('Observation noise covariance must be positive definite.') from error

    # Extracted from GaussianLatentStore.apply_observation. Matrix products
    # and the one solve accept leading dimensions directly; there is no
    # independent per-batch solver or loop over candidate conditionals.
    innovation = jacobian @ covariance @ jacobian.swapaxes(-1, -2) + noise
    innovation = .5*(innovation + innovation.swapaxes(-1, -2))
    gain = np.linalg.solve(innovation, jacobian @ covariance).swapaxes(-1, -2)
    correction = gain @ residual if not batch else (gain @ residual[..., None])[..., 0]
    posterior_mean = mean - correction
    joseph = np.eye(d, dtype=float) - gain @ jacobian
    posterior_covariance = (joseph @ covariance @ joseph.swapaxes(-1, -2)
                            + gain @ noise @ gain.swapaxes(-1, -2))
    posterior_covariance = .5*(posterior_covariance + posterior_covariance.swapaxes(-1, -2))
    if any(not np.isfinite(value).all() for value in
           (posterior_mean, posterior_covariance, innovation, gain)):
        raise FloatingPointError('Gaussian conditioning produced nonfinite moments; no numerical repair applied.')
    return posterior_mean, posterior_covariance, innovation, gain


def _condition_isotropic_precision_family(mean,covariance,jacobian,residual,precisions):
    """Exact fixed-prior Gaussian family; no rank truncation or noise repair."""
    mean,covariance,jacobian,residual,precisions=(np.asarray(value,dtype=float)
        for value in (mean,covariance,jacobian,residual,precisions))
    if (mean.ndim!=1 or residual.ndim!=1 or not len(mean) or not len(residual)
            or covariance.shape!=(len(mean),len(mean))
            or jacobian.shape!=(len(residual),len(mean))):
        raise ValueError('Isotropic precision family requires fixed unbatched Gaussian core dimensions.')
    if any(not np.isfinite(value).all() for value in (mean,covariance,jacobian,residual,precisions)):
        raise ValueError('Gaussian conditioning inputs and precisions must be finite.')
    if precisions.size<1 or np.any(precisions<=0):
        raise ValueError('Isotropic observation precisions must be positive and nonempty.')
    if not np.allclose(covariance,covariance.T,rtol=1e-10,atol=1e-12):
        raise ValueError('Gaussian covariance inputs must be symmetric.')
    d,m=len(mean),len(residual)
    with np.errstate(over='raise',divide='raise',invalid='raise'):
        noise=np.eye(m)/precisions[...,None,None]
        # Keep the established Joseph implementation for singular priors and
        # tolerated near-symmetry. No eigenvalue threshold/floor, factor repair
        # or pseudoinverse is introduced by the specialization.
        if not np.array_equal(covariance,covariance.T):
            return condition_gaussian_linear(mean,covariance,jacobian,residual,noise)
        try:
            root=np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            return condition_gaussian_linear(mean,covariance,jacobian,residual,noise)
        # Full V is needed when m<d: its d-m columns are unobserved directions,
        # not zero posterior variance. No singular value is dropped, however
        # small; a zero singular value retains variance and contributes no gain.
        left,singular,right=np.linalg.svd(jacobian@root,full_matrices=(m<d))
        basis=root@right.T
        count=len(singular)
        denominator=1.+precisions[...,None]*singular**2
        variance=np.ones(precisions.shape+(d,))
        variance[...,:count]=1./denominator
        posterior_covariance=(basis*variance[...,None,:])@basis.T
        posterior_covariance=.5*(posterior_covariance+posterior_covariance.swapaxes(-1,-2))
        scale=precisions[...,None]*singular/denominator
        gain=(basis[:,:count]*scale[...,None,:])@left[:,:count].T
        posterior_mean=mean-(gain@residual[...,None])[...,0]
        innovation=jacobian@covariance@jacobian.T+noise
        innovation=.5*(innovation+innovation.swapaxes(-1,-2))
    if any(not np.isfinite(value).all() for value in
           (posterior_mean,posterior_covariance,innovation,gain)):
        raise FloatingPointError('Gaussian conditioning produced nonfinite moments; no numerical repair applied.')
    return posterior_mean,posterior_covariance,innovation,gain


__all__ = ['condition_gaussian_linear']
