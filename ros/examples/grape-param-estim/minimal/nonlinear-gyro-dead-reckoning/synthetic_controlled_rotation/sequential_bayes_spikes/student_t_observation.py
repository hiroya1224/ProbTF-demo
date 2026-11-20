"""Marginal Student-t gyro factors and numerical importance-grid placement.

The supplied matrix is the marginal observation covariance, not the Student-t
scale matrix. The gamma precision is integrated analytically. Optimizing a
Gaussian proposal changes integration nodes only, never the posterior law.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammaln


def student_t_scale(covariance, degrees_of_freedom):
    nu = float(degrees_of_freedom)
    if not np.isfinite(nu) or nu <= 2.:
        raise ValueError('Finite Student-t df > 2 is required to preserve the declared covariance')
    return np.asarray(covariance, float)*((nu-2.)/nu)


def student_t_log_density(values, mean, covariance, degrees_of_freedom):
    nu = float(degrees_of_freedom)
    scale = student_t_scale(covariance, nu)
    root = np.linalg.cholesky(scale)
    residual = np.asarray(values, float)-np.asarray(mean, float)
    white = np.linalg.solve(root, residual.T).T
    distance = np.sum(white**2, axis=-1)
    return (gammaln(.5*(nu+3.))-gammaln(.5*nu)-1.5*np.log(nu*np.pi)
            -np.log(np.diag(root)).sum()-.5*(nu+3.)*np.log1p(distance/nu))


def fit_proposal(flow, prior_mean, covariance, times, observations, warm_start,
                 max_nfev, degrees_of_freedom):
    """Fit a finite Gaussian integration proposal to the Student-t target.

    Grouped transformed residuals reproduce the multivariate t negative log
    likelihood, not three independent univariate t likelihoods. The original
    Gaussian initial law remains a quadratic prior. Full target/proposal
    weights are still used by the caller for every reported moment.
    """
    nu = float(degrees_of_freedom)
    root_prior = np.linalg.cholesky(covariance)
    root_scale = np.linalg.cholesky(student_t_scale(covariance, nu))
    def residual(value):
        states = flow.evaluate(np.asarray(value)[None, :], times)
        prior = np.linalg.solve(root_prior, value-prior_mean)
        white = np.linalg.solve(root_scale, (states[1:, 0, :3]-observations[1:]).T).T
        squared = np.sum(white**2, axis=1)
        ratio = np.full_like(squared, (nu+3.)/nu)
        positive = squared > 0.
        ratio[positive] = (nu+3.)*np.log1p(squared[positive]/nu)/squared[positive]
        return np.r_[prior, (np.sqrt(ratio)[:, None]*white).reshape(-1)]
    results = []
    for initial in (np.asarray(warm_start, float), prior_mean):
        result = least_squares(residual, initial, jac='3-point', max_nfev=max_nfev,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun).all():
            raise FloatingPointError('Nonfinite Student-t proposal parameters')
        results.append(result)
    best = min(results, key=lambda result: float(result.fun@result.fun))
    precision = best.jac.T@best.jac
    root_precision = np.linalg.cholesky(.5*(precision+precision.T))
    inverse = np.linalg.solve(root_precision, np.eye(3))
    record = dict(placement='Gaussian numerical importance proposal fitted to initial Gaussian times all acquired multivariate Student-t factors',
        target_replaced_by_proposal=False, student_df=nu,
        scale_matrix_rule='Sigma=(nu-2)/nu * marginal R',
        chosen_cost=float(best.cost), chosen_optimality=float(best.optimality),
        chosen_status=int(best.status), chosen_success=bool(best.success), chosen_nfev=int(best.nfev),
        starts_cost=[float(result.cost) for result in results],
        starts_nfev=[int(result.nfev) for result in results],
        maximum_function_evaluations_per_start=int(max_nfev),
        finite_unconverged_proposal_does_not_reject_observation=True,
        covariance='Inverse Gauss-Newton matrix of grouped transformed Student-t residuals plus original prior; integration proposal only')
    return best.x.copy(), inverse.T@inverse, record
