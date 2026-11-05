"""Exact anisotropic observation factors for attitude and body acceleration.

The caller supplies one conditional prior a|q ~ N(m + F R(q).T g, S).
Acceleration is integrated out analytically. These routines return observation
factors only, never a prior-times-observation posterior attitude distribution.
Quaternion coefficients use Hamilton wxyz with R mapping body to world.
"""

import numpy as np

from probtf.bingham import _rotation_quadratic_forms


_ROTATION_FORMS = np.asarray(_rotation_quadratic_forms()).reshape(3, 3, 4, 4)


def _array(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(name + ' must be finite with shape ' + str(shape))
    return value


def _symmetric(value):
    # Antisymmetric arithmetic roundoff does not contribute to a quadratic form.
    return .5 * (value + value.T)


def quaternion_bilinear_form(x, z):
    """Return symmetric H(x,z) satisfying q.T H q = x.T R(q) z."""
    x = _array(x, (3,), 'world vector')
    z = _array(z, (3,), 'body vector')
    return np.einsum('i,j,ijab->ab', x, z, _ROTATION_FORMS)


def _chol_solve(factor, value):
    return np.linalg.solve(factor.T, np.linalg.solve(factor, value))


def acceleration_factor(y, m, F, S, Sigma_a, g):
    """Marginalize body acceleration and return the current exact factor.

    With h=R(q).T g and G=F-I, the observation model is
    y=a-h+n, a|q~N(m+Fh,S), n~N(0,Sigma_a). Therefore

      L_a(q) = N(y; m+G h, V), V=S+Sigma_a,
      log L_a(q) = q.T A q - .5 h.T Wg h + q-independent terms.

    S and Sigma_a are supplied covariance matrices; V must admit the needed
    positive-definite solve. No isotropization, jitter, clipping or numerical
    concentration threshold is used. The returned conditional covariance uses
    the algebraically equivalent Joseph form to preserve its Gram structure.
    """
    y, m = _array(y, (3,), 'acceleration observation'), _array(m, (3,), 'conditional intercept')
    F = _array(F, (3, 3), 'conditional coupling')
    S = _symmetric(_array(S, (3, 3), 'conditional acceleration covariance'))
    Sigma_a = _symmetric(_array(Sigma_a, (3, 3), 'accelerometer covariance'))
    g = _array(g, (3,), 'world gravity')
    G = F - np.eye(3)
    V = _symmetric(S + Sigma_a)
    lower = np.linalg.cholesky(V)
    d = y - m
    inverse_d = _chol_solve(lower, d)
    inverse_G = _chol_solve(lower, G)
    gain = _chol_solve(lower, S.T).T
    remainder = np.eye(3) - gain
    posterior_covariance = _symmetric(remainder @ S @ remainder.T + gain @ Sigma_a @ gain.T)
    return dict(A=quaternion_bilinear_form(g, G.T @ inverse_d),
                Wg=_symmetric(G.T @ inverse_G),
                updated_m=m + gain @ d,
                updated_F=F - gain @ G,
                updated_S=posterior_covariance,
                V=V, G=G, d=d, gain=gain)


def magnetic_factor(y, Sigma_m, b):
    """Return L_m(q)=N(y; R(q).T b,Sigma_m) as quadratic/quartic coefficients.

    Its logarithm is q.T A q - .5 h_b.T Wb h_b plus a q-independent
    constant. No previous attitude prior or acceleration conditional is used.
    """
    y = _array(y, (3,), 'magnetic observation')
    Sigma_m = _symmetric(_array(Sigma_m, (3, 3), 'magnetic covariance'))
    b = _array(b, (3,), 'world magnetic field')
    lower = np.linalg.cholesky(Sigma_m)
    information = _chol_solve(lower, y)
    return dict(A=quaternion_bilinear_form(b, information),
                Wb=_symmetric(_chol_solve(lower, np.eye(3))))
