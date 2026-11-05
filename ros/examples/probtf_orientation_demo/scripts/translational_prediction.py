"""Shared-increment moment prediction for attitude and body acceleration.

The gyro supplies a nominal Hamilton increment ``delta_bar`` and independent
identity-centred Bingham noise ``epsilon``; delta = delta_bar * epsilon.
The vector block of its parameter is -2*(Sigma_omega*dt**2)^-1.  This is an
explicit Bingham process model with the usual high-concentration angular scale,
not a claim that exponentiating Gaussian gyro noise gives an exact Bingham.

Every attitude and acceleration moment uses this same random delta.  All
expectations below contract the existing quaternion second/fourth moments;
there are no particles, generated candidate rotations or tangent updates.
"""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

_CORE = Path(__file__).resolve().parents[4] / 'ros/core/probtf_core/src'
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
from probtf.bingham import (
    bingham_second_moment, bingham_fourth_moment,
    quaternion_product_second_moment, _ROTATION_ENTRY_FORMS,
)


_FORMS = np.asarray(_ROTATION_ENTRY_FORMS).reshape(3, 3, 4, 4)
_DIFFERENCE_FORMS = _FORMS - np.eye(3)[:, :, None, None] * np.eye(4)


def _symmetric(matrix):
    return .5 * (matrix + matrix.T)


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(name + ' must be a finite vector (3,)')
    return array


def _matrix(value, name, symmetric=False):
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError(name + ' must be a finite matrix (3,3)')
    if symmetric:
        # These moments are mathematically symmetric; assemble that quadratic
        # form without an optional numerical-skew stopping criterion.
        return _symmetric(array)
    return array


def _readonly(value):
    array = np.asarray(value, dtype=float).copy()
    array.flags.writeable = False
    return array


def _left_product(quaternion):
    w, x, y, z = quaternion
    return np.array([[w, -x, -y, -z], [x, w, -z, y],
                     [y, z, w, -x], [z, -y, x, w]])


@lru_cache(maxsize=128)
def _noise_moments(rotation_covariance_key):
    covariance = np.asarray(rotation_covariance_key).reshape(3, 3)
    if not np.any(covariance):
        second = np.diag([1., 0., 0., 0.])
        fourth = np.zeros((4, 4, 4, 4)); fourth[0, 0, 0, 0] = 1.
        parameter = None  # exactly deterministic, not a finite Bingham
    else:
        lower = np.linalg.cholesky(covariance)
        inverse_lower = np.linalg.solve(lower, np.eye(3))
        precision = inverse_lower.T @ inverse_lower
        parameter = np.zeros((4, 4))
        parameter[1:, 1:] = -2. * _symmetric(precision)
        second = bingham_second_moment(parameter, integration_steps=200)
        fourth = bingham_fourth_moment(parameter, integration_steps=200)
    rotation_mean = np.einsum('ijab,ab->ij', _FORMS, second)
    rotation_second = np.einsum('ijab,klcd,abcd->ijkl',
                                _FORMS, _FORMS, fourth)
    # Computing Cov(R) from R-I avoids subtracting nearly equal order-one
    # values when the gyro increment is very concentrated. This is the same
    # quadratic-form expectation, with no spread alteration or regularizer.
    difference_mean = np.einsum('ijab,ab->ij', _DIFFERENCE_FORMS, second)
    difference_second = np.einsum('ijab,klcd,abcd->ijkl',
                                  _DIFFERENCE_FORMS, _DIFFERENCE_FORMS, fourth)
    centered_second = difference_second - np.einsum('ij,kl->ijkl',
                                                    difference_mean, difference_mean)
    return tuple(_readonly(value) if value is not None else None for value in
                 (parameter, second, fourth, rotation_mean, rotation_second,
                  difference_mean, difference_second, centered_second))


def gyro_increment(gyro, dt, gyro_covariance):
    """Return one shared Bingham increment law and its transport operators.

    ``mean_quaternion`` is the nominal mode quaternion delta_bar, not the
    arithmetic quaternion mean (which is zero for an antipodal law).
    ``nominal_rotation`` is Dbar and ``mean_rotation`` is E[D_delta].
    ``noise_rotation_second[a,i,b,j]`` is E[R_epsilon[a,i]R_epsilon[b,j]].
    ``average_transported_matrix(X)`` returns E[D_delta.T X D_delta].
    ``uncertainty_center(a,b)`` returns Cov(D_delta.T a,D_delta.T b).
    """
    gyro = _vector(gyro, 'gyro')
    covariance = _matrix(gyro_covariance, 'gyro_covariance', symmetric=True)
    dt = float(dt)
    if not np.isfinite(dt) or dt < 0:
        raise ValueError('dt must be finite and nonnegative')
    nominal = Rotation.from_rotvec(gyro * dt)
    quaternion = nominal.as_quat()[[3, 0, 1, 2]]
    nominal_rotation = nominal.as_matrix()
    key = tuple((covariance * dt**2).ravel())
    parameter, second, fourth, rotation_mean, rotation_second, difference_mean, \
        difference_second, centered_second = _noise_moments(key)

    def average_transported_matrix(matrix):
        matrix = _matrix(matrix, 'transported matrix')
        value = nominal_rotation.T @ matrix @ nominal_rotation
        # E[(I+DeltaR).T X (I+DeltaR)], retaining second-order noise terms.
        return (value + difference_mean.T @ value + value @ difference_mean
                + np.einsum('ab,aibj->ij', value, difference_second))

    def uncertainty_center(left, right):
        left = nominal_rotation.T @ _vector(left, 'left vector')
        right = nominal_rotation.T @ _vector(right, 'right vector')
        return np.einsum('a,b,aibj->ij', left, right, centered_second)

    product = _left_product(quaternion)
    increment_second = _symmetric(product @ second @ product.T)
    return dict(mean_quaternion=quaternion, nominal_rotation=nominal_rotation,
                mean_rotation=nominal_rotation @ rotation_mean,
                M2delta=increment_second, noise_parameter=parameter,
                noise_M2=second, noise_M4=fourth,
                noise_rotation_mean=rotation_mean,
                noise_rotation_second=rotation_second,
                average_transported_matrix=average_transported_matrix,
                uncertainty_center=uncertainty_center,
                model='delta=delta_bar*epsilon; epsilon is identity-centred Bingham')


def predict_joint_moments(Mq, Eh, Nh, Eb, Nb, mean_a, cov_a, cov_ah,
                          gyro, dt, Sigma, beta, Q, *, cov_hg=None):
    """Predict the required joint moments with exactly one shared increment.

    beta and Q specify a'=beta D_delta.T a + w, w independent N(0,Q).
    Passing a directly centred ``cov_hg`` avoids recovering the small radial
    variance from Nh-Eh*Eh.T at a very concentrated attitude. Nh itself is still
    transported, preserving the separate sufficient-statistic second moment.
    """
    eh, eb, ma = (_vector(Eh, 'Eh'), _vector(Eb, 'Eb'), _vector(mean_a, 'mean_a'))
    nh, nb = _matrix(Nh, 'Nh', True), _matrix(Nb, 'Nb', True)
    ca = _matrix(cov_a, 'cov_a', True)
    cah = _matrix(cov_ah, 'cov_ah')
    ch = (_symmetric(nh - np.outer(eh, eh)) if cov_hg is None
          else _matrix(cov_hg, 'cov_hg', True))
    process = _matrix(Q, 'Q', True)
    beta = float(beta)
    if not np.isfinite(beta):
        raise ValueError('beta must be finite')
    delta = gyro_increment(gyro, dt, Sigma)
    transport = delta['average_transported_matrix']
    center = delta['uncertainty_center']
    expected_rotation = delta['mean_rotation']
    return dict(Mq=quaternion_product_second_moment(Mq, delta['M2delta']),
                Ehg=expected_rotation.T @ eh, Ngg=_symmetric(transport(nh)),
                Ehb=expected_rotation.T @ eb, Nbb=_symmetric(transport(nb)),
                mean_a=beta * (expected_rotation.T @ ma),
                cov_a=_symmetric(beta**2 * (transport(ca) + center(ma, ma)) + process),
                cov_hg=_symmetric(transport(ch) + center(eh, eh)),
                cov_ah=beta * (transport(cah) + center(ma, eh)), delta=delta)


def conditional_parameters(mean_a, cov_a, mean_h, cov_h, cov_ah):
    """Regress a|h as N(m+Fh,S), preserving the full cross covariance.

    This is the finite conditional-family projection, not a claim that a
    noisy-increment push-forward remains in that family. The actual positive
    definite h covariance is solved by Cholesky, with no floor or jitter.
    """
    ma, mh = _vector(mean_a, 'mean_a'), _vector(mean_h, 'mean_h')
    ca, ch = _matrix(cov_a, 'cov_a', True), _matrix(cov_h, 'cov_h', True)
    cah = _matrix(cov_ah, 'cov_ah')
    lower = np.linalg.cholesky(ch)
    f = np.linalg.solve(lower.T, np.linalg.solve(lower, cah.T)).T
    return dict(m=ma - f @ mh, F=f, S=_symmetric(ca - f @ ch @ f.T))
