"""Local polynomial sensor-lag maps on the original irregular IMU clock.

The returned sparse operators retain every original timestamp. Covariances
refer to individual sensor samples, never to a sample mean. The static sensor
model is temporally independent; overlaps between polynomial windows remain
available through the complete sparse operators.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import sparse


def local_polynomial_signal(time, values, noise_covariance, window_seconds=.08,
                            degree=3, max_derivative=2):
    """Fit each local polynomial and expose its linear observation operators.

    Near an endpoint the time window is shifted into the observed interval.
    If that window has too few observations it is expanded to the nearest
    ``degree + 1`` original readings. No padding, resampling, or future sample
    fabrication is performed. This is an explicitly offline sensor model.

    ``covariances[d,e,k]`` is Cov(D_d y[k], D_e y[k]); full cross-time
    covariance is ``(operators[d] @ operators[e].T)[k,l] * R``.
    """
    stamps = np.asarray(time, dtype=float)
    measurements = np.asarray(values, dtype=float)
    covariance = np.asarray(noise_covariance, dtype=float)
    if (stamps.ndim != 1 or measurements.shape != (len(stamps), 3)
            or covariance.shape != (3, 3) or not np.isfinite(stamps).all()
            or not np.isfinite(measurements).all() or not np.isfinite(covariance).all()
            or np.any(np.diff(stamps) <= 0)):
        raise ValueError('Finite original increasing timestamps, Nx3 measurements and 3x3 noise are required.')
    degree, max_derivative = int(degree), int(max_derivative)
    if degree < max_derivative or max_derivative < 0 or len(stamps) < degree + 1:
        raise ValueError('The polynomial and original sample count must identify every requested derivative.')
    window_seconds = float(window_seconds)
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError('window_seconds must be positive and finite.')
    rows, columns = [], []
    coefficients = [[] for _ in range(max_derivative + 1)]
    counts = np.empty(len(stamps), dtype=np.int64)
    extents = np.empty((len(stamps), 2), dtype=float)
    for k, stamp in enumerate(stamps):
        lower = max(stamps[0], min(stamp - .5 * window_seconds,
                                   stamps[-1] - window_seconds))
        upper = lower + window_seconds
        first = int(np.searchsorted(stamps, lower, side='left'))
        stop = int(np.searchsorted(stamps, upper, side='right'))
        if stop - first < degree + 1:
            first = max(0, min(k - degree // 2, len(stamps) - degree - 1))
            stop = first + degree + 1
        indices = np.arange(first, stop)
        offsets = stamps[indices] - stamp
        scale = float(np.max(np.abs(offsets)))
        if scale == 0.:
            scale = 1.  # Unit coordinates for a single constant observation.
        design = np.vander(offsets / scale, N=degree + 1, increasing=True)
        q, r = np.linalg.qr(design, mode='reduced')
        inverse = np.linalg.solve(r, q.T)
        rows.extend([k] * len(indices))
        columns.extend(indices.tolist())
        for order in range(max_derivative + 1):
            weights = math.factorial(order) / scale ** order * inverse[order]
            coefficients[order].extend(weights.tolist())
        counts[k], extents[k] = len(indices), stamps[indices[[0, -1]]]
    operators = tuple(sparse.csr_matrix((weights, (rows, columns)),
                                       shape=(len(stamps), len(stamps)))
                      for weights in coefficients)
    derivatives = tuple(operator @ measurements for operator in operators)
    cross_covariances = np.empty((max_derivative + 1, max_derivative + 1,
                                  len(stamps), 3, 3), dtype=float)
    for d, left in enumerate(operators):
        for e, right in enumerate(operators):
            factors = np.asarray(left.multiply(right).sum(axis=1)).ravel()
            cross_covariances[d, e] = factors[:, None, None] * covariance
    result = dict(smooth=derivatives[0], operators=operators,
                  covariances=cross_covariances, noise_covariance=covariance.copy(),
                  smooth_covariance=cross_covariances[0, 0],
                  neighbor_count=counts, window_extent=extents,
                  point_indices=np.arange(len(stamps)), time=stamps.copy(),
                  window_seconds=window_seconds, degree=degree,
                  max_derivative=max_derivative,
                  temporal_noise_model='independent original individual sensor readings; shared windows retain full sparse operators',
                  method='irregular-time local polynomial, shifted endpoint windows, QR solve')
    if max_derivative >= 1:
        result.update(derivative=derivatives[1], derivative_covariance=cross_covariances[1, 1],
                      smooth_derivative_cross=cross_covariances[0, 1])
    if max_derivative >= 2:
        result.update(second_derivative=derivatives[2],
                      second_derivative_covariance=cross_covariances[2, 2])
    return result


def corrected_lag_signal(signal, tau, tau_variance=0.):
    """Invert y + tau*y_dot = x as an explicit local linear sensor map.

    Fixed-tau covariance includes the smooth/derivative cross term exactly.
    A supplied tau variance uses the first-order parameter sensitivity y_dot.
    Its unknown correlation with the sensor error is bounded by twice the
    sum of those two marginal covariances. This is a declared conservative
    linearization, not an exact nonlinear distribution of a fitted tau.
    """
    tau, tau_variance = float(tau), float(tau_variance)
    if not np.isfinite(tau) or tau < 0 or not np.isfinite(tau_variance) or tau_variance < 0:
        raise ValueError('Sensor time constant and its variance must be finite and nonnegative.')
    operator = signal['operators'][0] + tau * signal['operators'][1]
    mean = signal['smooth'] + tau * signal['derivative']
    factors = np.asarray(operator.multiply(operator).sum(axis=1)).ravel()
    fixed = factors[:, None, None] * signal['noise_covariance']
    parameter = tau_variance * np.einsum('ni,nj->nij', signal['derivative'], signal['derivative'])
    covariance = 2. * (fixed + parameter) if tau_variance else fixed.copy()
    return dict(mean=mean, covariance=covariance, fixed_tau_covariance=fixed,
                tau_covariance=parameter, operator=operator, tau=tau,
                tau_variance=tau_variance,
                covariance_method=('fixed linear operator; 2*(sensor + linearized tau) unknown-cross bound'
                                   if tau_variance else 'exact fixed-tau local linear operator covariance'))
