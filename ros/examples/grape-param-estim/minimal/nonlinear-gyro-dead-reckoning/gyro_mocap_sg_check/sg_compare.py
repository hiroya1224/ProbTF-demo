"""Irregular-clock quaternion SG derivative and original-clock gyro comparison.

This is a separate offline diagnostic. It neither corrects the previous
prediction nor uses Euler-angle derivatives as body angular velocity.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _time(time, name):
    value = np.asarray(time, float)
    if (value.ndim != 1 or len(value) < 2 or not np.isfinite(value).all()
            or np.any(np.diff(value) <= 0.)):
        raise ValueError(name+' must be finite and strictly increasing')
    return value


def continuous_quaternions(quaternions_xyzw):
    """Normalize and sign-align representatives without changing rotations."""
    q = np.asarray(quaternions_xyzw, float).copy()
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise ValueError('quaternions must be finite Nx4 xyzw')
    norm = np.linalg.norm(q, axis=1)
    if np.any(norm == 0.):
        raise ValueError('zero quaternion')
    q /= norm[:, None]
    for index in range(1, len(q)):
        if q[index] @ q[index-1] < 0.:
            q[index] *= -1.
    return q


def differentiate_quaternions(time, quaternions_xyzw, query,
                              window_seconds=.16, degree=3):
    """Centered Savitzky--Golay fits on actual, unresampled timestamps.

    Fit signed quaternion components by a common local polynomial. Normalize
    the fitted quaternion and differentiate that normalization, then compute
    body omega = 2 vec(conjugate(q) Hamilton qdot). The central finite-window
    SG fit is an offline approximation; it does not introduce a causal lag.
    No smoothing/filter output is inserted into the preceding predictor.
    """
    stamps = _time(time, 'time')
    q = continuous_quaternions(quaternions_xyzw)
    query = np.atleast_1d(np.asarray(query, float))
    width = float(window_seconds)
    order = int(degree)
    if (len(q) != len(stamps) or query.ndim != 1 or not np.isfinite(query).all()
            or not np.isfinite(width) or width <= 0. or order != degree or order < 1):
        raise ValueError('Invalid local polynomial inputs')
    half = width/2.
    if np.any(query-half < stamps[0]) or np.any(query+half > stamps[-1]):
        raise ValueError('Queries must have full centered-window source support')
    fitted = np.empty((len(query),4)); derivative = np.empty_like(fitted)
    counts = np.empty(len(query),int); residual = np.empty(len(query))
    condition = np.empty(len(query))
    for index, selected in enumerate(query):
        left = np.searchsorted(stamps, selected-half, side='left')
        right = np.searchsorted(stamps, selected+half, side='right')
        offsets = (stamps[left:right]-selected)/half
        if len(offsets) < order+1:
            raise ValueError('The supplied centered window has too few source measurements')
        design = np.polynomial.polynomial.polyvander(offsets, order)
        coefficients = np.linalg.lstsq(design, q[left:right], rcond=None)[0]
        raw, raw_dot = coefficients[0], coefficients[1]/half
        radius = np.linalg.norm(raw)
        if not np.isfinite(radius) or radius == 0.:
            raise FloatingPointError('The SG quaternion fit has zero/nonfinite norm')
        fitted[index] = raw/radius
        derivative[index] = (raw_dot-fitted[index]*(fitted[index]@raw_dot))/radius
        counts[index] = len(offsets)
        residual[index] = np.sqrt(np.mean((design@coefficients-q[left:right])**2))
        condition[index] = np.linalg.cond(design)
    vector, scalar = fitted[:, :3], fitted[:, 3:]
    vector_dot, scalar_dot = derivative[:, :3], derivative[:, 3:]
    omega = 2.*(scalar*vector_dot-scalar_dot*vector-np.cross(vector,vector_dot))
    return dict(time=query.copy(), quaternion_xyzw=fitted,
        quaternion_derivative_xyzw=derivative, omega_body=omega,
        source_points_per_fit=counts, quaternion_fit_rms=residual,
        design_condition=condition, quaternion_tangency_residual=np.sum(fitted*derivative,axis=1),
        window_seconds=width, degree=order,
        method='centered irregular-clock signed quaternion-component SG; differentiated normalization; body omega from q^-1 qdot',
        quaternion_order='xyzw', angular_velocity_frame='body',
        causal_filter=False, uniform_time_resampling=False)


def _metrics(observed, predicted):
    error = observed-predicted
    centered_o = observed-observed.mean(axis=0)
    centered_p = predicted-predicted.mean(axis=0)
    denominator = np.sqrt(np.sum(centered_o**2,axis=0)*np.sum(centered_p**2,axis=0))
    correlations = np.divide(np.sum(centered_o*centered_p,axis=0), denominator,
                             out=np.full(3,np.nan), where=denominator>0.)
    return dict(rmse_axis_rad_s=np.sqrt(np.mean(error**2,axis=0)),
        vector_rmse_rad_s=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
        mean_error_axis_rad_s=error.mean(axis=0), correlation_axis=correlations)


def compare_lags(gyro_time, gyro, sg_time, sg_omega,
                 window=(50.,55.), max_lag=.25, lag_step=.001):
    """Compare raw gyro(t) with mocap-SG body omega(t-lag).

    Positive lag means the recorded gyro features occur later than mocap's.
    It is a relative effective shift, not proof of a physical sensor delay.
    All trial shifts use the same original gyro rows and fixed support.
    No amplitude, axis rotation, bias or affine gain is fitted.
    """
    gt, st = _time(gyro_time,'gyro_time'), _time(sg_time,'sg_time')
    measured, derived = np.asarray(gyro,float), np.asarray(sg_omega,float)
    lower, upper = map(float,window)
    bound, step = float(max_lag), float(lag_step)
    if (measured.shape != (len(gt),3) or derived.shape != (len(st),3)
            or not np.isfinite(measured).all() or not np.isfinite(derived).all()
            or not lower < upper or bound <= 0. or step <= 0.):
        raise ValueError('Invalid lag-comparison inputs')
    support = ((gt>=lower)&(gt<=upper)&(gt-bound>=st[0])&(gt+bound<=st[-1]))
    times, values = gt[support], measured[support]
    if len(times)<3:
        raise ValueError('No common all-lag support for comparison')
    def at(lag):
        shifted = times-float(lag)
        return np.column_stack([np.interp(shifted,st,derived[:,axis]) for axis in range(3)])
    def score(lag):
        return float(np.mean(np.sum((values-at(lag))**2,axis=1)))
    count = max(2,int(round(2*bound/step)))
    lags = np.linspace(-bound,bound,count+1)
    mse = np.array([score(lag) for lag in lags])
    axis_mse = np.array([np.mean((values-at(lag))**2,axis=0) for lag in lags])
    best_index = int(np.argmin(mse))
    candidates = [(float(lags[best_index]),float(mse[best_index]))]
    if 0<best_index<len(lags)-1:
        refined=minimize_scalar(score,bounds=(lags[best_index-1],lags[best_index+1]),
            method='bounded',options={'xatol':1e-8})
        if np.isfinite(refined.x) and np.isfinite(refined.fun):
            candidates.append((float(refined.x),float(refined.fun)))
    best, _ = min(candidates,key=lambda pair:pair[1])
    axis_best = []
    for axis in range(3):
        selected=int(np.argmin(axis_mse[:,axis]))
        possibilities=[(float(lags[selected]),float(axis_mse[selected,axis]))]
        if 0<selected<len(lags)-1:
            def axis_score(lag): return float(np.mean((values[:,axis]-at(lag)[:,axis])**2))
            refined=minimize_scalar(axis_score,bounds=(lags[selected-1],lags[selected+1]),
                method='bounded',options={'xatol':1e-8})
            if np.isfinite(refined.fun): possibilities.append((float(refined.x),float(refined.fun)))
        axis_best.append(min(possibilities,key=lambda pair:pair[1])[0])
    unshifted, shifted = at(0.),at(best)
    return dict(time=times, gyro=values, gyro_source_indices=np.flatnonzero(support),
        mocap_sg_unshifted=unshifted,mocap_sg_shifted=shifted,
        lag_seconds=lags,vector_mse_rad2_s2=mse,axis_mse_rad2_s2=axis_mse,
        best_lag_seconds=best,best_axis_lag_seconds=np.array(axis_best),
        best_at_search_boundary=bool(best_index in (0,len(lags)-1)),
        before=_metrics(values,unshifted),after=_metrics(values,shifted),
        sign_convention='gyro(t) compared with mocap_SG(t-lag); positive lag means gyro later',
        fixed_support_count=len(times), requested_window_seconds=[lower,upper],
        effective_gyro_window_seconds=[float(times[0]),float(times[-1])],
        uniform_resampling_of_gyro=False,
        sg_shift_interpolation='linear interpolation of SG derivatives evaluated on original mocap timestamps',
        bias_or_scale_or_rotation_fitted=False)
