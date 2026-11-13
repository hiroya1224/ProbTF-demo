#!/usr/bin/env python3
"""Sparse-gyro controlled nonlinear propagation; all quadrature is deterministic."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
for directory in (HERE, ROOT/'ros/core/probtf_core/src'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from propagation import integrate_six_point_increment, propagate_orientation_second_moments
from propagation import quaternion_product_second_moment
from torque import build_torque_profile

METHODS = ('constant', 'linear_forced', 'nonlinear')
DEFAULT_NOISE = (.1, .5, 2., 5.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, Path): return str(item)
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, default=convert, ensure_ascii=False,
                                    indent=2, allow_nan=False)+'\n')


def quaternion_multiply(left, right):
    """Hamilton product in wxyz order, with NumPy broadcasting."""
    left, right = np.broadcast_arrays(np.asarray(left), np.asarray(right))
    scalar = left[..., :1]*right[..., :1] - np.sum(left[..., 1:]*right[..., 1:], axis=-1, keepdims=True)
    vector = (left[..., :1]*right[..., 1:] + right[..., :1]*left[..., 1:]
              + np.cross(left[..., 1:], right[..., 1:]))
    return np.concatenate((scalar, vector), axis=-1)


def finite_law_statistics(quaternions, weights, second_moments, initial_rpy):
    """Actual Euler moments of the finite cubature-product law, not a Gaussian fit.

    The principal second-moment eigenvector selects a local Euler branch.
    Euler angles are then unwrapped on that branch before weighted moments.
    No observed/mocap trajectory selects the branch after initial alignment.
    """
    _, eigenvectors = np.linalg.eigh(second_moments)
    representative = eigenvectors[:, :, -1]
    rpy_anchor = Rotation.from_quat(representative[:, [1,2,3,0]]).as_euler('xyz')
    rpy_anchor = np.unwrap(np.vstack((initial_rpy, rpy_anchor)), axis=0)[1:]
    shape = quaternions.shape[:-1]
    angles = Rotation.from_quat(quaternions.reshape(-1,4)[:, [1,2,3,0]]).as_euler('xyz').reshape(*shape,3)
    angles = rpy_anchor[:,None,:] + (angles-rpy_anchor[:,None,:]+np.pi) % (2*np.pi)-np.pi
    mean = np.einsum('tkd,k->td', angles, weights)
    centered = angles-mean[:,None,:]
    covariance = np.einsum('tki,k,tkj->tij', centered, weights, centered)
    explicit = np.einsum('tki,k,tkj->tij', quaternions, weights, quaternions)
    return representative, mean, covariance, explicit


def predict(data, torque_profile, sigma_deg_s, mode, output_dt=.01, max_step=.002):
    """Independent fresh measured-input law at each original 0.5 Hz timestamp.

    Within a block the six SAME initial angular-velocity candidates are
    integrated throughout. Product enumeration for Euler statistics exactly
    represents the finite six-point rule (6, 36, 216 branches); it is not MC.
    """
    start, end = np.asarray(data['time'],float)
    observations = np.asarray(data['sparse_gyro_time'],float)
    measured = np.asarray(data['sparse_gyro'],float)
    covariance = np.eye(3)*np.deg2rad(sigma_deg_s)**2
    q0 = np.asarray(data['initial_quaternion_wxyz'],float)
    parent = q0[None,:].copy()
    parent_weights = np.ones(1)
    initial_moment = np.outer(q0,q0)
    previous_rpy = Rotation.from_quat(q0[[1,2,3,0]]).as_euler('xyz')
    grid = np.unique(np.r_[np.linspace(start,end,int(round((end-start)/output_dt))+1), observations])
    result = {name:[] for name in ('time','quaternion_second_moment','representative_quaternion_wxyz',
        'rpy_mean_rad','rpy_covariance_rad2','omega_mean_rad_s','omega_covariance_rad2_s2','block_index')}
    details = {}
    moment_residuals = []
    for block,begin in enumerate(observations):
        finish = observations[block+1] if block+1<len(observations) else end
        times = grid[(grid>=begin)&(grid<=finish)]
        prediction = integrate_six_point_increment(measured[block],covariance,times,
            torque_profile['torque'],torque_profile['inertia'],mode=mode,max_step=max_step)
        delta = prediction['delta']
        mdelta = prediction['M2delta']
        moment = propagate_orientation_second_moments(initial_moment,mdelta)
        q = quaternion_multiply(parent[None,:,None,:], delta[:,None,:,:]).reshape(len(times),-1,4)
        weights = (parent_weights[:,None]*np.asarray(prediction['weights'])[None,:]).reshape(-1)
        representative, mean, cov, explicit = finite_law_statistics(q,weights,moment,previous_rpy)
        moment_residuals.append(float(np.max(np.abs(explicit-moment))))
        keep = slice(None) if block==len(observations)-1 else slice(None,-1)
        values = dict(time=times,quaternion_second_moment=moment,
            representative_quaternion_wxyz=representative,rpy_mean_rad=mean,rpy_covariance_rad2=cov,
            omega_mean_rad_s=prediction['mean_omega'],omega_covariance_rad2_s2=prediction['covariance_omega'],
            block_index=np.full(len(times),block,dtype=int))
        for name,value in values.items(): result[name].append(value[keep])
        for name,value in dict(time=times,delta_wxyz=delta,omega=prediction['omega'],
            parent_wxyz=parent,parent_weight=parent_weights).items():
            details[f'block_{block}_{name}'] = value
        parent = q[-1]
        parent_weights = weights
        initial_moment = moment[-1]
        previous_rpy = mean[-1]
    result = {name:np.concatenate(value) for name,value in result.items()}
    result.update(details)
    result.update(final_law_quaternion_wxyz=parent,final_law_weights=parent_weights,
                  core_product_vs_enumeration_max_abs=np.asarray(moment_residuals))
    return result


def reference_rpy(data, query):
    rotations = Slerp(data['mocap_time'],Rotation.from_quat(data['mocap_quaternion_xyzw']))(query)
    raw = rotations.as_euler('xyz')
    initial = Rotation.from_quat(data['initial_quaternion_wxyz'][[1,2,3,0]]).as_euler('xyz')
    rpy = np.unwrap(np.vstack((initial,raw)),axis=0)[1:]
    return rpy, rotations


def errors(data, predicted):
    rpy, reference = reference_rpy(data,predicted['time'])
    mean = predicted['rpy_mean_rad']
    difference = (mean-rpy+np.pi)%(2*np.pi)-np.pi
    angular = (reference.inv()*Rotation.from_euler('xyz',mean)).magnitude()
    std = np.sqrt(np.diagonal(predicted['rpy_covariance_rad2'],axis1=1,axis2=2))
    # Plot/empirical comparisons, not a calibration guarantee for +/-2sigma.
    scores = dict(rpy_rmse_deg=np.rad2deg(np.sqrt(np.mean(difference**2,axis=0))),
        rpy_final_error_deg=np.rad2deg(difference[-1]),
        rpy_final_two_sigma_deg=np.rad2deg(2*std[-1]),
        geodesic_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(angular**2)))),
        geodesic_final_deg=float(np.rad2deg(angular[-1])),
        empirical_two_sigma_axis_fraction=np.mean(np.abs(difference)<=2*std,axis=0),
        max_core_product_enumeration_residual=float(np.max(predicted['core_product_vs_enumeration_max_abs'])))
    return dict(reference_rpy_rad=rpy,error_rpy_rad=difference,error_geodesic_rad=angular),scores


def run(args):
    from plot import render_case, render_overview
    args.output.mkdir(parents=True,exist_ok=True)
    data = dict(np.load(args.input,allow_pickle=False))
    input_metadata = json.loads(args.input.with_suffix('.json').read_text())
    torque = build_torque_profile(data,args.model)
    np.savez_compressed(args.output/'torque_profile.npz',**torque['arrays'])
    write_json(args.output/'torque_profile.json',torque['metadata'])
    started = time.monotonic()
    summaries = []
    for sigma in args.sigma_deg_s:
        label = format(sigma,'g').replace('.','p')
        destination = args.output/f'sigma_{label}_deg_s'
        destination.mkdir(exist_ok=True)
        cases = {}
        metrics = {}
        for mode in METHODS:
            print(f'sigma={sigma:g} deg/s, {mode}',flush=True)
            prediction = predict(data,torque,sigma,mode,max_step=args.max_step)
            reference,scores = errors(data,prediction)
            prediction.update(reference)
            np.savez_compressed(destination/(mode+'.npz'),**prediction)
            cases[mode] = prediction
            metrics[mode] = scores
        record = dict(sigma_deg_s=sigma,sigma_rad_s=float(np.deg2rad(sigma)),
            covariance_rad2_s2=np.eye(3)*np.deg2rad(sigma)**2,
            actual_observation_times=data['sparse_gyro_time'],
            observation_rate_hz=.5,measurement_values_randomly_perturbed=False,
            methods=metrics,band='mean +/- 2 weighted standard deviations of the finite cubature-product Euler law; not guaranteed credible intervals',
            uncertainty_scope='only independent individual sparse gyro candidates; conditional on nominal inertia, fixed actuator model, measured gimbal/command history and strictly-past residual torque mean')
        write_json(destination/'metrics.json',record)
        render_case(destination,data,torque,cases,record)
        summaries.append(record)
    render_overview(args.output,data,summaries)
    provenance = dict(schema='grape/nonlinear-sparse-gyro-dead-reckoning/v1',
        input_sha256=sha(args.input),input_metadata_sha256=sha(args.input.with_suffix('.json')),
        input_metadata=input_metadata,source_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
        model_sha256=sha(args.model),noise_standard_deviations_deg_s=args.sigma_deg_s,
        nominal_inertia_fixed=data['nominal_inertia'],
        thrust_lag_from_separate_published_prior_fit=True,
        torque_profile_provenance=torque['metadata'],
        core_product_source=dict(path=inspect.getsourcefile(quaternion_product_second_moment),
            sha256=sha(inspect.getsourcefile(quaternion_product_second_moment))),
        max_integrator_step_seconds=args.max_step,wall_seconds=time.monotonic()-started,
        initial_attitude='one mocap SLERP alignment, deterministic Dirac',
        attitude_observations_after_initialization=False,accel_or_magnetic_updates=False,
        recorded_control_replayed='same issued commands and measured gimbal history at all noise levels; controller is not rerun in a closed loop',
        gyro_refresh='independent fresh Gaussian measured input at each sparse timestamp; chapter-4 product-law approximation, not a joint Bayesian gyro update',
        within_interval='the same six Gaussian cubature points integrated to every intermediate time',
        uncertainty_representation='raw quaternion second moment via existing core product formula; finite product-law Euler moments by deterministic enumeration, no Bingham projection needed',
        no_monte_carlo=True,hidden_dense_evaluation_gyro_used_for_prediction=False,
        integration='RK4 for each six-point Euler/quaternion system, normalized quaternion after each finite step; finite step/interpolation/cubature errors remain',
        methods=METHODS,results=summaries)
    write_json(args.output/'summary.json',provenance)
    print('Completed:',args.output,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('/tmp/nonlinear-gyro-dead-reckoning-20261113/input.npz'))
    parser.add_argument('--output',type=Path,default=Path('/tmp/nonlinear-gyro-dead-reckoning-20261113/results'))
    parser.add_argument('--model',type=Path,default=HERE.parent/'grape_vehicle_model.json')
    parser.add_argument('--sigma-deg-s',type=float,nargs='+',default=list(DEFAULT_NOISE))
    parser.add_argument('--max-step',type=float,default=.002)
    args=parser.parse_args()
    if any(s<=0 or not np.isfinite(s) for s in args.sigma_deg_s) or args.max_step<=0:
        parser.error('positive finite sigmas and integration step required')
    run(args)


if __name__=='__main__': main()
