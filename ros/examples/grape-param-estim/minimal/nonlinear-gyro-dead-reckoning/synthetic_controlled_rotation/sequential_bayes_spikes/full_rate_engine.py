"""Every-reading Bayesian updates with causal numerical proposal refreshes.

The target remains the original W0 Gaussian times ALL acquired likelihoods.
Only integration-node placement is refreshed; a proposal is never substituted
for a posterior state. Prefix likelihoods are accumulated in native memory
without retaining the complete history-by-node tensor.
"""
from pathlib import Path
import hashlib
import importlib.util
import sys
import time

import numpy as np
from scipy.special import logsumexp

HERE=Path(__file__).resolve().parent


def _load(name,filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


_adaptive=_load('_full_rate_existing_adaptive','adaptive_engine.py')
_native=_load('_full_rate_prefix_native','full_rate_flow.py')
gaussian_tensor_points=_adaptive.gaussian_tensor_points
summarize_joint=_adaptive.summarize_joint
gaussian_log_density=_adaptive.gaussian_log_density
FullRateFlow=_native.FullRateFlow


def predict_full_rate(data,torque_profile,sigma_deg_s=5.,order=11,max_step=.002,
        mode='nonlinear',refresh_seconds=.1,proposal_max_nfev=24,
        initial_dense_seconds=.25,reanchor_displacement_sigma=1.,dense_startup_seconds=None):
    """Return one saved-moment case, with pre/post at EVERY acquired time.

    Both scheduled and displacement-triggered proposal refreshes evaluate
    the same causal target. A finite numerical discrepancy cannot reject an
    observation. Earlier published endpoints are never rewritten.
    """
    begun=time.monotonic();bounds=np.asarray(data['time'],float)
    acquired=np.asarray(data['sparse_gyro_time'],float)
    observed=np.asarray(data['sparse_gyro'],float)
    sigma=float(sigma_deg_s);refresh_seconds=float(refresh_seconds)
    if dense_startup_seconds is not None:initial_dense_seconds=dense_startup_seconds
    initial_dense_seconds=float(initial_dense_seconds)
    if (bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[1]<=bounds[0]
            or acquired.ndim!=1 or not len(acquired) or observed.shape!=(len(acquired),3)
            or not np.isfinite(acquired).all() or not np.isfinite(observed).all()
            or acquired[0]!=bounds[0] or acquired[-1]>bounds[1] or np.any(np.diff(acquired)<=0.)
            or not np.isfinite(sigma) or sigma<=0. or not np.isfinite(refresh_seconds) or refresh_seconds<=0.
            or not np.isfinite(initial_dense_seconds) or initial_dense_seconds<0.):
        raise ValueError('Finite increasing full-rate observations and positive settings required')
    if reanchor_displacement_sigma is not None:
        reanchor_displacement_sigma=float(reanchor_displacement_sigma)
        if not np.isfinite(reanchor_displacement_sigma) or reanchor_displacement_sigma<=0.:
            raise ValueError('Positive finite proposal-displacement threshold, or None, required')
    covariance=np.eye(3)*np.deg2rad(sigma)**2
    prior_mean=observed[0].copy();q0=np.asarray(data['initial_quaternion_wxyz'],float)
    flow=FullRateFlow(torque_profile,q0,*bounds,max_step=max_step,mode=mode)
    # Every sample in the early interval is cheap to reanchor. Subsequently
    # the schedule is defined from clock values, independently of readings.
    schedule=np.unique(np.r_[0,np.flatnonzero(acquired<=bounds[0]+initial_dense_seconds),
        np.searchsorted(acquired,np.arange(bounds[0],acquired[-1]+refresh_seconds,refresh_seconds))]).astype(int)
    schedule=schedule[schedule<len(acquired)]
    fields=('time','event_side','acquisition_index','quaternion_second_moment','rotation_mean',
        'rotation_covariance_colmajor','omega_mean_rad_s','omega_covariance_rad2_s2',
        'rotation_omega_covariance_colmajor','rpy_proxy_rad','rpy_covariance_delta_rad2',
        'initial_omega_mean_rad_s','initial_omega_covariance_rad2_s2','core_vs_rotation_mean_max_abs',
        'target_likelihood_factor_count')
    output={key:[] for key in fields};refresh_records=[]
    refresh_indices=[];refresh_means=[];refresh_covariances=[];refresh_logs=[];refresh_prior_logs=[]
    prior_log_evidence=np.full(len(acquired),np.nan)
    displacement_at_observation=np.zeros(len(acquired));refresh_at_observation=np.zeros(len(acquired),bool)
    last_center=prior_mean.copy();published_prior=None;anchor=0;reason='initial';trigger_displacement=0.
    last_progress=-1
    def append(index,side,state,mass,nodes):
        moment=summarize_joint(state[:,3:],state[:,:3],mass,initial_nodes=nodes)
        count=max(0,index-1) if side<0 else index
        moment.update(time=float(acquired[index]),event_side=int(side),acquisition_index=int(index),
            target_likelihood_factor_count=int(count))
        for key in fields:output[key].append(np.asarray(moment[key]).copy())
        return moment
    while True:
        future=schedule[schedule>anchor]
        finish=int(future[0]) if len(future) else len(acquired)-1
        finish_is_refresh=bool(len(future))
        if anchor==0:
            center=prior_mean.copy();proposal_covariance=covariance.copy()
            placement=dict(initialization=True,first_reading_used_once=True)
        else:
            center,proposal_covariance,placement=_adaptive._fit_proposal(flow,prior_mean,covariance,
                acquired[:anchor+1],observed[:anchor+1],last_center,int(proposal_max_nfev))
        last_center=center.copy();root_proposal=np.linalg.cholesky(proposal_covariance)
        nodes,quadrature=gaussian_tensor_points(center,proposal_covariance,order)
        past_likelihood,states=flow.prefix_and_block(nodes,acquired,observed,anchor,finish,covariance)
        raw=np.log(quadrature)+gaussian_log_density(nodes,prior_mean,covariance)
        raw-=gaussian_log_density(nodes,center,proposal_covariance)
        raw+=past_likelihood
        normalizer=float(logsumexp(raw))
        if not np.isfinite(normalizer):raise FloatingPointError('Undefined full-rate prefix normalizer')
        prior_logs=raw-normalizer
        current=states[0]
        if anchor:
            regridded=summarize_joint(current[:,3:],current[:,:3],np.exp(prior_logs),initial_nodes=nodes)
            prior_mean_difference=float(np.max(abs(regridded['omega_mean_rad_s']-published_prior['omega_mean_rad_s'])))
            prior_covariance_difference=float(np.max(abs(regridded['omega_covariance_rad2_s2']-published_prior['omega_covariance_rad2_s2'])))
            raw+=gaussian_log_density(current[:,:3],observed[anchor],covariance)
        else:prior_mean_difference=prior_covariance_difference=0.
        posterior_normalizer=float(logsumexp(raw))
        if not np.isfinite(posterior_normalizer):raise FloatingPointError('Undefined full-rate posterior normalizer')
        logs=raw-posterior_normalizer;mass=np.exp(logs)
        prior_log_evidence[anchor]=posterior_normalizer-normalizer if anchor else 0.
        posterior=append(anchor,1,current,mass,nodes)
        refresh_at_observation[anchor]=True
        refresh_indices.append(anchor);refresh_means.append(center.copy());refresh_covariances.append(proposal_covariance.copy())
        refresh_logs.append(logs.copy());refresh_prior_logs.append(prior_logs.copy())
        refresh_records.append(dict(acquisition_index=int(anchor),time=float(acquired[anchor]),reason=reason,
            trigger_displacement_sigma=float(trigger_displacement),proposal=placement,
            regridded_prior_mean_max_difference_rad_s=prior_mean_difference,
            regridded_prior_covariance_max_difference_rad2_s2=prior_covariance_difference))
        if anchor==len(acquired)-1:break
        next_anchor=finish;next_reason='scheduled';next_displacement=0.;completed_final=False
        for index in range(anchor+1,finish+1):
            current=states[index-anchor]
            published_prior=append(index,-1,current,mass,nodes)
            if index==finish and finish_is_refresh:break
            likelihood=gaussian_log_density(current[:,:3],observed[index],covariance)
            update=logs+likelihood;log_normalizer=float(logsumexp(update))
            if not np.isfinite(log_normalizer):raise FloatingPointError('Undefined full-rate reading normalizer')
            updated_logs=update-log_normalizer;updated_mass=np.exp(updated_logs)
            candidate_mean=updated_mass@nodes
            displacement=float(np.max(abs(np.linalg.solve(root_proposal,candidate_mean-center))))
            displacement_at_observation[index]=displacement
            if reanchor_displacement_sigma is not None and displacement>reanchor_displacement_sigma:
                # This preliminary rule only chooses a better integration
                # grid. The new target re-evaluates every factor once before
                # the current posterior is published; no data are rejected.
                next_anchor=index;next_reason='numerical_displacement';next_displacement=displacement
                break
            logs=updated_logs;mass=updated_mass
            prior_log_evidence[index]=log_normalizer
            append(index,1,current,mass,nodes)
            if index==len(acquired)-1:completed_final=True
        if completed_final:break
        anchor=next_anchor;reason=next_reason;trigger_displacement=next_displacement
        progress=int(acquired[anchor]-bounds[0])
        if progress>last_progress:
            print(f'Full-rate {mode} order {order}: t={acquired[anchor]:.3f}s, '
                f'{len(refresh_indices)} proposal refreshes, wall={time.monotonic()-begun:.1f}s',flush=True)
            last_progress=progress
    arrays={key:np.stack(value) for key,value in output.items()}
    arrays['rpy_proxy_rad']=np.unwrap(arrays['rpy_proxy_rad'],axis=0)
    arrays.update(event_time=acquired.copy(),measurement_covariance_rad2_s2=covariance.copy(),
        initial_omega_nodes_rad_s=gaussian_tensor_points(prior_mean,covariance,order)[0],
        initial_quadrature_weights=gaussian_tensor_points(prior_mean,covariance,order)[1],
        refresh_acquisition_index=np.asarray(refresh_indices,int),
        refresh_proposal_mean_rad_s=np.asarray(refresh_means),
        refresh_proposal_covariance_rad2_s2=np.asarray(refresh_covariances),
        refresh_log_weights=np.asarray(refresh_logs),refresh_prior_log_weights=np.asarray(refresh_prior_logs),
        observation_log_predictive_evidence=prior_log_evidence,
        observation_proposal_refresh=refresh_at_observation,
        observation_proposal_displacement_sigma=displacement_at_observation,
        refresh_records=refresh_records,
        metadata=dict(schema='grape/full-rate-causal-bayes-rotation/v1',mode=mode,
            posterior='initial N(W0;y0,R) times ALL acquired j>=1 Gaussian gyro likelihoods once',
            acquired_observation_count=len(acquired),likelihood_count=len(acquired)-1,
            quadrature_order_per_axis=int(order),quadrature_nodes_per_refresh=int(order)**3,
            scheduled_refresh_seconds=refresh_seconds,initial_dense_seconds=initial_dense_seconds,
            reanchor_displacement_sigma=reanchor_displacement_sigma,
            actual_proposal_refresh_count=len(refresh_indices),
            proposal_is_not_posterior_state=True,proposal_density_divided_out=True,
            proposal_selection_uses_only_current_prefix=True,
            refresh_nodes_reconstruction='existing gaussian_tensor_points(refresh mean, covariance, order); no redundant per-observation node tensor',
            previous_online_outputs_rewritten=False,every_observation_updated=True,
            numerical_regridding_is_not_observation_rejection=True,
            monte_carlo=False,node_pruning=False,covariance_floor=False,process_noise=False,
            full_joint_quaternion_omega_law=True,
            native_step_count=int(flow.total_steps),maximum_quaternion_norm_correction=float(flow.maximum_norm_correction),
            native_source_sha256=hashlib.sha256((HERE/'full_rate_flow.cpp').read_bytes()).hexdigest(),
            included_physical_source_sha256=hashlib.sha256((HERE/'native_flow.cpp').read_bytes()).hexdigest(),
            wall_seconds=time.monotonic()-begun))
    return arrays
