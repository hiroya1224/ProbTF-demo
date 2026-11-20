"""Current-rate Markov Bayes filtering with Student-t gyro and Gaussian Q.

Only the omega integral uses deterministic quadrature. Quaternion second
moments and the full conditional rotation covariance travel with the same
omega candidates. Each prediction is a Gaussian omega transition mixture;
its product with the current Student-t factor is integrated without replacing
the posterior by a Gaussian. Repeated Nyström integration is an explicit
finite numerical approximation, not Monte Carlo or a covariance adjustment.
"""
from pathlib import Path
import hashlib
import importlib.util
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp


HERE = Path(__file__).resolve().parent
SHARED = HERE.parent/'synthetic_controlled_rotation/sequential_bayes_spikes'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


conditional = _load('_real_process_conditional_moments', SHARED/'process_noise_engine.py')
student = _load('_real_process_student_observation', SHARED/'student_t_observation.py')
gaussian_tensor_points = conditional.gaussian_tensor_points
mixture_moments = conditional.mixture_moments
transport_conditionals = conditional.transport_conditionals
ControlledFlow = conditional.ControlledFlow


def _fit_student_proposal(mean, covariance, observation, observation_covariance, student_df):
    """Use an approximate Gaussian prior ONLY to place integration points."""
    root_prior = np.linalg.cholesky(covariance)
    root_scale = np.linalg.cholesky(student.student_t_scale(observation_covariance,student_df))
    nu = float(student_df)
    def residual(value):
        prior = np.linalg.solve(root_prior,value-mean)
        white = np.linalg.solve(root_scale,value-observation)
        squared = float(white@white)
        ratio = ((nu+3.)/nu if squared == 0. else
                 (nu+3.)*np.log1p(squared/nu)/squared)
        return np.r_[prior,np.sqrt(ratio)*white]
    result = least_squares(residual,mean,jac='3-point',max_nfev=24,
                           ftol=1e-9,xtol=1e-9,gtol=1e-9)
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun).all():
        raise FloatingPointError('Nonfinite Student-t numerical proposal')
    precision = result.jac.T@result.jac
    root_precision = np.linalg.cholesky(.5*(precision+precision.T))
    inverse = np.linalg.solve(root_precision,np.eye(3))
    return result.x.copy(), inverse.T@inverse, dict(
        placement='Moment-Gaussian prior and multivariate Student-t fit for integration placement only; actual mixture times Student-t divided by proposal',
        cost=float(result.cost),optimality=float(result.optimality),nfev=int(result.nfev),
        success=bool(result.success),status=int(result.status),
        target_replaced_by_proposal=False,finite_unconverged_proposal_does_not_reject_observation=True)


def density_quadrature(weights, omega_means, omega_covariance,
        quaternion_moments, rotation_covariances, measurement,
        measurement_covariance, student_df=4., order=7,
        *, proposal_mean=None, proposal_covariance=None):
    """Integrate [sum_i w_i N(omega; f_i,Q)] t(y;omega), retaining q|omega."""
    mass = conditional._normalized(weights)
    prior = mixture_moments(mass,omega_means,omega_covariance,
                            quaternion_moments,rotation_covariances)
    if proposal_mean is None:
        center,proposal_covariance,placement = _fit_student_proposal(
            prior['omega_mean_rad_s'],prior['omega_covariance_rad2_s2'],
            np.asarray(measurement),measurement_covariance,student_df)
    else:
        center = np.asarray(proposal_mean,float)
        proposal_covariance = np.asarray(proposal_covariance,float)
        placement = dict(placement='Explicit saved Gaussian integration proposal',target_replaced_by_proposal=False)
    nodes,base_weights = gaussian_tensor_points(center,proposal_covariance,order)
    with np.errstate(divide='ignore'):
        log_mass = np.log(mass)
    log_components = conditional._log_gaussian(nodes[:,None,:],np.asarray(omega_means)[None,:,:],omega_covariance)
    log_components += log_mass[None,:]
    log_density = logsumexp(log_components,axis=1)
    responsibilities = np.exp(log_components-log_density[:,None])
    conditional_moments = (responsibilities@np.asarray(quaternion_moments).reshape(len(mass),16)).reshape(-1,4,4)
    means_r = conditional.conditional_rotation_means(quaternion_moments)
    conditional_mean_r = conditional.conditional_rotation_means(conditional_moments)
    covariances = (responsibilities@np.asarray(rotation_covariances).reshape(len(mass),81)).reshape(-1,9,9)
    # Centered total covariance, identical to the established Q engine.
    # Positive weighted Gram terms avoid cancellation in nearly rigid laws.
    for lower in range(0,len(nodes),64):
        upper = min(lower+64,len(nodes))
        centered = means_r[None,:,:]-conditional_mean_r[lower:upper,None,:]
        covariances[lower:upper] += np.einsum('ji,jia,jib->jab',
            responsibilities[lower:upper],centered,centered,optimize=True)
    covariances = .5*(covariances+covariances.swapaxes(1,2))
    log_likelihood = student.student_t_log_density(nodes,np.asarray(measurement),
                                                   measurement_covariance,student_df)
    log_base = np.log(base_weights)+log_density-conditional._log_gaussian(nodes,center,proposal_covariance)
    raw = log_base+log_likelihood
    log_normalizer = float(logsumexp(raw))
    if not np.isfinite(log_normalizer):
        raise FloatingPointError('Undefined Student-t process posterior normalizer')
    posterior_weights = np.exp(raw-log_normalizer)
    statistics = mixture_moments(posterior_weights,nodes,np.zeros((3,3)),
                                 conditional_moments,covariances)
    # Integrating the prior with these points is a diagnostic of the chosen
    # numerical region, never a reason to drop an observation or alter Q.
    prior_normalizer = float(logsumexp(log_base))
    prior_weights = np.exp(log_base-prior_normalizer)
    prior_numeric = mixture_moments(prior_weights,nodes,np.zeros((3,3)),
                                    conditional_moments,covariances)
    diagnostic = dict(log_predictive_evidence=log_normalizer,
        numerical_prior_log_integral=prior_normalizer,
        numerical_prior_integral_minus_one=float(np.expm1(prior_normalizer)),
        numerical_weight_effective_count=float(1/np.sum(posterior_weights**2)),
        zero_posterior_weight_count=int(np.count_nonzero(posterior_weights==0.)),
        maximum_responsibility_sum_error=float(np.max(abs(responsibilities.sum(axis=1)-1.))),
        finite_diagnostic_only=True,observation_rejected=False,
        gaussian_proposal_is_not_posterior=True,proposal=placement)
    for key in ('omega_mean_rad_s','omega_covariance_rad2_s2','quaternion_second_moment',
                'rotation_mean','rotation_covariance_colmajor','rotation_omega_covariance_colmajor'):
        diagnostic['prior_quadrature_maximum_'+key+'_difference'] = float(np.max(abs(prior_numeric[key]-prior[key])))
    return dict(nodes=nodes,weights=posterior_weights,quaternion_moments=conditional_moments,
        rotation_covariances=covariances,statistics=statistics,diagnostics=diagnostic,
        proposal_mean=center,proposal_covariance=proposal_covariance,
        base_weights=base_weights,log_unnormalized_weights=raw)


def predict_process_student(data,physical,sigma_deg_s=5.,process_sigma_deg_s=1.,
        student_df=4.,order=7,max_step=.002,mode='nonlinear',output_dt=.01):
    """Add Q once before each native gyro likelihood; keep dense physical law."""
    begun = time.monotonic()
    stamps = np.asarray(data['sparse_gyro_time'],float)
    observations = np.asarray(data['sparse_gyro'],float)
    start,end = np.asarray(data['time'],float)
    sigma,process_sigma,nu = float(sigma_deg_s),float(process_sigma_deg_s),float(student_df)
    if (stamps.ndim!=1 or len(stamps)<2 or observations.shape!=(len(stamps),3)
            or not np.isfinite(stamps).all() or not np.isfinite(observations).all()
            or np.any(np.diff(stamps)<=0.) or stamps[0]!=start or stamps[-1]>end
            or not np.isfinite([sigma,process_sigma,nu,output_dt]).all()
            or sigma<=0. or process_sigma<=0. or nu<=2. or output_dt<=0.):
        raise ValueError('Finite native clock and positive noise scales / Student-t df>2 required')
    q0 = np.asarray(data['initial_quaternion_wxyz'],float)
    if q0.shape!=(4,) or not np.isfinite(q0).all() or np.linalg.norm(q0)==0.:
        raise ValueError('Finite nonzero initial quaternion required')
    q0 = q0/np.linalg.norm(q0)
    r = np.eye(3)*np.deg2rad(sigma)**2
    q = np.eye(3)*np.deg2rad(process_sigma)**2
    nodes,mass = gaussian_tensor_points(observations[0],r,order)
    moments = np.repeat(np.outer(q0,q0)[None,:,:],len(nodes),axis=0)
    rotation_covariances = np.zeros((len(nodes),9,9))
    output,diagnostics,events = {},[],[]
    grid = start+np.arange(int(np.floor((end-start)/output_dt))+1)*output_dt
    grid = np.unique(np.r_[grid[grid<=end],end])
    total_steps,maximum_norm,last_progress = 0,0.,-1
    final_packet = {}
    def append(stamp,index,side,statistics,process_count):
        statistics = dict(statistics,time=np.asarray(stamp),event_side=np.asarray(side),
            acquisition_index=np.asarray(index),
            target_likelihood_factor_count=np.asarray(index-1 if side<0 else index),
            process_application_count=np.asarray(process_count))
        for key,value in statistics.items():
            output.setdefault(key,[]).append(np.asarray(value).copy())
    initial = mixture_moments(np.ones(1),observations[:1],r,
        np.outer(q0,q0)[None,:,:],np.zeros((1,9,9)))
    append(start,0,1,initial,0)
    for index in range(1,len(stamps)+1):
        is_tail = index==len(stamps)
        lower = stamps[index-1]
        upper = end if is_tail else stamps[index]
        dense_times = grid[(grid>lower)&(grid<=upper if is_tail else grid<upper)]
        if upper==lower:
            continue
        query = np.unique(np.r_[lower,dense_times,upper])
        flow = ControlledFlow(physical,np.array([1.,0.,0.,0.]),lower,upper,
                              max_step=max_step,mode=mode)
        states = flow.evaluate(nodes,query)
        for stamp,state in zip(query[1:],states[1:]):
            propagated_moments,propagated_covariances = transport_conditionals(
                moments,rotation_covariances,state[:,3:])
            if stamp<upper or is_tail:
                prediction = mixture_moments(mass,state[:,:3],np.zeros((3,3)),
                    propagated_moments,propagated_covariances)
                append(stamp,index-1,0,prediction,index-1)
            if stamp!=upper or is_tail:
                continue
            prior = mixture_moments(mass,state[:,:3],q,
                propagated_moments,propagated_covariances)
            append(upper,index,-1,prior,index)
            rule = density_quadrature(mass,state[:,:3],q,
                propagated_moments,propagated_covariances,observations[index],r,
                student_df=nu,order=order)
            append(upper,index,1,rule['statistics'],index)
            diagnostics.append(dict(acquisition_index=index,time=float(upper),**rule['diagnostics']))
            events.append(dict(acquisition_index=index,time=float(upper),likelihood_applications=1,
                process_noise_application_count=index,process_covariance_rad2_s2=q.tolist()))
            if index==len(stamps)-1:
                final_packet = dict(final_prior_component_weights=mass.copy(),
                    final_prior_component_omega_means_rad_s=state[:,:3].copy(),
                    final_prior_common_omega_covariance_rad2_s2=q.copy(),
                    final_prior_conditional_quaternion_second_moment=propagated_moments.copy(),
                    final_prior_conditional_rotation_covariance_colmajor=propagated_covariances.copy(),
                    final_measurement_rad_s=observations[index].copy(),
                    final_proposal_mean_rad_s=rule['proposal_mean'].copy(),
                    final_proposal_covariance_rad2_s2=rule['proposal_covariance'].copy(),
                    final_quadrature_nodes_rad_s=rule['nodes'].copy(),
                    final_quadrature_base_weights=rule['base_weights'].copy(),
                    final_quadrature_posterior_weights=rule['weights'].copy(),
                    final_posterior_time=np.asarray(upper),
                    final_conditional_quaternion_second_moment=rule['quaternion_moments'].copy(),
                    final_conditional_rotation_covariance_colmajor=rule['rotation_covariances'].copy())
            nodes,mass = rule['nodes'],rule['weights']
            moments,rotation_covariances = rule['quaternion_moments'],rule['rotation_covariances']
        total_steps += flow.total_steps
        maximum_norm = max(maximum_norm,flow.maximum_norm_correction)
        progress = int(upper-start)
        if progress>last_progress:
            last_progress=progress
            print(f'Student-t Q sigma {process_sigma:g} {mode} GH{order}: '
                  f't={upper-start:.3f}s, wall={time.monotonic()-begun:.1f}s',flush=True)
    arrays = {key:np.stack(value) for key,value in output.items()}
    arrays['rpy_proxy_rad'] = np.unwrap(arrays['rpy_proxy_rad'],axis=0)
    arrays.update(final_packet,event_time=stamps.copy(),dense_output_grid_time=grid,
        measurement_covariance_rad2_s2=r,process_covariance_rad2_s2=q,
        process_noise_application_count=np.asarray(len(stamps)-1),
        process_increment_times=stamps[1:].copy(),
        process_sigma_deg_s=np.asarray(process_sigma),observation_student_df=np.asarray(nu),
        observation_student_scale_rad2_s2=student.student_t_scale(r,nu),
        diagnostics=diagnostics,events=events,
        metadata=dict(schema='grape/real-student-markov-process-moments/v1',
            model=mode,observation_count=len(stamps),likelihood_count=len(stamps)-1,
            process_increment_count=len(stamps)-1,process_sigma_deg_s=process_sigma,
            process_covariance_rad2_s2=q.tolist(),student_df=nu,
            quadrature_order_per_axis=int(order),quadrature_nodes_per_step=int(order)**3,
            measurement_covariance_rad2_s2=r.tolist(),
            process_time_convention='Independent endpoint omega increment per native acquisition interval, before its likelihood; no increment at RK4 substeps/dense queries/final tail; not a continuous-time PSD',
            posterior='Continuous Gaussian omega transition mixture times current multivariate Student-t likelihood; prior information retained by Markov density integration',
            finite_representation='Deterministic tensor-GH Nystrom integration of current omega density; conditional quaternion second moments and full rotation covariance retained; posterior is not Gaussian',
            initial_reading_used_once=True,gaussian_proposal_is_not_posterior=True,
            future_observations_used=False,monte_carlo=False,covariance_floor_or_jitter=False,
            finite_diagnostics_reject_observations=False,core_changed=False,
            dense_output=dict(grid_count=len(grid),output_dt_seconds=float(output_dt),
                final_acquisition_time=float(stamps[-1]),final_unobserved_prediction_seconds=float(end-stamps[-1]),
                process_increments_at_dense_queries=False,interpolation_or_flat_tail_hold=False),
            native_step_count=int(total_steps),maximum_quaternion_norm_correction=float(maximum_norm),
            wall_seconds=time.monotonic()-begun,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return arrays
