"""Causal Bayes law with adaptive deterministic importance quadrature.

The posterior target is never replaced by the Gaussian used to place
integration nodes.  It always equals the original initial-rate Gaussian
times every acquired Gaussian gyro likelihood, each included once.  The
proposal affects numerical integration only.  All nodes are passed through
the same valid nonlinear controlled rotation flow; R/omega cross survives.
"""
from pathlib import Path
import importlib.util
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp

HERE=Path(__file__).resolve().parent


def _load(name,filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


_moments=_load('_adaptive_existing_joint_moments','engine.py')
_native=_load('_adaptive_existing_native_flow','native_flow.py')
gaussian_tensor_points=_moments.gaussian_tensor_points
summarize_joint=_moments.summarize_joint
ControlledFlow=_native.ControlledFlow


def gaussian_log_density(values,mean,covariance):
    values=np.asarray(values,float)
    root=np.linalg.cholesky(covariance)
    whitened=np.linalg.solve(root,(values-mean).T).T
    return -.5*np.sum(whitened**2,axis=-1)-1.5*np.log(2.*np.pi)-np.log(np.diag(root)).sum()


def _fit_proposal(flow,prior_mean,covariance,observation_times,observations,warm_start,max_nfev):
    """Finite deterministic local optimization used ONLY to place nodes."""
    root=np.linalg.cholesky(covariance)
    def residual(value):
        state=flow.evaluate(np.asarray(value)[None,:],observation_times)
        errors=np.vstack((value-prior_mean,state[1:,0,:3]-observations[1:]))
        return np.linalg.solve(root,errors.T).T.reshape(-1)
    candidates=[]
    for initial in (np.asarray(warm_start,float),prior_mean):
        result=least_squares(residual,initial,jac='3-point',max_nfev=max_nfev,
                             ftol=1e-9,xtol=1e-9,gtol=1e-9)
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun).all():
            raise FloatingPointError('Proposal optimization produced non-finite parameters')
        candidates.append(result)
    result=min(candidates,key=lambda item:float(np.dot(item.fun,item.fun)))
    precision=result.jac.T@result.jac
    # The original positive-definite Gaussian prior is included in the
    # residual/Jacobian. No regularizer, floor, or state covariance change.
    root_precision=np.linalg.cholesky(.5*(precision+precision.T))
    inverse_root=np.linalg.solve(root_precision,np.eye(3))
    proposal_covariance=inverse_root.T@inverse_root
    record=dict(placement='Gaussian importance proposal only; not posterior replacement',
        starts=['previous proposal center','original first-reading center'],
        chosen_cost=float(result.cost),chosen_optimality=float(result.optimality),
        chosen_status=int(result.status),chosen_success=bool(result.success),
        chosen_nfev=int(result.nfev),
        starts_cost=[float(r.cost) for r in candidates],
        starts_nfev=[int(r.nfev) for r in candidates],
        maximum_function_evaluations_per_start=int(max_nfev),
        finite_unconverged_proposal_does_not_reject_observation=True,
        jacobian='deterministic central differences (scipy least_squares 3-point)',
        covariance='inverse Gauss-Newton precision for proposal placement only')
    return result.x.copy(),proposal_covariance,record


def predict_adaptive(data,torque_profile,sigma_deg_s=5.,order=11,output_dt=.02,
                     max_step=.002,observation_sets=None,*,mode='nonlinear',
                     proposal_max_nfev=24):
    """Return online joint moments, using only already acquired likelihoods.

    Each observation changes a density over the SAME latent initial W0.
    Gaussian quadrature proposals move as that density concentrates; target
    evaluation includes the original prior and all past likelihood factors,
    divided by the proposal density. Re-evaluating a past factor at new
    numerical nodes is not multiplying that factor into the law twice.

    A previous interval's endpoint is retained as the published prior.
    The current posterior and future interval are generated using the new
    quadrature. Earlier online outputs are never overwritten. Differences
    between old-grid and new-grid evaluations of the same prior are saved
    as integration diagnostics, never used as rejection thresholds.
    """
    bounds=np.asarray(data['time'],float)
    if bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[1]<=bounds[0]:
        raise ValueError('Finite increasing experiment bounds required')
    start,end=bounds
    acquired=np.asarray(data['sparse_gyro_time'],float)
    measured=np.asarray(data['sparse_gyro'],float)
    if (acquired.ndim!=1 or not len(acquired) or measured.shape!=(len(acquired),3)
            or not np.isfinite(acquired).all() or not np.isfinite(measured).all()
            or acquired[0]!=start or acquired[-1]>end or np.any(np.diff(acquired)<=0.)):
        raise ValueError('Original increasing acquisition clock must start at initialization')
    sigma=float(sigma_deg_s)
    if not all(np.isfinite(v) and v>0 for v in (sigma,float(output_dt),float(max_step))):
        raise ValueError('Positive sigma/output interval/RK4 step required')
    if int(proposal_max_nfev)!=proposal_max_nfev or proposal_max_nfev<1:
        raise ValueError('Positive proposal iteration budget required')
    covariance=np.eye(3)*np.deg2rad(sigma)**2
    q0=np.asarray(data['initial_quaternion_wxyz'],float)
    cases={'default':measured} if observation_sets is None else dict(observation_sets)
    if not cases: raise ValueError('At least one observation case required')
    for name,observations in cases.items():
        observations=np.asarray(observations,float)
        if (observations.shape!=measured.shape or not np.isfinite(observations).all()
                or not np.array_equal(observations[0],measured[0])):
            raise ValueError('Cases must have finite measurements and the same first reading')
        cases[name]=observations
    grid=np.unique(np.r_[np.linspace(start,end,int(round((end-start)/output_dt))+1),acquired])
    fields=('time','event_side','acquisition_index','quaternion_second_moment','rotation_mean',
        'rotation_covariance_colmajor','omega_mean_rad_s','omega_covariance_rad2_s2',
        'rotation_omega_covariance_colmajor','rpy_proxy_rad','rpy_covariance_delta_rad2',
        'initial_omega_mean_rad_s','initial_omega_covariance_rad2_s2','core_vs_rotation_mean_max_abs')
    results={}
    for name,observations in cases.items():
        begun=time.monotonic()
        flow=ControlledFlow(torque_profile,q0,start,end,max_step=max_step,mode=mode)
        output={field:[] for field in fields}
        events=[];event_nodes=[];event_logs=[];event_prior_logs=[]
        event_proposal_mean=[];event_proposal_covariance=[]
        previous_center=measured[0].copy()
        published_prior=None
        prior_mean=measured[0]
        def append(stamp,side,acquisition,state,mass,nodes):
            moment=summarize_joint(state[:,3:],state[:,:3],mass,initial_nodes=nodes)
            moment.update(time=float(stamp),event_side=int(side),acquisition_index=int(acquisition))
            for key in fields: output[key].append(np.asarray(moment[key]).copy())
            return moment
        for acquisition,begin in enumerate(acquired):
            finish=acquired[acquisition+1] if acquisition+1<len(acquired) else end
            interval=grid[(grid>=begin)&(grid<=finish)]
            history=acquired[:acquisition+1]
            if acquisition==0:
                center=prior_mean.copy();proposal_covariance=covariance.copy()
                placement=dict(initialization=True,first_reading_used_once=True,
                    placement='initial posterior N(y0,R), no repeated y0 likelihood')
            else:
                center,proposal_covariance,placement=_fit_proposal(flow,prior_mean,covariance,
                    history,observations[:acquisition+1],previous_center,int(proposal_max_nfev))
            previous_center=center.copy()
            nodes,quadrature=gaussian_tensor_points(center,proposal_covariance,order)
            query=np.unique(np.r_[start,history,interval])
            state=flow.evaluate(nodes,query)
            history_indices=np.searchsorted(query,history)
            log_prior_density=gaussian_log_density(nodes,prior_mean,covariance)
            log_proposal=gaussian_log_density(nodes,center,proposal_covariance)
            raw_prior=np.log(quadrature)+log_prior_density-log_proposal
            for j in range(1,acquisition):
                raw_prior+=gaussian_log_density(state[history_indices[j],:,:3],observations[j],covariance)
            prior_normalizer=float(logsumexp(raw_prior))
            log_prior_mass=raw_prior-prior_normalizer
            raw_posterior=raw_prior.copy()
            if acquisition:
                raw_posterior+=gaussian_log_density(state[history_indices[-1],:,:3],
                                                     observations[acquisition],covariance)
            normalizer=float(logsumexp(raw_posterior))
            if not np.isfinite(normalizer):
                raise FloatingPointError('No finite adaptive importance normalizer')
            log_mass=raw_posterior-normalizer
            mass=np.exp(log_mass)
            # Avoid replacing the inherited law by proposal mean/covariance:
            # moments ALWAYS use these full target/proposal weights.
            event_nodes.append(nodes.copy());event_logs.append(log_mass.copy())
            event_prior_logs.append(log_prior_mass.copy())
            event_proposal_mean.append(center.copy())
            event_proposal_covariance.append(proposal_covariance.copy())
            current=state[history_indices[-1]]
            posterior=append(begin,1,acquisition,current,mass,nodes)
            if acquisition==0:
                event=dict(acquisition_index=0,time=float(begin),initialization=True,
                    first_reading_used_once=True,likelihood_applications=0,
                    log_predictive_evidence=None,proposal=placement)
            else:
                regridded=summarize_joint(current[:,3:],current[:,:3],np.exp(log_prior_mass),initial_nodes=nodes)
                event=dict(acquisition_index=int(acquisition),time=float(begin),initialization=False,
                    likelihood_applications=1,target_likelihood_factor_count=int(acquisition),
                    log_predictive_evidence=normalizer-prior_normalizer,
                    innovation_rad_s=(observations[acquisition]-published_prior['omega_mean_rad_s']).tolist(),
                    prior_omega_mean_rad_s=published_prior['omega_mean_rad_s'].tolist(),
                    posterior_omega_mean_rad_s=posterior['omega_mean_rad_s'].tolist(),
                    prior_omega_covariance_rad2_s2=published_prior['omega_covariance_rad2_s2'].tolist(),
                    posterior_omega_covariance_rad2_s2=posterior['omega_covariance_rad2_s2'].tolist(),
                    regridded_prior_omega_mean_rad_s=regridded['omega_mean_rad_s'].tolist(),
                    regridded_prior_omega_covariance_rad2_s2=regridded['omega_covariance_rad2_s2'].tolist(),
                    regridded_prior_mean_max_difference_rad_s=float(np.max(np.abs(
                        regridded['omega_mean_rad_s']-published_prior['omega_mean_rad_s']))),
                    regridded_prior_covariance_max_difference_rad2_s2=float(np.max(np.abs(
                        regridded['omega_covariance_rad2_s2']-published_prior['omega_covariance_rad2_s2']))),
                    minimum_quadrature_mass=float(mass.min()),maximum_quadrature_mass=float(mass.max()),
                    inverse_squared_mass_sum=float(1./np.dot(mass,mass)),
                    zero_mass_from_float_exp_count=int(np.count_nonzero(mass==0.)),
                    quadrature_concentration_is_not_an_acceptance_gate=True,
                    full_target_log_normalizer_diagnostic=normalizer,proposal=placement)
            events.append(event)
            for stamp in interval[1:]:
                row=int(np.searchsorted(query,stamp))
                next_acquisition=(acquisition+1 if acquisition+1<len(acquired) and stamp==finish else -1)
                side=-1 if next_acquisition>=0 else 0
                moment=append(stamp,side,next_acquisition,state[row],mass,nodes)
                if side==-1: published_prior=moment
        arrays={key:np.stack(value) for key,value in output.items()}
        arrays['rpy_proxy_rad']=np.unwrap(arrays['rpy_proxy_rad'],axis=0)
        arrays.update(event_time=acquired.copy(),event_initial_omega_nodes_rad_s=np.stack(event_nodes),
            initial_omega_nodes_rad_s=event_nodes[0].copy(),
            initial_quadrature_weights=gaussian_tensor_points(prior_mean,covariance,order)[1],
            event_proposal_mean_rad_s=np.stack(event_proposal_mean),
            event_proposal_covariance_rad2_s2=np.stack(event_proposal_covariance),
            event_log_weights=np.stack(event_logs),event_prior_log_weights=np.stack(event_prior_logs),
            measurement_covariance_rad2_s2=covariance.copy(),events=events,
            metadata=dict(schema='grape/adaptive-sequential-joint-bayes-rotation/v1',
                posterior='original initial Gaussian times every acquired Gaussian likelihood once',
                latent_variable='same initial body angular velocity W0, fixed q0/J/known torque',
                quadrature='adaptive deterministic tensor Gauss-Hermite importance integration',
                quadrature_order_per_axis=int(order),quadrature_nodes_per_event=int(order)**3,
                proposal_is_not_posterior_state=True,
                proposal_density_divided_out=True,past_likelihoods_reevaluated_not_double_counted=True,
                event_prior_weights='Previous target evaluated on current numerical proposal nodes; prior display endpoint retained from previous interval',
                noise_model='independent per-reading isotropic Gaussian; no process noise or bias state',
                first_reading='N(y0,R) once; y0 not multiplied again',
                flow='native C++ controlled nonlinear Euler/quaternion RK4, identical equations to Python reference',
                native_source_sha256=__import__('hashlib').sha256((HERE/'native_flow.cpp').read_bytes()).hexdigest(),
                maximum_rk4_step_seconds=float(max_step),native_steps_all_queries=int(flow.total_steps),
                maximum_quaternion_norm_correction=float(flow.maximum_norm_correction),
                monte_carlo=False,node_pruning=False,covariance_floor=False,
                earlier_online_outputs_rewritten=False,wall_seconds=time.monotonic()-begun))
        results[name]=arrays
    return results
