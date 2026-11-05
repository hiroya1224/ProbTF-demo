#!/usr/bin/env python3
"""Post-hoc integration checks of saved, unchanged quadratic/quartic laws.

This never re-estimates a trajectory or changes a saved attitude law. Multiple
local integration charts, including duplicate modes, are combined with a finer
global Haar rule by the balance-mixture importance denominator. This remains a
finite deterministic diagnostic, not a proof of discovering every possible mode.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.special import logsumexp

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT / 'ros/core/probtf_core/src'))
from probtf.bingham import bingham_log_normalizer
from translational_attitude_law import (
    ExponentialAttitudeLaw, _haar_grid, _hermite_grid,
    projective_basis, rotation_matrices,
)


def _local_log_density(q, reference, covariance):
    lower = np.linalg.cholesky(covariance)
    dots = q @ reference
    tangent = q @ projective_basis(reference)
    valid = dots != 0.
    z = np.zeros_like(tangent)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        z[valid] = tangent[valid] / dots[valid, None]
        white = np.linalg.solve(lower, z.T).T
        radius2 = np.einsum('ni,ni->n', z, z)
        mahalanobis = np.einsum('ni,ni->n', white, white)
        result = (-math.log(2.)-1.5*math.log(2.*math.pi)
            -float(np.log(np.diag(lower)).sum())-.5*mahalanobis
            +2.*np.log1p(radius2))
    result[~valid | ~np.isfinite(result)] = -np.inf
    return result


def multilocal_mis(law, reference, centers, order=7, radial=8, angles=24,
                  local_mixture_weight=.999999):
    """Integrate one even S3 law using all supplied local charts plus Haar."""
    alpha = float(local_mixture_weight)
    if not 0. < alpha < 1.:
        raise ValueError('Local mixture weight must lie strictly between zero and one')
    base, log_hermite = _hermite_grid(order)
    global_q, log_haar = _haar_grid(radial, angles)
    nodes, rule = [], []
    local_weight = alpha / len(centers) if centers else 0.
    for center, covariance in centers:
        lower = np.linalg.cholesky(covariance)
        z = math.sqrt(2.) * base @ lower.T
        radius2 = np.einsum('ni,ni->n', z, z)
        q = (center + z @ projective_basis(center).T) / np.sqrt(1.+radius2)[:, None]
        nodes.append(q)
        rule.append(math.log(local_weight)+log_hermite-1.5*math.log(math.pi))
    haar_weight = 1.-alpha if centers else 1.
    nodes.append(global_q)
    rule.append(math.log(haar_weight)+log_haar)
    q = np.concatenate(nodes)
    log_rule = np.concatenate(rule)
    component_densities = [np.full(len(q), math.log(haar_weight)-math.log(2.*math.pi**2))]
    for center, covariance in centers:
        component_densities.append(math.log(local_weight)+_local_log_density(q,center,covariance))
    log_mixture = logsumexp(np.stack(component_densities), axis=0)
    log_measure = log_rule-log_mixture
    relative = law.relative_log_density(q, reference)
    relative_logz = float(logsumexp(relative+log_measure))
    weight = np.exp(relative+log_measure-relative_logz)
    weight /= weight.sum()
    rotation = rotation_matrices(q)
    pitch = np.arctan2(-rotation[:,2,0], np.hypot(rotation[:,0,0],rotation[:,1,0]))
    mean_pitch = float(weight@pitch)
    return dict(logZ=float(law.log_density(reference))+relative_logz,
        Mqq=np.einsum('n,ni,nj->ij',weight,q,q),
        pitch_mean_rad=mean_pitch,
        pitch_variance_rad2=float(weight@((pitch-mean_pitch)**2)),
        local_centers=len(centers),local_nodes=len(centers)*len(base),
        global_nodes=len(global_q),total_nodes=len(q),
        integration_method='deterministic_multilocal_GH_and_global_Haar_MIS',
        local_mixture_weight=alpha if centers else 0.)


def diversified_projective_seeds():
    """48 fixed mode-search starts, never a probability-state approximation.

    The first 24 projective rotations are axes, signed coordinate pairs and
    half-hypercube vertices. The second copy has a fixed left quaternion
    rotation, independent of sensors, truth, or comparison outcomes.
    """
    seeds = [q for q in np.eye(4)]
    for i,j in itertools.combinations(range(4),2):
        for sign in (-1.,1.):
            q=np.zeros(4);q[i]=1./math.sqrt(2.);q[j]=sign/math.sqrt(2.)
            seeds.append(q)
    for signs in itertools.product((-1.,1.),repeat=3):
        seeds.append(np.r_[1.,signs]/2.)
    seeds=np.asarray(seeds)
    w,x,y,z=np.array([1.,2.,3.,4.])/math.sqrt(30.)
    left=np.array([[w,-x,-y,-z],[x,w,-z,y],[y,z,w,-x],[z,-y,x,w]])
    return np.concatenate((seeds,seeds@left.T))


def find_centers(law, stored_reference, max_seeds=6, *,
                 seed_strategy='diversified',seed_radial=8,seed_angles=24,
                 mode_iterations=20):
    seed_q, _ = _haar_grid(seed_radial,seed_angles)
    score = law.relative_log_density(seed_q, stored_reference)
    chosen = np.argsort(score)[-int(max_seeds):][::-1]
    seeds = [('stored_reference',np.asarray(stored_reference))]
    if seed_strategy=='diversified':
        seeds.extend(('fixed_projective_%d'%i,q)
                     for i,q in enumerate(diversified_projective_seeds()))
    elif seed_strategy!='haar_only':
        raise ValueError('Unknown mode-seed strategy')
    seeds.extend(('haar_rank_%d'%rank,seed_q[index])
                 for rank,index in enumerate(chosen,1))
    centers, diagnostics = [], []
    for origin,seed in seeds:
        entry = {'seed':origin}
        try:
            q, info = law.find_mode(seed,max_iterations=mode_iterations)
            entry.update(mode_wxyz=q.tolist(),optimizer=info)
            covariance = law.proposal_covariance(q)
            np.linalg.cholesky(covariance)
            centers.append((q,covariance))
            entry.update(proposal_covariance=covariance.tolist(),status='included')
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
            entry.update(status='proposal_unavailable',error=type(error).__name__+': '+str(error))
        diagnostics.append(entry)
    return centers, diagnostics


def _pitch_mean(moments):
    rotation = moments['R']
    pitch = np.arctan2(-rotation[:,2,0],np.hypot(rotation[:,0,0],rotation[:,1,0]))
    return float(moments['weights']@pitch)


def validate_state(law, reference, max_seeds=6, saved_pitch_rad=None, **seed_options):
    centers,seeds = find_centers(law,reference,max_seeds,**seed_options)
    refined = multilocal_mis(law,np.asarray(reference),centers)
    result = {'status':'computed','centers':seeds,
        'refined_logZ':refined['logZ'],'refined_Mqq':refined['Mqq'].tolist(),
        'refined_pitch_mean_deg':float(np.rad2deg(refined['pitch_mean_rad'])),
        'refined_pitch_std_deg':float(np.rad2deg(math.sqrt(refined['pitch_variance_rad2']))),
        'local_centers_used':refined['local_centers'],'global_nodes':refined['global_nodes'],
        'total_nodes':refined['total_nodes'],'local_mixture_weight':refined['local_mixture_weight']}
    if saved_pitch_rad is not None:
        result['saved_pitch_mean_deg'] = float(np.rad2deg(saved_pitch_rad))
        result['refined_minus_saved_pitch_mean_deg'] = float(np.rad2deg(refined['pitch_mean_rad']-saved_pitch_rad))
    try:
        covariance = law.proposal_covariance(reference)
        original = law.integrate(reference,covariance,order=7)
        original_pitch = _pitch_mean(original)
        result.update(production_logZ=original['logZ'],
            production_Mqq=original['Mqq'].tolist(),
            production_pitch_mean_deg=float(np.rad2deg(original_pitch)),
            logZ_change=float(refined['logZ']-original['logZ']),
            Mqq_frobenius_change=float(np.linalg.norm(refined['Mqq']-original['Mqq'])),
            pitch_mean_change_deg=float(np.rad2deg(refined['pitch_mean_rad']-original_pitch)))
        # Diagnostic labels only: no state acceptance/rejection and no updates
        # to the estimator, trajectory or published comparison statistics.
        result['large_change_diagnostic'] = (abs(result['logZ_change'])>.1 or
            result['Mqq_frobenius_change']>.01 or abs(result['pitch_mean_change_deg'])>.1)
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
        result.update(status='production_curvature_unavailable',
            production_error=type(error).__name__+': '+str(error))
    return result


def self_test():
    parameter = np.diag([-400.,0.,-400.,-400.])
    law = ExponentialAttitudeLaw(parameter,np.zeros((3,3)),np.zeros((3,3)),
                               [0.,0.,-9.80665],[.8,.2,.56])
    reference = np.array([1.,0.,0.,0.])
    # Retain the historical single-local-chart control even when production
    # integrate() has gained adaptive mode discovery.
    coarse = multilocal_mis(law,reference,[(reference,np.eye(3)*1e-4)],
                           order=7,radial=4,angles=12)
    centers,seeds = find_centers(law,reference)
    result = multilocal_mis(law,reference,centers)
    expected = bingham_log_normalizer(parameter,integration_steps=200)
    report = dict(test='known_remote_Bingham_peak_kappa400',
        expected_logZ=expected,original_wrong_chart_logZ=coarse['logZ'],
        multilocal_logZ=result['logZ'],absolute_logZ_error=abs(result['logZ']-expected),
        centers=len(centers),proposal_failures=sum(s['status']!='included' for s in seeds),
        no_estimator_state_changed=True)
    if report['absolute_logZ_error']>1e-5:
        raise AssertionError('Known remote-peak integration self-test failed: '+repr(report))
    return report


def _parallel_inputs(files,args,report):
    """Independent per-input diagnostics, retaining every returned state row."""
    workers=min(args.jobs,len(files))
    report['parallel_input_workers']=workers
    with tempfile.TemporaryDirectory(prefix='probtf-modes-check-') as folder:
        outputs=[]
        commands=[]
        for index,path in enumerate(files):
            output=Path(folder)/('%d.json'%index)
            command=[sys.executable,str(Path(__file__).resolve()),
                '--run-root',str(path.parent.resolve()),'--output',str(output),
                '--stride',str(args.stride),'--maxseeds',str(args.maxseeds),
                '--seed-strategy',args.seed_strategy,
                '--seed-radial',str(args.seed_radial),'--seed-angles',str(args.seed_angles),
                '--seed-iterations',str(args.seed_iterations),'--jobs','1']
            if args.extra_indices:
                command+=['--extra-indices']+[str(i) for i in args.extra_indices]
            outputs.append(output);commands.append(command)
        environment=dict(os.environ,OPENBLAS_NUM_THREADS='1')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(subprocess.run,command,stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,text=True,env=environment) for command in commands]
            for path,output,future in zip(files,outputs,futures):
                process=future.result()
                if process.returncode:
                    report['input_errors'].append({'input':str(path),
                        'error':'validator subprocess exited '+str(process.returncode),
                        'stdout':process.stdout,'stderr':process.stderr})
                    continue
                child=json.loads(output.read_text())
                report['states'].extend(child['states'])
                report['input_errors'].extend(child['input_errors'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--stride',type=int,default=2000)
    parser.add_argument('--extra-indices',type=int,nargs='*',default=[],
        help='Additional original sample indices to check beyond the regular stride')
    parser.add_argument('--maxseeds',type=int,default=6,
        help='Additional highest-density Haar mode-search starts, beyond fixed diversified starts')
    parser.add_argument('--seed-strategy',choices=('diversified','haar_only'),default='diversified',
        help='Independent fixed 24+rotated24 projective starts plus top Haar starts, or Haar only')
    parser.add_argument('--seed-radial',type=int,default=8)
    parser.add_argument('--seed-angles',type=int,default=24)
    parser.add_argument('--seed-iterations',type=int,default=20)
    parser.add_argument('--jobs',type=int,default=3,
        help='Parallel independent input files; never parallelize or alter a trajectory update')
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    if args.stride<1 or args.maxseeds<1 or args.jobs<1:
        parser.error('stride, maxseeds and jobs must be positive')
    if any(index<0 for index in args.extra_indices):
        parser.error('extra indices must be nonnegative original sample indices')
    if args.seed_radial<2 or args.seed_angles<4 or args.seed_iterations<1:
        parser.error('seed grid needs radial>=2, angles>=4, iterations>=1')
    if args.run_root is None and not args.self_test:
        parser.error('--run-root is required unless --self-test is selected')
    if args.run_root is not None and args.output is None:
        parser.error('--output is required for a saved-run diagnostic')
    started=time.monotonic()
    report={'schema':'posthoc-multilocal-attitude-quadrature/v1',
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'attitude_law_sha256':hashlib.sha256(Path(sys.modules[ExponentialAttitudeLaw.__module__].__file__).read_bytes()).hexdigest(),
        'estimator_state_changed':False,'sensors_rerun':False,
        'integration':'same saved density, all found local charts including duplicates, global Haar 8x24x24',
        'stride':args.stride,'extra_indices':sorted(set(args.extra_indices)),
        'maxseeds':args.maxseeds,
        'seed_strategy':args.seed_strategy,
        'mode_seed_grid':{'radial':args.seed_radial,'angles':args.seed_angles},
        'fixed_projective_seed_count':48 if args.seed_strategy=='diversified' else 0,
        'fixed_copy_left_rotation_wxyz':(np.array([1.,2.,3.,4.])/math.sqrt(30.)).tolist(),
        'mode_search_iterations':args.seed_iterations,
        'mode_search_starts_are_integration_geometry_only':True,
        'duplicate_mode_charts_retained':True,
        'local_quadrature_order':7,'global_radial_order':8,'global_angle_order':24,
        'local_total_proposal_weight':.999999,
        'nonblocking_diagnostic_thresholds':{'abs_logZ_change':.1,'Mq_frobenius_change':.01,'abs_pitch_mean_change_deg':.1},
        'scope':'finite post-hoc integration sensitivity; not a proof of discovering every mode',
        'states':[],'input_errors':[]}
    if args.self_test:
        report['self_test']=self_test()
    if args.run_root is not None:
        files=sorted(args.run_root.rglob('full_estimation.npz'))
        report['input_files_found']=len(files)
        if args.jobs>1 and len(files)>1:
            _parallel_inputs(files,args,report)
            files=[]
        else:
            report['parallel_input_workers']=1 if files else 0
        for path in files:
            try:
                settings=json.loads((path.parent/'simulation_settings.json').read_text())
                with np.load(path) as data:
                    indices=list(range(0,len(data['time_seconds']),args.stride))
                    if indices[-1]!=len(data['time_seconds'])-1:indices.append(len(data['time_seconds'])-1)
                    indices=sorted(set(indices)|{i for i in args.extra_indices
                                                  if i<len(data['time_seconds'])})
                    for index in indices:
                        for method in range(data['A'].shape[1]):
                            law=ExponentialAttitudeLaw(data['A'][index,method],data['Wg'][index,method],
                                data['Wb'][index,method],settings['gravity_world_m_s2'],settings['magnetic_field_world'])
                            try:
                                state=validate_state(law,data['mode_wxyz'][index,method],args.maxseeds,
                                    float(data['pitch_mean_rad'][index,method]),
                                    seed_strategy=args.seed_strategy,seed_radial=args.seed_radial,
                                    seed_angles=args.seed_angles,mode_iterations=args.seed_iterations)
                            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
                                state={'status':'diagnostic_unavailable','error':type(error).__name__+': '+str(error)}
                            state.update(input=str(path),scenario=settings.get('scenario',path.parent.name),
                                source_index=index,time_seconds=float(data['time_seconds'][index]),method_index=method,
                                method=('joint_marg','joint_imu','gravity_marg')[method])
                            for field in ('projection_moment_residual',
                                          'projection_initial_moment_residual','mode_gradient_norm'):
                                if field in data.files:
                                    state['saved_'+field]=float(data[field][index,method])
                            report['states'].append(state)
            except (OSError,ValueError,KeyError) as error:
                report['input_errors'].append({'input':str(path),'error':type(error).__name__+': '+str(error)})
    complete=[s for s in report['states'] if 'logZ_change' in s]
    report['summary']={'states':len(report['states']),'compared_states':len(complete),
        'large_change_diagnostics':sum(s.get('large_change_diagnostic',False) for s in complete),
        'max_abs_logZ_change':max((abs(s['logZ_change']) for s in complete),default=None),
        'max_Mqq_frobenius_change':max((s['Mqq_frobenius_change'] for s in complete),default=None),
        'max_abs_pitch_mean_change_deg':max((abs(s['pitch_mean_change_deg']) for s in complete),default=None),
        'elapsed_seconds':time.monotonic()-started}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:report[key] for key in ('self_test','summary') if key in report},indent=2))


if __name__=='__main__':
    main()
