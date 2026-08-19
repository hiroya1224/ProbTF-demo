#!/usr/bin/env python3
"""Forward-only soft-max continuation from a robust PID seed toward failure."""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
from tqdm.auto import tqdm

_HERE=Path(__file__).resolve().parent
_MINIMAL=_HERE.parent
_PROJECT_ROOT=_MINIMAL.parent
_SOURCE_ROOT=_PROJECT_ROOT/'src'
for _path in (_HERE,_MINIMAL,_SOURCE_ROOT):
    if str(_path) not in sys.path: sys.path.insert(0,str(_path))
from core import COVARIANCE_NAMES,actuator_parameters_from_estimate,load_estimate_json,quotient_to_scale_free_plants
from grape_param_estim.gimbalrotor_pid_postprocess import ScaleFreePlant,load_vehicle_model
from pid_safe_margin_full import _sobol_gain_coordinates
from pid_safe_margin_slices import GAIN_CAPS,GAIN_LABELS,ForwardMarginEvaluator,_recorded_gain_matrix,_sobol_quotient_coordinates
from single_bag_savgol_reports import source_commit,write_json

SCHEMA='grape-param-estim/pid-softmax-continuation/v1'
DEFAULT_WORKERS=min(12,os.cpu_count() or 1)

def _csv(text,cast):
    out=tuple(cast(x.strip()) for x in str(text).split(',') if x.strip())
    if not out: raise ValueError('empty schedule')
    return out

def _metric(row,gap):
    if not np.all(row.pole_valid_mask): return None
    r=np.asarray(row.spectral_radius,float); m=float(np.max(r)); hard=1.0-m
    if gap<=0.0: tau=0.0; sm=m
    else:
        tau=float(gap/math.log(float(r.size)))
        sm=float(m+tau*math.log(float(np.mean(np.exp((r-m)/tau)))))
    return {'delta_soft':float(1-sm),'delta_hard':float(hard),'smooth_max':sm,'hard_max':m,
            'stable_fraction':float(np.mean(r<1.0)),'tau':tau,'gap':float(gap)}

def _eval(ev,q,gap):
    row=ev.evaluate(q); return row,_metric(row,gap)

def _find_seed(ev,failure_q,count,seed,gap,progress):
    design,info=_sobol_gain_coordinates(count,seed)
    order=np.argsort(np.linalg.norm(design-failure_q[None,:],axis=1)); best=None; safe=bad=0
    bar=tqdm(order,desc='Global soft-safe seed scan',unit='gain',dynamic_ncols=True,disable=not progress)
    for idx in bar:
        _,met=_eval(ev,design[int(idx)],gap)
        if met is None: bad+=1; continue
        if met['delta_soft']>0:
            safe+=1
            if best is None or met['delta_soft']>best[1]['delta_soft']: best=(design[int(idx)].copy(),met)
        if progress: bar.set_postfix(safe=safe,unresolved=bad,best='n/a' if best is None else f"{best[1]['delta_soft']:.2e}")
    if best is None: raise RuntimeError('no soft-safe seed found')
    return best[0],best[1],{**dict(info),'soft_safe_count':safe,'unresolved_count':bad}

def _cloud(q,failure_q,step,ratio,count,sampler):
    v=failure_q-q; d=float(np.linalg.norm(v))
    if d==0: return q[None,:]
    h=min(step,d); center=q+h*v/d; radius=ratio*h
    pts=np.clip(center[None,:]+radius*(2*sampler.random(count)-1),0,1)
    return np.vstack((center[None,:],q[None,:],pts))

def _choose(ev,candidates,failure_q,current_distance,gap,progress,desc):
    good=[]
    bar=tqdm(range(len(candidates)),desc=desc,unit='candidate',leave=False,dynamic_ncols=True,disable=not progress)
    for i in bar:
        q=candidates[i]; _,met=_eval(ev,q,gap)
        if met is None: continue
        d=float(np.linalg.norm(q-failure_q))
        if met['delta_soft']>0 and d<current_distance-1e-12: good.append((d,-met['delta_soft'],q.copy(),met))
    if not good: return None,None,0
    good.sort(key=lambda x:(x[0],x[1])); x=good[0]; return x[2],x[3],len(good)

def _recover(ev,q,failure_q,gap,radius,count,sampler,progress):
    _,met=_eval(ev,q,gap)
    if met is not None and met['delta_soft']>0: return q.copy(),met,False
    pts=np.clip(q[None,:]+radius*(2*sampler.random(count)-1),0,1); pts=np.vstack((q[None,:],pts)); best=None
    bar=tqdm(range(len(pts)),desc='Refinement recovery',unit='candidate',leave=False,dynamic_ncols=True,disable=not progress)
    for i in bar:
        cand=pts[i]; _,val=_eval(ev,cand,gap)
        if val is None or val['delta_soft']<=0: continue
        item=(float(np.linalg.norm(cand-failure_q)),-val['delta_soft'],cand.copy(),val)
        if best is None or item[:2]<best[:2]: best=item
    if best is None: raise RuntimeError('refinement invalidated path point and recovery found no safe replacement')
    return best[2],best[3],True

def _stage(ev,start_q,failure_q,gap,stage,sampler,args,progress):
    q=start_q.copy(); _,met=_eval(ev,q,gap)
    if met is None or met['delta_soft']<=0: raise RuntimeError('stage start is not soft-safe')
    path=[{'stage':stage,'step':0,'q':q.tolist(),'distance':float(np.linalg.norm(q-failure_q)),**met}]
    h=float(args.initial_step); shrinks=attempts=0
    bar=tqdm(total=args.max_steps,desc=f'Stage {stage} continuation',unit='step',dynamic_ncols=True,disable=not progress)
    try:
        while len(path)-1<args.max_steps:
            d=float(np.linalg.norm(q-failure_q))
            if d<=args.minimum_step or h<args.minimum_step: break
            attempts+=1; candidates=_cloud(q,failure_q,h,args.radius_ratio,args.local_candidates,sampler)
            nq,nmet,ngood=_choose(ev,candidates,failure_q,d,gap,progress,f'Stage {stage} local cloud')
            if nq is None:
                h*=0.5; shrinks+=1
                if progress: bar.set_postfix(blocked=True,step=f'{h:.3f}',distance=f'{d:.3f}')
                continue
            old=d; q=nq; met=nmet; d=float(np.linalg.norm(q-failure_q))
            path.append({'stage':stage,'step':len(path),'q':q.tolist(),'distance':d,'accepted_step':h,**met})
            if ngood>=max(4,args.local_candidates//4): h=min(args.initial_step,h*1.25)
            bar.update(1)
            if progress: bar.set_postfix(distance=f'{d:.3f}',improvement=f'{old-d:.3f}',delta=f"{met['delta_soft']:.2e}",step=f'{h:.3f}')
    finally: bar.close()
    return q,path,{'attempts':attempts,'blocked_step_shrinks':shrinks}

def _plot(out,failure_q,seed_q,paths,final_q):
    fig,ax=plt.subplots(2,1,figsize=(13,9),constrained_layout=True)
    for s,path in enumerate(paths): ax[0].plot([p['distance'] for p in path],[p['delta_soft'] for p in path],marker='o',label=f'stage {s}')
    ax[0].axhline(0); ax[0].set_xlabel('normalized 12-D distance to failure'); ax[0].set_ylabel('soft robust margin'); ax[0].legend(); ax[0].grid(True,alpha=.25)
    x=np.arange(12); ax[1].plot(x,failure_q,marker='o',label='failure'); ax[1].plot(x,seed_q,marker='s',label='seed'); ax[1].plot(x,final_q,marker='^',label='final')
    ax[1].set_xticks(x); ax[1].set_xticklabels(GAIN_LABELS,rotation=55,ha='right'); ax[1].set_ylim(0,1); ax[1].set_ylabel('normalized gain q'); ax[1].legend(); ax[1].grid(True,alpha=.25)
    fig.savefig(out/'softmax_continuation.png',dpi=180); plt.close(fig)

def analyze(args):
    t0=time.perf_counter(); progress=not args.no_progress; counts=_csv(args.plant_counts,int); gaps=_csv(args.softmax_gaps,float)
    if len(counts)!=len(gaps): raise ValueError('plant-counts and softmax-gaps lengths differ')
    if any(n<=1 for n in counts) or any(b<=a for a,b in zip(counts,counts[1:])): raise ValueError('plant counts must strictly increase')
    if any(g<0 for g in gaps) or any(b>=a for a,b in zip(gaps,gaps[1:])): raise ValueError('softmax gaps must strictly decrease')
    ep=Path(args.estimate_json).expanduser().resolve(); est=load_estimate_json(ep); fg=_recorded_gain_matrix(est); caps=GAIN_CAPS.reshape(-1); fq=fg.reshape(-1)/caps
    vm=load_vehicle_model(Path(est['input']['vehicle_model'])); ap=actuator_parameters_from_estimate(est); dt=float(est['controller_timing']['median_seconds'])
    quotient,sampling=_sobol_quotient_coordinates(est,args.covariance,counts[-1],args.seed); plants=quotient_to_scale_free_plants(est,quotient,vm.parameters,ScaleFreePlant)
    sampler=qmc.Sobol(d=12,scramble=True,seed=args.local_seed); evaluators=[]; paths=[]; stages=[]
    try:
        first=ForwardMarginEvaluator(plants=plants[:counts[0]],vehicle_model=vm,actuator_parameters=ap,controller_dt=dt,workers=args.workers); evaluators.append(first)
        seed_q,seed_met,seed_info=_find_seed(first,fq,args.gain_samples,args.gain_seed,gaps[0],progress); q=seed_q.copy()
        for s,(n,gap) in enumerate(zip(counts,gaps)):
            if s==0: ev=first
            else:
                ev=ForwardMarginEvaluator(plants=plants[:n],vehicle_model=vm,actuator_parameters=ap,controller_dt=dt,workers=args.workers); evaluators.append(ev)
                q,_,recovered=_recover(ev,q,fq,gap,args.initial_step*args.radius_ratio,args.local_candidates*2,sampler,progress); stages[-1]['recovered_for_next_stage']=bool(recovered)
            q,path,stats=_stage(ev,q,fq,gap,s,sampler,args,progress); paths.append(path); stages.append({'stage':s,'plant_count':n,'gap':gap,'tau':gap/math.log(float(n)) if gap>0 else 0.0,'path':path,**stats})
        final_row,final_soft=_eval(evaluators[-1],q,gaps[-1]); final_hard=_metric(final_row,0.0); _,failure_soft=_eval(evaluators[-1],fq,gaps[-1])
        out=Path(args.output_dir).expanduser().resolve() if args.output_dir else ep.parent/'pid_softmax_continuation'; out.mkdir(parents=True,exist_ok=True); _plot(out,fq,seed_q,paths,q)
        payload={'schema':SCHEMA,'source_commit':source_commit(_PROJECT_ROOT),'case_name':str(est['case_name']),'estimate_json':str(ep),'definition':{'smooth_max':'m + tau*log(mean(exp((rho_i-m)/tau)))','delta_soft':'1-smooth_max','bound':'0 <= hard_max-smooth_max <= epsilon=tau*log(N)','selection':'closest-to-failure among soft-safe local candidates'},'plant_sampling':{**dict(sampling),'maximum_sample_count':counts[-1]},'schedules':{'plant_counts':list(counts),'softmax_gaps':list(gaps),'taus':[g/math.log(float(n)) if g>0 else 0.0 for n,g in zip(counts,gaps)]},'seed_scan':{**seed_info,'q':seed_q.tolist(),'gain_matrix':(seed_q.reshape(4,3)*GAIN_CAPS).tolist(),'metric':seed_met},'stages':stages,'final':{'q':q.tolist(),'gain_matrix':(q.reshape(4,3)*GAIN_CAPS).tolist(),'distance_to_failure':float(np.linalg.norm(q-fq)),'soft_metric':final_soft,'hard_metric':final_hard},'failure_on_largest_prefix':failure_soft,'elapsed_seconds':float(time.perf_counter()-t0),'files':{'figure':str(out/'softmax_continuation.png')}}
        write_json(out/'softmax_continuation.json',payload); np.savez_compressed(out/'softmax_continuation.npz',quotient_coordinates=quotient,failure_q=fq,seed_q=seed_q,final_q=q,gain_caps=GAIN_CAPS); return payload
    finally:
        for ev in evaluators: ev.close()

def _parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--estimate-json',required=True); p.add_argument('--covariance',choices=COVARIANCE_NAMES,default='conservative_fusion'); p.add_argument('--plant-counts',default='128,256,512,1024'); p.add_argument('--softmax-gaps',default='1e-3,5e-4,2.5e-4,1.25e-4'); p.add_argument('--gain-samples',type=int,default=256); p.add_argument('--local-candidates',type=int,default=64); p.add_argument('--initial-step',type=float,default=.20); p.add_argument('--minimum-step',type=float,default=.0125); p.add_argument('--radius-ratio',type=float,default=.75); p.add_argument('--max-steps',type=int,default=24); p.add_argument('--seed',type=int,default=0); p.add_argument('--gain-seed',type=int,default=1); p.add_argument('--local-seed',type=int,default=2); p.add_argument('--workers',type=int,default=DEFAULT_WORKERS); p.add_argument('--output-dir'); p.add_argument('--no-progress',action='store_true'); return p

def main():
    args=_parser().parse_args(); print(json.dumps(analyze(args),indent=2,sort_keys=True))
if __name__=='__main__': main()
