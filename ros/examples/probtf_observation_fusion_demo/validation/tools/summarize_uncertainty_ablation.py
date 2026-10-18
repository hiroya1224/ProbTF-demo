#!/usr/bin/env python3
"""Collect bounded numeric evidence and distance plots from diagnostic trials."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


CASES=('marker','imu_marker')
MODES=('baseline','oracle_goal','oracle_marker','true_K','oracle_body','all_oracle','known_landmark','oracle_body_goal')


def read_json(path):
    path=Path(path)
    if path.is_file():return json.loads(path.read_text())
    with gzip.open(str(path)+'.gz','rt')as stream:return json.load(stream)


def physical_trace(directory):
    path=directory/'physical_trajectory.jsonl'
    opener=open if path.exists()else gzip.open
    path=path if path.exists()else Path(str(path)+'.gz')
    values=[]
    with opener(path,'rt')as stream:
        for line in stream:
            item=json.loads(line)
            values.append((item['time'],item['physical_error_m'],*item['tip']))
    array=np.asarray(values)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(array[:,2:5],axis=0),axis=1))]
    reached=np.flatnonzero(array[:,1]<=.006)
    first=int(reached[0])if len(reached)else None
    scores=dict(first_reach_s=None if first is None else float(array[first,0]),
        first_reach_error_m=None if first is None else float(array[first,1]),
        arc_before_first_reach_m=None if first is None else float(arc[first]),
        net_shortening_per_arc_before_first_reach=(None if first is None or arc[first]==0
            else float((array[0,1]-array[first,1])/arc[first])),
        arc_after_first_reach_m=None if first is None else float(arc[-1]-arc[first]),
        maximum_error_after_first_reach_m=None if first is None else float(array[first:,1].max()),
        whole_trajectory_arc_m=float(arc[-1]),
        reach_threshold_m=.006,
        common_time_error_m={str(t):float(np.interp(t,array[:,0],array[:,1]))
            for t in (0.,30.,38.,60.,90.,120.,180.)if t<=array[-1,0]},
        common_time_error_interpolation='linear between original physical frames')
    # This affects figure rendering only. Full trajectory scores come from
    # every physical frame in the original diagnostic summary.
    stride=max(1,int(np.ceil(len(array)/1800)))
    selected=np.unique(np.r_[np.arange(0,len(array),stride),len(array)-1,
        np.argmin(array[:,1]),[]if first is None else[first]]).astype(int)
    trace=np.c_[array[:,0],array[:,1],arc]
    return trace[selected],scores


def summarize(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];traces={}
    for case in CASES:
        for mode in MODES:
            directory=root/'live'/(case+'_'+mode)
            if not(directory/'diagnostic_summary.json').is_file():continue
            result=read_json(directory/'diagnostic_summary.json')
            setup=read_json(directory/'diagnostic_setup.json')
            cutoff=directory/'diagnostic_physical_cutoff.json'
            recovery=directory/'diagnostic_raw_recovery.json'
            row=dict(result,case=case,intervention=mode,raw_directory=str(directory),
                setup_sha256=hashlib.sha256((directory/'diagnostic_setup.json').read_bytes()).hexdigest(),
                requested_physical_duration_s=setup['duration_s'],
                asynchronous_timing_matched=False,
                diagnostic_source_sha256=setup.get('diagnostic_sha256'))
            if cutoff.is_file():
                data=read_json(cutoff)
                keys=('acquisition_requests','acquisition_archived','actual_motion_hz',
                    'actual_control_hz','actual_camera_hz','body_estimation_finished',
                    'body_estimation_drained','stiffness_jobs_completed','stiffness_jobs_submitted',
                    'stiffness_estimation_drained','all_enabled_observations_processed',
                    'observation_age_s','gradient_source_age_s','actual_estimate_hz',
                    'last_body_candidate_id','stiffness_failure','stiffness_worker_failed')
                row['physical_cutoff_selected_fields']={key:data[key]for key in keys if key in data}
                row['physical_cutoff_sha256']=hashlib.sha256(cutoff.read_bytes()).hexdigest()
            if recovery.is_file():
                row['raw_archival_recovery']=read_json(recovery)
            compression=directory/'lossless_compression.json'
            if compression.is_file():
                value=read_json(compression)
                row['lossless_storage']=dict(original_bytes=value['original_bytes'],
                    compressed_bytes=value['compressed_bytes'],file_count=len(value['files']),
                    manifest_sha256=hashlib.sha256(compression.read_bytes()).hexdigest())
            trace,path_scores=physical_trace(directory)
            if not np.isclose(path_scores['whole_trajectory_arc_m'],row['tip_arc_m'],rtol=1e-11,atol=1e-12):
                raise ValueError('Trajectory arc and recorded diagnostic summary disagree: '+str(directory))
            row['whole_physical_path_scores']=path_scores
            rows.append(row)
            traces[(case,mode)]=trace
    record=dict(schema='uncertainty_diagnostic_closed_loop_comparison_v1',
        source_root=str(root),production_numeric_commit='1244f3f',
        production_control_changed=False,truth_inputs_explicitly_diagnostic=True,
        Monte_Carlo_inference=False,replications_per_condition=1,
        original_gui_run_not_used_as_headless_baseline=True,
        uncertainty_only_causality_claimed=False,rows=rows)
    (output/'closed_loop_summary.json').write_text(json.dumps(record,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    labels=dict(baseline='Baseline',oracle_goal='True goal / narrow law',
        oracle_marker='True relative marker / narrow law',true_K='True stiffness action',
        oracle_body='True body / narrowed body',all_oracle='Combined truth assistance',
        known_landmark='Known landmark prior / normal updates',
        oracle_body_goal='True body + goal / estimated K')
    for axis,case in zip(axes,CASES):
        for index,mode in enumerate(MODES):
            values=traces.get((case,mode))
            if values is None:continue
            axis.plot(values[:,0],values[:,1]*1000,label=labels[mode],color='C'+str(index),lw=1.4)
            if values[-1,0]<179.999:
                axis.plot(values[-1,0],values[-1,1]*1000,'x',color='C'+str(index),ms=8)
        axis.axhline(6,color='black',ls=':',lw=.8,label='6 mm reach threshold')
        axis.set_title('Marker'if case=='marker'else'IMU + Marker')
        axis.set_xlabel('Physical source time [s]');axis.grid(alpha=.25)
        axis.set_xlim(0,180)
    axes[0].set_ylabel('Actual tip-to-switch distance [mm]')
    entries={}
    for axis in axes:
        axis_handles,axis_labels=axis.get_legend_handles_labels()
        for handle,label in zip(axis_handles,axis_labels):entries.setdefault(label,handle)
    legend=list(entries);handles=list(entries.values())
    fig.legend(handles,legend,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('Diagnostic truth interventions; unchanged physical model and controller')
    fig.tight_layout(rect=(0,.15,1,.94))
    fig.savefig(output/'closed_loop_distances.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for axis,case in zip(axes,CASES):
        for index,mode in enumerate(MODES):
            values=traces.get((case,mode))
            if values is None:continue
            axis.plot(values[:,2],values[:,1]*1000,label=labels[mode],color='C'+str(index),lw=1.4)
            axis.plot(values[-1,2],values[-1,1]*1000,'o',color='C'+str(index),ms=3)
        axis.axhline(6,color='black',ls=':',lw=.8,label='6 mm reach threshold')
        axis.set_title('Marker'if case=='marker'else'IMU + Marker')
        axis.set_xlabel('Cumulative actual tip travel [m]');axis.grid(alpha=.25)
    axes[0].set_ylabel('Actual tip-to-switch distance [mm]')
    fig.legend(handles,legend,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('Distance versus cumulative tip travel; all original physical frames scored')
    fig.tight_layout(rect=(0,.15,1,.94))
    fig.savefig(output/'closed_loop_paths.png',dpi=150);plt.close(fig)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    value=summarize(args.root,args.output)
    print('Recorded',len(value['rows']),'completed or explicitly failed trials')
