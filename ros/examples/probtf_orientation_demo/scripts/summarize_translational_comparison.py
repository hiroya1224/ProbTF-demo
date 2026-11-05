#!/usr/bin/env python3
"""Publish finished standalone IMU runs and a predeclared Madgwick gain control."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import platform
import scipy

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[3]/'ros/core/probtf_core/src'))
import numpy as np
from scipy.spatial.transform import Rotation
import compare_translational_acceleration as benchmark
from translational_madgwick import MadgwickBaseline
from translational_synthetic_data import estimator_inputs

EXTRA_IDS=('madgwick_marg_noise_gain','madgwick_imu_noise_gain')
EXTRA_LABELS=('Madgwick MARG (gyro-noise RMS gain)','Madgwick IMU (gyro-noise RMS gain)')
PLOT_FIELDS=('time_seconds','truth_rpy_deg','pitch_estimate_deg','pitch_variance_rad2',
    'rpy_error_deg','local_mode_rpy_error_deg','acceleration_world_mean',
    'acceleration_world_covariance','true_acceleration_world_m_s2',
    'extra_rpy_deg','extra_rpy_error_deg')


def run_covariance_gain_baselines(inputs):
    covariance=np.asarray(inputs['gyro_covariance'])
    beta=.5*float(np.sqrt(np.trace(covariance)))
    reference=np.asarray(inputs['magnetic_field_world'])
    q0=inputs['initial_wxyz']
    states=[MadgwickBaseline(q0,reference,beta),MadgwickBaseline(q0,beta=beta)]
    n=len(inputs['measurement_stamps']);quaternions=np.empty((n,2,4))
    for k in range(n):
        dt=0. if k==0 else float(inputs['measurement_stamps'][k]-inputs['measurement_stamps'][k-1])
        mag=inputs['magnetic_raw'][k] if inputs['magnetic_observed'][k] else None
        quaternions[k,0]=states[0].update(inputs['gyro_rad_s'][k],inputs['acceleration_m_s2'][k],dt,mag)
        quaternions[k,1]=states[1].update(inputs['gyro_rad_s'][k],inputs['acceleration_m_s2'][k],dt)
    return quaternions,beta


def evaluate_extra_baselines(arrays,quaternions,beta):
    n=len(arrays['time_seconds'])
    rpy=Rotation.from_quat(quaternions.reshape(-1,4)[:,[1,2,3,0]]).as_euler('xyz',degrees=True).reshape(n,2,3)
    errors=(rpy-arrays['truth_rpy_deg'][:,None,:]+180.)%360.-180.
    result={}
    for i,name in enumerate(EXTRA_IDS):
        pitch=errors[:,i,1]
        result[name]={'pitch_rmse_deg':float(np.sqrt(np.mean(pitch*pitch))),
            'pitch_mae_deg':float(np.mean(abs(pitch))), 'pitch_max_abs_deg':float(abs(pitch).max()),
            'rpy_rmse_deg':np.sqrt(np.mean(errors[:,i]**2,axis=0)).tolist(),
            'native_samples':n,'beta':beta,'gain_selection':'0.5 sqrt(trace(assumed gyro sample covariance)); no truth search'}
    arrays['extra_mode_wxyz']=quaternions
    arrays['extra_rpy_deg']=rpy
    arrays['extra_rpy_error_deg']=errors
    return result


def plot_gain_control(cases,output,beta,include_historical=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'legend.fontsize':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(len(cases),2,figsize=(13.,3.2*len(cases)),squeeze=False)
    for row,(case,arrays) in enumerate(cases.items()):
        t=arrays['time_seconds']
        for col,(joint,base) in enumerate(((0,3),(1,4))):
            ax=axes[row,col]
            ax.plot(t,arrays['rpy_error_deg'][:,joint,1],color=benchmark.COLORS[joint],lw=1.3,label=benchmark.LABELS[joint])
            if include_historical:
                ax.plot(t,arrays['rpy_error_deg'][:,base,1],color='#E67E22',ls='--',lw=1.2,label='Madgwick β=0.1')
            ax.plot(t,arrays['extra_rpy_error_deg'][:,col,1],color='black',ls=':',lw=1.25,label='Madgwick β={:.6f} (noise RMS)'.format(beta))
            ax.axhline(0,color='gray',alpha=.4,lw=.7)
            ax.set_title(benchmark.SCENARIO_LABELS[case]+(' | 9-axis' if col==0 else ' | 6-axis'))
            ax.set_ylabel('Signed pitch error [deg]');ax.set_xlabel('Time [s]');ax.grid(alpha=.23);ax.set_xlim(t[0],t[-1])
    for ax in axes[0]:ax.legend(loc='upper left',bbox_to_anchor=(0,1.38))
    benchmark.finalize_figure_layout(fig)
    prefix='pitch_gain_control' if include_historical else 'pitch_noise_gain_detail'
    for suffix in ('png','pdf','svg'):fig.savefig(output/(prefix+'.'+suffix),bbox_inches='tight',dpi=180)
    plt.close(fig)



def plot_rmse_summary(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=benchmark.METHODS+EXTRA_IDS
    labels=benchmark.LABELS+EXTRA_LABELS
    fig,ax=plt.subplots(figsize=(12.,6.1))
    y=np.arange(len(names))
    for i,(case,record) in enumerate(summary['cases'].items()):
        values=[record['statistics'][method]['pitch_rmse_deg'] for method in names]
        rows=y+(i-1)*.24
        ax.barh(rows,values,height=.22,label=benchmark.SCENARIO_LABELS[case],alpha=.85)
        for row,value in zip(rows,values):ax.text(value*1.04,row,'{:.3g}'.format(value),va='center',fontsize=8)
    ax.set_yticks(y);ax.set_yticklabels(labels);ax.invert_yaxis();ax.set_xscale('log')
    ax.set_xlabel('Pitch RMSE over all native outputs [deg] (log scale)')
    ax.grid(axis='x',alpha=.22);ax.legend(loc='upper left',bbox_to_anchor=(0,1.2),ncol=2)
    benchmark.finalize_figure_layout(fig)
    for suffix in ('png','pdf','svg'):fig.savefig(output/('pitch_rmse_summary.'+suffix),bbox_inches='tight',dpi=180)
    plt.close(fig)


def publish_validation_reports(run_root, publication):
    """Keep every validation state and metric; omit repetitive optimizer traces."""
    for name in ('fixed_state_quadrature_check.json','global_modes_check.json'):
        source=Path(run_root)/name
        if not source.exists():continue
        report=json.loads(source.read_text())
        def compact(value):
            if isinstance(value,list):return [compact(item) for item in value]
            if not isinstance(value,dict):return value
            traces=isinstance(value.get('centers'),list)
            result={key:compact(item) for key,item in value.items() if not (traces and key=='centers')}
            if traces:
                result['integration_center_diagnostics_count']=len(value['centers'])
            return result
        report=compact(report)
        report['publication_storage']={
            'all_checked_states_and_numeric_differences_retained':True,
            'omitted_only':'per-seed mode/proposal optimizer diagnostics under centers; all checked states and integral-comparison metrics retained',
            'full_report_path':str(source.resolve()),
            'full_report_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        benchmark.save_json(Path(publication)/name,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--publication',type=Path,required=True)
    args=parser.parse_args();args.publication.mkdir(parents=True,exist_ok=True)
    cases={};summary={'schema':'probtf.standalone_imu_translation_comparison.v1',
        'methods':dict(zip(benchmark.METHODS+EXTRA_IDS,benchmark.LABELS+EXTRA_LABELS)),
        'cases':{},'software_versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},'primary_metrics_use_all_native_outputs':True, 'noise_covariance_type':'per-reading sensor covariance, not continuous-time spectral density',
        'saved_plot_series_hz':50., 'saved_plot_series_rule':'all native output acquisition stamps; only necessary plot fields, no state-law intermediates','raw_and_full_law_storage':str(args.run_root),
        'ai_generated_images_or_video':False, 'pitch_point_policy':'posterior pitch mean for probabilistic laws; point pitch for Madgwick/gyro; local-mode errors separately retained', 'rpy_error_array_policy':'pitch uses posterior mean in the first three methods, while roll/yaw use local mode; this mixed array is not a single attitude estimate','source_hashes_at_execution':json.loads((args.run_root/'source_hashes.json').read_text()) if (args.run_root/'source_hashes.json').exists() else {}}
    for path in sorted(args.run_root.rglob('evaluated_timeseries.npz')):
        case=path.parent.name
        if case not in benchmark.SCENARIO_LABELS:continue
        if case in cases:raise ValueError('Duplicate completed case: '+case)
        arrays=dict(np.load(path,allow_pickle=False));measured=dict(np.load(path.parent/'input_data.npz',allow_pickle=False))
        meta=json.loads((path.parent/'simulation_settings.json').read_text())
        original=json.loads((path.parent/'summary.json').read_text())
        quaternions,beta=run_covariance_gain_baselines(estimator_inputs({'arrays':measured,'metadata':meta}))
        extra=evaluate_extra_baselines(arrays,quaternions,beta)
        for i,method in enumerate(benchmark.METHODS[:3]):
            width=2.*np.rad2deg(np.sqrt(arrays['pitch_variance_rad2'][:,i]))
            original['statistics'][method]['truth_inside_pitch_mean_plus_minus_2_std_fraction']=float(
                np.mean(np.abs(arrays['rpy_error_deg'][:,i,1])<=width))
        cases[case]=arrays
        summary['cases'][case]={'statistics':dict(original['statistics'],**extra),
            'combined_six_method_wall_seconds':original['wall_seconds'],
            'simulation_settings':meta,'native_samples':len(arrays['time_seconds']),
            'input_data_sha256':hashlib.sha256((path.parent/'input_data.npz').read_bytes()).hexdigest(),
            'full_state_sha256':hashlib.sha256((path.parent/'full_estimation.npz').read_bytes()).hexdigest()}
        np.savez_compressed(args.publication/(case+'_plot_series.npz'),**{k:arrays[k] for k in PLOT_FIELDS})
    expected=set(benchmark.SCENARIO_LABELS)
    if set(cases)!=expected:raise ValueError('Expected all three completed cases; got '+repr(set(cases)))
    cases={name:cases[name] for name in benchmark.SCENARIO_LABELS}
    summary['saved_plot_series_fields']=list(PLOT_FIELDS)
    summary['saved_plot_series_hz']=1./next(iter(summary['cases'].values()))['simulation_settings']['effective_dt_seconds']
    summary['additional_madgwick_beta']=beta
    summary['additional_gain_policy']=json.loads((HERE.parent/'config/translational_madgwick_gain_comparison.json').read_text())
    summary['full_experiment_settings']=json.loads(next(args.run_root.rglob('configured_experiment.json')).read_text())
    summary['full_experiment_settings']['scenarios']=list(cases)
    summary['source_hashes_at_publication']={k:hashlib.sha256((HERE.parents[3]/k).read_bytes()).hexdigest() for k in summary['source_hashes_at_execution']}
    completion=args.run_root/'source_hashes_at_completion.json'
    summary['source_hashes_at_completion']=json.loads(completion.read_text()) if completion.exists() else summary['source_hashes_at_publication']
    summary['estimator_sources_unchanged_during_execution']=(summary['source_hashes_at_execution']==summary['source_hashes_at_completion'] if summary['source_hashes_at_execution'] else None)
    revision=args.run_root/'post_run_plotting_revision.json'
    summary['post_run_plotting_revision']=json.loads(revision.read_text()) if revision.exists() else None
    benchmark.make_figures(cases,args.publication)
    plot_gain_control(cases,args.publication,beta)
    plot_gain_control(cases,args.publication,beta,include_historical=False)
    plot_rmse_summary(summary,args.publication)
    for svg in args.publication.glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    benchmark.save_json(args.publication/'summary.json',summary)
    publish_validation_reports(args.run_root,args.publication)
    for case in cases:
        print(case,json.dumps({k:round(v['pitch_rmse_deg'],6) for k,v in summary['cases'][case]['statistics'].items()}),flush=True)
    print('PUBLISHED',args.publication,flush=True)


if __name__=='__main__':main()
