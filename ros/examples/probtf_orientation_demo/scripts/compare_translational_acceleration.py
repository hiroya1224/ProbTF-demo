#!/usr/bin/env python3
"""Standalone synthetic pitch comparison; no drone fit or mocap is used."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO/'ros/core/probtf_core/src'))

import numpy as np
from scipy.spatial.transform import Rotation
from probtf.geometry.quaternion import quat_mul
from translational_joint_filter import JointTranslationAttitudeFilter
from translational_madgwick import MadgwickBaseline
from translational_synthetic_data import generate_comparison_data, estimator_inputs

METHODS = ('joint_marg','joint_imu','gravity_marg','madgwick_marg','madgwick_imu','gyro_only')
LABELS = ('Joint 9-axis (posterior pitch mean)', 'Joint 6-axis (posterior pitch mean)',
          'Gravity-only 9-axis (posterior pitch mean)', 'Madgwick MARG', 'Madgwick IMU', 'Gyro only')
COLORS = ('#008C46','#1565C0','#8E44AD','#E67E22','#C0392B','#777777')
STYLES = ('-','-','-.','--','--',':')
SCENARIO_LABELS = {'gravity_only':'Rotation, no translation',
    'smooth_translation':'Smooth translating flight',
    'sustained_translation':'Sustained acceleration pulses'}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def estimate_case(inputs, settings, output, progress=None):
    """Only calibrated readings, declared assumptions and one initial pose enter."""
    timestamp = np.asarray(inputs['measurement_stamps']);count=len(timestamp)
    kwargs = {name:settings[name] for name in ('quadrature_order','projection_iterations',
        'initial_attitude_std_deg','initial_acceleration_std','acceleration_time_constant',
        'acceleration_stationary_std')}
    common = (inputs['initial_wxyz'],inputs['gravity_world_m_s2'],inputs['magnetic_field_world'],
        inputs['gyro_covariance'],inputs['acceleration_covariance'],inputs['magnetic_covariance'])
    filters = [JointTranslationAttitudeFilter(*common,**kwargs),
        JointTranslationAttitudeFilter(*common,use_magnetic=False,**kwargs),
        JointTranslationAttitudeFilter(*common,estimate_translation=False,**kwargs)]
    baselines = [MadgwickBaseline(inputs['initial_wxyz'],inputs['magnetic_field_world'],settings['madgwick_beta']),
        MadgwickBaseline(inputs['initial_wxyz'],beta=settings['madgwick_beta'])]
    gyro_quaternion = np.asarray(inputs['initial_wxyz']).copy()
    arrays = {'time_seconds':timestamp.copy(), 'mode_wxyz':np.empty((count,6,4)),
        'pitch_variance_rad2':np.full((count,3),np.nan),
        'pitch_mean_rad':np.empty((count,3)),
        'acceleration_world_mean':np.empty((count,3,3)),
        'acceleration_world_covariance':np.empty((count,3,3,3)),
        'conditional_m':np.empty((count,3,3)), 'conditional_F':np.empty((count,3,3,3)),
        'conditional_S':np.empty((count,3,3,3)), 'A':np.empty((count,3,4,4)),
        'Wg':np.empty((count,3,3,3)), 'Wb':np.empty((count,3,3,3)),
        'projection_moment_residual':np.zeros((count,3)),
        'projection_initial_moment_residual':np.zeros((count,3)),
        'mode_gradient_norm':np.zeros((count,3))}
    started=last=time.monotonic()
    for k,stamp in enumerate(timestamp):
        dt=0. if k==0 else float(stamp-timestamp[k-1])
        magnetic=inputs['magnetic_raw'][k] if inputs['magnetic_observed'][k] else None
        for i,state in enumerate(filters):
            result=state.update(inputs['gyro_rad_s'][k],inputs['acceleration_m_s2'][k],dt,magnetic)
            arrays['mode_wxyz'][k,i]=result['quaternion_wxyz']
            for name in ('pitch_mean_rad','pitch_variance_rad2','acceleration_world_mean','acceleration_world_covariance',
                         'conditional_m','conditional_F','conditional_S','A','Wg','Wb'):
                arrays[name][k,i]=result[name]
            diagnostics=result['diagnostics']
            arrays['projection_moment_residual'][k,i]=diagnostics['projection'].get('moment_residual',0.)
            arrays['projection_initial_moment_residual'][k,i]=diagnostics['projection'].get('initial_moment_residual',0.)
            arrays['mode_gradient_norm'][k,i]=diagnostics['mode']['gradient_norm']
        arrays['mode_wxyz'][k,3]=baselines[0].update(inputs['gyro_rad_s'][k],inputs['acceleration_m_s2'][k],dt,magnetic)
        arrays['mode_wxyz'][k,4]=baselines[1].update(inputs['gyro_rad_s'][k],inputs['acceleration_m_s2'][k],dt)
        if k:
            delta=Rotation.from_rotvec(inputs['gyro_rad_s'][k]*dt).as_quat()[[3,0,1,2]]
            gyro_quaternion=quat_mul(gyro_quaternion,delta)
        arrays['mode_wxyz'][k,5]=gyro_quaternion
        if progress and time.monotonic()-last>=10.:
            progress(k+1,count,time.monotonic()-started)
            last=time.monotonic()
    arrays['rpy_deg']=Rotation.from_quat(arrays['mode_wxyz'].reshape(-1,4)[:,[1,2,3,0]]).as_euler('xyz',degrees=True).reshape(count,6,3)
    np.savez_compressed(output/'full_estimation.npz',**arrays)
    return arrays, time.monotonic()-started


def evaluate_case(arrays, dataset):
    truth=dataset['arrays']['truth_wxyz']
    truth_rpy=Rotation.from_quat(truth[:,[1,2,3,0]]).as_euler('xyz',degrees=True)
    mode_errors=(arrays['rpy_deg']-truth_rpy[:,None,:]+180.)%360.-180.
    pitch_estimate=arrays['rpy_deg'][:,:,1].copy()
    pitch_estimate[:,:3]=np.rad2deg(arrays['pitch_mean_rad'])
    errors=mode_errors.copy()
    errors[:,:3,1]=(pitch_estimate[:,:3]-truth_rpy[:,None,1]+180.)%360.-180.
    angles=np.rad2deg(2*np.arccos(np.minimum(1.,np.abs(np.einsum('nmi,ni->nm',arrays['mode_wxyz'],truth)))))
    arrays.update(truth_rpy_deg=truth_rpy, truth_wxyz=truth,
        rpy_error_deg=errors,local_mode_rpy_error_deg=mode_errors,pitch_estimate_deg=pitch_estimate,rotation_error_deg=angles,
        true_acceleration_world_m_s2=dataset['arrays']['acceleration_world_m_s2'])
    moving=np.linalg.norm(dataset['arrays']['acceleration_world_m_s2'],axis=1)>.2
    stats={}
    for i,method in enumerate(METHODS):
        pitch=errors[:,i,1]
        stats[method]={'pitch_rmse_deg':float(np.sqrt(np.mean(pitch*pitch))),
            'pitch_mae_deg':float(np.mean(np.abs(pitch))), 'pitch_max_abs_deg':float(np.max(np.abs(pitch))),
            'pitch_final_deg':float(pitch[-1]),'local_mode_rpy_rmse_deg':np.sqrt(np.mean(mode_errors[:,i]**2,axis=0)).tolist(),
            'local_mode_pitch_rmse_deg':float(np.sqrt(np.mean(mode_errors[:,i,1]**2))),
            'rotation_rmse_deg':float(np.sqrt(np.mean(angles[:,i]**2))),
            'native_samples':len(pitch),'dynamic_pitch_rmse_deg':float(np.sqrt(np.mean(pitch[moving]**2))) if moving.any() else None,
            'dynamic_evaluation_samples':int(moving.sum())}
        if i<3:
            stats[method].update(maximum_projection_moment_residual=float(arrays['projection_moment_residual'][:,i].max()),
                maximum_mode_gradient_norm=float(arrays['mode_gradient_norm'][:,i].max()),
                acceleration_world_rmse_m_s2=float(np.sqrt(np.mean(np.sum((arrays['acceleration_world_mean'][:,i]-arrays['true_acceleration_world_m_s2'])**2,axis=1)))))
    return stats


def plot_series_indices(stamps, rate_hz):
    """Nearest actual acquisition to a uniform display clock; no interpolation."""
    stamps=np.asarray(stamps,float)
    if stamps.ndim!=1 or not len(stamps) or rate_hz<=0:
        raise ValueError('A nonempty timestamp vector and positive display rate are required')
    targets=stamps[0]+np.arange(int(np.floor((stamps[-1]-stamps[0])*rate_hz))+1)/rate_hz
    right=np.minimum(np.searchsorted(stamps,targets),len(stamps)-1)
    left=np.maximum(right-1,0)
    chosen=np.where(abs(stamps[left]-targets)<=abs(stamps[right]-targets),left,right)
    return np.unique(np.r_[chosen,len(stamps)-1])


def finalize_figure_layout(fig, legend_columns=3):
    """One figure legend, with measured space reserved above all subplot titles."""
    handles_by_label={}
    for ax in fig.axes:
        handles,labels=ax.get_legend_handles_labels()
        for handle,label in zip(handles,labels):
            compact=label.replace(' (posterior pitch mean)','')
            if compact not in handles_by_label:handles_by_label[compact]=handle
        legend=ax.get_legend()
        if legend is not None:legend.remove()
    if handles_by_label:
        legend=fig.legend(list(handles_by_label.values()),list(handles_by_label),
            loc='upper center',bbox_to_anchor=(.5,1.),ncol=legend_columns,
            framealpha=1.,fontsize=9)
        fig.canvas.draw()
        box=legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        fig.tight_layout(rect=(0.,0.,1.,max(.5,box.y0-.025)))
    else:fig.tight_layout()


def make_figures(cases, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.labelsize':11,'legend.fontsize':9,'pdf.fonttype':42,
        'svg.fonttype':'none','figure.dpi':140,'savefig.dpi':180})
    for prefix,selected in [('pitch_marg_comparison',(0,2,3)),('pitch_imu_comparison',(1,4,5))]:
        fig,axes=plt.subplots(len(cases),2,figsize=(13.,3.35*len(cases)),squeeze=False)
        for row,(name,arrays) in enumerate(cases.items()):
            t=arrays['time_seconds']
            axes[row,0].plot(t,arrays['truth_rpy_deg'][:,1],color='black',lw=1.8,label='Synthetic truth')
            axes[row,1].axhline(0.,color='black',lw=.7,alpha=.5)
            for i in selected:
                axes[row,0].plot(t,arrays['pitch_estimate_deg'][:,i],color=COLORS[i],ls=STYLES[i],lw=1.2,label=LABELS[i])
                axes[row,1].plot(t,arrays['rpy_error_deg'][:,i,1],color=COLORS[i],ls=STYLES[i],lw=1.1,label=LABELS[i])
                if i==selected[0]:
                    sigma=np.rad2deg(np.sqrt(arrays['pitch_variance_rad2'][:,i]))
                    mean=arrays['pitch_estimate_deg'][:,i]
                    axes[row,0].fill_between(t,mean-2*sigma,mean+2*sigma,color=COLORS[i],alpha=.13,label='Posterior pitch mean ±2 std')
            axes[row,0].set_ylabel('Pitch [deg]');axes[row,1].set_ylabel('Signed pitch error [deg]')
            for ax in axes[row]:
                ax.set_title(SCENARIO_LABELS[name]);ax.set_xlabel('Time [s]');ax.grid(alpha=.22);ax.set_xlim(t[0],t[-1])
        axes[0,0].legend(loc='upper left',bbox_to_anchor=(0,1.50),ncol=2)
        axes[0,1].legend(loc='upper left',bbox_to_anchor=(0,1.36),ncol=1)
        finalize_figure_layout(fig)
        for suffix in ('png','pdf','svg'):fig.savefig(output/(prefix+'.'+suffix),bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(3,len(cases),figsize=(4.5*len(cases),8.5),squeeze=False)
    for col,(name,a) in enumerate(cases.items()):
        for k in range(3):
            ax=axes[k,col];t=a['time_seconds']
            ax.plot(t,a['true_acceleration_world_m_s2'][:,k],color='black',lw=1.7,label='Synthetic truth')
            for i in (0,1):
                ax.plot(t,a['acceleration_world_mean'][:,i,k],color=COLORS[i],lw=1.2,label=LABELS[i])
            ax.grid(alpha=.22);ax.set_xlabel('Time [s]');ax.set_ylabel('World '+'xyz'[k]+' acceleration [m/s²]')
            if k==0:ax.set_title(SCENARIO_LABELS[name])
    axes[0,0].legend(loc='upper left',bbox_to_anchor=(0,1.5),fontsize=8)
    finalize_figure_layout(fig)
    for suffix in ('png','pdf','svg'):fig.savefig(output/('translation_estimation.'+suffix),bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE.parent/'config/translational_comparison.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--publication',type=Path)
    parser.add_argument('--duration',type=float)
    parser.add_argument('--scenario',choices=('gravity_only','smooth_translation','sustained_translation'),action='append')
    parser.add_argument('--quadrature-order',type=int)
    args=parser.parse_args()
    settings=json.loads(args.config.read_text())
    if args.duration is not None:settings['duration_seconds']=args.duration
    if args.scenario:settings['scenarios']=args.scenario
    if args.quadrature_order is not None:settings['quadrature_order']=args.quadrature_order
    args.output.mkdir(parents=True,exist_ok=True)
    save_json(args.output/'configured_experiment.json',settings)
    cases={};summary={'settings':settings,'methods':dict(zip(METHODS,LABELS)),
        'truth_in_estimator':'one common initial quaternion and known fixed g/b only; no subsequent orientation/translation/generation covariance',
        'noise_policy':'full anisotropic calibrated sensor assumptions; separate mismatched generation covariances',
        'main_metric':'posterior mean pitch for probabilistic laws, point pitch for Madgwick/gyro; RMSE over every native output; local-mode errors retained separately',
        'projection':'finite forward KL retaining all quadratic/quartic coefficients, deterministic projective Gauss-Hermite integration',
        'madgwick_source':'https://x-io.co.uk/downloads/madgwick_internal_report.pdf',
        'madgwick_policy':'existing original normalized-gradient Euler implementation; fixed beta; no truth tuning; fresh magnetometer only',
        'calibration_parameters_estimated':False,'monte_carlo_estimation':False,'cases':{}}
    for name in settings['scenarios']:
        output=args.output/name;output.mkdir(exist_ok=True)
        dataset=generate_comparison_data(settings['duration_seconds'],settings['sample_interval_seconds'],settings['seed'],name)
        np.savez_compressed(output/'input_data.npz',**dataset['arrays'])
        save_json(output/'simulation_settings.json',dataset['metadata'])
        arrays,wall=estimate_case(estimator_inputs(dataset),settings,output,
            progress=lambda n,total,seconds:print(name,n,total,round(seconds,1),flush=True))
        statistics=evaluate_case(arrays,dataset)
        np.savez_compressed(output/'evaluated_timeseries.npz',**arrays)
        cases[name]=arrays;summary['cases'][name]={'statistics':statistics,'wall_seconds':wall,
            'dataset_sha256':hashlib.sha256((output/'input_data.npz').read_bytes()).hexdigest()}
        save_json(output/'summary.json',summary['cases'][name]);save_json(args.output/'summary.json',summary)
        print(name,json.dumps({key:value['pitch_rmse_deg'] for key,value in statistics.items()}),flush=True)
    publication=args.publication or args.output
    publication.mkdir(parents=True,exist_ok=True)
    make_figures(cases,publication)
    save_json(publication/'summary.json',summary)
    for name,arrays in cases.items():
        selected=plot_series_indices(arrays['time_seconds'],settings['output_series_rate_hz'])
        np.savez_compressed(publication/(name+'_plot_series.npz'),**{key:value[selected] for key,value in arrays.items()})
    print('COMPLETED',publication,flush=True)


if __name__=='__main__':main()
