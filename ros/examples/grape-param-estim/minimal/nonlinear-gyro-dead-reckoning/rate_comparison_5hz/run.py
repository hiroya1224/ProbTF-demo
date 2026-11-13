#!/usr/bin/env python3
"""Compare original gyro acquisitions at .5/2/5 Hz under frozen dynamics."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
if str(PARENT) not in sys.path:sys.path.append(str(PARENT))
from engine import predict_factorized
from torque import TorqueProfile

ARCHIVE=PARENT/'publication/bag8_50_55s'
RATES=(.5,2.,5.)
SIGMAS=(.1,.5,1.,2.,2.5,5.)
MODES=('constant','linear_forced','nonlinear')
COLORS={'constant':'#d67b20','linear_forced':'#009688','nonlinear':'#1769aa'}
LABELS={'constant':'Held angular velocity','linear_forced':'Controlled, no gyroscopic term',
        'nonlinear':'Controlled nonlinear Euler'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    def encode(item):
        if isinstance(item,np.ndarray):return item.tolist()
        if isinstance(item,np.generic):return item.item()
        if isinstance(item,Path):return str(item)
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value,default=encode,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def label(value):return format(value,'g').replace('.','p')


def acquisitions(data,metadata,rate):
    """First original native gyro at/after each requested bag-clock epoch."""
    start,end=metadata['requested_window_seconds']
    desired=start+np.arange(0.,end-start,1./float(rate))
    index=np.searchsorted(data['imu_time'],desired,side='left')
    original=np.asarray(data['imu_time'])[index]
    if len(np.unique(index))!=len(index) or original[0]!=data['time'][0] or original[-1]>=data['time'][1]:
        raise ValueError('Distinct original samples must share the fixed initialization and horizon')
    selected=dict(data)
    selected['sparse_gyro_time']=original.copy();selected['sparse_gyro']=data['gyro'][index].copy()
    evidence=dict(requested_time=desired,time=original,gyro=data['gyro'][index],source_index=index,
        header_stamp_ns=data['imu_stamp_ns'][index],record_stamp_ns=data['imu_record_ns'][index])
    return selected,evidence


def common_statistics(data,prediction):
    """Score every acquisition rate on exactly the same 501 time instants."""
    common=np.linspace(*data['time'],501)
    mean=np.column_stack([np.interp(common,prediction['time'],prediction['rpy_mean_rad'][:,a]) for a in range(3)])
    covariance=np.stack([np.column_stack([np.interp(common,prediction['time'],prediction['rpy_covariance_rad2'][:,a,b])
                                        for b in range(3)]) for a in range(3)],axis=1)
    reference=Slerp(data['mocap_time'],Rotation.from_quat(data['mocap_quaternion_xyzw']))(common)
    initial=Rotation.from_quat(data['initial_quaternion_wxyz'][[1,2,3,0]]).as_euler('xyz')
    rpy=np.unwrap(np.vstack((initial,reference.as_euler('xyz'))),axis=0)[1:]
    difference=(mean-rpy+np.pi)%(2*np.pi)-np.pi
    geodesic=(reference.inv()*Rotation.from_euler('xyz',mean)).magnitude()
    std=np.sqrt(np.diagonal(covariance,axis1=1,axis2=2))
    arrays=dict(time=common,rpy_proxy_rad=mean,rpy_covariance_delta_rad2=covariance,
        reference_rpy_rad=rpy,error_rpy_rad=difference,error_geodesic_rad=geodesic)
    metrics=dict(geodesic_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(geodesic**2)))),
        geodesic_max_deg=float(np.rad2deg(geodesic.max())),geodesic_final_deg=float(np.rad2deg(geodesic[-1])),
        rpy_proxy_rmse_deg=np.rad2deg(np.sqrt(np.mean(difference**2,axis=0))),
        final_rpy_two_sigma_delta_deg=np.rad2deg(2*std[-1]),
        common_score_count=len(common),
        exact_rotation_moment_internal_difference=float(np.max(prediction['core_vs_rotation_mean_max_abs'])))
    return arrays,metrics


def render_case(destination,rate,sigma,curves):
    base=curves['nonlinear'];time=base['time']-base['time'][0]
    with PdfPages(destination/'comparison.pdf') as pdf:
        fig,axes=plt.subplots(3,1,figsize=(10,8.5),sharex=True)
        for dimension,axis in enumerate(axes):
            axis.plot(time,np.rad2deg(base['reference_rpy_rad'][:,dimension]),'k',lw=1.7,label='Mocap reference')
            for mode in ('nonlinear','linear_forced','constant'):
                value=curves[mode];mean=np.rad2deg(value['rpy_proxy_rad'][:,dimension])
                std=np.rad2deg(np.sqrt(value['rpy_covariance_delta_rad2'][:,dimension,dimension]))
                if mode!='linear_forced':axis.fill_between(time,mean-2*std,mean+2*std,
                    color=COLORS[mode],alpha=.18 if mode=='nonlinear' else .08,lw=0)
                axis.plot(time,mean,color=COLORS[mode],lw=1.6 if mode=='nonlinear' else 1.1,
                          ls='-' if mode=='nonlinear' else '--' if mode=='constant' else ':',label=LABELS[mode])
            axis.set_ylabel(('Roll','Pitch','Yaw')[dimension]+' [deg]');axis.grid(alpha=.25)
        axes[0].legend(fontsize=8);axes[-1].set_xlabel('Time from common initialization [s]')
        fig.suptitle(f'Original gyro acquisitions {rate:g} Hz / sigma {sigma:g} deg/s per reading\n'
                     'Same nominal inertia, control and frozen past residual torque at all rates',fontsize=12)
        fig.text(.06,.014,'Centers and +/-2sigma are Euler moment proxies from the same exact rotation moments; no Gaussian coverage claim.',fontsize=8)
        fig.tight_layout(rect=(0,.035,1,.93));fig.savefig(destination/'rpy.png',dpi=160);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(4,1,figsize=(10,9),sharex=True)
        for dimension,axis in enumerate(axes[:3]):
            for mode,value in curves.items():
                width=2*np.rad2deg(np.sqrt(value['rpy_covariance_delta_rad2'][:,dimension,dimension]))
                axis.plot(time,width,color=COLORS[mode],label=LABELS[mode])
            axis.set_ylabel(('Roll','Pitch','Yaw')[dimension]+' 2sigma [deg]');axis.grid(alpha=.25)
        for mode,value in curves.items():axes[3].plot(time,np.rad2deg(value['error_geodesic_rad']),color=COLORS[mode],label=LABELS[mode])
        axes[3].set_ylabel('SO(3) error [deg]');axes[3].grid(alpha=.25);axes[0].legend(fontsize=8)
        axes[-1].set_xlabel('Time [s]');fig.suptitle(f'{rate:g} Hz / sigma {sigma:g} deg/s: local width and mocap error')
        fig.tight_layout(rect=(0,.02,1,.95));pdf.savefig(fig);plt.close(fig)


def render_rate_overview(destination,rate_curves,sigma=2.5):
    fig,axes=plt.subplots(len(rate_curves),3,figsize=(14,3.2*len(rate_curves)),sharex=True,squeeze=False)
    for row,(rate,curves) in enumerate(rate_curves.items()):
        base=curves['nonlinear'];time=base['time']-base['time'][0]
        for dimension,axis in enumerate(axes[row]):
            axis.plot(time,np.rad2deg(base['reference_rpy_rad'][:,dimension]),'k',lw=1.3,label='Mocap')
            for mode in ('nonlinear','constant'):
                value=curves[mode];mean=np.rad2deg(value['rpy_proxy_rad'][:,dimension])
                width=2*np.rad2deg(np.sqrt(value['rpy_covariance_delta_rad2'][:,dimension,dimension]))
                if mode=='nonlinear':axis.fill_between(time,mean-width,mean+width,color=COLORS[mode],alpha=.18)
                axis.plot(time,mean,color=COLORS[mode],ls='-' if mode=='nonlinear' else '--',label=LABELS[mode])
            axis.set_title(f'{rate:g} Hz / '+('Roll','Pitch','Yaw')[dimension]);axis.set_ylabel('deg');axis.grid(alpha=.25)
            if row==len(rate_curves)-1:axis.set_xlabel('Time [s]')
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'Acquisition rate comparison at sigma={sigma:g} deg/s / Euler moment proxy and local 2sigma',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(destination/'rate_overview_sigma_2p5.png',dpi=170)
    fig.savefig(destination/'rate_overview_sigma_2p5.pdf');plt.close(fig)


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    data=dict(np.load(args.input,allow_pickle=False));metadata=json.loads(args.input.with_suffix('.json').read_text())
    torque_arrays=dict(np.load(args.torque,allow_pickle=False))
    profile=dict(torque=TorqueProfile(torque_arrays['torque_profile_time'],torque_arrays['torque_actuator_body'],
                                     torque_arrays['residual_mean_body']),inertia=torque_arrays['nominal_inertia_body'])
    records=[];overview={};started=time.monotonic()
    for rate in args.rates:
        selected,evidence=acquisitions(data,metadata,rate)
        directory=args.output/('rate_'+label(rate)+'_hz');directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/'acquisitions.npz',**evidence)
        for sigma in args.sigmas:
            print('rate',rate,'sigma',sigma,flush=True)
            destination=directory/('sigma_'+label(sigma)+'_deg_s');destination.mkdir(exist_ok=True)
            curves={};metrics={}
            for mode in MODES:
                prediction=predict_factorized(selected,profile,sigma,mode,output_dt=.02,max_step=args.max_step)
                curve,metric=common_statistics(data,prediction);curves[mode]=curve;metrics[mode]=metric
                # Explicit names identify the local Euler display approximation.
                prediction['rpy_proxy_rad']=prediction.pop('rpy_mean_rad')
                prediction['rpy_covariance_delta_rad2']=prediction.pop('rpy_covariance_rad2')
                np.savez_compressed(destination/(mode+'.npz'),**prediction)
                np.savez_compressed(destination/(mode+'_plot.npz'),**curve)
            record=dict(rate_hz=rate,sigma_deg_s=sigma,covariance_rad2_s2=np.eye(3)*np.deg2rad(sigma)**2,
                original_gyro_count=len(evidence['time']),actual_observation_times=evidence['time'],
                same_measurements_for_all_sigmas=True,random_sensor_noise_added=False,methods=metrics,
                state='same factored independent six-point increment law, exact quaternion second moment and rotation-entry mean/full covariance',
                visualization='Euler center f(E[R]) and delta-method covariance Df Cov(vecR) Df.T; approximate Euler statistics, not a new candidate law')
            write_json(destination/'metrics.json',record);render_case(destination,rate,sigma,curves);records.append(record)
            if sigma==2.5:overview[rate]=curves
    if overview:render_rate_overview(args.output,overview)
    write_json(args.output/'summary.json',dict(schema='grape/acquisition-rate-and-noise-comparison/v1',
        input_source=dict(path=str(args.input.resolve()),sha256=sha(args.input)),
        torque_source=dict(path=str(args.torque.resolve()),sha256=sha(args.torque)),
        original_trajectory_and_control_unchanged=True,nominal_inertia=profile['inertia'],
        rates_hz=args.rates,sigmas_deg_s=args.sigmas,results=records,wall_seconds=time.monotonic()-started,
        source_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
        initial_attitude='same original initial mocap alignment once, no later corrections',
        measurement_law='independent individual-reading isotropic Gaussian, fresh at each original acquisition; no PSD or shared-bias interpretation',
        law_representation='factored six-point increments, fixed-size exact rotation moments instead of exponential enumeration',
        display_approximation='same Euler delta method at ALL rates/noises, kept out of state propagation',
        unselected_original_gyro_rows_used_for_prediction=False,
        original_gyro_stream_used_only_at_the_saved_acquisition_indices=True,no_monte_carlo=True))
    with (args.output/'comparison_table.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['rate_hz','sigma_deg_s','method','observations','so3_rmse_deg','so3_max_deg','so3_final_deg',
                         'roll_proxy_rmse_deg','pitch_proxy_rmse_deg','yaw_proxy_rmse_deg','roll_final_2sigma_deg','pitch_final_2sigma_deg','yaw_final_2sigma_deg'])
        for r in records:
            for mode,m in r['methods'].items():writer.writerow([r['rate_hz'],r['sigma_deg_s'],mode,r['original_gyro_count'],
                m['geodesic_rmse_deg'],m['geodesic_max_deg'],m['geodesic_final_deg'],*m['rpy_proxy_rmse_deg'],*m['final_rpy_two_sigma_delta_deg']])
    print('Completed',args.output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ARCHIVE/'input.npz')
    parser.add_argument('--torque',type=Path,default=ARCHIVE/'torque_profile.npz')
    parser.add_argument('--output',type=Path,default=Path('/tmp/grape5hz-20261113/results'))
    parser.add_argument('--rates',type=float,nargs='+',default=list(RATES))
    parser.add_argument('--sigmas',type=float,nargs='+',default=list(SIGMAS))
    parser.add_argument('--max-step',type=float,default=.002)
    args=parser.parse_args()
    if any(not np.isfinite(v) or v<=0 for v in (*args.rates,*args.sigmas,args.max_step)):
        parser.error('Positive finite rates, sigmas and integration step required')
    run(args)
