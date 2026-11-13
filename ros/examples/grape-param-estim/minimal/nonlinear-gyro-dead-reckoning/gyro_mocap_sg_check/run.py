#!/usr/bin/env python3
"""Offline mocap SG/gyro clock check and a separate differentiation roundtrip."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil

os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sg_compare import differentiate_quaternions, compare_lags
from roundtrip import run_roundtrip

HERE=Path(__file__).resolve().parent
WINDOWS=(.08,.16,.32)
COLORS=('#1783a8','#d16a24','#3a9959')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    def encode(item):
        if isinstance(item,np.ndarray): return item.tolist()
        if isinstance(item,np.generic): return item.item()
        if isinstance(item,Path): return str(item)
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value,default=encode,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def lag_metrics(result):
    return {name:result[name] for name in ('best_lag_seconds','best_axis_lag_seconds',
        'best_at_search_boundary','before','after','sign_convention','fixed_support_count',
        'requested_window_seconds','effective_gyro_window_seconds','bias_or_scale_or_rotation_fitted')}


def save_figure(figure,path,pdf):
    figure.tight_layout(rect=(0,.035,1,.93))
    figure.savefig(path,dpi=180)
    pdf.savefig(figure)
    plt.close(figure)


def comparison_figures(destination,data,comparisons):
    start=float(data['eval_interval'][0]);main=comparisons[1]
    with PdfPages(destination/'gyro_comparison.pdf') as pdf:
        for shifted in (False,True):
            fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
            for dimension,axis in enumerate(axes):
                axis.plot(main['time']-start,np.rad2deg(main['gyro'][:,dimension]),color='.35',lw=.9,label='Raw gyro')
                if shifted:
                    axis.plot(main['time']-start,np.rad2deg(main['mocap_sg_unshifted'][:,dimension]),
                              color='#1783a8',alpha=.5,lw=1,label='Mocap SG, no shift')
                    axis.plot(main['time']-start,np.rad2deg(main['mocap_sg_shifted'][:,dimension]),
                              color='#d16a24',lw=1.3,label=f'Mocap SG shifted: lag={1e3*main["best_lag_seconds"]:+.2f} ms')
                else:
                    for width,color,value in zip(WINDOWS,COLORS,comparisons):
                        axis.plot(value['time']-start,np.rad2deg(value['mocap_sg_unshifted'][:,dimension]),
                                  color=color,lw=1.2,label=f'Mocap SG window {width:g} s')
                axis.set_ylabel(('wx','wy','wz')[dimension]+' [deg/s]');axis.grid(alpha=.25)
            axes[0].legend(fontsize=8,loc='best');axes[-1].set_xlabel('Time from original 5 s interval start [s]')
            fig.suptitle('Raw gyro versus mocap SG body angular velocity'+
                         (' / shift-only alignment' if shifted else ' / original header clocks'),fontsize=13)
            fig.text(.08,.012,'Positive lag: gyro(t) matches mocap-SG(t-lag). No gain, bias or axis rotation fitted.',fontsize=9)
            save_figure(fig,destination/('gyro_aligned.png' if shifted else 'gyro_unshifted.png'),pdf)
        fig,axis=plt.subplots(figsize=(10,5))
        for width,color,value in zip(WINDOWS,COLORS,comparisons):
            axis.plot(value['lag_seconds']*1e3,np.rad2deg(np.sqrt(value['vector_mse_rad2_s2'])),
                      color=color,label=f'SG {width:g} s, best {value["best_lag_seconds"]*1e3:+.2f} ms')
        axis.axvline(0,color='k',ls=':',lw=1);axis.grid(alpha=.25);axis.legend()
        axis.set(xlabel='Lag [ms]; positive means gyro later',ylabel='Vector RMSE [deg/s]')
        fig.suptitle('Fixed-support relative time-shift scan, no amplitude correction')
        save_figure(fig,destination/'lag_scan.png',pdf)


def roundtrip_figures(destination,data,roundtrips):
    with PdfPages(destination/'sg_roundtrip.pdf') as pdf:
        for sparse in (False,True):
            fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
            base=roundtrips[1]['arrays'];time=base['time']-data['eval_interval'][0]
            for dimension,axis in enumerate(axes):
                axis.plot(time,np.rad2deg(base['reference_rpy_rad'][:,dimension]),'k',lw=1.5,label='Original mocap')
                for width,color,value in zip(WINDOWS,COLORS,roundtrips):
                    a=value['arrays'];key='sparse_rpy_rad' if sparse else 'full_rpy_rad'
                    axis.plot(a['time']-data['eval_interval'][0],np.rad2deg(a[key][:,dimension]),
                              color=color,lw=1.2,label=f'SG {width:g} s, integrated '+('at 0.5 Hz held rate' if sparse else 'at full rate'))
                axis.set_ylabel(('Roll','Pitch','Yaw')[dimension]+' [deg]');axis.grid(alpha=.25)
            axes[0].legend(fontsize=8,loc='best');axes[-1].set_xlabel('Time from original 5 s interval start [s]')
            fig.suptitle('Mocap -> SG body rate -> quaternion integration'+
                         (' / sparse 0.5 Hz' if sparse else ' / full-rate roundtrip'),fontsize=13)
            fig.text(.07,.012,'Offline consistency diagnostic. Initial mocap orientation used once; no later orientation resets.',fontsize=9)
            save_figure(fig,destination/('roundtrip_sparse.png' if sparse else 'roundtrip_full.png'),pdf)
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
        for width,color,value in zip(WINDOWS,COLORS,roundtrips):
            a=value['arrays'];t=a['time']-data['eval_interval'][0]
            axes[0].plot(t,np.rad2deg(a['full_error_geodesic_rad']),color=color,label=f'SG {width:g} s')
            axes[1].plot(t,np.rad2deg(a['sparse_error_geodesic_rad']),color=color,label=f'SG {width:g} s')
        for axis in axes: axis.grid(alpha=.25);axis.legend(fontsize=9);axis.set_ylabel('SO(3) error [deg]')
        axes[0].set_title('Full-rate reintegration');axes[1].set_title('0.5 Hz SG rate held until next acquisition')
        axes[1].set_xlabel('Time [s]');fig.suptitle('Differentiation/reintegration errors, shown on separate scales')
        save_figure(fig,destination/'roundtrip_errors.png',pdf)


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    data=dict(np.load(args.input,allow_pickle=False));records=[];comparisons=[];roundtrips=[]
    eval_window=tuple(data['eval_interval']);wide=(40.5,59.5);past=(40.5,49.5)
    for width in WINDOWS:
        print('SG window',width,flush=True)
        query=data['mocap_time'][(data['mocap_time']>=40)&(data['mocap_time']<=60)]
        sg=differentiate_quaternions(data['mocap_time'],data['mocap_quaternion_xyzw'],query,width,3)
        main=compare_lags(data['gyro_time'],data['gyro_rad_s'],query,sg['omega_body'],window=eval_window)
        broad=compare_lags(data['gyro_time'],data['gyro_rad_s'],query,sg['omega_body'],window=wide)
        calibration=compare_lags(data['gyro_time'],data['gyro_rad_s'],query,sg['omega_body'],window=past)
        # An independently fitted past shift is evaluated on the 5 s window;
        # it is not silently used to change either original time series.
        shifted=np.column_stack([np.interp(main['time']-calibration['best_lag_seconds'],query,sg['omega_body'][:,a]) for a in range(3)])
        past_error=main['gyro']-shifted
        roundtrip=run_roundtrip(data,window_seconds=width,degree=3,max_step=.002)
        label=f'window_{int(round(width*1000))}ms'
        np.savez_compressed(args.output/(label+'_sg.npz'),**{k:v for k,v in sg.items() if isinstance(v,np.ndarray)})
        np.savez_compressed(args.output/(label+'_comparison.npz'),**{k:v for k,v in main.items() if isinstance(v,np.ndarray)})
        np.savez_compressed(args.output/(label+'_roundtrip.npz'),**roundtrip['arrays'])
        record=dict(sg_window_seconds=width,primary=lag_metrics(main),
            wide=lag_metrics(broad),past_shift=lag_metrics(calibration),
            primary_using_past_shift=dict(lag_seconds=calibration['best_lag_seconds'],
                rmse_axis_rad_s=np.sqrt(np.mean(past_error**2,axis=0)),
                vector_rmse_rad_s=float(np.sqrt(np.mean(np.sum(past_error**2,axis=1))))),
            roundtrip=roundtrip['metrics'])
        write_json(args.output/(label+'_metrics.json'),record)
        records.append(record);comparisons.append(main);roundtrips.append(roundtrip)
    # Repeat the primary clock comparison using recorded receive times.
    rt=data['mocap_record_time'];rq=rt[(rt>=40)&(rt<=60)]
    record_sg=differentiate_quaternions(rt,data['mocap_quaternion_xyzw'],rq,.16,3)
    record_clock=compare_lags(data['gyro_record_time'],data['gyro_rad_s'],rq,record_sg['omega_body'],window=eval_window)
    comparison_figures(args.output,data,comparisons)
    roundtrip_figures(args.output,data,roundtrips)
    summary=dict(schema='grape/mocap-sg-gyro-and-roundtrip/v1',
        original_input_sha256=sha(args.input),original_metadata=json.loads(args.input.with_suffix('.json').read_text()),
        original_times_modified=False,gyro_bias_scale_rotation_fitted=False,
        sg='centered irregular-clock degree3 signed-quaternion fit with differentiated normalization',
        derivative='body omega = 2 vec(conjugate(q) Hamilton qdot), not Euler-angle derivatives',
        lag_definition='gyro(t) compared with mocap-SG(t-lag); positive means gyro later',
        lag_search_seconds=[-.25,.25],lag_grid_spacing_seconds=.001,
        fixed_comparison_support=True,primary_window_seconds=eval_window,
        broad_window_seconds=wide,past_lag_calibration_seconds=past,
        record_clock_primary=lag_metrics(record_clock),results=records,
        roundtrip_scope='offline centered-SG kinematic consistency test; not independent IMU estimation or a new torque-model replay',
        source_sha256={p.name:sha(p) for p in HERE.glob('*.py')})
    write_json(args.output/'summary.json',summary)
    with (args.output/'lag_table.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['sg_window_s','primary_lag_ms','wide_lag_ms','past_lag_ms','raw_vector_rmse_deg_s','shifted_vector_rmse_deg_s','past_shift_vector_rmse_deg_s'])
        for r in records:writer.writerow([r['sg_window_seconds'],r['primary']['best_lag_seconds']*1000,
            r['wide']['best_lag_seconds']*1000,r['past_shift']['best_lag_seconds']*1000,
            np.rad2deg(r['primary']['before']['vector_rmse_rad_s']),np.rad2deg(r['primary']['after']['vector_rmse_rad_s']),
            np.rad2deg(r['primary_using_past_shift']['vector_rmse_rad_s'])])
    with (args.output/'roundtrip_table.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['sg_window_s','method','so3_rmse_deg','so3_max_deg','so3_final_deg','roll_rmse_deg','pitch_rmse_deg','yaw_rmse_deg'])
        for record in records:
            for method in ('full_rate','sparse_0p5hz','full_rate_vs_fitted_mocap'):
                metric=record['roundtrip'][method]
                writer.writerow([record['sg_window_seconds'],method,metric['geodesic_rmse_deg'],
                    metric['geodesic_max_deg'],metric['geodesic_final_deg'],*metric['rpy_rmse_deg']])
    print('Completed',args.output,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('/tmp/gyro-mocap-sg-20261113/input.npz'))
    parser.add_argument('--output',type=Path,default=Path('/tmp/gyro-mocap-sg-20261113/results'))
    args=parser.parse_args();run(args)


if __name__=='__main__':main()
