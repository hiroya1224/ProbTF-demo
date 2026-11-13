"""Separate SG-derived kinematic rate comparison: full rate, 0.5/2/5 Hz.

This module never reads native gyro values, controller inputs or model torque.
The original mocap is used to create an offline centered-SG rate curve and
for read-only comparison.  The initial attitude is supplied once; later
attitude measurements never reset either integrated trajectory.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'gyro_mocap_sg_check'
DEFAULT_INPUT=PRIOR/'publication/bag8_50_55s/input.npz'


def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sg=_module('five_hz_existing_sg_compare',PRIOR/'sg_compare.py')
_kinematic=_module('five_hz_existing_roundtrip',PRIOR/'roundtrip.py')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_rate_curve(data,rate_time,omega_body,*,window_seconds=.32,max_step=.002):
    """Integrate one supplied SG curve at all three acquisition rates.

    All rate conditions use the exact same initial pose and output clock.
    A common clock equal to the earlier 0.5 Hz diagnostic's mocap/endpoint/
    2-second-knot union preserves its baseline statistics.  The integrator
    independently splits at every rate knot, including the new 5 Hz knots.
    """
    start,end=np.asarray(data['eval_interval'],float)
    source_time=np.asarray(data['mocap_time'],float)
    source_q=np.asarray(data['mocap_quaternion_xyzw'],float)
    rate_time=np.asarray(rate_time,float)
    rate=np.asarray(omega_body,float)
    if rate_time[0]>start or rate_time[-1]<end:
        raise ValueError('The SG rate curve must bracket the complete evaluation interval')
    sparse_times={label:start+np.arange(0.,end-start,period) for label,period in
                  (('0p5hz',2.),('2hz',.5),('5hz',.2))}
    native=source_time[(source_time>=start)&(source_time<=end)]
    query=np.unique(np.r_[start,native,sparse_times['0p5hz'],end])
    reference=Slerp(source_time,Rotation.from_quat(source_q))(query).as_quat()
    initial=reference[0,[3,0,1,2]]
    full=_kinematic.integrate_body_rate(query,initial,rate_time,rate,max_step=max_step)
    error,rpy_error,score=_kinematic._errors(full,reference)
    arrays=dict(time=query,rate_time=rate_time,sg_omega_body=rate,
        reference_quaternion_xyzw=reference,reference_rpy_rad=_kinematic._rpy(reference),
        full_q_wxyz=full,full_rpy_rad=_kinematic._rpy(full[:,[1,2,3,0]]),
        full_error_geodesic_rad=error,full_error_rpy_rad=rpy_error)
    metrics=dict(full_rate=score)
    for label,stamps in sparse_times.items():
        selected=np.column_stack([np.interp(stamps,rate_time,rate[:,axis]) for axis in range(3)])
        quaternion=_kinematic.integrate_body_rate(query,initial,stamps,selected,
            interpolation='zoh',max_step=max_step)
        error,rpy_error,score=_kinematic._errors(quaternion,reference)
        arrays.update({label+'_q_wxyz':quaternion,label+'_rpy_rad':_kinematic._rpy(quaternion[:,[1,2,3,0]]),
            label+'_error_geodesic_rad':error,label+'_error_rpy_rad':rpy_error,
            label+'_rate_time':stamps,label+'_rate_body':selected})
        metrics[label]=score
    metrics.update(window_seconds=float(window_seconds),sg_degree=3,
        comparison_count=len(query),same_comparison_clock=True,
        acquisition_counts={label:len(value) for label,value in sparse_times.items()},
        periods_seconds={'0p5hz':2.,'2hz':.5,'5hz':.2},
        input='same centered-SG body angular velocity derived from mocap',
        sampling_clock='exact requested start+k*period for each rate, excluding the end; not new native gyro observations',
        attitude_initialization='one raw-mocap SLERP at evaluation start, no later resets',
        full_rate_interpolation='piecewise linear, RK4; finite integration approximation',
        sparse_interpolation='zero-order hold with exact constant-rate right quaternion increments',
        gyro_measurement_values_used=False,force_torque_model_used=False,
        artificial_noise_added=False,uncertainty_propagated=False,
        result_role='offline kinematic sampling diagnostic, not an independent gyro/IMU estimate or controlled dynamics prediction',
        units='angles degrees in metrics; omega radians/second and quaternion wxyz in arrays')
    return dict(arrays=arrays,metrics=metrics)


def run_sg_benchmark(data,window_seconds=.32,*,max_step=.002,rate_grid_dt=.0025):
    """Build the unchanged SG curve on a numerical grid, then compare rates."""
    start,end=np.asarray(data['eval_interval'],float)
    if not np.isfinite(rate_grid_dt) or rate_grid_dt<=0.:
        raise ValueError('A finite positive numerical SG grid spacing is required')
    rate_time=np.linspace(start,end,int(np.ceil((end-start)/rate_grid_dt))+1)
    sg=_sg.differentiate_quaternions(data['mocap_time'],data['mocap_quaternion_xyzw'],
        rate_time,window_seconds=window_seconds,degree=3)
    result=compare_rate_curve(data,rate_time,sg['omega_body'],
        window_seconds=window_seconds,max_step=max_step)
    result['arrays']['sg_quaternion_xyzw']=sg['quaternion_xyzw']
    result['metrics'].update(numerical_sg_grid_max_step_seconds=float(np.max(np.diff(rate_time))),
        numerical_sg_grid_count=len(rate_time),
        sg_grid_role='numerical polynomial evaluations; not additional independently acquired mocap samples',
        source_sha256={name:_sha(path) for name,path in
            (('sg_compare.py',PRIOR/'sg_compare.py'),('roundtrip.py',PRIOR/'roundtrip.py'),
             ('sg_benchmark.py',Path(__file__)) )})
    return result


def _plot(result,destination):
    os.environ.setdefault('MPLBACKEND','Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    a=result['arrays'];time=a['time']-a['time'][0]
    labels=('Roll','Pitch','Yaw')
    with PdfPages(destination/'sg_benchmark.pdf') as pdf:
        figure,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
        for dimension,axis in enumerate(axes):
            axis.plot(time,np.rad2deg(a['reference_rpy_rad'][:,dimension]),'k',lw=1.8,label='Original mocap')
            for key,label,color in (('full','Full-rate SG curve','#529c65'),
                                    ('0p5hz','0.5 Hz SG rate, held','#d08028'),
                                    ('2hz','2 Hz SG rate, held','#9756b1'),
                                    ('5hz','5 Hz SG rate, held','#1b79a0')):
                axis.plot(time,np.rad2deg(a[key+'_rpy_rad'][:,dimension]),
                          color=color,lw=1.3,label=label)
            axis.set_ylabel(labels[dimension]+' [deg]');axis.grid(alpha=.25)
        axes[0].legend(fontsize=8);axes[-1].set_xlabel('Time from common initial pose [s]')
        figure.suptitle(f'Mocap-derived SG curve, window {result["metrics"]["window_seconds"]:g} s: acquisition-rate check')
        figure.tight_layout(rect=(0,.02,1,.94));figure.savefig(destination/'sg_benchmark.png',dpi=180)
        pdf.savefig(figure);plt.close(figure)
        figure,axis=plt.subplots(figsize=(10,5))
        for key,label,color in (('full','Full-rate SG curve','#529c65'),
                                ('0p5hz','0.5 Hz SG rate, held','#d08028'),
                                ('2hz','2 Hz SG rate, held','#9756b1'),
                                ('5hz','5 Hz SG rate, held','#1b79a0')):
            axis.plot(time,np.rad2deg(a[key+'_error_geodesic_rad']),color=color,label=label)
        axis.set(xlabel='Time [s]',ylabel='SO(3) error [deg]');axis.grid(alpha=.25);axis.legend()
        figure.suptitle('Kinematic diagnostic only; no independent gyro reconstruction or noise propagation')
        figure.tight_layout(rect=(0,.02,1,.94));figure.savefig(destination/'sg_benchmark_errors.png',dpi=180)
        pdf.savefig(figure);plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--output',type=Path,default=Path('/tmp/grape5hz-20261113'))
    parser.add_argument('--sg-window',type=float,default=.32)
    args=parser.parse_args()
    result=run_sg_benchmark(dict(np.load(args.input,allow_pickle=False)),args.sg_window)
    result['metrics']['input_sha256']=_sha(args.input)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'sg_benchmark.npz',**result['arrays'])
    (args.output/'sg_benchmark.json').write_text(json.dumps(result['metrics'],indent=2,allow_nan=False)+'\n')
    for label in ('0p5hz','2hz','5hz'):
        selected={key:value for key,value in result['arrays'].items()
            if key.startswith(label+'_') or key in ('time','reference_quaternion_xyzw','reference_rpy_rad')}
        np.savez_compressed(args.output/('sg_benchmark_'+label+'.npz'),**selected)
        metadata=dict(rate_label=label,metric=result['metrics'][label],
            result_role=result['metrics']['result_role'],input_sha256=result['metrics']['input_sha256'],
            original_result_summary='sg_benchmark.json',noise_propagation=False)
        (args.output/('sg_benchmark_'+label+'.json')).write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    _plot(result,args.output)
    print(json.dumps(result['metrics'],indent=2))


if __name__=='__main__':main()
