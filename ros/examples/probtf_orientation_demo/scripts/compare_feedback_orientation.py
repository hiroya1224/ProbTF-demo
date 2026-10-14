#!/usr/bin/env python3
"""Compare six observation-feedback choices and three all-sensor choices with Madgwick."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
def load_helper(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

original = load_helper('feedback_base', 'compare_rolling_bag.py')
madgwick = load_helper('feedback_madgwick', 'compare_madgwick_rolling.py')
CASE_IDS = ('gyro_only','gyro_acceleration_output','gyro_acceleration_feedback',
            'all_information_output','all_information_acceleration_feedback','all_information_feedback')
CASE_LABELS = ('角速度の時間発展のみ',
               '角速度と加速度（加速度は予測に使わない）',
               '角速度と加速度（加速度も予測に使う）',
               'すべての情報（加速度と地磁気は予測に使わない）',
               'すべての情報（加速度のみ予測に使う）',
               'すべての情報（加速度も地磁気も予測に使う）')
CASE_COLORS = ('#0072B2','#56B4E9','#009E73','#E69F00','#CC79A7','#D55E00')
CASE_STYLES = ('-','--','-','--','-.','-')
# Rows: acceleration/magnetic evidence in the output, then in the next prior.
COEFFICIENTS = np.array(((0,0,0,0),(1,0,0,0),(1,0,1,0),
                         (1,1,0,0),(1,1,1,0),(1,1,1,1)), dtype=int)
GROUP_IDS = ('gyro','gyro_acceleration','gyro_acceleration_magnetic')
CASE_GROUPS = (0,0,1,0,1,2)
ALL_CASES = (3,4,5)

def canonical_batch(parameters):
    return parameters - np.linalg.eigvalsh(parameters)[...,-1,None,None]*np.eye(4)

def direction_factors(acceleration, magnetic, reference, acceleration_concentration=20., magnetic_concentration=20.):
    gravity = np.array([original.gravity_bingham_evidence(
        original.SPECIFIC_FORCE_REFERENCE_WORLD, vector, acceleration_concentration) for vector in acceleration])
    field = np.array([original.magnetic_bingham_evidence(reference, vector, magnetic_concentration) for vector in magnetic])
    return gravity, field

def output_parameters(carried, gravity, magnetic):
    result = np.empty((len(carried),6,4,4))
    result[:,0] = carried[:,0]
    result[:,1] = canonical_batch(carried[:,0] + gravity)
    result[:,2] = carried[:,1]
    result[:,3] = canonical_batch(carried[:,0] + gravity + magnetic)
    result[:,4] = canonical_batch(carried[:,1] + magnetic)
    result[:,5] = carried[:,2]
    return result

def modes_from_parameters(parameters, initial):
    modes = np.linalg.eigh(parameters)[1][...,-1]
    for case in range(6):
        modes[:,case] = original.continuous_quaternions(modes[:,case])
        if modes[0,case] @ initial < 0:
            modes[:,case] *= -1
    modes[0,0] = initial
    return modes

def compare_six(gyro, acceleration, magnetic, timestamps, initial, reference,
                initial_concentration=100., gyro_variance=1e-4,
                acceleration_concentration=20., magnetic_concentration=20., integration_steps=120,
                progress=None):
    """Only retained evidence is propagated; output-only factors never modify a state."""
    gravity, field = direction_factors(acceleration, magnetic, reference,
                                      acceleration_concentration, magnetic_concentration)
    initial_parameter = initial_concentration*(np.outer(initial,initial)-np.eye(4))
    states = np.array([initial_parameter.copy() for _ in range(3)])
    carried = np.empty((len(timestamps),3,4,4))
    residuals = np.zeros((len(timestamps),3))
    started = last = time.monotonic()
    for k in range(len(timestamps)):
        if k:
            interval = timestamps[k]-timestamps[k-1]
            if interval <= 0:
                raise ValueError('Measurement timestamps must strictly increase.')
            for group in range(3):
                states[group], residuals[k,group] = original.propagate(
                    states[group],gyro[k],interval,np.eye(3)*gyro_variance,integration_steps)
        states[1] = original.canonical_bingham_parameter(states[1] + gravity[k])
        states[2] = original.canonical_bingham_parameter(states[2] + gravity[k] + field[k])
        carried[k] = states
        if progress and time.monotonic()-last >= 10:
            progress(k+1,len(timestamps),time.monotonic()-started)
            last = time.monotonic()
    output = output_parameters(carried,gravity,field)
    return output,modes_from_parameters(output,initial),carried,residuals

def from_five(base, metadata):
    if list(metadata['case_labels']) != list(original.CASE_IDS):
        raise ValueError('--input-five must contain the original five-condition results.')
    gravity, field = direction_factors(base['acceleration_m_s2'],base['magnetic_raw'],
        metadata['reference_magnetic_world_unit'],metadata['acceleration_direction_concentration'],
        metadata['magnetic_direction_concentration'])
    carried = base['display_parameter'][:,[0,1,4]].copy()
    output = output_parameters(carried,gravity,field)
    modes = modes_from_parameters(output,np.array(metadata['initial_quaternion_wxyz']))
    for new,old in ((0,0),(2,1),(5,4)):
        modes[:,new] = base['mode_wxyz'][:,old]
    return output,modes,carried,base['moment_fit_residual'][:,[0,1,3]].copy()

def evaluate(modes, truth):
    axis_errors, errors = original.rotation_error_series(modes,truth)
    stats = [{'rpy_rms_deg':np.sqrt(np.mean(axis_errors[:,i]**2,axis=0)).tolist(),
              'rms_deg':float(np.sqrt(np.mean(errors[:,i]**2))),
              'final_deg':float(errors[-1,i])} for i in range(modes.shape[1])]
    return axis_errors, errors, stats

def display_rpy(modes, reference_rpy):
    rpy = np.stack([original.rotations_from_wxyz(modes[:,i]).as_euler('xyz',degrees=True)
                    for i in range(modes.shape[1])],axis=1)
    return rpy+360*np.round((reference_rpy[:,None]-rpy)/360)

def plot_errors(arrays, output, labels, colors, styles, prefix):
    plt = original.configure_matplotlib()
    plt.rcParams.update({'font.size':12})
    for logarithmic in (False,True):
        figure,axes = plt.subplots(4,1,sharex=True,figsize=(8.1,9.4))
        for component,axis in enumerate(axes):
            values = arrays['rpy_error_deg'][:,:,component] if component < 3 else arrays['error_deg']
            if logarithmic:
                axis.set_yscale('log',nonpositive='mask')
            for case,label in enumerate(labels):
                plotted = np.ma.masked_less_equal(values[:,case],0) if logarithmic else values[:,case]
                axis.plot(arrays['time_seconds'],plotted,color=colors[case],ls=styles[case],lw=1.,label=label)
            axis.set_ylabel(('ロール誤差','ピッチ誤差','ヨー誤差','3軸の合成誤差')[component]+' / deg')
            axis.set_xlim(arrays['time_seconds'][0],arrays['time_seconds'][-1])
            if not logarithmic:
                axis.set_ylim(bottom=0)
            axis.grid(alpha=.22)
        axes[0].legend(ncol=1,loc='lower left',bbox_to_anchor=(0,1.02),frameon=False,fontsize=11.5)
        axes[-1].set_xlabel('経過時間 / s')
        figure.tight_layout()
        original.save_figure(figure,output,prefix+('_semilog' if logarithmic else ''))
        plt.close(figure)

def plot_orientation(arrays, output):
    plt = original.configure_matplotlib()
    plt.rcParams.update({'font.size':12})
    for name, values, truth, components in [
        ('rpy',arrays['rpy_deg'],arrays['mocap_rpy_deg'],('roll','pitch','yaw')),
        ('quaternion',arrays['mode_wxyz'],arrays['mocap_wxyz'],('w','x','y','z'))]:
        figure,axes = plt.subplots(len(components),1,sharex=True,figsize=(8.1,7.6 if name=='rpy' else 9.2))
        for component,axis in enumerate(axes):
            for case,label in enumerate(CASE_LABELS):
                axis.plot(arrays['time_seconds'],values[:,case,component],color=CASE_COLORS[case],
                          ls=CASE_STYLES[case],lw=1.,label=label)
            axis.plot(arrays['time_seconds'],truth[:,component],color='#222222',ls=':',lw=1.,label=original.REFERENCE_LABEL)
            axis.set_ylabel(components[component]+(' / deg' if name=='rpy' else ''))
            axis.set_xlim(arrays['time_seconds'][0],arrays['time_seconds'][-1])
            axis.grid(alpha=.22)
        axes[0].legend(ncol=1,loc='lower left',bbox_to_anchor=(0,1.02),frameon=False,fontsize=11.5)
        axes[-1].set_xlabel('経過時間 / s')
        figure.tight_layout()
        original.save_figure(figure,output,name)
        plt.close(figure)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag',type=Path,default=Path.home()/'rosbags/20250303_delta_circle_trajectory_flight_5_2025-03-03-13-20-43.bag')
    parser.add_argument('--input-five',type=Path,help='Reuse the three exactly equivalent recorded feedback trajectories.')
    parser.add_argument('--madgwick-input',type=Path,help='Reuse the recorded plain Madgwick nine-axis trajectory.')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--duration',type=float)
    parser.add_argument('--beta',type=float,default=.1)
    args = parser.parse_args()
    if args.input_five and args.duration is not None:
        parser.error('--duration applies only to a fresh bag replay.')
    args.output.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    if args.input_five:
        base = dict(np.load(args.input_five/'timeseries.npz'))
        metadata = json.loads((args.input_five/'summary.json').read_text())
        output,modes,carried,residuals = from_five(base,metadata)
        cache = {'five_timeseries_sha256':hashlib.sha256((args.input_five/'timeseries.npz').read_bytes()).hexdigest(),
                 'five_summary_sha256':hashlib.sha256((args.input_five/'summary.json').read_bytes()).hexdigest()}
    else:
        imu,mocap = original.read_bag(args.bag)
        selected,truth,reference,metadata = original.prepare_streams(imu,mocap,duration=args.duration)
        truth = original.continuous_quaternions(truth)
        if truth[0] @ metadata['initial_quaternion_wxyz'] < 0:
            truth *= -1
        reference_rpy = np.rad2deg(np.unwrap(original.rotations_from_wxyz(truth).as_euler('xyz'),axis=0))
        base = dict(time_seconds=selected[:,1]-selected[0,1],measurement_stamps=selected[:,1],
                    gyro_rad_s=selected[:,2:5],acceleration_m_s2=selected[:,5:8],magnetic_raw=selected[:,8:11],
                    mocap_wxyz=truth,mocap_rpy_deg=reference_rpy)
        output,modes,carried,residuals = compare_six(base['gyro_rad_s'],base['acceleration_m_s2'],
            base['magnetic_raw'],base['measurement_stamps'],np.array(metadata['initial_quaternion_wxyz']),reference,
            progress=lambda k,n,s: print('propagation {}/{}; {:.1f}s'.format(k,n,s),flush=True))
        metadata.update(bag_path=str(args.bag.resolve()),duration_seconds=float(base['time_seconds'][-1]),
            evaluated_imu_samples=len(selected),initial_bingham_concentration=100.,
            gyro_sample_covariance_rad2_s2=(np.eye(3)*1e-4).tolist(),acceleration_direction_concentration=20.,
            magnetic_direction_concentration=20.,integration_steps=120)
        cache = {}
    keys = ('time_seconds','measurement_stamps','gyro_rad_s','acceleration_m_s2','magnetic_raw','mocap_wxyz','mocap_rpy_deg')
    arrays = {key:base[key] for key in keys}
    axis_errors,errors,stats = evaluate(modes,arrays['mocap_wxyz'])
    arrays.update(display_parameter=output,mode_wxyz=modes,carried_parameter=carried,moment_fit_residual=residuals,
                  rpy_deg=display_rpy(modes,arrays['mocap_rpy_deg']),rpy_error_deg=axis_errors,error_deg=errors)
    plot_orientation(arrays,args.output)
    plot_errors(arrays,args.output,CASE_LABELS,CASE_COLORS,CASE_STYLES,'axis_errors')
    np.savez_compressed(args.output/'timeseries.npz',**arrays)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    retained = {CASE_IDS[i]:{'acceleration':bool(row[2]),'magnetic':bool(row[3])} for i,row in enumerate(COEFFICIENTS)}
    report = {key:metadata[key] for key in ('bag_path','duration_seconds','evaluated_imu_samples','calibration_seconds',
        'calibration_samples','initial_quaternion_wxyz','reference_magnetic_world_unit','initial_bingham_concentration',
        'gyro_sample_covariance_rad2_s2','acceleration_direction_concentration','magnetic_direction_concentration','integration_steps')}
    report.update(case_labels=dict(zip(CASE_IDS,CASE_LABELS)),statistics=dict(zip(CASE_IDS,stats)),
        output_and_retained_evidence_coefficients=COEFFICIENTS.tolist(),carried_group_ids=GROUP_IDS,
        case_carried_groups=dict(zip(CASE_IDS,CASE_GROUPS)),next_prediction_evidence=retained,
        prediction_convention='right-end angular velocity; retained Bingham distribution; all sensor samples',
        script_sha256=source_sha,reused_five_trajectories=bool(args.input_five),
        source_helpers_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                              for name in ('compare_rolling_bag.py','compare_madgwick_rolling.py')},
        maximum_moment_fit_residual=float(residuals.max()),**cache)
    (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if args.madgwick_input:
        previous = dict(np.load(args.madgwick_input/'timeseries.npz'))
        previous_report = json.loads((args.madgwick_input/'summary.json').read_text())
        if previous_report['madgwick_beta'] != args.beta:
            raise ValueError('Cached Madgwick coefficient differs from --beta.')
        for key in keys:
            np.testing.assert_array_equal(previous[key],arrays[key])
        selected = list(previous_report['case_labels']).index('madgwick_marg')
        marg = previous['mode_wxyz'][:,selected]
        marg_cache = hashlib.sha256((args.madgwick_input/'timeseries.npz').read_bytes()).hexdigest()
    else:
        marg = madgwick.run_madgwick(arrays,np.array(metadata['reference_magnetic_world_unit']),args.beta)[:,1]
        marg_cache = None
    comparison = {key:arrays[key] for key in keys}
    comparison_modes = np.concatenate((modes[:,ALL_CASES],marg[:,None]),axis=1)
    comparison_axes,comparison_error,comparison_stats = evaluate(comparison_modes,arrays['mocap_wxyz'])
    comparison.update(mode_wxyz=comparison_modes,rpy_deg=display_rpy(comparison_modes,arrays['mocap_rpy_deg']),
                      rpy_error_deg=comparison_axes,error_deg=comparison_error)
    labels = [CASE_LABELS[i] for i in ALL_CASES]+['Madgwick（9軸）']
    colors = [CASE_COLORS[i] for i in ALL_CASES]+['#882255']
    styles = [CASE_STYLES[i] for i in ALL_CASES]+[':']
    paired = args.output/'madgwick_comparison'
    paired.mkdir(exist_ok=True)
    plot_errors(comparison,paired,labels,colors,styles,'method_comparison')
    np.savez_compressed(paired/'timeseries.npz',**comparison)
    identifiers = [CASE_IDS[i] for i in ALL_CASES]+['madgwick_marg']
    paired_report = dict(report,case_labels=dict(zip(identifiers,labels)),statistics=dict(zip(identifiers,comparison_stats)),
                         madgwick_beta=args.beta,reused_madgwick_timeseries_sha256=marg_cache,
                         world_from_magnetic_north_heading_deg=float(np.rad2deg(madgwick.heading_frame(metadata['reference_magnetic_world_unit'])[1])))
    (paired/'summary.json').write_text(json.dumps(paired_report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'statistics':report['statistics'],'madgwick':comparison_stats[-1],
                      'elapsed_seconds':time.monotonic()-started,'output':str(args.output)},ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__':
    main()
