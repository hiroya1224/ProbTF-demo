#!/usr/bin/env python3
"""Compare six observation-feedback choices and three all-sensor choices with the Madgwick filter."""
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
CASE_COLORS = ('#0064B0','#14A6C9','#008C46','#B69D00','#7A3DB8','#C23B22')
CASE_STYLES = ('-','--','-','--','-.','-')
# Rows: acceleration/magnetic evidence in the output, then in the next prior.
COEFFICIENTS = np.array(((0,0,0,0),(1,0,0,0),(1,0,1,0),
                         (1,1,0,0),(1,1,1,0),(1,1,1,1)), dtype=int)
GROUP_IDS = ('gyro','gyro_acceleration','gyro_acceleration_magnetic')
CASE_GROUPS = (0,0,1,0,1,2)
ALL_CASES = (3,4,5)
MADGWICK_LABEL = 'Madgwick フィルタ（9軸）'
MADGWICK_COLOR = '#000000'
LINE_WIDTH = 1.35
CASE_MARKERS = ('o', 's', '^', 'D', 'v', 'P')
MARKER_INTERVAL_SECONDS = 3.0
DASH_PATTERNS = {'--': (14., 7.), '-.': (14., 5., 2., 5.), ':': (1.2, 3.3)}
COORDINATE_COLORS = ('#D84315','#008C45','#1565C0')
EXPERIMENT_START_STAMP = original.EXPERIMENT_START_STAMP

def canonical_batch(parameters):
    return parameters - np.linalg.eigvalsh(parameters)[...,-1,None,None]*np.eye(4)

def direction_factors(acceleration, magnetic, reference, acceleration_concentration=20., magnetic_concentration=20.):
    acceleration_weights = np.broadcast_to(acceleration_concentration, (len(acceleration),))
    magnetic_weights = np.broadcast_to(magnetic_concentration, (len(magnetic),))
    gravity = np.array([original.gravity_bingham_evidence(
        original.SPECIFIC_FORCE_REFERENCE_WORLD, vector, weight)
        for vector, weight in zip(acceleration, acceleration_weights)])
    field = np.array([original.magnetic_bingham_evidence(reference, vector, weight)
        for vector, weight in zip(magnetic, magnetic_weights)])
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
                progress=None, propagation=None):
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
                # Preserve the original gyro-only control calculation.  The
                # observed groups can use an inverse supporting calibrated,
                # highly concentrated direction likelihoods.
                propagate = original.propagate if group == 0 or propagation is None else propagation
                states[group], residuals[k,group] = propagate(
                    states[group],gyro[k],interval,np.eye(3)*gyro_variance,integration_steps)
        states[1] = original.canonical_bingham_parameter(states[1] + gravity[k])
        states[2] = original.canonical_bingham_parameter(states[2] + gravity[k] + field[k])
        carried[k] = states
        if progress and time.monotonic()-last >= 10:
            progress(k+1,len(timestamps),time.monotonic()-started)
            last = time.monotonic()
    output = output_parameters(carried,gravity,field)
    return output,modes_from_parameters(output,initial),carried,residuals

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

def configure_plotting():
    plt = original.configure_matplotlib()
    import scienceplots
    plt.style.use(['science', 'no-latex'])
    plt.rcParams.update({'font.family': ['IPAexGothic', 'DejaVu Serif'], 'font.size': 12,
        'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black',
        'ytick.color': 'black', 'axes.spines.top': True, 'axes.spines.right': True,
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    return plt


def plot_condition(axis, times, values, index, color, style, marker, label):
    # Keep every value in the line. Markers select original samples only.
    marker_times = np.arange(.5*(index+1), times[-1], MARKER_INTERVAL_SECONDS)
    marker_indices = np.searchsorted(times, marker_times).tolist()
    line, = axis.plot(times, values, color=color, ls=style, lw=LINE_WIDTH,
        marker=marker, markevery=marker_indices, markersize=4.5,
        markerfacecolor='white', markeredgewidth=.95, label=label)
    if style in DASH_PATTERNS:
        line.set_dashes(DASH_PATTERNS[style])
    return line


def plot_errors(arrays, output, labels, colors, styles, prefix):
    plt = configure_plotting()
    markers = CASE_MARKERS[3:]+('*',) if prefix == 'method_comparison' else CASE_MARKERS
    figure,axes = plt.subplots(4,1,sharex=True,figsize=(8.1,9.4))
    for component,axis in enumerate(axes):
        values = arrays['rpy_error_deg'][:,:,component] if component < 3 else arrays['error_deg']
        for case,label in enumerate(labels):
            phase = CASE_LABELS.index(label) if label in CASE_LABELS else len(CASE_LABELS)
            plot_condition(axis,arrays['time_seconds'],values[:,case],phase,
                colors[case],styles[case],markers[case],label)
        axis.set_ylabel(('ロール誤差','ピッチ誤差','ヨー誤差','3軸の合成誤差')[component]+' [deg]')
        axis.set_xlim(arrays['time_seconds'][0],arrays['time_seconds'][-1])
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.18)
    axes[0].legend(ncol=1,loc='lower left',bbox_to_anchor=(0,1.02),frameon=False,
        fontsize=11.5,handlelength=4.3,numpoints=1)
    axes[-1].set_xlabel('経過時間 [s]')
    figure.tight_layout()
    original.save_figure(figure,output,prefix)
    plt.close(figure)


def plot_orientation(arrays, output):
    plt = configure_plotting()
    figure,axes = plt.subplots(3,1,sharex=True,figsize=(8.1,7.6))
    for component,axis in enumerate(axes):
        for case,label in enumerate(CASE_LABELS):
            plot_condition(axis,arrays['time_seconds'],arrays['rpy_deg'][:,case,component],case,
                CASE_COLORS[case],CASE_STYLES[case],CASE_MARKERS[case],label)
        axis.plot(arrays['time_seconds'],arrays['mocap_rpy_deg'][:,component],
            color='#333333',ls=':',lw=1.,label=original.REFERENCE_LABEL)
        axis.set_ylabel(('roll','pitch','yaw')[component]+' [deg]')
        axis.set_xlim(arrays['time_seconds'][0],arrays['time_seconds'][-1])
        axis.grid(alpha=.18)
    axes[0].legend(ncol=1,loc='lower left',bbox_to_anchor=(0,1.02),frameon=False,
        fontsize=11.5,handlelength=4.3,numpoints=1)
    axes[-1].set_xlabel('経過時間 [s]')
    figure.tight_layout()
    original.save_figure(figure,output,'rpy')
    plt.close(figure)


def read_measurements(path):
    import rosbag
    import rospy
    imu, mocap, frames = [], [], set()
    boundary = original.measurement_start(path)
    read_options = {} if boundary is None else {'start_time':rospy.Time.from_sec(boundary)}
    with rosbag.Bag(str(path),'r') as bag:
        for topic,msg,stamp in bag.read_messages(topics=['/delta/imu','/delta/mocap/pose'],
                **read_options):
            measured = msg.stamp.to_sec() if topic == '/delta/imu' else msg.header.stamp.to_sec()
            if boundary is not None and measured < boundary:
                continue
            if topic == '/delta/imu':
                imu.append([stamp.to_sec(),measured,*msg.gyro_data,*msg.acc_data,*msg.mag_data])
            else:
                q,position = msg.pose.orientation,msg.pose.position
                mocap.append([stamp.to_sec(),measured,q.w,q.x,q.y,q.z,position.x,position.y,position.z])
                frames.add(msg.header.frame_id)
    imu,mocap = np.asarray(imu,dtype=float),np.asarray(mocap,dtype=float)
    if len(imu) < 2 or len(mocap) < 2 or len(frames) != 1:
        raise ValueError('Finite IMU and mocap streams in a common frame are required.')
    if not np.isfinite(imu).all() or not np.isfinite(mocap).all():
        raise ValueError('Non-finite measurement values.')
    if np.any(np.diff(imu[:,1]) <= 0) or np.any(np.diff(mocap[:,1]) <= 0):
        raise ValueError('Measurement timestamps must strictly increase.')
    return imu,mocap,next(iter(frames))


def prepare_measurements(imu,mocap,duration=None):
    selected,truth,reference,metadata = original.prepare_streams(imu,mocap[:,:6],duration=duration)
    truth = original.continuous_quaternions(truth)
    if truth[0] @ metadata['initial_quaternion_wxyz'] < 0:
        truth *= -1
    origin,end = selected[0,1],selected[-1,1]
    position = mocap[(mocap[:,1] >= origin) & (mocap[:,1] <= end)]
    arrays = dict(time_seconds=selected[:,1]-origin,measurement_stamps=selected[:,1],
        gyro_rad_s=selected[:,2:5],acceleration_m_s2=selected[:,5:8],magnetic_raw=selected[:,8:11],
        mocap_wxyz=truth,mocap_rpy_deg=np.rad2deg(np.unwrap(original.rotations_from_wxyz(truth).as_euler('xyz'),axis=0)),
        position_measurement_stamps=position[:,1],position_time_seconds=position[:,1]-origin,position_m=position[:,6:9])
    metadata.update(duration_seconds=float(arrays['time_seconds'][-1]),evaluated_imu_samples=len(selected),
        position_samples=len(position),initial_bingham_concentration=100.,
        gyro_sample_covariance_rad2_s2=(np.eye(3)*1e-4).tolist(),acceleration_direction_concentration=20.,
        magnetic_direction_concentration=20.,integration_steps=120)
    return arrays,reference,metadata


def plot_raw_measurements(arrays,output):
    plt = configure_plotting()
    records = [('position',arrays['position_time_seconds'],arrays['position_m'],'m'),
        ('gyro',arrays['time_seconds'],arrays['gyro_rad_s'],'rad/s'),
        ('acc',arrays['time_seconds'],arrays['acceleration_m_s2'],'m/s²'),
        ('mag',arrays['time_seconds'],arrays['magnetic_raw'],'記録値')]
    limits = {}
    for name,times,values,unit in records:
        figure,axes = plt.subplots(1,3,figsize=(11.4,2.45),sharex=True,sharey=name=='acc')
        for component,axis in enumerate(axes):
            axis.plot(times,values[:,component],color=COORDINATE_COLORS[component],lw=LINE_WIDTH)
            axis.set_ylabel(('x','y','z')[component]+' ['+unit+']')
            axis.set_xlim(0,arrays['time_seconds'][-1])
            axis.grid(alpha=.22)
        if name == 'acc':
            lower = min(0.,float(values[values <= 12.].min()))
            for axis in axes:
                axis.set_ylim(lower,12.)
        figure.supxlabel('経過時間 [s]',fontsize=12)
        figure.tight_layout()
        original.save_figure(figure,output,name)
        limits[name] = [list(axis.get_ylim()) for axis in axes]
        plt.close(figure)
    return limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag',type=Path,default=Path.home()/'rosbags/20250303_delta_circle_trajectory_flight_5_2025-03-03-13-20-43.bag')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--duration',type=float)
    parser.add_argument('--beta',type=float,default=.1)
    parser.add_argument('--render-only',action='store_true')
    args = parser.parse_args()
    if args.render_only and args.duration is not None:
        parser.error('--duration applies only to a bag replay.')
    args.output.mkdir(parents=True,exist_ok=True)
    paired = args.output/'madgwick_comparison'
    paired.mkdir(exist_ok=True)
    started = time.monotonic()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.render_only:
        arrays = dict(np.load(args.output/'timeseries.npz'))
        metadata = json.loads((args.output/'summary.json').read_text())
        comparison = dict(np.load(paired/'timeseries.npz'))
        comparison_metadata = json.loads((paired/'summary.json').read_text())
        if list(metadata['case_labels']) != list(CASE_IDS) or comparison_metadata['madgwick_beta'] != args.beta:
            raise ValueError('Saved cases or the Madgwick filter coefficient differ.')
        inference_sha = metadata['inference_script_sha256']
    else:
        imu,mocap,frame = read_measurements(args.bag)
        arrays,reference,metadata = prepare_measurements(imu,mocap,duration=args.duration)
        metadata.update(bag_path=str(args.bag.resolve()),mocap_position_frame=frame)
        output,modes,carried,residuals = compare_six(arrays['gyro_rad_s'],arrays['acceleration_m_s2'],
            arrays['magnetic_raw'],arrays['measurement_stamps'],np.array(metadata['initial_quaternion_wxyz']),reference,
            progress=lambda k,n,s: print('propagation {}/{}; {:.1f}s'.format(k,n,s),flush=True))
        axis_errors,errors,stats = evaluate(modes,arrays['mocap_wxyz'])
        arrays.update(display_parameter=output,mode_wxyz=modes,carried_parameter=carried,moment_fit_residual=residuals,
            rpy_deg=display_rpy(modes,arrays['mocap_rpy_deg']),rpy_error_deg=axis_errors,error_deg=errors)
        metadata.update(statistics=dict(zip(CASE_IDS,stats)),maximum_moment_fit_residual=float(residuals.max()))
        marg = madgwick.run_madgwick(arrays,reference,args.beta)[:,1]
        keys = ('time_seconds','measurement_stamps','gyro_rad_s','acceleration_m_s2','magnetic_raw','mocap_wxyz','mocap_rpy_deg')
        comparison = {key:arrays[key] for key in keys}
        comparison_modes = np.concatenate((modes[:,ALL_CASES],marg[:,None]),axis=1)
        axis_errors,errors,stats = evaluate(comparison_modes,arrays['mocap_wxyz'])
        comparison.update(mode_wxyz=comparison_modes,rpy_deg=display_rpy(comparison_modes,arrays['mocap_rpy_deg']),
                          rpy_error_deg=axis_errors,error_deg=errors)
        comparison_metadata = dict(statistics=dict(zip([CASE_IDS[i] for i in ALL_CASES]+['madgwick_marg'],stats)),
            madgwick_beta=args.beta,world_from_magnetic_north_heading_deg=float(np.rad2deg(madgwick.heading_frame(reference)[1])))
        inference_sha = script_sha
    labels = [CASE_LABELS[i] for i in ALL_CASES]+[MADGWICK_LABEL]
    colors = [CASE_COLORS[i] for i in ALL_CASES]+[MADGWICK_COLOR]
    styles = [CASE_STYLES[i] for i in ALL_CASES]+[':']
    metadata.update(case_labels=dict(zip(CASE_IDS,CASE_LABELS)),output_and_retained_evidence_coefficients=COEFFICIENTS.tolist(),
        carried_group_ids=GROUP_IDS,case_carried_groups=dict(zip(CASE_IDS,CASE_GROUPS)),
        next_prediction_evidence={CASE_IDS[i]:{'acceleration':bool(row[2]),'magnetic':bool(row[3])} for i,row in enumerate(COEFFICIENTS)},
        prediction_convention='right-end angular velocity; retained Bingham distribution; all measurement samples',
        inference_script_sha256=inference_sha,plot_script_sha256=script_sha,
        source_helpers_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                              for name in ('compare_rolling_bag.py','compare_madgwick_rolling.py')},
        plot_colors=dict(zip(CASE_IDS,CASE_COLORS)),plot_styles=dict(zip(CASE_IDS,CASE_STYLES)),
        plot_markers=dict(zip(CASE_IDS,CASE_MARKERS)),marker_interval_seconds=MARKER_INTERVAL_SECONDS,
        plot_dash_patterns=DASH_PATTERNS,plot_style=['science','no-latex'],axis_label_color='black',unit_format='square brackets',
        plot_linewidth=LINE_WIDTH,coordinate_colors=dict(zip(('x','y','z'),COORDINATE_COLORS)))
    plot_orientation(arrays,args.output)
    plot_errors(arrays,args.output,CASE_LABELS,CASE_COLORS,CASE_STYLES,'axis_errors')
    metadata['raw_plot_limits'] = plot_raw_measurements(arrays,args.output)
    comparison_metadata = dict(metadata,**comparison_metadata)
    comparison_metadata.update(inference_script_sha256=inference_sha,plot_script_sha256=script_sha,
        coordinate_colors=dict(zip(('x','y','z'),COORDINATE_COLORS)),case_labels=dict(zip([CASE_IDS[i] for i in ALL_CASES]+['madgwick_marg'],labels)),
        plot_colors=dict(zip([CASE_IDS[i] for i in ALL_CASES]+['madgwick_marg'],colors)),
        plot_styles=dict(zip([CASE_IDS[i] for i in ALL_CASES]+['madgwick_marg'],styles)),
        plot_markers=dict(zip([CASE_IDS[i] for i in ALL_CASES]+['madgwick_marg'],CASE_MARKERS[3:]+('*',))),
        plot_linewidth=LINE_WIDTH,plot_dash_patterns=DASH_PATTERNS,plot_style=['science','no-latex'],
        axis_label_color='black',unit_format='square brackets')
    plot_errors(comparison,paired,labels,colors,styles,'method_comparison')
    for dest,values,report in ((args.output,arrays,metadata),(paired,comparison,comparison_metadata)):
        np.savez_compressed(dest/'timeseries.npz',**values)
        (dest/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'statistics':metadata['statistics'],'madgwick_filter':comparison_metadata['statistics']['madgwick_marg'],
        'samples':metadata['evaluated_imu_samples'],'duration_seconds':metadata['duration_seconds'],
        'elapsed_seconds':time.monotonic()-started,'output':str(args.output)},ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__':
    main()
