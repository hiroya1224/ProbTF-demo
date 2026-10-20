#!/usr/bin/env python3
"""Render six saved filters and the two single-observation Bingham laws."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('feedback_comparison', HERE/'compare_feedback_orientation.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
import static_vector_noise


def single_observation_parameters(arrays, summary):
    """Use the same native direction factors, without a prior or recursion."""
    if 'assumed_sensor_noise' in summary or 'vector_noise_calibration' in summary:
        noise = summary.get('assumed_sensor_noise', summary.get('vector_noise_calibration'))
        acceleration_concentration = static_vector_noise.concentration_from_vector_noise(
            arrays['acceleration_m_s2'], noise['acceleration'])
        magnetic_concentration = static_vector_noise.concentration_from_vector_noise(
            arrays['magnetic_raw'], noise['magnetic'])
        for sensor, values in (('acceleration', acceleration_concentration), ('magnetic', magnetic_concentration)):
            key = sensor+'_direction_concentration'
            if key in arrays and not np.array_equal(arrays[key], values):
                raise ValueError('The saved '+sensor+' concentration differs from its noise calibration.')
    else:
        acceleration_concentration = summary['acceleration_direction_concentration']
        magnetic_concentration = summary['magnetic_direction_concentration']
    gravity, magnetic = comparison.direction_factors(
        arrays['acceleration_m_s2'], arrays['magnetic_raw'],
        np.asarray(summary['reference_magnetic_world_unit']),
        acceleration_concentration, magnetic_concentration)
    return np.stack((gravity, magnetic), axis=1)


def render(input_directory, output, fps=4., samples=180, duration=None):
    plt = comparison.configure_plotting()
    sys.path.insert(0, str(comparison.original.ROOT/'third_party/BinghamNLL/src'))
    from bingham.visualize.SO3s import draw_so3s
    from mpl_toolkits.mplot3d.art3d import Path3DCollection
    from probtf.distributions import BinghamOrientation
    from probtf.probability.sampling import sample_bingham_orientation

    arrays = dict(np.load(input_directory/'timeseries.npz'))
    summary = json.loads((input_directory/'summary.json').read_text())
    reference_label = summary.get('orientation_reference_label', 'モーションキャプチャ')
    labels = tuple(label.replace('（', ' (').replace('）', ')')
                   for label in summary['case_labels'].values())
    expected = tuple(label.replace('（', ' (').replace('）', ')') for label in comparison.CASE_LABELS)
    if tuple(summary['case_labels']) != tuple(comparison.CASE_IDS) or labels != expected:
        raise ValueError('The saved six conditions differ from the current comparison.')
    observation_parameters = single_observation_parameters(arrays, summary)
    observation_labels = ('加速度のみ (単独観測)', '地磁気のみ (単独観測)')
    t = arrays['time_seconds']
    end = float(t[-1]) if duration is None else min(float(t[-1]), duration)
    frame_times = np.arange(0., end, 1./fps)
    frame_times = np.append(frame_times, end)
    indices = np.minimum(np.searchsorted(t, frame_times), len(t)-1)
    output.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(21.6, 14.8), dpi=100)
    title_labels = tuple(label.replace(' (', '\n(')
        .replace('加速度と地磁気は予測', '加速度と地磁気は\n予測')
        .replace('加速度も地磁気も予測', '加速度も地磁気も\n予測')
        for label in labels+observation_labels)
    camera = np.array([1.4, -1.5, 1.])
    axes, point_clouds, mode_lines, truth_lines = [], [], [], []
    for case, label in enumerate(labels+observation_labels):
        row, column = (case//3, case%3) if case < 6 else (case-6, 3)
        center = .175+.23*column
        axis = figure.add_axes([center-.1, .628-.3*row, .2, .19], projection='3d')
        draw_so3s(axis, np.eye(3)[None], axes=None, axes_gt=None,
            point_of_view=camera, distance=9)
        clouds = [item for item in axis.collections if isinstance(item, Path3DCollection)]
        assert len(clouds) == 6
        axis.get_legend().remove()
        mode, truth = [], []
        for component, color in enumerate(comparison.COORDINATE_COLORS):
            if case < 6:
                mode.append(axis.plot([0, 0], [0, 0], [0, 0], color=color, lw=2.)[0])
            truth.append(axis.plot([0, 0], [0, 0], [0, 0], color='black', ls='--', lw=1.4)[0])
        # Reserve a separate title band: 3D tick/axis labels extend outside
        # the axes rectangle and must not share the next row's title space.
        figure.text(center, .878-.3*row, title_labels[case], fontsize=20,
                    ha='center', va='top', linespacing=1.12)
        for setter, label in zip((axis.set_xlabel, axis.set_ylabel, axis.set_zlabel), ('x', 'y', 'z')):
            setter(label, fontsize=18, labelpad=7)
        axis.tick_params(labelsize=15, pad=1)
        axis.set_xlim(-1.05, 1.05)
        axis.set_ylim(-1.05, 1.05)
        axis.set_zlim(-1.05, 1.05)
        axes.append(axis)
        point_clouds.append(clouds)
        mode_lines.append(mode)
        truth_lines.append(truth)
    from matplotlib.lines import Line2D
    divider = (axes[2].get_position().x1+axes[6].get_position().x0)/2
    figure.add_artist(Line2D([divider, divider],
        [.303, .9],
        transform=figure.transFigure, color='black', lw=1.5, ls=(0, (6, 5))))
    timeline = figure.add_axes([.065, .172, .91, .085])
    for case, label in enumerate(labels):
        comparison.plot_condition(timeline, t, arrays['error_deg'][:, case], case,
            comparison.CASE_COLORS[case], comparison.CASE_STYLES[case], comparison.CASE_MARKERS[case], title_labels[case])
    cursor = timeline.axvline(0., color='black', lw=1.)
    timeline.set_xlim(0., t[-1])
    timeline.set_ylim(0., max(10., float(arrays['error_deg'].max())*1.05))
    timeline.set_xlabel('経過時間 [s]', fontsize=20, labelpad=8)
    timeline.set_ylabel('3軸の合成誤差\n[deg]', fontsize=20, labelpad=8)
    timeline.tick_params(labelsize=18)
    from matplotlib.ticker import MaxNLocator
    timeline.yaxis.set_major_locator(MaxNLocator(nbins=4))
    timeline.grid(alpha=.2)
    legend_axis = figure.add_axes([.05, .012, .93, .095])
    legend_axis.set_axis_off()
    legend_axis.legend(*timeline.get_legend_handles_labels(), ncol=3, loc='center',
        fontsize=20, frameon=False, handlelength=2.2, columnspacing=2., labelspacing=.8)
    clock = figure.text(.5, .985, '', ha='center', va='top', fontsize=28)
    figure.text(.5, .942, '点群：Bingham 分布の軸端点\n実線：左6条件のモード軸　黒破線：'+reference_label,
        ha='center', va='top', fontsize=20, linespacing=1.2)
    width, height = figure.canvas.get_width_height()
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output/'bingham_comparison.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    generator = np.random.default_rng()
    snapshot_frames = set(np.round(np.linspace(0, len(indices)-1, 5)).astype(int))
    saved = 0
    try:
        for frame, index in enumerate(indices):
            truth = comparison.original.rotations_from_wxyz(arrays['mocap_wxyz'][index]).as_matrix()
            for case in range(8):
                parameter = (arrays['display_parameter'][index, case] if case < 6
                             else observation_parameters[index, case-6])
                orientation = BinghamOrientation.from_parameter_matrix(parameter)
                quaternions = sample_bingham_orientation(orientation, samples, rng=generator)
                rotations = comparison.original.rotations_from_wxyz(quaternions).as_matrix()
                if case < 6:
                    mode = comparison.original.rotations_from_wxyz(arrays['mode_wxyz'][index, case]).as_matrix()
                for component in range(3):
                    tips = rotations[:, :, component]
                    back = (tips*camera).sum(axis=1) < 0
                    for collection, selection in ((point_clouds[case][component], back),
                            (point_clouds[case][component+3], ~back)):
                        points = tips[selection]
                        collection._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
                    if case < 6:
                        mode_lines[case][component].set_data_3d(*np.column_stack((np.zeros(3), mode[:, component])))
                    truth_lines[case][component].set_data_3d(*np.column_stack((np.zeros(3), truth[:, component])))
            cursor.set_xdata([t[index], t[index]])
            clock.set_text(f't = {t[index]:.2f} [s]')
            figure.canvas.draw()
            process.stdin.write(figure.canvas.tostring_rgb())
            if frame in snapshot_frames:
                with plt.rc_context({'savefig.bbox': None}):
                    figure.savefig(output/f'bingham_snapshot_{saved:03d}.png', dpi=100)
                saved += 1
            if frame % max(1, int(fps*5)) == 0:
                print(f'video frame {frame+1}/{len(indices)} (t={t[index]:.2f}s)', flush=True)
    finally:
        process.stdin.close()
        status = process.wait()
        plt.close(figure)
    if status:
        raise RuntimeError(f'ffmpeg failed with exit status {status}')
    report = dict(frames=len(indices), fps=fps, width=width, height=height,
        displayed_start_seconds=float(t[indices[0]]), displayed_end_seconds=float(t[indices[-1]]),
        samples_per_distribution_per_frame=samples, condition_labels=labels,
        coordinate_colors=summary['coordinate_colors'], mocap_reference='black dashed axes', axis_label_color='black',
        plot_style=['science','no-latex'], unit_format='square brackets',
        visualizer='third_party/BinghamNLL/src/bingham/visualize/SO3s.py:draw_so3s',
        input_timeseries_sha256=hashlib.sha256((input_directory/'timeseries.npz').read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if summary.get('synthetic_data', False):
        report.pop('mocap_reference')
        report['orientation_reference'] = dict(label=reference_label, rendering='black dashed axes',
            source='synthetic trajectory; not a motion-capture measurement')
    report['typography'] = dict(condition_title_pt=20, legend_pt=20, clock_pt=28,
        explanation_pt=20, coordinate_label_pt=18, coordinate_tick_pt=15,
        timeline_label_pt=20, timeline_tick_pt=18,
        legend_condition_numbers=None,
        panel_titles=title_labels)
    observation_file = output/'single_observation_distributions.npz'
    np.savez_compressed(observation_file, display_parameter=observation_parameters,
        **{key: arrays[key] for key in ('time_seconds', 'measurement_stamps',
            'measurement_stamp_ns', 'bag_record_stamp_ns', 'bag_elapsed_seconds',
            'acceleration_direction_concentration', 'magnetic_direction_concentration') if key in arrays})
    report['single_observation_distributions'] = dict(
        labels=observation_labels, column=4, rows=['acceleration', 'magnetic'],
        acceleration_reference_world=comparison.original.SPECIFIC_FORCE_REFERENCE_WORLD.tolist(),
        magnetic_reference_world=summary['reference_magnetic_world_unit'],
        acceleration_concentration=summary.get('acceleration_direction_concentration',
            summary.get('direction_concentration_statistics', {}).get('acceleration')),
        magnetic_concentration=summary.get('magnetic_direction_concentration',
            summary.get('direction_concentration_statistics', {}).get('magnetic')),
        vector_noise_calibration=summary.get('vector_noise_calibration'),
        assumed_sensor_noise=summary.get('assumed_sensor_noise'),
        law='normalized native single-direction likelihood; no prior, gyro or temporal accumulation',
        mode_axes_drawn=False, unobserved_rotation='about the respective reference direction',
        parameters_file=observation_file.name, parameters_sha256=hashlib.sha256(observation_file.read_bytes()).hexdigest(),
        vertical_divider='black dashed; between columns3 and4')
    (output/'video.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=4.)
    parser.add_argument('--samples', type=int, default=180)
    parser.add_argument('--duration', type=float)
    args = parser.parse_args()
    if args.fps <= 0 or args.samples < 1 or (args.duration is not None and args.duration <= 0):
        parser.error('Frame rate, sample count, and duration must be positive.')
    print(json.dumps(render(args.input, args.output, args.fps, args.samples, args.duration), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
