#!/usr/bin/env python3
"""Render the six saved Bingham sequences with the existing SO(3) visualizer."""
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


def render(input_directory, output, fps=4., samples=180, duration=None):
    plt = comparison.configure_plotting()
    sys.path.insert(0, str(comparison.original.ROOT/'third_party/BinghamNLL/src'))
    from bingham.visualize.SO3s import draw_so3s
    from mpl_toolkits.mplot3d.art3d import Path3DCollection
    from probtf.distributions import BinghamOrientation
    from probtf.probability.sampling import sample_bingham_orientation

    arrays = dict(np.load(input_directory/'timeseries.npz'))
    summary = json.loads((input_directory/'summary.json').read_text())
    labels = tuple(label.replace('（', ' (').replace('）', ')')
                   for label in summary['case_labels'].values())
    expected = tuple(label.replace('（', ' (').replace('）', ')') for label in comparison.CASE_LABELS)
    if tuple(summary['case_labels']) != tuple(comparison.CASE_IDS) or labels != expected:
        raise ValueError('The saved six conditions differ from the current comparison.')
    t = arrays['time_seconds']
    end = float(t[-1]) if duration is None else min(float(t[-1]), duration)
    frame_times = np.arange(0., end, 1./fps)
    frame_times = np.append(frame_times, end)
    indices = np.minimum(np.searchsorted(t, frame_times), len(t)-1)
    output.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(16, 14.8), dpi=100)
    grid = figure.add_gridspec(3, 3, height_ratios=(3.6, 3.6, 1.8),
        left=.04, right=.96, top=.88, bottom=.14, hspace=.43)
    camera = np.array([1.4, -1.5, 1.])
    axes, point_clouds, mode_lines, truth_lines = [], [], [], []
    for case, label in enumerate(labels):
        axis = figure.add_subplot(grid[case//3, case%3], projection='3d')
        draw_so3s(axis, np.eye(3)[None], axes=None, axes_gt=None,
            point_of_view=camera, distance=9)
        clouds = [item for item in axis.collections if isinstance(item, Path3DCollection)]
        assert len(clouds) == 6
        axis.get_legend().remove()
        mode, truth = [], []
        for component, color in enumerate(comparison.COORDINATE_COLORS):
            mode.append(axis.plot([0, 0], [0, 0], [0, 0], color=color, lw=2.)[0])
            truth.append(axis.plot([0, 0], [0, 0], [0, 0], color='black', ls='--', lw=1.4)[0])
        axis.set_title(label.replace(' (', '\n('), fontsize=10, pad=10)
        for setter, label in zip((axis.set_xlabel, axis.set_ylabel, axis.set_zlabel), ('x', 'y', 'z')):
            setter(label)
        axis.set_xlim(-1.05, 1.05)
        axis.set_ylim(-1.05, 1.05)
        axis.set_zlim(-1.05, 1.05)
        axes.append(axis)
        point_clouds.append(clouds)
        mode_lines.append(mode)
        truth_lines.append(truth)
    timeline = figure.add_subplot(grid[2, :])
    for case, label in enumerate(labels):
        comparison.plot_condition(timeline, t, arrays['error_deg'][:, case], case,
            comparison.CASE_COLORS[case], comparison.CASE_STYLES[case], comparison.CASE_MARKERS[case], label)
    cursor = timeline.axvline(0., color='black', lw=1.)
    timeline.set_xlim(0., t[-1])
    timeline.set_ylim(0., max(10., float(arrays['error_deg'].max())*1.05))
    timeline.set_xlabel('経過時間 [s]')
    timeline.set_ylabel('3軸の合成誤差 [deg]')
    timeline.grid(alpha=.2)
    timeline.legend(ncol=2, loc='upper left', bbox_to_anchor=(0., -.28), fontsize=9, frameon=False, handlelength=4.3)
    clock = figure.text(.5, .98, '', ha='center', va='top', fontsize=15)
    figure.text(.5, .947, '点群：Bingham 分布の軸端点　赤・緑・青の実線：分布のモード　黒破線：モーションキャプチャ',
        ha='center', va='top', fontsize=11)
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
            for case in range(6):
                orientation = BinghamOrientation.from_parameter_matrix(arrays['display_parameter'][index, case])
                quaternions = sample_bingham_orientation(orientation, samples, rng=generator)
                rotations = comparison.original.rotations_from_wxyz(quaternions).as_matrix()
                mode = comparison.original.rotations_from_wxyz(arrays['mode_wxyz'][index, case]).as_matrix()
                for component in range(3):
                    tips = rotations[:, :, component]
                    back = (tips*camera).sum(axis=1) < 0
                    for collection, selection in ((point_clouds[case][component], back),
                            (point_clouds[case][component+3], ~back)):
                        points = tips[selection]
                        collection._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
                    mode_lines[case][component].set_data_3d(*np.column_stack((np.zeros(3), mode[:, component])))
                    truth_lines[case][component].set_data_3d(*np.column_stack((np.zeros(3), truth[:, component])))
            cursor.set_xdata([t[index], t[index]])
            clock.set_text(f't = {t[index]:.2f} [s]')
            figure.canvas.draw()
            process.stdin.write(figure.canvas.tostring_rgb())
            if frame in snapshot_frames:
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
