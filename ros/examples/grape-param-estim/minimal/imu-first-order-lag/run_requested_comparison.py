#!/usr/bin/env python3
"""Only the requested real-IMU sensor/trajectory/applied-torque comparison.

The old fit and recording are frozen.  Each attitude observer receives the
model-plus-applied-wrench angular increment; its GA posterior then drives the
next translational dynamics interval.  Magnetics are output-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/grape-requested-comparison-matplotlib')

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'publication/imu_native_20261031'
CASES = ('failure1', 'failure2', 'success')
LABELS = {'gravity_only': 'Gravity-only observation + dynamics feedback',
          'translation_aware': 'Translation-aware observation + dynamics feedback'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(serializable(value), indent=2,
                                   ensure_ascii=False, allow_nan=False) + '\n')


def estimator_inputs(data):
    """Keep measured clocks/commands and declared initial conditions only."""
    selected = {k: v for k, v in data.items()
                if not k.startswith('mocap_') and k != 'metadata'}
    meta = data.get('metadata', {})
    selected['metadata'] = {k: meta[k] for k in
        ('case', 'bag_sha256', 'measurement_clock', 'requested_window') if k in meta}
    return selected


def read_case(source, case):
    directory = Path(source)
    with np.load(directory / 'inputs' / (case + '.npz'), allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    data['metadata'] = json.loads((directory / 'inputs' / (case + '.json')).read_text())
    return data


def load_frozen_fit(source, case, data, model_path):
    """Restore the saved distribution; reevaluate its point without optimizing.

    Historical published fit_results.npz and run fit_arrays.npz have different
    filenames from load_fit_cache's arrays.npz.  This local conversion is
    explicit and cannot accidentally read the published replay arrays.npz.
    """
    from imu_model import ImuDynamicsProblem, load_fit_cache
    from run import validate_fit_inputs

    original = Path(source) / case
    arrays_path = original / 'fit_arrays.npz'
    if not arrays_path.exists():
        arrays_path = original / 'fit_results.npz'
    with np.load(arrays_path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    with tempfile.TemporaryDirectory(prefix='grape-frozen-fit-') as folder:
        folder = Path(folder)
        shutil.copyfile(original / 'estimate.json', folder / 'estimate.json')
        np.savez_compressed(folder / 'arrays.npz', **arrays)
        fit = load_fit_cache(folder, model_path)
    validate_fit_inputs(fit, data, model_path)
    coordinate = np.asarray(fit['estimate']['parameter_distribution']['reduced_log_tau_coordinate'])
    options = fit['estimate']['method']['options']
    problem = ImuDynamicsProblem(data, fit['model'], options)
    evaluation = problem.evaluate(coordinate)
    diagnostics = {}
    for key in ('residual', 'raw_residual_wrench', 'corrected_gyro', 'corrected_specific_force'):
        if key in arrays:
            diagnostics[key + '_reevaluation_max_abs_difference'] = float(
                np.max(np.abs(arrays[key] - evaluation[key])))
    # Only deterministic means/derivatives of this SAME candidate are filled.
    # No solver or covariance re-estimation is called here.
    fit['arrays'].update({key: value for key, value in evaluation.items()
                          if isinstance(value, np.ndarray)})
    fit['arrays'].update(time=np.asarray(data['time']).copy(),
        jacobian_reduced_log_tau=evaluation['jacobian'],
        gyro_smooth=problem.gyro['smooth'], gyro_derivative=problem.gyro['derivative'],
        gyro_second_derivative=problem.gyro['second_derivative'],
        accel_smooth=problem.accel['smooth'], accel_derivative=problem.accel['derivative'],
        accel_second_derivative=problem.accel['second_derivative'])
    fit['problem'], fit['evaluation'] = problem, evaluation
    fit['frozen_fit_provenance'] = dict(estimate_sha256=digest(original / 'estimate.json'),
        fit_arrays_sha256=digest(arrays_path), fit_arrays_path=str(arrays_path.resolve()),
        refitted=False, parameter_covariance_reestimated=False, **diagnostics)
    return fit


def load_angular_carrier(source, case, data):
    """Reuse the recorded physical model+applied-wrench angular dynamics."""
    path = Path(source) / case / 'arrays.npz'
    with np.load(path, allow_pickle=False) as z:
        model = {k[len('model_'):]: z[k].copy() for k in z.files if k.startswith('model_')}
    if not np.array_equal(model['time'], data['time']):
        raise ValueError('Frozen model carrier and original input clocks do not match')
    return model, dict(path=str(path.resolve()), sha256=digest(path),
        use='body angular dynamics and its shared source factors; absolute free-model pose/position is not substituted for the observer state')


def numeric_arrays(value):
    return {k: v for k, v in value.items()
            if isinstance(v, np.ndarray) and v.dtype.kind != 'O'}


def run_case(case, source, output, model_path, options):
    from imu_orientation_comparison import estimate_orientation_comparison
    from imu_coupled_replay import prepare_coupled_model, replay_coupled_model
    from focused_reports import write_requested_figures

    destination = Path(output) / case
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = {'case': case, 'status': 'running', 'stage': 'inputs'}

    def stage(name, **extra):
        status.update(stage=name, elapsed_seconds=time.monotonic() - started, **extra)
        write_json(destination / 'status.json', status)
        print(json.dumps(serializable(status), ensure_ascii=False), flush=True)

    try:
        full_data = read_case(source, case)
        data = estimator_inputs(full_data)
        fit = load_frozen_fit(source, case, data, model_path)
        common_model, carrier_source = load_angular_carrier(source, case, data)
        stage('model_increment_and_shared_wrench_profiles')
        shared = prepare_coupled_model(data, fit, common_model,
            options={'integration_max_step_seconds': .005})
        observer_options = dict(options)
        observer_options.update(shared['orientation_prediction_options'])
        observer_options['progress'] = lambda branch, done, total, seconds: stage(
            'orientation', branch=branch, rows_completed=done, rows_total=total,
            orientation_wall_seconds=seconds)
        stage('orientation')
        orientation = estimate_orientation_comparison(data, float(fit['taus'][1]),
            float(fit['taus'][2]), np.asarray(fit['tau_covariance_seconds'])[1:3, 1:3],
            options=observer_options)
        write_json(destination / 'orientation_metadata.json',
            {k: v['metadata'] for k, v in orientation.items() if k in LABELS})
        tracks = {}
        for method in ('gravity_only', 'translation_aware'):
            stage('attitude_feedback_dynamics', branch=method)
            track = replay_coupled_model(data, fit, shared, orientation[method],
                options={'integration_max_step_seconds': .005})
            tracks[LABELS[method]] = track
            np.savez_compressed(destination / (method + '_full.npz'),
                **numeric_arrays(track), **{'orientation__' + k: v
                    for k, v in numeric_arrays(orientation[method]).items()})
            write_json(destination / (method + '_metadata.json'), track.get('metadata', {}))
        stage('requested_five_figure_types')
        report = write_requested_figures(destination, case, full_data, common_model,
            fit['taus'], trajectories=tracks)
        np.savez_compressed(destination / 'figure_data.npz', **report['arrays'])
        metadata = dict(schema='grape/attitude-feedback-requested-comparison/v1', case=case,
            figures=report['metadata'], frozen_fit=fit['frozen_fit_provenance'],
            angular_carrier=carrier_source, observer_comparison=orientation['metadata'],
            shared_model=shared.get('metadata', {}),
            input_sha256=digest(Path(source) / 'inputs' / (case + '.npz')),
            input_metadata_sha256=digest(Path(source) / 'inputs' / (case + '.json')),
            native_rows=len(data['time']), native_time_range=[data['time'][0], data['time'][-1]],
            mocap_use='initial conditions only in estimation; original supported rows for plot reference',
            plot_files=report['files'], options=options,
            physical_parameters_and_three_taus_frozen=True, monte_carlo=False,
            body_sensor_response_shared_between_attitude_methods=True,
            total_wall_seconds=time.monotonic() - started)
        write_json(destination / 'comparison_metadata.json', metadata)
        stage('completed', status='completed')
        return metadata
    except Exception as error:
        stage('failed', status='failed', error=type(error).__name__ + ': ' + str(error),
              traceback=traceback.format_exc())
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=Path('/tmp/grape-attitude-feedback-20261106'))
    parser.add_argument('--vehicle-model', type=Path, default=HERE.parent / 'grape_vehicle_model.json')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--quadrature-order', type=int, default=7)
    parser.add_argument('--projection-iterations', type=int, default=3)
    parser.add_argument('--haar-radial', type=int, default=32)
    parser.add_argument('--haar-angles', type=int, default=32)
    parser.add_argument('--local-mixture-weight', type=float, default=1e-5)
    parser.add_argument('--integration-tolerance', type=float, default=1e-5)
    args = parser.parse_args(argv)
    if args.output.resolve() == args.source.resolve() or args.source.resolve() in args.output.resolve().parents:
        raise ValueError('Use a separate output; the historical result is preserved')
    from run import require_separate_output
    require_separate_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    options = {'quadrature_order': args.quadrature_order,
               'projection_iterations': args.projection_iterations,
               'window_seconds': .08, 'degree': 3, 'initial_concentration': 100.,
               'quadrature_options': dict(global_radial=args.haar_radial,
                   global_angles=args.haar_angles,
                   local_mixture_weight=args.local_mixture_weight,
                   adaptive_refinement=True,
                   refinement_tolerance=args.integration_tolerance)}
    files = [HERE / name for name in ('run_requested_comparison.py', 'imu_orientation.py',
        'imu_orientation_comparison.py', 'imu_coupled_replay.py', 'focused_reports.py',
        'imu_model.py', 'imu_replay.py', 'imu_signal.py')]
    core_scripts = HERE.parents[4] / 'ros/examples/probtf_orientation_demo/scripts'
    files.extend(sorted(core_scripts.glob('translational_*.py')))
    hashes = {str(p): digest(p) for p in files}
    write_json(args.output / 'source_hashes_at_start.json', hashes)
    results = []
    if len(args.cases) == 1 or args.jobs == 1:
        results = [run_case(c, args.source, args.output, args.vehicle_model, options) for c in args.cases]
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(args.cases)),
                mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(run_case, c, args.source, args.output, args.vehicle_model, options): c
                       for c in args.cases}
            completed = {futures[f]: f.result() for f in as_completed(futures)}
        results = [completed[c] for c in args.cases]
    reports = [str(args.output / c / 'requested_results.pdf') for c in args.cases]
    combined = args.output / 'requested_sensor_trajectory_torque_comparison.pdf'
    subprocess.run(['pdfunite', *reports, str(combined)], check=True)
    write_json(args.output / 'manifest.json', dict(
        schema='grape/attitude-feedback-requested-comparison/v1',
        completed_utc=datetime.now(timezone.utc).isoformat(), cases=results,
        source_hashes_at_start=hashes, source_hashes_at_completion={str(p): digest(p) for p in files},
        combined_report=combined.name, requested_figure_types=['gyro', 'accel', 'xyz', 'rpy', 'residual_torque']))
    print('COMPLETED', combined, flush=True)


if __name__ == '__main__':
    main()
