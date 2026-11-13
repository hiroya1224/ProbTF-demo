#!/usr/bin/env python3
"""Three-frequency motion and a focused RPY/gyro figure at .5 Hz, sigma 5."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
from trajectory import generate_dataset
from gyro_progress_data import extract_progress
from focused_plotting import render_focused
from audit import audit_dataset

_spec = importlib.util.spec_from_file_location('synthetic_focused_shared_runner', HERE/'run.py')
_shared = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _shared
_spec.loader.exec_module(_shared)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rate, sigma = .5, 5.
    data, metadata = generate_dataset(motion_profile='three_frequency')
    np.savez_compressed(output/'input.npz', **data)
    _shared.write_json(output/'input.json', metadata)
    selected, acquisitions = _shared.select_gyro(data, rate)
    np.savez_compressed(output/'acquisitions.npz', **acquisitions)
    profile = dict(inertia=data['inertia_body_kg_m2'],
                   torque=_shared.TorqueProfile(data['torque_time'], data['torque_body_nm']))
    audit = audit_dataset(data, profile, rates=(rate,))
    _shared.write_json(output/'audit.json', audit)
    rpy_curves, gyro_curves, metrics, events = {}, {}, {}, {}
    for method in ('nonlinear', 'linear_forced'):
        print('Three-frequency focused reconstruction:', method, flush=True)
        prediction = _shared.predict_factorized(selected, profile, sigma, method,
                                                output_dt=.01, max_step=.002)
        rpy_curves[method], metrics[method] = _shared.score(data, prediction)
        progress = extract_progress(prediction, acquisitions, sigma)
        curve = progress['arrays']
        curve['reference_omega_rad_s'] = np.column_stack([
            np.interp(curve['time'], data['imu_time'], data['gyro_true_body_rad_s'][:, axis]) for axis in range(3)])
        gyro_curves[method] = curve
        events[method] = dict(events=progress['events'], metadata=progress['metadata'])
        prediction['rpy_proxy_rad'] = prediction.pop('rpy_mean_rad')
        prediction['rpy_covariance_delta_rad2'] = prediction.pop('rpy_covariance_rad2')
        np.savez_compressed(output/(method+'.npz'), **prediction)
        np.savez_compressed(output/(method+'_rpy_plot.npz'), **rpy_curves[method])
        np.savez_compressed(output/(method+'_gyro_plot.npz'), **curve)
    _shared.write_json(output/'gyro_events.json', events)
    render_focused(output, rpy_curves, gyro_curves, acquisitions, rate=rate, sigma=sigma)
    source_paths = [HERE/name for name in ('focus.py', 'focused_plotting.py', 'trajectory.py',
                    'audit.py', 'run.py', 'gyro_progress_data.py')]
    summary = dict(schema='grape/synthetic-three-frequency-focused/v1',
        rate_hz=rate, assumed_sigma_deg_s=sigma, duration_seconds=5., generation=metadata,
        layout=dict(top=['roll', 'pitch', 'yaw'], bottom=['omega_x', 'omega_y', 'omega_z']),
        methods=metrics, gyro_moment_verification={m: e['metadata'] for m, e in events.items()},
        known_input='same fixed J and 1kHz generated inverse-Euler torque for both methods',
        predictor_keys=list(selected), ongoing_truth_used_in_prediction=False,
        gyro_update='fresh independent Gaussian at original acquisition; both before/after limits retained',
        bands='same law: Euler moment proxy and delta-method RPY covariance; direct angular-rate mean/full covariance; +/-2sigma without clipping',
        source_sha256={p.name: sha(p) for p in source_paths},
        shared_source_sha256={name: sha(HERE.parent/name) for name in
            ('propagation.py', 'torque.py', 'rate_comparison_5hz/engine.py')})
    _shared.write_json(output/'summary.json', summary)
    lines = ['# 3周波数の合成運動：0.5 Hz・sigma=5 deg/s', '',
             '[指定の2×3図PDF](attitude_and_gyro.pdf) / [PNG](attitude_and_gyro.png)', '',
             '上段はroll・pitch・yaw、下段はbody角速度のx・y・z。黒は人工真値、青は非線形Euler、'
             '赤破線はジャイロ項を省いた角速度動力学。橙の×は取得gyro。帯は同じ候補の±2sigma。', '',
             '各軸に3つの異なる周波数を、振幅のある成分として加えた。5秒間、観測は0・2・4秒。'
             '角速度は観測前後を同じ時刻に両方保持し、幅を見せるための分散操作を行っていない。', '',
             '5 deg/sは従来と同じ入力Gaussianの仮定標準偏差。生成gyroのノイズは0.2 deg/s、'
             '固定biasは[0.03,-0.02,0.04] deg/s。既知の同じ慣性・入力トルクを両方式へ渡す。', '',
             '| 軸 | 振幅 [deg] | 周波数 [Hz] | 位相 [rad] |', '|---|---|---|---|']
    for i, axis in enumerate(('roll', 'pitch', 'yaw')):
        lines.append('| '+axis+' | '+ ' | '.join(', '.join(f'{v:g}' for v in metadata[key][i])
                     for key in ('rpy_amplitude_deg', 'rpy_frequency_hz', 'rpy_phase_rad'))+' |')
    lines.extend(['', '元の2周波数の図・結果は保持。新しい入力・測定列・取得時刻・分布モーメント・描画数値を別保存した。',
                  '[検証](audit.json) / [数値要約](summary.json) / [方法と再現](../../README.md)', ''])
    (output/'INDEX.md').write_text('\n'.join(lines))
    files = [dict(path=str(p.relative_to(output)), bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json']
    _shared.write_json(output/'manifest.json', dict(schema='grape/synthetic-focused-publication/v1',
        files=files, bytes_excluding_manifest=sum(p['bytes'] for p in files),
        old_results_overwritten=False, raw_bags_videos_or_intermediates_included=False))
    print('Saved', output/'attitude_and_gyro.pdf', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/synthetic-three-frequency-20261113'))
    run(parser.parse_args().output)
