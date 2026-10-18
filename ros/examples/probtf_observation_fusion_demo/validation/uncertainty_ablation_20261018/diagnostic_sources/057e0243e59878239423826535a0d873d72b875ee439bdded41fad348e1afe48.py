#!/usr/bin/env python3
"""Stream compact posthoc diagnostics from selected uncertainty ablation runs.

Use after each run has stopped. Plain and losslessly archived .gz JSONL are
accepted. Compressed databases are left alone unless --restore-db is given;
then a temporary private copy is inspected read-only and removed afterwards.
No estimator/controller function is patched or executed by this audit.
"""

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
import traceback

import numpy as np


def exists(path):
    return path.is_file() or Path(str(path)+'.gz').is_file()


def open_text(path):
    return path.open() if path.is_file() else gzip.open(str(path)+'.gz', 'rt')


def rows(path):
    if not exists(path):
        return
    with open_text(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_json(path):
    if not exists(path):
        return None
    with open_text(path) as stream:
        return json.load(stream)


def stats(values):
    value = np.asarray(values, dtype=float)
    value = value[np.isfinite(value)]
    if not len(value):
        return dict(n=0)
    return dict(n=int(len(value)), min=float(value.min()), median=float(np.median(value)),
                p95=float(np.percentile(value, 95)), max=float(value.max()))


@contextmanager
def database_path(path, restore):
    if path.is_file():
        yield path
    elif Path(str(path)+'.gz').is_file() and restore:
        with tempfile.TemporaryDirectory(prefix='probtf-ablation-audit-') as directory:
            temporary = Path(directory)/path.name
            with gzip.open(str(path)+'.gz', 'rb') as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target, length=1 << 20)
            yield temporary
    else:
        yield None


def database_snapshot(path, restore):
    with database_path(path, restore) as actual:
        if actual is None:
            return dict(available=False,
                reason='database absent or archived; use --restore-db to inspect a private temporary copy')
        connection = sqlite3.connect('file:'+str(actual)+'?mode=ro', uri=True)
        try:
            count, payload, hashed, ready = connection.execute(
                'SELECT COUNT(*), SUM(payload_json IS NOT NULL), '
                'SUM(payload_hash IS NOT NULL), SUM(ready_wall_stamp IS NOT NULL) FROM events').fetchone()
            return dict(available=True, read_wall_stamp=time.time(), total_event_rows=count,
                payload_json_present=payload, payload_hash_present=hashed,
                ready_wall_stamp_present=ready, status_counts=connection.execute(
                    'SELECT status,COUNT(*) FROM events GROUP BY status').fetchall(),
                kind_status_counts=connection.execute('SELECT kind,status,COUNT(*),'
                    'MIN(stamp),MAX(stamp) FROM events GROUP BY kind,status').fetchall())
        finally:
            connection.close()


def _true_goal(run, setup):
    # Truth is scoring-only. Prefer a recorded capture from this run.
    first = next(rows(run/'diagnostic_control_truth.jsonl'), None)
    if first is not None:
        return np.asarray(first['goal_position']), 'original diagnostic control capture'
    import yaml
    from probtf_observation_fusion_demo.moving_scene import transform, inverse, apply
    scene = Path(__file__).resolve().parents[2]/'config/switch_press_scene.yaml'
    digest = hashlib.sha256(scene.read_bytes()).hexdigest()
    if setup.get('source_sha256', {}).get(str(scene)) != digest:
        raise ValueError('The physical scene configuration is not the archived source version.')
    settings = yaml.safe_load(scene.read_text())
    marker = settings['marker']
    pose = inverse(transform(settings['base_translation'])) @ transform(
        marker['world_translation'], marker['world_rpy'])
    return apply(pose, marker['goal_offset']), 'hash-verified original physical scene configuration'


def audit(run, *, restore_db=False):
    setup = load_json(run/'diagnostic_setup.json') or {}
    cutoff = load_json(run/'diagnostic_physical_cutoff.json')
    physical_summary = load_json(run/'diagnostic_summary.json')
    trajectory = list(rows(run/'physical_trajectory.jsonl'))
    last = trajectory[-1] if trajectory else None
    start = trajectory[0]['stamp'] if trajectory else None
    result = dict(case=run.name, diagnostic_only=True, audit_wall_stamp=time.time(),
        completed_physical_cutoff_present=cutoff is not None,
        physical_summary=physical_summary, physical_cutoff=cutoff,
        last_recorded_physical_frame=(None if last is None else
            {key:last[key] for key in ('time', 'stamp', 'command', 'actual_joints',
                                      'physical_error_m', 'reached')}),
        requested_duration_s=setup.get('duration_s'),
        observed_duration_s=None if last is None else last['time'],
        partial_run=cutoff is None, numerical_law_changed=False,
        truth_used_for_posthoc_scoring_only=True)
    branch_rows, event_ids, fits = [], set(), []
    for record in rows(run/'estimation.jsonl'):
        event = record.get('sensor_event', {})
        branches = record.get('variational_observation_update', {}).get('branch_diagnostics', [])
        if branches:
            event_ids.add(event.get('event_id'))
        for branch in branches:
            branch_rows.append(dict(event_id=event.get('event_id'),
                source_stamp=event.get('acquisition_stamp'), **{key:branch.get(key) for key in
                ('parent_label', 'finite_nonstationary_iterate_used', 'final_gradient_inf',
                 'stationarity_verified', 'completed_iterations', 'final_objective')}))
            for observation in branch.get('source_evaluator_diagnostics', {}).get('propagated_observations', []):
                residual = np.asarray(observation['predicted_mean'])-np.asarray(observation['observed_mean'])
                fits.append(dict(event_id=event.get('event_id'),
                    position_residual_m=float(np.linalg.norm(residual[:3])),
                    rotation_chart_residual_rad=float(np.linalg.norm(residual[3:]))))
    result.update(effective_events_in_K_saved_estimation=len(event_ids),
        variational_branch_diagnostics=branch_rows, variational_branches=len(branch_rows),
        finite_nonstationary_branch_count=sum(bool(r['finite_nonstationary_iterate_used']) for r in branch_rows),
        final_gradient_inf=stats([r['final_gradient_inf'] for r in branch_rows]),
        variational_fit_diagnostics=fits,
        final_fitted_position_residual_m=stats([r['position_residual_m'] for r in fits]),
        final_fitted_rotation_chart_residual_rad=stats([r['rotation_chart_residual_rad'] for r in fits]))
    adoptions = list(rows(run/'body_ready_adoptions.jsonl'))
    endpoint = None
    if cutoff is not None:
        indices = [i+1 for i,r in enumerate(adoptions) if r['candidate_id']==cutoff['last_body_candidate_id']]
        endpoint = indices[-1] if indices else None
    result['body_adoptions'] = dict(total_saved=len(adoptions), physical_cutoff_prefix=endpoint,
        cutoff_matched_by_candidate_id=endpoint is not None,
        unique_assimilated_event_ids=sorted({event for row in adoptions for event in row['assimilated_event_ids']}))
    source_ages, k_ages = [], []
    for row in rows(run/'commands.jsonl'):
        step, execution = row['step'], row['reaching_execution']
        if execution.get('gradient_source_age_s') is not None:
            source_ages.append(execution['gradient_source_age_s'])
        if step.get('source_stamp') is not None:
            k_ages.append(step['end_stamp']-step['source_stamp'])
    result['actual_interval_gradient_source_age_s'] = stats(source_ages)
    result['actual_interval_K_source_age_s'] = stats(k_ages)
    targets = (0., 1., 5., 10., 30., 60., 90., 120., 150., 179.)
    selected, distances, body_ages, statuses = {}, {}, [], {}
    for row in rows(run/'control.jsonl'):
        control = row['controller']
        statuses[control['status']] = statuses.get(control['status'], 0)+1
        if control.get('body_observation_source_stamp') is not None:
            body_ages.append(control['source_stamp']-control['body_observation_source_stamp'])
        if start is None:
            continue
        for target in targets:
            distance = abs(control['source_stamp']-start-target)
            if target not in distances or distance < distances[target]:
                distances[target] = distance
                selected[target] = dict(index=row['index'], request_id=row['request_id'],
                    source_stamp=control['source_stamp'], source_command=control['source_command'],
                    status=control['status'], selected_K=control.get('selected_K'),
                    origins={q['frame_label']:q for q in row['snapshot']['queries'] if q['axis']=='origin'})
    result.update(control_status_counts=statuses, control_body_observation_age_s=stats(body_ages))
    scored = []
    if selected:
        true_goal, source = _true_goal(run, setup)
        result['physical_goal_score_reference'] = source
        for target, sample in selected.items():
            frame = trajectory[sample['index']]
            if frame['stamp'] != sample['source_stamp'] or frame['command'] != sample['source_command']:
                raise ValueError('Selected control does not match the exact physical frame stamp/command.')
            origins = sample.pop('origins')
            hand, goal = np.asarray(origins['hand']['mean']), np.asarray(origins['switch_goal']['mean'])
            actual_hand = np.asarray(frame['tip'])
            hand_error, goal_error = hand-actual_hand, goal-true_goal
            denominator = np.linalg.norm(hand_error)*np.linalg.norm(goal_error)
            sample.update(target_t=target, actual_source_t=sample['source_stamp']-start,
                physical_error_m=frame['physical_error_m'],
                estimated_mean_distance_m=float(np.linalg.norm(hand-goal)),
                hand_mean_error_m=float(np.linalg.norm(hand_error)),
                goal_mean_error_m=float(np.linalg.norm(goal_error)),
                relative_mean_error_m=float(np.linalg.norm(hand_error-goal_error)),
                common_offset_cosine=None if denominator==0. else float(hand_error@goal_error/denominator),
                hand_error_vector=hand_error.tolist(), goal_error_vector=goal_error.tolist(),
                hand_position_covariance=origins['hand']['covariance'],
                goal_position_covariance=origins['switch_goal']['covariance'])
            scored.append(sample)
    result['selected_control_samples'] = scored
    previews = list(rows(run/'camera_previews.jsonl'))
    missing = [row for row in previews if not row['visible']]
    result['raw_preview_detection'] = dict(saved_count=len(previews), missing_count=len(missing),
        first_missing_stamp=None if not missing else missing[0]['stamp'],
        last_saved_stamp=None if not previews else previews[-1]['stamp'])
    interventions = list(rows(run/'diagnostic_marker_interventions.jsonl'))
    result['prepared_marker_interventions'] = dict(saved_count=len(interventions),
        applied_count=sum(bool(r.get('intervention_applied')) for r in interventions))
    result['posthoc_database_snapshot'] = database_snapshot(run/'observations.sqlite3', restore_db)
    result['notes'] = [
        'Physical cutoff counts differ from later raw recovery and estimator drain.',
        'estimation.jsonl is saved by the K worker and can omit body updates still queued for K.',
        'Raw preview detection differs from an available native/central pose observation.',
        'Last saved control artifacts can precede the last executed gradient; actual sample times are explicit.',
        'Gaussian corrected position means are not rigid poses.',
        'Missing cutoff or early failure never implies a fabricated completed 180 second run.']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+', help='Directory names under BASE/live')
    parser.add_argument('--base', type=Path, default=Path('/tmp/probtf_uncertainty_ablation_20261018'))
    parser.add_argument('--restore-db', action='store_true')
    args = parser.parse_args()
    output = args.base/'analysis'
    output.mkdir(exist_ok=True)
    for case in args.cases:
        if Path(case).name != case:
            parser.error('Cases must be direct directory names, not paths.')
        run = args.base/'live'/case
        try:
            result = audit(run, restore_db=args.restore_db)
        except Exception:
            result = dict(case=case, audit_failed=True, audit_failure=traceback.format_exc(),
                completed_physical_cutoff_present=exists(run/'diagnostic_physical_cutoff.json'),
                original_run_unchanged=True)
        path = output/(case+'_compact_audit.json')
        path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(case=case, output=str(path), audit_failed=result.get('audit_failed', False)),
                         sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
