#!/usr/bin/env python3
"""Read saved normal ROS trials without rerunning inference or injecting truth.

Physical truth is read only for posthoc distance scoring.  JSONL byte frontiers
are fixed before a read-only database snapshot, so an in-progress trial can be
reported without treating a later unfinished line as a completed result.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3
import time

import numpy as np


REPO = Path(__file__).resolve().parents[5]
SOURCE = REPO/'ros/examples/probtf_observation_fusion_demo/src/probtf_observation_fusion_demo'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def rows(path, frontiers, warnings):
    path = Path(path)
    frontier = frontiers.get(path.name, path.stat().st_size if path.exists() else 0)
    if not path.exists():
        return
    with path.open('rb') as stream:
        while stream.tell() < frontier:
            line = stream.readline(min(frontier-stream.tell(), 100000000))
            if not line.strip():
                continue
            if not line.endswith(b'\n'):
                warnings.append('Ignored incomplete final line in '+path.name)
                return
            yield json.loads(line)


def truth_flags(value, counter):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ('physical_truth_received', 'physical_truth_used_for_inference'):
                counter[str(child)] += 1
            if isinstance(child, (dict, list)):
                truth_flags(child, counter)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                truth_flags(child, counter)


def all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_keys(child)


def hash_comparison(path, key):
    record = read_json(path)
    if record is None:
        return dict(available=False, path=str(path))
    expected = record[key]
    changed, missing = [], []
    for name, checksum in expected.items():
        candidate = Path(name)
        candidate = candidate if candidate.is_absolute() else REPO/candidate
        if not candidate.is_file():
            missing.append(name)
        elif digest(candidate) != checksum:
            changed.append(name)
    return dict(available=True, path=str(path), manifest_sha256=digest(path),
        file_count=len(expected), rviz_file_count=sum(Path(name).suffix == '.rviz' for name in expected),
        changed_paths=changed, missing_paths=missing)


def selected(value, names):
    return None if value is None else {key: value.get(key) for key in names}


def right_jacobian(vector):
    angle = np.linalg.norm(vector)
    if angle == 0.:
        return np.eye(3)
    x, y, z = vector
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    if angle < 1e-5:
        return np.eye(3)-(.5-angle**2/24.)*cross+(1./6.-angle**2/120.)*(cross @ cross)
    return (np.eye(3)-(1.-np.cos(angle))/angle**2*cross
        +(angle-np.sin(angle))/angle**3*(cross @ cross))


def summarize(run):
    run = Path(run)
    warnings = []
    names = ('physical_trajectory.jsonl', 'inference_batch_events.jsonl', 'estimation.jsonl',
        'stiffness.jsonl', 'stiffness_ready_adoptions.jsonl', 'camera_pipeline.jsonl', 'shutdown.jsonl')
    frontiers = {name: (run/name).stat().st_size for name in names if (run/name).exists()}
    manifest_available = (run/'motion_manifest.json').is_file()
    manifest = read_json(run/'motion_manifest.json') or {}
    cutoff = read_json(run/'async_physical_cutoff.json') or read_json(run/'diagnostic_physical_cutoff.json')
    counts, checks, flags = Counter(), Counter(), Counter()
    first_physical = last_physical = None
    minimum = float('inf')
    reached = False
    for record in rows(run/'physical_trajectory.jsonl', frontiers, warnings):
        counts['physical_rows'] += 1
        first_physical = record if first_physical is None else first_physical
        last_physical = record
        minimum = min(minimum, record['physical_error_m'])
        reached |= bool(record['reached'])

    connection = sqlite3.connect('file:'+str(run/'observations.sqlite3')+'?mode=ro', uri=True)
    connection.execute('BEGIN')
    raw = {identity: (stamp, json.loads(command), kind, sequence) for identity, stamp,
        command, kind, sequence in connection.execute(
            'SELECT event_id,stamp,command_json,kind,sequence FROM events')}
    status_counts = dict(connection.execute('SELECT status,COUNT(*) FROM events GROUP BY status'))
    payload_keys, forbidden = {}, Counter()
    prohibited = {'camera_pose', 'marker_pose', 'true_K', 'actual_joints', 'physical_joint_position',
        'link_poses', 'physical_pose', 'physical_link_poses', 'simulator_stiffness'}
    for kind, payload in connection.execute('SELECT kind,payload_json FROM events WHERE payload_json IS NOT NULL'):
        value = json.loads(payload)
        payload_keys.setdefault(kind, set()).update(value)
        forbidden.update(set(all_keys(value)) & prohibited)
    acquisition_by_kind = {}
    for kind, count, start, end in connection.execute(
            'SELECT kind,COUNT(*),MIN(stamp),MAX(stamp) FROM events GROUP BY kind'):
        acquisition_by_kind[kind] = dict(archived_event_slots=count, first_acquisition_stamp=start,
            last_acquisition_stamp=end, archived_source_clock_hz=(count-1)/(end-start) if end > start else None)
    connection.close()

    previous_order = None
    last_body_progress_stamp = None
    progress_kinds, body_jobs = Counter(), {}
    for record in rows(run/'inference_batch_events.jsonl', frontiers, warnings):
        counts['body_progress_rows'] += 1
        progress_kinds[record['sensor_kind']] += 1
        actual = raw.get(record['event_id'])
        order = record['capture_stamp'], record['event_sequence']
        checks['progress_original_stamp_sequence_mismatch'] += (
            actual is None or actual[0] != order[0] or actual[3] != order[1])
        checks['progress_chronological_order_violation'] += previous_order is not None and order < previous_order
        previous_order = order
        last_body_progress_stamp = record['capture_stamp']
        metadata = record.get('result_metadata') or {}
        counts['body_fused_observations'] += bool(metadata.get('observation_fused'))

    previous_order = None
    previous_export = None
    previous_K_export_job = None
    export_order, exported_ids, export_reversals = Counter(), set(), []
    for record in rows(run/'estimation.jsonl', frontiers, warnings):
        counts['estimation_export_rows'] += 1
        truth_flags(record, flags)
        source = record['sequential_source']
        current = record['stages'][-1]['components']
        ids = [component['state_id'] for component in current]
        checks['canonical_candidate_ID_mismatch'] += ids != record['candidate_measure']['shape_source_state_ids']
        checks['terminal_canonical_ID_mismatch'] += ids != source['terminal_body_state_ids']
        checks['body_hand_query_ID_mismatch'] += any(component['hand_state_id'] != component['state_id'] for component in current)
        actual = raw.get(source['acquisition_event_id'])
        checks['export_original_t_q_sequence_mismatch'] += (actual is None or
            actual[0] != source['acquisition_stamp'] or actual[1] != source['acquisition_command']
            or actual[3] != source['acquisition_sequence'])
        order = source['acquisition_stamp'], source['acquisition_sequence']
        sequence = record['async_estimation']['stiffness_job_sequence']
        export_order['acquisition_order_reversals_in_mixed_writer_file'] += (
            previous_order is not None and order < previous_order)
        export_order['duplicate_acquisition_exports'] += source['acquisition_event_id'] in exported_ids
        exported_ids.add(source['acquisition_event_id'])
        if previous_order is not None and order < previous_order and len(export_reversals) < 3:
            export_reversals.append(dict(previous=previous_export,
                next=dict(event_id=source['acquisition_event_id'], stamp=order[0], K_job_sequence=sequence)))
        previous_order = order
        previous_export = dict(event_id=source['acquisition_event_id'], stamp=order[0], K_job_sequence=sequence)
        checks['body_batch_used_as_joint_update'] += not source['batch_is_scheduling_only']
        checks['body_batch_hazard_aggregated'] += bool(source['hazard_aggregation_applied'])
        if sequence is not None:
            counts['K_backed_estimation_exports'] += 1
            checks['K_backed_export_FIFO_violation'] += (previous_K_export_job is not None
                and sequence <= previous_K_export_job)
            previous_K_export_job = sequence
            body_jobs[sequence] = dict(ids=ids, source=source)
        else:
            counts['body_only_estimation_exports_without_K_job'] += 1

    archived = {int(path.stem.split('_')[0]): path for path in (run/'stiffness_jobs').glob('*.pickle')}
    targets, statuses, failures = Counter(), Counter(), Counter()
    completed, numerical_changes, previous_K = set(), 0, None
    first_K = last_K = None
    first_K_stamp = last_K_stamp = None
    rotation_max_errors = dict(prior_mean=0., prior_covariance=0., posterior_mean=0., posterior_covariance=0., canonical_rotation_covariance=0.)
    previous_factor_ids, previous_job = [], None
    for message in rows(run/'stiffness.jsonl', frontiers, warnings):
        sequence = message['job_sequence']
        completed.add(sequence)
        counts['K_worker_completed_rows'] += 1
        checks['K_worker_FIFO_violation'] += previous_job is not None and sequence <= previous_job
        previous_job = sequence
        record = message.get('stiffness_record') or {}
        truth_flags(record, flags)
        targets[record.get('matching_target')] += 1
        checks['K_matching_target_not_joint_rotation'] += record.get('matching_target') != 'joint_rotation'
        statuses[record.get('status', message['kind'])] += 1
        counts['K_density_updates'] += bool(record.get('update_applied'))
        if record.get('failure_reason'):
            failures[record['failure_reason']] += 1
        if sequence not in archived:
            checks['completed_K_missing_frozen_job'] += 1
            continue
        job = pickle.loads(archived[sequence].read_bytes())
        frozen = job['result']
        source = frozen.records['sequential_source']
        ids = [component.state.state_id for component in frozen.stages[-1].components]
        checks['K_canonical_body_ID_mismatch'] += ids != message['body_state_ids'] or ids != record['body_posterior_state_ids']
        checks['K_source_candidate_ID_mismatch'] += job['source_candidate_id'] != message['source_candidate_id']
        actual = raw.get(source['acquisition_event_id'])
        checks['K_original_DB_t_q_sequence_mismatch'] += (actual is None or
            actual[0] != source['acquisition_stamp'] or actual[1] != source['acquisition_command']
            or actual[3] != source['acquisition_sequence'])
        checks['K_original_record_t_q_mismatch'] += (
            record['source_stamp'] != source['acquisition_stamp']
            or record['source_command'] != source['acquisition_command']
            or message['source_stamp'] != source['acquisition_stamp']
            or message['source_command'] != source['acquisition_command'])
        if sequence in body_jobs:
            checks['K_completed_body_export_ID_mismatch'] += body_jobs[sequence]['ids'] != ids
            counts['K_to_completed_body_export_matched'] += 1
        else:
            counts['K_body_export_not_yet_present_at_frontier'] += 1
        marginal = record.get('rotation_marginal')
        if marginal:
            counts['K_rotation_marginal_records'] += 1
            checks['K_rotation_dimension_not_18'] += marginal['rotation_dimension'] != 18
            checks['K_translation_density_scored'] += bool(marginal['translation_density_scored'])
            checks['K_translation_conditioned'] += bool(marginal['translation_conditioning_used'])
            checks['K_independent_link_product_used'] += bool(marginal['independent_link_product_used'])
            branches = tuple(frozen.frozen_body_branches)
            if not branches:
                branches = (type('Branch', (), dict(reference_state=frozen.prior,
                    prior=frozen.frozen_body_prior,
                    posterior_components=frozen.frozen_body_posterior.components))(),)
            projected = frozen.projected_body_numerator
            for saved in marginal['branches']:
                branch = next(item for item in branches if item.reference_state.state_id == saved['source_state_id'])
                index = [6*branch.reference_state.edge_ids.index(name)+axis
                    for name in marginal['body_edge_ids'] for axis in (3, 4, 5)]
                counts['frozen_rotation_prior_comparisons'] += 1
                expected = branch.prior.mean[index], branch.prior.covariance[np.ix_(index, index)]
                for key, value, stored in zip(('prior_mean', 'prior_covariance'), expected,
                        (saved['prior_mean'], saved['prior_covariance'])):
                    rotation_max_errors[key] = max(rotation_max_errors[key], float(np.max(np.abs(value-stored))))
                children = (projected.posterior_components if projected is not None
                    and projected.reference_state.state_id == saved['source_state_id'] else
                    () if projected is not None else branch.posterior_components)
                by_label = {child.label: child for child in children}
                for child_record in saved['components']:
                    child = by_label[child_record['label']]
                    mean = child.distribution.mean[index]
                    covariance = child.distribution.covariance[np.ix_(index, index)]
                    counts['frozen_rotation_posterior_comparisons'] += 1
                    for key, value, stored in (('posterior_mean', mean, child_record['mean']),
                            ('posterior_covariance', covariance, child_record['covariance'])):
                        rotation_max_errors[key] = max(rotation_max_errors[key], float(np.max(np.abs(value-stored))))
                    off_diagonal = covariance.copy()
                    chart = np.zeros_like(covariance)
                    for link in range(6):
                        block = slice(3*link, 3*link+3)
                        off_diagonal[block, block] = 0.
                        chart[block, block] = right_jacobian(mean[block])
                    counts['posterior_with_nonzero_inter_link_rotation_cross'] += bool(np.any(off_diagonal != 0.))
                    canonical = next((body.state for body in frozen.stages[-1].components
                        if body.label == child.label), None)
                    if canonical is not None:
                        rotation_max_errors['canonical_rotation_covariance'] = max(
                            rotation_max_errors['canonical_rotation_covariance'], float(np.max(np.abs(
                                chart @ covariance @ chart.T-canonical.covariance[np.ix_(index, index)]))))
                        counts['canonical_rotation_covariance_comparisons'] += 1
        factors = list((record.get('posterior') or {}).get('factor_ids', ()))
        if factors:
            checks['K_old_factor_history_prefix_mismatch'] += factors[:len(previous_factor_ids)] != previous_factor_ids
            previous_factor_ids = factors
        K = (record.get('command_proposal') or {}).get('selected_K')
        if K is not None:
            counts['K_representative_records'] += 1
            numerical_changes += previous_K is not None and K != previous_K
            if first_K is None:
                first_K, first_K_stamp = K, record['source_stamp']
            last_K, last_K_stamp, previous_K = K, record['source_stamp'], K

    adopted = {row['job_sequence'] for row in rows(run/'stiffness_ready_adoptions.jsonl', frontiers, warnings)}
    shutdowns = list(rows(run/'shutdown.jsonl', frontiers, warnings))
    shutdown = shutdowns[-1] if shutdowns else None
    expected_source = manifest.get('implementation_sha256') or {}
    changed = [name for name, checksum in expected_source.items()
        if not (SOURCE/name).is_file() or digest(SOURCE/name) != checksum]
    physical = None if first_physical is None else dict(
        elapsed_physical_s=last_physical['time']-first_physical['time'],
        saved_physical_source_clock_hz=((counts['physical_rows']-1)/(
            last_physical['time']-first_physical['time']) if last_physical['time'] > first_physical['time'] else None),
        initial_error_m=first_physical['physical_error_m'], minimum_error_m=minimum,
        final_error_m=last_physical['physical_error_m'], reached_at_least_once=reached,
        terminal_reached=bool(last_physical['reached']), last_stamp=last_physical['stamp'],
        collision_checking_enabled=last_physical.get('collision_checking_enabled'),
        scope='saved physical rows at this byte frontier; truth read only for posthoc scoring')
    return dict(case=run.name, run_directory=str(run), jsonl_byte_frontiers=frontiers,
        warnings=warnings, physical_posthoc=physical,
        latest_manifest_available=manifest_available,
        latest_manifest=selected(manifest, ('finished', 'simulated_elapsed_s', 'duration',
            'estimation_drained', 'body_estimation_drained', 'stiffness_estimation_drained',
            'actual_motion_hz', 'actual_camera_hz', 'actual_joint_hz', 'actual_imu_hz',
            'actual_control_hz', 'actual_estimate_hz', 'acquisition_requests', 'acquisition_archived',
            'all_captured_sensor_data_archived', 'all_enabled_observations_processed',
            'stiffness_jobs_submitted', 'stiffness_jobs_completed', 'pending_stiffness_job_sequences',
            'stiffness_failure', 'artifact_failure', 'camera_preview_publication_route',
            'camera_preview_age_s', 'latest_estimated_K', 'implementation_unchanged_during_execution')),
        physical_cutoff=cutoff,
        shutdown=selected(shutdown, ('motion_finished', 'estimation_drained', 'acquisition_submitted',
            'acquisition_acknowledged', 'pending_stiffness_jobs', 'sensor_event_database')),
        archived_DB_status_counts=status_counts, archived_acquisition_by_kind=acquisition_by_kind,
        counts=dict(counts), source_and_law_checks=dict(checks),
        body_progress_sensor_kind_counts=dict(progress_kinds),
        last_body_progress_acquisition_stamp=last_body_progress_stamp,
        export_order_diagnostics=dict(counts=dict(export_order), reversal_examples=export_reversals,
            interpretation='estimation.jsonl interleaves immediate body-only/missing-observation exports and delayed FIFO K-worker exports; its global row order is completion order, not the chronological estimator recurrence',
            writer_sources='async_workers.py stiffness_worker writes completed non-initial K exports; event_inference_worker writes initial and no-K-job body exports'),
        matching_target_counts=dict(targets), K_status_counts=dict(statuses), K_failure_reasons=dict(failures),
        rotation_marginal_max_absolute_errors=rotation_max_errors,
        truth_flag_values=dict(flags), true_inference_truth_flag_count=flags.get('True', 0),
        raw_payload_keys={kind: sorted(keys) for kind, keys in payload_keys.items()},
        forbidden_truth_payload_key_counts=dict(forbidden), K_numeric_changes=numerical_changes,
        first_selected_K=first_K, last_worker_selected_K=last_K,
        first_K_acquisition_stamp=first_K_stamp, last_completed_K_acquisition_stamp=last_K_stamp,
        last_K_source_age_at_physical_frontier_s=(None if last_K_stamp is None or last_physical is None
            else last_physical['stamp']-last_K_stamp),
        last_completed_K_posterior_factor_count=len(previous_factor_ids),
        K_archived_job_count=len(archived), K_worker_completed_count=len(completed),
        K_coordinator_adopted_count=len(adopted), K_archived_not_worker_completed=sorted(set(archived)-completed),
        K_worker_completed_not_adopted=sorted(completed-adopted),
        source_hash_checks=dict(expected_module_count=len(expected_source), changed_current_modules=changed,
            manifest_reports_unchanged=manifest.get('implementation_unchanged_during_execution')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--protected-baseline', type=Path,
        default=Path('/tmp/probtf_no_concentration_authorized_baseline_20261018.json'))
    arguments = parser.parse_args()
    source_manifest = arguments.source_manifest or arguments.runs[0].parent/'source_manifest.json'
    report = dict(audited_wall_stamp=time.time(),
        method='stream fixed JSONL byte frontiers, query read-only archived event metadata, and compare saved joint rotational marginals with immutable estimated K-job laws; no equilibrium solve, inference replay, or truth injection',
        production_source_comparison=hash_comparison(source_manifest, 'production_sha256'),
        protected_core_and_rviz=hash_comparison(arguments.protected_baseline, 'protected_sha256'),
        limitations=['Each run has its own asynchronous computation and acquisition schedule.',
            'Body progress, completed K exports, K worker completion, adoption and shutdown are separate counts.',
            'An incomplete run is a partial observation, not an arrival or drain result.',
            'Rotation-law consistency does not guarantee hand-position accuracy or stiffness identifiability.',
            'Saved source/payload consistency checks do not prove estimation accuracy or stability.',
            'Collision checking is recorded separately; a distance condition does not establish collision-free success.'],
        runs=[summarize(run) for run in arguments.runs])
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(arguments.output), bytes=arguments.output.stat().st_size,
        cases=[dict(case=run['case'], physical=run['physical_posthoc'],
            checks=run['source_and_law_checks'], counts=run['counts']) for run in report['runs']]), ensure_ascii=False))


if __name__ == '__main__':
    main()
