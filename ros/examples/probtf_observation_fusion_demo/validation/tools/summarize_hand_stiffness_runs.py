#!/usr/bin/env python3
"""Read-only, small evidence summary of the 2026-10-19 baseline K trials.

No inference, control, or physics is rerun. Truth is used only in the saved
physical-distance summary. Original JSONL/SQLite archives remain under /tmp.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
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


def selected(record, names):
    return {key: record.get(key) for key in names}


def protected_baseline(path):
    record = json.loads(path.read_text())
    expected = record['protected_sha256']
    changed = [name for name, digest in expected.items()
               if not Path(name).is_file() or sha256(name) != digest]
    return dict(path=str(path), sha256=sha256(path), protected_file_count=len(expected),
        rviz_file_count=sum(Path(name).suffix == '.rviz' for name in expected),
        changed_current_paths=changed)


def summarize(run):
    import numpy as np
    from probtf_observation_fusion_demo.body_native import pose_query
    run = Path(run)
    setup = json.loads((run/'diagnostic_setup.json').read_text())
    physical = json.loads((run/'diagnostic_summary.json').read_text())
    cutoff = json.loads((run/'diagnostic_physical_cutoff.json').read_text())
    shutdown = list(rows(run/'shutdown.jsonl'))[-1]
    counts, errors, body_jobs, flags = Counter(), Counter(), {}, Counter()
    last_order, first_source, last_source = None, None, None
    database = sqlite3.connect('file:'+str(run/'observations.sqlite3')+'?mode=ro', uri=True)
    raw = {identity: (stamp, json.loads(command), kind, sequence)
           for identity, stamp, command, kind, sequence in database.execute(
               'SELECT event_id,stamp,command_json,kind,sequence FROM events')}
    payload_keys, forbidden = {}, Counter()
    prohibited = {'camera_pose', 'marker_pose', 'true_K', 'actual_joints',
                  'physical_joint_position', 'link_poses', 'physical_pose'}
    for kind, payload in database.execute('SELECT kind,payload_json FROM events WHERE payload_json IS NOT NULL'):
        keys = set(json.loads(payload))
        payload_keys.setdefault(kind, set()).update(keys)
        forbidden.update(keys & prohibited)
    database.close()

    progress_kinds, last_body_progress_stamp = Counter(), None
    for progress in rows(run/'inference_batch_events.jsonl'):
        counts['body_progress_event_rows'] += 1
        progress_kinds[progress.get('sensor_kind')] += 1
        metadata = progress.get('result_metadata') or {}
        counts['body_progress_fused_observations'] += bool(metadata.get('observation_fused'))
        actual = raw.get(progress['event_id'])
        errors['body_progress_original_stamp_mismatch'] += actual is None or actual[0] != progress['capture_stamp']
        last_body_progress_stamp = progress['capture_stamp']

    for record in rows(run/'estimation.jsonl'):
        counts['estimation_rows'] += 1
        truth_flags(record, flags)
        source = record['sequential_source']
        current = record['stages'][-1]['components']
        ids = [item['state_id'] for item in current]
        expected = record['candidate_measure']['shape_source_state_ids']
        errors['canonical_measure_ID_mismatch'] += ids != expected
        errors['hand_query_state_ID_mismatch'] += any(item['hand_state_id'] != item['state_id'] for item in current)
        errors['terminal_ID_mismatch'] += ids != source['terminal_body_state_ids']
        errors['async_capture_mismatch'] += (source['acquisition_stamp'] != record['async_estimation']['capture_stamp']
            or source['acquisition_command'] != record['async_estimation']['capture_command'])
        event = record.get('sensor_event') or {}
        if event:
            errors['sensor_event_source_mismatch'] += (event['event_id'] != source['acquisition_event_id']
                or event['acquisition_stamp'] != source['acquisition_stamp']
                or event['sent_command'] != source['acquisition_command'])
        actual = raw.get(source['acquisition_event_id'])
        if actual is None:
            errors['source_missing_from_archived_DB'] += 1
        else:
            errors['source_t_q_sequence_mismatch_to_DB'] += (
                actual[0] != source['acquisition_stamp'] or actual[1] != source['acquisition_command']
                or actual[3] != source['acquisition_sequence'])
        order = source['acquisition_stamp'], source['acquisition_sequence']
        errors['chronological_order_violation'] += last_order is not None and order < last_order
        last_order = order
        first_source = source if first_source is None else first_source
        last_source = source
        errors['batch_used_as_joint_update'] += not source['batch_is_scheduling_only']
        errors['hazard_aggregated_in_body'] += source['hazard_aggregation_applied']
        job = record['async_estimation']['stiffness_job_sequence']
        if job is not None:
            body_jobs[job] = dict(ids=ids, stamp=source['acquisition_stamp'],
                command=source['acquisition_command'], event=source['acquisition_event_id'])

    targets, statuses, failures, terminations, evidence_quality = (Counter() for _ in range(5))
    first_K, last_K, previous_K, K_changes, first_K_stamp, last_K_stamp = None, None, None, 0, None, None
    hand_paths, matched_K_jobs, posterior_factor_count = set(), 0, 0
    hand_query_checks = Counter()
    hand_query_errors = dict(mean_max_abs=0., covariance_max_abs=0., modal_rotation_max_abs=0.)
    archived_jobs = {int(path.stem.split('_')[0]): path for path in (run/'stiffness_jobs').glob('*.pickle')}
    for message in rows(run/'stiffness.jsonl'):
        counts['K_worker_completed_rows'] += 1
        record = message.get('stiffness_record') or {}
        truth_flags(record, flags)
        targets[record.get('matching_target')] += 1
        statuses[record.get('status', message.get('kind'))] += 1
        counts['K_density_updates'] += bool(record.get('update_applied'))
        counts['K_new_factor_records'] += record.get('new_factor_id') is not None
        for key in ('failure_reason', 'pending_failure', 'reason'):
            if record.get(key):
                failures[str(record[key])] += 1
        source = body_jobs.get(message.get('job_sequence'))
        if source is None:
            errors['K_job_without_saved_body_source'] += 1
        else:
            matched_K_jobs += 1
            errors['K_to_canonical_body_ID_mismatch'] += (
                source['ids'] != message['body_state_ids'] or source['ids'] != record['body_posterior_state_ids'])
            errors['K_original_t_q_mismatch'] += (
                source['stamp'] != message['source_stamp'] or source['stamp'] != record['source_stamp']
                or source['command'] != message['source_command'] or source['command'] != record['source_command'])
        path = tuple((name, inverse) for name, inverse in record.get('hand_path', ()))
        if path:
            hand_paths.add(path)
        marginal = record.get('hand_marginal')
        if marginal:
            counts['K_hand_marginal_records'] += 1
            errors['intermediate_link_density_was_scored'] += marginal['intermediate_link_density_scored']
            posterior_ids = {component['source_state_id'] for branch in marginal['branches']
                             for component in branch['components']}
            # Query IDs include the reference version and ':hand_K' component
            # label. Test the actual canonical 6D law, not metadata equality.
            hand_query_checks['query_ID_reexpression_difference'] += not set(record['body_posterior_state_ids']).issubset(posterior_ids)
            job = pickle.loads(archived_jobs[message['job_sequence']].read_bytes())
            frozen = job['result']
            child_queries = [component for branch in marginal['branches'] for component in branch['components']]
            for body, query_record in zip(frozen.stages[-1].components, child_queries):
                query = pose_query(body.state, frozen.hand_path, query_id='posthoc:canonical_hand', frame=frozen.root_frame)
                hand_query_checks['canonical_6D_queries_compared'] += 1
                for key, expected, actual in (
                        ('mean_max_abs', query.mean, query_record['mean']),
                        ('covariance_max_abs', query.covariance, query_record['covariance']),
                        ('modal_rotation_max_abs', query.rotation_wxyz, query_record['modal_rotation_wxyz'])):
                    difference = float(np.max(np.abs(np.asarray(expected)-actual)))
                    hand_query_errors[key] = max(hand_query_errors[key], difference)
                errors['archived_K_body_state_ID_mismatch'] += body.state.state_id not in record['body_posterior_state_ids']
        proposal = record.get('command_proposal') or {}
        value = proposal.get('selected_K')
        if value is not None:
            counts['K_representative_records'] += 1
            K_changes += previous_K is not None and value != previous_K
            if first_K is None:
                first_K, first_K_stamp = value, record['source_stamp']
            last_K, last_K_stamp, previous_K = value, record['source_stamp'], value
        selection = record.get('proposal_selection') or {}
        if selection.get('termination_reason'):
            terminations[selection['termination_reason']] += 1
        surprise = record.get('bayes_factor_surprise') or {}
        for name in ('keep_evidence', 'reset_evidence'):
            diagnostics = (surprise.get(name) or {}).get('diagnostics') or {}
            if diagnostics:
                evidence_quality['evidence_records'] += 1
                evidence_quality['finite_nonstationary_accepted'] += bool(diagnostics.get('finite_nonstationary_point_accepted'))
                evidence_quality['stationarity_verified'] += bool(diagnostics.get('stationarity_verified'))
        posterior_factor_count = len((record.get('posterior') or {}).get('factor_ids', ()))

    submitted = {int(name.stem.split('_')[0]) for name in (run/'stiffness_jobs').glob('*.pickle')}
    completed = {message['job_sequence'] for message in rows(run/'stiffness.jsonl')}
    adopted = {message['job_sequence'] for message in rows(run/'stiffness_ready_adoptions.jsonl')}
    expected_hashes = setup['source_sha256']
    changed, missing, rviz_count = [], [], 0
    for path, expected in expected_hashes.items():
        candidate = Path(path)
        rviz_count += candidate.suffix == '.rviz'
        if not candidate.is_file():
            missing.append(path)
        elif sha256(candidate) != expected:
            changed.append(path)
    manifests = ('diagnostic_setup.json', 'diagnostic_summary.json', 'diagnostic_physical_cutoff.json',
                 'estimation.jsonl', 'stiffness.jsonl', 'shutdown.jsonl')
    return dict(run_directory=str(run), case=setup['case'], intervention=setup['intervention'],
        run_start_git_HEAD=setup['git_HEAD'], inference_truth_supplied=setup['physical_truth_received'],
        file_sha256={name: sha256(run/name) for name in manifests},
        physical_posthoc=physical,
        physical_cutoff=selected(cutoff, ('physical_stamp', 'estimate_stamp', 'observation_age_s',
            'actual_motion_hz', 'actual_camera_hz', 'actual_control_hz', 'actual_joint_hz', 'actual_imu_hz',
            'acquisition_requests', 'acquisition_archived', 'all_captured_sensor_data_archived',
            'all_enabled_observations_processed', 'body_estimation_drained', 'stiffness_estimation_drained',
            'stiffness_jobs_submitted', 'stiffness_jobs_completed', 'pending_stiffness_job_sequences',
            'stiffness_failure', 'stiffness_worker_failed', 'artifact_failure', 'pending_display_artifact_count',
            'latest_estimated_K', 'active_gradient_K_est', 'implementation_unchanged_during_execution')),
        shutdown=selected(shutdown, ('acquisition_submitted', 'acquisition_acknowledged',
            'motion_finished', 'estimation_drained', 'pending_stiffness_jobs', 'sensor_event_database')),
        counts=dict(counts), chronological_source_checks=dict(errors),
        estimation_export_scope='With independent K worker, estimation.jsonl contains completed K-job exports; it does not enumerate every body-processed acquisition. Use body_progress_event_rows and archived K-job counts separately.',
        body_progress_sensor_kind_counts=dict(progress_kinds), last_body_progress_acquisition_stamp=last_body_progress_stamp,
        first_sequential_source=first_source, last_sequential_source=last_source,
        truth_flag_values=dict(flags), raw_payload_keys={kind: sorted(keys) for kind, keys in payload_keys.items()},
        true_inference_truth_flag_count=flags.get('True', 0),
        forbidden_truth_payload_key_counts=dict(forbidden), matched_K_to_body_jobs=matched_K_jobs,
        matching_target_counts=dict(targets), K_status_counts=dict(statuses),
        canonical_hand_query_checks=dict(hand_query_checks), canonical_hand_query_max_errors=hand_query_errors,
        K_failure_reasons=dict(failures), K_representative_termination_counts=dict(terminations),
        K_evidence_diagnostics=dict(evidence_quality), hand_paths=[list(path) for path in sorted(hand_paths)],
        K_numeric_changes=K_changes, first_selected_K=first_K, last_worker_selected_K=last_K,
        first_K_acquisition_stamp=first_K_stamp, last_completed_K_acquisition_stamp=last_K_stamp,
        last_completed_K_posterior_factor_count=posterior_factor_count,
        K_archived_job_count=len(submitted), K_worker_completed_count=len(completed),
        K_coordinator_adopted_count=len(adopted), K_archived_not_worker_completed=sorted(submitted-completed),
        K_worker_completed_not_adopted=sorted(completed-adopted),
        source_comparison=dict(manifest_count=len(expected_hashes), rviz_count=rviz_count,
            changed_current_paths=changed, missing_current_paths=missing,
            physical_run_reported_changed=physical['source_changed']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protected-baseline', type=Path,
        default=Path('/tmp/probtf_no_concentration_authorized_baseline_20261018.json'))
    args = parser.parse_args()
    report = dict(method='stream saved original JSONL, read-only raw-event metadata, and estimated frozen K-job objects; recompute deterministic canonical pose queries only; no observation update, control replay, or truth injection',
        protected_core_and_rviz=protected_baseline(args.protected_baseline),
        post_cutoff_edit_note=dict(source='implementation owner report after both trials completed',
            description='Three descriptive strings/docstrings in stiffness_identification.py were updated after physical execution and drain; source_sha256 now differs for this file. Trial cutoff and diagnostic_summary report no changes during execution. This report does not substitute current hashes for trial hashes.'),
        limitations=[
            'One asynchronous headless trial per sensor setting; no matched full-body-control comparison in this report.',
            'Physical truth is used only for posthoc distance and trajectory scoring.',
            'Physical cutoff, later drain, completed worker journal and coordinator adoption are distinct clocks/counts.',
            'Detailed completed K exports cover fewer acquisitions than the body progress journal; uncompleted K jobs are archived, not counted as completed density updates.',
            'Not all acquired raw was archived at shutdown; preserved physical-side journal is not the same as fused data.',
            'Source-ID/payload checks establish saved provenance consistency; they do not prove estimator accuracy or stability.',
            'Current-file hashes may differ from the completed trial after later docstring-only edits; execution integrity is taken from the source-frozen cutoff and diagnostic_summary, separately from current-file comparison.',
            'These 180-second trials did not reach; no causal or success claim follows from the implementation change.'],
        runs=[summarize(path) for path in args.runs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), bytes=args.output.stat().st_size,
        cases=[dict(case=row['case'], checks=row['chronological_source_checks'], counts=row['counts'])
               for row in report['runs']]), ensure_ascii=False))


if __name__ == '__main__':
    main()
