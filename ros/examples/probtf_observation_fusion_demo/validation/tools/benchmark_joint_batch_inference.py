#!/usr/bin/env python3
"""Benchmark one joint posterior from every raw acquisition in a saved prefix.

This is the user-authorized joint-batch approximation. It does not assert
numerical equivalence to the historical per-event assumed-density filter.
Original DB/BLOB/commands are read-only; bookkeeping changes use isolated copies.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import sqlite3
import time

from benchmark_batch_inference import (
    PACKAGE, clone_prefix, digest, rows, select_prefix, sha, source_hashes, worker_entry,
)


def _binaries():
    from probtf_observation_fusion_demo.auxiliary_equilibrium_native import _library_path
    from probtf_observation_fusion_demo.dynamic_physics import _library_path as physics_library
    return {name: dict(path=str(path), sha256=sha(path), ABI=2)
            for name, path in (('auxiliary_equilibrium', _library_path(None)),
                               ('physical_dynamics', physics_library(None)))}


def _ledger(run, selected):
    from probtf_observation_fusion_demo.async_state import ActionStep
    end = max(json.loads(row['payload_json'])['frame_index'] for row in selected)
    actions = []
    for row in rows(run/'commands.jsonl'):
        step = ActionStep(**row['step'])
        if step.index >= end:
            break
        if step.index != len(actions):
            raise ValueError('Original actual-command ledger has a gap.')
        actions.append(step)
    if len(actions) != end:
        raise ValueError('Original actual-command ledger does not reach the selected frontier.')
    return tuple(actions)


def _time_indexed_law_summary(record, selected):
    """Check the saved per-acquisition K factors, not only raw ID metadata.

    This audits the installed factor definition and its immutable source
    chain. Native mechanical routing itself is additionally exercised by
    the real-arm command/score regression; this is not a native call trace.
    """
    joint, stiffness = record.get('joint_batch', {}), record.get('stiffness', {})
    historical = joint.get('time_resolved_stiffness', {})
    product = stiffness.get('stiffness_batch_factorization', {})
    effective = list(record.get('observation_ids', ()))
    atomic = stiffness.get('time_indexed_body_factors', ())
    captured = {row['event_id']: row for row in selected}
    histories = historical.get('prefix_history', ())
    likelihoods = [{item['observation_id']: item for item in factors}
                   for factors in joint.get('likelihood_factors', ())]
    sources, source_ok, chain_ok, weights_ok, parameter_ok = [], [], [], [], []
    by_history = []
    for history in histories:
        previous, links = history.get('initial_prefix_id'), {}
        valid = isinstance(previous, str) and len(previous) == 64
        for link in history.get('factor_links', ()):
            matches = link.get('previous_prefix_id') == previous
            if link.get('effective'):
                try:
                    expected = hashlib.sha256(bytes.fromhex(previous)+link['factor_id'].encode()).hexdigest()
                except (ValueError, TypeError, KeyError):
                    expected = None
                matches &= link.get('prefix_id') == expected
                for identity in link.get('observation_ids', ()):
                    links[identity] = link
            else:
                matches &= link.get('prefix_id') == previous
            valid &= matches
            previous = link.get('prefix_id')
        valid &= previous == history.get('final_prefix_id')
        chain_ok.append(bool(valid)); by_history.append(links)
    def normalized(logs):
        if not logs or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in logs):
            return False
        peak = max(logs)
        return abs(peak+math.log(math.fsum(math.exp(value-peak) for value in logs))) <= 1e-8
    for index, item in enumerate(atomic):
        identity = item.get('observation_id')
        raw, provenance = captured.get(identity), item.get('prefix_source') or {}
        payload = {} if raw is None else json.loads(raw['payload_json'])
        original = raw is not None and (
            item.get('source_stamp') == raw['stamp'] and
            item.get('source_command') == json.loads(raw['command_json']) and
            provenance.get('event_sequence') == raw['sequence'] and
            provenance.get('payload_hash') == raw['payload_hash'] and
            provenance.get('acquisition_frame_index') == payload.get('frame_index') and
            provenance.get('prefix_index') == index and
            provenance.get('source_clock') == 'original acquisition' and
            provenance.get('comparison_clock') == 'original acquisition' and
            provenance.get('terminal_command_substituted') is False)
        prefixes = provenance.get('prefix_sources', ())
        prefix_matches = len(prefixes) == len(histories) == len(likelihoods) and bool(histories)
        for parent, prefix in enumerate(prefixes):
            link = by_history[parent].get(identity) if parent < len(by_history) else None
            likelihood = likelihoods[parent].get(identity) if parent < len(likelihoods) else None
            prefix_matches &= link is not None and likelihood is not None and (
                prefix.get('prefix_id') == link['prefix_id'] and
                prefix.get('previous_prefix_id') == link['previous_prefix_id'] and
                prefix.get('common_start_state_id') == likelihood['common_state_id'] and
                prefix.get('acquisition_state_id') == likelihood['acquisition_state_id'] and
                prefix.get('actual_ledger_prefix') == likelihood['actual_command_ledger'])
        marginal = item.get('body_marginal') or {}
        branches = marginal.get('branches', ())
        parameters = (marginal.get('frozen_factor_sha256') == item.get('factor_id') and
            marginal.get('full_joint_parameters_retained_in_immutable_factor') is True and
            marginal.get('repeated_full_matrix_JSON_expansion') is False and bool(branches))
        for branch in branches:
            parameters &= (isinstance(branch.get('prior_parameters_sha256'), str) and
                len(branch['prior_parameters_sha256']) == 64 and
                all(isinstance(child.get('parameters_sha256'), str) and len(child['parameters_sha256']) == 64
                    for child in branch.get('children', ())))
        normalized_weights = normalized([branch.get('log_weight') for branch in branches]) and normalized(
            [child.get('log_weight') for branch in branches for child in branch.get('children', ())])
        source_ok.append(bool(original)); chain_ok.append(bool(prefix_matches))
        weights_ok.append(bool(normalized_weights)); parameter_ok.append(bool(parameters))
        sources.append(dict(observation_id=identity, source_stamp=item.get('source_stamp'),
            source_command=item.get('source_command'), factor_id=item.get('factor_id'),
            raw_source_matches=bool(original), prefix_chain_matches=bool(prefix_matches),
            normalized_full_body_mixture_log_weights=bool(normalized_weights),
            frozen_parameter_digests_present=bool(parameters)))
    checks = dict(
        time_resolved_schema_present=bool(historical),
        one_atomic_K_factor_per_effective_acquisition=len(atomic) == len(effective),
        atomic_K_observation_ids_preserve_acquisition_order=[item.get('observation_id') for item in atomic] == effective,
        historical_observation_ids_match=historical.get('observation_ids') == effective,
        historical_ratio_count_matches=historical.get('historical_ratio_count') == len(effective),
        composite_atomic_factor_ids_match=(not effective or product.get('atomic_factor_ids') == [item.get('factor_id') for item in atomic]),
        composite_factorization_required=(not effective or product.get('atomic_factor_count') == len(effective)
            and product.get('all_acquisition_commands_requeried_for_each_trial_K') is True
            and product.get('endpoint_body_ratio_used_as_K_factor') is False),
        every_atomic_original_stamp_command_sequence_hash_and_frame_matches=all(source_ok) and len(source_ok) == len(effective),
        every_original_prefix_chain_matches=all(chain_ok) and bool(histories),
        every_full_body_prior_posterior_log_mixture_is_normalized=all(weights_ok) and len(weights_ok) == len(effective),
        frozen_atomic_parameter_digests_are_recorded=all(parameter_ok) and len(parameter_ok) == len(effective))
    return dict(checks=checks, all_checks_passed=all(checks.values()),
        audit_scope='saved acquisition-specific K factor definition and original raw/prefix provenance; native arithmetic routing is covered separately by real-arm regression',
        atomic_factor_count=len(atomic), distinct_acquisition_commands=len({tuple(item.get('source_command', ())) for item in atomic}),
        source_rows=sources)


def _saved_law_summary(work, selected, actions, batch_id):
    estimates = list(rows(work/'estimation.jsonl'))
    if len(estimates) != 1:
        raise ValueError('One joint batch must publish exactly one estimation record.')
    record = estimates[0]
    joint = record.get('joint_batch', {})
    stiffness = record.get('stiffness', {})
    policy = stiffness.get('joint_batch', {})
    posterior = stiffness.get('posterior') or {}
    evidence = stiffness.get('bayes_factor_surprise') or {}
    expected_ids = [row['event_id'] for row in selected]
    effective = list(record.get('observation_ids', ()))
    raw_set = set(expected_ids)
    captured = {row['event_id']: row for row in selected}
    factor_rows = joint.get('likelihood_factors', ())
    source_matches = []
    factor_metadata = []
    for parent_index, factors in enumerate(factor_rows):
        for factor in factors:
            identity = factor['observation_id']
            original = captured.get(identity)
            actual = original is not None and (
                factor['acquisition_stamp'] == original['stamp'] and
                factor['acquisition_command'] == json.loads(original['command_json']) and
                factor['acquisition_event']['payload_hash'] == original['payload_hash'])
            source_matches.append(actual)
            factor_metadata.append(dict(parent_index=parent_index, observation_id=identity,
                effective=factor['effective'], native_component_count=factor['native_component_count'],
                acquisition_stamp=factor['acquisition_stamp'],
                common_factor_stamp=factor['common_factor_stamp'],
                source_clock_and_command_and_payload_hash_match=actual))
    shared = evidence.get('identical_keep_reset_evidence_reused', False)
    integrals = [evidence.get('keep_evidence')]
    if not shared:
        integrals.append(evidence.get('reset_evidence'))
    actual_laplace = [item for item in integrals if item is not None and not item.get('exact', False)]
    canonical = record.get('internal_projection', {})
    endpoint = selected[-1]
    time_indexed = _time_indexed_law_summary(record, selected)
    checks = dict(
        one_estimation_handoff=len(estimates) == 1,
        all_raw_ids_retained_in_order=joint.get('raw_event_ids') == expected_ids,
        stiffness_context_retains_all_raw_ids_in_order=policy.get('acquired_event_ids') == expected_ids,
        effective_ids_distinct=len(set(effective)) == len(effective),
        effective_ids_are_raw_acquisitions=set(effective) <= raw_set,
        body_and_stiffness_effective_ids_match=policy.get('effective_observation_ids') == effective,
        native_likelihood_clock_command_and_payload_hashes_match=bool(source_matches) and all(source_matches),
        no_intermediate_body_projections=joint.get('intermediate_projection_count') == 0,
        one_terminal_body_projection=joint.get('terminal_projection_count') == 1,
        actual_ledger_retained=joint.get('actual_ledger') == [step.record() for step in actions],
        endpoint_stamp_is_original=joint.get('endpoint_stamp') == endpoint['stamp'],
        endpoint_command_is_original=joint.get('endpoint_command') == json.loads(endpoint['command_json']),
        one_or_zero_K_factor=bool(posterior) and len(posterior.get('factor_ids', ())) == (1 if effective else 0),
        K_factor_ids_include_every_effective_observation=(not effective or
            stiffness.get('new_factor_observation_ids') == effective),
        continuous_K_not_replaced_by_Gaussian=posterior.get('global_moments') is None and
            not (posterior.get('bayes_factor_surprise') or {}).get('density_projection_applied', False),
        batch_id_matches=joint.get('batch_id') == batch_id and policy.get('batch_id') == batch_id,
        no_sequential_equivalence_claim=joint.get('old_per_event_recursion_equivalent') is False,
        no_sampling=joint.get('monte_carlo_used') is False and policy.get('monte_carlo_used') is False)
    checks['time_indexed_K_factors_are_required_and_match_original_acquisitions'] = time_indexed['all_checks_passed']
    return dict(checks=checks, all_law_checks_passed=all(checks.values()),
        effective_observation_ids=effective, effective_sensor_event_count=len(effective),
        likelihood_factors_per_parent=[len(items) for items in factor_rows],
        effective_likelihoods_per_parent=[sum(bool(item['effective']) for item in items) for items in factor_rows],
        native_components_per_factor=[[item['native_component_count'] for item in items] for items in factor_rows],
        exact_joint_branch_counts=[item['exact_branch_count'] for item in joint.get('parent_fusions', ())],
        body_intermediate_projection_count=joint.get('intermediate_projection_count'),
        body_terminal_projection_count=joint.get('terminal_projection_count'),
        body_surprise=record.get('bayes_factor_surprise'),
        K_new_factor_count=int(stiffness.get('new_factor_id') is not None),
        K_cumulative_factor_count=len(posterior.get('factor_ids', ())),
        K_retained_BFS_updates=(posterior.get('bayes_factor_surprise') or {}).get('retained_event_count'),
        K_hazard=evidence.get('rho0'), K_batch_policy=policy,
        time_indexed_stiffness=time_indexed,
        K_evidence_integrals_computed_inferred_from_records=len(actual_laplace),
        K_evidence_unique_density_queries=sum(item.get('diagnostics', {}).get('evaluations', 0) for item in actual_laplace),
        K_evidence_seconds=sum(item.get('diagnostics', {}).get('seconds', 0.) for item in actual_laplace),
        K_evidence_shared_keep_reset=shared,
        K_selected=stiffness.get('command_proposal', {}).get('selected_K') if stiffness.get('command_proposal') else None,
        K_proposal_status=stiffness.get('proposal_status'),
        candidate_measure=record.get('candidate_measure'),
        canonical_state_id=canonical.get('internal_state_id'),
        source_stamp=joint.get('source_stamp'), endpoint_stamp=joint.get('endpoint_stamp'),
        endpoint_command=joint.get('endpoint_command'),
        likelihood_source_metadata=factor_metadata,
        inference_seconds=record.get('async_estimation', {}).get('inference_seconds'),
        processing_seconds=record.get('async_estimation', {}).get('total_seconds'),
        complete_estimation_record_sha256=digest(record))


def _run_prefix(run, work, selected, actions, templates, timeout):
    from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore
    work.mkdir(parents=True, exist_ok=False)
    database = work/'observations.sqlite3'
    copied = clone_prefix(run, database, selected)
    context = mp.get_context('spawn')
    requests, replies = context.Queue(), context.Queue()
    worker = context.Process(target=worker_entry, args=(requests, replies,
        (run/'robot.urdf').read_text(), str(run/'robot.urdf'), templates, str(work)))
    progress, error, usage, complete, law = [], None, None, None, None
    started = time.perf_counter()
    worker.start()
    def receive():
        deadline = None if timeout is None else time.monotonic()+timeout
        while True:
            try:
                return replies.get(timeout=1.)
            except Empty:
                if not worker.is_alive():
                    raise RuntimeError('Joint batch worker exited before its reply: '+str(worker.exitcode))
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError('Explicit benchmark reply timeout exceeded.')
    try:
        with SensorEventStore(database) as store:
            batch = store.claim_batch('joint-benchmark', max_stamp=selected[-1]['stamp'],
                                      snapshot_id='joint-fixed-prefix')
            if batch.event_ids != tuple(row['event_id'] for row in selected):
                raise AssertionError('The joint batch omitted or reordered a raw acquisition.')
            requests.put(dict(kind='sensor_event_batch', batch=batch, database=str(database),
                              ledger_steps=actions, joint_body_batch=True))
            while complete is None:
                message = receive()
                if message.get('failed'):
                    raise RuntimeError(message.get('traceback', 'Worker failed.'))
                if message.get('kind') == 'inference_batch_progress':
                    progress.append(message['event_id'])
                elif message.get('kind') == 'inference_batch_completed':
                    complete = message
            finished = time.perf_counter()
            statuses = store.stats()
            fused_ids = [row['event_id'] for row in store.connection.execute(
                "SELECT event_id,result_metadata_json FROM events ORDER BY stamp,sequence")
                if (json.loads(row['result_metadata_json']) or {}).get('observation_fused')]
        law = _saved_law_summary(work, selected, actions, batch.batch_id)
        requests.put(None)
        while usage is None:
            message = receive()
            if message.get('kind') == 'benchmark_worker_usage':
                usage = message
        worker.join(timeout=10.)
        if worker.is_alive():
            raise RuntimeError('Joint batch worker did not exit after its sentinel.')
    except Exception as failure:
        error = dict(type=type(failure).__name__, reason=str(failure))
        finished = time.perf_counter()
        worker.join(timeout=2.)
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=10.)
        with SensorEventStore(database) as store:
            statuses = store.stats()
        fused_ids = None
        while usage is None:
            try:
                message = replies.get_nowait()
            except Empty:
                break
            if message.get('kind') == 'benchmark_worker_usage':
                usage = message
    finally:
        requests.cancel_join_thread(); replies.cancel_join_thread()
        requests.close(); replies.close()
    raw_ids = [row['event_id'] for row in selected]
    checks = dict(no_worker_error=error is None,
        all_raw_completed=statuses['counts'].get('completed', 0) == len(selected),
        checkpoint_ids_preserve_complete_raw_order=progress == raw_ids,
        fused_DB_ids_match_law=law is not None and fused_ids == law['effective_observation_ids'],
        all_joint_law_checks_passed=law is not None and law['all_law_checks_passed'])
    elapsed = finished-started
    duration = selected[-1]['stamp']-selected[0]['stamp']
    return dict(status='passed' if all(checks.values()) else 'failed', error=error, checks=checks,
        wall_seconds=elapsed, selected_acquisition_span_seconds=duration,
        wall_seconds_divided_by_acquisition_span=None if duration <= 0. else elapsed/duration,
        fresh_process_startup_included=True, OS_page_cache_controlled=False,
        raw_events_per_wall_second=None if error else len(selected)/elapsed,
        effective_sensor_events_per_wall_second=None if error or law is None else law['effective_sensor_event_count']/elapsed,
        raw_events_completed=statuses['counts'].get('completed', 0),
        raw_completed_by_kind=dict(Counter(row['kind'] for row in selected)) if checks['all_raw_completed'] else None,
        database_status=statuses, worker_exitcode=worker.exitcode, worker_usage=usage,
        isolated_working_DB_copy=copied, joint_posterior=law,
        batch_record=None if complete is None else complete['batch_record'],
        output_directory=str(work),
        timing_scope='Actual production joint batch worker initialization, IPC, one joint body posterior, one continuous K update/proposal, raw completion journals; excludes isolated DB-copy setup and final worker teardown.',
        failed_trial_is_performance_measurement=False if error else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=('marker', 'imu_marker'))
    parser.add_argument('--prefix-seconds', nargs='+', type=float, default=[.1, .3, 1.])
    parser.add_argument('--full-recorded', action='store_true',
        help='Select every available raw event through the original final acquisition frontier.')
    parser.add_argument('--reply-timeout-seconds', type=float, default=None,
        help='Optional explicit watchdog; default has no work-time limit and detects actual worker death.')
    parser.add_argument('--concurrent-workload-note', default=None,
        help='Record other active work; a declared concurrent trial is not an isolated-rate guarantee.')
    options = parser.parse_args()
    if any(not math.isfinite(t) or t < 0. for t in options.prefix_seconds):
        parser.error('Acquisition prefix durations must be finite and nonnegative.')
    case = options.case or options.run.name
    if case not in ('marker', 'imu_marker'):
        parser.error('Specify --case marker or imu_marker.')
    run, output = options.run.resolve(), options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import yaml
    templates = tuple(yaml.safe_load((PACKAGE/'config'/(case+'.yaml')).read_text())['observations'])
    before, binaries = source_hashes(), _binaries()
    original_db = sha(run/'observations.sqlite3')
    result = dict(schema='probtf_actual_joint_batch_benchmark_v1', original_run=str(run), case=case,
        approximation='one common-chart likelihood product of every acquired effective sensor; one terminal body moment/Bingham projection; product of acquisition-specific K body ratios with one batch evidence/hazard update',
        historical_per_event_numeric_equivalence_claimed=False, monte_carlo_used=False,
        original_database_open_mode='ro', source_hashes_before=before, native_binaries=binaries,
        original_DB_sha256_before=original_db, original_ledger_sha256=sha(run/'commands.jsonl'),
        original_URDF_sha256=sha(run/'robot.urdf'), results=[],
        concurrent_workload_note=options.concurrent_workload_note,
        isolated_processing_rate_guaranteed=False,
        machine_logical_cpu_count=os.cpu_count(), load_average_before=list(os.getloadavg()))
    try:
        durations = [None] if options.full_recorded else options.prefix_seconds
        for number, duration in enumerate(durations):
            selected, coverage = select_prefix(run, duration, None, options.full_recorded)
            actions = _ledger(run, selected)
            label = 'full_recorded' if duration is None else 'prefix_%02d_%gs' % (number, duration)
            trial = _run_prefix(run, output/label,
                selected, actions, templates, options.reply_timeout_seconds)
            trial.update(requested_acquisition_duration_seconds=duration, coverage=coverage,
                         actual_command_ledger_intervals=len(actions))
            result['results'].append(trial)
            (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            print(json.dumps(dict(requested_duration=duration,status=trial['status'],
                wall_seconds=trial['wall_seconds'],raw_completed=trial['raw_events_completed'],
                effective_sensor_events=None if trial['joint_posterior'] is None else trial['joint_posterior']['effective_sensor_event_count'],
                error=trial['error']), indent=2), flush=True)
            if trial['status'] != 'passed':
                break
    finally:
        after = source_hashes()
        result.update(source_unchanged_during_benchmark=before == after,
            source_files_changed_during_benchmark=[key for key in sorted(set(before)|set(after)) if before.get(key) != after.get(key)],
            actual_native_binaries_unchanged=_binaries() == binaries,
            load_average_after=list(os.getloadavg()),
            original_database_bytes_unchanged=sha(run/'observations.sqlite3') == original_db,
            benchmark_script_sha256=sha(Path(__file__)))
        result['timings_provisional_due_to_source_change'] = not result['source_unchanged_during_benchmark'] or not result['actual_native_binaries_unchanged']
        result['all_checks_passed'] = bool(result['results']) and len(result['results']) == (1 if options.full_recorded else len(options.prefix_seconds)) and all(
            trial['status'] == 'passed' for trial in result['results']) and not result['timings_provisional_due_to_source_change'] and result['original_database_bytes_unchanged']
        (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(summary=str(output/'summary.json'),all_checks_passed=result['all_checks_passed']), indent=2))
    if not result['all_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
