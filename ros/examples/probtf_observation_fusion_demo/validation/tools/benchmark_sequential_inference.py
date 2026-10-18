#!/usr/bin/env python3
"""Replay actual raw acquisitions/commands with the per-event body operator.

The source DB is opened read-only. Numeric arrays stay in the requested output
directory for backend comparison; no simulator truth or fitted-K substitute is
fed to inference. K evidence, IPC, archives and live contention are not timed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import RLock
import time
import traceback

import cv2
import numpy as np
import yaml

from probtf_observation_fusion_demo.async_state import ActionStep, RecordedCommandPath
from probtf_observation_fusion_demo.body_native import backend_record, pose_query
from probtf_observation_fusion_demo.immutable_body_cache import clear_reexpression_cache
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore


PACKAGE = Path(__file__).resolve().parents[2]


def source_hashes():
    return {str(path.relative_to(PACKAGE)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ('src', 'config', 'native')
            for path in (PACKAGE/directory).rglob('*')
            if path.is_file() and path.suffix in ('.py', '.yaml', '.cpp', '.h')}


def open_readonly_store(path):
    store = object.__new__(SensorEventStore)
    store._lock = RLock()
    store.connection = sqlite3.connect('file:'+str(path.resolve())+'?mode=ro', uri=True)
    store.connection.row_factory = sqlite3.Row
    store.connection.execute('PRAGMA query_only=ON')
    return store


def main(args):
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    templates = tuple(yaml.safe_load((PACKAGE/'config'/f'{args.case}.yaml').read_text())['observations'])
    enabled = {item['sensor'] for item in templates}
    store = open_readonly_store(args.input_run/'observations.sqlite3')
    selected, sensor_count = [], 0
    try:
        for row in store.connection.execute('SELECT * FROM events ORDER BY stamp, sequence'):
            initialize = not selected and row['kind'] == 'joint'
            if not initialize and {'imu': 'imu', 'image': 'marker'}.get(row['kind']) not in enabled:
                continue
            event = store._event(row)
            if selected and args.duration is not None and event.stamp-selected[0].stamp > args.duration:
                break
            selected.append(event)
            sensor_count += int(event.kind != 'joint')
            if args.count and sensor_count >= args.count:
                break
    finally:
        store.connection.close()
    if not selected or selected[0].kind != 'joint':
        raise ValueError('Actual replay requires its original initial command notification.')
    maximum_index = int(selected[-1].payload['frame_index'])
    ledger = []
    with (args.input_run/'commands.jsonl').open() as stream:
        for line in stream:
            step = json.loads(line)['step']
            if step['index'] >= maximum_index:
                break
            ledger.append(ActionStep(**step))
    if len(ledger) != maximum_index or [step.index for step in ledger] != list(range(maximum_index)):
        raise ValueError('Raw acquisitions require a complete actual command ledger.')
    setup_started = time.perf_counter()
    inference = MovingInference((args.input_run/'robot.urdf').read_text(), templates)
    setup_seconds = time.perf_counter()-setup_started
    clear_reexpression_cache()
    previous, previous_index, rows, arrays = None, None, [], {}
    error = None
    for number, event in enumerate(selected):
        try:
            index = int(event.payload['frame_index'])
            steps = () if previous is None else tuple(ledger[previous_index:index])
            transport_started = time.perf_counter()
            path = None if previous is None else RecordedCommandPath(previous, steps,
                body_ids=inference.body_ids, actions=inference.actions)
            transport_seconds = time.perf_counter()-transport_started
            wall_started, cpu_started = time.perf_counter(), time.process_time()
            result = inference.update_sensor_event(event, event_index=index, command_map=path,
                probability_unit='per_effective_sensor_event')
            elapsed, cpu = time.perf_counter()-wall_started, time.process_time()-cpu_started
            canonical = result.stages[-1].components[0].state
            prefix = f'event_{number:06d}_'
            arrays[prefix+'covariance'] = canonical.covariance
            arrays[prefix+'translation'] = np.asarray([edge.translation for edge in canonical.transforms])
            arrays[prefix+'quaternion'] = np.asarray([edge.rotation_wxyz for edge in canonical.transforms])
            arrays[prefix+'bingham'] = np.asarray([np.zeros((4, 4)) if edge.bingham is None else
                                                  edge.bingham for edge in canonical.transforms])
            posterior = result.projected_body_numerator
            if posterior is not None:
                for k, component in enumerate(posterior.posterior_components):
                    arrays[prefix+f'posterior_{k}_mean'] = component.distribution.mean
                    arrays[prefix+f'posterior_{k}_covariance'] = component.distribution.covariance
            for k, component in enumerate(result.stages[-1].goal_components):
                arrays[prefix+f'goal_{k}_covariance'] = component.state.covariance
                arrays[prefix+f'goal_{k}_translation'] = np.asarray([edge.translation for edge in component.state.transforms])
                arrays[prefix+f'goal_{k}_quaternion'] = np.asarray([edge.rotation_wxyz for edge in component.state.transforms])
            jumps = []
            if previous is not None:
                before = path.push(previous.stages[-1].components[0].state, event.command,
                                   inference.body_ids, inference.actions)
                for link, link_path in inference.paths['links'].items():
                    old, new = pose_query(before, link_path), pose_query(canonical, link_path)
                    cosine = min(1., abs(float(np.dot(old.rotation_wxyz, new.rotation_wxyz))))
                    jumps.append(dict(link=link, mean_displacement_m=float(np.linalg.norm(new.mean[:3]-old.mean[:3])),
                        mode_displacement_m=float(np.linalg.norm(new.mode[:3]-old.mode[:3])),
                        modal_angle_rad=float(2.*np.arccos(cosine))))
            row = dict(event_id=event.event_id, sequence=event.sequence, kind=event.kind,
                stamp=event.stamp, command=list(event.command), frame_index=index,
                original_payload_hash=event.payload_hash, actual_interval_indices=[step.index for step in steps],
                body_update_s=elapsed, body_update_cpu_s=cpu, transport_setup_s=transport_seconds,
                effective_observation_ids=result.records['observation_ids'],
                body_state_id=canonical.state_id, body_dimension=canonical.mean.size,
                bayes_factor_surprise=result.records.get('bayes_factor_surprise'),
                goal_log_weights=[component.log_weight for component in result.stages[-1].goal_components],
                same_time_command_update_displacements=jumps)
            rows.append(row)
            print(args.case, number, event.event_id, f'{elapsed:.6f}s', flush=True)
            previous, previous_index = result, index
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            break
    native = backend_record()
    per_sensor = {}
    for kind in ('imu', 'image'):
        durations = [row['body_update_s'] for row in rows if row['kind'] == kind]
        per_sensor[kind] = dict(count=len(durations), total_s=sum(durations),
            median_s=float(np.median(durations)) if durations else None,
            p95_s=float(np.percentile(durations, 95)) if durations else None)
    summary = dict(schema='sequential_actual_raw_backend_benchmark_v1', case=args.case,
        input_run=str(args.input_run), backend=native, original_raw_preserved=True,
        raw_database_read_only=True, acquisition_span_s=selected[-1].stamp-selected[0].stamp,
        selected_sensor_count=sensor_count, completed_event_count=len(rows),
        initialization_s=setup_seconds, per_sensor=per_sensor,
        body_processing_total_s=sum(row['body_update_s']+row['transport_setup_s'] for row in rows),
        error=error, source_unchanged=hashes == source_hashes(), source_sha256=hashes,
        timing_excludes=['imports', 'initial raw DB read', 'K evidence/representative', 'IPC',
                        'archives', 'extra post-update link displacement diagnostics', 'live contention'],
        finite_results_used_without_additional_quality_thresholds=True,
        precision_or_reaching_success_claimed=False, events=rows)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=True)+'\n')
    np.savez_compressed(args.output/'numeric_laws.npz', **arrays)
    if error:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', type=Path, required=True)
    parser.add_argument('--case', choices=('marker', 'imu_marker'), required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
