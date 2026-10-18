#!/usr/bin/env python3
"""Stream a recorded running-gradient experiment into small, reproducible traces.

This is an offline diagnostic reader. It never imports the estimator, changes a
candidate law, or feeds physical truth back into control. A packet counts as
executed only when its ID occurs in commands.jsonl, including when post-motion
adoptions have the same final simulation stamp. No causal ablation is implied.

Example (the historical critical requests can be selected explicitly):
  /usr/bin/python3 analyze_running_gradient_trace.py RUN OUTPUT \
      --critical-request marker=82 --critical-request imu_marker=163

The optional runtime/*.jsonl monitor did not exist in older recordings. Its
inclusive nested stage totals must not be added as independent wall times.
"""
import argparse
import bisect
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


READ_DIAGNOSTICS = {}


CONTROL_FIELDS = (
    'source_stamp', 'source_command', 'status', 'reason',
    'body_observation_source_stamp', 'goal_observation_source_stamp',
    'parameter_density_id', 'selected_K', 'objective_at_source',
    'selected_objective_at_source', 'log_task_factor', 'log_view_factor',
    'log_select_factor', 'gradient', 'task_gradient', 'view_gradient',
    'weighted_view_gradient', 'computation_seconds',
    'task_computation_seconds_including_view_probes',
    'view_probe_computation_seconds', 'snapshot_update_count',
)


def stream(path):
    diagnostic = dict(complete_rows_read=0, unfinished_tail_bytes=0,
        note='Only newline-terminated JSONL rows are included; a live writer may still be appending the final row.')
    READ_DIAGNOSTICS[str(path.resolve())] = diagnostic
    with path.open() as source:
        for line in source:
            if not line.endswith('\n'):
                diagnostic['unfinished_tail_bytes'] = len(line.encode('utf-8'))
                return
            if line.strip():
                row = json.loads(line)
                diagnostic['complete_rows_read'] += 1
                yield row


def norm(value):
    return None if value is None else float(np.linalg.norm(value))


def cosine(left, right):
    if left is None or right is None:
        return None
    scale = norm(left)*norm(right)
    return None if scale == 0. else float(np.dot(left, right)/scale)


def stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return dict(count=len(values), minimum=min(values), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=max(values), total=sum(values))


def pick(record, keys):
    return {key: record.get(key) for key in keys}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def source_stat(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def compact_control(row, origin):
    controller = row['controller']
    view = controller.get('visibility_factor') or {}
    record = pick(row, ('index', 'request_id', 'event_id', 'capture_index',
                        'work_seconds', 'artifact_prefix'))
    record.update(pick(controller, CONTROL_FIELDS))
    record['source_time'] = controller['source_stamp']-origin
    for kind in ('body', 'goal'):
        stamp = controller.get(kind+'_observation_source_stamp')
        record[kind+'_observation_time'] = None if stamp is None else stamp-origin
    record.update(task_norm=norm(controller.get('task_gradient')),
                  weighted_view_norm=norm(controller.get('weighted_view_gradient')),
                  selected_norm=norm(controller.get('gradient')),
                  task_view_cosine=cosine(controller.get('task_gradient'),
                                          controller.get('weighted_view_gradient')),
                  view_penalty=view.get('penalty'), view_weight=view.get('weight'),
                  predicted_marker_pixels=[mode.get('predicted_pixel')
                      for mode in view.get('per_mode_record', ())],
                  view_optical_angles=[mode.get('optical_axis_angle_rad')
                      for mode in view.get('per_mode_record', ())])
    record['branches'] = [pick(branch, ('weight', 'log_weight', 'log_position',
        'log_orientation', 'log_position_pdet', 'relative_position_mean',
        'relative_position_covariance')) for branch in controller.get('source_branches', ())]
    return record


def load_controls(directory, output, origin, use_cache):
    source = directory/'control.jsonl'
    cache = output/(directory.name+'_control_compact.json')
    identity = dict(source=source_stat(source), origin_stamp=origin, format_revision=1)
    if use_cache and cache.exists():
        previous = json.loads(cache.read_text())
        if previous.get('identity') == identity:
            return previous['records']
    records = [compact_control(row, origin) for row in stream(source)]
    # Cache only copied scalar/vector diagnostics, never a retained raw record.
    write_json(cache, dict(identity=identity, records=records))
    return records


def damped_step(position, velocity, gradient, dt, eta, gamma):
    """Independent form of the same interval-constant-gradient exact solution."""
    if gamma == 0.:
        decay, a, b = 1., dt, .5*dt*dt
    else:
        x = gamma*dt
        decay = math.exp(-x)
        if abs(x) < 1e-4:
            a = dt*(1.-x/2.+x*x/6.-x*x*x/24.+x**4/120.)
            b = dt*dt*(.5-x/6.+x*x/24.-x*x*x/120.+x**4/720.)
        else:
            a = -math.expm1(-x)/gamma
            b = (dt-a)/gamma
    return (np.asarray(position)+a*np.asarray(velocity)-eta*b*np.asarray(gradient),
            decay*np.asarray(velocity)-eta*a*np.asarray(gradient))


def command_trace(directory, origin):
    records, errors = [], []
    for row in stream(directory/'commands.jsonl'):
        step = row['step']
        execution = row.get('reaching_execution') or {}
        gradient = execution.get('gradient')
        if gradient is not None and all(key in execution for key in (
                'position_before', 'velocity_before', 'dt', 'eta', 'gamma')):
            position, velocity = damped_step(execution['position_before'],
                execution['velocity_before'], gradient, execution['dt'],
                execution['eta'], execution['gamma'])
            errors.append(dict(position_l2=norm(position-execution['command']),
                               velocity_l2=norm(velocity-execution['velocity'])))
        records.append(dict(time=step['end_stamp']-origin,
            start_time=step['start_stamp']-origin, stamp=step['end_stamp'],
            command=step['end_command'], velocity=row['velocity'], gradient=gradient,
            acceleration=execution.get('acceleration'),
            gradient_source_age_s=execution.get('gradient_source_age_s'),
            gradient_source_command_distance_l2=execution.get('gradient_source_command_distance_l2'),
            gradient_source_command_distance_inf=execution.get('gradient_source_command_distance_inf'),
            source_stamp=execution.get('source_stamp'),
            parameter_density_id=execution.get('parameter_density_id'),
            selected_K=execution.get('selected_K'), eta=execution.get('eta'),
            gamma=execution.get('gamma'), status=execution.get('status'),
            gradient_packet_id=execution.get('gradient_packet_id'),
            coordinator_timings_s=row.get('coordinator_timings_s')))
    return records, errors


def nearest_physical(records, stamps, stamp):
    index = bisect.bisect_left(stamps, stamp)
    if index == len(stamps):
        index -= 1
    elif index > 0 and abs(stamps[index-1]-stamp) < abs(stamps[index]-stamp):
        index -= 1
    return records[index]


def adoption_trace(directory, origin, controls, commands, physical):
    by_request = {row['request_id']: row for row in controls}
    executed = {row['gradient_packet_id'] for row in commands}
    stamps = [row['stamp'] for row in physical]
    records = []
    for row in stream(directory/'gradient_adoptions.jsonl'):
        packet = row.get('packet') or row.get('gradient_packet') or {}
        record = pick(row, ('accepted', 'adoption_stamp', 'position', 'velocity',
            'gradient_source_age_s', 'gradient_source_command_distance_l2',
            'gradient_source_command_distance_inf', 'request_id',
            'evaluation_status', 'evaluation_reason', 'state_before', 'state_after'))
        record.update(time=row['adoption_stamp']-origin,
            source_time=None if packet.get('source_stamp') is None else packet['source_stamp']-origin,
            gradient=packet.get('gradient'), selected_K=packet.get('selected_K'),
            parameter_density_id=packet.get('parameter_density_id'),
            packet_id=packet.get('packet_id'),
            used_by_physical_tick=packet.get('packet_id') is not None
                and packet.get('packet_id') in executed,
            control=by_request.get(row.get('request_id')),
            physical_near_adoption=nearest_physical(physical, stamps, row['adoption_stamp']))
        control = record['control']
        record['body_observation_age_at_adoption_s'] = (None if control is None
            or control['body_observation_time'] is None else
            record['time']-control['body_observation_time'])
        records.append(record)
    return records


def gradient_jumps(adoptions):
    rows = []
    for before, after in zip(adoptions, adoptions[1:]):
        if before['gradient'] is None or after['gradient'] is None:
            continue
        old_K, new_K = before['selected_K'], after['selected_K']
        rows.append(dict(time=after['time'], request_id=after['request_id'],
            previous_request_id=before['request_id'],
            parameter_density_changed=after['parameter_density_id']!=before['parameter_density_id'],
            K_bitwise_unchanged=old_K is not None and new_K is not None
                and np.array_equal(old_K, new_K),
            delta_logK_inf=None if old_K is None or new_K is None else
                float(np.max(np.abs(np.log(np.asarray(new_K)/old_K)))),
            delta_gradient_norm=norm(np.asarray(after['gradient'])-before['gradient']),
            gradient_norm=norm(after['gradient']),
            gradient_cosine_previous=cosine(after['gradient'], before['gradient']),
            body_observation_age_at_adoption_s=after['body_observation_age_at_adoption_s'],
            gradient_source_age_s=after['gradient_source_age_s'],
            physical_error_near_adoption=after['physical_near_adoption']['physical_error_m']))
    return rows


def estimation_trace(directory, origin):
    records = []
    path = directory/'estimation.jsonl'
    if not path.exists():
        return records
    for row in stream(path):
        asynchronous = row.get('async_estimation') or {}
        source = row.get('joint_batch_source') or {}
        stiffness = row.get('stiffness') or {}
        proposal = stiffness.get('command_proposal') or {}
        stamp = asynchronous.get('capture_stamp')
        records.append(dict(capture_time=None if stamp is None else stamp-origin,
            event_id=asynchronous.get('event_id'), capture_index=asynchronous.get('capture_index'),
            inference_seconds=asynchronous.get('inference_seconds'),
            total_seconds=asynchronous.get('total_seconds'),
            body_observation_time=None if source.get('last_effective_observation_stamp') is None
                else source['last_effective_observation_stamp']-origin,
            effective_event_count=len(source.get('effective_event_ids', ())),
            selected_K=proposal.get('selected_K'),
            density_id=(stiffness.get('posterior') or {}).get('density_id'),
            proposal_status=stiffness.get('proposal_status')))
    return records


def timeline_csv(path, physical, commands, adoptions):
    fields = ('time', 'physical_error_m', 'physical_speed_norm',
        'physical_acceleration_norm', 'command_speed_norm', 'command_acceleration_norm',
        'gradient_source_age_s', 'body_observation_age_s',
        'gradient_source_command_distance_l2', 'source_time', 'adoption_time',
        'request_id', 'J', 'V', 'task_gradient_norm', 'weighted_view_gradient_norm',
        'selected_gradient_norm', 'task_view_cosine', 'selected_K', 'command', 'command_velocity')
    times = [record['adoption_stamp'] for record in adoptions]
    command_times = [record['stamp'] for record in commands]
    with path.open('w') as output:
        writer = csv.DictWriter(output, fields, lineterminator='\n')
        writer.writeheader()
        for row in physical:
            record = dict(time=row['time'], physical_error_m=row['physical_error_m'],
                physical_speed_norm=norm(row['joint_velocities']),
                physical_acceleration_norm=norm(row['joint_accelerations']))
            index = bisect.bisect_right(command_times, row['stamp'])-1
            if index >= 0:
                command = commands[index]
                record.update(command_speed_norm=norm(command['velocity']),
                    command_acceleration_norm=norm(command['acceleration']),
                    gradient_source_age_s=command['gradient_source_age_s'],
                    gradient_source_command_distance_l2=command['gradient_source_command_distance_l2'],
                    command=command['command'], command_velocity=command['velocity'])
            index = bisect.bisect_right(times, row['stamp'])-1
            if index >= 0:
                adopted = adoptions[index]
                record.update(adoption_time=adopted['time'], source_time=adopted['source_time'],
                              request_id=adopted['request_id'])
                control = adopted['control']
                if control is not None:
                    record.update(body_observation_age_s=None if control['body_observation_time'] is None
                        else row['time']-control['body_observation_time'],
                        J=control.get('objective_at_source'), V=control.get('view_penalty'),
                        task_gradient_norm=control['task_norm'],
                        weighted_view_gradient_norm=control['weighted_view_norm'],
                        selected_gradient_norm=control['selected_norm'],
                        task_view_cosine=control['task_view_cosine'], selected_K=control['selected_K'])
            writer.writerow(record)


def runtime_stages(directory, output):
    """Retain failed/open intervals too; this telemetry never gates execution."""
    records, open_intervals, mismatches = [], [], []
    for path in sorted((directory/'runtime').glob('*_progress.jsonl')):
        stacks = defaultdict(list)
        for row in stream(path):
            key = (row['role'], row['pid'])
            stack = stacks[key]
            if row['event'] == 'stage_started':
                stack.append(dict(start=row, direct_children_s=0.))
            elif row['event'] == 'stage_finished':
                entry = stack.pop() if stack else None
                if entry is None or entry['start']['stage'] != row['stage']:
                    mismatches.append(dict(path=path.name, sequence=row['sequence'],
                        stage=row['stage'], issue='stage finish without matching stack entry'))
                duration = row['duration_s']
                child_time = 0. if entry is None else entry['direct_children_s']
                records.append(dict(role=row['role'], pid=row['pid'], stage=row['stage'],
                    context=row.get('context'), end_wall_time_s=row['wall_time_s'],
                    end_monotonic_s=row['monotonic_s'], duration_s=duration,
                    cpu_s=row.get('cpu_s'),
                    exclusive_duration_s=None if entry is None else duration-child_time,
                    outcome=row.get('outcome'), error=row.get('error')))
                if stack:
                    stack[-1]['direct_children_s'] += duration
        for stack in stacks.values():
            open_intervals.extend(entry['start'] for entry in stack)
    groups = defaultdict(list)
    for record in records:
        groups[(record['role'], record['stage'])].append(record)
    statistics = [dict(role=role, stage=stage, wall=stats(r['duration_s'] for r in rows),
        cpu=stats(r['cpu_s'] for r in rows),
        exclusive_wall=stats(r['exclusive_duration_s'] for r in rows),
        failed_count=sum(r['outcome']=='failed' for r in rows))
        for (role, stage), rows in sorted(groups.items())]
    result = dict(available=bool(records or open_intervals),
        note='Wall totals are inclusive; nested stages overlap. Exclusive wall subtracts direct child durations only. Open/failed intervals are preserved.',
        stages=statistics, longest_intervals=sorted(records,
            key=lambda row: row['duration_s'], reverse=True)[:30],
        open_intervals=open_intervals, pairing_diagnostics=mismatches)
    write_json(output/(directory.name+'_runtime_summary.json'), result)
    with (output/(directory.name+'_runtime_stages.csv')).open('w') as target:
        writer = csv.DictWriter(target, ('role', 'pid', 'stage', 'context',
            'end_wall_time_s', 'end_monotonic_s', 'duration_s', 'cpu_s',
            'exclusive_duration_s', 'outcome', 'error'), lineterminator='\n')
        writer.writeheader()
        for record in records:
            writer.writerow(dict(record, context=json.dumps(record['context'], sort_keys=True)))
    return result


def select_critical(accepted, jumps, requested):
    if requested is not None:
        return next((row for row in accepted if row['request_id']==requested), None), 'explicit request'
    candidates = [row for row in jumps if row['parameter_density_changed'] and row['time'] >= 5.]
    if not candidates:
        candidates = jumps
    if not candidates:
        return None, 'no pair of executed gradient packets'
    event = max(candidates, key=lambda row: row['delta_gradient_norm'])
    return next(row for row in accepted if row['request_id']==event['request_id']), \
        'largest recorded gradient change after a density update at >=5s, or largest change if none; diagnostic selection only'


def analyze_case(directory, output, requested=None, use_cache=True):
    physical = [pick(row, ('time', 'stamp', 'command', 'actual_joints',
        'joint_velocities', 'joint_accelerations', 'tip', 'rigid_tip',
        'physical_error_m', 'rigid_command_error_m', 'reached', 'action_mode'))
        for row in stream(directory/'physical_trajectory.jsonl')]
    if not physical:
        raise ValueError('No physical frames in '+str(directory))
    origin = physical[0]['stamp']-physical[0]['time']
    controls = load_controls(directory, output, origin, use_cache)
    commands, errors = command_trace(directory, origin)
    adoptions = adoption_trace(directory, origin, controls, commands, physical)
    accepted = [row for row in adoptions if row['accepted'] and row['used_by_physical_tick']]
    jumps = gradient_jumps(accepted)
    event, selection = select_critical(accepted, jumps, requested)
    manifest_path = directory/'motion_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    timeline_csv(output/(directory.name+'_timeline.csv'), physical, commands, accepted)
    runtime = runtime_stages(directory, output)
    minimum = min(physical, key=lambda row: row['physical_error_m'])
    summary = dict(case=directory.name, run_directory=str(directory.resolve()),
        scope='This recorded run only; source identity is reported from its manifest. No causal ablation or comparison with another source is implied.',
        origin_stamp=origin,
        implementation_unchanged_during_execution=manifest.get('implementation_unchanged_during_execution'),
        implementation_sha256=manifest.get('implementation_sha256'),
        physical_model_source=manifest.get('physical_model_source'),
        actual_motion_hz=manifest.get('actual_motion_hz'),
        collision_checking_enabled=manifest.get('collision_checking_enabled'),
        motion_duration_s=physical[-1]['time'], minimum_physical_error_m=minimum['physical_error_m'],
        minimum_physical_error_time_s=minimum['time'],
        final_physical_error_m=physical[-1]['physical_error_m'],
        physical_reached=any(row['reached'] for row in physical),
        used_gradient_packet_count=len(accepted), all_control_results_count=len(controls),
        accepted_packets_never_used_by_a_physical_tick=[row['request_id'] for row in adoptions
            if row['accepted'] and not row['used_by_physical_tick']],
        adoption_reset_position_or_velocity_count=sum(row['state_before']!=row['state_after'] for row in accepted),
        held_gradient_exact_integration_position_error_l2=stats(row['position_l2'] for row in errors),
        held_gradient_exact_integration_velocity_error_l2=stats(row['velocity_l2'] for row in errors),
        maximum_held_gradient_age_at_physics_tick_s=max(
            (row['gradient_source_age_s'] or 0. for row in commands), default=None),
        maximum_source_command_distance_at_physics_tick_l2=max(
            (row['gradient_source_command_distance_l2'] or 0. for row in commands), default=None),
        command_speed_norm=stats(norm(row['velocity']) for row in commands),
        realized_speed_norm=stats(norm(row['joint_velocities']) for row in physical),
        realized_acceleration_norm=stats(norm(row['joint_accelerations']) for row in physical),
        estimation=estimation_trace(directory, origin),
        largest_gradient_jumps=sorted(jumps,key=lambda row:row['delta_gradient_norm'],reverse=True)[:10],
        largest_within_same_K_gradient_jump=max((row for row in jumps if row['K_bitwise_unchanged']),
            key=lambda row:row['delta_gradient_norm'],default=None),
        critical_selection=selection, critical_request_id=None if event is None else event['request_id'],
        critical_adoption_time_s=None if event is None else event['time'],
        runtime_stage_summary=runtime)
    runtime_by_request = defaultdict(lambda: defaultdict(list))
    with (output/(directory.name+'_runtime_stages.csv')).open() as stage_source:
        for stage in csv.DictReader(stage_source):
            context = json.loads(stage['context']) or {}
            request_id = context.get('request_id')
            if stage['role'] == 'control' and request_id is not None:
                runtime_by_request[request_id][stage['stage']].append(float(stage['duration_s']))
    segments = []
    for row in accepted:
        control = row['control']
        if control is None:
            continue
        key = (row['parameter_density_id'], control['body_observation_time'])
        if not segments or segments[-1]['key'] != key:
            segments.append(dict(key=key, start_time=row['time'], end_time=row['time'],
                selected_K=row['selected_K'], body_observation_time=control['body_observation_time'],
                rows=[]))
        segments[-1]['rows'].append(row)
        segments[-1]['end_time'] = row['time']
    for segment in segments:
        segment.pop('key')
        rows = segment.pop('rows')
        segment['control_count'] = len(rows)
        segment['adoption_source_age_s'] = stats(row['gradient_source_age_s'] for row in rows)
        segment['selected_gradient_norm'] = stats(row['control']['selected_norm'] for row in rows)
        segment['weighted_view_to_task_norm_ratio'] = stats(
            None if not row['control']['task_norm'] or row['control']['weighted_view_norm'] is None else
            row['control']['weighted_view_norm']/row['control']['task_norm'] for row in rows)
        for field in ('work_seconds', 'computation_seconds', 'view_probe_computation_seconds'):
            segment[field] = stats(row['control'][field] for row in rows)
        stage_values = defaultdict(list)
        for row in rows:
            for name, durations in runtime_by_request[row['request_id']].items():
                stage_values[name].extend(durations)
        segment['control_runtime_stages'] = {name: stats(values)
            for name,values in sorted(stage_values.items())}
    summary['used_density_segments'] = segments
    if event is not None:
        index = accepted.index(event)
        selected = accepted[max(0,index-1):index+2]
        summary['critical_before_after'] = []
        for row in selected:
            control = row['control'] or {}
            summary['critical_before_after'].append(dict(request_id=row['request_id'],
                adoption_time=row['time'], source_time=row['source_time'],
                body_observation_age_at_adoption_s=row['body_observation_age_at_adoption_s'],
                gradient_source_age_s=row['gradient_source_age_s'],
                source_command_distance_l2=row['gradient_source_command_distance_l2'],
                selected_K=row['selected_K'], command_velocity_norm=norm(row['velocity']),
                physical_error_m=row['physical_near_adoption']['physical_error_m'],
                **pick(control, ('body_observation_time', 'source_command', 'objective_at_source',
                    'view_penalty', 'view_weight', 'task_norm', 'weighted_view_norm', 'selected_norm'))))
        # Read gains from the actual first interval that executed this packet.
        active = next(row for row in commands if row['gradient_packet_id']==event['packet_id'])
        gamma, eta = active['gamma'], active['eta']
        control = event['control'] or {}
        contributions = dict(damping=(-gamma*np.asarray(event['velocity'])).tolist(),
            total=(-gamma*np.asarray(event['velocity'])-eta*np.asarray(event['gradient'])).tolist())
        for name, key in (('task','task_gradient'),('view','weighted_view_gradient')):
            contributions[name] = (None if control.get(key) is None else
                                   (-eta*np.asarray(control[key])).tolist())
        summary['critical_instantaneous_command_acceleration_contributions'] = contributions
        if index:
            before = accepted[index-1]['control']
            summary['critical_source_command_change_l2'] = norm(
                np.asarray(control['source_command'])-before['source_command'])
    summary['read_diagnostics'] = {path:record for path,record in READ_DIAGNOSTICS.items()
        if Path(path).parent == directory.resolve() or directory.resolve() in Path(path).parents}
    write_json(output/(directory.name+'_audit.json'), dict(summary=summary,
        controls=controls, adoptions=adoptions, gradient_jumps=jumps))
    return summary


def make_plot(output, summaries):
    os.environ.setdefault('MPLCONFIGDIR', str(output/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, len(summaries), figsize=(6.5*len(summaries),11),
                             sharex='col', squeeze=False)
    for column, (case, summary) in enumerate(summaries.items()):
        with (output/(case+'_timeline.csv')).open() as source:
            rows = list(csv.DictReader(source))
        time = [float(row['time']) for row in rows]
        def values(key):
            return [float(row[key]) if row[key] not in ('','None') else np.nan for row in rows]
        axes[0,column].plot(time,values('physical_error_m'))
        axes[0,column].set_title(case)
        axes[0,column].set_ylabel('Physical tip error [m]')
        axes[1,column].plot(time,values('command_speed_norm'),label='command')
        axes[1,column].plot(time,values('physical_speed_norm'),label='physical')
        axes[1,column].set_ylabel('Joint speed norm [rad/s]')
        for key,label in (('task_gradient_norm','task'),
                          ('weighted_view_gradient_norm','weighted view'),
                          ('selected_gradient_norm','selected')):
            axes[2,column].semilogy(time,values(key),label=label)
        axes[2,column].set_ylabel('Gradient norm')
        axes[3,column].plot(time,values('body_observation_age_s'),label='observation age')
        axes[3,column].plot(time,values('gradient_source_age_s'),label='gradient age')
        axes[3,column].set_ylabel('Age [s]')
        axes[3,column].set_xlabel('Physical elapsed time [s]')
        for axis in axes[:,column]:
            if summary['critical_adoption_time_s'] is not None:
                axis.axvline(summary['critical_adoption_time_s'],color='black',linestyle='--',alpha=.6)
            axis.grid(alpha=.2)
        for row in (1,2,3):
            axes[row,column].legend()
    fig.suptitle('Recorded motion and executed gradient sources; source provenance is in summary.json')
    fig.tight_layout()
    fig.savefig(output/'timeline.png',dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory',type=Path)
    parser.add_argument('output_directory',type=Path)
    parser.add_argument('--cases',nargs='+',default=['marker','imu_marker'])
    parser.add_argument('--critical-request',action='append',default=[],metavar='CASE=ID')
    parser.add_argument('--no-cache',action='store_true')
    parser.add_argument('--skip-plot',action='store_true')
    parser.add_argument('--runtime-only',action='store_true',
                        help='Read monitor files without requiring completed physical/control logs.')
    arguments = parser.parse_args(argv)
    critical = dict((case,int(number)) for case,number in
                    (value.split('=',1) for value in arguments.critical_request))
    arguments.output_directory.mkdir(parents=True,exist_ok=True)
    summaries = {}
    for case in arguments.cases:
        directory = arguments.run_directory/case
        if arguments.runtime_only:
            summaries[case] = runtime_stages(directory,arguments.output_directory)
        else:
            summaries[case] = analyze_case(directory,arguments.output_directory,
                critical.get(case),not arguments.no_cache)
        print(case+': analysis saved',flush=True)
    write_json(arguments.output_directory/'summary.json',summaries)
    if not arguments.skip_plot and not arguments.runtime_only:
        make_plot(arguments.output_directory,summaries)
    return summaries


if __name__ == '__main__':
    main()
