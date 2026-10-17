#!/usr/bin/env python3
"""Stream a recorded motion run into an evidence-only evaluation summary.

Large raw sensor/image/estimation payloads are never loaded. Missing telemetry
produces null, rather than a reconstructed success or an assumed zero. Distance
residence is time weighted using linear interpolation between physical samples;
this is a sampled-path diagnostic, not continuous collision certification.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def vector(value):
    if not isinstance(value, (list, tuple)) or not value:
        return None
    result = [number(item) for item in value]
    return result if all(item is not None for item in result) else None


def difference(left, right):
    left, right = vector(left), vector(right)
    if left is None or right is None or len(left) != len(right):
        return None
    return max(abs(a-b) for a, b in zip(left, right))


def norm(value):
    value = vector(value)
    return None if value is None else math.sqrt(sum(item*item for item in value))


class Stats:
    def __init__(self):
        self.count = 0
        self.mean = self.m2 = 0.
        self.minimum = self.maximum = None

    def add(self, value):
        value = number(value)
        if value is None:
            return
        self.count += 1
        delta = value-self.mean
        self.mean += delta/self.count
        self.m2 += delta*(value-self.mean)
        self.minimum = value if self.minimum is None else min(value, self.minimum)
        self.maximum = value if self.maximum is None else max(value, self.maximum)

    def record(self):
        if not self.count:
            return None
        return dict(count=self.count, mean=self.mean, minimum=self.minimum,
                    maximum=self.maximum, population_std=math.sqrt(max(0., self.m2/self.count)))


class Lines:
    def __init__(self, path, warnings):
        self.path, self.warnings = Path(path), warnings
        self.rows = self.invalid = 0
        self.digest = None

    def __iter__(self):
        if not self.path.exists():
            return
        digest = hashlib.sha256()
        with self.path.open('rb') as stream:
            for line_number, raw in enumerate(stream, 1):
                digest.update(raw)
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                    if not isinstance(row, dict):
                        raise ValueError('row is not an object')
                except (ValueError, UnicodeError) as error:
                    self.invalid += 1
                    if self.invalid <= 3:
                        self.warnings.append('%s:%d: %s' % (self.path.name, line_number, error))
                    continue
                self.rows += 1
                yield row
        self.digest = digest.hexdigest()

    def record(self):
        return dict(path=str(self.path.resolve()), exists=self.path.exists(), rows=self.rows,
                    invalid_rows=self.invalid, sha256=self.digest)


class Trace:
    """Bounded deterministic decimation; extrema and final point stay separate."""
    def __init__(self, limit):
        self.limit, self.stride, self.count, self.points = limit, 1, 0, []
        self.last = None

    def add(self, point):
        index = self.count
        self.count += 1
        self.last = point
        if index % self.stride == 0:
            self.points.append(point)
        if len(self.points) > self.limit:
            self.points = self.points[::2]
            self.stride *= 2

    def record(self):
        points = list(self.points)
        if self.last is not None and (not points or points[-1] != self.last):
            if len(points) >= self.limit:
                points[-1] = self.last
            else:
                points.append(self.last)
        return dict(raw_samples=self.count, retained_samples=len(points),
                    deterministic_index_stride=self.stride, final_point_added=True, samples=points)


def mapping(value):
    return value if isinstance(value, dict) else {}


def first(*values):
    return next((value for value in values if value is not None), None)


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_manifest(path, warnings):
    if not path.exists():
        warnings.append('motion_manifest.json is missing; run mode/settings are not assumed.')
        return {}
    try:
        return mapping(json.loads(path.read_text()))
    except (OSError, ValueError) as error:
        warnings.append('motion_manifest.json could not be read: '+str(error))
        return {}


def configuration(manifest, options):
    gains = first(manifest.get('running_gradient_config'), manifest.get('running_gradient_options'),
                  manifest.get('running_gradient_gains'), {})
    gains = mapping(gains)
    eta = first(options.eta, number(gains.get('eta')), number(manifest.get('running_gradient_eta')))
    gamma = first(options.gamma, number(gains.get('gamma')), number(manifest.get('running_gradient_gamma')))
    threshold = first(options.near_threshold, number(manifest.get('reach_tolerance_m')),
                      number(manifest.get('reach_tolerance')))
    threshold_source = 'explicit --near-threshold' if options.near_threshold is not None else 'run manifest'
    if threshold is None:
        # This is explicitly labelled current config, never a frozen old-run setting.
        path = Path(__file__).resolve().parents[1]/'config/switch_press_scene.yaml'
        try:
            for line in path.read_text().splitlines():
                if line.startswith('reach_tolerance:'):
                    threshold = number(line.split(':', 1)[1].split('#', 1)[0].strip())
                    threshold_source = 'current repository scene config (not an archived run snapshot): '+str(path)
                    break
        except OSError:
            pass
    return dict(eta=eta, gamma=gamma,
                gain_source=('explicit CLI override' if options.eta is not None or options.gamma is not None
                             else 'run manifest' if eta is not None or gamma is not None else None),
                near_threshold_m=threshold,
                near_threshold_source=threshold_source if threshold is not None else None)


def physical_summary(lines, threshold, samples):
    trace, errors, intervals = Trace(samples), Stats(), Stats()
    previous = first_point = last_point = minimum_point = None
    residence = covered = 0.
    reached = Counter()
    modes = Counter()
    motion_steps = 0
    for row in lines:
        stamp = number(first(row.get('time'), row.get('simulated_elapsed_s'), row.get('stamp')))
        error = number(first(row.get('physical_error_m'), row.get('tip_goal_distance_m')))
        modes[str(row.get('action_mode', 'unrecorded'))] += 1
        reached[str(row.get('reached'))] += 1
        if stamp is None or error is None:
            previous = None
            continue
        point = dict(time_s=stamp, physical_distance_m=error)
        errors.add(error)
        trace.add(point)
        if first_point is None:
            first_point = point
        last_point = point
        if minimum_point is None or error < minimum_point['physical_distance_m']:
            minimum_point = point
        if previous is not None and stamp > previous['time_s']:
            dt = stamp-previous['time_s']
            intervals.add(dt)
            covered += dt
            if threshold is not None:
                before = previous['physical_distance_m']
                if before <= threshold and error <= threshold:
                    residence += dt
                elif (before <= threshold) != (error <= threshold):
                    inside_fraction = abs(threshold-(before if before <= threshold else error))/abs(error-before)
                    residence += dt*inside_fraction
            motion_steps += 1
        previous = point
    duration = None if first_point is None or last_point is None else last_point['time_s']-first_point['time_s']
    return dict(distance_m=errors.record(), initial=first_point, minimum=minimum_point, final=last_point,
                duration_s=duration, actual_simulation_sample_hz=(motion_steps/duration if duration else None),
                physical_interval_s=intervals.record(), distance_coverage_s=covered,
                near_threshold_m=threshold,
                near_residence_s=residence if threshold is not None and covered > 0. else None,
                near_residence_fraction=(residence/covered if threshold is not None and covered > 0. else None),
                near_residence_method='linear distance interpolation between consecutive valid physical samples',
                reached_samples=dict(reached), action_mode_samples=dict(modes), distance_trace=trace.record())


def coefficients(gamma, dt):
    if gamma == 0.:
        return 1., dt, .5*dt*dt
    x = gamma*dt
    if abs(x) < 1e-3:
        return (math.exp(-x), dt*(1.-x/2.+x*x/6.-x**3/24.+x**4/120.),
                dt*dt*(.5-x/6.+x*x/24.-x**3/120.+x**4/720.))
    a = -math.expm1(-x)/gamma
    return math.exp(-x), a, (dt-a)/gamma


class KChanges:
    def __init__(self):
        self.count = self.changes = 0
        self.initial = self.final = self.minimum = self.maximum = None
        self.maximum_step = self.maximum_log_step = None
        self.density_ids = set()

    def add(self, value, density_id=None):
        value = vector(value)
        if value is None or any(item <= 0. for item in value):
            return
        if self.final is not None and len(value) != len(self.final):
            return
        self.count += 1
        if density_id is not None:
            self.density_ids.add(str(density_id))
        if self.initial is None:
            self.initial, self.minimum, self.maximum = list(value), list(value), list(value)
        else:
            delta = max(abs(a-b) for a, b in zip(value, self.final))
            log_delta = max(abs(math.log(a)-math.log(b)) for a, b in zip(value, self.final))
            self.changes += int(delta > 0.)
            self.maximum_step = delta if self.maximum_step is None else max(delta, self.maximum_step)
            self.maximum_log_step = log_delta if self.maximum_log_step is None else max(log_delta, self.maximum_log_step)
            self.minimum = [min(a, b) for a, b in zip(value, self.minimum)]
            self.maximum = [max(a, b) for a, b in zip(value, self.maximum)]
        self.final = list(value)

    def record(self):
        return None if not self.count else dict(samples=self.count, changed_samples=self.changes,
            distinct_density_ids=len(self.density_ids), initial=self.initial, final=self.final,
            per_joint_minimum=self.minimum, per_joint_maximum=self.maximum,
            maximum_absolute_step=self.maximum_step, maximum_absolute_log_step=self.maximum_log_step)


def command_summary(lines, configuration):
    join_position, join_velocity, equation_position, equation_velocity = Stats(), Stats(), Stats(), Stats()
    age, distance, lag, speed, acceleration = (Stats() for _ in range(5))
    statuses, policies, failures = Counter(), Counter(), Counter()
    previous_command = previous_velocity = None
    intervals, equation_intervals, held_intervals = 0, 0, 0
    held_duration = 0.
    no_goal_duration = image_missing_duration = None
    no_goal_samples = image_missing_samples = 0
    image_missing_hold_motion_duration = None
    image_missing_hold_motion_samples = 0
    no_goal_motion = no_goal_motion_duration = None
    no_goal_motion_samples = 0
    selected_k = KChanges()
    for row in lines:
        step = mapping(row.get('step'))
        execution = mapping(first(row.get('running_gradient_execution'), row.get('reaching_execution'),
                                  row.get('running_gradient')))
        behavior = mapping(row.get('behavior'))
        command = vector(first(step.get('end_command'), execution.get('command'), row.get('command')))
        before = vector(first(execution.get('position_before'), execution.get('command_before'),
                              step.get('start_command')))
        velocity = vector(first(execution.get('velocity'), row.get('velocity')))
        before_velocity = vector(first(execution.get('velocity_before'), execution.get('initial_velocity')))
        dt = number(first(execution.get('dt'),
                          (step['end_stamp']-step['start_stamp'] if isinstance(step.get('end_stamp'), (float, int))
                           and isinstance(step.get('start_stamp'), (float, int)) else None)))
        join_position.add(difference(before, previous_command))
        if before_velocity is not None:
            join_velocity.add(difference(before_velocity, previous_velocity))
        intervals += 1
        status = str(execution.get('status', 'unrecorded'))
        statuses[status] += 1
        policies[str(first(behavior.get('mode'), row.get('reaching_phase'), 'unrecorded'))] += 1
        if execution.get('stop_requested') or execution.get('halted'):
            failures[str(execution.get('reason', status))] += 1
        packet = mapping(execution.get('gradient_packet'))
        gradient = vector(first(execution.get('gradient'), packet.get('gradient')))
        age.add(execution.get('gradient_source_age_s'))
        distance.add(execution.get('gradient_source_command_distance_l2'))
        lag.add(execution.get('gradient_snapshot_update_lag'))
        speed.add(norm(velocity))
        acceleration.add(norm(execution.get('acceleration')))
        selected_k.add(first(step.get('selected_K'), execution.get('selected_K'), packet.get('selected_K')),
                       first(step.get('density_id'), execution.get('parameter_density_id'), packet.get('parameter_density_id')))
        eta, gamma = number(first(execution.get('eta'), configuration['eta'])), number(first(execution.get('gamma'), configuration['gamma']))
        if before_velocity is None:
            # This is recorded previous interval state, not an assumed initial zero.
            before_velocity = previous_velocity
        if (status == 'tracking' and gradient is not None and before is not None and before_velocity is not None
                and command is not None and velocity is not None and dt is not None and dt >= 0.
                and eta is not None and gamma is not None
                and len({len(value) for value in (before, before_velocity, command, velocity, gradient)}) == 1):
            try:
                decay, a_coeff, b_coeff = coefficients(gamma, dt)
                expected_command = [u+a_coeff*v-eta*b_coeff*g for u, v, g in zip(before, before_velocity, gradient)]
                expected_velocity = [decay*v-eta*a_coeff*g for v, g in zip(before_velocity, gradient)]
                equation_position.add(difference(command, expected_command))
                equation_velocity.add(difference(velocity, expected_velocity))
                equation_intervals += 1
            except (ValueError, OverflowError):
                pass
        if status == 'tracking' and dt is not None:
            held_intervals += 1
            held_duration += dt
        goal_available = first(execution.get('goal_available'), behavior.get('goal_available'))
        if goal_available is False and dt is not None:
            no_goal_samples += 1
            no_goal_duration = (no_goal_duration or 0.)+dt
            moving = difference(command, before)
            if moving is not None:
                no_goal_motion_samples += 1
                no_goal_motion_duration = (no_goal_motion_duration or 0.)+(dt if moving > 0. else 0.)
                no_goal_motion = bool(no_goal_motion) or moving > 0.
        marker_detected = first(behavior.get('marker_detected'), execution.get('marker_detected'), row.get('marker_detected'))
        if marker_detected is False and dt is not None:
            image_missing_samples += 1
            image_missing_duration = (image_missing_duration or 0.)+dt
            if status == 'tracking' and difference(command, before) is not None:
                image_missing_hold_motion_samples += 1
                image_missing_hold_motion_duration = (image_missing_hold_motion_duration or 0.)+(dt if difference(command, before) > 0. else 0.)
        previous_command, previous_velocity = command, velocity
    return dict(intervals=intervals, status_counts=dict(statuses), policy_counts=dict(policies),
        command_position_boundary_residual_inf_rad=join_position.record(),
        command_velocity_boundary_residual_inf_rad_s=join_velocity.record(),
        exact_hold_equation_position_residual_inf_rad=equation_position.record(),
        exact_hold_equation_velocity_residual_inf_rad_s=equation_velocity.record(),
        exact_hold_equation_verified_intervals=equation_intervals,
        command_speed_norm_rad_s=speed.record(), command_acceleration_norm_rad_s2=acceleration.record(),
        held_gradient_tracking_intervals=held_intervals, held_gradient_tracking_duration_s=held_duration,
        gradient_source_age_s=age.record(), gradient_source_command_distance_rad=distance.record(),
        gradient_snapshot_update_lag=lag.record(), selected_K_on_actual_intervals=selected_k.record(),
        fail_safe_tick_reasons=dict(failures),
        no_goal_telemetry_samples=no_goal_samples, no_goal_duration_s=no_goal_duration,
        no_goal_motion_detected=no_goal_motion, no_goal_motion_covered_samples=no_goal_motion_samples,
        no_goal_motion_duration_s=no_goal_motion_duration,
        raw_image_missing_telemetry_samples=image_missing_samples, raw_image_missing_duration_s=image_missing_duration,
        raw_image_missing_retained_gradient_motion_samples=image_missing_hold_motion_samples,
        raw_image_missing_retained_gradient_motion_duration_s=image_missing_hold_motion_duration,
        continuity_note='Equation residual, interval-state boundary continuity and packet-adoption invariance are separate checks; no unrecorded initial velocity is assumed.')


def adoption_summary(lines):
    counts, reasons = Counter(), Counter()
    age, displacement, lag, position_invariance, velocity_invariance = (Stats() for _ in range(5))
    selected_k = KChanges()
    for row in lines:
        counts['accepted' if row.get('accepted') is True else 'rejected'] += 1
        reasons[str(row.get('reason'))] += 1
        age.add(row.get('gradient_source_age_s'))
        displacement.add(row.get('gradient_source_command_distance_l2'))
        lag.add(row.get('gradient_snapshot_update_lag'))
        before = mapping(row.get('state_before'))
        after = mapping(row.get('state_after'))
        position_invariance.add(difference(first(row.get('position_before'), row.get('position'), before.get('position'), before.get('command')),
                                          first(row.get('position_after'), row.get('after_position'), after.get('position'), after.get('command'))))
        velocity_invariance.add(difference(first(row.get('velocity_before'), row.get('velocity'), before.get('velocity')),
                                          first(row.get('velocity_after'), row.get('after_velocity'), after.get('velocity'))))
        if row.get('accepted'):
            packet = mapping(first(row.get('gradient_packet'), row.get('packet')))
            selected_k.add(packet.get('selected_K'), packet.get('parameter_density_id'))
    return dict(counts=dict(counts), rejection_or_acceptance_reason_counts=dict(reasons),
                age_at_adoption_s=age.record(), command_displacement_at_adoption_rad=displacement.record(),
                snapshot_update_lag_at_adoption=lag.record(),
                packet_adoption_position_change_inf_rad=position_invariance.record(),
                packet_adoption_velocity_change_inf_rad_s=velocity_invariance.record(),
                selected_K_at_accepted_adoptions=selected_k.record())


def control_summary(lines):
    modes, statuses, reasons = Counter(), Counter(), Counter()
    computation, worker = Stats(), Stats()
    selected_k = KChanges()
    for row in lines:
        modes[str(first(row.get('control_mode'), mapping(row.get('controller')).get('control_mode'), 'unrecorded'))] += 1
        evaluation = mapping(row.get('gradient_evaluation'))
        if evaluation:
            statuses[str(evaluation.get('status'))] += 1
            reasons[str(evaluation.get('reason'))] += 1
            computation.add(evaluation.get('computation_seconds'))
            selected_k.add(evaluation.get('selected_K'), evaluation.get('parameter_density_id'))
        worker.add(row.get('work_seconds'))
    return dict(mode_counts=dict(modes), gradient_status_counts=dict(statuses), gradient_reason_counts=dict(reasons),
                gradient_computation_seconds=computation.record(), total_worker_seconds=worker.record(),
                selected_K_at_gradient_evaluation=selected_k.record())


def database_summary(path, warnings):
    if not path.exists():
        return None
    output = dict(path=str(path.resolve()), sensor_status_counts={}, sensors={}, errors=[],
                  result_metadata_only=True, raw_payloads_loaded=False)
    try:
        connection = sqlite3.connect('file:'+str(path.resolve())+'?mode=ro', uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute('BEGIN')
        try:
            for row in connection.execute('SELECT kind,status,COUNT(*) n FROM events GROUP BY kind,status'):
                output['sensor_status_counts'].setdefault(row['kind'], {})[row['status']] = row['n']
            for row in connection.execute('SELECT kind,COUNT(*) n,MIN(stamp) first_stamp,MAX(stamp) last_stamp, '
                                           'SUM(attempts) attempts,MAX(attempts) maximum_attempts FROM events GROUP BY kind'):
                span = row['last_stamp']-row['first_stamp']
                output['sensors'][row['kind']] = dict(count=row['n'], first_acquisition_stamp=row['first_stamp'],
                    last_acquisition_stamp=row['last_stamp'], acquisition_span_s=span,
                    actual_simulation_acquisition_hz=(row['n']-1)/span if span > 0. else None,
                    attempts=row['attempts'], maximum_attempts=row['maximum_attempts'])
            processing, waiting = Stats(), Stats()
            fused = Counter()
            for row in connection.execute('SELECT kind,result_metadata_json,claimed_wall_stamp,completed_wall_stamp,ready_wall_stamp '
                                           'FROM events WHERE result_metadata_json IS NOT NULL'):
                metadata = mapping(json.loads(row['result_metadata_json']))
                fused[row['kind']+'_fused' if metadata.get('observation_fused') else row['kind']+'_not_fused'] += 1
                processing.add(metadata.get('inference_seconds'))
                if row['claimed_wall_stamp'] is not None and row['ready_wall_stamp'] is not None:
                    waiting.add(row['claimed_wall_stamp']-row['ready_wall_stamp'])
            output.update(total_events=sum(sum(value.values()) for value in output['sensor_status_counts'].values()),
                          observation_fusion_counts=dict(fused), inference_seconds=processing.record(),
                          ready_to_claim_wall_delay_s=waiting.record())
            for row in connection.execute('SELECT event_id,kind,status,error FROM events WHERE error IS NOT NULL LIMIT 20'):
                output['errors'].append(dict(row))
            output['error_events'] = connection.execute('SELECT COUNT(*) FROM events WHERE error IS NOT NULL').fetchone()[0]
            output['backlog_events'] = connection.execute("SELECT COUNT(*) FROM events WHERE status IN ('pending','processing','awaiting_payload')").fetchone()[0]
            output['failed_or_replay_events'] = connection.execute("SELECT COUNT(*) FROM events WHERE status IN ('failed','late_requires_replay')").fetchone()[0]
        finally:
            connection.close()
    except (sqlite3.Error, ValueError) as error:
        warnings.append('Observation DB summary failed: '+str(error))
        output['query_error'] = str(error)
    return output


def failure_summary(run, warnings):
    summaries = {}
    for name in ('execution_failure.jsonl', 'shutdown.jsonl', 'gradient_failures.jsonl',
                 'running_gradient_fail_safe.jsonl'):
        lines = Lines(run/name, warnings)
        examples = []
        reasons = Counter()
        for row in lines:
            reasons[str(first(row.get('reason'), row.get('status'), row.get('failed'), 'unrecorded'))] += 1
            if len(examples) < 3:
                examples.append(row)
        if lines.path.exists():
            summaries[name] = dict(file=lines.record(), reason_counts=dict(reasons), first_examples=examples)
    return summaries


def evaluate(options):
    run, warnings = Path(options.run_dir).resolve(), []
    manifest = load_manifest(run/'motion_manifest.json', warnings)
    config = configuration(manifest, options)
    logs = {name: Lines(run/name, warnings) for name in
            ('physical_trajectory.jsonl', 'commands.jsonl', 'gradient_adoptions.jsonl', 'control.jsonl')}
    physical = physical_summary(logs['physical_trajectory.jsonl'], config['near_threshold_m'], options.distance_samples)
    commands = command_summary(logs['commands.jsonl'], config)
    adoptions = adoption_summary(logs['gradient_adoptions.jsonl'])
    control = control_summary(logs['control.jsonl'])
    mode = first(manifest.get('control_mode'), next(iter(control['mode_counts']), None))
    if mode in (None, 'unrecorded'):
        mode = 'unknown; not inferred from current repository default'
    return dict(schema='probtf_running_gradient_evaluation_v1', run_directory=str(run), control_mode=mode,
        evaluation_only=True, modifies_raw_inputs=False, old_run_performance_not_relabelled=True,
        configuration=config, physical=physical, commands=commands, gradient_adoptions=adoptions,
        control_worker=control, observations=database_summary(run/'observations.sqlite3', warnings),
        manifest={key: manifest.get(key) for key in (
            'execution_mode', 'execution_architecture', 'finished', 'estimation_drained', 'duration',
            'target_physics_hz', 'target_joint_hz', 'target_imu_hz', 'target_image_hz',
            'actual_motion_hz', 'actual_camera_hz', 'actual_joint_hz', 'actual_imu_hz',
            'actual_control_hz', 'actual_estimate_hz', 'reached', 'final_physical_error_m',
            'failure', 'collision_checking_enabled', 'reach_condition',
            'implementation_sha256', 'implementation_unchanged_during_execution')},
        files={name: lines.record() for name, lines in logs.items()},
        estimation_log=dict(path=str(run/'estimation.jsonl'), exists=(run/'estimation.jsonl').exists(),
            scanned=False, reason='Use SQLite event result metadata; avoid loading giant candidate records.'),
        failure_logs=failure_summary(run, warnings), warnings=warnings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--near-threshold', type=float, default=None, help='Physical distance tolerance, metres.')
    parser.add_argument('--distance-samples', type=int, default=1200, help='Bounded distance trace summary size.')
    parser.add_argument('--eta', type=float, default=None, help='Explicit logged-run gain when absent from manifest.')
    parser.add_argument('--gamma', type=float, default=None, help='Explicit logged-run damping when absent from manifest.')
    options = parser.parse_args(argv)
    if not options.run_dir.is_dir():
        parser.error('run_dir must be an existing directory')
    for name in ('near_threshold', 'eta', 'gamma'):
        value = getattr(options, name)
        if value is not None and (not math.isfinite(value) or value < 0. or name != 'gamma' and value == 0.):
            parser.error(name+' must be finite and positive (gamma may be zero)')
    if options.distance_samples < 2:
        parser.error('distance-samples must be at least 2')
    result = evaluate(options)
    encoded = json.dumps(finite_json(result), indent=2, sort_keys=True, allow_nan=False)+'\n'
    if options.output is None:
        sys.stdout.write(encoded)
    else:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(encoded)
        print(str(options.output.resolve()))


if __name__ == '__main__':
    main()
