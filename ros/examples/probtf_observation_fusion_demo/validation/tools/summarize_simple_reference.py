#!/usr/bin/env python3
"""Read-only audit of the minimal joint-law / Gaussian-K / finite-step loop.

Physical rows are used only for posthoc evaluation.  The model comparison
reuses saved auxiliary equilibrium angles and performs FK, never inference,
equilibrium fitting, command replay, true-K installation, or Monte Carlo.
"""

import argparse
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path
import sqlite3
import time
import xml.etree.ElementTree as ET

import numpy as np

from summarize_joint_rotation_runs import (
    REPO, all_keys, digest, hash_comparison, read_json, rows, selected, truth_flags,
)


def percentiles(values):
    return None if not values else dict(count=len(values),
        mean=float(np.mean(values)), total=float(np.sum(values)),
        median=float(np.median(values)), p95=float(np.percentile(values, 95)), maximum=float(max(values)))


class BodySurpriseAudit:
    """Inspect saved body-only change updates without changing their outcome."""
    def __init__(self):
        self.counts, self.checks, self.applied_by_kind = Counter(), Counter(), Counter()
        self.diagnostics = {name: [] for name in ('rho_error', 'prior_odds_error',
            'posterior_log_odds_residual', 'recorded_log_odds_residual_difference',
            'posterior_log_normalization_residual', 'posterior_change_probability_error',
            'transition_prior_log_normalization_residual', 'transition_prior_log_weight_residual')}
        self.examples = []
        self.posterior_change, self.log_bayes_factors = [], []
        self.strongest_event = None

    def observe(self, row, stiffness):
        surprise = row.get('body_surprise')
        if not isinstance(surprise, dict):
            self.counts['records_absent'] += 1
            return
        self.counts['records_present'] += 1
        enabled, applied = bool(surprise.get('enabled')), bool(surprise.get('applied'))
        self.counts['enabled'] += enabled
        self.counts['disabled'] += not enabled
        self.counts['applied'] += applied
        self.checks['applied_while_disabled'] += applied and not enabled
        self.checks['applied_without_fused_observation'] += applied and not row['observation_ids']
        self.checks['applied_to_unselected_observation'] += (
            applied and 'observation_selected' in row and not row['observation_selected'])
        if enabled:
            values = (surprise.get('change_probability'), surprise.get('prior_change_odds'))
            finite = all(value is not None and np.isfinite(value) for value in values)
            self.checks['enabled_rho_or_prior_odds_nonfinite'] += not finite
            if finite:
                self.diagnostics['rho_error'].append(abs(values[0]-.25))
                self.diagnostics['prior_odds_error'].append(abs(values[1]-1./3.))
        if not applied:
            return
        self.applied_by_kind[row['kind']] += 1
        self.checks['applied_observation_ID_mismatch'] += surprise.get('observation_ids') != row['observation_ids']
        self.checks['applied_internal_state_ID_mismatch'] += surprise.get('internal_state_id') != row['body_state_ids'][0]
        self.checks['applied_probability_unit_mismatch'] += surprise.get('probability_unit') != 'per_effective_sensor_event'
        log_keep = surprise.get('posterior_keep_log_probability')
        log_reset = surprise.get('posterior_change_log_probability')
        log_bf, rho = surprise.get('log_bayes_factor'), surprise.get('change_probability')
        values = (log_keep, log_reset, log_bf, rho, surprise.get('posterior_change_probability'))
        finite = all(value is not None and np.isfinite(value) for value in values) and 0. < rho < 1.
        self.checks['applied_odds_or_mass_nonfinite_or_undefined'] += not finite
        if finite:
            self.posterior_change.append(float(surprise['posterior_change_probability']))
            self.log_bayes_factors.append(float(log_bf))
            residual = log_reset-log_keep-(np.log(rho)-np.log1p(-rho)+log_bf)
            self.diagnostics['posterior_log_odds_residual'].append(float(residual))
            self.diagnostics['posterior_log_normalization_residual'].append(float(np.logaddexp(log_keep, log_reset)))
            self.diagnostics['posterior_change_probability_error'].append(float(
                surprise['posterior_change_probability']-np.exp(log_reset)))
            saved = surprise.get('joint_evidence_log_odds_consistency', {}).get('residual')
            self.checks['recorded_odds_residual_nonfinite_or_missing'] += saved is None or not np.isfinite(saved)
            if saved is not None and np.isfinite(saved):
                self.diagnostics['recorded_log_odds_residual_difference'].append(float(saved-residual))
        source_event = dict(event_id=row['event_id'], sequence=row['sequence'],
            original_stamp=row['acquisition_stamp'], original_command=row['acquisition_command'],
            canonical_state_id=row['body_state_ids'][0], rho=rho,
            prior_change_odds=surprise.get('prior_change_odds'),
            log_bayes_factor=log_bf, posterior_change_probability=surprise.get('posterior_change_probability'))
        if len(self.examples) < 3:
            self.examples.append(source_event)
        if finite and (self.strongest_event is None or
                source_event['posterior_change_probability'] > self.strongest_event['posterior_change_probability']):
            self.strongest_event = source_event
        if stiffness is None:
            self.checks['applied_without_K_update_record'] += 1
            return
        marginal = stiffness.get('body_marginal') or {}
        branches = marginal.get('branches', ())
        self.checks['applied_K_prior_branch_count_not_two'] += len(branches) != 2
        self.checks['applied_K_BFS_or_evidence_enabled'] += bool(
            stiffness.get('bayes_factor_surprise_enabled') or stiffness.get('evidence_computed'))
        self.checks['applied_K_canonical_numerator_ID_mismatch'] += (
            (stiffness.get('body_internal_approximation') or {}).get('internal_state_id')
                != row['body_state_ids'][0]
            or stiffness.get('body_posterior_state_ids') != row['body_state_ids'])
        if len(branches) == 2:
            self.checks['applied_K_prior_source_ID_mismatch'] += (
                [branch['source_state_id'] for branch in branches] != surprise.get('prior_source_state_ids'))
            self.checks['applied_K_prior_original_t_q_mismatch'] += any(
                branch['source_stamp'] != row['acquisition_stamp']
                or branch['source_command'] != row['acquisition_command'] for branch in branches)
            self.checks['applied_K_projected_numerator_reference_mismatch'] += (
                marginal.get('projected_numerator_source_state_id') != branches[0]['source_state_id'])
            self.checks['applied_K_canonical_numerator_component_mismatch'] += (
                len(branches[0]['components']) != 1 or len(branches[1]['components']) != 0
                or branches[0]['components'][0]['label'] != row['body_component_id'])
            if len(branches[0]['components']) == 1:
                child = branches[0]['components'][0]
                self.checks['applied_K_canonical_numerator_mass_mismatch'] += (
                    child['weight'] != 1. or child['log_weight'] != 0.)
            log_weights = [branch['log_weight'] for branch in branches]
            self.checks['applied_K_prior_weights_mismatch'] += [branch['weight'] for branch in branches] != [.75, .25]
            self.checks['applied_K_prior_log_weights_nonfinite'] += not np.all(np.isfinite(log_weights))
            if np.all(np.isfinite(log_weights)):
                self.diagnostics['transition_prior_log_normalization_residual'].append(float(np.logaddexp(*log_weights)))
                self.diagnostics['transition_prior_log_weight_residual'].append(float(
                    np.max(np.abs(np.asarray(log_weights)-np.log([.75, .25])))))

    def record(self):
        return dict(available=bool(self.counts['records_present']), counts=dict(self.counts),
            checks=dict(self.checks), applied_sensor_counts=dict(self.applied_by_kind),
            numerical_diagnostics={name: dict(count=len(values),
                max_absolute=float(np.max(np.abs(values))) if values else None)
                for name, values in self.diagnostics.items()},
            source_event_examples=self.examples,
            posterior_change_probability=percentiles(self.posterior_change),
            log_bayes_factor=percentiles(self.log_bayes_factors),
            highest_posterior_change_event=self.strongest_event,
            interpretation='Saved body change updates only; K surprise remains disabled. Finite residuals are diagnostics, not update acceptance or stopping conditions. Missing legacy records are not invented as disabled updates. The projected numerator reference ID identifies its actual Keep chart; canonical identity is checked separately.')


def source_coverage(stamps):
    if not stamps:
        return dict(count=0, original_time_range=None, maximum_gap_s=None)
    ordered = sorted(stamps)
    return dict(count=len(stamps), original_time_range=[ordered[0], ordered[-1]],
        maximum_gap_s=(max(np.diff(ordered)) if len(ordered)>1 else None))


def reaching_objective(diagnostics):
    """Identify the law's saved selection rule, including historical runs."""
    expression = diagnostics.get('objective')
    known = {
        'E_mu_c[||p_hand(B)-p_goal(E)||^2]': ('expected_squared_distance', True),
        '||E_mu_c[p_hand(B)]-E_mu_c[p_goal(E)]||^2': ('squared_mean_distance', False),
    }
    if expression in known:
        family, include_trace = known[expression]
        source = 'saved objective expression'
    else:
        include_trace = diagnostics.get('source_score', {}).get('difference_covariance_trace_in_objective')
        family = ('expected_squared_distance' if include_trace is True else
            'squared_mean_distance' if include_trace is False else 'unspecified')
        source = 'saved point-score declaration' if include_trace is not None else 'not recorded'
    return dict(expression=expression, family=family,
        covariance_trace_in_objective=include_trace, identification_source=source)


def declared_score(point_score, objective):
    include_trace = objective['covariance_trace_in_objective']
    if include_trace is None:
        return None
    difference = np.asarray(point_score['mean_difference'])
    return float(difference@difference+(np.trace(point_score['difference_covariance']) if include_trace else 0.))


def finite_plan_response(row, physical, stamps, by_stamp):
    """Compare one saved prediction with its actual, finite-duration response."""
    diagnostics = row['diagnostics']
    if not diagnostics['goal_observed']:
        return None
    start = by_stamp.get(row['adoption_stamp'])
    end_index = bisect_left(stamps, row['adoption_stamp']+row['duration'])
    if start is None or end_index == len(stamps):
        return None
    end = physical[end_index]
    before, after = diagnostics['source_score'], diagnostics['selected_score']
    objective = reaching_objective(diagnostics)
    declared_before, declared_after = (declared_score(item, objective) for item in (before, after))
    delta_predicted = np.asarray(after['hand_mean'])-before['hand_mean']
    delta_actual = np.asarray(end['tip'])-start['tip']
    before_squared = float(np.dot(before['mean_difference'], before['mean_difference']))
    after_squared = float(np.dot(after['mean_difference'], after['mean_difference']))
    denominator = np.linalg.norm(delta_predicted)*np.linalg.norm(delta_actual)
    toward_goal = -np.asarray(before['mean_difference'])
    goal_distance = np.linalg.norm(toward_goal)
    predicted_goal_projection = (None if goal_distance == 0. else
        float(np.dot(delta_predicted, toward_goal)/goal_distance))
    actual_goal_projection = (None if goal_distance == 0. else
        float(np.dot(delta_actual, toward_goal)/goal_distance))
    return dict(adoption_stamp=row['adoption_stamp'], evaluated_physical_stamp=end['stamp'],
        physical_sample_overshoot_s=end['stamp']-(row['adoption_stamp']+row['duration']),
        S_before=diagnostics['S_before'], S_after=diagnostics['S_after'],
        reaching_objective=objective,
        score_before_equation_residual=(None if declared_before is None else diagnostics['S_before']-declared_before),
        score_after_equation_residual=(None if declared_after is None else diagnostics['S_after']-declared_after),
        mean_squared_distance_before_m2=before_squared,
        mean_squared_distance_after_m2=after_squared,
        predicted_mean_squared_distance_change_m2=after_squared-before_squared,
        predicted_covariance_trace_before_m2=float(np.trace(before['difference_covariance'])),
        predicted_covariance_trace_after_m2=float(np.trace(after['difference_covariance'])),
        predicted_covariance_trace_change_m2=float(np.trace(after['difference_covariance'])
            -np.trace(before['difference_covariance'])),
        predicted_mean_distance_reduction_m=float(np.sqrt(before_squared)-np.sqrt(after_squared)),
        actual_physical_distance_reduction_m=start['physical_error_m']-end['physical_error_m'],
        predicted_hand_displacement_m=delta_predicted.tolist(),
        actual_hand_displacement_m=delta_actual.tolist(),
        predicted_hand_displacement_norm_m=float(np.linalg.norm(delta_predicted)),
        actual_hand_displacement_norm_m=float(np.linalg.norm(delta_actual)),
        predicted_hand_progress_along_estimated_goal_m=predicted_goal_projection,
        actual_hand_progress_along_estimated_goal_m=actual_goal_projection,
        predicted_hand_goal_alignment_cosine=(None if predicted_goal_projection is None or
            np.linalg.norm(delta_predicted) == 0. else float(predicted_goal_projection/np.linalg.norm(delta_predicted))),
        actual_hand_estimated_goal_alignment_cosine=(None if actual_goal_projection is None or
            np.linalg.norm(delta_actual) == 0. else float(actual_goal_projection/np.linalg.norm(delta_actual))),
        predicted_actual_hand_displacement_cosine=(None if denominator == 0.
            else float(np.dot(delta_predicted, delta_actual)/denominator)),
        selected_command_step_norm_rad=diagnostics['step_norm_rad'],
        auxiliary_active_joint_limits=diagnostics.get('selected_command_action', {}).get('active_joint_limits'))


def posthoc_physical_goal(first):
    """Read the unchanged world setting solely for posthoc scoring."""
    import yaml
    from scipy.spatial.transform import Rotation
    path = REPO/'ros/examples/probtf_observation_fusion_demo/config/switch_press_scene.yaml'
    expected = (first or {}).get('physics', {}).get('source', {}).get('scene_config_sha256')
    checksum = digest(path)
    if expected != checksum:
        return dict(available=False, reason='logged scene hash and current static world setting do not match',
            logged_scene_sha256=expected, current_scene_sha256=checksum)
    settings = yaml.safe_load(path.read_text())
    marker = settings['marker']
    goal = (np.asarray(marker['world_translation'])-settings['base_translation']
        +Rotation.from_euler('xyz', marker['world_rpy']).as_matrix()@marker['goal_offset'])
    return dict(available=True, source=str(path), scene_sha256=checksum,
        position_base_link=goal.tolist(), only_posthoc=True,
        reach_tolerance_m=settings['reach_tolerance'], physical_truth_fed_to_runtime=False)


def camera_detection_summary(previews):
    visible = [item['stamp'] for item in previews if item['visible']]
    intervals, missing_since, missing_count = [], None, 0
    for item in previews:
        if not item['visible']:
            missing_since = item['stamp'] if missing_since is None else missing_since
            missing_count += 1
        elif missing_since is not None:
            intervals.append(dict(first_missing_stamp=missing_since,
                recovered_stamp=item['stamp'], duration_s=item['stamp']-missing_since,
                missing_captures=missing_count))
            missing_since, missing_count = None, 0
    if missing_since is not None:
        intervals.append(dict(first_missing_stamp=missing_since, recovered_stamp=None,
            duration_s=previews[-1]['stamp']-missing_since, missing_captures=missing_count))
    return dict(captured_preview_count=len(previews), detected_count=len(visible),
        detected_source_coverage=source_coverage(visible),
        longest_detection_loss_intervals=sorted(intervals, key=lambda item: item['duration_s'], reverse=True)[:5],
        loss_interval_rule='From the first captured image with no detected marker to the first recovered capture; these are image timestamps, not exact continuous visibility boundaries.',
        detection_is_statistical_fusion=False)


def reconstruct(row):
    from probtf.spherical_law.first_order import BinghamTransform, JointTransformState
    covariance = np.asarray(row['body_covariance'])
    transforms = []
    for index, data in enumerate(row['local_transforms']):
        validated = BinghamTransform(data['edge_id'], np.asarray(data['translation']),
            bingham=None if data['bingham'] is None else np.asarray(data['bingham']),
            translation_covariance=covariance[6*index:6*index+3, 6*index:6*index+3],
            rotation_wxyz=np.asarray(data['rotation_wxyz']) if data['bingham'] is None else None)
        # Restore the saved immutable chart, as the native materializer does.
        # Re-eigendecomposing A can change the cached modal quaternion by
        # roundoff and would manufacture a different source hash here.
        edge = object.__new__(BinghamTransform)
        saved_rotation = np.array(data['rotation_wxyz'], dtype=float)
        saved_rotation.setflags(write=False)
        for key, value in dict(edge_id=validated.edge_id, translation=validated.translation,
                translation_covariance=validated.translation_covariance,
                bingham=validated.bingham, rotation_wxyz=saved_rotation).items():
            object.__setattr__(edge, key, value)
        transforms.append(edge)
    return JointTransformState(tuple(transforms), covariance,
        version=row['body_state_version'], stamp=row['acquisition_stamp'],
        command=tuple(row['acquisition_command']), component_id=row['body_component_id'],
        mean=np.asarray(row['body_chart_mean']))


def model_comparisons(sample_rows, model, physical):
    if not model:
        return dict(available=False, reason='simple_model.json is not present at this frontier')
    from deflecomp_core.robot.pinocchio_robot import RobotArm
    from probtf.geometry import quat_to_rotmat
    from probtf_observation_fusion_demo.body_native import point_query
    from scipy.spatial.transform import Rotation
    robot = RobotArm(model['urdf_path'], tip_link=model['hand_path'][-1][0], base_link=model['root_frame'])
    parents = {joint.find('child').get('link'): joint.find('parent').get('link')
        for joint in ET.parse(model['urdf_path']).getroot().findall('joint')}
    samples = []
    for row in sample_rows:
        try:
            state = reconstruct(row)
            proposal = row['stiffness']['command_proposal']
            theta = np.asarray(proposal['auxiliary_equilibrium'])
            robot._fk_update(theta)
            base = robot.data.oMf[robot.base_fid]
            pose = base.inverse()*robot.data.oMf[robot.tip_fid]
            hand = point_query(state, tuple(map(tuple, model['hand_path'])),
                query_id='posthoc:simple:hand', frame=model['root_frame'])
            actual = physical[row['frame_index']]
            angular = []
            by_id = {edge.edge_id: edge for edge in state.transforms}
            for name in model['body_edge_ids']:
                parent = robot.model.getFrameId(parents[name])
                child = robot.model.getFrameId(name)
                local = robot.data.oMf[parent].inverse()*robot.data.oMf[child]
                angular.append(float(np.linalg.norm(Rotation.from_matrix(
                    quat_to_rotmat(by_id[name].rotation_wxyz).T @ local.rotation).as_rotvec())))
            samples.append(dict(event_id=row['event_id'], original_stamp=row['acquisition_stamp'],
                original_sequence=row['sequence'],
                original_command=row['acquisition_command'], frame_index=row['frame_index'],
                saved_state_id=row['body_state_ids'][0], reconstructed_state_id=state.state_id,
                state_id_matches=state.state_id == row['body_state_ids'][0],
                selected_K=proposal['selected_K'], density_id=proposal['density_id'],
                original_t_q_matches_saved_proposal=(proposal['source_stamp'] == row['acquisition_stamp']
                    and proposal['source_command'] == row['acquisition_command']),
                physical_t_q_matches=(actual['stamp'] == row['acquisition_stamp']
                    and actual['command'] == row['acquisition_command']),
                model_hand=pose.translation.tolist(), estimated_hand_mode=hand.mode.tolist(),
                estimated_hand_mean=hand.mean.tolist(), physical_hand=actual['tip'],
                model_to_estimated_mode_m=float(np.linalg.norm(pose.translation-hand.mode)),
                model_to_estimated_mean_m=float(np.linalg.norm(pose.translation-hand.mean)),
                model_to_physical_m=float(np.linalg.norm(pose.translation-actual['tip'])),
                estimated_mean_to_physical_m=float(np.linalg.norm(hand.mean-actual['tip'])),
                estimated_mode_to_physical_m=float(np.linalg.norm(hand.mode-actual['tip'])),
                model_to_estimated_local_rotation_errors_rad=angular,
                local_rotation_RMS_rad=float(np.sqrt(np.mean(np.square(angular)))),
                local_rotation_max_rad=max(angular), equilibrium_resolved=False,
                physical_truth_used_only_for_posthoc_scoring=True))
        except Exception as error:
            samples.append(dict(event_id=row['event_id'], available=False,
                reason=type(error).__name__+': '+str(error)))
    return dict(available=True, sampling='nearest original acquisition to seven evenly spaced source-clock targets, with duplicate events removed',
        scope='saved K representative and saved auxiliary equilibrium at each original acquisition; posthoc FK only',
        interpretation='local rotation agreement, endpoint agreement with the estimated law, and physical endpoint agreement are separate quantities; no arrival or causal guarantee follows',
        samples=samples)


def selected_plan_model_comparisons(inputs, model, by_stamp):
    if not model or not inputs:
        return dict(available=False, reason='No complete goal-available selected plan with a saved model is present')
    from deflecomp_core.robot.pinocchio_robot import RobotArm
    robot = RobotArm(model['urdf_path'], tip_link=model['hand_path'][-1][0], base_link=model['root_frame'])
    samples = []
    for index in sorted(set(np.linspace(0, len(inputs)-1, min(7, len(inputs)), dtype=int))):
        source = inputs[index]
        try:
            robot._fk_update(np.asarray(source['auxiliary_equilibrium']))
            pose = robot.data.oMf[robot.base_fid].inverse()*robot.data.oMf[robot.tip_fid]
            actual = by_stamp.get(source['source_stamp'])
            samples.append(dict(source, model_hand=pose.translation.tolist(),
                model_to_same_selected_body_mean_m=float(np.linalg.norm(pose.translation-source['estimated_hand_mean'])),
                source_t_q_matches_physical=(actual is not None and actual['command'] == source['source_command']),
                model_to_physical_at_selection_source_m=(None if actual is None else
                    float(np.linalg.norm(pose.translation-actual['tip']))),
                physical_truth_used_only_for_posthoc_scoring=True, equilibrium_resolved=False))
        except Exception as error:
            samples.append(dict(source_stamp=source['source_stamp'], available=False,
                reason=type(error).__name__+': '+str(error)))
    return dict(available=True,
        scope='Actual K representative and saved auxiliary equilibrium captured by the selector; compare FK with the same saved hand mean used by S at that exact source command. No equilibrium is resolved.',
        samples=samples)


def summarize(run):
    run = Path(run)
    warnings, counts, checks, flags = [], Counter(), Counter(), Counter()
    names = ('physical_trajectory.jsonl', 'commands.jsonl', 'simple_updates.jsonl',
        'simple_plans.jsonl', 'observation_selection.jsonl', 'camera_previews.jsonl',
        'execution_failure.jsonl', 'shutdown.jsonl')
    frontiers = {name: (run/name).stat().st_size for name in names if (run/name).exists()}
    manifest = read_json(run/'motion_manifest.json')
    model = read_json(run/'simple_model.json')
    physical, by_stamp = {}, {}
    first = last = None
    minimum, minimum_stamp, reached = float('inf'), None, False
    for index, row in enumerate(rows(run/'physical_trajectory.jsonl', frontiers, warnings)):
        first = row if first is None else first
        last = row
        physical[index] = {key: row[key] for key in ('stamp', 'time', 'command', 'tip', 'physical_error_m')}
        by_stamp[row['stamp']] = physical[index]
        if row['physical_error_m'] < minimum:
            minimum, minimum_stamp = row['physical_error_m'], row['stamp']
        reached |= bool(row['reached'])
    counts['physical_rows'] = len(physical)
    physical_goal = posthoc_physical_goal(first)
    targets = [] if first is None else np.linspace(first['stamp'], last['stamp'], 7)
    closest = [None for _ in targets]
    closest_distance = [float('inf') for _ in targets]

    connection = sqlite3.connect('file:'+str(run/'observations.sqlite3')+'?mode=ro', uri=True)
    connection.execute('BEGIN')
    raw = {identity: (stamp, json.loads(command), sequence, status)
        for identity, stamp, command, sequence, status in connection.execute(
            'SELECT event_id,stamp,command_json,sequence,status FROM events')}
    DB_counts = dict(connection.execute('SELECT status,COUNT(*) FROM events GROUP BY status'))
    DB_sensor_counts = dict(connection.execute('SELECT kind,COUNT(*) FROM events GROUP BY kind'))
    forbidden, payload_keys = Counter(), {}
    prohibited = {'actual_joints', 'true_K', 'camera_pose', 'marker_pose', 'physical_pose', 'physical_link_poses'}
    for kind, payload in connection.execute('SELECT kind,payload_json FROM events WHERE payload_json IS NOT NULL'):
        value = json.loads(payload)
        payload_keys.setdefault(kind, set()).update(value)
        forbidden.update(set(all_keys(value)) & prohibited)
    connection.close()
    seen, previous_order, last_processed_stamp = set(), None, None
    statuses, kinds, failures, K_skip_reasons = Counter(), Counter(), Counter(), Counter()
    body_seconds, K_seconds, environment_cross = [], [], []
    body_by_kind, K_by_kind = {}, {}
    body_surprise_audit = BodySurpriseAudit()
    observation_coverage = {}
    first_K = last_K = previous_K = None
    last_K_stamp = None
    largest_Dirac_row, largest_local_translation_change = 0., 0.
    initial_translations = None
    for row in rows(run/'simple_updates.jsonl', frontiers, warnings):
        counts['processed_event_rows'] += 1
        identity = row['event_id']
        checks['duplicate_processed_event_ID'] += identity in seen
        seen.add(identity)
        actual = raw.get(identity)
        order = row['acquisition_stamp'], row['sequence']
        checks['original_t_q_sequence_mismatch'] += (actual is None or actual[0] != order[0]
            or actual[1] != row['acquisition_command'] or actual[2] != order[1])
        checks['processing_chronological_violation'] += previous_order is not None and order < previous_order
        previous_order, last_processed_stamp = order, order[0]
        kinds[row['kind']] += 1
        coverage = observation_coverage.setdefault(row['kind'],
            dict(processed=[], eligible=[], selected=[], fused=[], unselected=[]))
        coverage['processed'].append(row['acquisition_stamp'])
        for label, active in (('eligible', row.get('observation_eligible', False)),
                ('selected', row.get('observation_selected', False)),
                ('fused', bool(row['observation_ids'])),
                ('unselected', row.get('observation_eligible', False)
                    and not row.get('observation_selected', False))):
            if active:
                coverage[label].append(row['acquisition_stamp'])
        if row.get('observation_eligible') is not None:
            checks['fused_observation_not_selected'] += (bool(row['observation_ids'])
                and not row.get('observation_selected', False))
            checks['unselected_observation_fused_or_K_updated'] += (
                row['observation_eligible'] and not row.get('observation_selected', False)
                and (bool(row['observation_ids']) or row.get('stiffness') is not None))
        counts['fused_observation_events'] += bool(row['observation_ids'])
        counts['goal_observed_rows'] += bool(row['simple_inference']['goal_observed'])
        checks['simple_canonical_state_ID_mismatch'] += row['simple_inference']['state_id'] != row['body_state_ids'][0]
        covariance = np.asarray(row['body_covariance'])
        transforms = row['local_transforms']
        ids = [item['edge_id'] for item in transforms]
        body_ids = model['body_edge_ids'] if model else [name for name in ids if name.startswith('link_')]
        translation = [6*ids.index(name)+axis for name in body_ids for axis in range(3)]
        rotation = [6*ids.index(name)+axis for name in body_ids for axis in (3, 4, 5)]
        environment = [coordinate for i, name in enumerate(ids)
            if name not in body_ids and (name == 'marker' or name.startswith('marker'))
            for coordinate in range(6*i, 6*i+6)]
        largest_Dirac_row = max(largest_Dirac_row, float(np.max(np.abs(covariance[translation, :]))))
        if environment:
            environment_cross.append(float(np.linalg.norm(covariance[np.ix_(rotation, environment)])))
        current_translations = [transforms[ids.index(name)]['translation'] for name in body_ids]
        if initial_translations is None:
            initial_translations = np.asarray(current_translations)
        largest_local_translation_change = max(largest_local_translation_change,
            float(np.max(np.abs(np.asarray(current_translations)-initial_translations))))
        body_seconds.append(row['body_seconds'])
        K_seconds.append(row['K_seconds'])
        body_by_kind.setdefault(row['kind'], []).append(row['body_seconds'])
        K_by_kind.setdefault(row['kind'], []).append(row['K_seconds'])
        record = row.get('stiffness')
        body_surprise_audit.observe(row, record)
        if record is None or record.get('matching_target') is None:
            counts['events_without_K_update_record'] += 1
            checks['informative_observation_without_K_update_record'] += bool(row['observation_ids'])
            K_skip_reasons['no update record' if record is None else
                record.get('reason', record.get('status', 'uninformative event descriptor'))] += 1
            # A missing update is not a newly timestamped K estimate.  Retain
            # the last actual K source and never invent t/q/ID correspondence.
            continue
        truth_flags(record, flags)
        statuses[record['status']] += 1
        checks['K_matching_target_not_joint_rotation'] += record['matching_target'] != 'joint_rotation'
        checks['K_source_t_q_mismatch'] += (record['source_stamp'] != row['acquisition_stamp']
            or record['source_command'] != row['acquisition_command'])
        checks['K_body_source_ID_mismatch'] += record['body_posterior_state_ids'] != row['body_state_ids']
        checks['K_BFS_enabled'] += bool(record.get('bayes_factor_surprise_enabled'))
        checks['K_evidence_computed'] += bool(record.get('evidence_computed'))
        checks['K_retained_history_factor_count'] += int(record.get('retained_factor_count', 0))
        counts['K_Gaussian_updates'] += bool(record['update_applied'])
        counts['K_unavailable_projection_rows'] += len(record.get('simple_projection_failures', ()))
        for failure in record.get('simple_projection_failures', ()):
            failures[failure.get('unavailable_reason', 'unspecified')] += 1
        posterior = record.get('posterior') or {}
        checks['K_history_factor_IDs_retained'] += len(posterior.get('factor_ids', ()))
        checks['K_all_history_factors_reevaluated'] += bool(posterior.get('all_history_factors_reevaluated'))
        proposal = record.get('command_proposal')
        if proposal:
            K = proposal['selected_K']
            counts['K_numeric_changes'] += previous_K is not None and K != previous_K
            first_K = K if first_K is None else first_K
            last_K, previous_K, last_K_stamp = K, K, row['acquisition_stamp']
            for target_index, target in enumerate(targets):
                distance = abs(row['acquisition_stamp']-target)
                # At a shared sensor stamp use the last completed original
                # sequence, including its actual K update, rather than an
                # earlier sensor result carrying a different same-time law.
                if distance <= closest_distance[target_index]:
                    closest[target_index], closest_distance[target_index] = row, distance

    selection_modes, selection_batches, selection_counts = Counter(), 0, Counter()
    for row in rows(run/'observation_selection.jsonl', frontiers, warnings):
        selection_batches += 1
        selection_modes[row['mode']] += 1
        selected_ids = row['selected_event_ids']
        unselected_ids = row['unselected_event_ids']
        checks['selection_batch_repeated_ID'] += len(selected_ids+unselected_ids)-len(set(selected_ids+unselected_ids))
        checks['selection_batch_unknown_raw_ID'] += len(set(selected_ids+unselected_ids)-raw.keys())
        for sensor, data in row['per_sensor'].items():
            selection_counts[sensor+'_eligible'] += data['available_count']
            selection_counts[sensor+'_selected'] += data['selected_count']
            if data['selected_count'] >= 2:
                checks['selection_batch_endpoint_range_mismatch'] += data['selected_time_range'] != data['available_time_range']

    plans, S_changes, selected_steps, responses, goal_predictions, plan_model_inputs = [], [], [], [], [], []
    objective_families, objective_expressions, objective_residuals = Counter(), Counter(), []
    active_limits = Counter()
    physical_stamps = [row['stamp'] for row in physical.values()]
    for row in rows(run/'simple_plans.jsonl', frontiers, warnings):
        counts['selected_plan_rows'] += 1
        diagnostics = row['diagnostics']
        objective = reaching_objective(diagnostics)
        objective_families[objective['family']] += 1
        objective_expressions[objective['expression'] or 'not recorded'] += 1
        truth_flags(diagnostics, flags)
        checks['plan_actual_start_vs_selection_source_mismatch'] += row['start_command'] != diagnostics['source_command']
        checks['plan_body_goal_law_ID_mismatch'] += diagnostics['body_source_state_ids'] != diagnostics['goal_source_state_ids']
        checks['plan_body_reconstructed_from_K'] += bool(diagnostics['body_reconstructed_from_stiffness'])
        if diagnostics['goal_observed']:
            for score_name, point_name in (('S_before', 'source_score'), ('S_after', 'selected_score')):
                point_score = diagnostics[point_name]
                expected_score = declared_score(point_score, objective)
                if expected_score is not None:
                    objective_residuals.append(diagnostics[score_name]-expected_score)
                flag = point_score.get('difference_covariance_trace_in_objective')
                checks['plan_objective_point_score_declaration_mismatch'] += (
                    flag is not None and objective['covariance_trace_in_objective'] is not None
                    and flag != objective['covariance_trace_in_objective'])
            counts['selected_goal_available_plans'] += 1
            checks['plan_S_increased'] += diagnostics['S_after'] > diagnostics['S_before']
            S_changes.append(diagnostics['S_after']-diagnostics['S_before'])
            selected_steps.append(diagnostics['step_norm_rad'])
            command_map = diagnostics.get('command_map', {})
            if command_map.get('anchor_equilibrium') is not None:
                plan_model_inputs.append(dict(request_id=row['request_id'],
                    source_stamp=diagnostics['source_stamp'], source_command=diagnostics['source_command'],
                    adoption_stamp=row['adoption_stamp'],
                    original_observation_stamp=diagnostics.get('original_observation_stamp'),
                    source_state_id=diagnostics['body_source_state_ids'][0],
                    selected_K=command_map['selected_K'], density_id=command_map['density_id'],
                    auxiliary_equilibrium=command_map['anchor_equilibrium'],
                    estimated_hand_mean=diagnostics['source_score']['hand_mean']))
            goal_predictions.append(dict(stamp=row['adoption_stamp'],
                goal_mean=diagnostics['source_score']['goal_mean'],
                hand_mean=diagnostics['source_score']['hand_mean'],
                original_observation_stamp=diagnostics.get('original_observation_stamp'),
                body_acquisition_stamp=diagnostics.get('body_acquisition_stamp'),
                state_id=diagnostics['body_source_state_ids'][0]))
            response = finite_plan_response(row, physical, physical_stamps, by_stamp)
            if response is not None:
                responses.append(response)
            for coordinate, active in enumerate(diagnostics.get('selected_command_action', {}).get('active_joint_limits', ())):
                active_limits[str(coordinate)] += bool(active)
        plans.append(dict(adoption_stamp=row['adoption_stamp'], start=row['start_command'],
            target=row['target_command'], duration=row['duration']))

    previews = list(rows(run/'camera_previews.jsonl', frontiers, warnings))
    checkpoints = []
    for label, stamp in (() if first is None else (('physical_minimum', minimum_stamp),
            ('40_physical_seconds', first['stamp']+40.),
            ('70_physical_seconds', first['stamp']+70.), ('physical_end', last['stamp']))):
        if stamp > last['stamp'] or not goal_predictions:
            continue
        prediction = min(goal_predictions, key=lambda item: abs(item['stamp']-stamp))
        actual = min(physical.values(), key=lambda item: abs(item['stamp']-stamp))
        preview = min(previews, key=lambda item: abs(item['stamp']-stamp)) if previews else None
        fused_images = [value for value in observation_coverage.get('image', {}).get('fused', ()) if value <= stamp]
        checkpoints.append(dict(label=label, requested_stamp=stamp, prediction=prediction,
            prediction_time_offset_s=prediction['stamp']-stamp,
            goal_mean_error_m=(None if not physical_goal['available'] else
                float(np.linalg.norm(np.asarray(prediction['goal_mean'])-physical_goal['position_base_link']))),
            hand_mean_error_m=float(np.linalg.norm(np.asarray(prediction['hand_mean'])-actual['tip'])),
            physical_distance_m=actual['physical_error_m'],
            nearest_camera_detection=preview,
            last_fused_image_stamp=max(fused_images) if fused_images else None,
            fused_image_age_s=stamp-max(fused_images) if fused_images else None))

    previous_step, active_plan = None, -1
    maximum_reference_error = 0.
    densities, K_versions = {}, set()
    for row in rows(run/'commands.jsonl', frontiers, warnings):
        counts['actual_command_intervals'] += 1
        if previous_step is not None:
            checks['ledger_interval_clock_gap'] += row['start_stamp'] != previous_step['end_stamp']
            checks['ledger_interval_command_gap'] += row['start_command'] != previous_step['end_command']
        previous_step = row
        while active_plan+1 < len(plans) and plans[active_plan+1]['adoption_stamp'] < row['end_stamp']:
            active_plan += 1
        if active_plan >= 0:
            plan = plans[active_plan]
            start = by_stamp.get(plan['adoption_stamp'])
            end = by_stamp.get(row['end_stamp'])
            if start is not None and end is not None:
                fraction = min(1., max(0., (end['time']-start['time'])/plan['duration']))
                expected = np.asarray(plan['start'])+fraction*(np.asarray(plan['target'])-plan['start'])
                maximum_reference_error = max(maximum_reference_error,
                    float(np.max(np.abs(expected-row['end_command']))))
        if row['selected_K'] is not None:
            K_versions.add(row['density_id'])
            previous_K = densities.setdefault(row['density_id'], row['selected_K'])
            checks['same_density_ID_changed_recorded_K'] += row['selected_K'] != previous_K
            checks['interval_K_source_in_future'] += row['source_stamp'] is not None and row['source_stamp'] > row['start_stamp']

    sample_rows = {row['event_id']: row for row in closest if row is not None}
    comparisons = model_comparisons(sorted(sample_rows.values(), key=lambda row: (row['acquisition_stamp'], row['sequence'])), model, physical)
    for sample in comparisons.get('samples', ()):
        counts['posthoc_model_samples'] += 1
        counts['posthoc_model_samples_unavailable'] += not sample.get('available', True)
        if sample.get('available', True):
            checks['posthoc_reconstructed_state_ID_mismatch'] += not sample['state_id_matches']
            checks['posthoc_K_proposal_original_t_q_mismatch'] += not sample['original_t_q_matches_saved_proposal']
            checks['posthoc_physical_original_t_q_mismatch'] += not sample['physical_t_q_matches']
    shutdowns = list(rows(run/'shutdown.jsonl', frontiers, warnings))
    saved_hashes = {} if manifest is None else manifest.get('source_sha256', {})
    package = REPO/'ros/examples/probtf_observation_fusion_demo'
    source_changes = [name for name, checksum in saved_hashes.items()
        if not (package/name).is_file() or digest(package/name) != checksum]
    return dict(case=run.name, run_directory=str(run), jsonl_byte_frontiers=frontiers,
        warnings=warnings, counts=dict(counts), source_law_and_control_checks=dict(checks),
        physical_posthoc=None if last is None else dict(elapsed_s=last['time']-first['time'],
            saved_physical_source_clock_hz=(len(physical)-1)/(last['time']-first['time']) if last['time'] > first['time'] else None,
            minimum_error_m=minimum, final_error_m=last['physical_error_m'],
            minimum_error_original_stamp=minimum_stamp,
            reached_at_least_once=reached, terminal_reached=bool(last['reached']),
            collision_checking_enabled=last['collision_checking_enabled']),
        last_processed_original_stamp=last_processed_stamp,
        last_processed_source_age_at_physical_frontier_s=(None if last is None or last_processed_stamp is None else last['stamp']-last_processed_stamp),
        raw_DB_status_counts=DB_counts, archived_sensor_kind_counts=DB_sensor_counts,
        raw_DB_original_time_range=(None if not raw else
            [min(item[0] for item in raw.values()), max(item[0] for item in raw.values())]),
        frontier_interpretation='Each JSONL file is read only up to its recorded byte frontier; raw metadata uses one read-only SQLite transaction. During a running trial these snapshots need not have exactly the same source-clock frontier.',
        archived_event_count=len(raw), processed_unique_event_count=len(seen),
        archived_not_processed_count=len(set(raw)-seen),
        observation_source_coverage_by_sensor_kind={kind: {label: source_coverage(stamps)
            for label, stamps in coverage.items()} for kind, coverage in observation_coverage.items()},
        observation_selection=dict(batch_count=selection_batches, modes=dict(selection_modes),
            claimed_batch_counts=dict(selection_counts),
            interpretation='Raw acquisition, context processing, selected observations and actually fused observations are counted separately; selection batches may still be in progress at this frontier.'),
        processed_sensor_kind_counts=dict(kinds), K_status_counts=dict(statuses),
        body_bayes_factor_surprise=body_surprise_audit.record(),
        K_projection_failures=dict(failures), K_skipped_update_reasons=dict(K_skip_reasons),
        first_selected_K=first_K, last_selected_K=last_K,
        last_K_original_stamp=last_K_stamp, body_update_seconds=percentiles(body_seconds),
        K_update_seconds=percentiles(K_seconds), body_environment_rotation_cross_norm=percentiles(environment_cross),
        body_update_seconds_by_sensor_kind={kind: percentiles(values) for kind, values in body_by_kind.items()},
        K_update_seconds_by_sensor_kind={kind: percentiles(values) for kind, values in K_by_kind.items()},
        maximum_body_translation_full_covariance_row_abs=largest_Dirac_row,
        maximum_known_local_translation_change_m=largest_local_translation_change,
        selected_S_changes=percentiles(S_changes), selected_step_norm_rad=percentiles(selected_steps),
        reaching_objective=dict(plan_family_counts=dict(objective_families),
            saved_expression_counts=dict(objective_expressions),
            score_equation_residual_count=len(objective_residuals),
            maximum_absolute_score_equation_residual=(float(np.max(np.abs(objective_residuals)))
                if objective_residuals else None),
            covariance_interpretation='Covariance and full hand/goal cross remain diagnostic data. Their trace contributes to S only for the explicitly recorded expected-squared-distance objective; squared-mean-distance selection scores corrected means alone.'),
        finite_plan_posthoc_response=dict(
            comparison_count=len(responses),
            scope='Each plan uses its saved fixed-law prediction and physical frames at adoption and the first frame at or after its finite duration. Physical settling after this boundary and changes from subsequent observations are not attributed to this single plan. Covariance-trace changes remain diagnostics and are not described as score contributions for squared-mean-distance selection.',
            aggregate={key: percentiles([item[key] for item in responses if item[key] is not None])
                for key in ('predicted_mean_squared_distance_change_m2',
                    'predicted_covariance_trace_change_m2', 'predicted_mean_distance_reduction_m',
                    'actual_physical_distance_reduction_m', 'predicted_hand_displacement_norm_m',
                    'actual_hand_displacement_norm_m', 'predicted_actual_hand_displacement_cosine',
                    'predicted_hand_progress_along_estimated_goal_m',
                    'actual_hand_progress_along_estimated_goal_m',
                    'predicted_hand_goal_alignment_cosine',
                    'actual_hand_estimated_goal_alignment_cosine')},
            auxiliary_joint_limit_active_plan_counts=dict(active_limits),
            temporal_samples=([responses[index] for index in sorted(set(
                np.linspace(0, len(responses)-1, min(7, len(responses)), dtype=int)))] if responses else [])),
        physical_goal_posthoc=physical_goal,
        goal_and_detection_checkpoints_posthoc=checkpoints,
        camera_detection=camera_detection_summary(previews),
        actual_interval_K_density_versions=len(K_versions), finite_reference_max_abs_error_rad=maximum_reference_error,
        latest_manifest_available=manifest is not None,
        latest_manifest=selected(manifest, ('finished', 'simulated_elapsed_s', 'duration', 'estimation_drained',
            'all_captured_sensor_data_archived', 'processed_events', 'acquisition_requests', 'acquisition_archived',
            'fused_observations', 'thinned_observations', 'observation_selection',
            'body_bfs_enabled', 'body_bayes_factor_surprise', 'K_bfs_enabled', 'K_evidence_enabled',
            'actual_motion_hz', 'actual_camera_hz', 'actual_joint_hz', 'actual_imu_hz', 'failure')),
        shutdown=shutdowns[-1] if shutdowns else None,
        execution_failures=list(rows(run/'execution_failure.jsonl', frontiers, warnings)),
        truth_flag_values=dict(flags), forbidden_truth_payload_key_counts=dict(forbidden),
        raw_payload_keys={kind: sorted(keys) for kind, keys in payload_keys.items()},
        reconstructed_model_comparison=comparisons,
        selected_plan_model_comparison=selected_plan_model_comparisons(plan_model_inputs, model, by_stamp),
        source_hash_comparison=dict(expected_module_count=len(saved_hashes), changed_current_modules=source_changes))


def compact_report(report, full_path):
    compact = {key: report[key] for key in ('audited_wall_stamp', 'method',
        'production_source_comparison', 'protected_core_and_rviz', 'limitations')}
    compact['compiled_library_comparison'] = report.get('compiled_library_comparison')
    compact['full_report_outside_git'] = str(full_path)
    compact['runs'] = []
    for run in report['runs']:
        row = selected(run, ('case', 'run_directory', 'physical_posthoc', 'counts',
            'source_law_and_control_checks', 'last_processed_original_stamp',
            'last_processed_source_age_at_physical_frontier_s', 'raw_DB_status_counts',
            'archived_sensor_kind_counts', 'archived_event_count', 'processed_unique_event_count',
            'archived_not_processed_count', 'K_status_counts', 'K_projection_failures',
            'first_selected_K', 'last_selected_K', 'last_K_original_stamp',
            'maximum_body_translation_full_covariance_row_abs',
            'maximum_known_local_translation_change_m', 'finite_reference_max_abs_error_rad',
            'latest_manifest', 'truth_flag_values', 'forbidden_truth_payload_key_counts',
            'source_hash_comparison', 'physical_goal_posthoc', 'camera_detection'))
        row['reaching_objective'] = run.get('reaching_objective')
        row['body_bayes_factor_surprise'] = run.get('body_bayes_factor_surprise')
        row['execution_failures'] = run.get('execution_failures', [])
        row['observation_coverage_by_sensor_kind'] = {
            kind: dict(counts={name: data['count'] for name, data in coverage.items()},
                selected=coverage['selected'], fused=coverage['fused'])
            for kind, coverage in run['observation_source_coverage_by_sensor_kind'].items()}
        response = run['finite_plan_posthoc_response']
        row['finite_plan_posthoc_response'] = selected(response, ('comparison_count', 'scope',
            'auxiliary_joint_limit_active_plan_counts'))
        row['finite_plan_posthoc_response']['aggregate'] = {
            key: dict(count=data['count'], median=data['median'], mean=data['mean'])
            for key, data in response['aggregate'].items() if data is not None}
        row['goal_and_detection_checkpoints_posthoc'] = [dict(
            label=point['label'], requested_stamp=point['requested_stamp'],
            goal_mean=point['prediction']['goal_mean'],
            goal_mean_error_m=point['goal_mean_error_m'],
            hand_mean_error_m=point['hand_mean_error_m'],
            physical_distance_m=point['physical_distance_m'],
            nearest_camera_detection=point['nearest_camera_detection'],
            fused_image_age_s=point['fused_image_age_s'])
            for point in run['goal_and_detection_checkpoints_posthoc'] if point['label'] != 'physical_minimum']
        samples = run['reconstructed_model_comparison'].get('samples', ())
        row['last_posthoc_model_sample'] = samples[-1] if samples else None
        samples = run.get('selected_plan_model_comparison', {}).get('samples', ())
        row['last_selected_plan_model_sample'] = samples[-1] if samples else None
        compact['runs'].append(row)
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--compact-output', type=Path,
        help='Also save selected scalar checks and a few posthoc values for Git; full raw data stays outside Git.')
    parser.add_argument('--protected-baseline', type=Path,
        default=Path('/tmp/probtf_no_concentration_authorized_baseline_20261018.json'))
    arguments = parser.parse_args()
    source_manifest = arguments.source_manifest or arguments.runs[0].parent/'source_manifest.json'
    source_contents = read_json(source_manifest) or {}
    report = dict(audited_wall_stamp=time.time(),
        method='saved original observations, finite-command intervals and one current Gaussian K record per update; uniform-time posthoc FK samples reuse saved auxiliary equilibrium angles; no estimator/control rerun or truth injection',
        production_source_comparison=hash_comparison(source_manifest, 'production_sha256'),
        compiled_library_comparison=(hash_comparison(source_manifest, 'compiled_library_sha256')
            if 'compiled_library_sha256' in source_contents else
            dict(available=False, reason='compiled library hashes are absent from this source manifest')),
        protected_core_and_rviz=hash_comparison(arguments.protected_baseline, 'protected_sha256'),
        limitations=['Current JSONL byte frontiers and physical completion are recorded separately from full raw/estimation drain.',
            'K is an effective auxiliary-model parameter; orientation fit and physical hand accuracy are separate measurements.',
            'Moment queries use the existing Bingham-derived local approximation; no exact nonlinear distribution claim.',
            'Posthoc physical scoring does not install true K, realized joints or target truth into estimation.',
            'Matching IDs, finite reference checks and passing audits do not establish arrival or stability.',
            'Collision checking remains a separately reported simulation setting.'],
        runs=[summarize(run) for run in arguments.runs])
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if arguments.compact_output is not None:
        compact = compact_report(report, arguments.output)
        arguments.compact_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.compact_output.write_text(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(arguments.output), bytes=arguments.output.stat().st_size,
        runs=[dict(case=run['case'], counts=run['counts'], checks=run['source_law_and_control_checks'],
            physical=run['physical_posthoc']) for run in report['runs']]), ensure_ascii=False))


if __name__ == '__main__':
    main()
