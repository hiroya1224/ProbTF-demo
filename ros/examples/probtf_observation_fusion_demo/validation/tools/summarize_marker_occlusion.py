#!/usr/bin/env python3
"""Read-only original-clock audit of the optional physical marker occluder.

This consumes saved observations and posthoc physical rows. It does not render
new sensor input, run inference, install physical truth, or alter a run.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

import numpy as np


def rows(path):
    with Path(path).open() as source:
        for line in source:
            yield json.loads(line)


def timing(values):
    return (dict(count=len(values), median=float(np.median(values)),
        p95=float(np.percentile(values, 95)), maximum=float(np.max(values)))
        if values else None)


def visibility_periods(samples):
    """Preserve every loss/reacquisition; do not assume an IMU-only tail."""
    periods = []
    for item in samples:
        visible = bool(item['visible'])
        if not periods or periods[-1]['visible'] != visible:
            periods.append(dict(visible=visible, first_stamp=item['stamp'],
                                last_stamp=item['stamp'], samples=1))
        else:
            periods[-1]['last_stamp'] = item['stamp']
            periods[-1]['samples'] += 1
    seen = False
    reacquisitions = 0
    for period in periods:
        if period['visible']:
            reacquisitions += seen
            seen = True
    return dict(periods=periods, reacquisitions=int(reacquisitions),
        terminal_lost_interval=(periods[-1] if periods and not periods[-1]['visible'] else None),
        initial_seen=bool(samples and samples[0]['visible']))


def after_loss_counts(updates, plans, terminal_loss_stamp):
    """Use original acquisition stamps for fused observations, never arrival order."""
    counts = Counter()
    if terminal_loss_stamp is None:
        return dict(available=False, reason='No terminal lost interval in saved camera previews.')
    for item in updates:
        if item['acquisition_stamp'] >= terminal_loss_stamp:
            counts['processed_'+item['kind']] += 1
            if item['observation_ids']:
                counts['fused_'+item['kind']] += 1
                counts['fused_with_goal_latched'] += bool(item.get('simple_inference', {}).get('goal_observed'))
    for item in plans:
        if item['adoption_stamp'] >= terminal_loss_stamp:
            counts['plans_adopted'] += 1
            counts['plans_with_retained_goal'] += bool(item['diagnostics']['goal_observed'])
            counts['plans_sending_nonzero_command_increment'] += bool(
                np.linalg.norm(np.asarray(item['target_command'])-item['start_command']) > 0.)
            observation_stamp = item['diagnostics'].get('original_observation_stamp')
            counts['plans_using_observation_acquired_after_loss'] += bool(
                observation_stamp is not None and observation_stamp >= terminal_loss_stamp)
    return dict(available=True, source_stamp=terminal_loss_stamp, counts=dict(counts))


def summarize(run):
    run = Path(run)
    manifest = json.loads((run/'motion_manifest.json').read_text())
    complete = (manifest['finished'] and manifest['estimation_drained']
                and manifest['simulated_elapsed_s'] >= manifest['duration'])
    if not complete:
        raise RuntimeError('Final summary requires completed physical duration and complete estimation drain.')
    preview = list(rows(run/'camera_previews.jsonl'))
    occlusion = list(rows(run/'physical_camera_occlusion.jsonl'))
    by_frame = {item['index']: item for item in preview}
    by_event = {item['event_id']: item for item in occlusion}
    physical = list(rows(run/'physical_trajectory.jsonl'))
    plans = list(rows(run/'simple_plans.jsonl'))
    start = physical[0]['stamp']
    visibility = visibility_periods(preview)
    checks = Counter()
    checks['camera_preview_duplicate_frame_index'] = len(preview)-len(by_frame)
    checks['physical_camera_duplicate_event_ID'] = len(occlusion)-len(by_event)
    checks['camera_preview_original_stamp_regressions'] = sum(
        second['stamp'] < first['stamp'] for first, second in zip(preview, preview[1:]))
    checks['camera_preview_original_frame_regressions'] = sum(
        second['index'] <= first['index'] for first, second in zip(preview, preview[1:]))
    fractions = []
    image_fusions = []
    fused_counts = Counter()
    connection = sqlite3.connect('file:'+str(run/'observations.sqlite3')+'?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    for item in occlusion:
        raw = connection.execute('SELECT * FROM events WHERE event_id=?', (item['event_id'],)).fetchone()
        metadata = json.loads(raw['payload_json']) if raw else {}
        checks['physical_preview_raw_event_missing'] += raw is None
        checks['physical_preview_original_stamp_mismatch'] += bool(raw and item['stamp'] != raw['stamp'])
        checks['physical_preview_original_command_mismatch'] += bool(raw and item['command'] != json.loads(raw['command_json']))
        checks['physical_preview_original_image_hash_mismatch'] += item['image_sha256'] != metadata.get('image_sha256')
        checks['physical_preview_original_frame_mismatch'] += item['frame_index'] != metadata.get('frame_index')
        shown = by_frame.get(item['frame_index'])
        checks['physical_preview_camera_preview_missing'] += shown is None
        if shown:
            checks['physical_preview_camera_preview_stamp_mismatch'] += shown['stamp'] != item['stamp']
            checks['physical_preview_camera_preview_visible_mismatch'] += shown['visible'] != item['marker_visible']
            checks['physical_preview_camera_preview_marker_ids_mismatch'] += shown['visible_marker_ids'] != item['visible_marker_ids']
        sheet = item['physical_occluder']
        fractions.append(sheet['occluded_board_fraction'])
        checks['distance_trigger_used'] += bool(sheet['distance_trigger_used'])
        checks['sensor_events_disabled'] += bool(sheet['sensor_events_disabled'])
        checks['physical_preview_truth_boundary_scope_mismatch'] += item['scope'] != 'physical image generation and detector preview only; not an estimator input'
    for item in rows(run/'simple_updates.jsonl'):
        if item['observation_ids']:
            fused_counts[item['kind']] += 1
            if item['kind'] == 'image':
                camera = by_event.get(item['event_id'])
                checks['fused_image_without_physical_preview'] += camera is None
                checks['fused_image_with_zero_original_decoded_markers'] += bool(camera and not camera['visible_marker_ids'])
                image_fusions.append(dict(event_id=item['event_id'], stamp=item['acquisition_stamp'],
                    relative_time_s=item['acquisition_stamp']-start,
                    visible_marker_ids=None if camera is None else camera['visible_marker_ids']))
        inference = item.get('simple_inference', {})
        checks['simple_inference_physical_truth_received'] += bool(inference.get('physical_truth_received'))
        checks['simple_inference_monte_carlo_used'] += bool(inference.get('monte_carlo_used'))
        if inference.get('goal_observed'):
            checks['body_state_simple_state_ID_mismatch'] += item['body_state_ids'] != [inference.get('state_id')]
    tail = visibility['terminal_lost_interval']
    tail_stamp = None if tail is None else tail['first_stamp']
    tail_counts = after_loss_counts(rows(run/'simple_updates.jsonl'), plans, tail_stamp)
    if tail_stamp is not None:
        commands = np.asarray([item['command'] for item in physical if item['stamp'] >= tail_stamp])
        tail_counts['actual_sent_command_path_length_rad'] = float(np.linalg.norm(np.diff(commands, axis=0), axis=1).sum()) if len(commands)>1 else 0.
        tail_counts['actual_sent_command_endpoint_displacement_rad'] = float(np.linalg.norm(commands[-1]-commands[0])) if len(commands)>1 else 0.
    imu_targets = Counter()
    for item in connection.execute("SELECT payload_json FROM events WHERE kind='imu'"):
        imu_targets[json.loads(item[0]).get('imu_target_link')] += 1
    final_tick_request_walls = [item['received_wall_stamp'] for item in rows(run/'acquisition_requests.jsonl')
        if item['stamp'] == physical[-1]['stamp']]
    final_tick_wall = min(final_tick_request_walls) if final_tick_request_walls else None
    if tail_stamp is not None and final_tick_wall is not None:
        fusion_wall_counts = Counter()
        latency_after_loss = []
        for event in connection.execute('SELECT kind, stamp, received_wall_stamp, completed_wall_stamp, result_metadata_json FROM events WHERE stamp>=?', (tail_stamp,)):
            metadata = json.loads(event['result_metadata_json']) if event['result_metadata_json'] else {}
            if metadata.get('observation_fused') and event['completed_wall_stamp'] is not None:
                phase = ('before_final_physical_tick' if event['completed_wall_stamp'] <= final_tick_wall
                         else 'during_drain_after_final_physical_tick')
                fusion_wall_counts[event['kind']+'_'+phase] += 1
                latency_after_loss.append(event['completed_wall_stamp']-event['received_wall_stamp'])
        tail_counts['fusion_completion_wall_counts'] = dict(fusion_wall_counts)
        tail_counts['fused_observation_acquisition_to_completion_wall_s'] = timing(latency_after_loss)
        tail_counts['final_physical_tick_request_wall_boundary'] = final_tick_wall
        tail_counts['wall_boundary_definition'] = 'Earliest acquisition-request received_wall_stamp attached to the saved final physical tick; an explicit logged boundary proxy, not a rewritten sensor time.'
        effective_stamps = [item['diagnostics'].get('original_observation_stamp') for item in plans
            if item['adoption_stamp'] >= tail_stamp and item['diagnostics'].get('original_observation_stamp') is not None]
        tail_counts['latest_effective_observation_source_stamp_used_during_physical_motion'] = max(effective_stamps) if effective_stamps else None
    connection.close()
    for plan in plans:
        diagnostics = plan['diagnostics']
        if diagnostics['goal_observed']:
            checks['plan_body_goal_source_ID_mismatch'] += diagnostics['body_source_state_ids'] != diagnostics['goal_source_state_ids']
            for score_name in ('source_score', 'selected_score'):
                score = diagnostics[score_name]
                checks['point_hand_goal_joint_state_ID_mismatch'] += not (score['hand_state_id'] == score['goal_state_id'] == score['joint_source_state_id'])
    camera_timing = {key: [] for key in ('render_seconds', 'preview_detection_seconds',
        'foreground_queue_seconds', 'total_work_seconds')}
    for item in rows(run/'camera_pipeline.jsonl'):
        if item['readiness'] == 'preview':
            for key in camera_timing:
                if key in item.get('timing', {}):
                    camera_timing[key].append(item['timing'][key])
    publication_timing = {key: [] for key in ('input_to_publish_wall_age_s', 'ready_to_publish_wall_age_s')}
    for item in rows(run/'camera_live_publication.jsonl'):
        for key in publication_timing:
            if key in item:
                publication_timing[key].append(item[key])
    stamps = np.asarray([item['stamp'] for item in preview])
    finite_physical = [item for item in physical if np.isfinite(item['physical_error_m'])]
    best = min(finite_physical, key=lambda item:item['physical_error_m'])
    last = finite_physical[-1]
    return dict(scope='Actual new physical occlusion run; all acquisition identities retained. Physical distances are posthoc only.',
        run=str(run), original_start_stamp=start,
        completion=dict(physical_elapsed_s=manifest['simulated_elapsed_s'], physical_finished=manifest['finished'],
            estimation_drained=manifest['estimation_drained'], failure=manifest['failure']),
        physical_occluder=manifest['physical_marker_occluder'],
        camera=dict(preview_count=len(preview), physical_occlusion_record_count=len(occlusion),
            original_time_range_s=[float(stamps[0]-start), float(stamps[-1]-start)],
            original_capture_gap_max_s=float(np.max(np.diff(stamps))) if len(stamps)>1 else None,
            visible_count=sum(item['visible'] for item in preview),
            missing_count=sum(not item['visible'] for item in preview),
            initial_marker_ids=preview[0]['visible_marker_ids'] if preview else None,
            final_marker_ids=preview[-1]['visible_marker_ids'] if preview else None,
            occluded_board_fraction=dict(first=fractions[0], final=fractions[-1],
                minimum=min(fractions), maximum=max(fractions), fully_covered_frames=sum(value == 1. for value in fractions)),
            pipeline_wall_seconds={key: timing(value) for key, value in camera_timing.items()},
            direct_ROS_publication_wall_seconds={key: timing(value) for key, value in publication_timing.items()},
            **visibility),
        fused_sensor_counts=dict(fused_counts), original_image_fusion_events=image_fusions,
        post_terminal_camera_loss=tail_counts, original_raw_imu_target_counts=dict(imu_targets),
        checks=dict(checks),
        physical_posthoc=dict(minimum_distance_mm=1000.*best['physical_error_m'], minimum_at_s=best['time'],
            final_distance_mm=1000.*last['physical_error_m'], any_reached=any(item['reached'] for item in finite_physical),
            terminal_reached=last['reached'], collision_checking_enabled=manifest['collision_checking_enabled'],
            reach_condition=last['reach_condition']),
        limitations=['The last lost interval is defined from actual preview source times, not assumed from a distance trigger.',
            'Raw image decoding and selected observation fusion have different counts; zero original decoded markers are never counted as a marker observation.',
            'Continuation counts establish execution and IMU fusion, not successful arrival or accurate estimation.',
            'Final drained fusion counts include updates completed after physical motion ended; the separately recorded wall-boundary counts identify this distinction.',
            'ROSpublish and preview source gaps do not measure RViz display completion.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), completion=result['completion'],
        physical_posthoc=result['physical_posthoc'], checks=result['checks'])))


if __name__ == '__main__':
    main()
