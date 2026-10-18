#!/usr/bin/env python3
"""Read existing camera-publication journals; never rerun or alter sources."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = fraction*(len(values)-1)
    low, high = int(index), min(int(index)+1, len(values)-1)
    return values[low]+(index-low)*(values[high]-values[low])


def stats(values):
    return dict(count=len(values), median=quantile(values, .5), p95=quantile(values, .95),
        p99=quantile(values, .99), max=max(values) if values else None)


def gaps(values):
    differences = [right-left for left, right in zip(values, values[1:])]
    return dict(stats(differences),
        over_0_2_s=sum(value > .2 for value in differences),
        over_0_5_s=sum(value > .5 for value in differences),
        over_1_s=sum(value > 1. for value in differences),
        over_3_s=sum(value > 3. for value in differences))


def summarize(path):
    manifest = json.loads((path/'motion_manifest.json').read_text())
    if not manifest['finished'] or manifest['simulated_elapsed_s'] < 180.-1.e-6:
        raise ValueError(str(path)+' has not completed 180 physical seconds; no prefix is exported as final.')
    publications = list(rows(path/'camera_live_publication.jsonl'))
    captures = {item['frame_index']: item for item in rows(path/'acquisition_requests.jsonl')
                if item['kind'] == 'image'}
    ordered = list(captures.values())
    checks = Counter()
    for item in publications:
        original = captures.get(item['frame_index'])
        checks['source_frame_absent_from_original_capture'] += original is None
        if original is not None:
            checks['original_source_stamp_mismatch'] += original['stamp'] != item['source_stamp']
            checks['original_received_wall_stamp_mismatch'] += original['received_wall_stamp'] != item['received_wall_stamp']
        checks['source_pixels_changed_true'] += bool(item['source_pixels_changed'])
        checks['new_observation_created_true'] += bool(item['new_observation_created'])
        checks['direct_route_false'] += not item['bypasses_coordinator_and_scene_publication_queues']
    for previous, current in zip(publications, publications[1:]):
        checks['nonincreasing_source_frame_index'] += current['frame_index'] <= previous['frame_index']
        checks['source_stamp_reversal'] += current['source_stamp'] < previous['source_stamp']
        checks['publish_wall_stamp_reversal'] += current['publish_wall_stamp'] < previous['publish_wall_stamp']
    selected_manifest = {key: manifest.get(key) for key in (
        'simulated_elapsed_s', 'duration', 'finished', 'reached', 'final_physical_error_m',
        'collision_checking_enabled', 'reach_condition', 'actual_motion_hz', 'actual_camera_hz',
        'actual_control_hz', 'acquisition_requests', 'acquisition_archived',
        'all_captured_sensor_data_archived', 'all_enabled_observations_processed',
        'estimation_drained', 'body_estimation_drained', 'stiffness_estimation_drained',
        'pending_stiffness_job_sequences', 'stiffness_failure', 'stiffness_worker_failed',
        'camera_preview_publication_route', 'implementation_unchanged_during_execution')}
    probe = json.loads((path/'subscriber_latency_probe.json').read_text())
    names = ('camera_live_publication.jsonl', 'acquisition_requests.jsonl',
             'motion_manifest.json', 'subscriber_latency_probe.json')
    return dict(case=path.name, run_directory=str(path), source_sha256={name: sha256(path/name) for name in names},
        manifest=selected_manifest, publication_count=len(publications), capture_count=len(captures),
        captures_without_publication_at_read=len(set(captures)-{item['frame_index'] for item in publications}),
        input_to_publish_wall_age_s=stats([item['input_to_publish_wall_age_s'] for item in publications]),
        ready_to_publish_wall_age_s=stats([item['ready_to_publish_wall_age_s'] for item in publications]),
        publication_wall_gaps_s=gaps([item['publish_wall_stamp'] for item in publications]),
        original_capture_wall_gaps_s=gaps([item['received_wall_stamp'] for item in ordered]),
        original_capture_source_clock_gaps_s=gaps([item['stamp'] for item in ordered]),
        published_source_clock_gaps_s=gaps([item['source_stamp'] for item in publications]),
        source_identity_checks=dict(checks),
        first_publication={key: publications[0][key] for key in ('frame_index', 'source_stamp', 'publish_wall_stamp')},
        last_publication={key: publications[-1][key] for key in ('frame_index', 'source_stamp', 'publish_wall_stamp')},
        subscriber_probe=probe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(scope='complete 180-physical-second direct acquisition-worker ROS publication trials',
        limits=[
            'ROS publish-call return is measured, not RViz rendering completion.',
            'Subscriber reception latency is a separate short probe, not a continuous 180-second measurement.',
            'Frame indices identify physical frames; their gaps are not dropped-camera-frame counts.',
            'Publication gaps and original acquisition gaps are reported separately. The direct route cannot create captures during a coordinator/physics pause.',
            'Large original acquisition gaps, including roughly three-second stalls, remain; this display-route change does not cure coordinator stalls.',
            'Physical reaching and raw archival/inference drain are independent of preview freshness.',
            'Current journals are summarized at read time after physical completion; later drain/publication can append to them.'],
        runs=[summarize(path) for path in args.runs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), bytes=args.output.stat().st_size,
        runs=[dict(case=item['case'], checks=item['source_identity_checks'],
            input_to_publish=item['input_to_publish_wall_age_s'],
            publication_gaps=item['publication_wall_gaps_s']) for item in report['runs']])))


if __name__ == '__main__':
    main()
