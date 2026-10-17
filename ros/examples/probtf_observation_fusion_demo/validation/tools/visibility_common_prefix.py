#!/usr/bin/env python3
"""Evaluate original records in a common acquisition/source-time prefix.

No raw file is rewritten and no failed run is relabelled complete. Control
computation is selected by its original source stamp; actual packet adoption
is separately selected by adoption stamp, so a late computation is not
declared available earlier than it arrived.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


CLI = Path(__file__).resolve().parents[2]/'scripts/evaluate_running_gradient.py'
SPEC = importlib.util.spec_from_file_location('probtf_visibility_existing_cli', CLI)
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


class PrefixLines:
    """Stream the full original file for its hash, yielding selected rows only."""
    def __init__(self, path, warnings, selector, cutoff, rule):
        self.original = E.Lines(path, warnings)
        self.rows = self.excluded = self.unknown_time = 0
        self.selector, self.cutoff, self.rule = selector, cutoff, rule
        self.minimum_selected = self.maximum_selected = self.maximum_original = None

    def __iter__(self):
        for row in self.original:
            stamp = E.number(self.selector(row))
            if stamp is None:
                self.unknown_time += 1
                continue
            self.maximum_original = stamp if self.maximum_original is None else max(self.maximum_original, stamp)
            if stamp > self.cutoff:
                self.excluded += 1
                continue
            self.rows += 1
            self.minimum_selected = stamp if self.minimum_selected is None else min(self.minimum_selected, stamp)
            self.maximum_selected = stamp if self.maximum_selected is None else max(self.maximum_selected, stamp)
            yield row

    def record(self):
        return dict(original_file=self.original.record(), selected_rows=self.rows,
            excluded_later_rows=self.excluded, unknown_time_rows_not_selected=self.unknown_time,
            selector=self.rule, inclusive_cutoff=self.cutoff,
            minimum_selected_time=self.minimum_selected, maximum_selected_time=self.maximum_selected,
            maximum_original_time=self.maximum_original, original_row_timestamps_modified=False)


def compact(value):
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()
                if key not in ('distance_trace', 'projection_trace', 'prediction_trace')}
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


def first_failure(path):
    if not path.exists():
        return None
    with path.open() as stream:
        line = next((line for line in stream if line.strip()), None)
    if line is None:
        return None
    row = json.loads(line)
    stiffness = E.mapping(row.get('stiffness'))
    event = E.mapping(row.get('sensor_event'))
    return dict(file=str(path), event_id=row.get('event_id'), sensor_kind=row.get('sensor_kind'),
        acquisition_stamp=event.get('acquisition_stamp'), status=stiffness.get('status'),
        failure_reason=stiffness.get('failure_reason'))


def evaluate_run(label, run, duration, epoch):
    warnings = []
    manifest = E.load_manifest(run/'motion_manifest.json', warnings)
    options = SimpleNamespace(eta=None, gamma=None, near_threshold=None)
    config = E.configuration(manifest, options)
    cutoff = epoch+duration
    logs = dict(
        physical=PrefixLines(run/'physical_trajectory.jsonl', warnings,
            lambda row: row.get('time'), duration, 'physical.time <= prefix_duration_s'),
        commands=PrefixLines(run/'commands.jsonl', warnings,
            lambda row: E.mapping(row.get('step')).get('end_stamp'), cutoff,
            'commands.step.end_stamp <= epoch+prefix_duration_s; whole actual intervals only'),
        adoptions=PrefixLines(run/'gradient_adoptions.jsonl', warnings,
            lambda row: row.get('adoption_stamp'), cutoff,
            'gradient_adoptions.adoption_stamp <= epoch+prefix_duration_s'),
        control=PrefixLines(run/'control.jsonl', warnings,
            lambda row: E.mapping(row.get('gradient_evaluation')).get('source_stamp'), cutoff,
            'control.gradient_evaluation.source_stamp <= epoch+prefix_duration_s'),
        previews=PrefixLines(run/'camera_previews.jsonl', warnings,
            lambda row: row.get('stamp'), cutoff, 'camera_previews.stamp <= epoch+prefix_duration_s'),
        acquisitions=PrefixLines(run/'acquisition_requests.jsonl', warnings,
            lambda row: row.get('stamp'), cutoff, 'acquisition_requests.stamp <= epoch+prefix_duration_s'))
    physical = E.physical_summary(logs['physical'], config['near_threshold_m'], 40)
    commands = E.command_summary(logs['commands'], config)
    adoptions = E.adoption_summary(logs['adoptions'])
    control = E.control_summary(logs['control'], 40)
    geometry = E.scene_geometry(run, manifest, None, warnings)
    camera = E.camera_summary(logs['previews'], logs['acquisitions'], geometry, 40)
    last_image = camera['raw_detector']['last_acquisition_stamp']
    camera['evaluation_window'] = dict(requested_end_acquisition_stamp=cutoff,
        last_observed_preview_acquisition_stamp=last_image,
        unobserved_tail_to_requested_window_end_s=None if last_image is None else cutoff-last_image,
        censor_at_last_observed_acquisition_not_requested_boundary=True,
        unknown_tail_extended_as_visible_or_missing=False)
    implementations = E.mapping(manifest.get('implementation_sha256'))
    failure = first_failure(run/'estimation_failure.jsonl')
    return compact(dict(label=label, run_directory=str(run), prefix_duration_s=duration,
        acquisition_epoch=epoch, configuration=config, visibility_factor_config=manifest.get('visibility_factor_config'),
        physical=physical, camera_visibility=camera, commands=commands,
        gradient_adoptions=adoptions, control_worker=control,
        source_run_status=dict(requested_full_duration_s=manifest.get('duration'),
            original_finished=manifest.get('finished'), original_estimation_drained=manifest.get('estimation_drained'),
            original_simulated_elapsed_s=manifest.get('simulated_elapsed_s'),
            physical_log_advanced_beyond_prefix=(logs['physical'].maximum_original is not None
                and logs['physical'].maximum_original > duration),
            original_manifest_failure=manifest.get('failure'), first_estimation_failure=failure,
            first_execution_failure=first_failure(run/'execution_failure.jsonl'),
            prefix_analysis_does_not_claim_full_run_completed=True),
        source_identity=dict(implementation_unchanged_during_execution=manifest.get('implementation_unchanged_during_execution'),
            implementation_manifest_sha256=hashlib.sha256(json.dumps(implementations, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest(),
            implementation_file_count=len(implementations), physical_model_source=manifest.get('physical_model_source')),
        filters={key: value.record() for key, value in logs.items()}, warnings=warnings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=30.)
    parser.add_argument('--epoch', type=float, default=1792080000.)
    parser.add_argument('--output', type=Path, default=Path('/tmp/probtf_visibility_common30_20261018.json'))
    parser.add_argument('--include-prior', action='store_true')
    parser.add_argument('--zero-root', type=Path, default=Path('/tmp/probtf_visibility_zero60_20261018_0827'))
    parser.add_argument('--positive-root', type=Path, default=Path('/tmp/probtf_visibility_positive60_20261018_0826'))
    options = parser.parse_args()
    sources = [('formal_zero60', options.zero_root),
               ('formal_positive60', options.positive_root)]
    if options.include_prior:
        sources.append(('earlier_positive90_separate_repeat', Path('/tmp/probtf_visibility_positive_20261018_0823')))
    result = dict(schema='probtf_visibility_common_prefix_evaluation_v1', evaluation_only=True,
        original_raw_records_rewritten=False, failed_source_runs_relabelled_complete=False,
        source_time_clip_duration_s=options.duration, source_time_clip_epoch=options.epoch,
        existing_cli_path=str(CLI), existing_cli_sha256=hashlib.sha256(CLI.read_bytes()).hexdigest(),
        helper_path=str(Path(__file__).resolve()), helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        comparison_limitations='Same requested duration/seed/settings are not wall-time-synchronized inference or an identical executed path. Control timing/statistics use source-time selection; adoption uses actual adoption time separately.',
        control_selection_note='A gradient whose source is before the cutoff can finish after it; this source-window summary does not assert earlier availability. Actual adoption-window metrics are separate.',
        runs={})
    for group, base in sources:
        for case in ('marker', 'imu_marker'):
            label = group+'_'+case
            result['runs'][label] = evaluate_run(label, base/case, options.duration, options.epoch)
            options.output.write_text(json.dumps(E.finite_json(result), indent=2, sort_keys=True, allow_nan=False)+'\n')
            row = result['runs'][label]
            print(json.dumps(dict(label=label, minimum=row['physical']['minimum'], final=row['physical']['final'],
                detection_frame_fraction=row['camera_visibility']['raw_detector']['detected_frame_fraction'],
                detection_time_fraction=row['camera_visibility']['raw_detector']['time_weighted_visible_fraction'],
                original_finished=row['source_run_status']['original_finished'])), flush=True)
    print(str(options.output), flush=True)


if __name__ == '__main__':
    main()
