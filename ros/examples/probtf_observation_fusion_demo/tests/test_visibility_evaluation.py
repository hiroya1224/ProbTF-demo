"""Posthoc visibility metrics use acquired times and preserve factor provenance."""

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/evaluate_running_gradient.py'
SPEC = importlib.util.spec_from_file_location('visibility_evaluation_cli', SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def pose(x=0., y=0., z=0.):
    return [[1., 0., 0., x], [0., 1., 0., y], [0., 0., 1., z], [0., 0., 0., 1.]]


def geometry():
    return dict(status='available', camera=dict(width=100., height=100., fx=100., fy=100.,
        cx=50., cy=50., near_plane_m=.02), board_sizes_m=dict(white_support=[.27, .27],
        black_region=[.252, .252]), exposure_time_s=0., motion_blur_samples=1)


def test_detection_frame_fraction_time_weighting_and_unrecovered_loss_are_distinct():
    timeline = evaluation.VisibilityTimeline()
    for stamp, detected in zip([100., 101., 102., 103., 104.], [True, False, False, True, False]):
        timeline.add(stamp, detected)
    result = timeline.record()
    assert result['detected_frame_fraction'] == .4
    assert result['time_weighted_visible_fraction'] == .5
    assert result['time_coverage_s'] == 4.
    assert result['reacquisition_time_s']['count'] == 1
    assert result['reacquisition_time_s']['mean'] == 2.
    assert result['right_censored_loss_episodes'] == 1
    last = result['loss_episodes'][-1]
    assert last['right_censored']
    assert last['reacquisition_time_s'] is None
    assert last['censored_loss_duration_lower_bound_s'] == 0.
    assert last['observation_end_acquisition_stamp'] == 104.


def test_initial_detection_is_not_called_reacquisition_and_unknown_gaps_are_censored():
    timeline = evaluation.VisibilityTimeline()
    timeline.add(5., False)
    timeline.add(6., False)
    timeline.add(7., True)
    timeline.add(8., False)
    timeline.add(9., None)
    timeline.add(10., True)
    result = timeline.record()
    assert result['initial_acquisition_to_first_detection_s'] == 2.
    assert result['reacquisition_time_s'] is None
    assert result['left_censored_loss_episodes'] == 1
    assert result['right_censored_loss_episodes'] == 1
    assert result['time_coverage_s'] == 3.
    assert result['loss_episodes'][0]['left_censored']
    assert result['loss_episodes'][1]['observation_end_acquisition_stamp'] == 8.


def test_camera_acquisition_detection_and_true_board_visibility_are_separate(tmp_path):
    requests = [dict(kind='image', frame_index=i, stamp=100.+i, camera_pose=pose(),
        marker_pose=marker) for i, marker in enumerate([pose(z=1.), pose(x=.37, z=1.), pose(z=-1.)])]
    previews = [dict(index=i, stamp=100.+i, visible=True, visible_marker_ids=[21]) for i in range(3)]
    # This unmatched preview is never attached to the nearest acquisition.
    previews.insert(1, dict(index=99, stamp=100.5, visible=False, visible_marker_ids=[]))
    write_lines(tmp_path/'acquisition_requests.jsonl', requests)
    write_lines(tmp_path/'camera_previews.jsonl', previews)
    warnings = []
    result = evaluation.camera_summary(evaluation.Lines(tmp_path/'camera_previews.jsonl', warnings),
        evaluation.Lines(tmp_path/'acquisition_requests.jsonl', warnings), geometry(), 20)
    assert result['raw_detector']['detected_frame_fraction'] == .75
    assert result['acquisition']['image_requests'] == 3
    assert result['acquisition']['image_acquisition_hz'] == 1.
    assert result['acquisition']['previews_per_image_request'] == 4./3.
    truth = result['physical_geometry']
    points = truth['projection_trace']['samples']
    assert points[0]['instantaneous']['white_support']['all_corners_inside']
    assert not points[1]['instantaneous']['white_support']['all_corners_inside']
    assert points[1]['instantaneous']['black_region']['all_corners_inside']
    assert not points[2]['instantaneous']['black_region']['all_corners_inside']
    assert points[2]['instantaneous']['black_region']['pixels'] == [None]*4
    assert truth['detector_geometry_pairs']['white_support:some_corners_outside:detected'] == 2
    assert truth['physical_truth_used_only_for_posthoc_projection']
    assert not truth['controller_input']


def test_exposure_truth_uses_the_renderer_temporal_poses_not_only_the_last_pose(tmp_path):
    settings = geometry()
    settings.update(exposure_time_s=1., motion_blur_samples=3)
    request = dict(kind='image', frame_index=1, stamp=101., camera_pose=pose(), marker_pose=pose(z=1.),
        camera_history=[[0., pose(x=-.6)], [1., pose()]])
    write_lines(tmp_path/'acquisition_requests.jsonl', [request])
    result = evaluation.camera_summary(evaluation.Lines(tmp_path/'no_previews', []),
        evaluation.Lines(tmp_path/'acquisition_requests.jsonl', []), settings, 20)
    point = result['physical_geometry']['projection_trace']['samples'][0]
    assert point['instantaneous']['white_support']['all_corners_inside']
    assert not point['exposure']['white_support']['all_corners_inside_every_render_sample']
    assert point['exposure']['white_support']['temporal_pose_samples'] == 3
    assert point['detected'] is None


def test_bad_truth_transform_is_unavailable_not_an_invented_projection():
    bad = pose(z=1.)
    bad[0][0] = 2.
    assert evaluation.project_board(pose(), bad, [.27, .27], geometry()['camera']) is None
    behind = evaluation.project_board(pose(), pose(z=-1.), [.27, .27], geometry()['camera'])
    assert not behind['all_corners_inside']


def test_missing_frame_identity_or_exposure_metadata_is_not_assumed_from_time(tmp_path):
    write_lines(tmp_path/'acquisition_requests.jsonl', [dict(kind='image', stamp=100.,
        camera_pose=pose(), marker_pose=pose(z=1.))])
    write_lines(tmp_path/'camera_previews.jsonl', [dict(stamp=100., visible=True)])
    settings = geometry()
    settings['exposure_time_s'] = None
    result = evaluation.camera_summary(evaluation.Lines(tmp_path/'camera_previews.jsonl', []),
        evaluation.Lines(tmp_path/'acquisition_requests.jsonl', []), settings, 20)
    assert result['raw_detector']['detected_frame_fraction'] == 1.
    assert result['physical_geometry']['detector_geometry_pairs'] == {}
    point = result['physical_geometry']['projection_trace']['samples'][0]
    assert point['detected'] is None
    assert 'exposure' not in point
    assert result['physical_geometry']['exposure']['white_support']['all_corners_inside_frame_fraction'] is None
    assert evaluation.exposure_poses({}, dict(exposure_time_s=.1, motion_blur_samples=7))[0] is None


def test_factor_metrics_preserve_gradients_signs_modes_and_check_the_same_source():
    source = dict(body_state_ids=['body'], goal_state_ids=['goal-a', 'goal-b'], source_stamp=12.,
        source_command=[.2, .3], selected_K=[20., 30.], parameter_density_id='K-law',
        body_observation_source_stamp=5., goal_observation_source_stamp=3.)
    factor = dict(source_ids=deepcopy(source), status='ok', enabled=True, weight=2., penalty=.5,
        per_mode_record=[dict(index=0, source_component_state_id='goal-a', source_state_id='trial-state',
            source_stamp=12., position_mean=[.1, .2, 1.], position_covariance=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
            optical_axis_angle_rad=.3, predicted_pixel=[51., 72.], front_or_back='front', status='ok')])
    record = dict(source, status='ok', task_gradient=[1.,0.], view_gradient=[-1.,1.],
        weighted_view_gradient=[-2.,2.], selected_gradient=[-1.,2.], gradient=[-1.,2.],
        log_task_factor=-4., log_view_factor=-1., log_select_factor=-5., visibility_factor=factor,
        computation_seconds=.7, view_probe_computation_seconds=.2,
        task_and_FD_overhead_seconds_excluding_view_probes=.4,
        task_computation_seconds_including_view_probes=.6, view_probe_count=13)
    result = evaluation.control_summary([dict(request_id=1, control_mode='running_gradient',
        gradient_evaluation=record, work_seconds=1.)])
    factors = result['selection_factors']
    stats = factors['statistics']
    assert stats['task_gradient_norm']['mean'] == 1.
    assert stats['view_gradient_norm']['mean'] == pytest.approx(math.sqrt(2.))
    assert stats['task_view_cosine']['mean'] == pytest.approx(-1./math.sqrt(2.))
    assert stats['gradient_sum_residual_inf']['maximum'] == 0.
    assert stats['weighted_view_scale_residual_inf']['maximum'] == 0.
    assert stats['view_log_factor_residual']['maximum'] == 0.
    assert stats['factor_log_sum_residual']['maximum'] == 0.
    assert stats['view_probe_computation_seconds']['mean'] == .2
    assert stats['task_and_FD_overhead_seconds_excluding_view_probes']['mean'] == .4
    assert stats['task_computation_seconds_including_view_probes']['mean'] == .6
    assert stats['view_probe_count']['mean'] == 13.
    assert stats['visibility_computation_seconds'] is None
    assert factors['opposed_gradient_fraction'] == 1.
    assert factors['factor_source_identity_checks']['rows_fully_checkable'] == 1
    assert not factors['factor_source_identity_checks'].get('rows_with_mismatch')
    trace = factors['prediction_trace']['samples'][0]
    assert trace['source_stamp'] == 12.
    assert trace['body_observation_source_stamp'] == 5.
    assert trace['goal_observation_source_stamp'] == 3.
    # Trial-state IDs need not equal the unmodified component ID.
    assert not trace['factor_source_mismatches']
    broken = deepcopy(record)
    broken['visibility_factor']['source_ids']['selected_K'] = [999.,999.]
    mismatch = evaluation.control_summary([dict(gradient_evaluation=broken)])
    assert mismatch['selection_factors']['factor_source_identity_checks']['selected_K:mismatch'] == 1


def test_old_running_gradient_logs_remain_usable_with_missing_new_metrics_null(tmp_path):
    write_lines(tmp_path/'control.jsonl', [dict(control_mode='running_gradient',
        gradient_evaluation=dict(status='ok', gradient=[1.], computation_seconds=.2))])
    options = SimpleNamespace(run_dir=tmp_path, eta=None, gamma=None, near_threshold=.006, distance_samples=10)
    result = evaluation.evaluate(options)
    assert result['schema'] == 'probtf_running_gradient_evaluation_v1'
    assert result['control_worker']['gradient_computation_seconds']['mean'] == .2
    stats = result['control_worker']['selection_factors']['statistics']
    assert stats['log_view_factor'] is None
    assert stats['task_gradient_norm'] is None
    assert stats['selected_gradient_norm']['mean'] == 1.
    assert stats['view_probe_computation_seconds'] is None
    assert stats['task_and_FD_overhead_seconds_excluding_view_probes'] is None
    assert stats['task_computation_seconds_including_view_probes'] is None
    assert stats['view_probe_count'] is None
    assert result['camera_visibility']['raw_detector']['detected_frame_fraction'] is None
    assert result['camera_visibility']['physical_geometry']['configuration']['status'] == 'unavailable'
    assert not result['estimation_log']['scanned']


def test_scene_geometry_requires_run_hash_or_an_explicit_archived_configuration(tmp_path):
    config = tmp_path/'outside_scene.yaml'
    config.write_text('camera: {width: 100, height: 100, focal_length: 100}\n'
        'marker_board: {enabled: true, rows: 3, columns: 3, pitch: 0.09, marker_size: 0.072}\n')
    run = tmp_path/'run'
    run.mkdir()
    checksum = hashlib.sha256(config.read_bytes()).hexdigest()
    result = evaluation.scene_geometry(run, {}, config, [])
    assert result['status'] == 'available'
    assert result['board_sizes_m']['white_support'] == [.27,.27]
    assert result['board_sizes_m']['black_region'] == [.252,.252]
    wrong = dict(physical_model_source=dict(scene_config_sha256='unrelated-frozen-source'))
    assert evaluation.scene_geometry(run, wrong, config, [])['status'] == 'unavailable'
    matching = dict(physical_model_source=dict(scene_config_sha256=checksum))
    assert evaluation.scene_geometry(run, matching, config, [])['matches_run_hash']
