"""Posthoc audit handles gzip and unfinished trials without changing data."""

import gzip
import json

from uncertainty_ablation_audit import audit, rows


def write_lines(path, values):
    path.write_text(''.join(json.dumps(value)+'\n' for value in values))


def test_gzip_stream_has_the_original_records(tmp_path):
    path = tmp_path/'records.jsonl'
    records = [dict(id=1, stamp=.2), dict(id=2, stamp=.7)]
    with gzip.open(str(path)+'.gz', 'wt') as stream:
        stream.write(''.join(json.dumps(record)+'\n' for record in records))
    assert list(rows(path)) == records


def test_partial_trial_keeps_last_recorded_time_and_exact_source(tmp_path):
    run = tmp_path/'marker_partial'
    run.mkdir()
    command = [0.]*6
    (run/'diagnostic_setup.json').write_text(json.dumps(dict(duration_s=180.,
        physical_truth_received=True, intervention='oracle_marker')))
    estimation = []
    for event_id, predicted in [('imu:00000001', [.1, -.2]),
            ('image:00000001', [.003, .004, 0., 0., 0., .02])]:
        observation = dict(predicted_mean=predicted, observed_mean=[0.]*len(predicted))
        branch = dict(final_gradient_inf=5., finite_nonstationary_iterate_used=True,
            source_evaluator_diagnostics=dict(propagated_observations=[observation]))
        estimation.append(dict(sensor_event=dict(event_id=event_id),
            variational_observation_update=dict(branch_diagnostics=[branch])))
    write_lines(run/'estimation.jsonl', estimation)
    write_lines(run/'physical_trajectory.jsonl', [dict(time=0., stamp=1000.,
        command=command, actual_joints=command, physical_error_m=.2,
        reached=False, tip=[.2, 0., 0.]), dict(time=2., stamp=1002.,
        command=command, actual_joints=command, physical_error_m=.1,
        reached=False, tip=[.1, 0., 0.])])
    write_lines(run/'diagnostic_control_truth.jsonl', [dict(goal_position=[0., 0., 0.])])
    queries = [dict(frame_label=name, axis='origin', mean=mean,
        covariance=[[1e-6, 0., 0.], [0., 1e-6, 0.], [0., 0., 1e-6]])
        for name,mean in [('hand', [.12, 0., 0.]), ('switch_goal', [.02, 0., 0.])]]
    write_lines(run/'control.jsonl', [dict(index=1, request_id=7,
        controller=dict(source_stamp=1002., source_command=command,
            status='ok', body_observation_source_stamp=1001.),
        snapshot=dict(queries=queries))])
    before = (run/'physical_trajectory.jsonl').read_bytes()
    result = audit(run)
    assert result['partial_run'] is True
    assert result['completed_physical_cutoff_present'] is False
    assert result['requested_duration_s'] == 180.
    assert result['observed_duration_s'] == 2.
    assert result['last_recorded_physical_frame']['stamp'] == 1002.
    assert all(row['actual_source_t']==2. for row in result['selected_control_samples'])
    assert result['posthoc_database_snapshot']['available'] is False
    assert result['audit_truth_used_for_posthoc_scoring_only'] is True
    assert result['audit_changed_trial_law'] is False
    assert result['trial_truth_assisted'] is True
    assert result['trial_intervention'] == 'oracle_marker'
    assert result['final_fitted_position_residual_m']['n'] == 1
    assert result['final_fitted_position_residual_m']['max'] == .005
    assert result['final_fitted_gravity_tangent2_residual_norm']['n'] == 1
    assert result['variational_fit_diagnostics'][0]['sensor'] == 'marker'
    assert result['variational_gravity_fit_diagnostics'][0]['sensor'] == 'imu'
    assert (run/'physical_trajectory.jsonl').read_bytes() == before


def test_missing_trial_does_not_fabricate_a_completed_run(tmp_path):
    result = audit(tmp_path/'missing')
    assert result['partial_run'] is True
    assert result['observed_duration_s'] is None
    assert result['last_recorded_physical_frame'] is None
    assert result['selected_control_samples'] == []
