"""Posthoc deserialization preserves the exact saved canonical chart."""

import importlib
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from probtf_observation_fusion_demo.simple_control import joint_reach_score
from probtf_observation_fusion_demo.simple_inference import SimpleInference
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore
from probtf_observation_fusion_demo.joint_rotation_stiffness import JointRotationDensityRatio
from test_simple_inference import acquired, events


def serialized_state(result):
    state = result.stages[-1].components[0].state
    return json.loads(json.dumps(dict(
        acquisition_stamp=state.stamp, acquisition_command=list(state.command),
        body_state_version=state.version, body_component_id=state.component_id,
        body_chart_mean=state.mean.tolist(), body_covariance=state.covariance.tolist(),
        local_transforms=[dict(edge_id=edge.edge_id, translation=edge.translation.tolist(),
            rotation_wxyz=edge.rotation_wxyz.tolist(),
            bingham=None if edge.bingham is None else edge.bingham.tolist())
            for edge in state.transforms])))


def test_initial_and_later_bingham_candidates_keep_exact_ID_and_root_score_after_JSON_restore(
        acquired, events, monkeypatch):
    tools = Path(__file__).resolve().parents[1]/'validation/tools'
    monkeypatch.syspath_prepend(str(tools))
    summary = importlib.import_module('summarize_simple_reference')
    inference = SimpleInference(acquired[0], acquired[2])
    results = [inference.update_sensor_event(events('joint:0', 'joint', frame_index=0))]
    results.append(inference.update_sensor_event(events('image:0', 'image', frame_index=0)))
    results.append(inference.update_sensor_event(events('imu:1', 'imu', stamp=1000.01, frame_index=1)))
    metadata = json.loads(json.dumps(dict(hand_path=inference.paths['hand'],
        marker_path=inference.paths['marker'], goal_offset=inference.settings['control']['goal_offset'],
        root_frame=inference.settings['root_frame'])))
    for result in results:
        original = result.stages[-1].components[0].state
        restored = summary.reconstruct(serialized_state(result))
        assert original.state_id == restored.state_id
        np.testing.assert_array_equal(original.mean, restored.mean)
        np.testing.assert_array_equal(original.covariance, restored.covariance)
        for saved, actual in zip(original.transforms, restored.transforms):
            np.testing.assert_array_equal(saved.rotation_wxyz, actual.rotation_wxyz)
            np.testing.assert_array_equal(saved.translation, actual.translation)
            if saved.bingham is not None:
                np.testing.assert_array_equal(saved.bingham, actual.bingham)
        arguments = dict(hand_path=tuple(map(tuple, metadata['hand_path'])),
            marker_path=tuple(map(tuple, metadata['marker_path'])),
            goal_offset=metadata['goal_offset'], root_frame=metadata['root_frame'])
        before, source = joint_reach_score(original, **arguments)
        after, comparison = joint_reach_score(restored, **arguments)
        assert before == after
        assert source['joint_source_state_id'] == comparison['joint_source_state_id']
        assert source['difference_covariance'] == comparison['difference_covariance']


def test_uninformative_records_without_K_do_not_create_fake_K_source_metadata(
        acquired, monkeypatch, tmp_path):
    tools = Path(__file__).resolve().parents[1]/'validation/tools'
    monkeypatch.syspath_prepend(str(tools))
    summary = importlib.import_module('summarize_simple_reference')
    inference = SimpleInference(acquired[0], acquired[2])
    command = acquired[1].command
    output = tmp_path/'no_K_events'
    output.mkdir()
    updates, physical = [], []
    with SensorEventStore(output/'observations.sqlite3') as store:
        for index, missing in enumerate((None, dict(status='no_observation', reason='empty likelihood'))):
            stamp = 1000.+.1*index
            event = store.append('joint:'+str(index), 'joint', stamp, command,
                dict(time=.1*index, frame_index=index))
            state = inference.prior_at(command, stamp, index)
            row = serialized_state(SimpleNamespace(stages=(SimpleNamespace(
                components=(SimpleNamespace(state=state),)),)))
            row.update(event_id=event.event_id, sequence=event.sequence, kind='joint',
                frame_index=index, body_state_ids=[state.state_id], body_seconds=.001,
                K_seconds=0., simple_inference=dict(state_id=state.state_id, goal_observed=False),
                observation_ids=[], stiffness=missing,
                observation_eligible=False, observation_selected=False)
            updates.append(row)
            physical.append(dict(stamp=stamp, time=.1*index, command=list(command), tip=[0., 0., 0.],
                physical_error_m=1., reached=False, collision_checking_enabled=False))
    for name, records in (('simple_updates.jsonl', updates), ('physical_trajectory.jsonl', physical)):
        (output/name).write_text(''.join(json.dumps(record)+'\n' for record in records))
    report = summary.summarize(output)
    assert report['counts']['events_without_K_update_record'] == 2
    assert report['source_law_and_control_checks']['informative_observation_without_K_update_record'] == 0
    assert report['last_K_original_stamp'] is None
    assert report['first_selected_K'] is None and report['last_selected_K'] is None
    assert report['latest_manifest_available'] is False
    assert report['K_update_seconds_by_sensor_kind']['joint']['maximum'] == 0.
    coverage = report['observation_source_coverage_by_sensor_kind']['joint']
    assert report['archived_sensor_kind_counts'] == {'joint': 2}
    assert coverage['processed']['count'] == 2
    assert coverage['processed']['original_time_range'] == [1000., 1000.1]
    assert coverage['eligible']['count'] == coverage['selected']['count'] == coverage['fused']['count'] == 0
    assert report['source_law_and_control_checks']['fused_observation_not_selected'] == 0
    assert not report['body_bayes_factor_surprise']['available']
    assert report['body_bayes_factor_surprise']['counts']['records_absent'] == 2


def test_saved_body_change_audit_checks_actual_odds_transition_priors_and_canonical_K_numerator(
        acquired, events, monkeypatch):
    tools = Path(__file__).resolve().parents[1]/'validation/tools'
    monkeypatch.syspath_prepend(str(tools))
    summary = importlib.import_module('summarize_simple_reference')
    inference = SimpleInference(acquired[0], acquired[2], body_surprise_enabled=True)
    event = events('image:body_change_audit', 'image', frame_index=0)
    result = inference.update_sensor_event(event, event_index=0)
    state = result.stages[-1].components[0].state
    assert result.records['bayes_factor_surprise']['applied']
    ratio = JointRotationDensityRatio(result.frozen_body_branches, inference.body_ids,
        posterior_branch=result.projected_body_numerator)
    row = dict(event_id=event.event_id, sequence=event.sequence, kind='image',
        acquisition_stamp=event.stamp, acquisition_command=list(event.command),
        observation_ids=result.records['observation_ids'], observation_selected=True,
        body_state_ids=[state.state_id], body_component_id=state.component_id,
        body_surprise=result.records['bayes_factor_surprise'])
    stiffness = dict(body_marginal=ratio.record(), bayes_factor_surprise_enabled=False,
        evidence_computed=False, body_posterior_state_ids=[state.state_id],
        body_internal_approximation=result.records['internal_projection'])
    auditor = summary.BodySurpriseAudit()
    auditor.observe(row, stiffness)
    valid = auditor.record()
    assert valid['counts']['enabled'] == valid['counts']['applied'] == 1
    assert all(value == 0 for value in valid['checks'].values())
    diagnostics = valid['numerical_diagnostics']
    assert diagnostics['rho_error']['max_absolute'] == 0.
    assert diagnostics['prior_odds_error']['max_absolute'] == 0.
    assert diagnostics['posterior_log_odds_residual']['max_absolute'] < 2.e-9
    assert diagnostics['posterior_log_normalization_residual']['max_absolute'] < 2.e-12
    assert valid['source_event_examples'][0]['canonical_state_id'] == state.state_id

    wrong = deepcopy(row)
    wrong['observation_ids'] = []
    wrong['observation_selected'] = False
    wrong['body_surprise']['change_probability'] = .5
    wrong['body_surprise']['posterior_keep_log_probability'] = np.log(.4)
    wrong['body_surprise']['posterior_change_log_probability'] = np.log(.4)
    wrong_stiffness = deepcopy(stiffness)
    wrong_stiffness['body_internal_approximation']['internal_state_id'] = 'foreign_canonical_law'
    wrong_stiffness['bayes_factor_surprise_enabled'] = True
    bad = summary.BodySurpriseAudit()
    bad.observe(wrong, wrong_stiffness)
    diagnostic = bad.record()
    assert diagnostic['checks']['applied_without_fused_observation'] == 1
    assert diagnostic['checks']['applied_to_unselected_observation'] == 1
    assert diagnostic['checks']['applied_K_canonical_numerator_ID_mismatch'] == 1
    assert diagnostic['checks']['applied_K_BFS_or_evidence_enabled'] == 1
    # Finite inconsistency is reported as a residual; this read-only audit
    # never turns a numerical tolerance into an inference stopping gate.
    assert diagnostic['numerical_diagnostics']['rho_error']['max_absolute'] == .25
    assert diagnostic['numerical_diagnostics']['posterior_log_normalization_residual']['max_absolute'] > .2

    no_change = summary.BodySurpriseAudit()
    for enabled in (False, True):
        skipped = dict(row, observation_ids=[], body_surprise=dict(enabled=enabled,
            applied=False, change_probability=.25, prior_change_odds=1./3.))
        no_change.observe(skipped, None)
    skipped = no_change.record()
    assert skipped['counts']['applied'] == 0
    assert all(value == 0 for value in skipped['checks'].values())


def test_plan_response_separates_mean_covariance_descent_and_actual_motion(monkeypatch):
    tools = Path(__file__).resolve().parents[1]/'validation/tools'
    monkeypatch.syspath_prepend(str(tools))
    summary = importlib.import_module('summarize_simple_reference')
    before = dict(hand_mean=[0., 0., 0.], mean_difference=[-1., 0., 0.],
        difference_covariance=np.eye(3).tolist())
    after = dict(hand_mean=[.1, 0., 0.], mean_difference=[-.9, 0., 0.],
        difference_covariance=(.5*np.eye(3)).tolist())
    row = dict(adoption_stamp=10., duration=.5, diagnostics=dict(goal_observed=True,
        S_before=4., S_after=2.31, source_score=before, selected_score=after,
        step_norm_rad=.02, selected_command_action=dict(active_joint_limits=[False, True])))
    physical = {0: dict(stamp=10., tip=[0., 0., 0.], physical_error_m=1.),
        1: dict(stamp=10.51, tip=[-.02, 0., 0.], physical_error_m=1.02)}
    result = summary.finite_plan_response(row, physical, [10., 10.51],
        {frame['stamp']: frame for frame in physical.values()})
    assert result['predicted_mean_squared_distance_change_m2'] == pytest.approx(-.19)
    assert result['predicted_covariance_trace_change_m2'] == -1.5
    assert result['predicted_mean_distance_reduction_m'] == pytest.approx(.1)
    assert result['actual_physical_distance_reduction_m'] == pytest.approx(-.02)
    assert result['predicted_actual_hand_displacement_cosine'] == -1.
    assert result['predicted_hand_goal_alignment_cosine'] == 1.
    assert result['actual_hand_estimated_goal_alignment_cosine'] == -1.
    assert result['predicted_hand_progress_along_estimated_goal_m'] == pytest.approx(.1)
    assert result['actual_hand_progress_along_estimated_goal_m'] == pytest.approx(-.02)
    assert result['physical_sample_overshoot_s'] == pytest.approx(.01)
