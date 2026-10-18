"""The simple HUD shows its normalized Gaussian and the actual action source."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from visualization_msgs.msg import Marker

from probtf_observation_fusion_demo.simple_display import simple_stiffness_marker


@pytest.fixture
def source():
    mean = np.log(np.arange(1., 7.)*100.)
    lower = np.eye(6)*.2
    lower[3, 0], lower[5, 1] = .03, -.04
    covariance = lower @ lower.T
    density = dict(density_id='simple_Gaussian:123456789',
        representation='single normalized full-covariance Gaussian in log K',
        mean=mean.tolist(), covariance=covariance.tolist(),
        normalization_constant=1., bayes_factor_surprise_enabled=False,
        evidence_computed=False, components=[dict(weight=1., log_weight=0.,
            x_est=mean.tolist(), P_est=covariance.tolist())])
    snapshot = SimpleNamespace(stamp=12.5, command=(.1, .2, .3, .4, .5, .6),
                               frame_id='root')
    proposal = dict(density_id=density['density_id'], source_stamp=snapshot.stamp,
        source_command=list(snapshot.command), source_prior_state_id='prior:2',
        body_posterior_state_ids=['canonical:4'], selected_logK=mean.tolist(),
        selected_K=np.exp(mean).tolist(), representative_from_prior=False)
    record = dict(source_stamp=snapshot.stamp, source_command=list(snapshot.command),
        source_prior_state_id='prior:2', body_posterior_state_ids=['canonical:4'],
        status='updated_single_gaussian', posterior=density, prior=deepcopy(density),
        command_proposal=proposal, update_applied=True)
    return snapshot, record


def test_actual_normalized_gaussian_text_retains_exact_hud_style_without_mutation(source):
    snapshot, record = source
    original = deepcopy(record)
    marker = simple_stiffness_marker(snapshot, record, [4])
    assert marker.header.frame_id == 'root' and marker.header.stamp.to_sec() == 12.5
    assert (marker.ns, marker.id, marker.type, marker.action) == (
        'estimated_stiffness', 0, Marker.TEXT_VIEW_FACING, Marker.ADD)
    assert (marker.pose.position.x, marker.pose.position.y, marker.pose.position.z) == (.72, 0., .70)
    assert (marker.pose.orientation.x, marker.pose.orientation.y,
            marker.pose.orientation.z, marker.pose.orientation.w) == (0., 0., 0., 1.)
    assert (marker.scale.x, marker.scale.y, marker.scale.z) == (0., 0., .023)
    assert (marker.color.r, marker.color.g, marker.color.b, marker.color.a) == (.10, .12, .19, 1.)
    assert 'sequential posterior approximation' in marker.text
    assert 'Representative exp(mean log K) [Nm/rad]: 100, 200, 300, 400, 500, 600' in marker.text
    assert 'sigma(log K): 0.2, 0.2, 0.2, 0.2022, 0.2, 0.204' in marker.text
    assert 'Normalized full-covariance log-K Gaussian' in marker.text
    assert 'BFS off | evidence not computed' in marker.text
    assert 'unknown' not in marker.text.lower()
    assert record == original


@pytest.mark.parametrize('status,from_prior,expected', [
    ('prior_retained', True, 'initial prior representative'),
    ('single_gaussian_projection_unavailable', False, 'update unavailable; prior retained'),
    ('prediction_with_retained_parameters', False, 'retained between observations'),
])
def test_initial_unavailable_and_retained_states_are_distinct(source, status, from_prior, expected):
    snapshot, record = source
    record.update(status=status, update_applied=False, prior_retained=True,
                  retained_prior=deepcopy(record['posterior']), posterior=None)
    record['command_proposal']['representative_from_prior'] = from_prior
    assert expected in simple_stiffness_marker(snapshot, record, [4]).text


def test_direct_action_keeps_old_estimate_time_separate_from_current_action_time(source):
    snapshot, record = source
    original_proposal = deepcopy(record['command_proposal'])
    original_proposal['source_stamp'] = 10.
    original_proposal['source_command'] = [.01]*6
    original_proposal['source_prior_state_id'] = 'prior:old'
    original_proposal['body_posterior_state_ids'] = ['canonical:old']
    action = dict(estimated_density_id=record['posterior']['density_id'],
        estimated_source_stamp=10., selected_K=deepcopy(original_proposal['selected_K']))
    correction = dict(density_id=record['posterior']['density_id'],
        source_stamp=snapshot.stamp, source_command=list(snapshot.command),
        body_posterior_state_ids=record['body_posterior_state_ids'],
        selected_K=deepcopy(original_proposal['selected_K']),
        action_stiffness=deepcopy(action),
        captured_parameter_packet=dict(action_stiffness=deepcopy(action)))
    record.update(command_proposal=original_proposal,
        action_command_proposal=deepcopy(source[1]['command_proposal']),
        command_correction=correction, action_stiffness=action,
        status='prediction_with_retained_parameters', captured_status='updated_single_gaussian')
    # The new proposal identifies the current body/action query, while the
    # retained estimator proposal still identifies its original acquisition.
    record['action_command_proposal'].update(source_stamp=snapshot.stamp,
        source_command=list(snapshot.command), source_prior_state_id='prior:2',
        body_posterior_state_ids=['canonical:4'])
    assert 'retained between observations' in simple_stiffness_marker(snapshot, record, [4]).text
    record['action_stiffness']['estimated_source_stamp'] = snapshot.stamp
    with pytest.raises(ValueError, match='original estimate'):
        simple_stiffness_marker(snapshot, record, [4])


@pytest.mark.parametrize('changed', ('density', 'time', 'command', 'body', 'prior', 'logK', 'K', 'component', 'normalization'))
def test_mismatched_gaussian_and_action_sources_are_rejected(source, changed):
    snapshot, record = source
    proposal = record['command_proposal']
    if changed == 'density':
        proposal['density_id'] = 'different'
    elif changed == 'time':
        proposal['source_stamp'] += 1.
    elif changed == 'command':
        proposal['source_command'][0] += .01
    elif changed == 'body':
        proposal['body_posterior_state_ids'] = ['other']
    elif changed == 'prior':
        proposal['source_prior_state_id'] = 'other'
    elif changed == 'logK':
        proposal['selected_logK'][0] += .01
    elif changed == 'K':
        proposal['selected_K'][0] += .01
    elif changed == 'component':
        record['posterior']['components'][0]['P_est'][0][0] += .01
    elif changed == 'normalization':
        record['posterior']['normalization_constant'] = None
    with pytest.raises(ValueError):
        simple_stiffness_marker(snapshot, record, [4])


def test_actual_needed_covariance_failure_is_explicit_and_small_variance_is_not_changed(source):
    snapshot, record = source
    tiny = np.eye(6)*1.e-20
    density = record['posterior']
    density['covariance'] = tiny.tolist()
    density['components'][0]['P_est'] = tiny.tolist()
    before = deepcopy(record)
    assert 'sigma(log K): 1e-10' in simple_stiffness_marker(snapshot, record, [4]).text
    assert record == before
    density['covariance'][0][0] = -1.e-20
    density['components'][0]['P_est'][0][0] = -1.e-20
    with pytest.raises(np.linalg.LinAlgError):
        simple_stiffness_marker(snapshot, record, [4])


def test_simple_publish_accepts_explicit_prediction_age_and_rejects_wrong_or_future_sources(source):
    from probtf_observation_fusion_demo.motion_display import MotionDisplay

    snapshot, record = source
    frame = SimpleNamespace(stamp=12.75, command=(.2, .3, .4, .5, .6, .7))
    execution = dict(execution_mode='simple', physical_stamp=frame.stamp,
        command=list(frame.command), displayed_prediction_stamp=snapshot.stamp,
        displayed_prediction_command=list(snapshot.command),
        displayed_prediction_age_s=frame.stamp-snapshot.stamp)
    display = MotionDisplay.__new__(MotionDisplay)
    reached = []

    class SourceGuardPassed(Exception):
        pass

    def capture_source(actual_snapshot, actual_record):
        reached.append((actual_snapshot, actual_record))
        raise SourceGuardPassed()

    # Stop immediately after the real source-clock guard, before ROS/render
    # setup. Both physical and candidate values retain their original source.
    display.stiffness_marker = capture_source
    with pytest.raises(SourceGuardPassed):
        display.publish(frame, 0, snapshot, 0, stiffness_record=record,
                        execution=execution, publish_scene=False)
    assert reached == [(snapshot, record)]
    for changed in ('age', 'future', 'physical_stamp', 'physical_command',
                    'prediction_stamp', 'prediction_command'):
        invalid, predicted = deepcopy(execution), deepcopy(snapshot)
        if changed == 'age':
            invalid['displayed_prediction_age_s'] = 0.
        elif changed == 'future':
            predicted.stamp = 13.
            invalid['displayed_prediction_stamp'] = predicted.stamp
            invalid['displayed_prediction_age_s'] = frame.stamp-predicted.stamp
        elif changed == 'physical_stamp':
            invalid['physical_stamp'] += .1
        elif changed == 'physical_command':
            invalid['command'][0] += .1
        elif changed == 'prediction_stamp':
            invalid['displayed_prediction_stamp'] += .1
        elif changed == 'prediction_command':
            invalid['displayed_prediction_command'][0] += .1
        with pytest.raises(AssertionError, match='source clocks are inconsistent'):
            display.publish(frame, 0, predicted, 0, stiffness_record=record,
                            execution=invalid, publish_scene=False)
    assert len(reached) == 1
