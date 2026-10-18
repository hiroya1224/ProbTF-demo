"""Optional request-level diagnostics preserve the exact inferred candidate."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rospkg
import yaml

from probtf_observation_fusion_demo.async_artifacts import capture_candidate_identity, identity_hash
from probtf_observation_fusion_demo.model import run, Stage, WeightedState
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.stiffness_body_consistency import (
    KBodyConsistencyEvaluator, _support_diagnostic,
)
from probtf_switch_press_demo.scene import write_robot


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def candidate(tmp_path_factory):
    """URDF-only prior and an explicit captured-theta fixture, no simulator K."""
    settings = yaml.safe_load((PACKAGE/'config/switch_press_scene.yaml').read_text())
    original = Path(rospkg.RosPack().get_path('symaware_grasp'))/'urdf/simple_six_dof_arm.urdf'
    path = write_robot(original, tmp_path_factory.mktemp('K_body_diagnostic')/'robot.urdf', settings)
    inference = MovingInference(path.read_text(), ())
    command = np.asarray(settings['initial_command'], dtype=float)
    state = inference.prior_at(command, 12.5, 1)
    # The no-observation model supplies the real downstream result schema.
    common = deepcopy(inference.common)
    common['update'] = dict(stamp=state.stamp, command=command.tolist(),
                            prior_version=state.version)
    result = run(common, (), prior=state, paths=inference.paths,
                 image=np.zeros((16, 16), dtype=np.uint8))
    result = replace(result, stages=(Stage('canonical', (WeightedState('canonical', 1., state),)),))
    body_ids = [item.state.state_id for item in result.stages[-1].components]
    action = dict(estimated_packet_id='K_fixture_packet', estimated_density_id='K_fixture_density',
        estimated_source_stamp=10., selected_K=[100.]*6,
        anchor_command=command.tolist(), anchor_stamp=state.stamp,
        auxiliary_equilibrium=command.tolist(), anchor_prior_state_id=state.state_id,
        anchor_body_state_ids=body_ids)
    action_json = json.dumps(action, sort_keys=True, separators=(',', ':'))
    reference = dict(estimated_packet_id=action['estimated_packet_id'], action_json=action_json,
        packet_id=hashlib.sha256(('direct-action\n'+action['estimated_packet_id']+'\n'+action_json).encode()).hexdigest())
    identity = capture_candidate_identity(result, parameter_identity=dict(
        estimated_packet_id=action['estimated_packet_id'], density_id=action['estimated_density_id'],
        selected_K=action['selected_K'], action_packet_ref=reference))
    return path, result, identity


def test_same_captured_command_has_rigid_modal_agreement_and_keeps_corrected_mean(candidate, monkeypatch):
    path, result, identity = candidate
    evaluator = KBodyConsistencyEvaluator(path)
    before = capture_candidate_identity(result, parameter_identity=identity['parameter'])
    covariance = result.stages[-1].components[0].state.covariance.copy()
    calls = []
    actual_fk = evaluator.robot._fk_update
    def fk(theta):
        calls.append(theta.copy())
        return actual_fk(theta)
    monkeypatch.setattr(evaluator.robot, '_fk_update', fk)
    record = evaluator.evaluate(result, identity, request=dict(request_id='request-1', index=2))
    assert record['status'] == 'available'
    assert record['candidate_id'] == identity_hash(identity)
    assert record['request_id'] == 'request-1'
    assert record['control_gating'] is record['source_laws_modified'] is False
    assert record['physical_truth_received'] is record['equilibrium_solve_performed'] is False
    assert len(calls) == record['FK_evaluations'] == 1
    assert record['K_source_age_s'] == 2.5
    assert record['density_id'] == identity['parameter']['density_id']
    component = record['components'][0]
    assert set(result.link_paths).issubset(component['global_links'])
    assert set(component['global_links']) == set(result.link_paths)|{'hand', 'camera'}
    for local in component['local_links'].values():
        assert local['modal_position_residual_m'] < 1.e-14
        assert local['modal_angular_residual_rad'] < 1.e-14
    for global_link in component['global_links'].values():
        assert global_link['modal_position_residual_m'] < 1.e-14
        assert global_link['modal_angular_residual_rad'] < 1.e-14
        assert global_link['source_state_id'] == component['state_id']
    # A mean correction is intentionally not forced back to deterministic FK.
    assert component['global_links']['hand']['corrected_mean_position_residual_m'] > 1.e-3
    assert component['body_joint']['full_cross_covariance_used']
    assert component['body_joint']['mahalanobis_squared'] < 1.e-20
    np.testing.assert_array_equal(result.stages[-1].components[0].state.covariance, covariance)
    assert capture_candidate_identity(result, parameter_identity=identity['parameter']) == before
    json.dumps(record, allow_nan=False)


def test_full_cross_covariance_changes_mahalanobis_without_modifying_it():
    covariance = np.array([[2., .8], [.8, 1.]])
    residual = np.array([1., -2.])
    before = covariance.copy()
    record = _support_diagnostic(residual, covariance)
    assert record['mahalanobis_squared'] == pytest.approx(residual @ np.linalg.solve(covariance, residual))
    assert record['mahalanobis_squared'] != pytest.approx(residual @ np.linalg.solve(np.diag(np.diag(covariance)), residual))
    np.testing.assert_array_equal(covariance, before)
    assert record['covariance_modified'] is False


def test_different_captured_equilibrium_is_compared_without_recentering_the_body(candidate):
    path, result, original = candidate
    identity = deepcopy(original)
    reference = identity['parameter']['action_packet_ref']
    action = json.loads(reference['action_json'])
    action['auxiliary_equilibrium'][2] += .03
    reference['action_json'] = json.dumps(action, sort_keys=True, separators=(',', ':'))
    reference['packet_id'] = hashlib.sha256(('direct-action\n'+reference['estimated_packet_id']+
        '\n'+reference['action_json']).encode()).hexdigest()
    before = result.stages[-1].components[0].state.state_id
    record = KBodyConsistencyEvaluator(path).evaluate(result, identity)
    assert record['status'] == 'available'
    component = record['components'][0]
    assert component['local_links']['link_3']['modal_angular_residual_rad'] == pytest.approx(.03)
    assert component['global_links']['hand']['modal_position_residual_m'] > .005
    assert component['body_joint']['mahalanobis_squared'] > 0.
    assert component['body_joint']['rank'] > 6
    assert result.stages[-1].components[0].state.state_id == before
    assert record['selected_K'] == original['parameter']['selected_K']
    assert record['source_command'] == original['command']


def test_support_mismatch_is_unavailable_not_fake_finite_density():
    record = _support_diagnostic(np.array([.2, .4]), np.diag([1., 0.]))
    assert record['status'] == 'unavailable'
    assert record['mahalanobis_squared'] is None
    assert record['projected_mahalanobis_squared'] == pytest.approx(.04)
    assert record['affine_support_residual_norm'] == .4
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize('field', ('anchor_command', 'anchor_stamp', 'anchor_body_state_ids', 'selected_K'))
def test_inconsistent_source_or_K_is_only_an_unavailable_diagnostic(candidate, field):
    path, result, original = candidate
    identity = deepcopy(original)
    action = json.loads(identity['parameter']['action_packet_ref']['action_json'])
    if field == 'anchor_command':
        action[field][0] += .1
    elif field == 'anchor_stamp':
        action[field] += .1
    elif field == 'anchor_body_state_ids':
        action[field] = ['another-body']
    else:
        action[field][0] += 1.
    identity['parameter']['action_packet_ref']['action_json'] = json.dumps(action)
    record = KBodyConsistencyEvaluator(path).evaluate(result, identity)
    assert record['status'] == 'unavailable'
    assert record['exception_type'] == 'ValueError'
    assert record['control_gating'] is False


def test_missing_packet_or_failed_optional_FK_cannot_refuse_control(candidate, monkeypatch):
    path, result, identity = candidate
    evaluator = KBodyConsistencyEvaluator(path)
    without = deepcopy(identity)
    without['parameter']['action_packet_ref'] = None
    missing = evaluator.evaluate(result, without)
    assert missing['status'] == 'unavailable'
    assert missing['reason'] == 'no captured direct-action K packet'
    def failed(*args):
        raise np.linalg.LinAlgError('optional FK audit failure')
    monkeypatch.setattr(evaluator.robot, '_fk_update', failed)
    record = evaluator.evaluate(result, identity)
    assert record['status'] == 'unavailable'
    assert record['reason'] == 'optional FK audit failure'
    assert record['exception_type'] == 'LinAlgError'
    assert record['control_gating'] is record['source_laws_modified'] is False


def test_optional_model_initialization_failure_is_reported_without_losing_source(candidate, monkeypatch):
    from probtf_observation_fusion_demo import stiffness_body_consistency as module
    path, result, identity = candidate
    def failed(*args, **kwargs):
        raise ValueError('diagnostic-only URDF model is unavailable')
    monkeypatch.setattr(module, 'RobotArm', failed)
    evaluator = KBodyConsistencyEvaluator(path)
    record = evaluator.evaluate(result, identity, request=dict(request_id='still-publish'))
    assert record['status'] == 'unavailable'
    assert record['initialization_error']['exception_type'] == 'ValueError'
    assert record['request_id'] == 'still-publish'
    assert record['candidate_id'] == identity_hash(identity)
    assert record['body_source_state_ids'] == [result.stages[-1].components[0].state.state_id]
    assert record['selected_K'] == identity['parameter']['selected_K']
    assert record['control_gating'] is record['source_laws_modified'] is False
