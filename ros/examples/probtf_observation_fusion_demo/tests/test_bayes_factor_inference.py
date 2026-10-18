"""Analytic predictive evidence and the actual observed-data update boundary."""

from dataclasses import replace
import io
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, logsumexp
from scipy.stats import multivariate_normal
import yaml
from visualization_msgs.msg import MarkerArray

from probtf.spherical_law.first_order import BinghamTransform, JointTransformState
from probtf_observation_fusion_demo.display import make_markers, snapshot_record, verify_wire_identity
from probtf_observation_fusion_demo.model import WeightedState, run
from probtf_observation_fusion_demo.moving_control import control_components
from probtf_observation_fusion_demo.moving_inference import MovingInference
from legacy_marker import legacy_marker_scene
from probtf_observation_fusion_demo.node import build_snapshot
from probtf_observation_fusion_demo.sensors import ADAPTERS
from test_sequential_inference import _analytic_sensor, _scalar_inputs


PACKAGE = Path(__file__).resolve().parents[1]


def test_finite_joint_evidence_odds_roundoff_is_diagnostic_and_does_not_change_the_law(
        sensor_problem, monkeypatch):
    from probtf_observation_fusion_demo.joint_batch_fusion import add_log_evidence_parts
    scene, inputs, cases = sensor_problem
    reference = MovingInference(scene.robot_description, cases['imu'], config_path=scene.legacy_inference_config)
    expected = reference.update(inputs, 0)
    original_run = MovingInference._run_observation_update
    discrepancy = 1e-4
    injections = []
    def rounded_evidence(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        if result.sensor_posteriors and kwargs.get('joint_evidence'):
            assert result.records['variational_observation_update']['method'] == 'variational_moments'
            # Perturb only a redundant evidence diagnostic. All actual
            # posterior components, masses, covariances and IDs are intact.
            evidence = result.records['joint_observation_evidence']['parents'][1]
            evidence['log_evidence'] += discrepancy
            evidence['log_evidence_parts'] = list(add_log_evidence_parts(
                evidence['log_evidence_parts'], (discrepancy,)))
            injections.append(result)
        return result
    monkeypatch.setattr(MovingInference, '_run_observation_update', rounded_evidence)
    inference = MovingInference(scene.robot_description, cases['imu'], config_path=scene.legacy_inference_config)
    actual = inference.update(inputs, 0)
    assert len(injections) == 1
    diagnostic = actual.records['bayes_factor_surprise']['joint_evidence_log_odds_consistency']
    assert not diagnostic['verified']
    assert not diagnostic['update_gate']
    assert diagnostic['residual'] == pytest.approx(-discrepancy, abs=1e-10)
    assert diagnostic['warning']
    for actual_component, reference_component in zip(actual.stages[-1].components, expected.stages[-1].components):
        assert actual_component.state.state_id == reference_component.state.state_id
        assert actual_component.weight == reference_component.weight
        assert actual_component.log_weight == reference_component.log_weight
        np.testing.assert_array_equal(actual_component.state.covariance, reference_component.state.covariance)
    # A finite diagnostic discrepancy does not stall the next fresh event.
    next_inputs = replace(inputs, stamp=1000.05, time=.05)
    next_result = inference.update(next_inputs, 1)
    assert next_result.records['bayes_factor_surprise']['applied']
    assert inference.last_result is next_result


def test_nonfinite_joint_evidence_odds_remains_an_actual_failure(sensor_problem, monkeypatch):
    scene, inputs, cases = sensor_problem
    original_run = MovingInference._run_observation_update
    injections = []
    def invalid_evidence(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        if result.sensor_posteriors and kwargs.get('joint_evidence'):
            assert result.records['variational_observation_update']['method'] == 'variational_moments'
            evidence = result.records['joint_observation_evidence']['parents'][1]
            evidence['log_evidence'] = np.nan
            evidence['log_evidence_parts'] = [np.nan, 0.]
            injections.append(result)
        return result
    monkeypatch.setattr(MovingInference, '_run_observation_update', invalid_evidence)
    inference = MovingInference(scene.robot_description, cases['imu'], config_path=scene.legacy_inference_config)
    with pytest.raises(ValueError, match='must have finite log masses'):
        inference.update(inputs, 0)
    assert len(injections) == 1


@pytest.mark.parametrize('values', [(.0, .0), (.4, 1.2), (8., 9.)])
def test_joint_bayes_factor_uses_normalized_correlated_sensor_evidence(values, monkeypatch):
    common, paths, original, observations = _scalar_inputs()
    priors = tuple(replace(item, weight=weight, log_weight=np.log(weight))
                   for item, weight in zip(original, (.75, .25)))
    observations = tuple(dict(item, value=value) for item, value in zip(observations, values))
    monkeypatch.setitem(ADAPTERS, 'analytic_test', _analytic_sensor)
    image = np.zeros((2, 2), np.uint8)
    result = run(common, observations, image, paths=paths, prior_components=priors,
                 joint_evidence=True)
    evidence = result.records['joint_observation_evidence']
    expected = []
    marginal_product = []
    for parent in priors:
        mean = parent.state.transforms[0].translation[0]
        variance = parent.state.covariance[0, 0]
        covariance = variance*np.ones((2, 2))+np.diag([.25, .4])
        expected.append(multivariate_normal.logpdf(values, np.full(2, mean), covariance))
        marginal_product.append(sum(multivariate_normal.logpdf(value, mean, variance+noise)
                                    for value, noise in zip(values, (.25, .4))))
    np.testing.assert_allclose([item['log_evidence'] for item in evidence['parents']], expected,
                               rtol=1e-12, atol=1e-11)
    assert evidence['log_evidence'] == pytest.approx(logsumexp(np.log([.75, .25])+expected), abs=1e-11)
    assert not np.isclose(expected[1]-expected[0], marginal_product[1]-marginal_product[0])
    actual_reset_log_mass = logsumexp([item.log_weight
        for item in result.frozen_body_branches[1].posterior_components])
    gamma = expit(expected[1]-expected[0]-np.log(3.))
    assert np.exp(actual_reset_log_mass) == pytest.approx(gamma, abs=1e-12)
    reordered = run(common, observations[::-1]+observations[:1], image, paths=paths,
                    prior_components=priors, joint_evidence=True)
    assert reordered.records['joint_observation_evidence'] == evidence


@pytest.fixture(scope='module')
def sensor_problem(tmp_path_factory):
    scene = legacy_marker_scene(tmp_path_factory.mktemp('bayes_factor_inputs'))
    inputs = scene.sensor_inputs(scene.step(0., 1000.))
    cases = {case: yaml.safe_load((PACKAGE/'config'/(case+'.yaml')).read_text())['observations']
             for case in ('no_sensors', 'imu', 'marker', 'imu_marker')}
    return scene, inputs, cases


@pytest.mark.parametrize('case', ['no_sensors', 'imu', 'marker', 'imu_marker'])
def test_actual_sensor_update_publishes_the_same_change_record_and_complete_candidate_laws(
        sensor_problem, case, monkeypatch):
    scene, inputs, cases = sensor_problem
    def prohibited(*args, **kwargs):
        raise AssertionError('Bayes-factor inference must not sample.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, prohibited)
    inference = MovingInference(scene.robot_description, cases[case], config_path=scene.legacy_inference_config)
    result = inference.update(inputs, 0)
    record = result.records['bayes_factor_surprise']
    assert record['change_probability'] == .25
    assert record['prior_change_odds'] == pytest.approx(1./3.)
    assert record['applied'] == (case != 'no_sensors')
    if record['applied']:
        assert len(result.frozen_body_branches) == 2
        assert result.projected_body_numerator is not None
        assert record['posterior_change_probability'] == pytest.approx(.25, abs=1e-10)
        assert record['log_bayes_factor'] == pytest.approx(0., abs=1e-10)
        if 'marker' in case:
            native_count = len(next(item.detector_result for item in result.local_estimates
                if item.detector_result is not None).record.distribution.components)
            assert len(result.stages[-1].goal_components) == native_count
            assert result.records['fresh_goal_construction']['native_component_count'] == native_count
        else:
            assert len(result.stages[-1].goal_components) == 2
        planes = [item.diagnostics['tangent_basis'] for item in result.local_estimates
                  if item.posterior is not None and item.diagnostics['sensor'] == 'imu']
        if planes:
            np.testing.assert_array_equal(planes[0], planes[1])
    else:
        assert record['log_bayes_factor'] is None and record['posterior_change_probability'] is None
        assert result.stages[-1].components[0].state is result.prior
    body, goals = control_components(result)
    assert len(body) == 1
    assert record['internal_state_id'] == body[0].state.state_id
    assert record['goal_source_state_ids'] == [item.state.state_id for item in goals]
    snapshot = build_snapshot(result, result.stages[-1], inference.common)
    assert snapshot.surprise_record is record
    assert snapshot_record(snapshot)['bayes_factor_surprise'] is record
    display = inference.settings['display']
    messages = make_markers(snapshot, seed=display['seed'], point_size=display['point_size'])
    buffer = io.BytesIO(); messages.serialize(buffer)
    actual = MarkerArray().deserialize(buffer.getvalue())
    verify_wire_identity(snapshot, actual, seed=display['seed'], point_size=display['point_size'])
    assert inference.update(inputs, 0) is result


def test_missing_image_holds_goal_and_has_no_bayes_factor_trial(sensor_problem):
    scene, inputs, cases = sensor_problem
    inference = MovingInference(scene.robot_description, cases['marker'], config_path=scene.legacy_inference_config)
    first = inference.update(inputs, 0)
    missing = replace(inputs, image=np.full_like(inputs.image, 255), stamp=1000.05, time=.05)
    second = inference.update(missing, 1)
    record = second.records['bayes_factor_surprise']
    assert not record['applied'] and record['posterior_change_probability'] is None
    assert second.stages[-1].components[0].state is second.prior
    assert second.records['fixed_goal_memory'] == first.records['fixed_goal_memory']
    for old, held in zip(first.stages[-1].goal_components, second.stages[-1].goal_components):
        assert old.weight == held.weight and old.log_weight == held.log_weight
        np.testing.assert_array_equal(old.state.covariance, held.state.covariance)


def test_imu_conflict_allows_expansion_without_moving_the_fixed_goal(sensor_problem):
    scene, inputs, cases = sensor_problem
    inference = MovingInference(scene.robot_description, cases['imu_marker'], config_path=scene.legacy_inference_config)
    first = inference.update(inputs, 0)
    memory = first.records['fixed_goal_memory']
    previous = first
    for index in range(1, 7):
        gravity = inputs.gravity_in_sensor.copy()
        if index == 6:
            gravity = gravity+np.array([.16, -.11, .08])
        packet = replace(inputs, image=np.full_like(inputs.image, 255),
                         gravity_in_sensor=gravity, stamp=1000.+index*.05, time=index*.05)
        current = inference.update(packet, index)
        assert current.records['fixed_goal_memory'] == memory
        assert current.records['bayes_factor_surprise']['applied']
        assert len(current.stages[-1].goal_components) == len(first.stages[-1].goal_components)
        for old, held in zip(previous.stages[-1].goal_components, current.stages[-1].goal_components):
            assert old.weight == held.weight and old.log_weight == held.log_weight
            assert old.state.transforms[-1] is held.state.transforms[-1]
            np.testing.assert_array_equal(old.state.covariance[60:, 60:], held.state.covariance[60:, 60:])
        if index == 5:
            before_conflict = current
        previous = current
    surprise = current.records['bayes_factor_surprise']
    assert surprise['posterior_change_probability'] > .9
    assert surprise['posterior_change_probability'] > before_conflict.records['bayes_factor_surprise']['posterior_change_probability']
    change = current.stages[-1].components[0].state.covariance-before_conflict.stages[-1].components[0].state.covariance
    assert np.linalg.eigvalsh(change)[-1] > 1e-6
