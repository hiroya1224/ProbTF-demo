"""Analytic checks of retained joint laws, global hypothesis masses and dropout."""
from dataclasses import replace
from pathlib import Path
import math

import numpy as np
import pytest
from scipy.special import logsumexp
import yaml

from probtf.spherical_law.first_order import BinghamTransform, JointTransformState, linear_query, point_query, reexpress_state
from probtf.spherical_law.fusion import PosteriorComponent, SensorPosterior, condition_gaussian, freeze_query
from probtf_observation_fusion_demo.model import WeightedState, component_log_weight, run
from probtf_observation_fusion_demo.moving_control import MovingController, command_state
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.sensors import ADAPTERS, LocalEstimate

PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def moving(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp('retained_body_laws'))
    frame = scene.step(0., 1000.)
    cases = {name: yaml.safe_load((PACKAGE/'config'/f'{name}.yaml').read_text())['observations']
             for name in ('no_sensors', 'imu', 'marker', 'imu_marker')}
    return scene, scene.sensor_inputs(frame), cases


def test_new_imu_information_accumulates_in_upstream_laws_without_mc(moving, monkeypatch):
    scene, inputs, cases = moving
    def prohibited(*args, **kwargs):
        raise AssertionError('No Monte Carlo is allowed in any inference step.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, prohibited)
    inference = MovingInference(scene.robot_description, cases['imu'])
    first = inference.update(inputs, 0)
    second = inference.update(replace(inputs, stamp=1000.05, time=.05), 1)
    previous = first.stages[-1].components[0].state
    np.testing.assert_array_equal(second.prior.covariance, previous.covariance)
    current = second.stages[-1].components[0].state
    for edge in ('link_1', 'link_2', 'link_3'):
        j = previous.edge_ids.index(edge)
        block = slice(6*j+3, 6*j+6)
        assert np.trace(current.covariance[block, block]) < np.trace(previous.covariance[block, block])
    # Identical tick retransmission cannot add that information twice.
    assert inference.update(replace(inputs, stamp=1000.05, time=.05), 1) is second
    changed = replace(inputs, stamp=1000.05, time=.05, gravity_in_sensor=inputs.gravity_in_sensor+[.01, 0., 0.])
    with pytest.raises(ValueError, match='different observation'):
        inference.update(changed, 1)


def test_marker_loss_retains_single_internal_memory_and_all_goal_modes(moving):
    scene, inputs, cases = moving
    inference = MovingInference(scene.robot_description, cases['marker'])
    first = inference.update(inputs, 0)
    assert len(first.stages[-1].components) == 1
    assert len(first.stages[-1].goal_components) >= 2
    command = inputs.command.copy(); command[1] += .025
    missing = replace(inputs, command=command, stamp=1000.05, time=.05,
                      image=np.full_like(inputs.image, 255))
    second = inference.update(missing, 1)
    assert second.records['observation_ids'] == []
    assert second.records['valid_observation_count'] == 0
    assert len(second.stages[-1].components) == len(first.stages[-1].components)
    assert len(second.stages[-1].goal_components) == len(first.stages[-1].goal_components)
    for old, prior, posterior, provenance in zip(first.stages[-1].components,
            second.stages[0].components, second.stages[-1].components, second.records['prediction']):
        expected = command_state(old.state, command, inference.body_ids, inference.actions)
        assert posterior.state is prior.state
        assert posterior.weight == old.weight
        assert posterior.log_weight == old.log_weight
        np.testing.assert_array_equal(prior.state.covariance, expected.covariance)
        marker = prior.state.edge_ids.index('marker')
        np.testing.assert_array_equal(prior.state.transforms[marker].translation,
                                      old.state.transforms[marker].translation)
        np.testing.assert_array_equal(prior.state.transforms[marker].bingham,
                                      old.state.transforms[marker].bingham)
        assert np.linalg.norm(prior.state.covariance[:36, 6*marker:6*marker+6]) > 1e-6
        assert provenance['source_state_id'] == old.state.state_id
        assert provenance['transported_state_id'] == prior.state.state_id
        assert prior.state.stamp == 1000.05 and prior.state.command == tuple(command)
    for old, retained in zip(first.stages[-1].goal_components, second.stages[-1].goal_components):
        expected = command_state(old.state, command, inference.body_ids, inference.actions)
        assert retained.weight == old.weight and retained.log_weight == old.log_weight
        np.testing.assert_array_equal(retained.state.covariance, expected.covariance)
        marker = expected.edge_ids.index('marker')
        np.testing.assert_array_equal(retained.state.transforms[marker].translation,
                                      expected.transforms[marker].translation)
        np.testing.assert_array_equal(retained.state.transforms[marker].bingham,
                                      expected.transforms[marker].bingham)


def test_internal_state_stays_single_and_current_goal_mixture_does_not_form_history_products(moving):
    scene, inputs, cases = moving
    inference = MovingInference(scene.robot_description, cases['marker'])
    first = inference.update(inputs, 0)
    second = inference.update(replace(inputs, stamp=1000.05, time=.05), 1)
    initial_count = len(first.stages[-1].goal_components)
    assert initial_count >= 2
    assert len(first.stages[-1].components) == len(second.stages[-1].components) == 1
    assert len(second.frozen_body_branches) == 2
    assert [branch.weight for branch in second.frozen_body_branches] == [.75, .25]
    assert len(second.stages[-1].goal_components) == initial_count
    assert len(second.stages[-2].components) == initial_count
    native = next(stage for stage in second.stages if stage.label == 'fused_change_candidates')
    displayed = {item.label: item for item in native.components}
    logs = []
    for branch in second.frozen_body_branches:
        assert branch.reference_state.component_id == branch.label
        assert len(branch.posterior_components) == initial_count//2
        for raw in branch.posterior_components:
            actual = displayed[raw.label]
            expected = reexpress_state(branch.reference_state, raw.distribution.mean,
                raw.distribution.covariance, version=actual.state.version, component_id=actual.state.component_id)
            assert actual.state.state_id == expected.state_id
            assert actual.log_weight == raw.log_weight
            logs.append(raw.log_weight)
    assert logsumexp(logs) == pytest.approx(0., abs=1e-12)
    numerator = second.projected_body_numerator
    assert numerator.reference_state is second.frozen_body_branches[0].reference_state
    assert numerator.weight == 1. and numerator.log_weight == 0.
    assert len(numerator.posterior_components) == 1
    projected = numerator.posterior_components[0]
    internal = second.stages[-1].components[0]
    expected = reexpress_state(numerator.reference_state, projected.distribution.mean,
        projected.distribution.covariance, version=internal.state.version,
        component_id=internal.state.component_id)
    assert projected.label == internal.label and projected.weight == internal.weight == 1.
    assert expected.state_id == internal.state.state_id
    assert second.records['valid_observation_count'] == 1
    assert len(second.sensor_posteriors) == 2
    for goal, raw in zip(second.stages[-1].goal_components, second.stages[-2].components):
        assert goal.weight == raw.weight and goal.log_weight == raw.log_weight
        internal = second.stages[-1].components[0].state
        np.testing.assert_array_equal(goal.state.covariance[:60, :60], internal.covariance[:60, :60])
    assert second.records['internal_projection']['source_component_count'] == initial_count


def test_disabled_change_model_retains_the_legacy_single_prior_update(moving, tmp_path):
    scene, inputs, cases = moving
    settings = yaml.safe_load((PACKAGE/'config'/'moving_estimation.yaml').read_text())
    settings['bayes_factor_surprise']['enabled'] = False
    path = tmp_path/'disabled.yaml'
    path.write_text(yaml.safe_dump(settings))
    inference = MovingInference(scene.robot_description, cases['marker'], config_path=path)
    first = inference.update(inputs, 0)
    second = inference.update(replace(inputs, stamp=1000.05, time=.05), 1)
    assert not second.records['bayes_factor_surprise']['enabled']
    assert not second.records['bayes_factor_surprise']['applied']
    assert len(second.stages[0].components) == len(second.frozen_body_branches) == 1
    assert second.projected_body_numerator is None
    assert len(second.sensor_posteriors) == 1
    assert len(second.stages[-1].goal_components) == len(first.stages[-1].goal_components) == 2
    np.testing.assert_array_equal(second.prior.covariance,
                                  first.stages[-1].components[0].state.covariance)


def _scalar_inputs():
    common = dict(root_frame='base', update={}, prior=dict(independence='initial product supplied',
        marker=dict(status='unknown')), known_geometry={}, camera={}, marker={})
    path = (('edge', False),)
    paths = dict(links={'edge': path}, hand=path, camera=path, marker=path, camera_marker=path)
    priors = []
    for label, mean, variance, weight in [('left', 0., .8, .3), ('right', 2., 1.1, .7)]:
        edge = BinghamTransform('edge', [mean, 0., 0.], rotation_wxyz=[1., 0., 0., 0.],
            translation_covariance=np.diag([variance, 0., 0.]))
        state = JointTransformState.from_independent((edge,), version=7, stamp=5., command=(.1,), component_id=label)
        priors.append(WeightedState(label, weight, state, math.log(weight)))
    observations = tuple(dict(sensor='analytic_test', sensor_id=name, observation_id='obs:'+name,
        value=value, variance=variance) for name, value, variance in [('a', .4, .25), ('b', 1.2, .4)])
    return common, paths, tuple(priors), observations


def _analytic_sensor(state, observation, paths, common, image):
    query = linear_query(point_query(state, paths['hand']), [[1., 0., 0.]])
    frozen = freeze_query(state, query)
    law, evidence = condition_gaussian(frozen.distribution, [observation['value']], [[observation['variance']]])
    posterior = SensorPosterior(frozen, observation['sensor_id'], observation['observation_id'],
        (PosteriorComponent('observed', law),))
    return LocalEstimate(posterior, {}, log_evidence=evidence, component_log_weights=(0.,))


def test_global_sensor_evidence_and_common_prior_correction_match_analytic_mixture(monkeypatch):
    common, paths, priors, observations = _scalar_inputs()
    monkeypatch.setitem(ADAPTERS, 'analytic_test', _analytic_sensor)
    image = np.zeros((2, 2), np.uint8)
    result = run(common, observations, image, paths=paths, prior_components=priors)
    reverse = run(common, tuple(reversed(observations)), image, paths=paths, prior_components=priors)
    log_masses = []
    for parent, updated in zip(priors, result.stages[-1].components):
        mean = parent.state.transforms[0].translation[0]
        variance = parent.state.covariance[0, 0]
        covariance = variance*np.ones((2, 2))+np.diag([.25, .4])
        error = np.array([.4, 1.2])-mean
        log_evidence = -.5*(2*np.log(2*np.pi)+np.linalg.slogdet(covariance)[1]
                            +error@np.linalg.solve(covariance, error))
        log_masses.append(parent.log_weight+log_evidence)
        expected_variance = 1./(1./variance+1./.25+1./.4)
        expected_mean = expected_variance*(mean/variance+.4/.25+1.2/.4)
        np.testing.assert_allclose(updated.state.transforms[0].translation[0], expected_mean, atol=1e-12)
        np.testing.assert_allclose(updated.state.covariance[0, 0], expected_variance, atol=1e-12)
    expected_logs = np.array(log_masses)-logsumexp(log_masses)
    np.testing.assert_allclose([item.log_weight for item in result.stages[-1].components], expected_logs, atol=1e-12)
    assert result.records['valid_observation_count'] == 2
    for actual, swapped in zip(result.stages[-1].components, reverse.stages[-1].components):
        assert actual.label == swapped.label
        assert actual.log_weight == swapped.log_weight
        np.testing.assert_array_equal(actual.state.covariance, swapped.state.covariance)
        np.testing.assert_array_equal(actual.state.transforms[0].translation, swapped.state.transforms[0].translation)


def test_retained_mixture_over_five_times_matches_batch_gaussian_information(monkeypatch):
    """A temporal test must verify accumulated evidence, not just smaller P."""
    common, paths, priors, _ = _scalar_inputs()
    initial = priors
    monkeypatch.setitem(ADAPTERS, 'analytic_test', _analytic_sensor)
    values = np.array([.4, .7, .3, 1.0, .6])
    variances = np.array([.25, .4, .2, .3, .35])
    image = np.zeros((2, 2), np.uint8)
    for index, (value, variance) in enumerate(zip(values, variances)):
        priors = tuple(replace(item, state=replace(item.state, stamp=5.+index,
                        version=3*index+1, component_id=item.label)) for item in priors)
        observation = dict(sensor='analytic_test', sensor_id='sensor',
            observation_id='time:'+str(index), value=float(value), variance=float(variance))
        result = run(common, (observation,), image, paths=paths, prior_components=priors)
        priors = result.stages[-1].components
        log_masses = []
        for original, current in zip(initial, priors):
            mean0 = original.state.transforms[0].translation[0]
            variance0 = original.state.covariance[0, 0]
            used_variance, used_values = variances[:index+1], values[:index+1]
            expected_variance = 1./(1./variance0+np.sum(1./used_variance))
            expected_mean = expected_variance*(mean0/variance0+np.sum(used_values/used_variance))
            np.testing.assert_allclose(current.state.transforms[0].translation[0], expected_mean, atol=1e-12)
            np.testing.assert_allclose(current.state.covariance[0, 0], expected_variance, atol=1e-12)
            innovation = np.diag(used_variance)+variance0*np.ones((index+1, index+1))
            error = used_values-mean0
            evidence = -.5*((index+1)*np.log(2*np.pi)+np.linalg.slogdet(innovation)[1]
                            +error@np.linalg.solve(innovation, error))
            log_masses.append(original.log_weight+evidence)
        expected_logs = np.asarray(log_masses)-logsumexp(log_masses)
        np.testing.assert_allclose([item.log_weight for item in priors], expected_logs, atol=1e-12)


def test_positive_underflow_hypothesis_is_not_pruned(monkeypatch):
    common, paths, priors, observations = _scalar_inputs()
    monkeypatch.setitem(ADAPTERS, 'analytic_test', _analytic_sensor)
    priors = (replace(priors[0], weight=1., log_weight=0.),
              replace(priors[1], weight=0., log_weight=-1000.))
    result = run(common, observations, np.zeros((2, 2), np.uint8), paths=paths, prior_components=priors)
    assert len(result.stages[-1].components) == 2
    tiny = result.stages[-1].components[1]
    assert tiny.weight == 0. and np.isfinite(tiny.log_weight)
    assert tiny.log_weight < -900.
    assert len(result.frozen_body_branches) == 2


def test_native_image_log_masses_survive_float_underflow_and_the_next_tick(moving, monkeypatch):
    from probtf_observation_fusion_demo import sensors
    scene, inputs, cases = moving
    original = sensors.PoseMixtureEstimator.estimate
    def tiny_image_mode(*args, **kwargs):
        detected = original(*args, **kwargs)
        assert len(detected.weights) >= 2
        logs = (0.,)+tuple(-1000.-index for index in range(len(detected.weights)-1))
        weights = tuple(math.exp(value) for value in logs)
        return replace(detected, weights=weights, log_masses=logs)
    monkeypatch.setattr(sensors.PoseMixtureEstimator, 'estimate', tiny_image_mode)
    inference = MovingInference(scene.robot_description, cases['marker'])
    first = inference.update(inputs, 0)
    assert len(first.stages[-1].components) == 1
    assert len(first.stages[-1].goal_components) >= 2
    tiny = first.stages[-1].goal_components[1]
    assert tiny.weight == 0. and math.isfinite(tiny.log_weight)
    assert first.local_estimates[0].posterior.components[1].log_weight == tiny.log_weight
    inference.templates = ()
    second = inference.update(replace(inputs, stamp=1000.05, time=.05), 1)
    assert len(second.stages[-1].components) == len(first.stages[-1].components) == 1
    assert len(second.stages[-1].goal_components) == len(first.stages[-1].goal_components)
    for old, retained in zip(first.stages[-1].goal_components, second.stages[-1].goal_components):
        assert old.log_weight == retained.log_weight
        np.testing.assert_array_equal(old.state.covariance, retained.state.covariance)


def test_image_detection_and_hessian_refinement_run_once_per_observation(moving, monkeypatch):
    from probtf_observation_fusion_demo import sensors
    scene, inputs, cases = moving
    calls = dict(detect=0, refine=0)
    original_detect = sensors.ArucoCornerDetector.detect
    original_estimate = sensors.PoseMixtureEstimator.estimate
    def detect(*args, **kwargs):
        calls['detect'] += 1
        return original_detect(*args, **kwargs)
    def estimate(*args, **kwargs):
        calls['refine'] += 1
        return original_estimate(*args, **kwargs)
    monkeypatch.setattr(sensors.ArucoCornerDetector, 'detect', detect)
    monkeypatch.setattr(sensors.PoseMixtureEstimator, 'estimate', estimate)
    inference = MovingInference(scene.robot_description, cases['marker'])
    inference.update(inputs, 0)
    second = inference.update(replace(inputs, stamp=1000.05, time=.05), 1)
    assert len(second.stages[0].components) == 2
    assert len(second.stages[-1].components) == 1
    assert len(second.stages[-1].goal_components) >= 2
    assert calls == dict(detect=2, refine=2)
    image_only = [estimate.detector_result for estimate in second.local_estimates]
    assert all(value is image_only[0] for value in image_only)


def test_two_sensor_fusion_after_a_real_command_keeps_declared_dirac_coordinates(moving):
    scene, inputs, cases = moving
    inference = MovingInference(scene.robot_description, cases['imu_marker'])
    first = inference.update(inputs, 0)
    control = inference.settings['control']
    controller = MovingController(inference, scene.robot_description, control['goal_offset'],
        control['goal_orientation_wxyz'], control, integration_steps=control['integration_steps'])
    command, _ = controller.step(first, .05)
    physical_frame = scene.step(.05, inputs.stamp+.05, command=command)
    second = inference.update(scene.sensor_inputs(physical_frame), 1)
    assert second.records['valid_observation_count'] == 2
    assert len(second.stages[-1].components) == 1
    assert len(second.stages[-1].goal_components) >= 2
    for branch in second.frozen_body_branches:
        fixed = np.flatnonzero(np.all(branch.prior.covariance == 0., axis=1))
        assert fixed.size > 0
        for child in branch.posterior_components:
            np.testing.assert_array_equal(child.distribution.mean[fixed], 0.)
            np.testing.assert_array_equal(child.distribution.covariance[fixed, :], 0.)
    for component in second.stages[-1].components:
        for j, edge in enumerate(component.state.transforms):
            if edge.bingham is None:
                np.testing.assert_array_equal(component.state.covariance[6*j+3:6*j+6, :], 0.)
