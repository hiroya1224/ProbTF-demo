"""Acquisition likelihood pullbacks keep native modes and all shared cross."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from probtf.spherical_law.fusion import Gaussian, condition_gaussian, lift_sensor_posterior
from probtf_observation_fusion_demo.async_state import ActionStep
from probtf_observation_fusion_demo.batch_observation_factors import (
    AcquisitionTransport, build_observation_factor, prepare_batch_images, transport_state_to_acquisition,
)
from probtf_observation_fusion_demo import batch_observation_factors as factors_module
from probtf_observation_fusion_demo.moving_inference import MovingInference
from legacy_marker import legacy_marker_scene


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def problem(tmp_path_factory):
    scene = legacy_marker_scene(tmp_path_factory.mktemp('batch_factor'), imu_target_link='tool0')
    inputs = scene.sensor_inputs(scene.step(0., 1000.))
    templates = tuple(yaml.safe_load((PACKAGE/'config/imu_marker.yaml').read_text())['observations'])
    inference = MovingInference(scene.robot_description, templates, config_path=scene.legacy_inference_config)
    state = inference.prior_at(inputs.command, inputs.stamp, 0)
    covariance = state.covariance.copy()
    covariance[3, 9] = covariance[9, 3] = .1*np.sqrt(covariance[3, 3]*covariance[9, 9])
    state = replace(state, covariance=covariance)
    command = np.array(inputs.command)+np.array([.02, 0., -.01, 0., 0., 0.])
    increment = (.013, 0., -.004, 0., 0., 0.)
    step = ActionStep(0, 1000., 1000.1, tuple(inputs.command), tuple(command), increment)
    acquisition, chart, ledger = transport_state_to_acquisition(state, (step,),
        inference.body_ids, inference.actions, acquisition_stamp=1000.1, acquisition_command=command)
    return inference, inputs, templates, state, acquisition, chart, ledger, step


def _observation(problem, sensor):
    inference, inputs, templates, *_ = problem
    observation = dict(next(item for item in templates if item['sensor'] == sensor))
    observation['observation_id'] = 'original:'+sensor+':1'
    if sensor == 'imu':
        observation.update(sensor_id='imu_tool0', target_link='tool0', gravity_in_root=[0., 0., -1.],
            gravity_in_sensor=np.asarray(inputs.gravity_in_sensor).tolist(), direction_sigma_rad=inputs.imu_sigma_rad)
    return observation


def _factor(problem, sensor, prepared=None):
    inference, inputs, _, common, acquisition, chart, ledger, _ = problem
    observation = _observation(problem, sensor)
    return build_observation_factor(common, acquisition, observation, inference.paths,
        inference.common, inputs.image, common_to_acquisition=chart,
        event_metadata=dict(event_id=observation['observation_id'], sequence=7,
            acquisition_stamp=acquisition.stamp, sent_command=list(acquisition.command),
            observation_id=observation['observation_id']), ledger_record=ledger, prepared_marker=prepared)


@pytest.mark.parametrize('sensor', ['imu', 'marker'])
def test_pulled_likelihood_matches_existing_sensor_lift_with_complete_cross(problem, sensor):
    _, _, _, common, acquisition, chart, _, _ = problem
    factor = _factor(problem, sensor)
    assert factor.effective and factor.observation_ids == ('original:'+sensor+':1',)
    acquisition_lifts = lift_sensor_posterior(factor.acquisition_estimate.posterior)
    common_lifts = lift_sensor_posterior(factor.common_posterior)
    assert len(common_lifts) == len(acquisition_lifts) == len(factor.components)
    for component, acquired, pulled in zip(factor.components, acquisition_lifts, common_lifts):
        direct, _ = condition_gaussian(Gaussian(common.mean, common.covariance),
            component.observed-component.offset, component.covariance, component.affine)
        np.testing.assert_allclose(pulled.distribution.mean, chart.T @ acquired.distribution.mean,
                                   rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(pulled.distribution.covariance,
            chart.T @ acquired.distribution.covariance @ chart, rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(direct.mean, pulled.distribution.mean, rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(direct.covariance, pulled.distribution.covariance, rtol=2e-10, atol=2e-11)
        assert not component.affine.flags.writeable
    assert factor.common_stamp == 1000. and factor.acquisition_stamp == 1000.1
    assert factor.common_command != factor.acquisition_command
    assert factor.record['common_factor_clock_role'].startswith('frozen coordinate reference')
    assert factor.record['acquisition_event']['sequence'] == 7
    assert factor.acquisition_estimate.posterior.query.stamp == acquisition.stamp
    assert factor.common_posterior.query.stamp == common.stamp
    with pytest.raises(TypeError): factor.record['acquisition_stamp'] = common.stamp


def test_native_image_log_weights_are_retained_before_body_overlap(problem):
    factor = _factor(problem, 'marker')
    native = factor.acquisition_estimate.diagnostics['components']
    assert len(native) >= 2
    for component, original in zip(factor.components, native):
        assert component.label == original['label']
        assert component.log_weight == original['image_log_weight']+original['chart_log_volume']
        np.testing.assert_array_equal(component.observed, original['image_only_mean'])
        np.testing.assert_array_equal(component.covariance, original['image_only_covariance'])
    assert factor.prepared_marker.detected is factor.acquisition_estimate.detector_result


def test_underflowed_native_image_mode_is_not_pruned(problem):
    inference, inputs, _, _, acquisition, *_ = problem
    prepared = factors_module.prepare_marker_observation(inference.common, inputs.image, acquisition.stamp)
    assert len(prepared.detected.weights) == 2
    detected = replace(prepared.detected, log_masses=(0., -1000.), weights=(1., 0.))
    factor = _factor(problem, 'marker', replace(prepared, detected=detected))
    assert len(factor.components) == 2
    assert np.isfinite(factor.components[1].log_weight)
    assert factor.components[1].log_weight < -990.


def test_images_are_preprocessed_once_per_original_event_independent_of_body(problem, monkeypatch):
    inference, inputs, *_ = problem
    event = SimpleNamespace(kind='image', event_id='image:1', stamp=1000.1,
        command=tuple(inputs.command), payload=dict(image=inputs.image, image_stamp=1000.1))
    original = factors_module.prepare_marker_observation
    calls = []
    def prepare(common, image, stamp):
        calls.append((image, stamp))
        return original(common, image, stamp)
    monkeypatch.setattr(factors_module, 'prepare_marker_observation', prepare)
    result = prepare_batch_images((event, event), inference.common)
    assert len(calls) == 1 and set(result) == {'image:1'}
    assert result['image:1'].image is inputs.image
    changed = SimpleNamespace(**dict(vars(event), payload=dict(image=np.zeros_like(inputs.image), image_stamp=1000.1)))
    with pytest.raises(ValueError, match='changed acquisition'):
        prepare_batch_images((event, changed), inference.common)


def test_missing_marker_is_not_invented_as_an_effective_observation(problem):
    inference, inputs, _, common, acquisition, chart, ledger, _ = problem
    image = np.full_like(inputs.image, 255)
    factor = build_observation_factor(common, acquisition, _observation(problem, 'marker'),
        inference.paths, inference.common, image, common_to_acquisition=chart, ledger_record=ledger)
    assert not factor.effective and factor.components == () and factor.observation_ids == ()
    assert factor.acquisition_estimate.posterior is None
    assert factor.record['observation_id'] == 'original:marker:1'
    assert factor.acquisition_stamp == acquisition.stamp


def test_closed_command_history_retains_actual_K_dependent_net_action(problem):
    inference, _, _, common, _, _, _, first = problem
    second = ActionStep(1, first.end_stamp, 1000.2, first.end_command, first.start_command,
        (-.005, 0., .001, 0., 0., 0.))
    state, chart, record = transport_state_to_acquisition(common, (first, second),
        inference.body_ids, inference.actions, acquisition_stamp=1000.2,
        acquisition_command=common.command)
    assert state.command == common.command
    assert record['joint_increment'][0] == pytest.approx(.008)
    assert not np.array_equal(chart, np.eye(len(common.mean)))
    assert not np.array_equal(state.transforms[0].bingham, common.transforms[0].bingham)
    np.testing.assert_allclose(state.covariance, chart @ common.covariance @ chart.T, atol=1e-14)


def test_ledger_or_original_acquisition_mismatch_is_rejected(problem):
    inference, inputs, _, common, acquisition, chart, ledger, first = problem
    with pytest.raises(ValueError, match='endpoint differs'):
        transport_state_to_acquisition(common, (first,), inference.body_ids, inference.actions,
            acquisition_stamp=1000.15, acquisition_command=first.end_command)
    with pytest.raises(ValueError, match='original acquisition'):
        build_observation_factor(common, acquisition, _observation(problem, 'imu'),
            inference.paths, inference.common, inputs.image, common_to_acquisition=chart,
            event_metadata=dict(acquisition_stamp=1000.2), ledger_record=ledger)


def test_shared_ledger_prefix_cache_matches_each_full_prefix_bitwise(problem):
    inference, _, _, common, _, _, _, first = problem
    second = ActionStep(1, first.end_stamp, 1000.2, first.end_command, first.start_command,
        (-.005, 0., .001, 0., 0., 0.))
    transport = AcquisitionTransport(common, (first, second), inference.body_ids, inference.actions)
    assert len(transport.ledger_record['steps']) == 2
    for steps, stamp, command in (((), common.stamp, common.command),
            ((first,), first.end_stamp, first.end_command),
            ((first, second), second.end_stamp, second.end_command)):
        expected, expected_chart, _ = transport_state_to_acquisition(common, steps,
            inference.body_ids, inference.actions, acquisition_stamp=stamp, acquisition_command=command)
        actual, chart, record = transport.advance_to(stamp, command)
        assert actual.state_id == expected.state_id
        np.testing.assert_array_equal(actual.covariance, expected.covariance)
        np.testing.assert_array_equal(chart, expected_chart)
        assert record['ledger_id'] == transport.ledger_id
        assert record['step_count'] == len(steps) and 'steps' not in record
        assert record['source_prefix_chain_sha256'] == transport.at(common.stamp, common.command)[2]['prefix_chain_sha256']
    with pytest.raises(ValueError, match='backwards'):
        transport.advance_to(first.end_stamp, first.end_command)


def test_shared_ledger_validation_does_not_accept_an_unexecuted_endpoint(problem):
    inference, _, _, common, _, _, _, first = problem
    transport = AcquisitionTransport(common, (first,), inference.body_ids, inference.actions)
    with pytest.raises(ValueError, match='endpoint differs'):
        transport.at(first.end_stamp+.01, first.end_command)
    changed = replace(first, start_stamp=first.start_stamp-.1)
    with pytest.raises(ValueError, match='contiguous'):
        AcquisitionTransport(common, (changed,), inference.body_ids, inference.actions)
