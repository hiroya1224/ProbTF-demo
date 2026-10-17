"""Individual acquired sensors update one retained joint candidate measure."""

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from probtf.spherical_law.first_order import point_query
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore


CONFIG = Path(__file__).resolve().parents[1]/'config'
UNIT = 'per_effective_sensor_event'


@pytest.fixture(scope='module')
def acquired(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp('sensor_event_inputs'))
    inputs = scene.sensor_inputs(scene.step(0., 1000.))
    templates = yaml.safe_load((CONFIG/'imu_marker.yaml').read_text())['observations']
    return scene.robot_description, inputs, templates


@pytest.fixture
def event_factory(tmp_path, acquired):
    _, inputs, _ = acquired
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        def event(identifier, kind, *, stamp=1000., command=None, **extra):
            payload = dict(time=stamp-1000.)
            if kind == 'image':
                payload.update(image=inputs.image, image_stamp=stamp)
            elif kind == 'imu':
                payload.update(gravity_in_sensor=inputs.gravity_in_sensor,
                    imu_target_link=inputs.imu_target_link, imu_sigma_rad=inputs.imu_sigma_rad)
            payload.update(extra)
            return store.append(identifier, kind, stamp,
                inputs.command if command is None else command, payload)
        yield event


def inference(acquired):
    description, _, templates = acquired
    return MovingInference(description, templates)


def update(model, event, index=0):
    return model.update_sensor_event(event, event_index=index, probability_unit=UNIT)


def test_imu_then_image_same_stamp_same_physical_frame_are_separate_updates(acquired, event_factory):
    model = inference(acquired)
    imu = update(model, event_factory('imu:0', 'imu', frame_index=0))
    imu_body = imu.stages[-1].components[0].state
    image_hash = hashlib.sha256(acquired[1].image.tobytes()).hexdigest()
    image = update(model, event_factory('image:0', 'image', frame_index=0,
        image_path='capture_000000.png', image_sha256=image_hash))
    assert imu.records['observation_ids'] == ['imu:0']
    assert image.records['observation_ids'] == ['image:0']
    assert imu.records['valid_observation_count'] == image.records['valid_observation_count'] == 1
    assert image.records['prediction'][0]['source_state_id'] == imu_body.state_id
    assert [item['sensor'] for item in model.last_observations] == ['marker']
    assert image.records['sensor_event']['event_index'] == 0
    assert image.records['sensor_event']['sequence'] > imu.records['sensor_event']['sequence']
    assert image.records['bayes_factor_surprise']['probability_unit'] == UNIT
    assert image.records['fixed_goal_memory_event'] == 'marker_observation_update'
    assert imu.records['sensor_event']['acquisition_metadata'] == {'frame_index': 0}
    assert image.records['sensor_event']['acquisition_metadata'] == dict(frame_index=0,
        image_path='capture_000000.png', image_sha256=image_hash)
    assert not image.records['sensor_event']['acquisition_metadata_is_observation']


def test_image_then_imu_never_reuses_display_pixels_as_an_observation(acquired, event_factory):
    model = inference(acquired)
    image = update(model, event_factory('image:0', 'image'))
    memory = deepcopy(image.records['fixed_goal_memory'])
    imu = update(model, event_factory('imu:1', 'imu', stamp=1000.01))
    assert imu.records['observation_ids'] == ['imu:1']
    assert [item['sensor'] for item in model.last_observations] == ['imu']
    assert all(item.detector_result is None for item in imu.local_estimates)
    assert imu.records['fixed_goal_memory'] == memory
    assert imu.records['fixed_goal_memory_event'] == 'no_image_event_hold'
    before = image.stages[-1].components[0].state
    assert imu.records['prediction'][0]['source_state_id'] == before.state_id
    assert np.linalg.norm(before.covariance-np.diag(np.diag(before.covariance))) > 0.
    assert np.linalg.norm(imu.stages[-1].components[0].state.covariance-
                          np.diag(np.diag(imu.stages[-1].components[0].state.covariance))) > 0.


def test_joint_notification_does_not_create_bayes_factor_or_mass(acquired, event_factory):
    model = inference(acquired)
    initial = update(model, event_factory('joint:0', 'joint'))
    assert initial.records['valid_observation_count'] == 0
    assert initial.records['observation_ids'] == []
    assert not initial.records['bayes_factor_surprise']['applied']
    assert initial.records['candidate_measure']['log_total_mass'] == 0.
    assert initial.records['candidate_measure']['log_mass_increment'] == 0.
    np.testing.assert_array_equal(initial.stages[-1].components[0].state.covariance,
                                  initial.prior.covariance)
    assert model.candidate_measure.components[0].state is initial.stages[-1].components[0].state


def test_nonunit_mass_accumulates_proxy_evidence_and_names_same_shape(acquired, event_factory):
    model = inference(acquired)
    first = update(model, event_factory('imu:0', 'imu'))
    second = update(model, event_factory('imu:1', 'imu', stamp=1000.01), 1)
    e0 = first.records['joint_observation_evidence']['log_evidence']
    e1 = second.records['joint_observation_evidence']['log_evidence']
    record = second.records['candidate_measure']
    assert record['log_total_mass'] == e0+e1
    assert record['previous_log_total_mass'] == e0
    assert record['log_mass_increment'] == e1
    assert record['log_total_mass'] != 0.
    assert record['normalized_shape_view']
    state = second.stages[-1].components[0].state
    assert model.candidate_measure.components[0].state is state
    assert record['shape_source_state_ids'] == [state.state_id]
    assert record['shape_source_commands'] == [list(state.command)]
    assert record['shape_source_stamps'] == [state.stamp]
    assert record['bayes_factor_hypothesis_scale']['keep_log_total_mass'] == e0
    assert record['bayes_factor_hypothesis_scale']['reset_log_total_mass'] == e0


def test_duplicate_is_bitwise_identity_and_older_consumed_event_cannot_rollback(acquired, event_factory):
    model = inference(acquired)
    event = event_factory('imu:0', 'imu')
    first = update(model, event)
    mass = model.candidate_measure
    assert update(model, event) is first
    assert model.candidate_measure is mass
    update(model, event_factory('imu:1', 'imu', stamp=1000.01), 1)
    with pytest.raises(ValueError, match='older sensor event'):
        update(model, event)


def test_missing_image_retains_mass_and_all_root_goal_parameters(acquired, event_factory):
    model = inference(acquired)
    first = update(model, event_factory('image:0', 'image'))
    memory = deepcopy(first.records['fixed_goal_memory'])
    log_mass = model.candidate_measure.log_total_mass
    blank = np.full_like(acquired[1].image, 255)
    missing = update(model, event_factory('image:missing', 'image', stamp=1000.1, image=blank), 1)
    assert missing.records['valid_observation_count'] == 0
    assert not missing.records['bayes_factor_surprise']['applied']
    assert model.candidate_measure.log_total_mass == log_mass
    assert missing.records['candidate_measure']['log_mass_increment'] == 0.
    assert missing.records['fixed_goal_memory_event'] == 'image_missing_hold'
    assert missing.records['fixed_goal_memory'] == memory


def test_event_change_probability_unit_requires_explicit_configuration(acquired, event_factory):
    model = inference(acquired)
    model.surprise_settings.pop('event_probability_unit', None)
    with pytest.raises(ValueError, match='explicitly configured'):
        model.update_sensor_event(event_factory('imu:0', 'imu'))
    assert model.last_result is None
    assert model.change_probability == .25


def test_disabled_sensor_event_is_empty_update_in_same_joint_pipeline(acquired, event_factory):
    description, _, _ = acquired
    model = MovingInference(description, ())
    result = model.update_sensor_event(event_factory('imu:disabled', 'imu'))
    assert not result.records['sensor_event']['enabled']
    assert result.records['valid_observation_count'] == 0
    assert not result.records['bayes_factor_surprise']['applied']
    assert result.records['candidate_measure']['log_total_mass'] == 0.


def test_decreasing_acquisition_stamp_rejected_before_state_update(acquired, event_factory):
    model = inference(acquired)
    update(model, event_factory('imu:later', 'imu', stamp=1000.1), 1)
    last = model.last_result
    with pytest.raises(ValueError, match='timestamp'):
        update(model, event_factory('imu:older', 'imu'), 2)
    assert model.last_result is last


def test_no_monte_carlo_and_only_declared_sensor_payload_is_accepted(acquired, event_factory, monkeypatch):
    model = inference(acquired)
    event = event_factory('imu:0', 'imu')
    def forbidden(*args, **kwargs):
        raise AssertionError('The retained candidate pipeline must not sample.')
    for name in ('normal', 'multivariate_normal', 'default_rng', 'random', 'randn', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    result = update(model, event)
    assert not result.records['sensor_event']['monte_carlo_used']
    truth = event_factory('imu:truth', 'imu', stamp=1000.1, true_joint_angles=[0.]*6)
    with pytest.raises(ValueError, match='Undeclared'):
        update(model, truth, 1)


def test_non_surprise_sensor_event_records_ordinary_proxy_evidence(acquired, event_factory):
    model = inference(acquired)
    model.surprise_enabled = False
    result = model.update_sensor_event(event_factory('imu:0', 'imu'))
    assert not result.records['bayes_factor_surprise']['applied']
    assert result.records['joint_observation_evidence'] is not None
    assert result.records['candidate_measure']['log_total_mass'] == result.records[
        'joint_observation_evidence']['log_evidence']


def test_zero_body_change_probability_is_ordinary_fusion(acquired, event_factory, tmp_path):
    settings = yaml.safe_load((CONFIG/'moving_estimation.yaml').read_text())
    settings['bayes_factor_surprise']['change_probability'] = 0.
    path = tmp_path/'zero_change.yaml'
    path.write_text(yaml.safe_dump(settings))
    zero = MovingInference(acquired[0], acquired[2], config_path=path)
    ordinary = inference(acquired)
    ordinary.surprise_enabled = False
    event = event_factory('imu:zero', 'imu')
    actual = zero.update_sensor_event(event)
    expected = ordinary.update_sensor_event(event)
    assert actual.records['valid_observation_count'] == 1
    assert not actual.records['bayes_factor_surprise']['applied']
    assert actual.records['bayes_factor_surprise']['reason'] == 'zero change probability: ordinary observation fusion'
    assert actual.records['bayes_factor_surprise']['prior_change_odds'] == 0.
    assert actual.stages[-1].components[0].state.state_id == expected.stages[-1].components[0].state.state_id
    assert actual.records['candidate_measure'] == expected.records['candidate_measure']


@pytest.mark.parametrize('kind,metadata,message', [
    ('imu', {'frame_index': -1}, 'frame_index'),
    ('imu', {'frame_index': 0.5}, 'frame_index'),
    ('imu', {'frame_index': True}, 'frame_index'),
    ('image', {'image_path': 0}, 'image_path'),
    ('image', {'image_path': ''}, 'image_path'),
    ('image', {'image_sha256': 'bad hash'}, 'image_sha256'),
    ('image', {'image_sha256': '0'*64}, 'acquired pixels'),
    ('imu', {'image_path': 'unobserved.png'}, 'Undeclared'),
    ('image', {'true_marker_pose': [0.]*7}, 'Undeclared'),
])
def test_acquisition_metadata_validation_never_updates_candidate(acquired, event_factory,
                                                               kind, metadata, message):
    model = inference(acquired)
    event = event_factory('bad_metadata', kind, **metadata)
    with pytest.raises(ValueError, match=message):
        update(model, event)
    assert model.last_result is None
    assert model.candidate_measure is None


def test_event_index_and_ledger_physical_frame_must_match(acquired, event_factory):
    model = inference(acquired)
    event = event_factory('wrong_frame', 'joint', frame_index=4,
                          command_velocity=[0.]*6)
    with pytest.raises(ValueError, match='physical frame_index disagree'):
        update(model, event, 3)
    result = update(model, event, 4)
    assert result.records['sensor_event']['acquisition_metadata'] == {'frame_index': 4}
    assert result.records['valid_observation_count'] == 0
    with pytest.raises(ValueError, match='physical frame_index disagree'):
        update(model, event, 3)


def test_default_event_index_uses_acquired_frame_before_db_sequence(acquired, event_factory):
    model = inference(acquired)
    event = event_factory('frame_notification', 'joint', frame_index=24)
    assert event.sequence != 24
    result = model.update_sensor_event(event)
    assert result.records['sensor_event']['event_index'] == 24
    assert result.records['moving_input_family']['frame_index'] == 24
