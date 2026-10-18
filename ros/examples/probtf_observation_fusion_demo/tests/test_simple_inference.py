"""The minimal loop keeps body and environment in one canonical law."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.node import build_snapshot
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore
from probtf_observation_fusion_demo.simple_inference import SimpleInference


CONFIG = Path(__file__).resolve().parents[1]/'config'


@pytest.fixture(scope='module')
def acquired(tmp_path_factory):
    # Physical poses are used to generate sensor pixels/directions only. The
    # inference constructor receives known URDF geometry, never physical K/q.
    scene = MovingSwitchScene(tmp_path_factory.mktemp('simple_sensor_inputs'))
    inputs = scene.sensor_inputs(scene.step(0., 1000.))
    templates = tuple(yaml.safe_load((CONFIG/'imu_marker.yaml').read_text())['observations'])
    return scene.robot_description, inputs, templates


@pytest.fixture
def events(tmp_path, acquired):
    inputs = acquired[1]
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        def make(identity, kind, stamp=1000., command=None, **extra):
            payload = dict(time=stamp-1000.)
            if kind == 'image':
                payload.update(image=inputs.image, image_stamp=stamp)
            elif kind == 'imu':
                payload.update(gravity_in_sensor=inputs.gravity_in_sensor,
                    imu_target_link=inputs.imu_target_link, imu_sigma_rad=inputs.imu_sigma_rad)
            payload.update(extra)
            return store.append(identity, kind, stamp, inputs.command if command is None else command, payload)
        yield make


def state(result):
    stage = result.stages[-1]
    assert len(stage.components) == 1 and not stage.goal_components
    return stage.components[0].state


def test_initial_translation_Diracs_preserve_rotation_and_environment_priors(acquired):
    description, inputs, templates = acquired
    old = MovingInference(description, templates).prior_at(inputs.command, inputs.stamp, 0)
    simple = SimpleInference(description, templates)
    assert simple.observation_update_settings['max_iterations'] == 160
    assert simple.settings['observation_update']['max_iterations'] == 160
    initial = simple.prior_at(inputs.command, inputs.stamp, 0)
    assert not simple.goal_observed and not simple.surprise_enabled
    for name in simple.body_ids:
        index = initial.edge_ids.index(name)
        part = slice(6*index, 6*index+3)
        assert np.count_nonzero(initial.covariance[part]) == 0
        np.testing.assert_array_equal(initial.transforms[index].translation, old.transforms[index].translation)
        np.testing.assert_array_equal(initial.transforms[index].bingham, old.transforms[index].bingham)
        np.testing.assert_array_equal(initial.covariance[6*index+3:6*index+6, 6*index+3:6*index+6],
                                      old.covariance[6*index+3:6*index+6, 6*index+3:6*index+6])
    marker = initial.edge_ids.index(simple.marker.edge_id)
    part = slice(6*marker, 6*marker+6)
    np.testing.assert_array_equal(initial.covariance[part, part], old.covariance[part, part])
    assert initial.covariance.shape == old.covariance.shape


def test_initial_joint_notification_has_no_goal_or_observation_update(acquired, events):
    model = SimpleInference(acquired[0], acquired[2])
    result = model.update_sensor_event(events('joint:0', 'joint', frame_index=0))
    current = state(result)
    assert result.records['observation_ids'] == []
    assert result.records['simple_goal_observed'] is False
    assert result.records['simple_inference']['goal_observed'] is False
    assert not result.records['bayes_factor_surprise']['enabled']
    assert result.records['candidate_measure']['log_mass_increment'] == 0.
    assert current is model.components[0].state
    assert not model.goal_components and model.fixed_goal_memory is None


def test_real_marker_and_IMU_updates_keep_shared_environment_cross_without_cut(acquired, events, monkeypatch):
    import probtf_observation_fusion_demo.moving_inference as old_module
    def forbidden(*args, **kwargs):
        raise AssertionError('The minimal reference must not redraw a body or construct independent fresh goals.')
    monkeypatch.setattr(old_module, 'fresh_marker_goals', forbidden)
    monkeypatch.setattr(old_module, 'transition_priors', forbidden)
    monkeypatch.setattr(old_module, 'retain_goal_components', forbidden)
    model = SimpleInference(acquired[0], acquired[2])
    image = model.update_sensor_event(events('image:0', 'image', frame_index=0))
    first = state(image)
    marker_index = first.edge_ids.index(model.marker.edge_id)
    marker_indices = np.arange(6*marker_index, 6*marker_index+6)
    body_rotation = np.array([6*first.edge_ids.index(name)+offset
                             for name in model.body_ids for offset in (3, 4, 5)])
    assert np.linalg.norm(first.covariance[np.ix_(body_rotation, marker_indices)]) > 0.
    assert model.goal_observed and image.records['simple_goal_observed']
    assert 'fresh_goal_construction' not in image.records
    assert not image.records['internal_projection']['conditional_goals_constructed']
    assert image.records['internal_projection']['source_component_count'] >= 1
    metadata = image.records['fixed_goal_memory']
    assert metadata['metadata_only'] and not metadata['independent_goal_law']
    assert 'goal_components' not in metadata and 'root_marker_parameters' not in metadata
    imu = model.update_sensor_event(events('imu:1', 'imu', stamp=1000.01, frame_index=1))
    current = state(imu)
    assert imu.records['prediction'][0]['source_state_id'] == first.state_id
    assert current is model.components[0].state
    assert np.linalg.norm(current.covariance[np.ix_(body_rotation, marker_indices)]) > 0.
    assert imu.records['fixed_goal_memory']['observation_stamp'] == 1000.
    assert imu.records['fixed_goal_memory']['source_stamp'] == 1000.01
    assert imu.records['simple_inference']['state_id'] == current.state_id
    assert model.goal_observed and not model.goal_components
    branch = imu.frozen_body_branches[0]
    assert branch.reference_state.state_id == imu.prior.state_id
    np.testing.assert_array_equal(branch.posterior_components[0].distribution.covariance,
                                  imu.frozen_body_posterior.components[0].distribution.covariance)


def test_missing_image_and_duplicate_preserve_joint_law_and_observed_latch(acquired, events):
    model = SimpleInference(acquired[0], acquired[2])
    first = model.update_sensor_event(events('image:0', 'image', frame_index=0))
    before = state(first)
    marker = before.edge_ids.index(model.marker.edge_id)
    blank = np.full_like(acquired[1].image, 255)
    missing_event = events('image:missing', 'image', stamp=1000.1, frame_index=1, image=blank)
    missing = model.update_sensor_event(missing_event)
    current = state(missing)
    assert model.goal_observed and missing.records['simple_goal_observed']
    assert missing.records['valid_observation_count'] == 0
    np.testing.assert_array_equal(current.covariance, before.covariance)
    np.testing.assert_array_equal(current.transforms[marker].translation, before.transforms[marker].translation)
    assert model.update_sensor_event(missing_event) is missing


def test_display_body_marker_and_switch_are_queries_of_the_same_joint_state(acquired, events):
    model = SimpleInference(acquired[0], acquired[2])
    result = model.update_sensor_event(events('image:0', 'image', frame_index=0))
    current = state(result)
    settings = deepcopy(model.common['display'])
    snapshot = build_snapshot(result, result.stages[-1], model.common)
    assert {point.query.state_id for point in snapshot.points} == {current.state_id}
    assert {'hand', 'marker', 'switch_goal'}.issubset({point.frame_label for point in snapshot.points})
    assert model.common['display'] == settings
    assert result.records['candidate_measure']['shape_source_state_ids'] == [current.state_id]


def test_preserve_original_sensor_ID_stamp_command_and_reject_truth_fields(acquired, events):
    model = SimpleInference(acquired[0], acquired[2])
    event = events('imu:original', 'imu', stamp=1000.125, frame_index=7)
    result = model.update_sensor_event(event)
    assert result.records['observation_ids'] == [event.event_id]
    record = result.records['sensor_event']
    assert record['acquisition_stamp'] == event.stamp
    assert record['sent_command'] == list(event.command)
    assert record['sequence'] == event.sequence
    assert result.prior.stamp == event.stamp and result.prior.command == event.command
    bad = events('imu:truth', 'imu', stamp=1000.2, frame_index=8, actual_joints=[0.]*6)
    with pytest.raises(ValueError, match='Undeclared sensor-event fields'):
        model.update_sensor_event(bad)


def test_two_native_branches_merge_full_joint_within_between_and_environment_cross(acquired, events, monkeypatch):
    from test_body_projection import _result
    model = SimpleInference(acquired[0], acquired[2])
    initial = model.update_sensor_event(events('joint:0', 'joint', frame_index=0))
    native = _result(initial.prior, weights=(.3, .7))
    children = native.frozen_body_branches[0].posterior_components
    raw = replace(initial, stages=native.stages,
        frozen_body_branches=native.frozen_body_branches,
        frozen_body_posterior=replace(initial.frozen_body_posterior, components=children),
        records=dict(stages=[dict(label='fused', sensor_ids=[], components=[])]))
    mean = sum(child.weight*child.distribution.mean for child in children)
    covariance = sum(child.weight*(child.distribution.covariance+
        np.outer(child.distribution.mean-mean, child.distribution.mean-mean)) for child in children)
    def forbidden(*args, **kwargs):
        raise AssertionError('The joint moment approximation must not use Monte Carlo.')
    for name in ('default_rng', 'normal', 'random', 'multivariate_normal'):
        monkeypatch.setattr(np.random, name, forbidden)
    projected = model._project_internal_update(raw, ())
    law = projected.frozen_body_posterior.components[0].distribution
    np.testing.assert_allclose(law.mean, mean, atol=1.e-17)
    np.testing.assert_allclose(law.covariance, covariance, atol=1.e-17)
    assert projected.records['internal_projection']['source_component_count'] == 2
    assert [item['weight'] for item in projected.records['internal_projection']['sources']] == [.3, .7]
    assert law.covariance[3, 60] != 0.
    assert state(projected).covariance[3, 60] != 0.
    assert not projected.stages[-1].goal_components
