"""One internal body and every current goal mode share actual IK/wire sources."""

from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from probtf.geometry import axis_angle_to_quat, quat_to_rotmat
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates, point_query,
)
from probtf_observation_fusion_demo.display import (
    make_markers, serialized_marker_array, snapshot_record, verify_wire_identity,
)
from probtf_observation_fusion_demo.model import WeightedState, load_inputs
from probtf_observation_fusion_demo.moving_control import MovingController
from probtf_observation_fusion_demo.node import build_snapshot, verify_adapter_identity


CONFIG = Path(__file__).resolve().parents[1]/'config'


def _inputs(underflow=False):
    """Explicit test inputs: 66 coordinates and the same full P_BB in both goals."""
    body = tuple(BinghamTransform('link_'+str(index+1), [.08, 0., .01],
        bingham_from_rotation_coordinates([1., 0., 0., 0.], .002*np.eye(3)),
        .0001*np.eye(3)) for index in range(6))
    fixed = tuple(BinghamTransform(name, translation, rotation_wxyz=[1., 0., 0., 0.])
                  for name, translation in (
                      ('tool0', [.02, 0., 0.]), ('press_rod', [.04, 0., 0.]),
                      ('rod_tip', [.04, 0., 0.]), ('camera', [.01, .03, .02])))
    marker = BinghamTransform('marker', [.6, -.1, .3],
        bingham_from_rotation_coordinates([1., 0., 0., 0.], .003*np.eye(3)), .0003*np.eye(3))
    state = JointTransformState.from_independent(body+fixed+(marker,),
        version=18, stamp=1234.25, command=(0.,)*6, component_id='internal')
    assert state.covariance.shape == (66, 66)
    internal = WeightedState('internal', 1., state, 0.)
    weights, logs = ((1., 0.), (0., -1000.)) if underflow else ((.3, .7), tuple(np.log([.3, .7])))
    goals = []
    for index, angle in enumerate((-.12, .18)):
        candidate_marker = BinghamTransform('marker', [.6, -.1+.03*index, .3],
            bingham_from_rotation_coordinates(axis_angle_to_quat([0., 0., 1.], angle),
                                              .003*np.eye(3)), .0003*np.eye(3))
        covariance = state.covariance.copy()
        # Distinct valid environment conditionals retain body/marker cross
        # blocks, while their complete 60-coordinate body marginal is shared.
        covariance[5, 60] = covariance[60, 5] = (
            (-1.)**index*.2*np.sqrt(covariance[5, 5]*covariance[60, 60]))
        candidate = JointTransformState(body+fixed+(candidate_marker,), covariance,
            version=state.version, stamp=state.stamp, command=state.command,
            component_id='goal:observation:mode:'+str(index))
        np.testing.assert_array_equal(candidate.covariance[:60, :60], state.covariance[:60, :60])
        assert all(actual is expected for actual, expected in zip(candidate.transforms[:10], state.transforms[:10]))
        goals.append(WeightedState(candidate.component_id, weights[index], candidate, logs[index]))
    stage = SimpleNamespace(label='fused', components=(internal,), goal_components=tuple(goals), sensor_ids=())
    chain = tuple((edge.edge_id, False) for edge in body)
    paths = {edge.edge_id: chain[:index+1] for index, edge in enumerate(body)}
    result = SimpleNamespace(prior=state, stages=(stage,), root_frame='base_link',
        link_paths=paths, hand_path=chain+(('tool0', False), ('press_rod', False), ('rod_tip', False)),
        camera_path=chain+(('tool0', False), ('camera', False)),
        marker_path=(('marker', False),), local_estimates=(), sensor_posteriors=(),
        records={'observation_ids': []})
    common, _ = load_inputs(CONFIG/'common.yaml', CONFIG/'no_sensors.yaml')
    inference = SimpleNamespace(body_ids=tuple(paths),
        actions=tuple({'joint_index': index, 'axis': [0., 0., 1.]} for index in range(6)),
        paths={'hand': result.hand_path, 'marker': result.marker_path},
        settings={'root_frame': result.root_frame})
    urdf = '<robot name="test">'+''.join(
        '<joint name="joint_{}" type="revolute"><limit lower="-3" upper="3"/></joint>'.format(index)
        for index in range(6))+'</robot>'
    controller = MovingController(inference, urdf, [0., -.24, .045], [.5, -.5, .5, .5],
        {'max_joint_velocity': .12, 'max_joint_acceleration': .3, 'damping': .025})
    return common, result, controller


@pytest.mark.parametrize('underflow', [False, True])
def test_one_body_and_all_goal_modes_use_the_actual_ik_sources_on_ros_wire(underflow):
    common, result, controller = _inputs(underflow)
    stage = result.stages[-1]
    _, diagnostics = controller.step(result, .05)
    snapshot = build_snapshot(result, stage, common)
    settings = common['display']
    message = make_markers(snapshot, settings['seed'], settings['point_size'])
    payload = verify_wire_identity(snapshot, message, seed=settings['seed'], point_size=settings['point_size'])
    wire = MarkerArray().deserialize(payload)
    assert serialized_marker_array(wire) == payload
    metadata = String(json.dumps(snapshot_record(snapshot)))
    stream = io.BytesIO()
    metadata.serialize(stream)
    received = json.loads(String().deserialize(stream.getvalue()).data)

    body = stage.components[0]
    goals = {item.label: item for item in stage.goal_components}
    assert diagnostics['body_source_state_ids'] == [body.state.state_id]
    assert diagnostics['source_state_ids'] == diagnostics['goal_source_state_ids'] == [g.state.state_id for g in goals.values()]
    assert diagnostics['source_weights'] == [g.weight for g in goals.values()]
    assert diagnostics['source_log_weights'] == [g.log_weight for g in goals.values()]
    body_points = [point for point in snapshot.points if point.frame_label != 'marker']
    goal_points = [point for point in snapshot.points if point.frame_label == 'marker']
    assert len(body_points) == 4*(len(result.link_paths)+2)
    assert len(goal_points) == 4*len(goals)
    assert {point.query.state_id for point in body_points} == {body.state.state_id}
    assert {point.query.state_id for point in goal_points} == set(diagnostics['goal_source_state_ids'])
    assert all(point.component_id == body.label for point in body_points)
    assert len({point.component_id for point in goal_points}) == 2
    rendered = [m for m in wire.markers if m.type == Marker.SPHERE_LIST and m.ns.startswith('gaussian/')]
    assert len(rendered) == len(snapshot.points) == len(received['queries'])
    endpoints = {'origin': [0., 0., 0.], 'x': [settings['axis_length'], 0., 0.],
                 'y': [0., settings['axis_length'], 0.], 'z': [0., 0., settings['axis_length']]}
    paths = dict(result.link_paths, hand=result.hand_path, camera=result.camera_path, marker=result.marker_path)
    for point, record, marker in zip(snapshot.points, received['queries'], rendered):
        source = goals[point.component_id] if point.frame_label == 'marker' else body
        expected = point_query(source.state, paths[point.frame_label], endpoints[point.axis], frame=result.root_frame)
        assert record['source_state_id'] == expected.state_id == source.state.state_id
        assert record['component_id'] == record['source_component_id'] == source.label
        assert record['weight'] == source.weight
        assert record['log_weight'] == source.log_weight
        assert record['source_command'] == diagnostics['source_command'] == list(source.state.command)
        assert record['source_stamp'] == diagnostics['source_stamp'] == source.state.stamp
        assert record['source_version'] == source.state.version == snapshot.version
        assert marker.header.frame_id == snapshot.frame_id
        assert marker.header.stamp.to_sec() == snapshot.stamp
        np.testing.assert_array_equal(record['mean'], expected.mean)
        np.testing.assert_array_equal(record['covariance'], expected.covariance)
        np.testing.assert_allclose(marker.color.a, .85*source.weight, atol=1e-7)
    assert np.linalg.norm(stage.goal_components[0].state.covariance[:60, 60:]) > 0.
    assert not np.array_equal(stage.goal_components[0].state.covariance[:60, 60:],
                              stage.goal_components[1].state.covariance[:60, 60:])
    if underflow:
        assert any(record['frame_label'] == 'marker' and record['weight'] == 0.
                   and record['log_weight'] == -1000. for record in received['queries'])


@pytest.mark.parametrize('field', ['stamp', 'command', 'version'])
def test_marker_goal_from_another_observation_command_or_version_is_rejected(field):
    common, result, _ = _inputs()
    goal = result.stages[-1].goal_components[0]
    changes = {'stamp': goal.state.stamp+.05, 'command': (.01,)+(0.,)*5,
               'version': goal.state.version+1}
    state = replace(goal.state, **{field: changes[field]})
    result.stages[-1].goal_components = (replace(goal, state=state), result.stages[-1].goal_components[1])
    with pytest.raises(AssertionError, match='joint-state query'):
        build_snapshot(result, result.stages[-1], common)


@pytest.mark.parametrize('change', ['internal_marker_substitution', 'weight', 'log_weight', 'missing_goal'])
def test_goal_display_cannot_substitute_internal_marker_or_drop_goal_sources(change):
    common, result, _ = _inputs()
    stage = result.stages[-1]
    snapshot = build_snapshot(result, stage, common)
    points = list(snapshot.points)
    index = next(index for index, point in enumerate(points) if point.frame_label == 'marker')
    point = points[index]
    if change == 'internal_marker_substitution':
        wrong = point_query(result.prior, result.marker_path, frame=result.root_frame)
        points[index] = replace(point, query=wrong)
    elif change == 'weight':
        points[index] = replace(point, weight=.1)
    elif change == 'log_weight':
        points[index] = replace(point, log_weight=-1000.)
    else:
        points = [point for point in points if point.component_id != stage.goal_components[1].label]
    with pytest.raises(AssertionError):
        verify_adapter_identity(replace(snapshot, points=tuple(points)), stage)


def test_stage_without_separate_goals_keeps_the_existing_snapshot_and_wire_path():
    common, result, _ = _inputs()
    source = result.stages[-1]
    old = SimpleNamespace(label=source.label, components=source.components, sensor_ids=source.sensor_ids)
    empty = SimpleNamespace(**vars(old), goal_components=())
    first = build_snapshot(result, old, common)
    second = build_snapshot(result, empty, common)
    assert snapshot_record(first) == snapshot_record(second)
    settings = common['display']
    assert serialized_marker_array(make_markers(first, settings['seed'], settings['point_size'])) == serialized_marker_array(
        make_markers(second, settings['seed'], settings['point_size']))


@pytest.mark.parametrize('case', ['marker', 'imu_marker'])
def test_change_weighted_projected_body_and_all_goal_modes_match_controller_and_ros_wire(case, tmp_path):
    from probtf.spherical_law.first_order import reexpress_state
    from probtf_observation_fusion_demo.moving_inference import MovingInference
    from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
    scene = MovingSwitchScene(tmp_path/case)
    templates = yaml.safe_load((CONFIG/(case+'.yaml')).read_text())['observations']
    inference = MovingInference(scene.robot_description, templates)
    inference.update(scene.sensor_inputs(scene.step(0., 1000.)), 0)
    result = inference.update(scene.sensor_inputs(scene.step(.05, 1000.05)), 1)
    assert result.records['bayes_factor_surprise']['applied']
    assert len(result.frozen_body_branches) == 2
    stage = result.stages[-1]
    body = stage.components[0]
    assert len(stage.components) == 1
    detected = next(estimate.detector_result for estimate in result.local_estimates
                    if estimate.detector_result is not None)
    assert len(stage.goal_components) == len(detected.record.distribution.components)
    assert [goal.weight for goal in stage.goal_components] == list(detected.weights)
    expected_logs = np.asarray(detected.log_masses)-np.logaddexp.reduce(detected.log_masses)
    np.testing.assert_array_equal([goal.log_weight for goal in stage.goal_components],expected_logs)
    assert all(goal.label.endswith(native.component_id) for goal,native in
               zip(stage.goal_components,detected.record.distribution.components))
    numerator = result.projected_body_numerator
    assert len(numerator.posterior_components) == 1
    raw = numerator.posterior_components[0]
    expected_body = reexpress_state(numerator.reference_state,
        raw.distribution.mean, raw.distribution.covariance,
        version=body.state.version, component_id=body.label)
    assert expected_body.state_id == body.state.state_id

    control = inference.settings['control']
    controller = MovingController(inference, scene.robot_description, control['goal_offset'],
        control['goal_orientation_wxyz'], control, integration_steps=control['integration_steps'])
    _, diagnostics = controller.step(result, .05)
    snapshot = build_snapshot(result, stage, inference.common)
    assert snapshot.surprise_record is result.records['bayes_factor_surprise']
    settings = inference.settings['display']
    message = make_markers(snapshot, settings['seed'], settings['point_size'])
    payload = verify_wire_identity(snapshot, message,
        seed=settings['seed'], point_size=settings['point_size'])
    assert serialized_marker_array(MarkerArray().deserialize(payload)) == payload
    wire = String(json.dumps(snapshot_record(snapshot)))
    stream = io.BytesIO(); wire.serialize(stream)
    received = json.loads(String().deserialize(stream.getvalue()).data)
    assert received['bayes_factor_surprise'] == result.records['bayes_factor_surprise']
    assert diagnostics['body_source_state_ids'] == [body.state.state_id]
    assert diagnostics['goal_source_state_ids'] == [item.state.state_id for item in stage.goal_components]
    assert diagnostics['source_log_weights'] == [item.log_weight for item in stage.goal_components]
    goals = {item.label: item for item in stage.goal_components}
    endpoints = {'origin': np.zeros(3), 'x': [settings['axis_length'], 0., 0.],
                 'y': [0., settings['axis_length'], 0.], 'z': [0., 0., settings['axis_length']]}
    paths = dict(result.link_paths, hand=result.hand_path,
                 camera=result.camera_path, marker=result.marker_path)
    for point, record in zip(snapshot.points, received['queries']):
        actual = goals[point.component_id] if point.frame_label in ('marker', 'switch_goal') else body
        if point.frame_label == 'switch_goal':
            path = result.marker_path
            endpoint = (np.asarray(control['goal_offset'])
                +quat_to_rotmat(control['goal_orientation_wxyz']) @ endpoints[point.axis])
        elif point.frame_label.startswith('IMU gravity at '):
            target = point.frame_label[len('IMU gravity at '):]
            path = result.link_paths[target]
            estimate = next(item for item in result.local_estimates
                if item.observed_direction is not None and item.diagnostics['target_link'] == target)
            endpoint = (np.zeros(3) if point.axis == 'origin' else
                        settings['axis_length']*estimate.observed_direction)
        else:
            path, endpoint = paths[point.frame_label], endpoints[point.axis]
        query = point_query(actual.state, path, endpoint, frame=result.root_frame)
        assert record['source_state_id'] == query.state_id == actual.state.state_id
        assert record['weight'] == actual.weight
        assert record['log_weight'] == actual.log_weight
        assert record['source_command'] == list(actual.state.command)
        assert record['source_stamp'] == actual.state.stamp
        np.testing.assert_array_equal(record['mean'], query.mean)
        np.testing.assert_array_equal(record['covariance'], query.covariance)
    assert {point.query.state_id for point in snapshot.points
            if point.frame_label not in ('marker', 'switch_goal')} == {body.state.state_id}
    assert {point.query.state_id for point in snapshot.points
            if point.frame_label in ('marker', 'switch_goal')} == {item.state.state_id for item in stage.goal_components}
