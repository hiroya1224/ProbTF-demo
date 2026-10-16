"""Bold goal arrows are the same full goal mixture as the core and IK."""

from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path

import numpy as np
import pytest
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from probtf.geometry import quat_mul, quat_to_rotmat
from probtf.spherical_law.first_order import point_query
from probtf_observation_fusion_demo import display as adapter
from probtf_observation_fusion_demo.display import (
    AXIS_COLORS, goal_axis_records, goal_pose_records, make_markers, serialized_marker_array,
    snapshot_record, verify_wire_identity,
)
from probtf_observation_fusion_demo.model import load_inputs
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.node import build_snapshot

from test_internal_state_display import _inputs


PACKAGE = Path(__file__).resolve().parents[1]


def _fixture_snapshot(underflow=False):
    common, result, controller = _inputs(underflow)
    common['control'] = dict(goal_offset=controller.goal_offset,
                             goal_orientation_wxyz=controller.goal_orientation)
    result.records['fixed_goal_memory'] = dict(available=True,
        observation_stamp=1234.20, observation_command=[-.01]*6,
        observation_version=15, observation_ids=['marker:frame:00004'])
    return common, result, controller, build_snapshot(result, result.stages[-1], common)


def _wire(snapshot, common):
    settings = common['display']
    message = make_markers(snapshot, settings['seed'], settings['point_size'])
    payload = verify_wire_identity(snapshot, message,
        seed=settings['seed'], point_size=settings['point_size'])
    return MarkerArray().deserialize(payload)


def _arrows(message):
    return [marker for marker in message.markers if marker.ns.startswith('goal_axes/')]


def _coordinates(marker):
    return np.array([[point.x, point.y, point.z] for point in marker.points])


@pytest.mark.parametrize('underflow', [False, True])
def test_every_goal_component_has_three_bold_arrows_with_exact_core_endpoints_and_wire_provenance(underflow):
    common, result, controller, snapshot = _fixture_snapshot(underflow)
    _, diagnostics = controller.step(result, .05)
    message = _wire(snapshot, common)
    arrows = _arrows(message)
    assert len(arrows) == 3*len(result.stages[-1].goal_components) == 6
    metadata = String(json.dumps(snapshot_record(snapshot)))
    stream = io.BytesIO(); metadata.serialize(stream)
    received = json.loads(String().deserialize(stream.getvalue()).data)
    assert len(received['goal_axes']) == len(arrows)
    sources = {item.label: item for item in result.stages[-1].goal_components}
    point_records = {item['query_id']: item for item in received['queries']}
    control_goals = {item['label']: item for item in diagnostics['root_goal_candidates']}
    rotation = quat_to_rotmat(controller.goal_orientation)
    for arrow, record in zip(arrows, received['goal_axes']):
        source = sources[record['component_id']]
        assert arrow.type == Marker.ARROW and arrow.action == Marker.ADD
        assert arrow.ns == record['namespace'] == 'goal_axes/'+source.label
        assert arrow.id == record['marker_id'] == ('x', 'y', 'z').index(record['axis'])
        assert arrow.header.frame_id == record['source_frame'] == result.root_frame
        assert arrow.header.stamp.to_sec() == record['source_stamp'] == source.state.stamp
        assert record['source_state_id'] == source.state.state_id
        assert record['source_component_id'] == source.state.component_id
        assert record['source_version'] == snapshot.version == source.state.version
        assert record['source_command'] == list(source.state.command)
        assert record['weight'] == source.weight and record['log_weight'] == source.log_weight
        assert json.loads(arrow.text) == record
        assert record['goal_observation_stamp'] == 1234.20
        assert record['goal_observation_version'] == 15
        assert record['goal_observation_command'] == [-.01]*6
        assert record['goal_observation_ids'] == ['marker:frame:00004']
        endpoint = np.zeros(3)
        endpoint[arrow.id] = common['display']['axis_length']
        expected_origin = point_query(source.state, result.marker_path, controller.goal_offset)
        expected_end = point_query(source.state, result.marker_path,
            controller.goal_offset+rotation @ endpoint)
        np.testing.assert_array_equal(_coordinates(arrow), [expected_origin.mode, expected_end.mode])
        np.testing.assert_array_equal(record['start'], point_records[record['origin_query_id']]['mode'])
        np.testing.assert_array_equal(record['end'], point_records[record['endpoint_query_id']]['mode'])
        controlled = next(item for item in control_goals[source.label]['points'] if item['axis'] == record['axis'])
        np.testing.assert_array_equal(point_records[record['endpoint_query_id']]['mean'], controlled['mean'])
        assert record['operation'] == 'modal frame query of same goal law'
        np.testing.assert_array_equal([arrow.scale.x, arrow.scale.y, arrow.scale.z], [.012, .024, .024])
        # The core modal endpoints set the complete arrow length. The
        # existing mean endpoints remain unchanged in the thin axes/clouds.
        assert np.linalg.norm(np.diff(_coordinates(arrow), axis=0)) == pytest.approx(
            np.linalg.norm(expected_end.mode-expected_origin.mode), rel=1e-14)
        expected_color = np.array([*AXIS_COLORS[record['axis']], .95*source.weight], dtype=np.float32)
        np.testing.assert_array_equal([arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a], expected_color)
    if underflow:
        tiny = [json.loads(arrow.text) for arrow in arrows if json.loads(arrow.text)['weight'] == 0.]
        assert len(tiny) == 3 and all(item['log_weight'] == -1000. for item in tiny)

    poses = goal_pose_records(snapshot)
    assert received['goal_poses'] == poses and len(poses) == len(sources)
    for index, (pose, source) in enumerate(zip(poses, sources.values())):
        assert pose['child_frame_id'] == 'estimated_switch_goal_'+str(index)
        assert pose['component_index'] == index
        assert pose['component_id'] == source.label
        assert pose['source_state_id'] == source.state.state_id
        assert pose['source_stamp'] == source.state.stamp
        assert pose['source_command'] == list(source.state.command)
        assert pose['weight'] == source.weight and pose['log_weight'] == source.log_weight
        marker = source.state.transforms[-1]
        expected_rotation = quat_to_rotmat(quat_mul(marker.rotation_wxyz, controller.goal_orientation))
        expected_position = marker.translation+quat_to_rotmat(marker.rotation_wxyz) @ controller.goal_offset
        np.testing.assert_allclose(pose['translation'], expected_position, atol=1e-15)
        np.testing.assert_allclose(quat_to_rotmat(pose['rotation_wxyz']), expected_rotation, atol=1e-14)
        assert pose['axis_length_m'] == pytest.approx(common['display']['axis_length'], abs=1e-15)
        for axis_index, axis in enumerate(('x', 'y', 'z')):
            arrow = arrows[3*index+axis_index]
            np.testing.assert_array_equal(pose['modal_endpoints'][axis], _coordinates(arrow)[1])
            assert pose['endpoint_query_ids'][axis] == json.loads(arrow.text)['endpoint_query_id']


def test_appended_goal_axes_leave_every_existing_particle_axis_label_and_size_byte_identical(monkeypatch):
    common, _, _, snapshot = _fixture_snapshot()
    settings = common['display']
    current = make_markers(snapshot, settings['seed'], settings['point_size'])
    with monkeypatch.context() as context:
        context.setattr(adapter, 'goal_axis_records', lambda _: [])
        existing = make_markers(snapshot, settings['seed'], settings['point_size'])
    assert serialized_marker_array(MarkerArray(current.markers[:-6])) == serialized_marker_array(existing)
    for marker in existing.markers:
        if marker.ns.startswith('gaussian/'):
            assert marker.scale.x == marker.scale.y == marker.scale.z == settings['point_size']
        elif marker.ns.startswith('axes/'):
            assert marker.scale.x == .00225


@pytest.mark.parametrize('field', ['endpoint', 'thickness', 'metadata', 'missing_component'])
def test_actual_goal_arrow_wire_tampering_is_rejected(field):
    common, _, _, snapshot = _fixture_snapshot()
    message = _wire(snapshot, common)
    arrow = _arrows(message)[0]
    if field == 'endpoint':
        arrow.points[1].x += .01
    elif field == 'thickness':
        arrow.scale.x *= 2.
    elif field == 'metadata':
        data = json.loads(arrow.text); data['log_weight'] -= 1.
        arrow.text = json.dumps(data)
    else:
        missing = _arrows(message)[-1].ns
        message.markers = [marker for marker in message.markers if marker.ns != missing]
    settings = common['display']
    with pytest.raises(AssertionError, match='differs from the core snapshot'):
        verify_wire_identity(snapshot, message, seed=settings['seed'], point_size=settings['point_size'])


@pytest.mark.parametrize('field', ['version', 'stamp', 'command', 'weight', 'log_weight', 'missing_axis'])
def test_goal_axes_reject_incomplete_or_mismatched_source_queries(field):
    _, _, _, snapshot = _fixture_snapshot()
    points = list(snapshot.points)
    index = next(index for index, point in enumerate(points)
                 if point.frame_label == 'switch_goal' and point.axis == 'x')
    original = points[index]
    if field == 'missing_axis':
        del points[index]
    elif field in ('weight', 'log_weight'):
        points[index] = replace(original, **{field: getattr(original, field)-.01})
    else:
        value = {'version': original.query.version+1, 'stamp': original.query.stamp+.01,
                 'command': (.01,)+(0.,)*5}[field]
        points[index] = replace(original, query=replace(original.query, **{field:value}))
    with pytest.raises(ValueError, match='Bold goal axes'):
        goal_axis_records(replace(snapshot, points=tuple(points)))


@pytest.mark.parametrize('invalid_basis', ['zero_length', 'reflection'])
def test_modal_TF_does_not_project_invalid_mean_or_modal_axes_into_a_rotation(invalid_basis):
    _, _, _, snapshot = _fixture_snapshot()
    points = list(snapshot.points)
    index = next(index for index, point in enumerate(points)
                 if point.frame_label == 'switch_goal' and point.axis == 'x')
    target = points[index]
    origin = next(point for point in points if point.frame_label == 'switch_goal'
                  and point.component_id == target.component_id and point.axis == 'origin')
    mode = (origin.query.mode if invalid_basis == 'zero_length' else
            2.*origin.query.mode-target.query.mode)
    points[index] = replace(target, query=replace(target.query, mode=mode))
    with pytest.raises(ValueError, match='Modal goal TF queries'):
        goal_pose_records(replace(snapshot, points=tuple(points)))


def test_new_marker_updates_goal_arrows_and_missing_images_hold_the_remembered_root_target(tmp_path):
    scene = MovingSwitchScene(tmp_path)
    inputs = scene.sensor_inputs(scene.step(0., 1000.))
    _, observations = load_inputs(PACKAGE/'config/common.yaml', PACKAGE/'config/imu_marker.yaml')
    inference = MovingInference(scene.robot_description, observations)
    first = inference.update(inputs, 0)
    assert first.records['fixed_goal_memory']['available']
    common = inference.common
    initial_snapshot = build_snapshot(first, first.stages[-1], common)
    initial = _arrows(_wire(initial_snapshot, common))
    memory = deepcopy(first.records['fixed_goal_memory'])
    missing = inference.update(replace(inputs, image=np.full_like(inputs.image, 255),
        stamp=1000.05, time=.05, command=inputs.command+.001), 1)
    assert missing.records['fixed_goal_memory'] == memory
    missing_snapshot = build_snapshot(missing, missing.stages[-1], common)
    retained = _arrows(_wire(missing_snapshot, common))
    assert len(retained) == len(initial)
    for before, after in zip(initial, retained):
        np.testing.assert_array_equal(_coordinates(after), _coordinates(before))
        old, new = json.loads(before.text), json.loads(after.text)
        assert (new['weight'], new['log_weight']) == (old['weight'], old['log_weight'])
        assert new['goal_observation_stamp'] == old['goal_observation_stamp'] == 1000.
        assert new['source_stamp'] == 1000.05 and old['source_stamp'] == 1000.

    reacquired = inference.update(replace(inputs, stamp=1000.10, time=.10), 2)
    assert reacquired.records['fixed_goal_memory_event'] == 'marker_observation_update'
    updated = _arrows(_wire(build_snapshot(reacquired, reacquired.stages[-1], common), common))
    assert len(updated) == 3*len(reacquired.stages[-1].goal_components)
    assert all(json.loads(arrow.text)['goal_observation_stamp'] == 1000.10 for arrow in updated)
    assert all(json.loads(arrow.text)['source_stamp'] == 1000.10 for arrow in updated)
    # This repeated valid observation changes the actual inferred target;
    # there is no retained first-frame pose in the renderer.
    assert any(not np.array_equal(_coordinates(a), _coordinates(b)) for a, b in zip(initial, updated))


def test_goal_axis_rendering_uses_no_sampling_or_source_mutation(monkeypatch):
    common, _, _, snapshot = _fixture_snapshot()
    before = [(point.query.mean.copy(), point.query.covariance.copy()) for point in snapshot.points]
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Goal axes must use existing core queries without sampling.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    first, second = _wire(snapshot, common), _wire(snapshot, common)
    assert serialized_marker_array(first) == serialized_marker_array(second)
    for point, (mean, covariance) in zip(snapshot.points, before):
        np.testing.assert_array_equal(point.query.mean, mean)
        np.testing.assert_array_equal(point.query.covariance, covariance)
