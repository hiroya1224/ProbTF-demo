"""The remembered switch target and the actual RViz wire use the IK sources."""

from dataclasses import replace
import json

import numpy as np
import pytest
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from probtf.geometry import quat_to_rotmat
from probtf.spherical_law.first_order import point_query
from probtf_observation_fusion_demo.display import (
    make_markers, snapshot_record, verify_wire_identity,
)
from probtf_observation_fusion_demo.node import build_snapshot, verify_adapter_identity

from test_internal_state_display import _inputs


@pytest.mark.parametrize('underflow', [False, True])
def test_switch_origin_and_axes_are_the_same_goal_mixture_as_control_and_wire(underflow):
    common, result, controller = _inputs(underflow)
    common['control'] = dict(goal_offset=controller.goal_offset,
                             goal_orientation_wxyz=controller.goal_orientation)
    result.records['fixed_goal_memory'] = dict(available=True, observation_stamp=1234.20)
    _, control = controller.step(result, .05)
    snapshot = build_snapshot(result, result.stages[-1], common)
    settings = common['display']
    markers = make_markers(snapshot, settings['seed'], settings['point_size'])
    wire = MarkerArray().deserialize(verify_wire_identity(snapshot, markers,
        seed=settings['seed'], point_size=settings['point_size']))
    metadata = String(json.dumps(snapshot_record(snapshot)))
    received = json.loads(metadata.data)
    assert snapshot.goal_memory is result.records['fixed_goal_memory']
    assert received['fixed_goal_memory'] == result.records['fixed_goal_memory']
    records = [item for item in received['queries'] if item['frame_label'] == 'switch_goal']
    points = [point for point in snapshot.points if point.frame_label == 'switch_goal']
    actual_markers = [marker for marker in wire.markers
        if marker.type == Marker.SPHERE_LIST and marker.ns.startswith('gaussian/switch_goal/')]
    assert len(points) == len(records) == len(actual_markers) == 8
    assert control['fixed_goal_memory_available']
    assert control['goal_memory_observation_stamp'] == 1234.20
    sources = {item.label: item for item in result.stages[-1].goal_components}
    diagnostics = {item['label']: item for item in control['root_goal_candidates']}
    assert set(sources) == set(diagnostics)
    endpoint = dict(origin=np.zeros(3), x=[settings['axis_length'], 0., 0.],
                    y=[0., settings['axis_length'], 0.], z=[0., 0., settings['axis_length']])
    rotation = quat_to_rotmat(controller.goal_orientation)
    for point, record, marker in zip(points, records, actual_markers):
        source = sources[point.component_id]
        expected = point_query(source.state, result.marker_path,
            controller.goal_offset+rotation @ endpoint[point.axis], frame=result.root_frame)
        control_point = next(item for item in diagnostics[source.label]['points']
                             if item['axis'] == point.axis)
        assert expected.state_id == record['source_state_id'] == diagnostics[source.label]['source_state_id']
        assert record['weight'] == source.weight == diagnostics[source.label]['weight']
        assert record['log_weight'] == source.log_weight == diagnostics[source.label]['log_weight']
        np.testing.assert_array_equal(record['mean'], expected.mean)
        np.testing.assert_array_equal(record['covariance'], expected.covariance)
        np.testing.assert_array_equal(control_point['mean'], expected.mean)
        np.testing.assert_array_equal(control_point['covariance'], expected.covariance)
        np.testing.assert_allclose(marker.color.a, .85*source.weight, atol=1e-7)
        assert marker.scale.x == pytest.approx(settings['point_size'])
        assert len(marker.points) == settings['marker_samples']
    if underflow:
        assert any(record['log_weight'] == -1000. and record['weight'] == 0.
                   for record in records)


@pytest.mark.parametrize('omission', ['mode', 'all', 'axis'])
def test_switch_target_display_rejects_missing_target_components_or_axes(omission):
    common, result, controller = _inputs()
    common['control'] = dict(goal_offset=controller.goal_offset,
                             goal_orientation_wxyz=controller.goal_orientation)
    result.records['fixed_goal_memory'] = dict(available=True, observation_stamp=1234.20)
    stage = result.stages[-1]
    snapshot = build_snapshot(result, stage, common)
    missing = stage.goal_components[-1].label
    points = tuple(point for point in snapshot.points
                   if not (point.frame_label == 'switch_goal' and (
                       omission == 'all' or (point.component_id == missing
                           and (omission == 'mode' or point.axis == 'x')))))
    with pytest.raises(AssertionError, match='switch-goal'):
        verify_adapter_identity(replace(snapshot, points=points), stage)
