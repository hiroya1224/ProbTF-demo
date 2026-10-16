"""Actual TF wire and bold arrows query each complete goal component."""

from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rospy
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from probtf.geometry import quat_to_rotmat
from probtf.spherical_law.first_order import point_query
from probtf_observation_fusion_demo.display import (
    goal_pose_records, make_markers, snapshot_record, verify_wire_identity,
)
from probtf_observation_fusion_demo.model import load_inputs
from probtf_observation_fusion_demo.motion_display import MotionDisplay
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.node import build_snapshot

from test_goal_axes_display import _fixture_snapshot


PACKAGE = Path(__file__).resolve().parents[1]


class CaptureTF:
    def __init__(self):
        self.payloads = []
        self.calls = []

    def sendTransform(self, transforms):
        # Audit what a real broadcaster sends over /tf, including float64
        # values and ROS integer seconds/nanoseconds, rather than the Python
        # records from which those messages happened to be constructed.
        message = TFMessage(transforms=list(transforms))
        stream = io.BytesIO()
        message.serialize(stream)
        self.payloads.append(stream.getvalue())
        self.calls.append(TFMessage().deserialize(stream.getvalue()))


class CaptureSources:
    def __init__(self):
        self.payloads = []
        self.calls = []

    def publish(self,message):
        stream = io.BytesIO()
        message.serialize(stream)
        self.payloads.append(stream.getvalue())
        self.calls.append(json.loads(String().deserialize(stream.getvalue()).data))


def _display(monkeypatch):
    wall = [8000.5]
    monkeypatch.setattr(rospy.Time,'now',staticmethod(lambda:rospy.Time.from_sec(wall[0])))
    display = MotionDisplay.__new__(MotionDisplay)
    display.transforms = CaptureTF()
    display.goal_tf_sources = CaptureSources()
    return display,wall


def _publish(display,frame,snapshot,index=0):
    metadata = dict(execution_mode='async',physical_stamp=frame.stamp,
        command=list(frame.command),displayed_prediction_stamp=snapshot.stamp,
        displayed_prediction_command=list(snapshot.command),
        displayed_prediction_age_s=frame.stamp-snapshot.stamp)
    display.publish(frame,index,snapshot,index,execution=metadata,publish_scene=False)
    return display.transforms.calls[-1]


def _goals(message):
    return [transform for transform in message.transforms
            if transform.child_frame_id.startswith('estimated_switch_goal_')]


def _translation(transform):
    value = transform.transform.translation
    return np.array([value.x,value.y,value.z])


def _quaternion(transform):
    value = transform.transform.rotation
    return np.array([value.w,value.x,value.y,value.z])


def _assert_source_wire(display,snapshot,physical,render_stamp,transforms):
    source = display.goal_tf_sources.calls[-1]
    expected = (rospy.Time.from_sec(render_stamp)
        -rospy.Duration.from_sec(physical.stamp-snapshot.stamp))
    assert source['source_stamp'] == snapshot.stamp
    assert source['physical_source_stamp'] == physical.stamp
    assert source['render_stamp_ros'] == rospy.Time.from_sec(render_stamp).to_sec()
    assert source['source_age_s'] == physical.stamp-snapshot.stamp
    assert source['clock_mapping'] == 'render_ros_stamp - (physical_sim_stamp - candidate_sim_stamp)'
    assert source['tf_stamp_ros_parts'] == dict(secs=expected.secs,nsecs=expected.nsecs)
    assert source['tf_stamp_ros'] == expected.to_sec()
    assert source['goal_poses'] == goal_pose_records(snapshot)
    assert len(source['goal_poses']) == len(transforms)
    for transform,pose in zip(transforms,source['goal_poses']):
        assert transform.header.stamp == expected
        assert (transform.header.stamp.secs,transform.header.stamp.nsecs) == (
            source['tf_stamp_ros_parts']['secs'],source['tf_stamp_ros_parts']['nsecs'])
        assert transform.header.frame_id == pose['source_frame']
        assert transform.child_frame_id == pose['child_frame_id']
        np.testing.assert_array_equal(_translation(transform),pose['translation'])
        np.testing.assert_array_equal(_quaternion(transform),pose['rotation_wxyz'])
    return source


def _arrow_wire(snapshot,common):
    settings = common['display']
    markers = make_markers(snapshot,settings['seed'],settings['point_size'])
    payload = verify_wire_identity(snapshot,markers,
        seed=settings['seed'],point_size=settings['point_size'])
    return [marker for marker in MarkerArray().deserialize(payload).markers
            if marker.ns.startswith('goal_axes/')]


@pytest.mark.parametrize('underflow',[False,True])
def test_all_goal_TF_wire_poses_equal_core_modal_queries_and_actual_arrow_endpoints(
        tmp_path,monkeypatch,underflow):
    common,result,controller,snapshot = _fixture_snapshot(underflow)
    scene = MovingSwitchScene(tmp_path)
    physical = scene.step(0.,stamp=snapshot.stamp+.2)
    display,wall = _display(monkeypatch)
    before = json.dumps(snapshot_record(snapshot),sort_keys=True)
    message = _publish(display,physical,snapshot)
    goals,records,arrows = _goals(message),goal_pose_records(snapshot),_arrow_wire(snapshot,common)
    assert len(goals) == len(result.stages[-1].goal_components) == 2
    assert len(arrows) == 3*len(goals)
    source_metadata = _assert_source_wire(display,snapshot,physical,wall[0],goals)
    sources = {item.label:item for item in result.stages[-1].goal_components}
    rotation_offset = quat_to_rotmat(controller.goal_orientation)
    for index,(transform,record) in enumerate(zip(goals,records)):
        assert transform.child_frame_id == record['child_frame_id'] == 'estimated_switch_goal_'+str(index)
        assert transform.header.frame_id == record['source_frame'] == result.root_frame
        assert transform.header.stamp.to_sec() == source_metadata['tf_stamp_ros']
        assert record['source_stamp'] == snapshot.stamp
        assert transform.header.stamp != rospy.Time.from_sec(snapshot.stamp)
        assert transform.header.stamp.to_sec() != physical.stamp
        assert transform.header.stamp.to_sec() != wall[0]
        np.testing.assert_array_equal(_translation(transform),record['translation'])
        np.testing.assert_array_equal(_quaternion(transform),record['rotation_wxyz'])
        source = sources[record['component_id']]
        independent_origin = point_query(source.state,result.marker_path,controller.goal_offset)
        np.testing.assert_array_equal(_translation(transform),independent_origin.mode)
        assert record['source_state_id'] == source.state.state_id
        assert record['weight'] == source.weight and record['log_weight'] == source.log_weight
        rotation = quat_to_rotmat(_quaternion(transform))
        for axis_index,axis in enumerate(('x','y','z')):
            arrow = arrows[3*index+axis_index]
            assert arrow.type == Marker.ARROW
            assert arrow.header.frame_id == transform.header.frame_id
            # Candidate/ARROW source time stays in its acquisition clock.
            # Only TF transport applies the explicitly recorded ROS mapping.
            assert arrow.header.stamp == rospy.Time.from_sec(record['source_stamp'])
            provenance = json.loads(arrow.text)
            assert provenance['source_state_id'] == record['source_state_id']
            assert provenance['child_frame_id'] == transform.child_frame_id
            assert provenance['log_weight'] == record['log_weight']
            start,end = [np.array([p.x,p.y,p.z]) for p in arrow.points]
            np.testing.assert_array_equal(start,_translation(transform))
            np.testing.assert_array_equal(end,record['modal_endpoints'][axis])
            endpoint = np.zeros(3);endpoint[axis_index] = common['display']['axis_length']
            independent_end = point_query(source.state,result.marker_path,
                controller.goal_offset+rotation_offset@endpoint)
            np.testing.assert_array_equal(end,independent_end.mode)
            np.testing.assert_allclose(_translation(transform)+rotation@endpoint,end,atol=1e-14,rtol=0.)
    physical_transforms = [t for t in message.transforms if t not in goals]
    assert physical_transforms and all(t.header.stamp == rospy.Time.from_sec(wall[0]) for t in physical_transforms)
    assert all(t.header.frame_id == 'base_link' for t in physical_transforms)
    assert len(physical_transforms) == 2*(len(physical.link_poses)-1)
    assert json.dumps(snapshot_record(snapshot),sort_keys=True) == before
    if underflow:
        assert records[1]['weight'] == 0. and records[1]['log_weight'] == -1000.
        assert goals[1].child_frame_id == 'estimated_switch_goal_1'
    # The true scene's switch is deliberately far from these test candidates.
    assert all(np.linalg.norm(_translation(t)-scene.goal) > .1 for t in goals)


def test_goal_heartbeat_duplicates_are_omitted_while_physical_TF_keeps_advancing(tmp_path,monkeypatch):
    _,_,_,snapshot = _fixture_snapshot()
    scene = MovingSwitchScene(tmp_path)
    first = scene.step(0.,stamp=snapshot.stamp+.2)
    second = scene.step(.05,stamp=snapshot.stamp+.25,
        command=scene.initial_command+np.array([.01,-.005,.003,.006,0.,0.]))
    display,wall = _display(monkeypatch)
    initial = _publish(display,first,snapshot)
    original_goals = deepcopy(_goals(initial))
    assert len(original_goals) == 2
    original_source = _assert_source_wire(display,snapshot,first,wall[0],original_goals)
    wall[0] += .05
    heartbeat = _publish(display,second,snapshot,1)
    assert _goals(heartbeat) == []
    assert heartbeat.transforms and all(t.header.stamp == rospy.Time.from_sec(wall[0]) for t in heartbeat.transforms)
    physical1 = next(t for t in initial.transforms if t.child_frame_id == 'tool0')
    physical2 = next(t for t in heartbeat.transforms if t.child_frame_id == 'tool0')
    assert not np.array_equal(_translation(physical1),_translation(physical2))
    for index in range(3):
        wall[0] += .05
        assert _goals(_publish(display,second,snapshot,index+2)) == []
    all_goals = [t for message in display.transforms.calls for t in _goals(message)]
    assert all_goals == original_goals
    assert display.goal_tf_sources.calls == [original_source]


def test_each_new_goal_source_is_published_on_first_hold_and_reobservation(tmp_path,monkeypatch):
    scene = MovingSwitchScene(tmp_path)
    common,observations = load_inputs(PACKAGE/'config/common.yaml',PACKAGE/'config/imu_marker.yaml')
    inference = MovingInference(scene.robot_description,observations)
    common = inference.common
    physical0 = scene.step(0.,stamp=1000.)
    first = inference.update(scene.sensor_inputs(physical0),0)
    assert first.records['fixed_goal_memory_event'] == 'marker_observation_update'
    snapshot0 = build_snapshot(first,first.stages[-1],common)
    display,wall = _display(monkeypatch)
    tf0 = _goals(_publish(display,physical0,snapshot0,0))
    records0 = goal_pose_records(snapshot0)
    assert len(tf0) == len(first.stages[-1].goal_components)
    source0 = _assert_source_wire(display,snapshot0,physical0,wall[0],tf0)

    physical1 = scene.step(.05,stamp=1000.05,command=scene.initial_command+.001)
    inputs = scene.sensor_inputs(physical1)
    held = inference.update(replace(inputs,image=np.full_like(inputs.image,255)),1)
    assert held.records['fixed_goal_memory_event'] == 'image_missing_hold'
    assert held.records['fixed_goal_memory']['observation_stamp'] == 1000.
    snapshot1 = build_snapshot(held,held.stages[-1],common)
    wall[0] += .05
    tf1 = _goals(_publish(display,physical1,snapshot1,1))
    records1 = goal_pose_records(snapshot1)
    source1 = _assert_source_wire(display,snapshot1,physical1,wall[0],tf1)
    assert len(tf1) == len(tf0)
    for before,after,old_source,new_source in zip(tf0,tf1,records0,records1):
        np.testing.assert_array_equal(_translation(after),_translation(before))
        np.testing.assert_array_equal(_quaternion(after),_quaternion(before))
        assert before.header.stamp.to_sec() == source0['tf_stamp_ros']
        assert after.header.stamp.to_sec() == source1['tf_stamp_ros']
        assert after.header.stamp > before.header.stamp
        assert new_source['source_state_id'] != old_source['source_state_id']
        assert new_source['goal_observation_stamp'] == old_source['goal_observation_stamp'] == 1000.
    assert _goals(_publish(display,physical1,snapshot1,1)) == []

    physical2 = scene.step(.10,stamp=1000.10,command=scene.initial_command)
    reacquired = inference.update(scene.sensor_inputs(physical2),2)
    assert reacquired.records['fixed_goal_memory_event'] == 'marker_observation_update'
    snapshot2 = build_snapshot(reacquired,reacquired.stages[-1],common)
    wall[0] += .05
    tf2 = _goals(_publish(display,physical2,snapshot2,2))
    records2 = goal_pose_records(snapshot2)
    _assert_source_wire(display,snapshot2,physical2,wall[0],tf2)
    assert len(tf2) == len(reacquired.stages[-1].goal_components)
    assert all(t.header.stamp > tf1[0].header.stamp for t in tf2)
    assert all(r['goal_observation_stamp'] == 1000.10 for r in records2)
    assert all(r['source_state_id'] not in {s['source_state_id'] for s in records1} for r in records2)
    assert any(not np.array_equal(_translation(a),_translation(b))
               or not np.array_equal(_quaternion(a),_quaternion(b)) for a,b in zip(tf0,tf2))
    assert _goals(_publish(display,physical2,snapshot2,2)) == []
    assert len(display.goal_tf_sources.calls) == 3


def test_physical_truth_changes_cannot_recenter_an_immutable_goal_snapshot(tmp_path,monkeypatch):
    _,_,_,snapshot = _fixture_snapshot()
    scene = MovingSwitchScene(tmp_path)
    physical = scene.step(0.,stamp=snapshot.stamp+.1)
    shifted = {}
    for name,pose in physical.link_poses.items():
        shifted[name] = pose.copy()
        shifted[name][:3,3] += [4.,-3.,2.]
    changed_truth = replace(physical,link_poses=shifted,rigid_link_poses=shifted)
    first,_ = _display(monkeypatch)
    second,_ = _display(monkeypatch)
    before = _goals(_publish(first,physical,snapshot))
    after = _goals(_publish(second,changed_truth,snapshot))
    assert before == after
    assert len(before) == len(goal_pose_records(snapshot))


def test_replay_can_rewind_simulation_sources_without_rewinding_ROS_TF(tmp_path,monkeypatch):
    common,result,_,initial = _fixture_snapshot()
    source_stage = result.stages[-1]
    def later_component(component):
        return replace(component,state=replace(component.state,
            stamp=component.state.stamp+2.,version=component.state.version+1))
    later_stage = SimpleNamespace(**dict(vars(source_stage),
        components=tuple(map(later_component,source_stage.components)),
        goal_components=tuple(map(later_component,source_stage.goal_components))))
    later_result = SimpleNamespace(**dict(vars(result),prior=later_stage.components[0].state,
        stages=(later_stage,)))
    later = build_snapshot(later_result,later_stage,common)
    scene = MovingSwitchScene(tmp_path/'first_run')
    physical0 = scene.step(0.,stamp=initial.stamp+.2)
    physical1 = scene.step(.05,stamp=later.stamp+.2)
    replay_scene = MovingSwitchScene(tmp_path/'replay')
    replay_physical = replay_scene.step(0.,stamp=initial.stamp+.2)
    display,wall = _display(monkeypatch)
    original_record = json.dumps(snapshot_record(initial),sort_keys=True)
    first = _goals(_publish(display,physical0,initial))
    _assert_source_wire(display,initial,physical0,wall[0],first)
    wall[0] += 2.
    second = _goals(_publish(display,physical1,later,1))
    _assert_source_wire(display,later,physical1,wall[0],second)
    wall[0] += 8.
    replayed = _goals(_publish(display,replay_physical,initial,0))
    replay_source = _assert_source_wire(display,initial,replay_physical,wall[0],replayed)
    assert initial.stamp < later.stamp
    assert replay_source['source_stamp'] == initial.stamp
    assert replay_source['goal_poses'] == goal_pose_records(initial)
    assert len(first) == len(second) == len(replayed) == 2
    for before,advanced,replay in zip(first,second,replayed):
        assert before.header.stamp < advanced.header.stamp < replay.header.stamp
        np.testing.assert_array_equal(_translation(replay),_translation(before))
        np.testing.assert_array_equal(_quaternion(replay),_quaternion(before))
    assert json.dumps(snapshot_record(initial),sort_keys=True) == original_record
    assert len(display.goal_tf_sources.calls) == 3
