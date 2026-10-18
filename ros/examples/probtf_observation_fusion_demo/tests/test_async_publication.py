"""Only the publication process draws/serializes; every raw event survives."""
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import pickle
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
from visualization_msgs.msg import MarkerArray

from probtf_observation_fusion_demo import async_motion, async_publication
from probtf_observation_fusion_demo import motion_display
from probtf_observation_fusion_demo.async_publication import (
    PublicationPacket, PublicationRenderer, publication_worker,
)
from probtf_observation_fusion_demo.display import make_markers, snapshot_record
from probtf_observation_fusion_demo.node import build_snapshot
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene

from test_measure_display_provenance import acquired, measured_source
from test_async_execution import Context


def wire(message):
    output = io.BytesIO()
    message.serialize(output)
    return output.getvalue()


class Publisher:
    def __init__(self, topic, message_type, **settings):
        self.topic, self.message_type, self.settings = topic, message_type, settings
        self.messages = []
    def publish(self, message):
        self.messages.append(message)


class Display:
    def __init__(self, scene, case, sensors):
        self.scene, self.case, self.sensors = scene, case, sensors
        self.calls = []
    def publish(self, frame, index, snapshot, capture_index, **kwargs):
        assert self.scene.trajectory[index]['stamp'] == frame.stamp
        assert kwargs['execution']['physical_stamp'] == frame.stamp
        assert kwargs['execution']['displayed_prediction_stamp'] == snapshot.stamp
        self.calls.append((frame,index,snapshot,capture_index,kwargs))


@pytest.fixture
def render_context(measured_source, tmp_path):
    inference, source = measured_source
    snapshot = build_snapshot(source, source.stages[-1], inference.common)
    display = inference.common['display']
    data = wire(make_markers(snapshot, display['seed'], display['point_size']))
    control = dict(request_id=1,event_id='image:acquired:0',index=0,capture_index=0,
        snapshot=snapshot,markers_wire=data,diagnostics=dict(
            body_source_state_ids=[item.state.state_id for item in source.stages[-1].components],
            goal_source_state_ids=[item.state.state_id for item in source.stages[-1].goal_components]))
    clock=[10.]
    specification=dict(output=str(tmp_path),case='marker',sensors=['marker'],display_hz=10.)
    renderer=PublicationRenderer(specification,scene_factory=lambda _:SimpleNamespace(trajectory=[]),
        display_factory=Display,publisher_factory=Publisher,clock=lambda:clock[0])
    return renderer,clock,control,source


def packet_for(control, *, sequence=0,index=0,start=0,rows=None,raw=(),preview=None,stamp=None):
    snapshot=control['snapshot']
    stamp=snapshot.stamp if stamp is None else stamp
    frame=SimpleNamespace(stamp=stamp,command=snapshot.command)
    row=dict(stamp=stamp,command=list(snapshot.command),command_interval_mode='searching')
    return PublicationPacket(sequence=sequence,enqueued_monotonic_s=9.,frame=frame,
        frame_index=index,trajectory_start=start,trajectory_rows=(row,) if rows is None else rows,
        raw_events=raw,metadata=dict(physical_stamp=stamp,command=list(snapshot.command),
            displayed_prediction_stamp=snapshot.stamp,
            displayed_prediction_command=list(snapshot.command),
            displayed_prediction_age_s=stamp-snapshot.stamp,estimation_drained=True),
        control=control,preview=preview)


def test_publication_process_preserves_wire_snapshot_pixels_and_every_raw_event(render_context, acquired):
    renderer,clock,control,source=render_context
    pixels=acquired[1].image
    raw=(dict(kind='joint',stamp=source.prior.stamp,command=source.prior.command,
              payload={'command_velocity':[.01]*6}),
         dict(kind='imu',stamp=source.prior.stamp,payload=dict(gravity_in_sensor=[0.,0.,-1.],
              imu_target_link='link_3',imu_sigma_rad=.018)),
         dict(kind='image',stamp=source.prior.stamp,image=pixels))
    packet=packet_for(control,raw=raw,preview=(0,pixels,source.prior.stamp))
    packet=pickle.loads(pickle.dumps(packet))
    acknowledgement=renderer.process(packet)
    assert acknowledgement['raw_counts']=={'joint':1,'imu':1,'image':1}
    assert wire(renderer.distributions.messages[0])==control['markers_wire']
    actual=MarkerArray().deserialize(wire(renderer.distributions.messages[0]))
    assert wire(actual)==control['markers_wire']
    metadata=json.loads(renderer.snapshots.messages[0].data)
    assert metadata==snapshot_record(control['snapshot'])
    assert metadata['candidate_measure']==source.records['candidate_measure']
    assert metadata['sensor_event']==source.records['sensor_event']
    assert renderer.raw_images.messages[0].data==pixels.tobytes()
    assert renderer.images.messages[0].data==pixels.tobytes()
    assert renderer.images.messages[0].height==pixels.shape[0]
    assert renderer.images.messages[0].width==pixels.shape[1]
    assert renderer.joint_commands.messages[0].position==list(source.prior.command)
    assert renderer.joint_commands.messages[0].velocity==[.01]*6
    assert renderer.raw_imu.messages[0].linear_acceleration.z==-9.80665
    manifest=json.loads((renderer.output/'motion_manifest.json').read_text())
    assert manifest['displayed_snapshot']['body_source_ids']==control['diagnostics']['body_source_state_ids']
    assert manifest['publication_process']['packet_sequence']==0
    assert manifest['publication_process']['queue_delay_s']==1.
    assert manifest['estimation_drained']
    assert not packet.control['snapshot'].points[0].query.covariance.flags.writeable


def test_original_camera_preview_is_sent_before_slow_scene_and_raw_publication(
        render_context, acquired, monkeypatch):
    renderer, clock, control, source = render_context
    pixels = acquired[1].image
    source_stamp = source.prior.stamp-3.
    physical_stamp = source.prior.stamp+.5
    original_scene_publish = renderer.display.publish
    original_raw_publish = renderer.joint_commands.publish
    seen = []

    def assert_preview_already_published():
        assert len(renderer.images.messages) == 1
        message = renderer.images.messages[0]
        assert message.header.stamp.to_sec() == source_stamp
        assert message.header.frame_id == 'camera_optical_frame'
        assert message.data == pixels.tobytes()
        assert (message.height, message.width) == pixels.shape[:2]

    def publish_raw(message):
        assert_preview_already_published()
        seen.append('raw')
        original_raw_publish(message)

    def slow_scene_publish(*args, **kwargs):
        assert_preview_already_published()
        seen.append('scene')
        # A blocked scene/manifest path must not hold back this packet's
        # already completed camera image or give it the current frame stamp.
        clock[0] += 10.
        original_scene_publish(*args, **kwargs)

    monkeypatch.setattr(renderer.joint_commands, 'publish', publish_raw)
    monkeypatch.setattr(renderer.display, 'publish', slow_scene_publish)
    event = dict(kind='joint', stamp=physical_stamp, command=source.prior.command,
        payload=dict(command_velocity=[.01]*6))
    packet = packet_for(control, stamp=physical_stamp, raw=(event,),
        preview=(0, pixels, source_stamp))
    acknowledgement = renderer.process(packet)
    assert seen == ['raw', 'scene']
    assert renderer.images.settings['queue_size'] == 1
    assert acknowledgement['raw_counts'] == dict(joint=1, imu=0, image=0)
    assert renderer.joint_commands.messages[0].header.stamp.to_sec() == physical_stamp
    assert len(renderer.trajectory) == 1
    assert renderer.trajectory[0]['stamp'] == physical_stamp
    assert len(renderer.distributions.messages) == len(renderer.snapshots.messages) == 1


def test_trajectory_deltas_cache_source_and_never_repeat_snapshot_or_drop_raw(render_context):
    renderer,clock,control,source=render_context
    renderer.process(packet_for(control))
    clock[0]+=.025
    packet=packet_for(control,sequence=1,index=1,start=1,stamp=source.prior.stamp+.025,
        raw=(dict(kind='joint',stamp=source.prior.stamp+.025,command=source.prior.command,
                  payload={'command_velocity':[0.]*6}),)*2)
    packet=replace(packet,control=None)
    ack=renderer.process(packet)
    assert len(renderer.trajectory)==2
    assert renderer.trajectory[-1]['command_interval_mode']=='searching'
    assert len(renderer.distributions.messages)==len(renderer.snapshots.messages)==1
    assert len(renderer.display.calls)==2
    assert renderer.display.calls[-1][-1]['publish_scene'] is False
    assert len(renderer.joint_commands.messages)==2
    assert ack['raw_counts']['joint']==2
    assert renderer.display.calls[-1][2] is renderer.current['snapshot']
    assert renderer.display.calls[-1][0].stamp-source.prior.stamp==pytest.approx(.025)


@pytest.mark.parametrize('change,match',[
    ({'sequence':2},'sequence'),({'trajectory_start':2},'gap, overlap'),
    ({'frame_index':3},'endpoint'),
])
def test_publication_rejects_missing_or_repeated_delta_before_any_raw_publish(render_context,change,match):
    renderer,_,control,_=render_context
    packet=replace(packet_for(control),**change)
    with pytest.raises(ValueError,match=match):renderer.process(packet)
    assert not renderer.joint_commands.messages
    assert not renderer.distributions.messages


def test_worker_owns_anonymous_ROS_node_and_drains_before_stopped(render_context,monkeypatch):
    renderer,_,control,_=render_context
    jobs,replies=Queue(),Queue()
    jobs.put(packet_for(control));jobs.put(None)
    initialized=[];shutdown=[]
    monkeypatch.setattr(async_publication,'PublicationRenderer',lambda _:renderer)
    monkeypatch.setattr(async_publication.rospy,'init_node',lambda *a,**k:initialized.append((a,k)))
    monkeypatch.setattr(async_publication.rospy,'get_name',lambda:'/independent_publication')
    monkeypatch.setattr(async_publication.rospy,'is_shutdown',lambda:False)
    monkeypatch.setattr(async_publication.rospy,'signal_shutdown',lambda reason:shutdown.append(reason))
    publication_worker(jobs,replies,renderer.specification)
    messages=[replies.get_nowait() for _ in range(3)]
    assert [item['kind'] for item in messages]==['ready','published','stopped']
    assert initialized[0][1]['anonymous'] and initialized[0][1]['disable_signals']
    assert messages[-1]['last_sequence']==0 and messages[-1]['graceful']
    assert shutdown


def test_worker_reports_display_failure_to_parent_and_does_not_claim_drain(render_context,monkeypatch):
    renderer,_,control,_=render_context
    jobs,replies=Queue(),Queue();jobs.put(packet_for(control));jobs.put(None)
    def fail(*_args,**_kwargs):raise RuntimeError('display failed')
    monkeypatch.setattr(renderer.display,'publish',fail)
    monkeypatch.setattr(async_publication,'PublicationRenderer',lambda _:renderer)
    monkeypatch.setattr(async_publication.rospy,'init_node',lambda *a,**k:None)
    monkeypatch.setattr(async_publication.rospy,'get_name',lambda:'/independent_publication')
    monkeypatch.setattr(async_publication.rospy,'is_shutdown',lambda:False)
    monkeypatch.setattr(async_publication.rospy,'signal_shutdown',lambda _:None)
    publication_worker(jobs,replies,renderer.specification)
    assert replies.get_nowait()['kind']=='ready'
    error=replies.get_nowait()
    assert error['failed'] and 'display failed' in error['traceback']
    assert not (renderer.output/'publication_shutdown.jsonl').exists()


@pytest.fixture
def live_wrapper(render_context,monkeypatch,tmp_path):
    renderer,_,control,source=render_context
    sessions=[];timers=[]
    class Session:
        def __init__(self,observations_path,output,**kwargs):
            self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
            self.context=Context();self.period=1/120.;self.rates={'joint':40.}
            self.settings={'display':{'publish_hz':10.}}
            self.scene=SimpleNamespace(config={},robot_description='',urdf='robot.urdf',
                texture_path='tag.png',world_base=np.eye(4),marker=np.eye(4),goal=np.zeros(3),
                tip_offset=np.zeros(3),imu_target_link='link_3',duration=kwargs.get('duration'),
                trajectory=[dict(stamp=source.prior.stamp,command=list(source.prior.command))])
            self.templates=();self.frames=[SimpleNamespace(stamp=source.prior.stamp,command=source.prior.command)]
            self.frame=self.frames[-1];self.current_control=control
            self.preview_capture_index=None;self.raw_publications=[]
            self.started=None;self.failure=None;self.closed=False;self.steps=0
            self.duration=kwargs.get('duration');sessions.append(self)
        def manifest(self):return dict(physical_stamp=self.frame.stamp,command=list(self.frame.command))
        def advance(self):self.steps+=1
        def start(self):self.started=0.;return True
        def close(self):self.closed=True
    class Timer:
        def __init__(self,*args):self.args=args;self.stopped=False;timers.append(self)
        def shutdown(self):self.stopped=True
    monkeypatch.setattr(async_motion,'AsyncMotionSession',Session)
    monkeypatch.setattr(async_motion.rospy,'Timer',Timer)
    monkeypatch.setattr(async_motion.rospy,'Service',lambda *a,**k:object())
    monkeypatch.setattr(async_motion.rospy,'on_shutdown',lambda _:None)
    monkeypatch.setattr(async_motion.rospy,'logerr',lambda *a:None)
    def forbidden(*a,**k):raise AssertionError('Main physical process must not own ROS publishers.')
    monkeypatch.setattr(async_motion.rospy,'Publisher',forbidden)
    live=async_motion.AsyncMotionLive('marker.yaml',tmp_path/'live','marker',True,duration=3.)
    yield live,sessions,timers
    live.close()


def test_main_timer_only_enqueues_deltas_and_new_control_without_drawing(live_wrapper):
    live,sessions,_=live_wrapper
    current=live.session
    current.raw_publications=[dict(kind='joint',stamp=current.frame.stamp)]
    live.publish(None)
    first=live.publication_requests.get_nowait()
    assert first.sequence==0 and len(first.trajectory_rows)==1
    assert first.control['snapshot'] is current.current_control['snapshot']
    assert current.raw_publications==[]
    live.publish(None)
    second=live.publication_requests.get_nowait()
    assert second.sequence==1 and second.trajectory_start==1 and second.trajectory_rows==()
    assert second.control is None
    assert second.metadata['publication_dispatch']['send_sequence']==1
    live.advance(None)
    assert current.steps==1 and current.started is not None


def test_publication_error_stops_physics_and_records_original_failure(live_wrapper):
    live,_,timers=live_wrapper
    live.publication_replies.put(dict(failed=True,worker='publication',traceback='original publication failure'))
    live.advance(None)
    assert live.closed and live.session.closed
    assert live.session.steps==0
    assert 'original publication failure' in live.session.failure['error']
    assert all(timer.stopped for timer in timers)
    assert (live.session.output/'execution_failure.jsonl').exists()


def test_real_display_scene_fields_are_preserved_and_full_draw_executes(measured_source,monkeypatch,tmp_path):
    inference,source=measured_source
    scene=MovingSwitchScene(tmp_path/'source_scene')
    frame=scene.step(0.,source.prior.stamp,render_camera=False)
    settings=inference.common['display']
    specification=async_publication.publication_specification(SimpleNamespace(
        scene=scene,output=tmp_path,templates=inference.templates,settings=inference.settings),'marker')
    class Broadcaster:
        def __init__(self):self.messages=[]
        def sendTransform(self,messages):self.messages.extend(messages)
    monkeypatch.setattr(motion_display,'TransformBroadcaster',Broadcaster)
    monkeypatch.setattr(motion_display,'StaticTransformBroadcaster',Broadcaster)
    monkeypatch.setattr(motion_display.rospy,'Publisher',Publisher)
    monkeypatch.setattr(motion_display.rospy,'set_param',lambda *a:None)
    monkeypatch.setattr(motion_display.rospy.Time,'now',staticmethod(lambda:motion_display.rospy.Time.from_sec(1234.)))
    renderer=PublicationRenderer(specification)
    assert renderer.scene.duration==scene.duration
    # Audit the actual adapter's attribute references, not a fake display's
    # reduced interface, so a newly required field cannot silently disappear.
    import ast,inspect
    tree=ast.parse(inspect.getsource(motion_display.MotionDisplay))
    fields={node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute)
        and isinstance(node.value,ast.Attribute) and node.value.attr=='scene'
        and isinstance(node.value.value,ast.Name) and node.value.value.id=='self'}
    assert not fields-set(vars(renderer.scene))
    snapshot=build_snapshot(source,source.stages[-1],inference.common)
    control=dict(request_id=1,event_id='image:acquired:0',index=0,capture_index=0,
        snapshot=snapshot,markers_wire=wire(make_markers(snapshot,settings['seed'],settings['point_size'])),
        diagnostics=dict(body_source_state_ids=[c.state.state_id for c in source.stages[-1].components],
                         goal_source_state_ids=[c.state.state_id for c in source.stages[-1].goal_components]))
    metadata=dict(execution_mode='async',physical_stamp=frame.stamp,command=list(frame.command),
        displayed_prediction_stamp=snapshot.stamp,displayed_prediction_command=list(snapshot.command),
        displayed_prediction_age_s=0.,observation_age_s=0.,actual_motion_hz=120.,actual_camera_hz=12.,
        actual_control_hz=1.,actual_estimate_hz=1.,finished=False,estimation_drained=False)
    packet=PublicationPacket(0,0.,frame,0,0,tuple(scene.trajectory),(),metadata,control)
    renderer.process(packet)
    assert renderer.display.publisher.messages
    assert renderer.display.transforms.messages
    assert wire(renderer.distributions.messages[0])==control['markers_wire']
    status=next(marker for marker in renderer.display.publisher.messages[0].markers if marker.ns=='status')
    assert '/ {:.1f} s'.format(scene.duration) in status.text


def test_dead_publisher_uses_exception_only_main_latched_failure_status(live_wrapper,monkeypatch):
    live,_,_=live_wrapper
    monkeypatch.setattr(async_motion.rospy,'Publisher',Publisher)
    live.publication_process.running=False
    live.publication_process.exitcode=2
    live.advance(None)
    assert live.closed
    assert live.failure_publisher.topic=='/observation_fusion/motion_metadata'
    assert live.failure_publisher.settings['latch']
    failure=json.loads(live.failure_publisher.messages[-1].data)
    assert failure['failed'] and 'exited unexpectedly: 2' in failure['error']


def test_replay_and_close_replace_both_sessions_and_publication_processes(live_wrapper):
    live,sessions,timers=live_wrapper
    live.advance(None)
    old,old_worker=live.session,live.publication_process
    response=live.replay(None)
    assert response.success
    assert old.closed and not old_worker.is_alive()
    assert live.session is not old and live.publication_process is not old_worker
    assert live.session.duration==3.
    assert live.trajectory_transmitted==0 and live.publication_sequence==0
    assert live.session.output.name=='live_replay_001'
    live.close();live.close()
    assert live.session.closed and all(timer.stopped for timer in timers)
    assert not live.publication_process.is_alive()
