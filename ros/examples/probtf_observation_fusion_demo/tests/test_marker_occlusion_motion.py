"""Physical pixels cause marker loss while the acquired joint goal remains usable."""

from copy import deepcopy
from pathlib import Path
from queue import Empty
from types import SimpleNamespace
import io

import cv2
import numpy as np
import pytest
import rospy
import yaml
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker, MarkerArray

from probtf_observation_fusion_demo import async_workers
from probtf_observation_fusion_demo.board_camera import BoardCamera
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.motion_display import MotionDisplay
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore
from probtf_observation_fusion_demo.simple_control import SimpleSelector
from probtf_observation_fusion_demo.simple_inference import SimpleInference
from probtf_observation_fusion_demo.simple_motion import SimpleMotionSession
from test_async_execution import Context
from test_motion_display import Capture


CONFIG = Path(__file__).resolve().parents[1]/'config'
NEAR_COMMAND = np.array([-.0634057, -1.2239886, 1.1909804, -.0520916, -.7922031, .0002226])
FORBIDDEN_SENSOR_KEYS = {'actual_joints', 'true_K', 'camera_pose', 'marker_pose',
    'physical_goal', 'marker_occluder'}


def detected_ids(image):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        _, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
    else:
        _, ids, _ = cv2.aruco.detectMarkers(image, dictionary)
    return set() if ids is None else set(ids.ravel().tolist())


@pytest.mark.parametrize('override', [None, False, True])
def test_resolved_physical_occlusion_reaches_actual_acquisition_camera_without_sensor_side_flags(
        tmp_path, monkeypatch, override):
    session = SimpleMotionSession(CONFIG/'imu_marker.yaml', tmp_path/'camera',
        context=Context(), duration=.1, marker_occlusion_enabled=override)
    try:
        worker = session.workers[0]
        specification = worker.args[2]
        expected = deepcopy(session.scene.config['camera'])
        assert expected['marker_occluder']['enabled'] is bool(override)
        assert specification['camera'] == expected
        cameras = []
        original_factory = async_workers.make_camera

        def factory(*args, **kwargs):
            camera = original_factory(*args, **kwargs)
            cameras.append(camera)
            return camera

        monkeypatch.setattr(async_workers, 'make_camera', factory)
        with worker.args[0].mutex:
            image_request = next(item for item in worker.args[0].queue if item['kind'] == 'image')
        old_camera = BoardCamera(session.scene.texture_path, specification['board'],
            expected, specification['seed'])
        old_pixels = old_camera.render(image_request['camera_pose'], image_request['marker_pose'],
            image_request['camera_history'])
        worker.args[0].put(None)
        worker.target(*worker.args)
        replies = []
        while True:
            try:
                replies.append(session.acquisition_replies.get_nowait())
            except Empty:
                break
        assert not any(item.get('failed') for item in replies), replies
        assert len(cameras) == 1 and cameras[0].settings == expected
        image = next(item for item in replies if item['kind'] == 'image_ready')
        assert image['stamp'] == image_request['stamp']
        assert image['image'].shape == (expected['height'], expected['width'])
        assert image['preview']['visible'] is bool(detected_ids(image['image']))
        if override:
            assert not np.array_equal(image['image'], old_pixels)
            diagnostic = image['camera_diagnostics']['marker_occluder']
            assert diagnostic['enabled'] and not diagnostic['distance_trigger_used']
            assert not diagnostic['sensor_events_disabled']
        else:
            assert type(cameras[0]) is BoardCamera
            np.testing.assert_array_equal(image['image'], old_pixels)
        with SensorEventStore(session.output/'observations.sqlite3') as store:
            events = store.events()
        assert {event.kind for event in events} == {'joint', 'imu', 'image'}
        saved = next(event for event in events if event.kind == 'image')
        np.testing.assert_array_equal(saved.payload['image'], image['image'])
        assert saved.stamp == image_request['stamp'] and saved.command == tuple(image_request['command'])
        assert all(not (FORBIDDEN_SENSOR_KEYS & event.payload.keys()) for event in events)
    finally:
        session.close()


def test_real_marker_acquisition_then_physical_occlusion_retains_joint_goal_and_IMU_selection(
        tmp_path, monkeypatch):
    # All physical poses remain in the renderer. The estimator receives only
    # these original pixels, gravity directions, sent commands and timestamps.
    scene = MovingSwitchScene(tmp_path/'physical_world', marker_occlusion_enabled=True)
    visible = scene.step(0., stamp=1000.)
    scene.step(4., stamp=1004., command=NEAR_COMMAND)
    hidden = scene.step(4.2, stamp=1004.2, command=NEAR_COMMAND)
    tail = scene.step(4.21, stamp=1004.21, command=NEAR_COMMAND, render_camera=False)
    assert detected_ids(visible.image)
    assert not detected_ids(hidden.image)
    assert hidden.camera_diagnostics['marker_occluder']['current_occluded_board_fraction'] > .95
    assert not np.array_equal(hidden.image, np.full_like(hidden.image, 255))
    templates = tuple(yaml.safe_load((CONFIG/'imu_marker.yaml').read_text())['observations'])
    model = SimpleInference(scene.robot_description, templates)
    assert {item['sensor'] for item in model.templates} == {'imu', 'marker'}

    def forbid(*args, **kwargs):
        raise AssertionError('Inference and finite selection do not sample Monte Carlo candidates.')
    for name in ('normal', 'random', 'randn', 'default_rng', 'multivariate_normal'):
        monkeypatch.setattr(np.random, name, forbid)

    with SensorEventStore(tmp_path/'sensor_events.sqlite3') as store:
        first_event = store.append('image:visible', 'image', visible.stamp, visible.command,
            dict(time=visible.time, frame_index=0, image=visible.image, image_stamp=visible.stamp))
        missing_event = store.append('image:physically_occluded', 'image', hidden.stamp, hidden.command,
            dict(time=hidden.time, frame_index=1, image=hidden.image, image_stamp=hidden.stamp))
        imu_event = store.append('imu:occluded_tail', 'imu', tail.stamp, tail.command,
            dict(time=tail.time, frame_index=2, gravity_in_sensor=tail.gravity_in_sensor,
                imu_target_link=scene.imu_target_link, imu_sigma_rad=scene.imu_sigma_rad))
        first = model.update_sensor_event(first_event, event_index=0)
        before = first.stages[-1].components[0].state
        assert model.goal_observed and first.records['observation_ids'] == [first_event.event_id]
        missing = model.update_sensor_event(missing_event, event_index=1)
        carried = missing.stages[-1].components[0].state
        marker = before.edge_ids.index(model.marker.edge_id)
        coordinates = np.arange(6*marker, 6*marker+6)
        assert missing.records['observation_ids'] == []
        assert not missing.records['bayes_factor_surprise']['applied']
        np.testing.assert_allclose(carried.transforms[marker].translation,
            before.transforms[marker].translation, rtol=0., atol=1.e-13)
        np.testing.assert_allclose(carried.transforms[marker].bingham,
            before.transforms[marker].bingham, rtol=2.e-12, atol=1.e-13)
        np.testing.assert_allclose(carried.covariance[np.ix_(coordinates, coordinates)],
            before.covariance[np.ix_(coordinates, coordinates)], rtol=2.e-12, atol=1.e-14)
        assert model.goal_observed and not missing.stages[-1].goal_components
        assert missing.records['fixed_goal_memory']['observation_stamp'] == visible.stamp
        imu = model.update_sensor_event(imu_event, event_index=2)
        current = imu.stages[-1].components[0].state
        assert imu.records['observation_ids'] == [imu_event.event_id]
        assert imu.records['simple_inference']['goal_observed']
        assert imu.records['fixed_goal_memory']['observation_stamp'] == visible.stamp
        assert not imu.stages[-1].goal_components
        assert {item['sensor'] for item in model.templates} == {'imu', 'marker'}
        assert all(not (FORBIDDEN_SENSOR_KEYS & event.payload.keys()) for event in store.events())
    selector = SimpleSelector(model, scene.robot_description, model.settings['simple_reference'])
    q_next, diagnostics = selector.select(imu)
    assert diagnostics['goal_observed'] and diagnostics['S_after'] < diagnostics['S_before']
    assert np.linalg.norm(q_next-np.asarray(current.command)) > 0.
    assert diagnostics['body_source_state_ids'] == diagnostics['goal_source_state_ids'] == [current.state_id]
    assert not diagnostics['physical_truth_received']


def test_rviz_occluder_wire_uses_the_same_resolved_physical_geometry_as_camera(tmp_path, monkeypatch):
    config = yaml.safe_load((CONFIG/'switch_press_scene.yaml').read_text())
    # Nondefault dimensions and offset make an accidental display-only default
    # or a base-frame interpretation distinguishable from the renderer's box.
    geometry = dict(center=[.031, -.141, .29], size=[.31, .13],
        thickness=.007, intensity=79, enabled=False)
    config['camera']['marker_occluder'] = geometry
    config_path = tmp_path/'physical_scene.yaml'
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(rospy.Time, 'now', staticmethod(lambda: rospy.Time.from_sec(8000.)))
    for enabled in (False, True):
        scene = MovingSwitchScene(tmp_path/str(enabled), config_path,
            marker_occlusion_enabled=enabled)
        frame = scene.step(0., stamp=1000.)
        display = MotionDisplay.__new__(MotionDisplay)
        display.scene, display.case, display.sensor_names = scene, 'imu_marker', ('imu', 'marker')
        display.tag_texture = cv2.imread(str(scene.texture_path), cv2.IMREAD_GRAYSCALE)
        display.transforms, display.publisher = Capture(), Capture()
        display.marker_id = 0
        display.publish(frame, 0, SimpleNamespace(observation_ids=(), stamp=frame.stamp), 0)
        stream = io.BytesIO()
        display.publisher.messages.serialize(stream)
        wire = MarkerArray().deserialize(stream.getvalue())
        boxes = [marker for marker in wire.markers if marker.ns == 'marker_occluder']
        if not enabled:
            assert not boxes
            assert type(scene.camera) is BoardCamera
            continue
        assert len(boxes) == 1
        box = boxes[0]
        resolved = scene.camera.occluder_config
        assert {key: resolved[key] for key in geometry} == dict(geometry, enabled=True)
        assert resolved['frame'] == 'marker'
        assert box.type == Marker.CUBE and box.header.frame_id == 'base_link'
        expected_position = scene.marker[:3, :3]@np.asarray(resolved['center'])+scene.marker[:3, 3]
        np.testing.assert_allclose([box.pose.position.x, box.pose.position.y, box.pose.position.z],
            expected_position, rtol=0., atol=1.e-14)
        actual_rotation = Rotation.from_quat([box.pose.orientation.x, box.pose.orientation.y,
            box.pose.orientation.z, box.pose.orientation.w]).as_matrix()
        np.testing.assert_allclose(actual_rotation, scene.marker[:3, :3], rtol=0., atol=1.e-14)
        np.testing.assert_allclose([box.scale.x, box.scale.y, box.scale.z],
            [*resolved['size'], resolved['thickness']], rtol=0., atol=1.e-14)
        np.testing.assert_allclose([box.color.r, box.color.g, box.color.b, box.color.a],
            [resolved['intensity']/255.]*3+[1.], rtol=0., atol=2.e-8)
