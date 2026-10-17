"""Pixels and readings cross the sensor boundary; world geometry does not."""
from pathlib import Path
from queue import Queue
import hashlib

import cv2
import numpy as np

from probtf_observation_fusion_demo.async_workers import acquisition_worker
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore


def test_archive_worker_preserves_all_same_time_raw_sensors_and_exact_pixels(tmp_path):
    scene = MovingSwitchScene(tmp_path/'physical')
    frame = scene.step(0., 1000., render_camera=False)
    requests, replies = Queue(), Queue()
    common = dict(frame_index=0, stamp=frame.stamp, time=frame.time,
                  command=frame.command.tolist(), received_wall_stamp=1700000000.)
    requests.put(dict(common, event_id='j0', kind='joint',
                      payload=dict(command_velocity=[0.]*6)))
    requests.put(dict(common, event_id='i0', kind='imu', payload=dict(
        gravity_in_sensor=frame.gravity_in_sensor,
        imu_target_link=scene.imu_target_link, imu_sigma_rad=scene.imu_sigma_rad)))
    requests.put(dict(common, event_id='c0', kind='image',
        camera_pose=frame.camera_pose, marker_pose=scene.marker,
        camera_history=tuple(scene.camera_history)))
    requests.put(None)
    specification = dict(board=scene.config['marker_board'], camera=scene.config['camera'],
        texture=str(scene.texture_path), seed=scene.config['seed'],
        marker_size=scene.config['marker']['size'], marker_ids=tuple(range(17,26)))
    database = tmp_path/'observations.sqlite3'
    acquisition_worker(requests,replies,specification,str(database),str(tmp_path))
    answers = [replies.get_nowait() for _ in range(3)]
    assert not any(item.get('failed') for item in answers), answers
    with SensorEventStore(database) as store:
        events = store.events()
        assert [item.event_id for item in events] == ['j0','i0','c0']
        assert [item.kind for item in events] == ['joint','imu','image']
        assert all(item.stamp == frame.stamp for item in events)
        assert all(item.command == tuple(frame.command) for item in events)
        assert store.stats()['counts'] == dict(pending=3)
        np.testing.assert_array_equal(events[1].payload['gravity_in_sensor'],frame.gravity_in_sensor)
        pixels = events[2].payload['image']
        archived = cv2.imread(str(tmp_path/events[2].payload['image_path']),cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(pixels,archived)
        np.testing.assert_array_equal(pixels,answers[2]['image'])
        assert events[2].payload['image_sha256'] == hashlib.sha256(pixels.tobytes()).hexdigest()
        forbidden = {'camera_pose','marker_pose','camera_history','joints','stiffness','goal'}
        assert all(not (forbidden & set(item.payload)) for item in events)
        assert events[2].payload['image_stamp'] == frame.stamp
        assert tuple(pixels.shape) == (720,960)
