"""Direct measured-image publication preserves pixels, source clock and order."""

import builtins
import json
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from probtf_observation_fusion_demo.camera_live_preview import make_live_preview_publisher
from probtf_observation_fusion_demo import async_workers as workers


class Image:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None, frame_id=None)


class Ros:
    def __init__(self, initialized=False):
        self.initialized, self.messages, self.initializations, self.shutdowns = initialized, [], [], []
        self.core = SimpleNamespace(is_initialized=lambda: self.initialized)
        self.Time = SimpleNamespace(from_sec=lambda stamp: ('original_source_stamp', stamp))
        self.publisher_calls, self.unregistered = [], 0
    def init_node(self, name, **kwargs):
        self.initializations.append((name, kwargs))
        self.initialized = True
    def Publisher(self, topic, message_type, **kwargs):
        self.publisher_calls.append((topic, message_type, kwargs))
        return SimpleNamespace(publish=self.messages.append, unregister=self.unregister)
    def unregister(self):
        self.unregistered += 1
    def signal_shutdown(self, reason):
        self.shutdowns.append(reason)


def test_headless_factory_does_not_import_ros(monkeypatch):
    original = builtins.__import__
    def forbid_ros(name, *args, **kwargs):
        if name == 'rospy' or name.startswith('sensor_msgs'):
            raise AssertionError('A headless camera must not import ROS.')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', forbid_ros)
    assert make_live_preview_publisher({}) is None
    assert make_live_preview_publisher(dict(direct_camera_preview=False)) is None


def test_direct_publisher_sends_identical_annotated_pixels_and_original_clock(tmp_path):
    ros = Ros()
    publisher = make_live_preview_publisher(dict(direct_camera_preview=True,
        camera_live_preview_output=str(tmp_path)), ros=ros, image_type=Image,
        wall_clock=lambda: 202.)
    pixels = np.arange(4*8*3, dtype=np.uint8).reshape(4, 8, 3)[:, ::2]
    before = pixels.copy()
    preview = dict(index=14, stamp=1792080001.25, image=pixels)
    record = publisher.publish(preview, received_wall_stamp=200., ready_wall_stamp=201.8)
    assert ros.initializations == [('probtf_camera_live_preview', dict(anonymous=True,
        disable_signals=True, argv=['probtf_camera_live_preview']))]
    assert ros.publisher_calls == [('/observation_fusion/detection_image', Image,
                                    dict(queue_size=1, latch=True))]
    message = ros.messages[0]
    assert message.header.stamp == ('original_source_stamp', preview['stamp'])
    assert message.header.frame_id == 'camera_optical_frame'
    assert message.encoding == 'bgr8' and message.is_bigendian is False
    assert (message.height, message.width, message.step) == (4, 4, 12)
    np.testing.assert_array_equal(np.frombuffer(message.data, dtype=np.uint8).reshape(4, 4, 3), before)
    np.testing.assert_array_equal(pixels, before)
    assert record['input_to_publish_wall_age_s'] == 2.
    assert record['ready_to_publish_wall_age_s'] == pytest.approx(.2)
    saved = json.loads((tmp_path/'camera_live_publication.jsonl').read_text())
    assert saved == record
    assert saved['source_stamp'] == preview['stamp']
    assert saved['bypasses_coordinator_and_scene_publication_queues']
    assert not saved['new_observation_created'] and not saved['source_pixels_changed']
    publisher.close(); publisher.close()
    assert ros.unregistered == 1 and len(ros.shutdowns) == 1


def test_an_existing_ros_node_is_never_shutdown_by_preview_helper(tmp_path):
    ros = Ros(initialized=True)
    publisher = make_live_preview_publisher(dict(direct_camera_preview=True,
        camera_live_preview_output=str(tmp_path)), ros=ros, image_type=Image)
    publisher.close()
    assert not ros.initializations and not ros.shutdowns
    assert ros.unregistered == 1


def test_worker_publishes_after_detection_before_ipc_or_archival_and_closes_owned_helper(tmp_path, monkeypatch):
    from probtf_observation_fusion_demo import camera_live_preview
    events = []
    class Camera:
        def __init__(self, *args, **kwargs):
            self.diagnostics = {}
        def render(self, *args):
            events.append('render')
            return np.full((24, 32), 255, dtype=np.uint8)
    class Direct:
        def publish(self, preview, **kwargs):
            events.append('direct_publish')
            assert preview['stamp'] == 1001.25 and preview['index'] == 7
            assert preview['image'].shape == (24, 32, 3)
            assert kwargs['received_wall_stamp'] == 200.
        def close(self):
            events.append('direct_close')
    def factory(specification):
        assert specification['direct_camera_preview'] is True
        assert specification['camera_live_preview_output'] == str(tmp_path)
        return Direct()
    class Replies(Queue):
        def put(self, message, *args, **kwargs):
            if message.get('kind') == 'preview_ready':
                events.append('preview_ipc')
            return super().put(message, *args, **kwargs)
    original_preview, original_write = workers._camera_preview, workers._write_camera_image
    def preview(*args, **kwargs):
        result = original_preview(*args, **kwargs)
        events.append('detected')
        return result
    def write(*args, **kwargs):
        events.append('png_write')
        assert 'direct_publish' in events
        return original_write(*args, **kwargs)
    monkeypatch.setattr(workers, 'make_camera', Camera)
    monkeypatch.setattr(workers, '_camera_preview', preview)
    monkeypatch.setattr(workers, '_write_camera_image', write)
    monkeypatch.setattr(camera_live_preview, 'make_live_preview_publisher', factory)
    requests, replies = Queue(), Replies()
    requests.put(dict(kind='image', event_id='image:7', frame_index=7,
        stamp=1001.25, time=1.25, command=[.1, -.2], received_wall_stamp=200.,
        camera_pose=None, marker_pose=None, camera_history=()))
    requests.put(None)
    workers.acquisition_worker(requests, replies,
        dict(board=dict(enabled=True), camera={}, texture='fixture', seed=21, marker_size=.072,
             marker_ids=(21,), direct_camera_preview=True),
        str(tmp_path/'observations.sqlite3'), str(tmp_path))
    messages = [replies.get_nowait() for _ in range(replies.qsize())]
    assert not any(message.get('failed') for message in messages), messages
    assert events == ['render', 'detected', 'direct_publish', 'preview_ipc', 'png_write', 'direct_close']
    assert [message['kind'] for message in messages] == ['preview_ready', 'image_ready']


@pytest.mark.parametrize('value', ['false', 1, None])
def test_live_mode_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError):
        make_live_preview_publisher(dict(direct_camera_preview=value))
