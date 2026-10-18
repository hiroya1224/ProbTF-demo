"""Fresh camera previews and complete FIFO archives share the same pixels."""

import hashlib
from queue import Empty, Queue
from threading import Event, Thread, get_ident
import time

import cv2
import numpy as np
import pytest

from probtf_observation_fusion_demo import async_workers as workers
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore


def specification():
    return dict(board=dict(enabled=True), texture='unused-synthetic-input',
        camera={}, seed=21, marker_size=.072, marker_ids=(21,))


def request(kind, index, stamp):
    result = dict(kind=kind, event_id=kind+':'+str(index), frame_index=index,
        stamp=stamp, time=stamp-1000., command=[.01*index, -.2],
        received_wall_stamp=1700000000.+index)
    if kind == 'image':
        result.update(camera_pose=index, marker_pose=None, camera_history=())
    else:
        result['payload'] = dict(reading=[index, .25])
    return result


def install_camera(monkeypatch):
    rendered = []
    class Camera:
        def __init__(self, *args):
            self.diagnostics = dict(synthetic_estimated_sensor_fixture=True)
        def render(self, camera_pose, marker_pose, history):
            # One deterministic new capture per request; no shared mutable
            # buffer and no extra capture for the preview or archive.
            image = ((np.arange(32*48).reshape(32, 48)+17*camera_pose) % 256).astype(np.uint8)
            rendered.append((camera_pose, image))
            return image
    monkeypatch.setattr(workers, 'BoardCamera', Camera)
    return rendered


def drain(queue):
    messages = []
    while True:
        try:
            messages.append(queue.get_nowait())
        except Empty:
            return messages


def test_blocked_png_does_not_block_previews_but_raw_and_database_order_wait(tmp_path, monkeypatch):
    rendered = install_camera(monkeypatch)
    entered_write, release_write = Event(), Event()
    original_write = workers._write_camera_image
    def blocked_write(path, image):
        entered_write.set()
        if not release_write.wait(5.):
            raise RuntimeError('test did not release the archival barrier')
        original_write(path, image)
    monkeypatch.setattr(workers, '_write_camera_image', blocked_write)
    requests, replies = Queue(), Queue()
    raw = [request('joint', 0, 1000.), request('image', 0, 1000.),
           request('imu', 1, 1000.1), request('image', 1, 1000.1)]
    for item in raw:
        requests.put(item)
    requests.put(dict(kind='database_operation', operation='select_batch',
        through_stamp=1000.1, snapshot_id='all-original-captures', acquisition_watermark=(4, 0)))
    requests.put(None)
    database = tmp_path/'observations.sqlite3'
    worker = Thread(target=workers.acquisition_worker,
        args=(requests, replies, specification(), str(database), str(tmp_path)))
    worker.start()
    messages = []
    try:
        assert entered_write.wait(3.)
        deadline = time.monotonic()+3.
        while sum(item.get('kind') == 'preview_ready' for item in messages) < 2:
            messages.append(replies.get(timeout=max(.01, deadline-time.monotonic())))
            assert not messages[-1].get('failed'), messages[-1]
        previews = [item for item in messages if item.get('kind') == 'preview_ready']
        assert [item['event_id'] for item in previews] == ['image:0', 'image:1']
        assert worker.is_alive()  # Normal shutdown is waiting for the writer.
        assert not any(item.get('kind') == 'image_ready' for item in messages)
        assert not any(item.get('kind') == 'database_batch_selection' for item in messages)
        assert not any(item.get('event_id') == 'imu:1' for item in messages)
        for item in previews:
            assert item['archival_complete'] is False
            assert item['observation_event_created'] is False
            assert item['command'] == raw[1 if item['frame_index'] == 0 else 3]['command']
            np.testing.assert_array_equal(item['image'], rendered[item['frame_index']][1])
        assert len(rendered) == 2
    finally:
        release_write.set()
        worker.join(5.)
    assert not worker.is_alive()
    messages.extend(drain(replies))
    assert not any(item.get('failed') for item in messages), messages
    archived = [item for item in messages if item.get('kind') in ('raw_ready', 'image_ready')]
    assert [item['event_id'] for item in archived] == [item['event_id'] for item in raw]
    assert [item['event_sequence'] for item in archived] == [1, 2, 3, 4]
    selection = [item for item in messages if item.get('kind') == 'database_batch_selection']
    assert len(selection) == 1
    assert selection[0]['batch'].event_ids == tuple(item['event_id'] for item in raw)
    with SensorEventStore(database) as store:
        events = store.events()
        assert [item.event_id for item in events] == [item['event_id'] for item in raw]
        for event, source in zip(events, raw):
            assert event.stamp == source['stamp']
            assert event.command == tuple(source['command'])
            if event.kind == 'image':
                image = event.payload['image']
                np.testing.assert_array_equal(image, rendered[source['frame_index']][1])
                np.testing.assert_array_equal(image, cv2.imread(
                    str(tmp_path/event.payload['image_path']), cv2.IMREAD_UNCHANGED))
                assert event.payload['image_stamp'] == source['stamp']
                assert event.payload['image_sha256'] == hashlib.sha256(image.tobytes()).hexdigest()
                assert not ({'camera_pose', 'marker_pose', 'camera_history'} & set(event.payload))
    for item in [item for item in messages if item.get('kind') == 'image_ready']:
        assert item['archival_complete'] is True
        assert item['acquisition_timing']['image_write_seconds'] >= 0.


def test_database_connection_is_owned_only_by_archive_thread(tmp_path, monkeypatch):
    install_camera(monkeypatch)
    foreground, owners = get_ident(), []
    class OwnedStore(SensorEventStore):
        def __init__(self, path):
            owners.append(get_ident())
            super().__init__(path)
        def append(self, *args, **kwargs):
            assert get_ident() == owners[0]
            return super().append(*args, **kwargs)
        def close(self):
            assert get_ident() == owners[0]
            super().close()
    monkeypatch.setattr(workers, 'SensorEventStore', OwnedStore)
    requests, replies = Queue(), Queue()
    requests.put(request('joint', 0, 1000.)); requests.put(None)
    workers.acquisition_worker(requests, replies, specification(),
        str(tmp_path/'observations.sqlite3'), str(tmp_path))
    messages = drain(replies)
    assert owners and owners[0] != foreground
    assert [item['kind'] for item in messages] == ['raw_ready']


def test_image_write_failure_is_explicit_and_never_acknowledged_as_archived(tmp_path, monkeypatch):
    install_camera(monkeypatch)
    def failing_write(path, image):
        raise IOError('deliberate archive failure')
    monkeypatch.setattr(workers, '_write_camera_image', failing_write)
    requests, replies = Queue(), Queue()
    requests.put(request('image', 0, 1000.)); requests.put(None)
    workers.acquisition_worker(requests, replies, specification(),
        str(tmp_path/'observations.sqlite3'), str(tmp_path))
    messages = drain(replies)
    assert [item.get('kind') for item in messages if not item.get('failed')] == ['preview_ready']
    failures = [item for item in messages if item.get('failed')]
    assert len(failures) == 1
    assert failures[0]['worker'] == 'acquisition'
    assert 'deliberate archive failure' in failures[0]['traceback']
    assert not any(item.get('kind') == 'image_ready' for item in messages)
    with SensorEventStore(tmp_path/'observations.sqlite3') as store:
        assert store.events()[0].status == 'awaiting_payload'


def test_historical_archive_helper_still_writes_exact_pixels_and_returns_preview(tmp_path):
    image = ((np.arange(32*48).reshape(32, 48)*7) % 256).astype(np.uint8)
    before = image.copy()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, 'DetectorParameters')
                  else cv2.aruco.DetectorParameters_create())
    detector = cv2.aruco.ArucoDetector(dictionary, parameters) if hasattr(cv2.aruco, 'ArucoDetector') else None
    preview = workers._archive_image(tmp_path/'raw.png', image, 7, 12.5,
                                    dictionary, parameters, detector, (21,))
    assert preview['index'] == 7 and preview['stamp'] == 12.5
    assert preview['image'].shape == (32, 48, 3)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(cv2.imread(str(tmp_path/'raw.png'), cv2.IMREAD_UNCHANGED), before)
