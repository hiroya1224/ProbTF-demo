"""A live preview is not a durable acquisition acknowledgement."""
import json
from queue import Queue

import numpy as np

from probtf_observation_fusion_demo.event_motion import AsyncMotionSession


def session(tmp_path):
    value = object.__new__(AsyncMotionSession)
    value.output = tmp_path
    value.acquisition_replies = Queue()
    value.acquisition_completed = 0
    value.preview_capture_index = None
    value.marker_preview_visible = None
    value.last_marker_missing_capture = None
    value.raw_publications = []
    return value


def message(index, kind='preview_ready', visible=True):
    image = np.full((3, 4), index, dtype=np.uint8)
    return dict(kind=kind, frame_index=index, stamp=100.+index,
        image=image, preview=dict(index=index, stamp=100.+index,
            image=image, visible=visible, visible_marker_ids=[17] if visible else []))


def test_preview_keeps_original_pixels_stamp_without_claiming_archival(tmp_path):
    value = session(tmp_path)
    preview = message(2)
    value.acquisition_replies.put(preview)
    value.collect_archives()
    assert value.acquisition_completed == 0
    assert value.raw_publications == []
    assert value.preview_image is preview['image']
    assert value.preview_capture_index == 2
    archive = dict(preview, kind='image_ready')
    value.acquisition_replies.put(archive)
    value.collect_archives()
    assert value.acquisition_completed == 1
    assert len(value.raw_publications) == 1
    assert value.raw_publications[0]['image'] is preview['image']
    assert value.raw_publications[0]['stamp'] == 102.
    rows = [json.loads(line) for line in (tmp_path/'camera_previews.jsonl').read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['stamp'] == 102.


def test_older_archive_ack_cannot_roll_back_live_detection_or_duplicate_preview(tmp_path):
    value = session(tmp_path)
    for item in [message(2, visible=False), message(3), message(2, 'image_ready', False)]:
        value.acquisition_replies.put(item)
    value.collect_archives()
    assert value.acquisition_completed == 1
    assert value.preview_capture_index == 3
    assert value.marker_preview_visible
    assert value.last_marker_missing_capture == 2
    np.testing.assert_array_equal(value.preview_image, np.full((3, 4), 3))
    assert value.raw_publications[0]['frame_index'] == 2
    assert len((tmp_path/'camera_previews.jsonl').read_text().splitlines()) == 2
