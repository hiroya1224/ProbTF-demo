"""Publish already measured camera previews without coordinator/scene queues.

ROS is imported only for the explicitly enabled live-view path.  This helper
does not produce observations, transform candidates, or replacement pixels.
"""

import json
from pathlib import Path
import time

import numpy as np


class LiveCameraPreviewPublisher:
    def __init__(self, specification, *, ros, image_type, wall_clock=None):
        self.ros, self.image_type = ros, image_type
        self.wall_clock = time.time if wall_clock is None else wall_clock
        self.output = Path(specification['camera_live_preview_output'])
        self.owns_node = not ros.core.is_initialized()
        self.closed = False
        if self.owns_node:
            # Spawned workers inherit the parent's argv, including its private
            # ROS node-name remapping. It must not rename this separate node.
            ros.init_node('probtf_camera_live_preview', anonymous=True,
                disable_signals=True, argv=['probtf_camera_live_preview'])
        self.publisher = ros.Publisher('/observation_fusion/detection_image',
            image_type, queue_size=1, latch=True)

    def publish(self, preview, *, received_wall_stamp, ready_wall_stamp=None):
        if self.closed:
            raise RuntimeError('The direct camera preview publisher is closed.')
        pixels = np.asarray(preview['image'])
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError('The annotated camera preview must contain the original BGR8 pixels.')
        ready = self.wall_clock() if ready_wall_stamp is None else float(ready_wall_stamp)
        message = self.image_type()
        message.header.stamp = self.ros.Time.from_sec(float(preview['stamp']))
        message.header.frame_id = 'camera_optical_frame'
        message.height, message.width = pixels.shape[:2]
        message.encoding, message.is_bigendian = 'bgr8', False
        message.step = int(message.width)*3
        message.data = pixels.tobytes(order='C')
        self.publisher.publish(message)
        published = self.wall_clock()
        record = dict(frame_index=int(preview['index']), source_stamp=float(preview['stamp']),
            received_wall_stamp=float(received_wall_stamp), ready_wall_stamp=ready,
            publish_wall_stamp=published,
            input_to_publish_wall_age_s=published-float(received_wall_stamp),
            ready_to_publish_wall_age_s=published-ready,
            frame_id=message.header.frame_id, encoding=message.encoding,
            width=int(message.width), height=int(message.height),
            topic='/observation_fusion/detection_image', publisher_queue_size=1,
            source_pixels_changed=False, new_observation_created=False,
            bypasses_coordinator_and_scene_publication_queues=True,
            publish_boundary='ROS publish call returned; RViz delivery/render completion is not measured')
        with (self.output/'camera_live_publication.jsonl').open('a') as stream:
            stream.write(json.dumps(record, separators=(',', ':'))+'\n')
        return record

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.publisher.unregister()
        if self.owns_node:
            self.ros.signal_shutdown('direct camera preview worker completed')


def make_live_preview_publisher(specification, *, ros=None, image_type=None, wall_clock=None):
    enabled = specification.get('direct_camera_preview', False)
    if not isinstance(enabled, bool):
        raise ValueError('direct_camera_preview must be an explicit boolean.')
    if not enabled:
        return None
    if ros is None:
        import rospy
        ros = rospy
    if image_type is None:
        from sensor_msgs.msg import Image
        image_type = Image
    return LiveCameraPreviewPublisher(specification, ros=ros, image_type=image_type,
                                      wall_clock=wall_clock)
