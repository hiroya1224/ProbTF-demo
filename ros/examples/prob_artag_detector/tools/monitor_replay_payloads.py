#!/usr/bin/env python3
"""Read-only subscriber-side audit of recorded AR replay payloads.

Hashes concatenate the exact ROS serialization in per-topic receive order.
No detector, inference, re-stamping, or republishing is performed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import signal
import threading
import time

import rospy
from probtf_msgs.msg import ProbabilisticTransformStamped
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage


def headers(message):
    if hasattr(message, 'header'):
        return [message.header]
    return [transform.header for transform in getattr(message, 'transforms', [])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--ready', required=True)
    args = parser.parse_args()
    lock = threading.RLock()
    ended = threading.Event()
    start_wall = time.time()
    start_mono = time.monotonic()
    topic_types = {
        '/prob_artag_demo/probtf': ProbabilisticTransformStamped,
        '/prob_artag_detector/debug_image': Image,
        '/tf': TFMessage,
        '/tf_static': TFMessage,
    }
    digests = {name: hashlib.sha256() for name in topic_types}
    stats = {
        name: {
            'count': 0, 'serialized_bytes': 0,
            'first_header_ns': None, 'last_header_ns': None,
            'first_wall_time': None, 'last_wall_time': None,
            'first_monotonic': None, 'last_monotonic': None,
            'message_type': cls._type,
            'md5sum': cls._md5sum,
        }
        for name, cls in topic_types.items()
    }
    clock_stats = {
        'count': 0, 'first_clock_ns': None, 'last_clock_ns': None,
        'first_wall_time': None, 'last_wall_time': None,
        'first_monotonic': None, 'last_monotonic': None,
    }
    errors = []

    def message_callback(message, name):
        wall, mono = time.time(), time.monotonic()
        try:
            output = io.BytesIO()
            message.serialize(output)
            payload = output.getvalue()
            source_headers = headers(message)
            with lock:
                row = stats[name]
                if row['count'] == 0:
                    row['first_wall_time'] = wall
                    row['first_monotonic'] = mono
                    if source_headers:
                        row['first_header_ns'] = source_headers[0].stamp.to_nsec()
                row['count'] += 1
                row['serialized_bytes'] += len(payload)
                row['last_wall_time'] = wall
                row['last_monotonic'] = mono
                if source_headers:
                    row['last_header_ns'] = source_headers[-1].stamp.to_nsec()
                digests[name].update(payload)
        except Exception as error:
            with lock:
                errors.append({'topic': name, 'error': repr(error), 'wall_time': wall})

    def clock_callback(message):
        wall, mono = time.time(), time.monotonic()
        with lock:
            if clock_stats['count'] == 0:
                clock_stats['first_clock_ns'] = message.clock.to_nsec()
                clock_stats['first_wall_time'] = wall
                clock_stats['first_monotonic'] = mono
            clock_stats['count'] += 1
            clock_stats['last_clock_ns'] = message.clock.to_nsec()
            clock_stats['last_wall_time'] = wall
            clock_stats['last_monotonic'] = mono

    def stop(signum, _frame):
        ended.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    rospy.init_node('ar_recorded_bag_replay_payload_monitor', anonymous=True, disable_signals=True)
    subscriptions = [
        rospy.Subscriber(topic, cls, message_callback, callback_args=topic,
                         queue_size=1000, buff_size=2**27, tcp_nodelay=True)
        for topic, cls in topic_types.items()
    ]
    subscriptions.append(rospy.Subscriber('/clock', Clock, clock_callback,
                                         queue_size=10000, buff_size=2**22,
                                         tcp_nodelay=True))
    ready = Path(args.ready)
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text(json.dumps({
        'node': rospy.get_name(), 'pid': __import__('os').getpid(),
        'wall_time': time.time(), 'monotonic': time.monotonic(),
        'topics': list(topic_types) + ['/clock'],
    }, indent=2) + '\n')
    while not ended.wait(0.05) and not rospy.is_shutdown():
        pass
    for subscription in subscriptions:
        subscription.unregister()
    rospy.signal_shutdown('Replay payload monitor stopped')
    with lock:
        result = {
            'method': 'sha256_of_concatenated_original_ROS_serialization_per_topic',
            'started_wall_time': start_wall, 'started_monotonic': start_mono,
            'stopped_wall_time': time.time(), 'stopped_monotonic': time.monotonic(),
            'topics': {
                topic: dict(row, sha256=digests[topic].hexdigest())
                for topic, row in stats.items()
            },
            'clock': dict(clock_stats), 'errors': list(errors),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(output), 'counts': {
        topic: row['count'] for topic, row in stats.items()},
        'clock_count': clock_stats['count'], 'errors': len(errors)}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
