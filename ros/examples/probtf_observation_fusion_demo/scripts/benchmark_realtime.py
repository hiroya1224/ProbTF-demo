#!/usr/bin/env python3
"""Measure fresh scene/inference/control/display work without prerecorded frames.

Example (use a dedicated ROS master when publishing):
  rosrun probtf_observation_fusion_demo benchmark_realtime.py --publish --frames 40

The default live executor supplies every fixed simulation-time tick. Slower
computation slows the measured wall rate rather than pretending to execute at
20 Hz or skipping an observation. No physical frame advances under an old law.
No file cache, saved image, or previously prepared distribution is used.
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import rospy
import rospkg
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from probtf_observation_fusion_demo.display import snapshot_record
from probtf_observation_fusion_demo.motion_display import MotionDisplay
from probtf_observation_fusion_demo.motion_node import MotionSession


def serialize(message):
    stream = io.BytesIO()
    message.serialize(stream)
    return stream.getvalue()


def statistics(values):
    array = np.asarray(values, dtype=float)
    return {"count": array.size, "mean": float(array.mean()),
            "median": float(np.median(array)), "p95": float(np.percentile(array, 95)),
            "maximum": float(array.max()), "minimum": float(array.min())}


def implementation_hashes(package):
    """Fingerprint the actual Python implementations used by this run."""
    root = package.parents[2]
    paths = [Path(__file__).resolve()]
    paths.extend((package/'src/probtf_observation_fusion_demo').glob('*.py'))
    paths.extend((package/'config').glob('*.yaml'))
    paths.extend((root/'ros/core/probtf_core/src/probtf/spherical_law').glob('*.py'))
    paths.extend(root/path for path in (
        'ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/model/equilibrium.py',
        'ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/model/sensitivity.py',
        'ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/model/spring.py',
        'ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/robot/pinocchio_robot.py',
        'ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/estimator/initialization.py',
        'ros/examples/deflecomp/deflecomp_ros/config/estimator.yaml',
        'ros/examples/probtf_switch_press_demo/config/independent_transform_laws.yaml',
        'ros/examples/prob_artag_detector/src/prob_artag_detector/detector.py',
        'ros/examples/prob_artag_detector/src/prob_artag_detector/estimator.py',
        'ros/core/probtf_core/src/probtf/bingham.py'))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


class WireReceiver:
    """Receive exact typed-topic bytes; do not reconstruct another display law."""
    def __init__(self, prefix):
        self.lock = threading.Lock()
        self.records = {"markers": [], "image": [], "snapshot": [], "stiffness": []}
        self.subscribers = [rospy.Subscriber(prefix+"/"+name, rospy.AnyMsg,
            self.receive, callback_args=name, queue_size=100)
            for name in self.records]

    def receive(self, message, name):
        received = (time.perf_counter(), hashlib.sha256(message._buff).hexdigest())
        with self.lock:
            self.records[name].append(received)

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()


def benchmark(case, frames, package, output, publish, source_hashes):
    source_matches_at_start = source_hashes == implementation_hashes(package)
    load_average = os.getloadavg()
    session = MotionSession(package/("config/"+case+".yaml"), output/case)
    scene, inference, observations = session.scene, session.inference, session.templates
    period = session.period
    receiver, publishers, physical_display = None, {}, None
    physical_subscribers = []
    physical_received = {"scene": [], "tf": []}
    expected_hud = []
    expected = {"markers": [], "image": [], "snapshot": [], "stiffness": []}
    if publish:
        prefix = "/observation_fusion_benchmark/"+case
        receiver = WireReceiver(prefix)
        publishers = {
            "markers": rospy.Publisher(prefix+"/markers", MarkerArray, queue_size=2),
            "image": rospy.Publisher(prefix+"/image", Image, queue_size=2),
            "snapshot": rospy.Publisher(prefix+"/snapshot", String, queue_size=2),
            "stiffness": rospy.Publisher(prefix+"/stiffness", String, queue_size=2),
        }
        for name, topic in (("scene", "/observation_fusion/scene"), ("tf", "/tf")):
            physical_subscribers.append(rospy.Subscriber(topic, rospy.AnyMsg,
                lambda message, selected: physical_received[selected].append(
                    (time.perf_counter(), bytes(message._buff))),
                callback_args=name, queue_size=100))
        physical_display = MotionDisplay(scene, case, [item["sensor"] for item in observations])
        deadline = time.monotonic()+5.0
        while (not all(publisher.get_num_connections() for publisher in publishers.values())
               or physical_display.publisher.get_num_connections() == 0):
            if time.monotonic() > deadline:
                raise RuntimeError("Benchmark subscribers did not connect.")
            time.sleep(.01)
    records = []
    started = time.perf_counter()
    for index in range(frames):
        cycle_start = time.perf_counter()
        if session.finished:
            raise RuntimeError("The benchmark requested more ticks than the common experiment duration.")
        tick = session.observe()
        frame, result, snapshot, markers = tick.frame, tick.result, tick.snapshot, tick.markers
        elapsed = frame.time
        timings = dict(tick.timings)
        stiffness_record = snapshot.stiffness_record
        if stiffness_record['status'].startswith('failed'):
            (output/(case+'.stiffness_failure_{:03d}.json'.format(index))).write_text(
                json.dumps({'body': result.records, 'stiffness': stiffness_record}, indent=2)+'\n')
        previous = time.perf_counter()
        # Serialize exactly once. Forwarding AnyMsg preserves these typed
        # payloads, including their common source time/command/version.
        payloads = {
            "markers": serialize(markers),
            "image": serialize(tick.image),
            "snapshot": serialize(String(json.dumps(snapshot_record(snapshot)))),
            "stiffness": serialize(String(json.dumps(stiffness_record))),
        }
        timings["serialization"] = time.perf_counter()-previous
        previous = time.perf_counter()
        if publish:
            for name, publisher in publishers.items():
                expected[name].append(hashlib.sha256(payloads[name]).hexdigest())
                publisher.publish(rospy.AnyMsg().deserialize(payloads[name]))
        timings["publication"] = time.perf_counter()-previous
        previous = time.perf_counter()
        if physical_display is not None:
            physical_display.publish(frame, index, snapshot, index,
                stiffness_record=stiffness_record, execution=dict(execution_mode='live',
                    simulated_step_s=period, actual_update_hz=(index/(time.perf_counter()-started) if index else None)))
            expected_hud.append(serialize(MotionDisplay.stiffness_marker(snapshot, stiffness_record)))
        timings["physical_display_and_tf"] = time.perf_counter()-previous
        session.finish(tick)
        diagnostics = result.records['controller']
        timings['command_selection'] = tick.timings['command_selection']
        timings["cycle_work"] = time.perf_counter()-cycle_start
        records.append({"index": index, "scene_elapsed": elapsed,
            "source_stamp": snapshot.stamp, "command": list(snapshot.command),
            "source_version": snapshot.version, "timings_s": timings,
            "valid_observations": len(snapshot.observation_ids),
            "components": len(result.stages[-1].components),
            "objective_evaluations": diagnostics.get("objective_evaluation_count", 0),
            "stiffness_status": stiffness_record['status'],
            "stiffness_update_applied": stiffness_record['update_applied'],
            "stiffness_failure_reason": stiffness_record.get('failure_reason'),
            "stiffness_information_rank": stiffness_record['information_rank'],
            "stiffness_model_evaluations": stiffness_record.get('model_evaluation_count'),
            "stiffness_prior": stiffness_record['prior'],
            "stiffness_posterior": stiffness_record['posterior'],
            "stiffness_prior_retained": stiffness_record.get('prior_retained', False),
            "stiffness_retained_prior": stiffness_record.get('retained_prior'),
            "stiffness_information_eigenvalues": stiffness_record['information_eigenvalues'],
            "source_identity_verified": all(point.query.command == snapshot.command
                and point.query.stamp == snapshot.stamp and point.query.version == snapshot.version
                for point in snapshot.points)})
        # A fast cycle can wait for the requested interval. A slow one must
        # report its actual wall rate, with no replay or skipped source updates.
        remaining = period-(time.perf_counter()-cycle_start)
        if remaining > 0:
            time.sleep(remaining)
    elapsed_wall = time.perf_counter()-started
    session.write_manifest(mode='live_benchmark')
    summary = {"case": case, "frames": len(records), "wall_seconds": elapsed_wall,
        "fresh_updates_per_second": len(records)/elapsed_wall,
        "nominal_command_interval_s": period,
        "clock": "fixed logical simulation ticks; wall rate measured independently; no prepared replay or skipped tick",
        "simulated_elapsed_s": session.last_tick.frame.time,
        "execution_mode": "live_benchmark",
        "realtime_factor": session.last_tick.frame.time/elapsed_wall,
        "offline_precomputation": False, "published": publish,
        "stiffness_enabled": True,
        "rviz_gpu_rendering_measured": False,
        "implementation_sha256": source_hashes,
        "source_files_unchanged_during_measurement": (
            source_matches_at_start and source_hashes == implementation_hashes(package)),
        "source_fingerprint_policy": "one baseline at process start, shared by all four cases",
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "load_average_at_start": list(load_average),
        "cpu_load_note": "Measured during other demo preparation; affinity is reported, not a claim of an isolated CPU.",
        "scene_config_sha256": scene.config_sha256,
        "estimation_config_sha256": hashlib.sha256((package/"config/moving_estimation.yaml").read_bytes()).hexdigest(),
        "stiffness_config_sha256": hashlib.sha256((package/"config/stiffness_identification.yaml").read_bytes()).hexdigest(),
        "stage_seconds": {name: statistics([row["timings_s"][name] for row in records])
                          for name in records[0]["timings_s"]},
        "missed_nominal_deadlines": sum(row["timings_s"]["cycle_work"] > period for row in records),
        "valid_observations_histogram": dict(Counter(row["valid_observations"] for row in records)),
        "stiffness_status_histogram": dict(Counter(row['stiffness_status'] for row in records)),
        "stiffness_update_applied_count": sum(row['stiffness_update_applied'] for row in records),
        "stiffness_failure_reasons": sorted(set(row['stiffness_failure_reason'] for row in records
                                               if row['stiffness_failure_reason'] is not None)),
        "stiffness_failed_update_count": sum(row['stiffness_status'].startswith('failed') for row in records),
        "applicability_limit": "Short-window throughput only; does not establish long-horizon or global Laplace applicability. Failed updates retain prior and do not supply a posterior.",
        "source_identity_verified": all(row["source_identity_verified"] for row in records)}
    if receiver is not None:
        deadline = time.monotonic()+5.0
        while time.monotonic() < deadline:
            with receiver.lock:
                if all(len(receiver.records[name]) >= frames for name in expected):
                    break
            time.sleep(.01)
        with receiver.lock:
            summary["received_wire"] = {}
            for name, received in receiver.records.items():
                actual = [digest for _, digest in received]
                intervals = np.diff([stamp for stamp, _ in received])
                summary["received_wire"][name] = {
                    "count": len(received), "bytes_identical_in_order": actual == expected[name],
                    "average_hz": float(1./intervals.mean()) if intervals.size else None,
                    "interval_seconds": statistics(intervals) if intervals.size else None}
                if actual != expected[name]:
                    raise AssertionError("Actual ROS wire data differs from this snapshot: "+name)
        receiver.close()
        for subscriber in physical_subscribers:
            subscriber.unregister()
        summary["received_physical_display"] = {}
        for name, received in physical_received.items():
            intervals = np.diff([stamp for stamp, _ in received])
            summary["received_physical_display"][name] = {
                "count": len(received),
                "average_hz": float(1./intervals.mean()) if intervals.size else None}
        actual_hud = []
        for _, payload in physical_received['scene']:
            received_scene = MarkerArray().deserialize(payload)
            hud = [marker for marker in received_scene.markers if marker.ns == 'estimated_stiffness']
            if len(hud) != 1:
                raise AssertionError('Actual ROS scene must contain one stiffness HUD from the updater record.')
            actual_hud.append(serialize(hud[0]))
        summary['stiffness_hud_wire_identical_in_order'] = actual_hud == expected_hud
        if actual_hud != expected_hud:
            raise AssertionError('Actual ROS stiffness HUD differs from this exact body/stiffness snapshot.')
        physical_display.publisher.unregister()
        for publisher in publishers.values():
            publisher.unregister()
    (output/(case+".frames.jsonl")).write_text("".join(json.dumps(row)+"\n" for row in records))
    (output/(case+".json")).write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--cases", nargs="+", default=["no_sensors", "imu", "marker", "imu_marker"],
                        choices=["no_sensors", "imu", "marker", "imu_marker"])
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/probtf_live_benchmark")/
                        datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args(rospy.myargv()[1:])
    if args.frames <= 1:
        parser.error("--frames must be greater than one")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.publish:
        rospy.init_node("observation_fusion_live_benchmark", anonymous=True)
    package = Path(rospkg.RosPack().get_path("probtf_observation_fusion_demo"))
    # Imports are fixed for this Python process. Re-reading a changed file at
    # the next case would misattribute its hash to an already imported module.
    source_hashes = implementation_hashes(package)
    result = []
    for case in args.cases:
        summary = benchmark(case, args.frames, package, args.output, args.publish, source_hashes)
        result.append(summary)
        print(json.dumps(summary), flush=True)
    (args.output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
