"""One playback and ROS adapter for every observation list."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rospy
import rospkg
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import MarkerArray

from probtf.spherical_law.first_order import point_query
from .display import (
    DisplayPoint, DisplaySnapshot, make_markers, save_snapshot, snapshot_record,
    verify_measure_identity,
)
from .model import component_log_weight, load_inputs, run
from .moving_control import switch_goal_queries


def build_snapshot(result, stage, common):
    """All display means/covariances are references to these core query results."""
    settings = common["display"]
    paths = dict(result.link_paths)
    paths.update(hand=result.hand_path, camera=result.camera_path, marker=result.marker_path)
    goal_components = tuple(getattr(stage, 'goal_components', ()))
    if goal_components:
        # The internal body is displayed once. Current observation goal modes
        # have their own joint sources; they supply only the marker queries.
        del paths['marker']
    points = []
    axis_length = float(settings["axis_length"])
    endpoints = (("origin", np.zeros(3)), ("x", [axis_length, 0, 0]),
                 ("y", [0, axis_length, 0]), ("z", [0, 0, axis_length]))

    def append_endpoint(component, label, path, axis, endpoint):
        state = component.state
        query = point_query(
            state, path, endpoint,
            query_id="{}/{}/{}/{}".format(stage.label, component.label, label, axis),
            frame=result.root_frame,
        )
        if (query.version != state.version or query.stamp != state.stamp
                or tuple(query.command) != tuple(state.command)):
            raise AssertionError("Core query provenance differs from its joint snapshot.")
        count = settings["marker_samples"] if label in ('marker', 'switch_goal') else settings["body_samples"]
        points.append(DisplayPoint(query, label, axis, component.label,
                                   component.weight, int(count), component_log_weight(component)))

    observed = {}
    for estimate in result.local_estimates:
        if estimate.observed_direction is not None:
            key = (estimate.posterior.observation_id, estimate.diagnostics['target_link'])
            if key in observed and not np.array_equal(
                    observed[key].observed_direction, estimate.observed_direction):
                raise AssertionError('One physical IMU observation has inconsistent displayed directions.')
            observed[key] = estimate
    for component in stage.components:
        state = component.state
        for label, path in paths.items():
            for axis, endpoint in endpoints:
                append_endpoint(component, label, path, axis, endpoint)
        for estimate in observed.values():
            target = estimate.diagnostics["target_link"]
            path = result.link_paths[target]
            for axis, endpoint in (("origin", np.zeros(3)),
                                   ("observed", axis_length * estimate.observed_direction)):
                query = point_query(state, path, endpoint,
                                    query_id="{}/{}/imu:{}/{}".format(
                                        stage.label, component.label, target, axis),
                                    frame=result.root_frame)
                points.append(DisplayPoint(query, "IMU gravity at " + target, axis,
                                           component.label, component.weight,
                                           int(settings["body_samples"]), component_log_weight(component)))
    for component in goal_components:
        for axis, endpoint in endpoints:
            append_endpoint(component, 'marker', result.marker_path, axis, endpoint)
    if 'control' in common:
        control = common['control']
        for component in goal_components or stage.components:
            for axis, query in switch_goal_queries(component.state,
                    result.marker_path, control['goal_offset'],
                    control['goal_orientation_wxyz'], axis_length,
                    frame=result.root_frame,
                    query_prefix='{}/{}/switch_goal'.format(stage.label, component.label)):
                points.append(DisplayPoint(query, 'switch_goal', axis, component.label,
                    component.weight, int(settings['marker_samples']), component_log_weight(component)))
    state = stage.components[0].state
    hand = next(p.query for p in points if p.frame_label == "hand" and p.axis == "origin")
    summary = (
        "{} | observations: {} | version: {}\n"
        "t={:.3f}  command={}\n"
        "hand covariance trace: within={:.6g}, between={:.6g}\n"
        "{} component(s); axes are core first-order Gaussian queries"
    ).format(stage.label, len(stage.sensor_ids), state.version, state.stamp,
             list(state.command), float(np.trace(hand.within)),
             float(np.trace(hand.between)), len(stage.components))
    observation_ids = tuple(dict.fromkeys(p.observation_id for p in result.sensor_posteriors
                                         if p.sensor_id in stage.sensor_ids))
    snapshot = DisplaySnapshot(tuple(points), result.root_frame, state.stamp,
                               tuple(state.command), state.version, stage.label,
                               observation_ids, summary,
                               goal_memory=result.records.get('fixed_goal_memory'),
                               behavior=result.records.get('behavior'),
                               surprise_record=result.records.get('bayes_factor_surprise'),
                               candidate_measure=(result.records.get('candidate_measure')
                                                  if stage is result.stages[-1] else None),
                               sensor_event=result.records.get('sensor_event'))
    verify_adapter_identity(snapshot, stage)
    return snapshot


def verify_adapter_identity(snapshot, stage):
    """The adapter owns no second covariance/mean model or mutable parameter copy."""
    components = {component.label: component for component in stage.components}
    goal_components = {component.label: component
                       for component in getattr(stage, 'goal_components', ())}
    for point in snapshot.points:
        sources = (goal_components if point.frame_label in ('marker', 'switch_goal')
                   and goal_components else components)
        if point.component_id not in sources:
            raise AssertionError("A displayed point belongs to a different body or goal source.")
        source = sources[point.component_id]
        query = point.query
        if (point.weight != source.weight or point.log_weight != component_log_weight(source)
                or query.version != source.state.version
                or query.stamp != source.state.stamp
                or tuple(query.command) != tuple(source.state.command)
                or query.state_id != source.state.state_id
                or query.frame != snapshot.frame_id
                or query.version != snapshot.version or query.stamp != snapshot.stamp
                or tuple(query.command) != tuple(snapshot.command)):
            raise AssertionError("Adapter input is not the estimator's joint-state query.")
        if query.mean.flags.writeable or query.covariance.flags.writeable:
            raise AssertionError("Display query must be immutable.")
        if not all(np.all(np.isfinite(value)) for value in
                   (query.mean,query.covariance,query.within,query.between)):
            raise ValueError('A displayed core query contains nonfinite values.')
    if goal_components and {point.component_id for point in snapshot.points
                            if point.frame_label == 'marker'} != set(goal_components):
        raise AssertionError("The marker display lost an observation goal mixture component.")
    switch_points = [point for point in snapshot.points if point.frame_label == 'switch_goal']
    if switch_points or snapshot.goal_memory is not None:
        sources = goal_components or components
        if ({point.component_id for point in switch_points} != set(sources)
                or len(switch_points) != 4*len(sources)
                or any({point.axis for point in switch_points if point.component_id == label}
                       != {'origin', 'x', 'y', 'z'} for label in sources)):
            raise AssertionError('The switch-goal display lost a retained target component or axis.')
    verify_measure_identity(snapshot)


def image_message(image, stamp, frame):
    pixels = np.asarray(image)
    message = Image()
    message.header.stamp = rospy.Time.from_sec(float(stamp))
    message.header.frame_id = frame
    message.height, message.width = pixels.shape[:2]
    message.encoding = "mono8" if pixels.ndim == 2 else "bgr8"
    message.is_bigendian = 0
    message.step = int(pixels.strides[0])
    message.data = pixels.tobytes()
    return message


def prepare(common_path, observations_path, output):
    common, observations = load_inputs(common_path, observations_path)
    result = run(common, observations)
    output.mkdir(parents=True, exist_ok=False)
    (output / "estimation.json").write_text(json.dumps(result.records, indent=2) + "\n")
    cv2.imwrite(str(output / "observation.png"), result.image)
    cv2.imwrite(str(output / "detection.png"), result.detection_image)
    snapshots = []
    messages = []
    for index, stage in enumerate(result.stages):
        snapshot = build_snapshot(result, stage, common)
        marker_array = make_markers(snapshot, common["display"]["seed"],
                                    common["display"]["point_size"])
        save_snapshot(output / "stage_{:02d}".format(index), snapshot, marker_array,
                      seed=common["display"]["seed"], point_size=common["display"]["point_size"])
        snapshots.append(snapshot)
        messages.append(marker_array)
    return common, result, tuple(snapshots), tuple(messages)


class Playback:
    def __init__(self, common, result, snapshots, messages, output, autostart):
        self.common = common
        self.result = result
        self.snapshots = snapshots
        self.messages = messages
        self.output = output
        self.started = time.monotonic() if autostart else None
        self.last_index = None
        self.publisher = rospy.Publisher("/observation_fusion/distributions", MarkerArray,
                                         queue_size=1, latch=True)
        self.prior_publisher = rospy.Publisher("/observation_fusion/prior", MarkerArray,
                                               queue_size=1, latch=True)
        self.metadata = rospy.Publisher("/observation_fusion/snapshot", String,
                                        queue_size=1, latch=True)
        self.image_publisher = rospy.Publisher("/observation_fusion/detection_image", Image,
                                               queue_size=1, latch=True)
        self.service = rospy.Service("/observation_fusion/replay", Trigger, self.replay)
        self.prior_publisher.publish(messages[0])
        self.image_publisher.publish(image_message(result.detection_image,
                                                   snapshots[0].stamp, "camera_image"))
        self.timer = rospy.Timer(rospy.Duration(0.1), self.advance)
        self.advance(None)

    def replay(self, _request):
        # Replays one fixed-time update from its same initial prior; it does not
        # fuse the recorded image into an already conditioned posterior again.
        self.started = time.monotonic()
        self.last_index = None
        self.advance(None)
        return TriggerResponse(True, "Replaying the same immutable update from its common prior.")

    def advance(self, _event):
        duration = float(self.common["display"]["stage_seconds"])
        elapsed = 0 if self.started is None else time.monotonic() - self.started
        index = min(int(elapsed / duration), len(self.snapshots) - 1)
        if index == self.last_index:
            return
        self.last_index = index
        snapshot = self.snapshots[index]
        # The message verified and saved from this snapshot is exactly what is
        # published, including frame, stamp, scales, colors and all sample points.
        self.publisher.publish(self.messages[index])
        self.metadata.publish(String(json.dumps(snapshot_record(snapshot))))
        rospy.loginfo("%s; observations=%s; record=%s", snapshot.stage,
                      snapshot.observation_ids, self.output)


def main():
    package = Path(rospkg.RosPack().get_path("probtf_observation_fusion_demo"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--common", default=str(package / "config/common.yaml"))
    parser.add_argument("--observations", default=str(package / "config/imu_marker.yaml"))
    parser.add_argument("--output")
    args = parser.parse_args(rospy.myargv()[1:])
    default_output = Path("/tmp/probtf_observation_fusion_demo") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.headless:
        output = Path(args.output) if args.output else default_output
        _, result, snapshots, _ = prepare(args.common, args.observations, output)
        print(json.dumps({"output": str(output), "stages": [s.stage for s in snapshots],
                          "valid_observations": len(result.sensor_posteriors)}))
        return
    rospy.init_node("observation_fusion_demo")
    common_path = rospy.get_param("~common_config", args.common)
    observations_path = rospy.get_param("~observations_config", args.observations)
    output = Path(rospy.get_param("~output_dir", args.output or str(default_output)))
    common, result, snapshots, messages = prepare(common_path, observations_path, output)
    common["display"]["stage_seconds"] = float(rospy.get_param(
        "~stage_seconds", common["display"]["stage_seconds"]))
    playback = Playback(common, result, snapshots, messages, output,
                        bool(rospy.get_param("~autostart", True)))
    rospy.spin()


if __name__ == "__main__":
    main()
