#!/usr/bin/env python3

"""Visualize deflecomp frame points queried from the Prob-TF v2 runtime."""

import numpy as np
import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from deflecomp_core.robot.urdf_info import (
    infer_base_link,
    infer_tip_link,
    load_urdf_model_info,
)
from deflecomp_ros.probtf_consumer import (
    covariance_axis_segments,
    lookup_point_moment,
)
from probtf.graph import ProbTfGraphError
from probtf.temporal import TemporalPolicy
from probtf_ros import RosProbTfListener


_COLORS = (
    (0.18, 0.55, 0.95),
    (0.95, 0.38, 0.22),
    (0.20, 0.75, 0.42),
    (0.76, 0.45, 0.90),
)


def _string_list(value):
    entries = value.split(",") if isinstance(value, str) else value
    return tuple(
        entry
        for entry in (str(item).strip().strip("/") for item in entries)
        if entry
    )


def _vector3_param(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("~{} must contain three finite values.".format(name))
    return result


def _point(values):
    message = Point()
    message.x, message.y, message.z = (float(value) for value in values)
    return message


class DeflecompProbTfPointMomentsNode:
    def __init__(self):
        dynamic_topic = rospy.get_param("~dynamic_topic", "/deflecomp/probtf")
        static_topic = rospy.get_param(
            "~static_topic",
            "/deflecomp/probtf_static",
        )
        self.listener = RosProbTfListener(
            dynamic_topic=dynamic_topic,
            static_topic=static_topic,
            max_records_per_edge=int(rospy.get_param("~max_records_per_edge", 500)),
        )

        urdf_path = str(rospy.get_param("~urdf_path", "")).strip()
        if not urdf_path:
            raise ValueError("~urdf_path is required.")
        model_info = load_urdf_model_info(urdf_path)
        self.target_frame = str(
            rospy.get_param("~target_frame", "")
        ).strip().strip("/") or infer_base_link(model_info)
        tip_frame = str(rospy.get_param("~tip_frame", "")).strip().strip("/")
        tip_frame = tip_frame or infer_tip_link(model_info)
        explicit_sources = _string_list(rospy.get_param("~source_frames", ()))
        if explicit_sources:
            self.source_frames = explicit_sources
        else:
            prefixes = _string_list(
                rospy.get_param("~frame_prefixes", ("ref", "cmd", "equil"))
            )
            self.source_frames = tuple(
                "{}/{}".format(prefix, tip_frame) for prefix in prefixes
            )
        if not self.source_frames:
            raise ValueError("At least one source frame is required.")

        self.source_point = _vector3_param(
            rospy.get_param("~source_point", (0.0, 0.0, 0.0)),
            "source_point",
        )
        self.sigma_scale = float(rospy.get_param("~sigma_scale", 2.0))
        self.point_scale = float(rospy.get_param("~point_scale", 0.025))
        self.axis_width = float(rospy.get_param("~axis_width", 0.006))
        self.lookup_rate_hz = float(rospy.get_param("~lookup_rate_hz", 10.0))
        if self.lookup_rate_hz <= 0.0:
            raise ValueError("~lookup_rate_hz must be positive.")
        if self.point_scale <= 0.0 or self.axis_width <= 0.0:
            raise ValueError("~point_scale and ~axis_width must be positive.")

        marker_topic = rospy.get_param(
            "~marker_topic",
            "/deflecomp/probtf_point_moments",
        )
        self.publisher = rospy.Publisher(marker_topic, MarkerArray, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / self.lookup_rate_hz),
            self._on_timer,
        )
        rospy.on_shutdown(self.listener.unregister)
        rospy.loginfo(
            "deflecomp Prob-TF consumer: %s <- %s via %s and %s",
            self.target_frame,
            ", ".join(self.source_frames),
            dynamic_topic,
            static_topic,
        )

    def _on_timer(self, event):
        del event
        observations = []
        for source_frame in self.source_frames:
            try:
                observations.append(
                    lookup_point_moment(
                        self.listener,
                        self.target_frame,
                        source_frame,
                        self.source_point,
                        policy=TemporalPolicy.LATEST_COMMON,
                    )
                )
            except (ProbTfGraphError, RuntimeError, ValueError) as error:
                rospy.logwarn_throttle(
                    3.0,
                    "Deflecomp Prob-TF lookup %s <- %s is unavailable: %s",
                    self.target_frame,
                    source_frame,
                    error,
                )
        self.publisher.publish(self._markers(observations))

    def _markers(self, observations):
        output = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        output.markers.append(clear)
        lifetime = rospy.Duration.from_sec(2.5 / self.lookup_rate_hz)
        for index, observation in enumerate(observations):
            color = _COLORS[index % len(_COLORS)]
            namespace = observation.source_frame.replace("/", "_")
            stamp = rospy.Time.from_sec(observation.resolved_stamp)

            mean = Marker()
            mean.header.frame_id = observation.target_frame
            mean.header.stamp = stamp
            mean.ns = namespace
            mean.id = 2 * index
            mean.type = Marker.SPHERE
            mean.action = Marker.ADD
            mean.pose.position = _point(observation.mean)
            mean.pose.orientation.w = 1.0
            mean.scale.x = self.point_scale
            mean.scale.y = self.point_scale
            mean.scale.z = self.point_scale
            mean.color.r, mean.color.g, mean.color.b = color
            mean.color.a = 0.95
            mean.lifetime = lifetime
            output.markers.append(mean)

            axes = Marker()
            axes.header.frame_id = observation.target_frame
            axes.header.stamp = stamp
            axes.ns = namespace
            axes.id = 2 * index + 1
            axes.type = Marker.LINE_LIST
            axes.action = Marker.ADD
            axes.pose.orientation.w = 1.0
            axes.scale.x = self.axis_width
            axes.color.r, axes.color.g, axes.color.b = color
            axes.color.a = 0.8
            axes.lifetime = lifetime
            for start, end in covariance_axis_segments(
                observation.mean,
                observation.covariance,
                self.sigma_scale,
            ):
                axes.points.extend((_point(start), _point(end)))
            output.markers.append(axes)
        return output


def main():
    rospy.init_node("deflecomp_probtf_point_moments")
    DeflecompProbTfPointMomentsNode()
    rospy.spin()


if __name__ == "__main__":
    main()
