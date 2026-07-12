#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import rospy
from scipy.spatial.transform import Rotation
from std_msgs.msg import String

from probtf.symbolic_urdf import parse_symbolic_urdf
from probtf_estimators.materialization import summarize_transform_for_materialization
from probtf_msgs.msg import ProbabilisticTransformStamped
from probtf_ros.v2_conversions import transform_distribution_from_msg


class SymbolicUrdfMaterializerNode:
    def __init__(self):
        template_path = Path(rospy.get_param("~template_file")).expanduser()
        self.template = parse_symbolic_urdf(template_path.read_text())
        self.output_parameter = rospy.get_param("~output_parameter", "/robot_description")
        output_file = rospy.get_param("~output_file", "")
        self.output_file = Path(output_file).expanduser() if output_file else None
        self.position_variance_threshold = float(
            rospy.get_param("~position_variance_threshold", 1e-4)
        )
        self.orientation_concentration_threshold = float(
            rospy.get_param("~orientation_concentration_threshold", 20.0)
        )
        self.values = dict(rospy.get_param("~static_substitutions", {}))
        self.bindings = rospy.get_param("~bindings")
        if set(self.bindings) | set(self.values) != set(self.template.placeholder_names):
            raise ValueError(
                "bindings and static_substitutions must cover every symbolic URDF placeholder"
            )
        self.publisher = rospy.Publisher("~materialized_urdf", String, queue_size=1, latch=True)
        self.parent_frame_id = rospy.get_param("~parent_frame_id", "").lstrip("/")
        self.child_frame_id = rospy.get_param("~child_frame_id", "").lstrip("/")
        for placeholder, binding in self.bindings.items():
            field = binding.get("field")
            if field not in ("position", "orientation_rpy"):
                raise ValueError("binding field must be 'position' or 'orientation_rpy'")
        self.subscriber = rospy.Subscriber(
            rospy.get_param("~input_topic", "/probtf"),
            ProbabilisticTransformStamped,
            self._update,
            queue_size=10,
        )

    def _update(self, message):
        if self.parent_frame_id and message.header.frame_id.lstrip("/") != self.parent_frame_id:
            return
        if self.child_frame_id and message.child_frame_id.lstrip("/") != self.child_frame_id:
            return
        try:
            summary = summarize_transform_for_materialization(
                transform_distribution_from_msg(message),
                integration_steps=rospy.get_param("~integration_steps", 120),
            )
        except (TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            rospy.logwarn_throttle(2.0, "ProbTF materialization rejected: %s", error)
            return

        position_ready = (
            float(np.max(np.linalg.eigvalsh(summary.position_covariance)))
            <= self.position_variance_threshold
        )
        orientation_ready = (
            summary.orientation_concentration_gap
            >= self.orientation_concentration_threshold
        )
        for placeholder, binding in self.bindings.items():
            field = binding["field"]
            if field == "position" and position_ready:
                self.values[placeholder] = summary.position_mean.tolist()
            elif field == "orientation_rpy" and orientation_ready:
                quaternion = summary.orientation_mode_wxyz
                self.values[placeholder] = Rotation.from_quat(
                    [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
                ).as_euler("xyz").tolist()
        self._materialize_if_complete()

    def _materialize_if_complete(self):
        if set(self.values) != set(self.template.placeholder_names):
            return
        urdf = self.template.materialize(self.values)
        rospy.set_param(self.output_parameter, urdf)
        if self.output_file is not None:
            self.output_file.write_text(urdf)
        self.publisher.publish(String(data=urdf))


if __name__ == "__main__":
    rospy.init_node("probtf_symbolic_urdf_materializer")
    SymbolicUrdfMaterializerNode()
    rospy.spin()
