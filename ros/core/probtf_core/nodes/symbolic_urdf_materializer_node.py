#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import rospy
from scipy.spatial.transform import Rotation
from std_msgs.msg import String

from probtf.symbolic_urdf import parse_symbolic_urdf
from probtf_msgs.msg import ProbabilisticTF


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
        self.subscribers = []
        for placeholder, binding in self.bindings.items():
            field = binding.get("field")
            if field not in ("position", "orientation_rpy"):
                raise ValueError("binding field must be 'position' or 'orientation_rpy'")
            self.subscribers.append(
                rospy.Subscriber(
                    binding["topic"],
                    ProbabilisticTF,
                    self._update,
                    callback_args=(placeholder, field),
                    queue_size=10,
                )
            )

    def _update(self, message, callback_args):
        placeholder, field = callback_args
        if message.header.frame_id and message.header.frame_id != message.parent_frame_id:
            rospy.logwarn_throttle(2.0, "Ignoring ProbTF with conflicting parent frame fields")
            return
        if field == "position":
            covariance = np.asarray(message.position_covariance, dtype=float).reshape(3, 3)
            if float(np.max(np.linalg.eigvalsh(covariance))) > self.position_variance_threshold:
                return
            self.values[placeholder] = [
                message.position_mean.x,
                message.position_mean.y,
                message.position_mean.z,
            ]
        else:
            parameter = np.asarray(message.orientation_bingham.matrix, dtype=float).reshape(4, 4)
            eigenvalues = np.linalg.eigvalsh(0.5 * (parameter + parameter.T))
            if float(eigenvalues[-1] - eigenvalues[-2]) < self.orientation_concentration_threshold:
                return
            quaternion = message.orientation_mode
            self.values[placeholder] = Rotation.from_quat(
                [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
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
