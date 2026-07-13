from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from deflecomp_ros.probtf_consumer import (
    covariance_axis_segments,
    lookup_point_moment,
)
from probtf_ros import ProbTfListener
from probtf_ros.tf_bridge import deterministic_tf_to_record


ROOT = Path(__file__).resolve().parents[2]
DEFLECOMP_ROS = ROOT / "ros" / "examples" / "deflecomp" / "deflecomp_ros"


class Stamp:
    def __init__(self, seconds):
        self.seconds = float(seconds)

    def to_sec(self):
        return self.seconds


class Vector3:
    def __init__(self, values):
        self.x, self.y, self.z = (float(value) for value in values)


class Quaternion:
    def __init__(self, wxyz=(1.0, 0.0, 0.0, 0.0)):
        self.w, self.x, self.y, self.z = (float(value) for value in wxyz)


class TransformMessage:
    def __init__(self, parent, child, stamp, translation):
        self.header = type("Header", (), {})()
        self.header.frame_id = parent
        self.header.stamp = Stamp(stamp)
        self.child_frame_id = child
        self.transform = type("Transform", (), {})()
        self.transform.translation = Vector3(translation)
        self.transform.rotation = Quaternion()


def test_tf_imported_v2_records_drive_path_lookup_and_point_moments():
    listener = ProbTfListener()
    listener.receive_record(
        deterministic_tf_to_record(
            TransformMessage(
                "base_link",
                "ref/base_link",
                0.0,
                (1.0, 0.0, 0.0),
            ),
            authority="static_transform_publisher",
            is_static=True,
            edge_id="base_to_ref",
        )
    )
    listener.receive_record(
        deterministic_tf_to_record(
            TransformMessage(
                "ref/base_link",
                "ref/tool",
                2.0,
                (0.0, 2.0, 0.0),
            ),
            authority="robot_state_publisher",
            edge_id="ref_base_to_tool",
        )
    )

    observation = lookup_point_moment(
        listener,
        "base_link",
        "ref/tool",
        (0.0, 0.0, 3.0),
    )

    assert observation.target_frame == "base_link"
    assert observation.source_frame == "ref/tool"
    assert observation.resolved_stamp == 2.0
    assert observation.edge_ids == ("ref_base_to_tool", "base_to_ref")
    np.testing.assert_allclose(observation.mean, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(observation.covariance, np.zeros((3, 3)))
    assert observation.approximation.lossy is False


def test_covariance_axes_are_derived_from_point_moments():
    mean = np.array([1.0, 2.0, 3.0])
    segments = covariance_axis_segments(
        mean,
        np.diag([1.0, 4.0, 9.0]),
        sigma_scale=2.0,
    )

    assert len(segments) == 3
    for start, end in segments:
        np.testing.assert_allclose(0.5 * (start + end), mean)
    np.testing.assert_allclose(
        [np.linalg.norm(end - start) for start, end in segments],
        [4.0, 8.0, 12.0],
    )


def test_deflecomp_frames_launch_scopes_imported_tf_to_v2_topics():
    launch_path = DEFLECOMP_ROS / "launch" / "deflecomp_frames.launch"
    root = ET.parse(str(launch_path)).getroot()
    launch_args = {
        element.attrib["name"]: element.attrib["default"]
        for element in root.findall("arg")
    }
    assert launch_args["probtf_tf_import_rate_hz"] == "50.0"
    assert launch_args["probtf_marker_rate_hz"] == "50.0"

    runtime_group = root.find("./group[@ns='deflecomp']")
    assert runtime_group is not None
    assert runtime_group.attrib["if"] == "$(arg enable_probtf_runtime)"

    bridge = runtime_group.find("include")
    assert bridge is not None
    assert "probtf_bridge.launch" in bridge.attrib["file"]
    bridge_args = {
        element.attrib["name"]: element.attrib["value"]
        for element in bridge.findall("arg")
    }
    assert bridge_args["import_tf"] == "true"
    assert bridge_args["export_tf"] == "false"
    assert bridge_args["probtf_topic"] == "$(arg probtf_topic)"
    assert bridge_args["probtf_static_topic"] == "$(arg probtf_static_topic)"
    assert bridge_args["tf_import_max_rate_hz"] == "$(arg probtf_tf_import_rate_hz)"

    deflecomp_include = next(
        element
        for element in root.findall("include")
        if "deflecomp.launch" in element.attrib.get("file", "")
    )
    deflecomp_args = {
        element.attrib["name"]: element.attrib["value"]
        for element in deflecomp_include.findall("arg")
    }
    assert deflecomp_args["particle_scan_enabled"] == "$(arg particle_scan_enabled)"

    consumer = runtime_group.find("node[@type='probtf_point_moments_node.py']")
    assert consumer is not None
    parameters = {
        element.attrib["name"]: element.attrib["value"]
        for element in consumer.findall("param")
    }
    assert parameters["dynamic_topic"] == "$(arg probtf_topic)"
    assert parameters["static_topic"] == "$(arg probtf_static_topic)"
    assert parameters["lookup_rate_hz"] == "$(arg probtf_marker_rate_hz)"

    rviz = root.find("node[@pkg='rviz']")
    gui = root.find("./group[@ns='ref']/node[@pkg='joint_state_publisher_gui']")
    headless = root.find("./group[@ns='ref']/node[@pkg='joint_state_publisher']")
    assert rviz is not None and rviz.attrib["if"] == "$(arg viewer)"
    assert gui is not None and gui.attrib["if"] == "$(arg viewer)"
    assert headless is not None and headless.attrib["unless"] == "$(arg viewer)"


def test_runtime_consumer_uses_v2_listener_without_stiffness_pose_encoding():
    consumer_module = (
        DEFLECOMP_ROS / "src" / "deflecomp_ros" / "probtf_consumer.py"
    ).read_text(encoding="utf-8")
    consumer_node = (
        DEFLECOMP_ROS / "nodes" / "probtf_point_moments_node.py"
    ).read_text(encoding="utf-8")
    config = (DEFLECOMP_ROS / "config" / "probtf_runtime.yaml").read_text(
        encoding="utf-8"
    )
    estimator_node = (DEFLECOMP_ROS / "nodes" / "deflecomp_node.py").read_text(
        encoding="utf-8"
    )

    assert "RosProbTfListener" in consumer_node
    assert "lookup_path" in consumer_module
    assert "lookup_point_moments" in consumer_module
    assert "ProbabilisticTF" not in consumer_module + consumer_node
    assert "kp_" not in consumer_module + consumer_node + config
    assert "probtf" not in estimator_node.lower()
    assert "marker_max_age" in consumer_node
    assert "lookup_rate_hz: 50.0" in config
    assert "now - observation.resolved_stamp" in consumer_node
    assert "_COLORS[source_index % len(_COLORS)]" in consumer_node
