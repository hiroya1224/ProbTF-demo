import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest
import rospy
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from prob_artag_detector import (
    CameraModel,
    MarkerObservation,
    PoseMixtureEstimator,
    approximate_camera_model,
    ippe_square_object_points,
    load_camera_calibration,
    project_points,
)
from prob_artag_detector.ros_markers import build_pose_mixture_markers
from probtf_ros import transform_distribution_from_msg, transform_distribution_to_msg


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "ros"
    / "examples"
    / "prob_artag_detector"
)


def test_fov_fallback_has_size_matched_centered_intrinsics():
    camera = approximate_camera_model(640, 480, horizontal_fov_deg=60.0)
    expected_focal = 320.0 / math.tan(math.radians(30.0))
    assert camera.width == 640
    assert camera.height == 480
    assert camera.camera_matrix[0, 0] == pytest.approx(expected_focal)
    assert camera.camera_matrix[1, 1] == pytest.approx(expected_focal)
    assert camera.camera_matrix[0, 2] == pytest.approx(319.5)
    assert camera.camera_matrix[1, 2] == pytest.approx(239.5)
    assert not camera.has_distortion
    with pytest.raises(ValueError, match="width and height"):
        approximate_camera_model(0, 480)
    with pytest.raises(ValueError, match="horizontal_fov_deg"):
        approximate_camera_model(640, 480, 180.0)


def test_ros_calibration_yaml_loads_and_invalid_distortion_is_rejected(tmp_path):
    calibration_path = tmp_path / "camera.yaml"
    calibration_path.write_text(
        """image_width: 640
image_height: 480
camera_name: test_camera
distortion_model: plumb_bob
camera_matrix:
  rows: 3
  cols: 3
  data: [610.0, 0.0, 320.0, 0.0, 608.0, 240.0, 0.0, 0.0, 1.0]
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.01, -0.02, 0.0, 0.0, 0.0]
""",
        encoding="utf-8",
    )
    calibration = load_camera_calibration(calibration_path)
    assert calibration.camera_name == "test_camera"
    assert calibration.distortion_model == "plumb_bob"
    assert calibration.model.width == 640
    assert calibration.model.camera_matrix[0, 0] == pytest.approx(610.0)

    info = SimpleNamespace(
        width=640,
        height=480,
        K=calibration.model.camera_matrix.reshape(-1),
        D=np.zeros(4),
        distortion_model="equidistant",
    )
    with pytest.raises(ValueError, match="distortion_model"):
        CameraModel.from_camera_info(info)
    info.distortion_model = "plumb_bob"
    info.width = 0
    with pytest.raises(ValueError, match="width and height"):
        CameraModel.from_camera_info(info)
    info.width = 640
    info.binning_x = 2
    with pytest.raises(ValueError, match="Binned"):
        CameraModel.from_camera_info(info)
    info.binning_x = 0
    info.roi = SimpleNamespace(
        x_offset=1, y_offset=0, width=0, height=0, do_rectify=False
    )
    with pytest.raises(ValueError, match="ROI"):
        CameraModel.from_camera_info(info)

    malformed_path = tmp_path / "malformed.yaml"
    malformed_path.write_text("camera_matrix: [", encoding="utf-8")
    with pytest.raises(ValueError, match="could not be parsed"):
        load_camera_calibration(malformed_path)


def _mixture_result(fallback=False):
    camera = CameraModel(
        np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        640,
        480,
    )
    rotation, _ = cv2.Rodrigues(np.array([0.12, -0.08, 0.03]))
    translation = np.array([0.02, -0.01, 0.8])
    corners = project_points(
        ippe_square_object_points(0.12), rotation, translation, camera
    )
    observation = MarkerObservation(7, corners, np.eye(8) * 0.25)
    return PoseMixtureEstimator(0.12).estimate(
        observation,
        camera,
        "camera_optical_frame",
        "apriltag_7",
        3.0,
        camera_model_source_id=(
            "camera_model:fallback_hfov_60.000deg" if fallback else "camera_model:test"
        ),
        camera_model_is_approximate=fallback,
    )


def test_rviz_markers_keep_every_mode_and_identify_tf_projection():
    result = _mixture_result()
    header = Header(stamp=rospy.Time.from_sec(3.0), frame_id="camera_optical_frame")
    output = build_pose_mixture_markers(
        (result,),
        header,
        0.12,
        lifetime=rospy.Duration.from_sec(0.5),
    )
    assert output.markers[0].action == Marker.DELETEALL
    assert len(output.markers) == 1 + 4 * len(result.weights)
    typed = {
        marker_type: [item for item in output.markers if item.type == marker_type]
        for marker_type in (Marker.CUBE, Marker.LINE_LIST, Marker.SPHERE, Marker.TEXT_VIEW_FACING)
    }
    assert all(len(items) == len(result.weights) for items in typed.values())
    assert all(item.header.frame_id == "camera_optical_frame" for item in output.markers)
    assert sum("[TF]" in item.text for item in typed[Marker.TEXT_VIEW_FACING]) == 1
    assert all(item.scale.x > 0.0 for item in typed[Marker.SPHERE])
    assert all(
        "conditional_translation_2sigma_at_mode" in item.ns
        for item in typed[Marker.SPHERE]
    )
    assert all(item.lifetime.to_sec() == pytest.approx(0.5) for item in output.markers[1:])

    cleared = build_pose_mixture_markers((), header, 0.12)
    assert len(cleared.markers) == 1
    assert cleared.markers[0].action == Marker.DELETEALL


def test_fallback_calibration_is_explicit_in_wire_provenance():
    result = _mixture_result(fallback=True)
    message = transform_distribution_to_msg(
        result.record, time_factory=lambda value: value
    )
    record = transform_distribution_from_msg(message)
    source_id = "camera_model:fallback_hfov_60.000deg"
    assert source_id in record.provenance.source_ids
    assert "uncalibrated fallback" in record.provenance.detail
    assert "not included" in record.approximation.detail
    assert all(
        source_id in component.provenance.source_ids
        and "uncalibrated fallback" in component.approximation.detail
        for component in record.distribution.components
    )


def test_real_camera_launch_wires_camera_detector_bridge_and_rviz():
    launch_path = PACKAGE_ROOT / "launch" / "prob_artag_real_camera_demo.launch"
    root = ET.parse(str(launch_path)).getroot()
    nodes = {item.attrib.get("name"): item for item in root.findall("node")}
    assert nodes["prob_artag_camera"].attrib["type"] == "prob_artag_camera_demo_node.py"
    assert nodes["prob_artag_camera_optical_tf"].attrib["pkg"] == "tf2_ros"
    assert nodes["prob_artag_rviz"].attrib["pkg"] == "rviz"

    bridge_include = next(
        item
        for item in root.findall(".//include")
        if "probtf_bridge.launch" in item.attrib["file"]
    )
    bridge_args = {item.attrib["name"]: item.attrib["value"] for item in bridge_include}
    assert bridge_args["import_tf"] == "false"
    assert bridge_args["export_tf"] == "true"
    assert bridge_args["tf_export_policy"] == "highest_weight_component_mode"
    assert bridge_args["probtf_topic"] == "$(arg probtf_topic)"

    arguments = {item.attrib["name"]: item.attrib["default"] for item in root.findall("arg")}
    assert arguments["probtf_topic"] == "/prob_artag_demo/probtf"

    rviz_text = (PACKAGE_ROOT / "rviz" / "prob_artag_real_camera_demo.rviz").read_text(
        encoding="utf-8"
    )
    assert "Fixed Frame: camera_link" in rviz_text
    assert "Marker Topic: /prob_artag_detector/markers" in rviz_text
    assert "Image Topic: /prob_artag_detector/debug_image" in rviz_text
