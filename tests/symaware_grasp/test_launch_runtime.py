from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


_PACKAGE = (
    Path(__file__).resolve().parents[2]
    / "ros"
    / "examples"
    / "symaware_grasp"
)


def test_launches_use_native_v2_topics_and_static_broadcaster():
    for name in ("probabilistic_tf_demo.launch", "prob_tf_link_cloud.launch"):
        path = _PACKAGE / "launch" / name
        ET.parse(str(path))
        text = path.read_text(encoding="utf-8")
        assert "probtf_static_broadcaster.py" in text
        assert "/probtf_static" in text
        assert "ProbabilisticTF" not in text
        assert "grasp_target_ptfs" not in text
        assert "selected_grasp_target_prob_tf" not in text


def test_application_messages_embed_complete_v2_transform_payloads():
    for name in ("ObjectBelief.msg", "GraspTarget.msg", "SelectedGraspTarget.msg"):
        text = (_PACKAGE / "msg" / name).read_text(encoding="utf-8")
        assert "probtf_msgs/ProbabilisticTransformStamped transform" in text
    assert not (_PACKAGE / "msg" / "HandBelief.msg").exists()


def test_installed_runtime_has_no_legacy_tree_or_manual_sample_scripts():
    cmake = (_PACKAGE / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "sample_check_simple_six_dof.py" not in cmake
    assert "show_link_prob_tf.py" not in cmake
    assert "ptf_demo_publisher.py" not in cmake
    assert "hand_prob_tf_publisher.py" not in cmake
    assert not (_PACKAGE / "src" / "symaware_grasp" / "prob_tf" / "tree.py").exists()
    assert not (_PACKAGE / "src" / "symaware_grasp" / "ptf_utils.py").exists()
    assert not (_PACKAGE / "src" / "symaware_grasp" / "ee_belief.py").exists()
    assert not (_PACKAGE / "src" / "symaware_grasp" / "distribution_metrics.py").exists()


def test_grasp_demo_launch_is_visualization_only_until_explicit_ik_run():
    launch_path = _PACKAGE / "launch" / "probabilistic_tf_demo.launch"
    root = ET.parse(str(launch_path)).getroot()
    nodes = {element.attrib["name"]: element for element in root.findall("node")}

    assert "symmetry_aware_ik_node" not in nodes
    assert "hand_prob_tf_publisher" not in nodes
    assert "hand_prob_tf_visualizer" not in nodes
    assert "selected_grasp_target_visualizer" not in nodes
    assert "object_prob_tf_visualizer" in nodes

    object_parameters = {
        element.attrib.get("name", element.attrib.get("param")): (
            element.attrib["value"]
            if "value" in element.attrib
            else (element.text or "").strip()
        )
        for element in nodes["object_prob_tf_visualizer"]
        if element.tag in ("param", "rosparam")
    }
    assert object_parameters["geometry_marker_topic"] == "/symaware_grasp/object_geometry"
    assert object_parameters["geometry_type"] == "cylinder"
    assert object_parameters["geometry_scale"] == "[0.10, 0.10, 0.18]"

    launch_text = launch_path.read_text(encoding="utf-8")
    assert "rosrun symaware_grasp symmetry_aware_ik_node.py" in launch_text
    assert "hand_belief" not in launch_text
    assert "ik_method" not in launch_text
    assert "selected_target_topic" not in launch_text


def test_demo_grasp_is_identity_so_target_belief_is_used_without_propagation():
    config = yaml.safe_load(
        (_PACKAGE / "config" / "grasp_library.yaml").read_text(encoding="utf-8")
    )
    candidates = config["objects"]["demo_cylinder"]["candidates"]

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["grasp_id"] == "cylinder_side_grasp"
    assert candidate["object_to_grasp_position"] == [0.0, 0.0, 0.0]
    assert candidate["approach_axis"] == [1.0, 0.0, 0.0]
    assert candidate["finger_axis"] == [0.0, 0.0, 1.0]
    assert candidate["weight"] == 1.0


def test_ik_runtime_is_pointwise_only_without_hand_distribution():
    solver = (
        _PACKAGE / "src" / "symaware_grasp" / "symmetry_aware_ik.py"
    ).read_text(encoding="utf-8")
    node = (_PACKAGE / "scripts" / "symmetry_aware_ik_node.py").read_text(
        encoding="utf-8"
    )

    assert 'METHOD_POINTWISE = "symmetry_aware_pointwise"' in solver
    assert "Bhattacharyya" not in solver + node
    assert "bhattacharyya" not in solver + node
    assert "EndEffectorBeliefModel" not in solver + node
    assert "METHOD_DETERMINISTIC" not in solver + node
    assert "deterministic_ik_result" not in node
    assert "SelectedGraspTarget" not in node
    assert "selected_target_publisher" not in node
    assert "selected_target_to_msg" not in node
    assert "wait_for_fresh_target_array" in node
    assert "listener_ready_time" in node


def test_rviz_scene_contains_cylinder_and_parallel_gripper_without_hand_cloud():
    rviz = (_PACKAGE / "rviz" / "probabilistic_tf_demo.rviz").read_text(
        encoding="utf-8"
    )
    assert "Marker Topic: /symaware_grasp/object_geometry" in rviz
    assert "Cylinder Object" in rviz
    assert "Hand Uncertainty" not in rviz
    assert "/hand_axes_cloud" not in rviz
    assert "Selected Grasp" not in rviz
    assert "/selected_grasp_axes_cloud" not in rviz
    assert "/selected_grasp_mode_axes" not in rviz

    urdf_root = ET.parse(str(_PACKAGE / "urdf" / "simple_six_dof_arm.urdf")).getroot()
    tool = urdf_root.find("./link[@name='tool0']")
    assert tool is not None
    visuals = {element.attrib.get("name") for element in tool.findall("visual")}
    assert {
        "gripper_palm",
        "gripper_left_finger",
        "gripper_right_finger",
        "gripper_left_pad",
        "gripper_right_pad",
    }.issubset(visuals)


def test_link_cloud_samples_only_after_listener_point_moment_lookup():
    source = (_PACKAGE / "scripts" / "prob_tf_link_cloud_node.py").read_text(encoding="utf-8")
    lookup_index = source.index("self.listener.lookup_point_moments")
    sample_index = source.index("self.rng.multivariate_normal")

    assert lookup_index < sample_index
    assert "sample_transform_distribution" not in source
    assert "update_sample" not in source


def test_link_cloud_launch_connects_joint_sliders_to_dynamic_v2_records():
    launch_path = _PACKAGE / "launch" / "prob_tf_link_cloud.launch"
    root = ET.parse(str(launch_path)).getroot()
    arguments = {element.attrib["name"]: element for element in root.findall("arg")}
    nodes = {element.attrib["name"]: element for element in root.findall("node")}

    assert arguments["joint_gui"].attrib["default"] == "$(arg rviz)"
    assert nodes["prob_tf_joint_sliders"].attrib["pkg"] == "joint_state_publisher_gui"
    broadcaster_params = {
        element.attrib["name"]: element.attrib["value"]
        for element in nodes["probtf_static_broadcaster"].findall("param")
    }
    assert broadcaster_params["follow_joint_states"] == "true"
    assert broadcaster_params["joint_states_topic"] == "joint_states"

    package_text = (_PACKAGE / "package.xml").read_text(encoding="utf-8")
    assert "<exec_depend>joint_state_publisher_gui</exec_depend>" in package_text
