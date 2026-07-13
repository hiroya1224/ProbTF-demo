from pathlib import Path
import xml.etree.ElementTree as ET


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
    for name in ("ObjectBelief.msg", "HandBelief.msg", "GraspTarget.msg", "SelectedGraspTarget.msg"):
        text = (_PACKAGE / "msg" / name).read_text(encoding="utf-8")
        assert "probtf_msgs/ProbabilisticTransformStamped transform" in text


def test_installed_runtime_has_no_legacy_tree_or_manual_sample_scripts():
    cmake = (_PACKAGE / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "sample_check_simple_six_dof.py" not in cmake
    assert "show_link_prob_tf.py" not in cmake
    assert "ptf_demo_publisher.py" not in cmake
    assert not (_PACKAGE / "src" / "symaware_grasp" / "prob_tf" / "tree.py").exists()
    assert not (_PACKAGE / "src" / "symaware_grasp" / "ptf_utils.py").exists()


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
