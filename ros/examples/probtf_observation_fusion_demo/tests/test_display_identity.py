"""Core parameters and actual ROS wire data, rather than empirical covariance."""

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from probtf.spherical_law.first_order import point_query
from probtf_observation_fusion_demo.display import (
    make_markers, query_record, serialized_marker_array, verify_wire_identity,
)
from probtf_observation_fusion_demo.model import load_inputs, run
from probtf_observation_fusion_demo.node import build_snapshot, verify_adapter_identity


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def combined():
    common, observations = load_inputs(PACKAGE / "config/common.yaml",
                                       PACKAGE / "config/imu_marker.yaml")
    result = run(common, observations)
    stage = result.stages[-1]
    return common, result, stage, build_snapshot(result, stage, common)


def test_adapter_numbers_weights_and_provenance_are_core_queries(combined):
    common, result, stage, snapshot = combined
    paths = dict(result.link_paths)
    paths.update(hand=result.hand_path, camera=result.camera_path, marker=result.marker_path)
    components = {component.label: component for component in stage.components}
    axis = common["display"]["axis_length"]
    points = {"origin": [0, 0, 0], "x": [axis, 0, 0],
              "y": [0, axis, 0], "z": [0, 0, axis]}
    for display in snapshot.points:
        component = components[display.component_id]
        if display.axis == "observed" or display.frame_label.startswith("IMU gravity at "):
            estimate = next(e for e in result.local_estimates if e.observed_direction is not None)
            path = paths[estimate.diagnostics["target_link"]]
            point = np.zeros(3) if display.axis == "origin" else axis * estimate.observed_direction
        else:
            path, point = paths[display.frame_label], points[display.axis]
        source = point_query(component.state, path, point,
                             display.query.query_id, result.root_frame)
        record = query_record(display)
        np.testing.assert_array_equal(record["mean"], source.mean)
        np.testing.assert_array_equal(record["covariance"], source.covariance)
        assert record["weight"] == component.weight
        assert record["source_state_id"] == component.state.state_id
        assert record["source_stamp"] == common["update"]["stamp"]
        assert record["source_command"] == common["update"]["command"]
        assert record["source_frame"] == result.root_frame
    assert snapshot.observation_ids == ("gravity_recording_001", "marker_image_001")
    # A plausible-looking cloud from a stale version is still a hard failure.
    stale = replace(snapshot.points[0].query, version=snapshot.version - 1)
    wrong = replace(snapshot, points=(replace(snapshot.points[0], query=stale),) + snapshot.points[1:])
    with pytest.raises(AssertionError):
        verify_adapter_identity(wrong, stage)


def test_actual_ros_wire_message_is_exact_and_mutation_is_detected(combined):
    common, _, _, snapshot = combined
    settings = common["display"]
    message = make_markers(snapshot, settings["seed"], settings["point_size"])
    payload = verify_wire_identity(snapshot, message, seed=settings["seed"],
                                   point_size=settings["point_size"])
    assert payload == serialized_marker_array(message)
    point_marker = message.markers[1]
    assert point_marker.scale.x == settings["point_size"]
    assert point_marker.header.frame_id == snapshot.frame_id
    assert point_marker.header.stamp.to_sec() == snapshot.stamp
    assert len(point_marker.points) == settings["body_samples"]
    point_marker.points[0].x += .001
    with pytest.raises(AssertionError):
        verify_wire_identity(snapshot, message, seed=settings["seed"],
                             point_size=settings["point_size"])


def test_render_seed_does_not_modify_any_source_distribution(combined):
    common, _, stage, snapshot = combined
    states = tuple(component.state.state_id for component in stage.components)
    before = tuple((point.query.mean.copy(), point.query.covariance.copy()) for point in snapshot.points)
    first = serialized_marker_array(make_markers(snapshot, 29))
    second = serialized_marker_array(make_markers(snapshot, 311))
    assert first != second
    assert states == tuple(component.state.state_id for component in stage.components)
    for point, (mean, covariance) in zip(snapshot.points, before):
        np.testing.assert_array_equal(point.query.mean, mean)
        np.testing.assert_array_equal(point.query.covariance, covariance)


def test_original_rviz_bytes_and_copied_layout_are_preserved():
    original_path = PACKAGE.parent / "probtf_switch_press_demo/rviz/switch_press_demo.rviz"
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == (
        "3ca46dbb7e5eb1a7502bc584f933eab20c08a5f561ea77e03b0857f4aff92c58")
    original = yaml.safe_load(original_path.read_text())
    copied = yaml.safe_load((PACKAGE / "rviz/observation_fusion_demo.rviz").read_text())
    assert copied["Window Geometry"] == original["Window Geometry"]
    first = copied["Visualization Manager"]["Views"]
    second = original["Visualization Manager"]["Views"]
    # Frame names necessarily follow the actual new input root; camera position,
    # orbit, distance and all saved view parameters otherwise stay unchanged.
    for value in [second["Current"]] + second["Saved"]:
        value["Target Frame"] = "base_link"
    assert first == second
    settings = yaml.safe_load((PACKAGE / "config/common.yaml").read_text())["display"]
    body = next(d for d in original["Visualization Manager"]["Displays"]
                if d.get("Name") == "ProbTF robot joints")
    marker = next(d for d in original["Visualization Manager"]["Displays"]
                  if d.get("Name") == "ProbTF observed AR marker")
    # The user explicitly requested a 0.75 rendering scale on 2026-10-16.
    # The protected RViz files themselves still remain byte-identical.
    assert settings["point_size"] == .75*body["Point Size"] == .75*marker["Point Size"]
    assert settings["axis_length"] == .75*body["Axis Length"] == .75*marker["Axis Length"]
    assert settings["body_samples"] == body["Sample Count"]
    assert settings["marker_samples"] == marker["Sample Count"]
