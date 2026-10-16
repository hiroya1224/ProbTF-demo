"""Information boundaries and actual multi-hypothesis sensor integration."""

from copy import deepcopy
import io
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
for source in (ROOT / "ros/core/probtf_core/src",
               ROOT / "ros/examples/prob_artag_detector/src",
               ROOT / "ros/estimators/probtf_estimators/src",
               ROOT / "ros/examples/probtf_observation_fusion_demo/src"):
    sys.path.insert(0, str(source))

from prob_artag_detector.camera import ippe_square_object_points, pose_jacobian
from probtf.geometry import quat_to_rotmat
from probtf.spherical_law.first_order import JointTransformState, point_query
from probtf.spherical_law.fusion import lift_sensor_posterior
from probtf_observation_fusion_demo.model import load_inputs, run
from probtf_observation_fusion_demo.scene import camera_model, render_image
from probtf_observation_fusion_demo.sensors import ADAPTERS, _marker_mode_law


CONFIG = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(scope="module")
def inputs():
    loaded = {name: load_inputs(CONFIG / "common.yaml", CONFIG / (name + ".yaml"))
              for name in ("no_sensors", "imu", "marker", "imu_marker")}
    image = render_image(loaded["no_sensors"][0])
    return loaded, image


@pytest.fixture(scope="module")
def results(inputs):
    loaded, image = inputs
    return {name: run(common, observations, image=image)
            for name, (common, observations) in loaded.items()}


def test_observation_lists_are_the_only_case_difference(inputs, results):
    loaded, _ = inputs
    common = loaded["no_sensors"][0]
    assert loaded["no_sensors"][1] == ()
    assert loaded["imu_marker"][1] == loaded["imu"][1] + loaded["marker"][1]
    for name, (other, _) in loaded.items():
        assert other == common
        prior = results[name].prior
        np.testing.assert_array_equal(prior.covariance, results["no_sensors"].prior.covariance)
        assert prior.command == tuple(common["update"]["command"])
        assert prior.stamp == common["update"]["stamp"]
    no_information = results["no_sensors"]
    assert no_information.stages[-1].components[0].state is no_information.prior
    assert no_information.records["valid_observation_count"] == 0


def test_imu_has_no_heading_information_and_downstream_relative_law_stays(results):
    result = results["imu"]
    posterior = result.sensor_posteriors[0]
    query = posterior.query
    assert query.distribution.mean.shape == (2,)
    difference = (np.linalg.inv(posterior.components[0].distribution.covariance)
                  - np.linalg.inv(query.distribution.covariance))
    information = query.jacobian.T @ difference @ query.jacobian
    assert np.linalg.matrix_rank(information, tol=1e-8) == 2
    # The direct observation is invariant to world yaw on the first link.
    yaw = np.zeros(result.prior.mean.size)
    yaw[5] = 1.0
    np.testing.assert_allclose(information @ yaw, 0., atol=1e-12)
    updated = result.stages[-1].components[0].state
    i = result.prior.edge_ids.index("link_3")
    np.testing.assert_allclose(updated.transforms[i].bingham,
                               result.prior.transforms[i].bingham, atol=1e-10)
    np.testing.assert_allclose(updated.covariance[6*i:6*i+6, 6*i:6*i+6],
                               result.prior.covariance[6*i:6*i+6, 6*i:6*i+6], atol=1e-14)


def test_lifted_target_matches_each_completed_sensor_posterior(results):
    result = results["imu_marker"]
    for posterior in result.sensor_posteriors:
        lifted = lift_sensor_posterior(posterior)
        query = posterior.query
        assert query.prior_version == result.prior.version
        assert query.command == result.prior.command
        assert query.stamp == result.prior.stamp
        for target, body in zip(posterior.components, lifted):
            np.testing.assert_allclose(query.intercept + query.jacobian @ body.distribution.mean,
                                       target.distribution.mean, atol=1e-11)
            np.testing.assert_allclose(query.jacobian @ body.distribution.covariance @ query.jacobian.T,
                                       target.distribution.covariance, atol=1e-12)
            np.testing.assert_allclose(body.within + body.between, body.distribution.covariance, atol=1e-12)


def test_camera_keeps_actual_image_modes_weights_and_unknown_marker(results):
    camera = results["marker"]
    estimate = camera.local_estimates[0]
    detected = estimate.detector_result
    assert len(detected.weights) >= 2  # This fixed planar image really is ambiguous.
    assert estimate.diagnostics["detector_weights"] == list(detected.weights)
    assert len(estimate.posterior.components) == len(detected.weights)
    assert [component.label for component in estimate.posterior.components] == [
        component.component_id for component in detected.record.distribution.components]
    combined = results["imu_marker"]
    assert len(combined.stages[-1].components) == len(detected.weights)
    assert combined.local_estimates[1].diagnostics == estimate.diagnostics
    for component in combined.stages[-1].components:
        state = component.state
        index = state.edge_ids.index("marker")
        assert np.all(np.linalg.eigvalsh(state.covariance[6*index:6*index+6, 6*index:6*index+6]) > 0.)


def test_camera_adapter_recovers_the_existing_image_hessian_gaussian(inputs, results):
    common = inputs[0]["marker"][0]
    estimate = results["marker"].local_estimates[0]
    covariance_pixels = np.asarray(estimate.diagnostics["corner_covariance"])
    camera = camera_model(common["camera"])
    points = ippe_square_object_points(common["marker"]["size"])
    for component in estimate.detector_result.record.distribution.components:
        rotation = quat_to_rotmat(component.orientation.mode_wxyz)
        recovered = _marker_mode_law(component, rotation)
        jacobian = pose_jacobian(points, rotation, component.translation.mean_at_reference, camera)
        hessian = jacobian.T @ np.linalg.solve(covariance_pixels, jacobian)
        np.testing.assert_allclose(np.linalg.inv(recovered.covariance), hessian,
                                   rtol=2e-9, atol=1e-6)


def test_retained_cross_blocks_affect_the_next_hand_query(results):
    result = results["imu_marker"]
    state = result.stages[-1].components[0].state
    diagonal = np.zeros_like(state.covariance)
    for i in range(len(state.transforms)):
        diagonal[6*i:6*i+6, 6*i:6*i+6] = state.covariance[6*i:6*i+6, 6*i:6*i+6]
    assert np.linalg.norm(state.covariance - diagonal) > 1e-4
    artificial_independence = JointTransformState(state.transforms, diagonal,
                                                 version=state.version, stamp=state.stamp,
                                                 command=state.command, component_id="test_removed_cross")
    original = point_query(state, result.hand_path)
    altered = point_query(artificial_independence, result.hand_path)
    assert np.linalg.norm(original.covariance - altered.covariance) > 1e-5
    assert np.linalg.norm(original.mean - altered.mean) > 1e-7


def test_hidden_physics_does_not_enter_estimation_and_no_rng_is_needed(inputs, results, monkeypatch):
    loaded, image = inputs
    def forbidden(*args, **kwargs):
        raise AssertionError("Randomness is forbidden in propagation, lifting and fusion.")
    for name in ("default_rng", "normal", "random", "randn", "multivariate_normal", "RandomState"):
        monkeypatch.setattr(np.random, name, forbidden)
    for name, (common, observations) in loaded.items():
        hidden_changed = deepcopy(common)
        hidden_changed["physical_scene"] = {"must_not_be_read": True}
        actual = run(hidden_changed, observations, image=image)
        assert actual.records == results[name].records


def test_duplicate_observation_is_not_new_information(inputs, results):
    common, observations = inputs[0]["imu_marker"]
    duplicated = run(common, observations + observations, image=inputs[1])
    assert duplicated.records == results["imu_marker"].records


def test_sensor_adapters_do_not_receive_physical_truth_or_render_settings(inputs, results, monkeypatch):
    def checked(adapter):
        def estimate(state, observation, paths, known, image):
            assert "physical_scene" not in known
            assert "display" not in known
            return adapter(state, observation, paths, known, image)
        return estimate
    for name, adapter in tuple(ADAPTERS.items()):
        monkeypatch.setitem(ADAPTERS, name, checked(adapter))
    common, observations = inputs[0]["imu_marker"]
    assert run(common, observations, image=inputs[1]).records == results["imu_marker"].records


def test_camera_plot_queries_same_core_posterior_and_ros_image_bytes(inputs, results, monkeypatch):
    from sensor_msgs.msg import Image
    from probtf_observation_fusion_demo import sensors
    from probtf_observation_fusion_demo.node import image_message
    captured = []
    original_plot = sensors._camera_xy_inset
    def capture_plot(image, queries):
        captured.extend(queries)
        return original_plot(image, queries)
    monkeypatch.setattr(sensors, "_camera_xy_inset", capture_plot)
    common, observations = inputs[0]["marker"]
    actual = run(common, observations, image=inputs[1])
    estimate = actual.local_estimates[0]
    assert len(captured) == len(estimate.posterior.components)
    for query, component in zip(captured, estimate.posterior.components):
        assert query.source is component
        assert np.shares_memory(query.mean, component.distribution.mean)
        assert np.shares_memory(query.covariance, component.distribution.covariance)
        np.testing.assert_array_equal(query.mean, component.distribution.mean[:2])
        np.testing.assert_array_equal(query.covariance, component.distribution.covariance[:2, :2])
        assert not query.mean.flags.writeable and not query.covariance.flags.writeable
    bounds = estimate.diagnostics["image_xy_query"]
    np.testing.assert_array_equal(bounds["plot_bounds_low"], np.min([
        query.mean - 2*np.sqrt(np.diag(query.covariance)) for query in captured], axis=0))
    np.testing.assert_array_equal(bounds["plot_bounds_high"], np.max([
        query.mean + 2*np.sqrt(np.diag(query.covariance)) for query in captured], axis=0))
    # Rendering reaches the actual fixed-size ROS image without changing inference.
    assert actual.records == results["marker"].records
    assert actual.detection_image.shape == (480, 640, 3)
    message = image_message(actual.detection_image, actual.prior.stamp, "camera_image")
    stream = io.BytesIO()
    message.serialize(stream)
    wire = Image().deserialize(stream.getvalue())
    np.testing.assert_array_equal(np.frombuffer(wire.data, dtype=np.uint8).reshape(480, 640, 3),
                                  actual.detection_image)
    assert wire.width == 640 and wire.height == 480 and wire.encoding == "bgr8"
