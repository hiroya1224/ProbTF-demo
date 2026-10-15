"""Full primitive-path moments and estimator/display law identity."""
from dataclasses import replace
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from probtf.bingham import bingham_second_moment, quaternion_product_second_moment
from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, OrientationKind, TransformComponent,
    TransformDistribution, TransformDistributionStamped,
)
from probtf.geometry import (
    axis_angle_to_quat, quat_right_matrix, quat_to_rotmat, rotation_action_matrix,
    rotation_vector_from_quaternion, rpy_to_quat,
)
from probtf.probability.moments import (
    PointMomentSummary, forward_component_point_moments, mixture_point_moments,
    rotation_vector_moments,
)
from probtf.provenance import TransformProvenance
from probtf_switch_press_demo.model import (
    BODY_FRAMES, COMPARISON_METHOD, Model, TransformPath, _Prepared, _fixed,
    _matched_orientation, summarize,
)
from probtf_switch_press_demo.scene import apply, inverse, transform, write_robot


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ros/examples/probtf_switch_press_demo"
Q = np.array([-.15, -1.85, 2.35, 0., -1.25, 0.])


@pytest.fixture
def model(tmp_path):
    scene = yaml.safe_load((PACKAGE / "config/default.yaml").read_text())
    urdf = write_robot(ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
                       tmp_path / "rod.urdf", scene)
    return Model.from_config(urdf, rod_length=scene["rod"]["length"],
        hand_camera=transform(scene["camera"]["translation"], scene["camera"]["rpy"]))


def _coupled_component(name="coupled", weight=1., mode=None, offset=None):
    if mode is None:
        mode = rpy_to_quat(.2, -.15, .35)
    # Anisotropic orientation with a general quaternion eigensystem.
    right = quat_right_matrix(mode)
    orientation = BinghamOrientation.from_parameter_matrix(
        right @ np.diag([0., -37., -61., -94.]) @ right.T, mode)
    coupling = np.array([[.05, -.01, .03, .02, .06, -.01, -.02, .01, .04],
                         [-.03, .02, .01, .04, -.02, .05, .01, -.04, .02],
                         [.02, .01, -.04, -.01, .03, .02, .06, .01, -.03]])
    return TransformComponent(name, weight, orientation,
        ConditionalGaussianTranslation([.02, -.03, .6] if offset is None else offset,
            np.diag([.002, .003, .004]) ** 2, coupling))


def _reference_moments(components, point=(0., 0., 0.)):
    result = PointMomentSummary(point, np.zeros((3, 3)))
    for component in reversed(tuple(components)):
        result = forward_component_point_moments(component, result, integration_steps=480)
    return result


def _marker_record():
    components = (_coupled_component("left", .3),
                  _coupled_component("right", .7, rpy_to_quat(-.2, .25, -.3), [.04, .01, .64]))
    return TransformDistributionStamped("camera_optical_frame", "switch_marker", 12.345,
        "image_marker", "detector", TransformDistribution(components),
        provenance=TransformProvenance(source_ids=("actual_camera_image_17",), method="native_detector"))


def test_cached_moment_transport_agrees_with_native_full_path(model):
    prepared = _Prepared.from_component(_coupled_component(), 480)
    moved = prepared.right_rotation(axis_angle_to_quat([.3, .7, -.2], -.41))
    input_point = PointMomentSummary([.2, -.04, .09], np.diag([.01, .02, .03]) ** 2)
    native = forward_component_point_moments(moved.component, input_point, 480)
    cached = moved.forward(input_point)
    np.testing.assert_allclose(cached.mean, native.mean, atol=1e-12)
    np.testing.assert_allclose(cached.covariance, native.covariance, atol=1e-12)
    np.testing.assert_allclose(moved.quaternion_moment,
        bingham_second_moment(moved.component.orientation.backend_parameter_matrix(), 480), atol=1e-12)

    path = model.chain(Q)
    direct = _reference_moments(path)
    hand = summarize(path)
    np.testing.assert_allclose(hand.mean, direct.mean, atol=1e-12)
    np.testing.assert_allclose(hand.covariance, direct.covariance, atol=1e-12)
    moment = np.diag([1., 0., 0., 0.])
    for item in path.prepared:
        moment = quaternion_product_second_moment(moment, item.quaternion_moment)
    np.testing.assert_allclose(hand.orientation_moment, moment, atol=1e-13)
    np.testing.assert_allclose(bingham_second_moment(hand.orientation_parameter, 480), moment, atol=1e-8)


def test_long_attachment_retains_both_position_rotation_cross_terms():
    component = _coupled_component()
    prepared = _Prepared.from_component(component, 480)
    rod = _fixed([.65, 0., 0.], rpy_to_quat(.1, .2, -.1), "rod", 480)
    path = TransformPath((prepared, rod), dict(frame_id="root", source_id="native_test", stamp=3.))
    hand = summarize(path)
    moments = rotation_vector_moments(component.orientation, 480)
    gamma = moments.second_vector - np.outer(moments.mean_vector, moments.mean_vector)
    action = rotation_action_matrix([.65, 0., 0.])
    coupling = component.translation.rotation_coupling
    expected_covariance = ((coupling + action) @ gamma @ (coupling + action).T
                           + component.translation.residual_covariance)
    expected_mean = (component.translation.mean_at_reference
        - coupling @ rotation_vector_from_quaternion(component.orientation.reference_quaternion_wxyz)
        + (coupling + action) @ moments.mean_vector)
    np.testing.assert_allclose(hand.mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(hand.covariance, expected_covariance, atol=1e-12)
    naive = (coupling @ gamma @ coupling.T + action @ gamma @ action.T
             + component.translation.residual_covariance)
    assert np.linalg.norm(expected_covariance - naive) > .0001


def test_goal_keeps_coupling_through_camera_marker_attachment_then_common_gaussian(model):
    marker = _marker_record()
    goal_offset = np.array([.02, -.24, .045])
    goal_quaternion = rpy_to_quat(.3, -.5, .8)
    target = model.target(Q, marker, goal_offset, goal_quaternion)
    camera = model.chain(Q, "camera", marker.stamp)
    attachment = _fixed(goal_offset, goal_quaternion, "goal", 480)
    expected = []
    for branch in marker.distribution.normalized_components():
        components = tuple(camera) + (branch.component, attachment.component)
        moments = _reference_moments(components)
        expected.append((branch.weight, moments))
    common = mixture_point_moments(expected)
    np.testing.assert_allclose(target.mean, common.mean, atol=1e-12)
    np.testing.assert_allclose(target.covariance, common.covariance, atol=1e-12)
    np.testing.assert_array_equal(target.weights, [.3, .7])
    within = sum(weight * point.covariance for weight, point in expected)
    assert np.linalg.norm(target.covariance - within) > .0001
    for index, (_weight, point) in enumerate(expected):
        np.testing.assert_allclose(target.branch_summaries[index].mean, point.mean, atol=1e-12)
        np.testing.assert_allclose(target.branch_summaries[index].covariance, point.covariance, atol=1e-12)
    other_orientation = model.target(Q, marker, goal_offset, [1., 0., 0., 0.])
    np.testing.assert_allclose(other_orientation.mean, target.mean, atol=1e-12)
    np.testing.assert_allclose(other_orientation.covariance, target.covariance, atol=1e-12)
    assert np.linalg.norm(other_orientation.branch_summaries[0].orientation_moment
                          - target.branch_summaries[0].orientation_moment) > .1
    assert target.metadata["observation_command"] == tuple(Q)
    assert target.metadata["observation_stamp"] == marker.stamp
    assert target.metadata["observation_source_ids"] == marker.provenance.source_ids


def test_nominal_geometry_uses_same_configured_family_not_physical_state(model):
    nominal = model.nominal_roots(Q)
    rigid = model.knowngeometry.transforms(Q, .185)
    for frame in BODY_FRAMES:
        np.testing.assert_allclose(nominal[frame], rigid[frame], atol=1e-14)
    np.testing.assert_allclose(nominal["camera_optical_frame"], rigid["camera_optical_frame"], atol=1e-14)
    first = model.hand(Q)
    second = model.hand(Q + [.02, -.01, .03, 0, .01, 0])
    assert np.linalg.norm(first.mean - second.mean) > .005
    assert np.linalg.norm(first.covariance - second.covariance) > 1e-5
    # The model receives no plant object or stiffness values.
    assert "stiffness" not in json.dumps(model.descriptor)
    assert first.metadata["source_fingerprint"] == second.metadata["source_fingerprint"]
    assert first.metadata["revision"] == second.metadata["revision"]


def _assert_native_comparison_matches(summary, record):
    orientations = summary.orientations if hasattr(summary, "orientations") else (summary.orientation,)
    weights = summary.weights if hasattr(summary, "weights") else (1.,)
    assert len(record.distribution.components) == len(weights)
    for component, orientation, weight in zip(record.distribution.components, orientations, weights):
        assert component.orientation is orientation
        assert component.raw_weight == weight
        np.testing.assert_array_equal(component.translation.mean_at_reference, summary.mean)
        np.testing.assert_array_equal(component.translation.residual_covariance, summary.covariance)
        np.testing.assert_array_equal(component.translation.rotation_coupling, np.zeros((3, 9)))
        detail = json.loads(component.provenance.detail)
        for key in ("frame_id", "source_id", "revision", "source_fingerprint", "command", "stamp"):
            expected = summary.metadata[key]
            assert detail[key] == (list(expected) if isinstance(expected, tuple) else expected)
    assert record.stamp == summary.metadata["stamp"]
    assert record.parent_frame_id == summary.metadata["frame_id"]


def test_published_primitives_hand_and_goal_keep_their_actual_laws(model):
    hand = model.hand(Q, stamp=12.345)
    records = model.display_records(Q, stamp=12.345, hand_summary=hand)
    assert tuple(record.child_frame_id for record in records) == BODY_FRAMES
    path = model.chain(Q, stamp=12.345)
    for record, (parent, child), prepared in zip(records[:-1], model.edges[:-1], path.prepared):
        assert (record.parent_frame_id, record.child_frame_id) == (parent, child)
        assert record.stamp == hand.metadata["stamp"]
        assert record.provenance.method == "configured_native_local_transform_family"
        # The display consumes the actual command-transported primitive. It
        # cannot replace a prefix by a separate independent X/R product law.
        assert record.distribution.components == (prepared.component,)
        assert record.distribution.components[0] is prepared.component
        detail = json.loads(record.provenance.detail)
        assert detail["command"] == list(Q)
        assert detail["primitive_dependence"] == "independent"
        assert detail["source_fingerprint"] == hand.metadata["source_fingerprint"]
        assert detail["revision"] == hand.metadata["revision"]
    assert records[-1].provenance.method == COMPARISON_METHOD
    _assert_native_comparison_matches(hand, records[-1])
    target = model.target(Q, _marker_record(), [.0, -.24, .045], rpy_to_quat(0, .5, 0))
    target_record = target.to_record("switch_goal", stamp=99.)
    _assert_native_comparison_matches(target, target_record)
    assert target_record.stamp == 12.345
    assert json.loads(target_record.provenance.detail)["publication_stamp"] == 99.
    assert target_record.provenance.method == "indet_ik_common_gaussian_bingham_mixture"
    # Rendering cannot mutate the frozen estimation snapshot.
    with pytest.raises(ValueError):
        hand.covariance[0, 0] *= 2
    with pytest.raises(TypeError):
        hand.metadata["revision"] = "other"
    with pytest.raises(ValueError, match="command"):
        model.display_records(Q + [.01, 0, 0, 0, 0, 0], stamp=12.345, hand_summary=hand)
    with pytest.raises(ValueError, match="stamp|time"):
        model.display_records(Q, stamp=99., hand_summary=hand)
    wrong_source = replace(hand, metadata=dict(hand.metadata, source_fingerprint="different_law"))
    with pytest.raises(ValueError, match="source"):
        model.display_records(Q, stamp=12.345, hand_summary=wrong_source)


def test_intermediate_display_keeps_the_cross_covariance_of_the_native_prefix(model):
    """A display-only X/R factorization loses a large, real upstream term."""
    records = model.display_records(Q, stamp=1.)
    native_prefix = tuple(record.distribution.components[0] for record in records[:-1])
    internal_prefix = tuple(model.chain(Q, "tool0", stamp=1.))
    assert all(shown is used for shown, used in zip(native_prefix, internal_prefix))
    axis_endpoint = np.array([.1, 0., 0.])
    endpoint = _reference_moments(native_prefix, axis_endpoint)
    exact = _reference_moments(internal_prefix, axis_endpoint)
    np.testing.assert_array_equal(endpoint.mean, exact.mean)
    np.testing.assert_array_equal(endpoint.covariance, exact.covariance)

    # Compute the deliberately incorrect factorization without a Bingham fit:
    # keep every source R, but remove source translations to get Cov(R v).
    orientation_only = tuple(replace(component, translation=ConditionalGaussianTranslation(
        np.zeros(3), np.zeros((3, 3)), np.zeros((3, 9)))) for component in internal_prefix)
    origin = _reference_moments(internal_prefix)
    rotated_axis = _reference_moments(orientation_only, axis_endpoint)
    wrongly_independent = origin.covariance + rotated_axis.covariance
    missing_cross = exact.covariance - wrongly_independent
    assert np.linalg.norm(missing_cross) / np.linalg.norm(exact.covariance) > .25
    assert np.trace(missing_cross) > .001


def test_independence_must_be_explicit_and_inputs_remain_direct(model):
    source = copy.deepcopy(model.descriptor)
    del source["independent"]
    with pytest.raises(ValueError, match="explicitly declared"):
        Model("unused.urdf", source, .325, np.eye(4))
    marker = _marker_record()
    wrong_frame = replace(marker, parent_frame_id="world")
    with pytest.raises(ValueError, match="optical camera"):
        model.target(Q, wrong_frame, [0, 0, 0])


def test_exact_dirac_and_uniform_require_structural_source_not_a_threshold():
    fixed = _fixed([.2, .1, -.04], rpy_to_quat(.1, .2, .3), "fixed", 480)
    summary = summarize(TransformPath((fixed,), {}))
    assert summary.orientation.kind is OrientationKind.DIRAC
    assert summary.orientation_parameter is None
    uniform = replace(fixed.component, orientation=BinghamOrientation.uniform())
    summary = summarize(TransformPath((_Prepared.from_component(uniform, 480), fixed), {}))
    assert summary.orientation.kind is OrientationKind.UNIFORM
    np.testing.assert_array_equal(summary.orientation_parameter, np.zeros((4, 4)))
    with pytest.raises(RuntimeError, match="not numerically resolved"):
        _matched_orientation(np.diag([1., 0., 0., 0.]), 480)


def test_isotropic_source_projection_cache_is_equivariant_and_anisotropy_falls_back(model):
    for q in (Q, Q + [.1, -.08, .03, .05, .07, -.1], np.zeros(6)):
        path = model.chain(q)
        cached = summarize(path)
        # Removing only the structural symmetry certificate forces the existing
        # general Bingham fit against the exact same propagated moment.
        general = summarize(TransformPath(tuple(replace(item, isotropic_factor=None)
            for item in path.prepared), path.metadata, path.integration_steps))
        np.testing.assert_array_equal(cached.mean, general.mean)
        np.testing.assert_array_equal(cached.covariance, general.covariance)
        np.testing.assert_array_equal(cached.orientation_moment, general.orientation_moment)
        np.testing.assert_allclose(bingham_second_moment(cached.orientation_parameter, 480),
                                   general.orientation_moment, atol=1e-8)
        np.testing.assert_allclose(bingham_second_moment(general.orientation_parameter, 480),
                                   cached.orientation_moment, atol=1e-8)
    coupled = _Prepared.from_component(_coupled_component(), 480)
    assert coupled.isotropic_factor is None
    anisotropic = summarize(TransformPath((coupled,), {}))
    np.testing.assert_allclose(bingham_second_moment(anisotropic.orientation_parameter, 480),
                               coupled.quaternion_moment, atol=1e-7)


def test_free_anisotropic_coupled_local_input_and_cache_source_context(model):
    source = model.descriptor
    source["reference_command"] = [.05, -.1, .3, .02, -.2, .1]
    source["revision"] = 17
    edge = source["edges"][2]
    edge["orientation"] = dict(kind="bingham", mode_wxyz=rpy_to_quat(.2, -.3, .4).tolist(),
                                eigenvalues=[0., -23., -51., -92.])
    edge["translation"]["mean"] = [.33, .015, -.008]
    coupling = _coupled_component().translation.rotation_coupling
    edge["translation"]["rotation_coupling"] = coupling.tolist()
    lower = np.array([[.002, 0, 0], [.0005, .003, 0], [-.0002, .0007, .004]])
    edge["translation"]["covariance"] = (lower @ lower.T).tolist()
    configured = Model.from_config(ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf", source)
    path = configured.chain(Q, stamp=4.)
    assert path.prepared[2].isotropic_factor is None
    component = tuple(path)[2]
    at_reference = tuple(configured.chain(configured.reference_command))[2]
    delta = axis_angle_to_quat([0, 1, 0], Q[2] - configured.reference_command[2])
    right = quat_right_matrix(delta)
    np.testing.assert_allclose(component.orientation.parameter_matrix(),
        right @ at_reference.orientation.parameter_matrix() @ right.T, atol=1e-12)
    np.testing.assert_allclose(component.translation.rotation_coupling,
        coupling @ np.kron(quat_to_rotmat(delta), np.eye(3)), atol=1e-14)
    projected = summarize(path)
    full = _reference_moments(path)
    np.testing.assert_allclose(projected.mean, full.mean, atol=1e-12)
    np.testing.assert_allclose(projected.covariance, full.covariance, atol=1e-12)
    np.testing.assert_allclose(bingham_second_moment(projected.orientation_parameter, 480),
                               projected.orientation_moment, atol=1e-6)
    assert projected.metadata["moment_match_max_residual"] < 1e-6
    first, second = configured.chain(Q, stamp=4.), configured.chain(Q, stamp=7.)
    assert all(a is b for a, b in zip(first.prepared, second.prepared))
    assert first.metadata["stamp"] == 4.
    assert second.metadata["stamp"] == 7.
    assert first.metadata["revision"] == "17"
    with pytest.raises(ValueError):
        first.prepared[2].quaternion_moment[0, 0] = .9
    # Mutating an exported description does not alter law/version/cache state.
    description = configured.descriptor
    description["edges"][2]["translation"]["mean"][0] = 99.
    assert configured.descriptor["edges"][2]["translation"]["mean"][0] == .33
    _assert_native_comparison_matches(projected,
        configured.display_records(Q, 4., hand_summary=projected)[-1])


def test_known_camera_mount_keeps_marker_visible_at_desired_tool_pose():
    """Task attachments and requested pose must admit observing the whole tag.

    This geometric compatibility test uses no robot pose, stiffness, simulated
    truth or estimated marker law. It checks the configured final task pose.
    """
    config = yaml.safe_load((PACKAGE / "config/default.yaml").read_text())
    camera, marker = config["camera"], config["marker"]
    marker_tip = np.eye(4)
    marker_tip[:3, :3] = quat_to_rotmat(marker["goal_orientation_wxyz"])
    marker_tip[:3, 3] = marker["goal_offset"]
    np.testing.assert_allclose(marker_tip[:3, 0], [0, 0, -1], atol=1e-14)
    tool_tip = transform([config["rod"]["length"], 0, 0])
    marker_tool = marker_tip @ inverse(tool_tip)
    half = marker["size"] / 2
    corners = np.array([[-half, half, 0], [half, half, 0],
                        [half, -half, 0], [-half, -half, 0]])
    image_size = np.array([camera["width"], camera["height"]])

    def projected_corners(mount_rpy):
        marker_camera = marker_tool @ transform(camera["translation"], mount_rpy)
        points = apply(inverse(marker_camera), corners)
        assert np.all(points[:, 2] > .02)
        return camera["focal_length"] * points[:, :2] / points[:, 2, None] + image_size / 2

    pixels = projected_corners(camera["rpy"])
    assert np.all(pixels / image_size >= .15)
    assert np.all(pixels / image_size <= .85)
    # The former straight optical mount made the final task/FOV constraints
    # incompatible: both upper tag corners were 72 pixels outside the image.
    old_pixels = projected_corners([-np.pi / 2, 0, -np.pi / 2])
    np.testing.assert_allclose(old_pixels[:2, 1], [-72., -72.], atol=1e-10)
    assert np.any(old_pixels[:, 1] < 0)
