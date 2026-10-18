"""Real board pixels become one central law without losing raw mixtures."""

from copy import deepcopy
from dataclasses import replace
import io
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from visualization_msgs.msg import MarkerArray

from probtf.geometry import quat_to_rotmat
from probtf.spherical_law.first_order import (
    MAX_FIRST_ORDER_ROTATION_VARIANCE, bingham_rotation_coordinates,
)
from probtf_observation_fusion_demo.body_native import point_query
from probtf_observation_fusion_demo import model, nonlinear_observation, sensors
from probtf_observation_fusion_demo.display import make_markers, verify_wire_identity
from probtf_observation_fusion_demo.marker_search import marker_observed
from probtf_observation_fusion_demo.marker_board import fuse_marker_board
from probtf_observation_fusion_demo.fresh_goal import fresh_marker_goals
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.node import build_snapshot
from test_marker_board import SETTINGS as ANALYTIC_BOARD_SETTINGS, detection


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def board_capture(tmp_path_factory):
    # Truth remains on the simulator side. The inference below receives only
    # pixels, sent command metadata and the existing known URDF attachments.
    scene = MovingSwitchScene(tmp_path_factory.mktemp('real_board_pipeline'))
    frame = scene.step(0., 1000.)
    # The enlarged board makes all nine tags detectable in this frame. Cover
    # only its central printed tag, as an explicit partial-occlusion fixture;
    # every other rendered marker pixel remains in the input. The estimator
    # receives the occluded image, never this masking geometry or a truth pose.
    settings = yaml.safe_load((PACKAGE/'config/moving_estimation.yaml').read_text())['marker']
    detector = sensors.ArucoCornerDetector(family=settings['family'],
        corner_sigma_px=settings['corner_sigma_px'], adaptive_covariance=False)
    observations = detector.detect(frame.image)
    image = frame.image.copy()
    for observation in observations:
        if observation.marker_id == 21:
            center = np.mean(observation.corners_px, axis=0)
            polygon = np.rint(center+1.08*(observation.corners_px-center)).astype(np.int32)
            cv2.fillConvexPoly(image, polygon, 255)
    remaining = {item.marker_id for item in detector.detect(image)}
    assert remaining == {item.marker_id for item in observations}-{21}
    assert remaining
    image.setflags(write=False)
    templates = yaml.safe_load((PACKAGE/'config/marker.yaml').read_text())['observations']
    return scene, replace(scene.sensor_inputs(frame), image=image), templates


@pytest.fixture(scope='module')
def board_pipeline(board_capture):
    scene, inputs, templates = board_capture
    patch = pytest.MonkeyPatch()
    prepared_calls, affine_calls = [], []
    prepare = sensors.prepare_marker_observation

    def prepare_once(common, image, stamp):
        prepared = prepare(common, image, stamp)
        prepared_calls.append(prepared)
        return prepared

    def forbidden_affine(*_args, **_kwargs):
        affine_calls.append(True)
        raise AssertionError('The nonlinear board update must not re-lift an affine sensor posterior.')

    def forbidden(*_args, **_kwargs):
        raise AssertionError('The board observation/goal pipeline must not use Monte Carlo.')

    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        patch.setattr(np.random, name, forbidden)
    patch.setattr(nonlinear_observation, 'prepare_marker_observation', prepare_once)
    patch.setitem(model.PREPARERS, 'marker', forbidden_affine)
    patch.setitem(model.ADAPTERS, 'marker', forbidden_affine)
    patch.setattr(model, 'lift_sensor_posterior', forbidden_affine)
    try:
        inference = MovingInference(scene.robot_description, templates)
        assert inference.observation_update_method == 'nonlinear_fixed_prior'
        result = inference.update(inputs, 0)
        yield SimpleNamespace(scene=scene, inputs=inputs, inference=inference, result=result,
            prepared=tuple(prepared_calls), affine_calls=tuple(affine_calls))
    finally:
        patch.undo()


def test_real_partial_board_uses_visible_noncentral_ids_and_preserves_every_raw_native_mode(board_pipeline):
    pipeline = board_pipeline
    assert len(pipeline.prepared) == 1
    prepared = pipeline.prepared[0]
    assert prepared.image.shape == (720, 960)
    np.testing.assert_array_equal(prepared.image, pipeline.inputs.image)
    observed = {item.marker_id for item in prepared.observations}
    assert observed and 21 not in observed
    assert observed.issubset(set(range(17, 26)))
    assert prepared.detected is not None
    assert len(prepared.detected.record.distribution.components) == 1
    assert prepared.detected.weights == (1.,)
    assert prepared.detected.log_masses == (0.,)

    board = prepared.board_fusion
    native = dict(prepared.individual_results)
    assert set(native) == set(board['usable_marker_ids'])
    assert set(native).issubset(observed)
    assert board['fusion_order'] == sorted(native)
    assert board['shared_body_prior_used'] is False
    assert board['body_camera_uncertainty_used'] is False
    assert board['body_camera_uncertainty_lift_count_in_this_helper'] == 0
    assert board['downstream_body_camera_lift_required_once'] is True
    assert board['central_component_count'] == 1
    raw = {item['marker_id']: item for item in board['raw_observations']}
    assert set(raw) == set(native)
    assert sum(item['native_component_count'] for item in raw.values()) > len(raw)
    for identity, detected in native.items():
        record = raw[identity]
        original = detected.record.distribution.components
        assert record['stamp'] == pipeline.inputs.stamp
        assert record['native_component_count'] == len(original)
        assert len(record['components']) == len(original)
        for saved, component, weight, log_mass in zip(record['components'], original,
                detected.weights, detected.log_masses):
            assert saved['native_component_id'] == component.component_id
            assert saved['weight'] == weight and saved['native_log_mass'] == log_mass
            np.testing.assert_array_equal(saved['native_bingham'], component.orientation.parameter_matrix())
            np.testing.assert_array_equal(saved['native_translation_mean'],
                component.translation.mean_at_reference)
            np.testing.assert_array_equal(saved['native_translation_residual_covariance'],
                component.translation.residual_covariance)
            np.testing.assert_array_equal(saved['native_translation_rotation_coupling'],
                component.translation.rotation_coupling)
    assert board['monte_carlo_used'] is False and board['physical_truth_received'] is False


def test_keep_reset_share_one_prepared_central_observation_and_one_image_update(board_pipeline):
    pipeline = board_pipeline
    result, prepared = pipeline.result, pipeline.prepared[0]
    assert len(pipeline.prepared) == 1
    assert pipeline.affine_calls == ()
    assert result.records['local_gaussian_lift_applied'] is False
    assert result.records['frozen_local_posteriors'] == []
    assert result.records['lift_total_covariance'] == []
    assert len(result.frozen_body_branches) == 2
    estimates = [item for item in result.local_estimates if item.posterior is not None]
    assert len(estimates) == 2
    for estimate in estimates:
        assert estimate.detector_result is prepared.detected
        assert estimate.board_fusion is prepared.board_fusion
        assert estimate.diagnostics['marker_board_fusion'] is prepared.board_fusion
        assert len(estimate.posterior.components) == 1
        assert len(estimate.posterior.component_queries) == 1
        assert [query.state_id for query in estimate.posterior.component_queries] == (
            estimate.diagnostics['joint_posterior_source_state_ids'])
    assert len({item.posterior.observation_id for item in estimates}) == 1
    surprise = result.records['bayes_factor_surprise']
    assert surprise['applied'] is True
    assert len(surprise['observation_ids']) == 1
    assert len(result.stages[-1].components) == 1
    assert len(result.stages[-1].goal_components) == 1


def test_fresh_board_goal_uses_the_same_central_source_and_known_switch_offset_on_wire(board_pipeline):
    pipeline = board_pipeline
    result, prepared = pipeline.result, pipeline.prepared[0]
    stage = result.stages[-1]
    record = result.records['fresh_goal_construction']
    board = prepared.board_fusion
    central = prepared.detected.record.distribution.components[0]
    assert record['observation_kind'] == 'fused_board_center'
    assert record['marker_board_fusion'] is board
    assert record['central_component_count'] == record['goal_component_count'] == 1
    assert record['raw_marker_count'] == len(prepared.individual_results)
    assert record['raw_native_component_count'] == sum(
        item['native_component_count'] for item in board['raw_observations'])
    assert record['raw_native_component_count'] > record['central_component_count']
    assert record['components'][0]['native_component_id'] == central.component_id
    assert 'derived central input' in record['component_record_semantics']
    assert record['capture_stamp'] == pipeline.inputs.stamp
    assert record['capture_command'] == list(pipeline.inputs.command)
    assert record['canonical_internal_state_id'] == stage.components[0].state.state_id
    assert record['old_internal_marker_used_for_goal_construction'] is False
    assert record['image_already_used_in_body_update'] is True
    assert record['observation_fused_again'] is False

    goal = stage.goal_components[0]
    assert goal.weight == 1. and goal.log_weight == 0.
    assert record['goal_source_state_ids'] == [goal.state.state_id]
    offset = np.asarray(pipeline.inference.common['control']['goal_offset'])
    assert np.linalg.norm(offset) > .2
    expected = point_query(goal.state, result.marker_path, offset)
    marker = goal.state.transforms[goal.state.edge_ids.index('marker')]
    np.testing.assert_allclose(expected.mode,
        marker.translation+quat_to_rotmat(marker.rotation_wxyz) @ offset, atol=1e-14)
    snapshot = build_snapshot(result, stage, pipeline.inference.common)
    point = next(item for item in snapshot.points
                 if item.frame_label == 'switch_goal' and item.axis == 'origin')
    assert point.query.state_id == goal.state.state_id
    np.testing.assert_array_equal(point.query.mean, expected.mean)
    np.testing.assert_array_equal(point.query.covariance, expected.covariance)
    assert snapshot.goal_memory['image_relative_source'] == record
    display = pipeline.inference.settings['display']
    markers = make_markers(snapshot, display['seed'], display['point_size'])
    stream = io.BytesIO()
    markers.serialize(stream)
    wire = MarkerArray().deserialize(stream.getvalue())
    verify_wire_identity(snapshot, wire, seed=display['seed'], point_size=display['point_size'])
    assert {item.component_id for item in snapshot.points if item.frame_label == 'switch_goal'} == {goal.label}


def test_all_board_images_missing_preserves_the_root_goal_and_uses_existing_loss_policy(board_pipeline):
    pipeline = board_pipeline
    first = pipeline.result
    before = first.stages[-1].goal_components[0]
    memory = deepcopy(first.records['fixed_goal_memory'])
    command = pipeline.inputs.command+np.array([.001, -.002, .001, 0., .001, -.001])
    missing = replace(pipeline.inputs, image=np.full_like(pipeline.inputs.image, 255),
        command=command, stamp=pipeline.inputs.stamp+.05, time=.05)
    current = pipeline.inference.update(missing, 1)
    assert marker_observed(current) is False
    assert current.records['fixed_goal_memory_event'] == 'image_missing_hold'
    assert current.records['fixed_goal_memory'] == memory
    assert current.records['bayes_factor_surprise']['applied'] is False
    assert 'fresh_goal_construction' not in current.records
    assert current.records['internal_projection']['goal_construction'].startswith('retained root-fixed task goal')
    assert len(current.stages[-1].components) == len(current.stages[-1].goal_components) == 1
    after = current.stages[-1].goal_components[0]
    assert (after.weight, after.log_weight) == (before.weight, before.log_weight)
    old_index = before.state.edge_ids.index('marker')
    new_index = after.state.edge_ids.index('marker')
    assert after.state.transforms[new_index] is before.state.transforms[old_index]
    np.testing.assert_array_equal(after.state.covariance[6*new_index:6*new_index+6,
        6*new_index:6*new_index+6], before.state.covariance[6*old_index:6*old_index+6,
        6*old_index:6*old_index+6])
    assert np.linalg.norm(after.state.covariance[:6*new_index, 6*new_index:6*new_index+6]) > 0.
    offset = pipeline.inference.common['control']['goal_offset']
    old_query = point_query(before.state, first.marker_path, offset)
    new_query = point_query(after.state, current.marker_path, offset)
    np.testing.assert_array_equal(new_query.mean, old_query.mean)
    # The stored marker law and P_EE above are bit-identical. Re-querying a
    # larger joint support can differ by floating-point eigensolver roundoff.
    np.testing.assert_allclose(new_query.covariance, old_query.covariance,
        atol=1e-18, rtol=1e-13)


def _fresh_domain_handoff(pipeline, fusion):
    """Exercise the derived-observation/task boundary on a frozen body.

    These analytic native-mode fixtures are deliberately unresolved pose
    likelihoods. They are a concentration-domain contract test, separate
    from the real-pixel body update and its accuracy above.
    """
    original = pipeline.result
    estimates = tuple(replace(item,
        posterior=replace(item.posterior, observation_id='analytic_board_domain_handoff'),
        detector_result=fusion.detected, board_fusion=fusion.record)
        for item in original.local_estimates if item.posterior is not None)
    source = replace(original, local_estimates=estimates)
    canonical = original.stages[-1].components[0].state
    goals, record = fresh_marker_goals(source, canonical)
    stage = replace(original.stages[-1], goal_components=goals)
    # No fixed target was captured by this isolated analytic handoff. Avoid
    # attaching the physical fixture's old target-memory record to these laws.
    display_result = replace(source, stages=(stage,), records={})
    snapshot = build_snapshot(display_result, stage, pipeline.inference.common)
    settings = pipeline.inference.settings['display']
    stream = io.BytesIO()
    make_markers(snapshot, settings['seed'], settings['point_size']).serialize(stream)
    wire = MarkerArray().deserialize(stream.getvalue())
    verify_wire_identity(snapshot, wire, seed=settings['seed'], point_size=settings['point_size'])
    for goal in goals:
        points = [item for item in snapshot.points
                  if item.frame_label == 'switch_goal' and item.component_id == goal.label]
        assert len(points) == 4
        assert all(item.query.state_id == goal.state.state_id
                   and item.weight == goal.weight and item.log_weight == goal.log_weight
                   for item in points)
    return goals, record, snapshot


def test_broad_central_single_fit_preserves_every_narrow_mode_through_fresh_goal_and_wire(board_pipeline):
    native = detection(21, angles=(-.4, .4), log_masses=np.log([.7, .3]),
        coupling=.02*np.eye(3), stamp=board_pipeline.inputs.stamp)
    fusion = fuse_marker_board({21: native}, ANALYTIC_BOARD_SETTINGS)
    assert MAX_FIRST_ORDER_ROTATION_VARIANCE == .1
    projection = fusion.record['steps'][0]['projection']
    assert projection['deferred'] and not projection['applied']
    assert projection['quaternion_second_moment_fit_max_error'] < 1e-6
    with pytest.raises(ValueError, match='concentration domain'):
        bingham_rotation_coordinates(np.asarray(projection['bingham_parameter']))
    assert fusion.record['final_mixture_fallback_used'] is True
    assert fusion.record['central_distribution']['component_count'] == 2
    for component in fusion.detected.record.distribution.components:
        _, covariance = bingham_rotation_coordinates(component.orientation.parameter_matrix())
        assert np.max(np.linalg.eigvalsh(covariance)) <= MAX_FIRST_ORDER_ROTATION_VARIANCE
    goals, record, snapshot = _fresh_domain_handoff(board_pipeline, fusion)
    assert len(goals) == record['central_component_count'] == record['goal_component_count'] == 2
    assert record['marker_board_fusion'] is fusion.record
    assert record['raw_marker_count'] == 1 and record['raw_native_component_count'] == 2
    np.testing.assert_array_equal([item.weight for item in goals], fusion.detected.weights)
    np.testing.assert_allclose([item.log_weight for item in goals], fusion.detected.log_masses,
        atol=1e-14, rtol=0.)
    assert len({item.query.state_id for item in snapshot.points if item.frame_label == 'switch_goal'}) == 2
    assert record['component_pruning_used'] is False


def test_intermediate_broad_fit_defers_until_next_marker_before_fresh_single_goal(board_pipeline):
    stamp = board_pipeline.inputs.stamp
    native = {
        17: detection(17, angles=(-.4, .4), log_masses=np.log([.7, .3]),
            coupling=.02*np.eye(3), stamp=stamp),
        18: detection(18, angles=(.4, .42), log_masses=np.log([.6, .4]), stamp=stamp),
    }
    fusion = fuse_marker_board(native, ANALYTIC_BOARD_SETTINGS)
    assert fusion.record['deferred_projection_marker_ids'] == [17]
    assert fusion.record['steps'][0]['projection']['retained_position_rotation_cross_covariance'] is not None
    assert fusion.record['steps'][1]['prior_branch_count'] == 2
    assert len(fusion.record['steps'][1]['branch_products']) == 4
    assert fusion.record['steps'][1]['projection']['applied'] is True
    assert fusion.record['final_mixture_fallback_used'] is False
    goals, record, _snapshot = _fresh_domain_handoff(board_pipeline, fusion)
    assert len(goals) == record['central_component_count'] == 1
    assert record['raw_marker_count'] == 2 and record['raw_native_component_count'] == 4
    assert goals[0].weight == 1. and goals[0].log_weight == 0.
    assert record['marker_board_fusion'] is fusion.record
