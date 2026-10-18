"""Fresh task goals use native image modes, without replacing the body/map."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from prob_artag_detector.models import EstimationDiagnostics, PoseMixtureResult
from probtf.distributions.bingham_orientation import BinghamOrientation
from probtf.distributions.conditional_translation import ConditionalGaussianTranslation
from probtf.distributions.stamped import TransformDistributionStamped
from probtf.distributions.transform_component import TransformComponent
from probtf.distributions.transform_distribution import TransformDistribution
from probtf.geometry import axis_angle_to_quat, quat_to_rotmat
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
)
from probtf_observation_fusion_demo.body_native import point_query, pose_query
from probtf.spherical_law.support import covariance_support
from probtf_observation_fusion_demo.fresh_goal import fresh_marker_goals
from probtf_observation_fusion_demo.model import Stage
from probtf_observation_fusion_demo.sensors import LocalEstimate


def _inputs(scale=1., *, underflow=False):
    def edge(name, translation, quaternion, rotation_variance, position_variance):
        return BinghamTransform(name, translation,
            bingham_from_rotation_coordinates(quaternion,
                scale*np.diag(rotation_variance)), scale*position_variance*np.eye(3))
    body = (
        edge('body1', [.12, -.03, .17], axis_angle_to_quat([0., 0., 1.], .17),
             [.001, .0015, .002], .0002),
        edge('body2', [.23, .01, .04], axis_angle_to_quat([0., 1., 0.], -.31),
             [.0017, .0011, .0014], .0001))
    camera = BinghamTransform('camera', [.06, .02, .1],
        rotation_wxyz=axis_angle_to_quat([1., 0., 0.], .12))
    marker = edge('marker', [7., -3., 1.], [1., 0., 0., 0.],
                  [.003, .004, .005], .02)
    canonical = JointTransformState.from_independent(body+(camera, marker),
        version=8, stamp=1234., command=(.1, -.2), component_id='internal:8')
    covariance = canonical.covariance.copy()
    for left, right, correlation in ((3, 10, .17), (1, 8, -.11), (5, 18, .13)):
        covariance[left, right] = covariance[right, left] = (
            correlation*np.sqrt(covariance[left, left]*covariance[right, right]))
    canonical = replace(canonical, covariance=covariance)
    native = []
    logs = (0., -1000.) if underflow else tuple(np.log([.3, .7]))
    weights = (1., 0.) if underflow else (.3, .7)
    for index in range(2):
        quaternion = axis_angle_to_quat([0., 1., 0.], -.07+.13*index)
        orientation = BinghamOrientation.from_parameter_matrix(
            bingham_from_rotation_coordinates(quaternion,
                scale*np.diag([.0003, .0005, .0004])))
        translation = ConditionalGaussianTranslation(
            [.07+.015*index, -.03+.01*index, .52], scale*.00003*np.eye(3),
            np.array([[.01, -.02, .03, -.01, .02, .01, .02, -.01, .04],
                      [.03, .01, -.01, .02, -.03, .04, -.02, .01, -.01],
                      [-.02, .04, .01, .03, .01, -.02, .01, .03, .02]]))
        native.append(TransformComponent('image:'+str(index), weights[index],
                                         orientation, translation))
    record = TransformDistributionStamped('camera', 'marker', canonical.stamp,
        'image_marker', 'test_image_detector', TransformDistribution(tuple(native)))
    detected = PoseMixtureResult(record, EstimationDiagnostics(2, 2, 2, 0, 0, 0, ()),
        tuple(quat_to_rotmat(item.orientation.mode_wxyz) for item in native),
        tuple(item.translation.mean_at_reference for item in native),
        logs, weights, (0, 1))
    estimates = tuple(LocalEstimate(SimpleNamespace(
        observation_id='image:frame:1', sensor_id='image',
        query=SimpleNamespace(stamp=canonical.stamp, command=canonical.command,
                              state_id=parent, component_id=parent)),
        dict(sensor='marker', detected=True), detector_result=detected)
        for parent in ('keep', 'reset'))
    result = SimpleNamespace(local_estimates=estimates, prior=canonical,
        camera_path=(('body1', False), ('body2', False), ('camera', False)),
        marker_path=(('marker', False),), root_frame='base_link',
        stages=(Stage('fused_change_candidates', ()),),
        records=dict(stiffness=dict(factor_ids=['retained:0', 'retained:1'])))
    return result, canonical


def _augmented(canonical, result, component):
    index = canonical.edge_ids.index('marker')
    p = canonical.covariance.copy()
    p[6*index:6*index+6] = 0.
    p[:, 6*index:6*index+6] = 0.
    p[6*index:6*index+6, 6*index:6*index+6] = component['image_only_covariance']
    transforms = list(canonical.transforms)
    transforms[index] = BinghamTransform('marker', component['translation'],
        component['bingham'], np.asarray(component['image_only_covariance'])[:3, :3])
    metadata = component['augmented_joint_state']
    return JointTransformState(tuple(transforms), p, version=metadata['version'],
        stamp=metadata['stamp'], command=metadata['command'],
        component_id=metadata['component_id'])


def test_native_modes_are_deduplicated_across_change_parents_and_keep_all_log_masses():
    result, canonical = _inputs()
    goals, record = fresh_marker_goals(result, canonical)
    assert len(goals) == record['native_component_count'] == 2
    assert record['sensor_parent_evaluation_count'] == 2
    detected = result.local_estimates[0].detector_result
    assert [g.weight for g in goals] == list(detected.weights)
    np.testing.assert_allclose([g.log_weight for g in goals], np.log(detected.weights),
                               rtol=0., atol=1e-15)
    assert record['observation_fused_again'] is False
    assert record['exact_joint_bayesian_posterior_claimed'] is False
    assert record['old_internal_marker_used_for_goal_construction'] is False
    assert record['goal_source_state_ids'] == [g.state.state_id for g in goals]
    assert result.records['stiffness']['factor_ids'] == ['retained:0', 'retained:1']
    assert result.prior is canonical


def test_finite_own_chart_roundoff_is_diagnostic_and_does_not_drop_native_modes(monkeypatch):
    import probtf_observation_fusion_demo.fresh_goal as module
    from probtf.spherical_law.fusion import Gaussian
    result, canonical = _inputs()
    expected, _ = fresh_marker_goals(result, canonical)
    original = module._marker_mode_chart
    def with_roundoff(*args):
        law, volume = original(*args)
        mean = law.mean.copy()
        mean[3] += 2e-12
        return Gaussian(mean, law.covariance), volume
    monkeypatch.setattr(module, '_marker_mode_chart', with_roundoff)
    actual, record = fresh_marker_goals(result, canonical)
    assert [item.state.state_id for item in actual] == [item.state.state_id for item in expected]
    assert len(actual) == 2
    assert all(not item['numerical_chart_diagnostics']['legacy_rotation_origin_tolerance_met']
               for item in module._native_modes(result.local_estimates[0].detector_result))
    assert record['native_component_count'] == 2


def test_changed_retained_map_cannot_pull_the_fresh_goal_away_from_the_image():
    result, canonical = _inputs()
    goals, _ = fresh_marker_goals(result, canonical)
    marker = canonical.transforms[-1]
    changed_marker = BinghamTransform(marker.edge_id, [-8., 5., 4.],
        marker.bingham, marker.translation_covariance)
    changed = replace(canonical, transforms=canonical.transforms[:-1]+(changed_marker,))
    actual, record = fresh_marker_goals(result, changed)
    assert changed.state_id != canonical.state_id
    assert [g.state.state_id for g in actual] == [g.state.state_id for g in goals]
    assert record['canonical_internal_state_id'] == changed.state_id
    np.testing.assert_array_equal(changed.covariance, canonical.covariance)
    assert changed.covariance[5, 18] != 0.  # Internal old-map cross is retained.


def test_new_relative_image_position_changes_the_goal_at_the_same_capture_body():
    result, canonical = _inputs()
    goals, _ = fresh_marker_goals(result, canonical)
    detected = result.local_estimates[0].detector_result
    modes = tuple(replace(component, translation=replace(component.translation,
        mean_at_reference=component.translation.mean_at_reference+np.array([.04, -.02, .01])))
        for component in detected.record.distribution.components)
    record = replace(detected.record, distribution=TransformDistribution(modes))
    new_detected = replace(detected, record=record,
        translations=tuple(item.translation.mean_at_reference for item in modes))
    changed = SimpleNamespace(**dict(vars(result), local_estimates=tuple(
        replace(item, detector_result=new_detected) for item in result.local_estimates)))
    updated, _ = fresh_marker_goals(changed, canonical)
    assert all(np.linalg.norm(old.state.transforms[-1].translation-new.state.transforms[-1].translation)>.04
               for old, new in zip(goals, updated))
    assert [g.weight for g in goals] == [g.weight for g in updated]


def test_joint_goal_uses_the_same_full_body_and_core_camera_pushforward_cross():
    result, canonical = _inputs()
    goals, record = fresh_marker_goals(result, canonical)
    for goal, mode in zip(goals, record['components']):
        state = goal.state
        np.testing.assert_array_equal(state.covariance[:18, :18], canonical.covariance[:18, :18])
        assert all(a is b for a, b in zip(state.transforms[:-1], canonical.transforms[:-1]))
        assert state.command == canonical.command and state.stamp == canonical.stamp
        assert state.version == canonical.version
        assert np.any(np.asarray(mode['image_only_covariance'])[:3, 3:])
        augmented = _augmented(canonical, result, mode)
        assert augmented.state_id == mode['augmented_joint_state']['state_id']
        query = pose_query(augmented, result.camera_path+result.marker_path,
                           query_id=mode['root_pose_query']['query_id'], frame=result.root_frame)
        chart = np.asarray(mode['root_marker_chart_jacobian'])
        expected_cross = query.cross_covariance[:18] @ chart.T
        np.testing.assert_array_equal(state.covariance[:18, 18:], expected_cross)
        np.testing.assert_allclose(state.covariance[18:, 18:], chart @ query.covariance @ chart.T,
                                   rtol=1e-13, atol=1e-17)
        np.testing.assert_array_equal(state.transforms[-1].translation, query.mean[:3])
        assert np.linalg.norm(expected_cross)>0.
        covariance_support(state.covariance, 'full goal test')
        np.testing.assert_array_equal(canonical.covariance[5, 18], result.prior.covariance[5, 18])


def test_camera_relative_closure_records_first_order_nonclosure_without_recentering():
    errors = []
    for scale in (.2, .1, .05):
        result, canonical = _inputs(scale)
        goals, record = fresh_marker_goals(result, canonical)
        values, covariance_errors, rotation_errors = [], [], []
        for goal, component in zip(goals, record['components']):
            closure = component['camera_relative_closure']
            actual = pose_query(goal.state,
                tuple((name, not inverse) for name, inverse in reversed(result.camera_path))
                    +result.marker_path,
                query_id=closure['query']['query_id'], frame='camera')
            np.testing.assert_array_equal(actual.mean, closure['query']['mean'])
            np.testing.assert_array_equal(actual.covariance, closure['query']['covariance'])
            np.testing.assert_array_equal(np.asarray(closure['mean_error']),
                np.asarray(closure['mean'])-component['image_only_mean'])
            np.testing.assert_array_equal(np.asarray(closure['covariance_error']),
                np.asarray(closure['covariance'])-component['image_only_covariance'])
            values.append(closure['translation_mean_error_m'])
            covariance_errors.append(np.linalg.norm(closure['covariance_error']))
            rotation_errors.append(closure['rotation_mean_error_rad'])
            assert closure['translation_mean_error_m']<4e-5
            assert not closure['exact_closure_claimed']
            assert closure['root_forward_pose_mean_preserved']
            assert not closure['mean_recentered_to_force_inverse_closure']
        errors.append((np.array(values), np.array(covariance_errors), np.array(rotation_errors)))
    # Both body and image P are scaled. The existing direction-dependent S2
    # mean approximation is not closed under collapsing a root pose. Its
    # inverse-composition position mean residual is O(P), and must not be
    # reported as exact closure or as an O(P**2) remainder.
    for previous, current in zip(errors, errors[1:]):
        assert np.all(current[0]>.48*previous[0])
        assert np.all(current[0]<.52*previous[0])
        assert np.all(current[1]<.27*previous[1])
        assert np.all(current[2]<.27*previous[2])


def test_forward_root_origin_is_preserved_and_offset_nonclosure_is_measured_honestly():
    errors = []
    for scale in (.2, .1):
        result, canonical = _inputs(scale)
        goals, record = fresh_marker_goals(result, canonical)
        mean_error, covariance_error = [], []
        for goal, component in zip(goals, record['components']):
            augmented = _augmented(canonical, result, component)
            for point in ([0., 0., 0.], [.08, -.14, .025]):
                direct = point_query(augmented, result.camera_path+result.marker_path, point)
                collapsed = point_query(goal.state, result.marker_path, point)
                if point == [0., 0., 0.]:
                    np.testing.assert_array_equal(direct.mean, collapsed.mean)
                    np.testing.assert_allclose(direct.covariance, collapsed.covariance,
                                               rtol=1e-13, atol=1e-17)
                else:
                    mean_error.append(np.linalg.norm(direct.mean-collapsed.mean))
                    covariance_error.append(np.linalg.norm(direct.covariance-collapsed.covariance))
        errors.append((np.array(mean_error), np.array(covariance_error)))
    assert np.all(errors[1][0]>.48*errors[0][0])
    assert np.all(errors[1][0]<.52*errors[0][0])
    assert np.all(errors[1][1]<.28*errors[0][1])


def test_underflowed_native_mode_survives_without_any_sampling(monkeypatch):
    result, canonical = _inputs(underflow=True)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Fresh goals must use analytic core moments.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    goals, record = fresh_marker_goals(result, canonical)
    assert len(goals) == 2
    assert goals[1].weight == 0. and goals[1].log_weight == -1000.
    assert record['components'][1]['native_log_mass'] == -1000.
    assert record['monte_carlo_used'] is record['component_pruning_used'] is False


def test_empty_observation_is_explicit_and_foreign_or_inconsistent_capture_is_rejected():
    result, canonical = _inputs()
    empty = SimpleNamespace(local_estimates=())
    goals, record = fresh_marker_goals(empty, canonical)
    assert goals == () and not record['applied']
    with pytest.raises(ValueError, match='capture metadata'):
        fresh_marker_goals(result, replace(canonical, stamp=canonical.stamp+.01))
    duplicate = result.local_estimates[-1]
    changed_detector = replace(duplicate.detector_result, log_masses=tuple(np.log([.4, .6])), weights=(.4, .6))
    foreign = SimpleNamespace(**dict(vars(result), local_estimates=(
        result.local_estimates[0], replace(duplicate, detector_result=changed_detector))))
    with pytest.raises(ValueError, match='same image-only modes'):
        fresh_marker_goals(foreign, canonical)
