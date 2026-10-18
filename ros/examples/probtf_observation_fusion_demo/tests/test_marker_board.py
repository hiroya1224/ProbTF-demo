"""Known-board push-forwards, full native mixtures and central-only ADF."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from prob_artag_detector.models import EstimationDiagnostics, PoseMixtureResult
from probtf.bingham import bingham_second_moment
from probtf.distributions.bingham_orientation import BinghamOrientation
from probtf.distributions.conditional_translation import ConditionalGaussianTranslation
from probtf.distributions.stamped import TransformDistributionStamped
from probtf.distributions.transform_component import TransformComponent
from probtf.distributions.transform_distribution import TransformDistribution
from probtf.geometry import (axis_angle_to_quat, quat_to_rotmat, skew,
    right_perturbation_vec_rotation_jacobian)
from probtf.spherical_law.first_order import (
    bingham_from_rotation_coordinates, bingham_rotation_coordinates, _right_jacobian,
)
from probtf.spherical_law.fusion import Gaussian
from probtf_observation_fusion_demo import marker_board as board_module
from probtf_observation_fusion_demo.marker_board import board_layout, fuse_marker_board
from probtf_observation_fusion_demo.sensors import _marker_mode_chart


SETTINGS = dict(rows=3, columns=3, marker_size=.048, pitch=.06,
                ids=[[17, 18, 19], [20, 21, 22], [23, 24, 25]])


def detection(marker_id, *, center=(.1, -.02, .8), angles=(.04,), offsets=None,
              log_masses=None, position_variance=2e-5, rotation_variance=2e-4,
              coupling=None, stamp=1234.5):
    marker = next(item for item in board_layout(SETTINGS) if item.marker_id == marker_id)
    logs = np.zeros(len(angles)) if log_masses is None else np.asarray(log_masses, dtype=float)
    weights = np.exp(logs-logsumexp(logs))
    offsets = [np.zeros(3)]*len(angles) if offsets is None else offsets
    components, rotations, translations = [], [], []
    for index, (angle, offset, weight) in enumerate(zip(angles, offsets, weights)):
        quaternion = axis_angle_to_quat([0., 1., 0.], angle)
        rotation = quat_to_rotmat(quaternion)
        translation = np.asarray(center)+rotation @ marker.center_to_marker_translation+offset
        c = np.zeros((3, 3)) if coupling is None else np.asarray(coupling)
        vec_coupling = c @ np.linalg.pinv(right_perturbation_vec_rotation_jacobian(quaternion))
        parameter = bingham_from_rotation_coordinates(quaternion, rotation_variance*np.eye(3))
        components.append(TransformComponent('tag{}:native{}'.format(marker_id, index), weight,
            BinghamOrientation.from_parameter_matrix(parameter),
            ConditionalGaussianTranslation(translation, position_variance*np.eye(3), vec_coupling)))
        rotations.append(rotation); translations.append(translation)
    frame = 'tag_'+str(marker_id)
    record = TransformDistributionStamped('camera', frame, stamp, 'camera__to__'+frame,
        'test_native_image_hessian', TransformDistribution(tuple(components)))
    diagnostics = EstimationDiagnostics(len(angles), len(angles), 0, 0, 0, 0, ())
    return PoseMixtureResult(record, diagnostics, tuple(rotations), tuple(translations),
        tuple(logs), tuple(weights), tuple(range(len(angles))))


def test_layout_keeps_old_center_frame_and_known_rigid_inverse_for_every_tag():
    layout = board_layout(SETTINGS)
    assert [item.marker_id for item in layout] == list(range(17, 26))
    assert layout[4].marker_id == 21
    assert layout[4].center_to_marker_translation == (0., 0., 0.)
    assert layout[0].center_to_marker_translation == (-.06, .06, 0.)
    assert layout[-1].center_to_marker_translation == (.06, -.06, 0.)
    for marker in layout:
        assert marker.size_m == .048
        np.testing.assert_array_equal(np.array(marker.center_to_marker_translation)+marker.marker_to_center_translation, 0.)
        assert marker.rotation_wxyz == (1., 0., 0., 0.)


def test_default_marker_and_pitch_are_enlarged_together_without_overlap():
    layout = board_layout({})
    assert all(item.size_m == .072 for item in layout)
    assert layout[0].center_to_marker_translation == (-.09, .09, 0.)
    assert layout[-1].center_to_marker_translation == (.09, -.09, 0.)
    assert layout[4].center_to_marker_translation == (0., 0., 0.)


@pytest.mark.parametrize('changed', [dict(rows=2), dict(pitch=.04), dict(marker_size=0.),
    dict(ids=[[17, 18, 19], [20, 21, 22], [23, 24, 24]])])
def test_invalid_board_geometry_is_not_silently_changed(changed):
    with pytest.raises(ValueError):
        board_layout(dict(SETTINGS, **changed))


def test_known_attachment_push_keeps_translation_rotation_cross_until_central_projection():
    native = detection(17, coupling=np.diag([.03, -.02, .01]))
    before = native.record.distribution.components[0]
    result = fuse_marker_board({17: native}, SETTINGS)
    source = result.record['raw_observations'][0]['components'][0]
    original = np.array(source['native_joint_covariance'])
    quaternion = before.orientation.mode_wxyz
    rotation = quat_to_rotmat(quaternion)
    offset = np.array(source['known_marker_to_center_translation'])
    jacobian = np.block([[np.eye(3), -rotation @ skew(offset)],
                         [np.zeros((3, 3)), np.eye(3)]])
    expected = jacobian @ original @ jacobian.T
    np.testing.assert_allclose(source['center_full_covariance'], expected, atol=2e-15, rtol=1e-10)
    omega = original[3:, 3:]
    expected_mean = (before.translation.mean_at_reference+rotation @ offset
        +.5*rotation @ (omega-np.trace(omega)*np.eye(3)) @ offset)
    np.testing.assert_allclose(source['center_mean_translation'], expected_mean, atol=2e-14)
    assert np.linalg.norm(np.array(source['center_full_covariance'])[:3, 3:]) > 1e-6
    projection = result.record['steps'][0]['projection']
    np.testing.assert_allclose(projection['discarded_position_rotation_cross_covariance'], expected[:3, 3:], atol=2e-15)
    final = result.detected.record.distribution.components[0]
    np.testing.assert_array_equal(final.translation.rotation_coupling, np.zeros((3, 9)))
    assert native.record.distribution.components[0] is before


def test_two_marker_density_product_matches_independent_full_six_dimensional_gaussian_formula():
    observations = {21: detection(21), 22: detection(22, center=(.102, -.019, .799),
                                                 coupling=.01*np.eye(3))}
    result = fuse_marker_board(observations, SETTINGS)
    step = result.record['steps'][1]
    product = step['branch_products'][0]
    m0, p0 = np.array(step['central_prior_mean']), np.array(step['central_prior_full_covariance'])
    m1, p1 = np.array(product['common_chart_mean']), np.array(product['common_chart_full_covariance'])
    covariance = np.linalg.inv(np.linalg.inv(p0)+np.linalg.inv(p1))
    mean = covariance @ (np.linalg.solve(p0, m0)+np.linalg.solve(p1, m1))
    np.testing.assert_allclose(product['product_mean'], mean, atol=2e-13)
    np.testing.assert_allclose(product['product_full_covariance'], covariance, atol=2e-14)
    np.testing.assert_allclose(product['prior_overlap_log_evidence'],
        multivariate_normal.logpdf(m1, m0, p0+p1), atol=2e-10)
    np.testing.assert_allclose(result.detected.translations[0], mean[:3], atol=2e-13)
    assert result.record['steps'][0]['central_prior_used'] is False
    assert step['central_prior_used'] is True
    assert result.record['shared_body_prior_used'] is False
    assert result.record['body_camera_uncertainty_lift_count_in_this_helper'] == 0


def test_native_mixture_moments_include_between_covariance_and_match_quaternion_second_moment(monkeypatch):
    native = detection(21, angles=(-.08, .08),
        offsets=([-.01, 0., 0.], [.01, 0., 0.]), log_masses=np.log([.3, .7]))
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Central marker fusion must use deterministic moments, not Monte Carlo.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    result = fuse_marker_board({21: native}, SETTINGS)
    step = result.record['steps'][0]
    projection = step['projection']
    weights = np.array(projection['branch_weights'])
    means = np.array([item['product_mean'] for item in step['branch_products']])
    covariances = np.array([item['product_full_covariance'] for item in step['branch_products']])
    mean = weights @ means
    within = sum(weight*p for weight, p in zip(weights, covariances))
    between = sum(weight*np.outer(m-mean, m-mean) for weight, m in zip(weights, means))
    np.testing.assert_allclose(projection['within_covariance'], within, atol=1e-15)
    np.testing.assert_allclose(projection['between_covariance'], between, atol=1e-15)
    np.testing.assert_allclose(result.detected.record.distribution.components[0].translation.residual_covariance,
                               (within+between)[:3, :3], atol=1e-15)
    assert between[0, 0] > 5e-5 and between[4, 4] > .001
    parameter = result.detected.record.distribution.components[0].orientation.parameter_matrix()
    np.testing.assert_allclose(bingham_second_moment(parameter),
        projection['quaternion_second_moment_target'], atol=2e-6, rtol=0.)
    assert projection['quaternion_second_moment_fit_max_error'] < 2e-6
    assert projection['bingham_fit_refinement_used']
    assert projection['bingham_fit_refinement_function_evaluations'] > 0
    assert len(result.record['raw_observations'][0]['components']) == 2
    assert result.record['central_component_count'] == 1
    assert result.record['raw_native_mixtures_retained']
    assert result.record['central_position_orientation_product_approximation']
    json.dumps(result.record, allow_nan=False)


def test_finite_inexact_central_bingham_fit_records_residual_and_keeps_all_native_modes(monkeypatch):
    import probtf_observation_fusion_demo.marker_board as module
    native = detection(21, angles=(-.03, .03), log_masses=np.log([.3, .7]))
    original_parameters = tuple(component.orientation.parameter_matrix().copy()
                                for component in native.record.distribution.components)
    incomplete_parameter = np.diag([-100., -110., -120., 0.])
    monkeypatch.setattr(module, 'match_bingham_to_second_moment',
                        lambda *args, **kwargs: incomplete_parameter.copy())
    monkeypatch.setattr(module, 'least_squares',
                        lambda *args, **kwargs: SimpleNamespace(x=np.log([100., 110., 120.]), nfev=1))
    result = fuse_marker_board({21: native}, SETTINGS)
    projection = result.record['steps'][0]['projection']
    assert projection['quaternion_second_moment_fit_max_error'] > 1e-6
    assert not projection['quaternion_second_moment_fit_tolerance_verified']
    assert not projection['quaternion_second_moment_fit_residual_is_update_gate']
    assert projection['quaternion_second_moment_fit_warning']
    assert projection['applied']
    assert result.record['central_component_count'] == 1
    assert len(result.record['raw_observations'][0]['components']) == 2
    assert result.record['raw_native_mixtures_retained']
    for component, original in zip(native.record.distribution.components, original_parameters):
        np.testing.assert_array_equal(component.orientation.parameter_matrix(), original)
    json.dumps(result.record, allow_nan=False)


def test_nonfinite_central_bingham_moment_remains_an_actual_failure(monkeypatch):
    import probtf_observation_fusion_demo.marker_board as module
    native = detection(21, angles=(-.03, .03))
    incomplete_parameter = np.diag([-100., -110., -120., 0.])
    actual_moment = module.bingham_second_moment
    def invalid_moment(parameter, **kwargs):
        if np.array_equal(parameter, incomplete_parameter):
            return np.full((4, 4), np.nan)
        return actual_moment(parameter, **kwargs)
    monkeypatch.setattr(module, 'match_bingham_to_second_moment',
                        lambda *args, **kwargs: incomplete_parameter.copy())
    monkeypatch.setattr(module, 'bingham_second_moment', invalid_moment)
    with pytest.raises(ValueError, match='nonfinite parameters or moments'):
        fuse_marker_board({21: native}, SETTINGS)


def test_underflowed_native_mode_can_become_the_dominant_product_without_pruning():
    first = detection(17, center=(0., 0., .8), angles=(0.,))
    second = detection(18, center=(0., 0., .8), angles=(0., 0.),
        offsets=([1., 0., 0.], [0., 0., 0.]), log_masses=(0., -1000.))
    assert second.weights[1] == 0.
    result = fuse_marker_board({17: first, 18: second}, SETTINGS)
    raw = result.record['raw_observations'][1]['components']
    assert raw[1]['weight'] == 0. and raw[1]['log_weight'] == -1000.
    projection = result.record['steps'][1]['projection']
    assert len(projection['branch_log_weights']) == 2
    assert projection['branch_log_weights'][1] > projection['branch_log_weights'][0]+1000.
    assert abs(result.detected.translations[0][0]) < 1e-4


def test_partial_board_uses_all_visible_tags_in_fixed_ID_order_and_returns_adapter_compatible_law():
    observations = {25: detection(25), 17: detection(17), 22: detection(22)}
    forward = fuse_marker_board(observations, SETTINGS)
    reverse = fuse_marker_board(dict(reversed(list(observations.items()))), SETTINGS)
    assert forward.record == reverse.record
    assert forward.record['fusion_order'] == [17, 22, 25]
    assert len(forward.record['steps']) == 3
    detected = forward.detected
    assert detected.record.parent_frame_id == 'camera'
    assert detected.record.child_frame_id == 'marker'
    assert detected.record.stamp == 1234.5
    assert detected.weights == (1.,) and detected.log_masses == (0.,)
    assert len(detected.record.distribution.components) == 1
    central = detected.record.distribution.components[0]
    law, volume = _marker_mode_chart(central, quat_to_rotmat(central.orientation.mode_wxyz))
    np.testing.assert_allclose(law.covariance[:3, 3:], 0., atol=0.)
    assert np.isfinite(volume)
    assert forward.record['central_distribution']['component_id'] == central.component_id
    assert forward.record['central_distribution']['bingham'] == central.orientation.parameter_matrix().tolist()
    assert forward.record['central_distribution']['rotation_wxyz'] == central.orientation.mode_wxyz.tolist()
    np.testing.assert_array_equal(forward.record['central_distribution']['full_product_covariance'], law.covariance)
    assert forward.record['body_camera_uncertainty_used'] is False
    assert forward.record['exact_full_native_cartesian_product_claimed'] is False
    assert forward.record['order_independent_claimed'] is False


def test_all_nine_native_two_mode_observations_stay_recorded_without_exponential_branch_enumeration():
    observations = {marker.marker_id: detection(marker.marker_id,
        angles=(.03, .05), log_masses=np.log([.8, .2])) for marker in board_layout(SETTINGS)}
    result = fuse_marker_board(observations, SETTINGS)
    assert len(result.record['raw_observations']) == len(result.record['steps']) == 9
    assert sum(item['native_component_count'] for item in result.record['raw_observations']) == 18
    assert [len(step['branch_products']) for step in result.record['steps']] == [2]*9
    assert all(step['projection']['branch_count'] == 2 for step in result.record['steps'])
    assert len(result.detected.record.distribution.components) == 1
    expected = {component.component_id for detected in observations.values()
                for component in detected.record.distribution.components}
    actual = {component['native_component_id'] for raw in result.record['raw_observations']
              for component in raw['components']}
    assert actual == expected
    assert all(step['projection']['quaternion_second_moment_fit_max_error'] < 1e-6
               for step in result.record['steps'])


def test_broad_intermediate_fit_is_accepted_after_using_all_native_modes_without_width_cutoff():
    first = detection(17, angles=(-.4, .4), log_masses=np.log([.7, .3]),
                      coupling=.02*np.eye(3))
    second = detection(18, angles=(.4, .42), log_masses=np.log([.6, .4]))
    result = fuse_marker_board({17: first, 18: second}, SETTINGS)
    assert result.record['deferred_projection_marker_ids'] == []
    before, after = result.record['steps']
    projection = before['projection']
    assert projection['applied'] and not projection['deferred']
    assert projection['single_proxy_domain_reason'] is None
    assert not projection['concentration_threshold_applied']
    assert projection['maximum_local_rotation_variance_rad2'] > .1
    assert projection['quaternion_second_moment_fit_max_error'] < 1e-6
    assert projection['discarded_position_rotation_cross_covariance'] is not None
    assert projection['retained_position_rotation_cross_covariance'] is None
    assert len(before['branch_products']) == projection['branch_count'] == 2
    assert len(projection['branch_log_weights']) == 2
    assert all(weight > 0. for weight in projection['branch_weights'])
    assert after['prior_branch_count'] == 1
    assert len(after['branch_products']) == 2
    assert result.record['maximum_live_branch_count'] == 2
    for product in after['branch_products']:
        np.testing.assert_array_equal(product['prior_branch_mean'],
            np.r_[projection['product_translation_mean'], np.zeros(3)])
        np.testing.assert_array_equal(product['prior_branch_full_covariance'], projection['product_covariance'])
        expected_log = (product['prior_branch_log_weight']+product['native_log_weight']
            +product['push_chart_log_volume']+product['common_chart_log_volume']
            +product['prior_overlap_log_evidence'])
        assert product['product_log_mass'] == expected_log
    assert after['projection']['applied'] and not after['projection']['deferred']
    assert len(result.detected.weights) == 1
    assert not result.record['final_mixture_fallback_used']
    assert sum(raw['native_component_count'] for raw in result.record['raw_observations']) == 4


def test_final_broad_single_Bingham_uses_every_native_branch_and_preserves_its_full_width():
    native = detection(21, angles=(-.4, .4), log_masses=np.log([.7, .3]),
                       coupling=.02*np.eye(3))
    result = fuse_marker_board({21: native}, SETTINGS)
    final = result.record['central_distribution']
    assert not result.record['final_mixture_fallback_used']
    assert not result.record['concentration_threshold_applied']
    assert result.record['central_component_count'] == final['component_count'] == 1
    assert final['stamp'] == native.record.stamp
    assert final['parent_frame_id'] == 'camera' and final['child_frame_id'] == 'marker'
    assert len(result.detected.record.distribution.components) == 1
    assert result.detected.seed_indices == (0,)
    projection = result.record['steps'][-1]['projection']
    assert len(projection['branch_log_weights']) == 2
    assert all(weight > 0. for weight in projection['branch_weights'])
    assert result.record['raw_observations'][0]['native_component_count'] == 2
    component = result.detected.record.distribution.components[0]
    law, _volume = _marker_mode_chart(component, quat_to_rotmat(component.orientation.mode_wxyz))
    _, omega = bingham_rotation_coordinates(component.orientation.parameter_matrix())
    assert np.max(np.linalg.eigvalsh(omega)) > .1
    np.testing.assert_allclose(law.covariance[3:, 3:], omega, atol=2e-13, rtol=1e-12)
    np.testing.assert_allclose(law.covariance, projection['product_covariance'], atol=2e-13, rtol=1e-12)
    np.testing.assert_allclose(bingham_second_moment(component.orientation.parameter_matrix()),
        projection['quaternion_second_moment_target'], atol=1e-6)
    assert result.detected.weights == (1.,) and result.detected.log_masses == (0.,)


def test_actual_nonunique_fitted_mode_defers_without_discarding_input_branches(monkeypatch):
    # Equal orthogonal quaternion modes have no unique central modal chart.
    # Supply that exact tied fitted Bingham, rather than emulating a width
    # threshold. Every individual branch has a proper finite inverse.
    # Deliberately inaccurate but proper finite fit: the optional refinement
    # is entered, and its zero gap cannot be represented by log parameters.
    # A nonzero scalar gauge must not change that conclusion or the law.
    tied_parameter = np.diag([0., -1000., 0., -1000.])+7.25*np.eye(4)
    monkeypatch.setattr(board_module, 'match_bingham_to_second_moment',
        lambda *args, **kwargs: tied_parameter.copy())
    branches = []
    for angle in (-np.pi/2., np.pi/2.):
        mean = np.array([.1, -.02, .8, 0., angle, 0.])
        inverse = np.linalg.inv(_right_jacobian(mean[3:]))
        covariance = np.zeros((6, 6))
        covariance[:3, :3] = 2e-5*np.eye(3)
        covariance[3:, 3:] = 2e-4*inverse@inverse.T
        branches.append(dict(label=str(angle), law=Gaussian(mean, covariance), log_mass=np.log(.5)))
    before = [(item['law'].mean.copy(), item['law'].covariance.copy()) for item in branches]
    proxy, record = board_module._project_central(branches, np.array([1., 0., 0., 0.]), 120)
    assert proxy is None and record['deferred'] and not record['applied']
    assert 'maximum' in record['single_proxy_domain_reason']
    assert record['core_bingham_fit_max_error'] > 5e-8
    assert not record['bingham_fit_refinement_used']
    assert 'zero modal gap' in record['bingham_fit_refinement_skipped_reason']
    np.testing.assert_array_equal(record['bingham_parameter'], tied_parameter)
    assert not record['quaternion_second_moment_fit_tolerance_verified']
    assert record['quaternion_second_moment_fit_warning']
    assert not record['concentration_threshold_applied']
    assert record['branch_count'] == len(record['branch_log_weights']) == 2
    assert record['retained_position_rotation_cross_covariance'] is not None
    for item, (mean, covariance) in zip(branches, before):
        np.testing.assert_array_equal(item['law'].mean, mean)
        np.testing.assert_array_equal(item['law'].covariance, covariance)


def test_optional_log_concentration_refinement_uses_scalar_gauge_invariant_gaps(monkeypatch):
    fit = board_module.match_bingham_to_second_moment
    def shifted_seed(target, **options):
        # Same proper Bingham gauge, with intentionally imperfect gaps so
        # that the existing optional refinement actually runs. Absolute
        # -eigenvalue concentrations would incorrectly reject this seed.
        return .5*fit(target, **options)+1e4*np.eye(4)
    monkeypatch.setattr(board_module, 'match_bingham_to_second_moment', shifted_seed)
    native = detection(21, angles=(-.4, .4), log_masses=np.log([.7, .3]))
    result = fuse_marker_board({21: native}, SETTINGS)
    projection = result.record['steps'][0]['projection']
    assert projection['core_bingham_fit_max_error'] > 5e-8
    assert projection['bingham_fit_refinement_used']
    assert projection['bingham_fit_refinement_skipped_reason'] is None
    assert projection['quaternion_second_moment_fit_max_error'] < 1e-6
    assert projection['applied']
    assert result.record['raw_observations'][0]['native_component_count'] == 2


def test_empty_unknown_mismatched_image_and_native_weight_errors_are_explicit():
    empty = fuse_marker_board({}, SETTINGS)
    assert empty.detected is None and not empty.record['applied']
    with pytest.raises(ValueError, match='outside'):
        fuse_marker_board({9: detection(21)}, SETTINGS)
    with pytest.raises(ValueError, match='same camera image'):
        fuse_marker_board({17: detection(17), 18: detection(18, stamp=1234.6)}, SETTINGS)
    bad = replace(detection(21), weights=(.5,))
    with pytest.raises(ValueError, match='log masses'):
        fuse_marker_board({21: bad}, SETTINGS)
