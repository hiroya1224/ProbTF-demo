"""Real-data adapter invariants, independent of any continuing mocap track."""
from pathlib import Path
import sys
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from imu_orientation_comparison import estimate_orientation_comparison
from imu_orientation import estimate_orientation
import imu_orientation_comparison as comparison_module
import translational_joint_filter as joint_filter_module


def measurements(count=8):
    time = np.arange(count) * .02
    return dict(time=time, gyro=np.zeros((count, 3)),
        accel=np.tile([0., 0., 9.80665], (count, 1)),
        mag=np.tile([30., 0., 15.], (count, 1)),
        mag_observed=np.ones(count, bool),
        gyro_noise_covariance=np.diag([1e-4, 2e-4, 3e-4]),
        accel_noise_covariance=np.diag([.01, .02, .03]),
        mag_noise_covariance=np.diag([.1, .2, .3]),
        initial_quaternion_wxyz=np.array([1., 0., 0., 0.]),
        initial_magnetic_reference_world=np.array([30., 0., 15.]),
        imu_stamp_ns=np.int64(1700000000000000000) + np.arange(count, dtype=np.int64) * 20000000,
        imu_source_index=np.arange(count, dtype=np.int64) * 3 + 17)


OPTIONS = dict(integration_steps=60, quadrature_order=7, projection_iterations=2)


def estimate(data):
    return estimate_orientation_comparison(data, .01, .02,
        lag_covariance_2x2=np.array([[1e-5, 2e-6], [2e-6, 2e-5]]), options=OPTIONS)


def test_both_branches_start_with_the_actual_same_bingham_law_and_skip_observations():
    data = measurements()
    result = estimate(data)
    new, old = result['translation_aware'], result['gravity_only']
    np.testing.assert_array_equal(new['A_recursive'][0], old['A_recursive'][0])
    np.testing.assert_array_equal(new['A_recursive'][0], np.diag([0., -100., -100., -100.]))
    for sequence in (new, old):
        np.testing.assert_array_equal(sequence['q_recursive_wxyz'][0], data['initial_quaternion_wxyz'])
        np.testing.assert_array_equal(sequence['q_output_wxyz'][0], data['initial_quaternion_wxyz'])
        assert not sequence['acceleration_updated'][0]
        assert not sequence['magnetic_updated'][0]
        np.testing.assert_array_equal(sequence['time'], data['time'])
        np.testing.assert_array_equal(sequence['imu_stamp_ns'], data['imu_stamp_ns'])
        np.testing.assert_array_equal(sequence['imu_source_index'], data['imu_source_index'])
        np.testing.assert_array_equal(sequence['mag_observed'], data['mag_observed'])
    # Same law, distinct finite deterministic integration rules. A100 has an
    # 8-degree initial proxy; GH7's actual moment error here is ~3.5e-7.
    np.testing.assert_allclose(new['mean_rotation_recursive'][0], old['mean_rotation_recursive'][0], atol=1e-6)
    np.testing.assert_allclose(new['rotation_second_moment_recursive'][0], old['rotation_second_moment_recursive'][0], atol=1e-6)


def test_magnetic_outputs_never_feed_back_any_part_of_joint_recursive_state():
    first = measurements(); second = {key: value.copy() for key, value in first.items()}
    second['mag'][1:] = [0., 30., 15.]
    a = estimate(first); b = estimate(second)
    for branch in ('translation_aware', 'gravity_only'):
        for key in ('A_recursive', 'Wg_recursive', 'Wb_recursive', 'q_recursive_wxyz',
                    'covariance_recursive', 'mean_rotation_recursive', 'rotation_second_moment_recursive'):
            np.testing.assert_array_equal(a[branch][key], b[branch][key])
        assert not np.allclose(a[branch]['mean_rotation_output'][-1], b[branch]['mean_rotation_output'][-1])
    for key in ('conditional_m_recursive', 'conditional_F_recursive', 'conditional_S_recursive'):
        np.testing.assert_array_equal(a['translation_aware'][key], b['translation_aware'][key])
    for key in ('m', 'F', 'S'):
        np.testing.assert_array_equal(a['translation_aware']['conditional_' + key + '_recursive'],
                                      a['translation_aware']['conditional_' + key + '_output'])


def test_positive_stationary_specific_force_uses_negative_physical_gravity():
    data = measurements(); data['mag_observed'][:] = False
    result = estimate(data)
    new = result['translation_aware']
    np.testing.assert_allclose(new['q_recursive_wxyz'], np.tile([1., 0., 0., 0.], (len(data['time']), 1)), atol=2e-7)
    np.testing.assert_array_equal(new['metadata']['physical_gravity_world'], [0., 0., -9.80665])
    for branch in ('translation_aware', 'gravity_only'):
        sequence = result[branch]
        assert not sequence['magnetic_updated'].any()
        np.testing.assert_array_equal(sequence['A_recursive'], sequence['A_output'])
        np.testing.assert_array_equal(sequence['mean_rotation_recursive'], sequence['mean_rotation_output'])
        for suffix in ('recursive', 'output'):
            full = sequence['rotation_second_moment_' + suffix]
            np.testing.assert_allclose(np.trace(full, axis1=1, axis2=2), 3., atol=1e-12)
            mean = sequence['mean_rotation_' + suffix].transpose(0, 2, 1).reshape(-1, 9)
            covariance = full - np.einsum('ni,nj->nij', mean, mean)
            assert np.linalg.eigvalsh(covariance).min() > -2e-12


def test_corrected_streams_retain_joint_tau_cross_and_full_overlap_operators():
    data = measurements()
    data['gyro'][:, 0] = 2. * data['time']
    data['accel'][:, 0] = .5 * data['time']
    result = estimate(data)
    for branch in ('translation_aware', 'gravity_only'):
        sequence = result[branch]
        assert np.linalg.norm(sequence['corrected_gyro_accel_cross_covariance']) > 0.
        for sensor in ('gyro', 'accel'):
            operator = sequence['corrected_' + sensor + '_operator']
            np.testing.assert_allclose(operator @ data[sensor], sequence['corrected_' + sensor])
            assert len(sequence['signal_operators'][sensor]) == 3
    np.testing.assert_array_equal(result['translation_aware']['corrected_gyro_covariance'],
                                  result['gravity_only']['corrected_gyro_covariance'])


def test_both_branches_use_supplied_model_increment_and_keep_raw_gyro_separate():
    data = measurements(); count = len(data['time'])
    model_rate = np.tile([0., 0., .4], (count, 1))
    model_covariance = np.tile(np.eye(3) * 1e-5, (count, 1, 1))
    source = np.zeros((count, 3, 31)); source[:, :, :3] = np.eye(3) * .02 * np.sqrt(1e-5)
    options = dict(OPTIONS, prediction_gyro=model_rate,
        prediction_gyro_covariance=model_covariance,
        prediction_increment_factor_source=source,
        prediction_gyro_source='test frozen model plus actual applied residual wrench')
    first = estimate_orientation_comparison(data, .01, .02, options=options)
    changed = {key: value.copy() for key, value in data.items()}
    changed['mag'][1:] = [0., 30., 15.]
    second = estimate_orientation_comparison(changed, .01, .02, options=options)
    for branch in ('translation_aware', 'gravity_only'):
        a, b = first[branch], second[branch]
        assert a['q_recursive_wxyz'][-1, 3] > .02
        np.testing.assert_allclose(a['corrected_gyro'], 0., atol=1e-13)
        np.testing.assert_array_equal(a['prediction_gyro'], model_rate)
        np.testing.assert_array_equal(a['prediction_gyro_covariance'], model_covariance)
        np.testing.assert_array_equal(a['prediction_increment_factor_source'], source)
        np.testing.assert_array_equal(a['A_recursive'], b['A_recursive'])
        np.testing.assert_array_equal(a['mean_rotation_recursive'], b['mean_rotation_recursive'])
        assert a['metadata']['explicit_model_increment_used']


def test_historical_default_prediction_remains_identical_to_its_explicit_input():
    data = measurements()
    default = estimate_orientation(data, .01, .02, options={'integration_steps': 60})
    explicit = estimate_orientation(data, .01, .02, options=dict(integration_steps=60,
        prediction_gyro=default['corrected_gyro'],
        prediction_gyro_covariance=default['corrected_gyro_covariance']))
    for key in ('A_recursive', 'A_output', 'q_recursive_wxyz', 'q_output_wxyz',
                'covariance_recursive', 'covariance_output', 'moment_fit_residual'):
        np.testing.assert_array_equal(default[key], explicit[key])
    assert not default['metadata']['explicit_model_increment_used']


def test_identical_optional_quadrature_settings_reach_every_new_law_query():
    data = measurements(count=4)
    numerical = dict(global_radial=2, global_angles=8,
        local_mixture_weight=.5, global_alignment='reference_covariance',
        adaptive_refinement=True, refinement_tolerance=1e-5)
    options = dict(OPTIONS, quadrature_order=3, projection_iterations=1,
                   quadrature_options=numerical)
    normalizers, projections = [], []
    original_integrate = comparison_module.ExponentialAttitudeLaw.integrate
    original_projection = joint_filter_module.forward_kl_projection
    def integrate(instance, *args, **kwargs):
        normalizers.append(dict(kwargs))
        return original_integrate(instance, *args, **kwargs)
    def project(*args, **kwargs):
        projections.append(dict(kwargs))
        return original_projection(*args, **kwargs)
    with mock.patch.object(comparison_module.ExponentialAttitudeLaw, 'integrate',
            autospec=True, side_effect=integrate), mock.patch.object(joint_filter_module,
            'forward_kl_projection', side_effect=project):
        result = estimate_orientation_comparison(data, .01, .02, options=options)
    assert len(projections) == len(data['time']) - 1
    assert len(normalizers) > len(projections) + 2
    # The outer policy must be identical at every caller. Its internal
    # coarse/fine quadrature queries intentionally use different node rules.
    outer_normalizers=[kwargs for kwargs in normalizers if kwargs.get('adaptive_refinement')]
    assert len(outer_normalizers) > len(projections) + 2
    for kwargs in outer_normalizers + projections:
        assert {key: kwargs[key] for key in numerical} == numerical
    assert result['translation_aware']['magnetic_updated'][1:].all()
    assert result['translation_aware']['metadata']['quadrature_options'] == numerical
    assert options['quadrature_options'] == numerical
