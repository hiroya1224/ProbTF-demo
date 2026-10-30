from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import imu_orientation as orientation


def measurements(n=24):
    t = np.arange(n) * .01
    return dict(time=t, gyro=np.tile([0., 0., .25], (n, 1)),
                accel=np.tile([0., 0., 9.80665], (n, 1)),
                mag=np.tile([30., 0., 15.], (n, 1)), mag_observed=np.ones(n, dtype=bool),
                gyro_noise_covariance=1e-4 * np.eye(3),
                accel_noise_covariance=.01 * np.eye(3), mag_noise_covariance=.1 * np.eye(3),
                initial_quaternion_wxyz=np.array([1., 0., 0., 0.]),
                initial_magnetic_reference_world=np.array([30., 0., 15.]))


def test_magnetic_information_changes_outputs_but_never_the_recurrence():
    first = measurements()
    second = {key: value.copy() for key, value in first.items()}
    second['mag'][1:] = [0., 30., 15.]
    a = orientation.estimate_orientation(first, .02, .03, options={'integration_steps': 60})
    b = orientation.estimate_orientation(second, .02, .03, options={'integration_steps': 60})
    np.testing.assert_array_equal(a['A_recursive'], b['A_recursive'])
    np.testing.assert_array_equal(a['q_recursive_wxyz'], b['q_recursive_wxyz'])
    np.testing.assert_array_equal(a['covariance_recursive'], b['covariance_recursive'])
    assert not np.allclose(a['A_output'][-1], b['A_output'][-1])
    assert not np.allclose(a['q_output_wxyz'][-1], b['q_output_wxyz'][-1])
    np.testing.assert_array_equal(a['q_recursive_wxyz'][0], first['initial_quaternion_wxyz'])
    np.testing.assert_array_equal(a['q_output_wxyz'][0], first['initial_quaternion_wxyz'])
    assert not a['acceleration_updated'][0] and not a['magnetic_updated'][0]


def test_missing_or_held_magnetometer_does_not_invent_an_output_update():
    data = measurements()
    data['mag_observed'][:] = False
    result = orientation.estimate_orientation(data, 0., 0., options={'integration_steps': 60})
    np.testing.assert_array_equal(result['A_recursive'], result['A_output'])
    np.testing.assert_array_equal(result['q_recursive_wxyz'], result['q_output_wxyz'])
    assert not result['magnetic_updated'].any()
    for k in (0, 12, 23):
        _, covariance = orientation.bingham_rotation_coordinates(result['A_output'][k])
        np.testing.assert_array_equal(covariance, result['covariance_output'][k])


def test_dynamic_acceleration_inflation_is_an_isotropic_upper_envelope():
    sensor = .01 * np.eye(3)
    prior = np.diag([1., 2., 0.])
    effective, isotropic, dynamic = orientation.conservative_acceleration_covariance(
        sensor, np.array([0., 0., 9.8]), prior, np.array([3., 1., 9.8]))
    np.testing.assert_allclose(dynamic, np.outer([3., 1., 0.], [3., 1., 0.]))
    np.testing.assert_allclose(effective, 2 * (sensor + prior + dynamic))
    assert np.linalg.eigvalsh(isotropic - effective).min() >= -1e-12
    np.testing.assert_array_equal(isotropic, np.eye(3) * np.linalg.eigvalsh(effective)[-1])


def test_lag_parameter_uncertainty_is_kept_in_sensor_covariance():
    data = measurements()
    data['gyro'][:, 0] = np.sin(data['time'] * 3)
    data['accel'][:, 0] = .5 * data['time']
    covariance = np.array([[.002, .0003], [.0003, .003]])
    result = orientation.estimate_orientation(data, .02, .03, covariance,
                                               options={'integration_steps': 60})
    np.testing.assert_array_equal(result['lag_covariance_2x2'], covariance)
    assert np.linalg.norm(result['corrected_gyro_accel_cross_covariance']) > 0
    assert np.linalg.eigvalsh(result['corrected_gyro_covariance']).min() > 0
