"""Display-window cropping must preserve the already propagated history."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[2]/'ros/examples/probtf_orientation_demo/scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('success_yaw_window', SCRIPTS/'compare_success_yaw_window.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_crop_keeps_each_propagated_state_and_original_clock_without_restart():
    stamps = 10.+np.arange(10.)
    positions = 10.+np.arange(0., 10., .5)
    parameters = np.arange(10*6*4*4).reshape(10, 6, 4, 4)
    arrays = dict(measurement_stamps=stamps, measurement_stamp_ns=(stamps*1e9).astype(np.int64),
        time_seconds=np.arange(10.), recurrence_elapsed_seconds=np.arange(10.),
        display_parameter=parameters, position_measurement_stamps=positions,
        position_time_seconds=positions-10., position_m=np.column_stack((positions, positions, positions)))
    result = analysis.crop_window(arrays, 13., 17.)
    np.testing.assert_array_equal(result['display_parameter'], parameters[3:8])
    np.testing.assert_array_equal(result['recurrence_elapsed_seconds'], np.arange(3., 8.))
    np.testing.assert_array_equal(result['measurement_stamp_ns'], arrays['measurement_stamp_ns'][3:8])
    np.testing.assert_array_equal(result['time_seconds'], np.arange(5.))
    np.testing.assert_array_equal(result['position_measurement_stamps'], np.arange(13., 17.5, .5))
    assert not np.shares_memory(result['display_parameter'], parameters)
    np.testing.assert_array_equal(arrays['time_seconds'], np.arange(10.))


def test_empty_or_single_sample_window_has_no_invented_endpoint():
    arrays = dict(measurement_stamps=np.array([1., 2., 3.]),
        position_measurement_stamps=np.array([1., 2., 3.]))
    for start, end in [(4., 5.), (1.1, 2.1)]:
        with pytest.raises(ValueError, match='two measured IMU samples'):
            analysis.crop_window(arrays, start, end)


def test_condition_label_typography_preserves_the_same_evidence_choice():
    assert analysis.display_label('角速度と加速度（加速度は予測に使わない）') == '角速度と加速度 (加速度は予測に使わない)'


def test_success_driver_passes_isotropic_calibration_to_every_prediction(tmp_path, monkeypatch):
    # Deliberately unequal axis variances and cross terms catch the former
    # accidental use of the empirical full matrix at the real CLI entry point.
    stamps = np.arange(51) * .04
    offsets = np.array([[1., 2., 3.], [-1., -2., -3.],
                        [2., -1., 0.], [-2., 1., 0.],
                        [0., 1., -2.], [0., -1., 2.]]) * 1e-3
    gyro = offsets[np.arange(len(stamps)) % len(offsets)]
    acceleration = np.array([0., 0., 9.8]) + 10*gyro
    magnetic = np.array([20., 0., -30.]) + 100*gyro
    imu = np.column_stack((stamps, stamps, gyro, acceleration, magnetic))
    mocap = np.column_stack((stamps, stamps, np.tile([1., 0., 0., 0.], (len(stamps), 1)),
                            np.zeros((len(stamps), 3))))
    ns = (stamps*1e9).astype(np.int64)
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps(dict(bag_path=str(tmp_path/'input.bag'),
        bag_sha256='fixed-test-source', bag_start_ns=0,
        selection={'selected_bag_elapsed_interval_s': [1., 2.]})))
    output = tmp_path/'output'
    monkeypatch.setattr(sys, 'argv', ['compare_success_yaw_window.py',
        '--selection-json', str(selection), '--output', str(output)])
    monkeypatch.setattr(analysis, 'sha256', lambda path: 'fixed-test-source')
    monkeypatch.setattr(analysis, 'read_native_measurements',
                        lambda path: (imu, mocap, ns, ns, 0.))
    prepared, calls = [], []
    original_prepare = analysis.comparison.prepare_measurements
    original_compare = analysis.comparison.compare_six

    def prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        prepared.append(result)
        return result

    def prediction(parameter, angular_velocity, dt, covariance, integration_steps):
        calls.append((angular_velocity.copy(), dt, covariance.copy()))
        return parameter.copy(), 0.

    class FinishedPrediction(Exception):
        pass

    def compare(*args, **kwargs):
        kwargs['propagation'] = prediction
        original_compare(*args, **kwargs)
        raise FinishedPrediction  # Do not render plots in a routing regression.

    monkeypatch.setattr(analysis.comparison, 'prepare_measurements', prepare)
    monkeypatch.setattr(analysis.comparison, 'compare_six', compare)
    with pytest.raises(FinishedPrediction):
        analysis.main()
    calibration = imu[stamps < 1.]
    raw_covariance = np.cov(calibration[:, 2:5], rowvar=False, ddof=1)
    expected = np.trace(raw_covariance)/3*np.eye(3)
    assert not np.allclose(raw_covariance, expected, rtol=1e-3, atol=0)
    arrays, _, metadata = prepared[0]
    assert len(calls) == 3*(len(arrays['measurement_stamps'])-1)
    for tick in range(1, len(arrays['measurement_stamps'])):
        for group in range(3):
            angular_velocity, dt, used = calls[3*(tick-1)+group]
            np.testing.assert_array_equal(angular_velocity, arrays['gyro_rad_s'][tick])
            assert dt == arrays['measurement_stamps'][tick]-arrays['measurement_stamps'][tick-1]
            np.testing.assert_allclose(used, expected, rtol=1e-15, atol=0)
    np.testing.assert_allclose(metadata['gyro_sample_covariance_rad2_s2'], expected, rtol=1e-15, atol=0)
    record = json.loads((output/'static_noise_calibration.json').read_text())
    np.testing.assert_allclose(record['angular_velocity']['sample_covariance'], raw_covariance,
                               rtol=1e-15, atol=0)
    np.testing.assert_allclose(record['angular_velocity']['isotropic_covariance'], expected,
                               rtol=1e-15, atol=0)
    for sensor, columns in [('acceleration', slice(5, 8)), ('magnetic', slice(8, 11))]:
        expected_noise = analysis.static_vector_noise.estimate_isotropic_vector_noise(calibration[:, columns])
        assert record[sensor] == expected_noise
