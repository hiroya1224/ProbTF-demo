"""Display-window cropping must preserve the already propagated history."""
import importlib.util
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
