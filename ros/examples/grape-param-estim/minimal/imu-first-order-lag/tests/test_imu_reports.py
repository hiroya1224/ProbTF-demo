from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import imu_reports as reports


def test_comparison_uses_native_mocap_times_and_never_extrapolates():
    time = np.array([1., 2., 3.])
    q = Rotation.from_euler("z", time * .2).as_quat()[:, [3, 0, 1, 2]]
    mt = np.array([.9, 1.1, 1.75, 2.8, 3.1])
    mq = Rotation.from_euler("z", mt * .2).as_quat()
    result = reports.orientation_comparison(mt, mq, time, q)
    np.testing.assert_array_equal(result['time'], mt[1:4])
    np.testing.assert_array_equal(result['mocap_index'], [1, 2, 3])
    np.testing.assert_allclose(result['so3_error_deg'], 0., atol=1e-12)


def test_sensor_response_is_exact_zoh_and_uses_raw_initial_measurement():
    time = np.array([10., 10.1, 10.3, 10.7])
    physical = np.full((4, 3), 4.)
    response = reports.first_order_measurement_response(time, physical, .2, np.ones(3))
    expected = 4. - 3. * np.exp(-(time - time[0]) / .2)
    np.testing.assert_allclose(response, np.tile(expected[:, None], (1, 3)), atol=1e-14)


def test_full_covariance_appendix_covers_all_cells_once_and_csv_preserves_cross(tmp_path):
    cover = np.zeros((17, 17), dtype=int)
    for rows, columns in reports.covariance_tiles(17):
        cover[rows, columns] += 1
    np.testing.assert_array_equal(cover, np.ones((17, 17)))
    covariance = np.arange(289.).reshape(17, 17)
    distribution = dict(physical_si_labels=[f"parameter_{i}" for i in range(17)],
                        physical_si_value=np.arange(17.), covariance_physical_si=covariance)
    reports._save_parameter_csv(tmp_path, distribution)
    parsed = np.genfromtxt(tmp_path / 'parameter_covariance_si.csv', delimiter=',', skip_header=1,
                          usecols=range(1, 18))
    np.testing.assert_array_equal(parsed, covariance)


def test_covariance_wireframes_preserve_rectangular_factor_axes():
    factor = np.column_stack((np.diag([3., 2., 1.]), np.zeros((3, 2))))
    circles = reports.covariance_wireframes(np.zeros(3), factor, count=65)
    for circle in circles:
        # The SVD wireframes lie on the actual 2-sigma ellipsoid.
        np.testing.assert_allclose(np.sum((circle / np.array([6., 4., 2.])[:, None])**2, axis=0), 1.)
    q = np.array([[1., 0., 0., 0.]])
    std = reports.rpy_standard_deviation(q, np.array([factor @ factor.T]), factor[None])
    np.testing.assert_allclose(std, np.rad2deg([[3., 2., 1.]]))


def test_applied_wrench_display_uses_exact_dynamics_mean_factor_and_clock():
    time = np.array([1., 1.01, 1.025])
    mean = np.arange(18.).reshape(3, 6)
    factor = np.arange(3 * 6 * 8.).reshape(3, 6, 8) / 100.
    model = {"time": time, "applied_residual_wrench_body": mean,
             "applied_residual_wrench_covariance_factor": factor}
    shown, kept_factor, std = reports.applied_wrench_distribution(model, time)
    np.testing.assert_array_equal(shown, mean)
    np.testing.assert_array_equal(kept_factor, factor)
    covariance = factor @ factor.transpose(0, 2, 1)
    np.testing.assert_allclose(std**2, np.diagonal(covariance, axis1=1, axis2=2))
    assert np.any(covariance[:, 0, 1] != 0)
    with pytest.raises(ValueError, match="original replay clock"):
        reports.applied_wrench_distribution(model, time + .001)
