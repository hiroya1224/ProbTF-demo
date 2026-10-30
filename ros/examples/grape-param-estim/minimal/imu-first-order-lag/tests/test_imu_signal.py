from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from imu_signal import local_polynomial_signal, corrected_lag_signal


def polynomial_data():
    t = np.cumsum(.006 * (1 + .1 * np.sin(np.arange(80))))
    y = np.column_stack([t ** 3, 2 * t ** 2, 3 * t + 1])
    first = np.column_stack([3 * t ** 2, 4 * t, np.full(len(t), 3.)])
    second = np.column_stack([6 * t, np.full(len(t), 4.), np.zeros(len(t))])
    return t, y, first, second


def test_irregular_polynomial_operators_keep_all_native_rows_and_derivatives():
    t, y, first, second = polynomial_data()
    result = local_polynomial_signal(t, y, .2 * np.eye(3))
    np.testing.assert_allclose(result['smooth'], y, atol=2e-12)
    np.testing.assert_allclose(result['derivative'], first, atol=2e-10)
    np.testing.assert_allclose(result['second_derivative'], second, atol=2e-8)
    np.testing.assert_array_equal(result['time'], t)
    np.testing.assert_array_equal(result['point_indices'], np.arange(len(t)))
    assert np.all(result['neighbor_count'] >= 4)
    assert result['window_extent'][0, 0] == t[0]
    assert result['window_extent'][-1, 1] == t[-1]


def test_fixed_tau_covariance_includes_sensor_derivative_cross_terms():
    t, y, first, _ = polynomial_data()
    covariance = .2 * np.eye(3)
    result = local_polynomial_signal(t, y, covariance)
    tau = .07
    corrected = corrected_lag_signal(result, tau)
    np.testing.assert_allclose(corrected['mean'], y + tau * first, atol=2e-11)
    expected = (result['covariances'][0, 0] + tau * result['covariances'][0, 1]
                + tau * result['covariances'][1, 0] + tau ** 2 * result['covariances'][1, 1])
    np.testing.assert_allclose(corrected['covariance'], expected, atol=1e-13)
    operator = corrected['operator']
    full = (operator @ operator.T).toarray()
    for k in (0, 20, 79):
        np.testing.assert_allclose(corrected['covariance'][k], full[k, k] * covariance)
    assert full[20, 21] != 0.  # Overlapping observations retain temporal dependence.


def test_tau_uncertainty_enlarges_covariance_without_changing_corrected_mean():
    t, y, _, _ = polynomial_data()
    result = local_polynomial_signal(t, y, .2 * np.eye(3))
    fixed = corrected_lag_signal(result, .05)
    uncertain = corrected_lag_signal(result, .05, tau_variance=.01)
    np.testing.assert_array_equal(fixed['mean'], uncertain['mean'])
    expected = 2 * (fixed['covariance'] + .01 * np.einsum('ni,nj->nij', result['derivative'], result['derivative']))
    np.testing.assert_allclose(uncertain['covariance'], expected)
    assert np.min(np.linalg.eigvalsh(uncertain['covariance'] - fixed['covariance'])) >= -1e-13
