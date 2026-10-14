"""Regression checks for the distinct temporal uses of magnetic evidence."""

import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np

root = Path.cwd() if 'PROBTF_COMPARISON_SCRIPT' in os.environ else Path(__file__).resolve().parents[2]
script = Path(os.environ.get('PROBTF_COMPARISON_SCRIPT',
    root / 'ros/examples/probtf_orientation_demo/scripts/compare_rolling_bag.py'))
spec = importlib.util.spec_from_file_location('rolling_comparison', script)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class RollingComparisonTest(unittest.TestCase):
    def test_sampling_a_concentrated_rotated_posterior(self):
        from probtf.bingham import bingham_second_moment
        from probtf.distributions import BinghamOrientation
        from probtf.probability.sampling import sample_bingham_orientation
        # A posterior from the actual rolling replay.  Its shifted precision
        # acquires a negative eigenvalue of order 1e-12 through roundoff.
        a = np.array([
            [-14983.789243654455,24754.917962714426,1317.3194740830736,-1413.1184887809507],
            [24754.917962714426,-41202.04141998859,1471.335761741219,616.7278321157274],
            [1317.3194740830736,1471.335761741219,-46215.125975479554,21301.303216664932],
            [-1413.1184887809507,616.7278321157274,21301.303216664932,-9976.766238687882],
        ])
        q = sample_bingham_orientation(BinghamOrientation.from_parameter_matrix(a),512)
        self.assertTrue(np.all(np.isfinite(q)))
        np.testing.assert_allclose(np.linalg.norm(q,axis=1),1.0,atol=1e-12)
        np.testing.assert_allclose(q.T@q/len(q),bingham_second_moment(a),atol=.02)

    def test_postprocessing_cannot_change_future_gyro_propagation(self):
        t = np.arange(8, dtype=float)*0.05
        gyro = np.tile([0.0, 0.0, 0.4], (len(t), 1))
        first = np.tile([1.0, 0.0, 0.0], (len(t), 1))
        second = np.tile([0.0, 1.0, 0.0], (len(t), 1))
        options = dict(initial_concentration=20.0, gyro_variance=1e-3,
                       magnetic_concentration=5.0, integration_steps=120)
        a, _, _ = comparison.compare_conditions(
            gyro, first, t, np.array([1.0,0.0,0.0,0.0]), np.array([1.0,0.0,0.0]), **options)
        b, _, _ = comparison.compare_conditions(
            gyro, second, t, np.array([1.0,0.0,0.0,0.0]), np.array([1.0,0.0,0.0]), **options)
        np.testing.assert_allclose(a[:,0], b[:,0], atol=1e-12)
        self.assertGreater(np.linalg.norm(a[-1,1]-b[-1,1]), 1.0)
        self.assertGreater(np.linalg.norm(a[-1,2]-b[-1,2]), 10.0)

    def test_single_postprocessing_factor_and_repeated_feedback_differ(self):
        from probtf.bingham import canonical_bingham_parameter
        t = np.arange(6, dtype=float)*0.1
        gyro = np.zeros((len(t),3))
        magnetic = np.tile([0.0,1.0,0.0], (len(t),1))
        q = np.array([1.0,0.0,0.0,0.0])
        initial = 20.0*(np.outer(q,q)-np.eye(4))
        evidence = comparison.magnetic_bingham_evidence([1.0,0.0,0.0], magnetic[0], 5.0)
        a, _, _ = comparison.compare_conditions(
            gyro, magnetic, t, q, np.array([1.0,0.0,0.0]),
            initial_concentration=20.0, gyro_variance=0.0, magnetic_concentration=5.0)
        for k in range(len(t)):
            np.testing.assert_allclose(a[k,0],initial,atol=1e-12)
            np.testing.assert_allclose(a[k,1],canonical_bingham_parameter(initial+evidence),atol=1e-12)
            np.testing.assert_allclose(a[k,2],canonical_bingham_parameter(initial+(k+1)*evidence),atol=1e-12)

    def test_warm_solver_preserves_core_prediction_moment_and_mode(self):
        from probtf.bingham import bingham_second_moment, bingham_mode
        from probtf_estimators.orientation_imu import predict_orientation_bingham
        a = np.diag([0.0,-4.0,-12.0,-30.0])
        gyro = np.array([0.3,-0.2,0.5])
        covariance = np.diag([0.02,0.03,0.01])
        predicted, residual = comparison.propagate(a,gyro,0.04,covariance)
        baseline = predict_orientation_bingham(a,gyro,0.04,covariance)
        np.testing.assert_allclose(bingham_second_moment(predicted),bingham_second_moment(baseline),atol=2e-5)
        self.assertGreater(abs(bingham_mode(predicted) @ bingham_mode(baseline)),1.0-1e-10)
        self.assertLess(residual,2e-5)


if __name__ == '__main__':
    unittest.main()
