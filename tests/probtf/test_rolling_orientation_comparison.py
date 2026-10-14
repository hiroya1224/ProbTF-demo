"""Regression checks for five gyro/acceleration/magnetic fusion conditions."""

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
    def test_rpy_errors_wrap_at_the_180_degree_boundary(self):
        rotation = comparison.Rotation
        truth = comparison.wxyz_from_rotations(rotation.from_euler('xyz',
            [[179.0, 0.0, 0.0], [0.0, 0.0, -179.0]], degrees=True))
        estimate = comparison.wxyz_from_rotations(rotation.from_euler('xyz',
            [[-179.0, 0.0, 0.0], [0.0, 0.0, 179.0]], degrees=True))
        # Opposite quaternion signs encode exactly the same reference attitude.
        axis_errors, combined = comparison.rotation_error_series(estimate[:, None], -truth)
        np.testing.assert_allclose(axis_errors[:, 0], [[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]], atol=1e-12)
        np.testing.assert_allclose(combined[:, 0], [2.0, 2.0], atol=1e-12)

    def test_combined_error_uses_the_relative_rotation_angle(self):
        rotation = comparison.Rotation
        truth = comparison.wxyz_from_rotations(rotation.identity(1))
        estimate = comparison.wxyz_from_rotations(rotation.from_euler('xyz',
            [[60.0, 60.0, 60.0]], degrees=True))
        axis_errors, combined = comparison.rotation_error_series(estimate[:, None], truth)
        np.testing.assert_allclose(axis_errors[0, 0], [60.0, 60.0, 60.0], atol=1e-12)
        np.testing.assert_allclose(combined[0, 0], 2.0*np.rad2deg(np.arccos(abs(estimate[0, 0]))), atol=1e-12)
        self.assertGreater(abs(combined[0, 0] - np.linalg.norm(axis_errors[0, 0])), 10.0)

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

    def test_sensor_selection_is_independent_between_conditions(self):
        t = np.arange(8, dtype=float)*0.05
        gyro = np.tile([0.0, 0.0, 0.4], (len(t), 1))
        acceleration = np.tile([0.0, 0.0, 9.81], (len(t), 1))
        tilted_acceleration = np.tile([9.81, 0.0, 0.0], (len(t), 1))
        first = np.tile([1.0, 0.0, 0.0], (len(t), 1))
        second = np.tile([0.0, 1.0, 0.0], (len(t), 1))
        options = dict(initial_concentration=20.0, gyro_variance=1e-3,
                       acceleration_concentration=5.0, magnetic_concentration=5.0, integration_steps=120)
        a, _, _ = comparison.compare_conditions(
            gyro, acceleration, first, t, np.array([1.0,0.0,0.0,0.0]), np.array([1.0,0.0,0.0]), **options)
        b, _, _ = comparison.compare_conditions(
            gyro, acceleration, second, t, np.array([1.0,0.0,0.0,0.0]), np.array([1.0,0.0,0.0]), **options)
        c, _, _ = comparison.compare_conditions(
            gyro, tilted_acceleration, first, t, np.array([1.0,0.0,0.0,0.0]), np.array([1.0,0.0,0.0]), **options)
        np.testing.assert_allclose(a[:,0], b[:,0], atol=1e-12)
        np.testing.assert_allclose(a[:,0], c[:,0], atol=1e-12)
        np.testing.assert_allclose(a[:,1], b[:,1], atol=1e-12)
        np.testing.assert_allclose(a[:,2], c[:,2], atol=1e-12)
        self.assertGreater(np.linalg.norm(a[-1,1]-c[-1,1]), 1.0)
        self.assertGreater(np.linalg.norm(a[-1,2]-b[-1,2]), 1.0)
        self.assertGreater(np.linalg.norm(a[-1,4]-b[-1,4]), 1.0)
        self.assertGreater(np.linalg.norm(a[-1,4]-c[-1,4]), 1.0)

    def test_recursive_fusion_and_observation_only_case_keep_distinct_history(self):
        from probtf.bingham import canonical_bingham_parameter
        t = np.arange(6, dtype=float)*0.1
        gyro = np.zeros((len(t),3))
        acceleration = np.tile([0.0,0.0,9.81], (len(t),1))
        magnetic = np.tile([0.0,1.0,0.0], (len(t),1))
        q = np.array([1.0,0.0,0.0,0.0])
        initial = 20.0*(np.outer(q,q)-np.eye(4))
        evidence = comparison.magnetic_bingham_evidence([1.0,0.0,0.0], magnetic[0], 5.0)
        gravity = comparison.gravity_bingham_evidence([0.0,0.0,1.0], acceleration[0], 5.0)
        a, _, _ = comparison.compare_conditions(
            gyro, acceleration, magnetic, t, q, np.array([1.0,0.0,0.0]),
            initial_concentration=20.0, gyro_variance=0.0,
            acceleration_concentration=5.0, magnetic_concentration=5.0)
        for k in range(len(t)):
            np.testing.assert_allclose(a[k,0],initial,atol=1e-12)
            np.testing.assert_allclose(a[k,1],canonical_bingham_parameter(initial+(k+1)*gravity),atol=1e-12)
            np.testing.assert_allclose(a[k,2],canonical_bingham_parameter(initial+(k+1)*evidence),atol=1e-12)
            np.testing.assert_allclose(a[k,3],canonical_bingham_parameter(gravity+evidence),atol=1e-12)
            np.testing.assert_allclose(a[k,4],canonical_bingham_parameter(initial+(k+1)*(gravity+evidence)),atol=1e-12)

    def test_acceleration_and_magnetic_case_has_no_gyro_or_past_state(self):
        t = np.arange(4, dtype=float)*0.05
        gyro = np.zeros((len(t),3))
        other_gyro = np.tile([0.8,0.6,0.4], (len(t),1))
        acceleration = np.tile([0.0,0.0,9.81], (len(t),1))
        magnetic = np.tile([1.0,0.0,0.0], (len(t),1))
        q = np.array([1.0,0.0,0.0,0.0])
        a, _, _ = comparison.compare_conditions(gyro, acceleration, magnetic, t, q, magnetic[0])
        b, _, _ = comparison.compare_conditions(other_gyro, acceleration, magnetic, t, q, magnetic[0])
        np.testing.assert_allclose(a[:,3],b[:,3],atol=1e-12)
        acceleration[:-1] = [9.81,0.0,0.0]
        magnetic[:-1] = [0.0,1.0,0.0]
        c, _, _ = comparison.compare_conditions(gyro, acceleration, magnetic, t, q, np.array([1.0,0.0,0.0]))
        np.testing.assert_allclose(a[-1,3],c[-1,3],atol=1e-12)
        self.assertGreater(np.linalg.norm(a[-1,4]-c[-1,4]),1.0)

    def test_two_correct_directions_recover_the_known_attitude(self):
        truth = comparison.Rotation.from_euler('xyz',[55.0,-25.0,80.0],degrees=True)
        acceleration = truth.inv().apply([0.0,0.0,9.81])[None,:]
        magnetic = truth.inv().apply([1.0,0.0,0.0])[None,:]
        _, modes, _ = comparison.compare_conditions(
            np.zeros((1,3)), acceleration, magnetic, np.array([0.0]),
            np.array([1.0,0.0,0.0,0.0]), np.array([1.0,0.0,0.0]))
        self.assertGreater(abs(modes[0,3] @ comparison.wxyz_from_rotations(truth)),1.0-1e-12)

    def test_gravity_evidence_does_not_constrain_world_vertical_yaw(self):
        tilt = comparison.Rotation.from_euler('xyz',[30.0,20.0,0.0],degrees=True)
        observed = tilt.inv().apply([0.0,0.0,1.0])
        factor = comparison.gravity_bingham_evidence([0.0,0.0,1.0],observed,20.0)
        attitudes = comparison.Rotation.from_euler('z',[-179.0,30.0,140.0],degrees=True)*tilt
        q = comparison.wxyz_from_rotations(attitudes)
        values = np.einsum('ni,ij,nj->n',q,factor,q)
        np.testing.assert_allclose(values,values[0],atol=1e-12)

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
