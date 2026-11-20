"""Independent normalized-density and causal-prefix checks for native Student-t.

Physics is evaluated separately with the established ControlledFlow, and
likelihoods are evaluated with scipy's multivariate distributions. No
generated truth or observation labels enter either implementation.
"""
from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
from scipy.stats import multivariate_normal, multivariate_t


HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


native = load('student_prefix_independent_physics', HERE/'native_flow.py')
prefix = load('student_prefix_under_test', HERE/'full_rate_flow.py')


class StudentPrefixTests(unittest.TestCase):
    def fixture(self, mode='nonlinear'):
        times = np.array([0., .007, .016, .024, .037, .05])
        readings = np.array([[.2, -.4, .6], [.23, -.41, .61],
            [3.2, -2.4, .3], [.26, -.39, .59], [.21, -.42, .63],
            [-.3, .2, 1.4]])
        nodes = np.array([[.12, -.31, .53], [.21, -.43, .62],
                          [.33, -.56, .74], [-.4, .5, -.6]])
        covariance = np.array([[.012, .002, -.001],
                               [.002, .018, .003],
                               [-.001, .003, .025]])
        q0 = np.array([.8, .2, -.3, .4]); q0 /= np.linalg.norm(q0)
        physical = dict(inertia=np.array([[.045, .002, -.001],
            [.002, .07, .003], [-.001, .003, .1]]),
            torque=np.array([.03, -.02, .01]))
        separate = native.ControlledFlow(physical, q0, 0., times[-1],
            max_step=.002, mode=mode).evaluate(nodes, times)
        flow = prefix.FullRateFlow(physical, q0, 0., times[-1], max_step=.002, mode=mode)
        return times, readings, nodes, covariance, separate, flow

    def test_normalized_student_t_prefix_matches_scipy_with_full_covariance(self):
        for mode in ('nonlinear', 'linear_forced'):
            for df in (3., 4., 9.):
                times, readings, nodes, covariance, separate, flow = self.fixture(mode)
                scale = (df-2.)/df*covariance
                for anchor in (0, 1, 3, 5):
                    with self.subTest(mode=mode, df=df, anchor=anchor):
                        logs, states = flow.prefix_and_block(nodes, times, readings,
                            anchor, len(times)-1, covariance, student_df=df)
                        reference = np.zeros(len(nodes))
                        for index in range(1, anchor):
                            reference += multivariate_t.logpdf(separate[index, :, :3],
                                loc=readings[index], shape=scale, df=df)
                        np.testing.assert_allclose(logs, reference, rtol=0., atol=3e-13)
                        np.testing.assert_array_equal(states, separate[anchor:])

    def test_default_gaussian_prefix_matches_existing_physics_and_density(self):
        times, readings, nodes, covariance, separate, flow = self.fixture()
        logs, states = flow.prefix_and_block(nodes, times, readings, 4, 5, covariance)
        explicit, explicit_states = flow.prefix_and_block(nodes, times, readings,
            4, 5, covariance, student_df=None)
        np.testing.assert_array_equal(logs, explicit)
        np.testing.assert_array_equal(states, explicit_states)
        np.testing.assert_array_equal(states, separate[4:])
        reference = sum(multivariate_normal.logpdf(separate[index, :, :3],
            mean=readings[index], cov=covariance) for index in range(1, 4))
        np.testing.assert_allclose(logs, reference, rtol=0., atol=3e-12)

    def test_current_and_future_readings_are_not_used_or_validated(self):
        times, readings, nodes, covariance, _separate, flow = self.fixture()
        original = flow.prefix_and_block(nodes, times, readings, 3, 5,
            covariance, student_df=4.)
        changed = readings.copy(); changed[3:] = np.nan
        changed[0] += 1000.  # y0 initializes outside this accumulated prefix.
        actual = flow.prefix_and_block(nodes, times, changed, 3, 5,
            covariance, student_df=4.)
        np.testing.assert_array_equal(actual[0], original[0])
        np.testing.assert_array_equal(actual[1], original[1])

    def test_degrees_of_freedom_must_admit_the_requested_covariance(self):
        times, readings, nodes, covariance, _separate, flow = self.fixture()
        for invalid in (-1., 0., 1., 2., np.nan, np.inf):
            with self.subTest(df=invalid), self.assertRaises(ValueError):
                flow.prefix_and_block(nodes, times, readings, 3, 5,
                    covariance, student_df=invalid)


if __name__ == '__main__':
    unittest.main()
