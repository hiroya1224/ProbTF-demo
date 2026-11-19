"""The corruption is reproducible, indexed and distinct from inference."""
from pathlib import Path
import importlib.util
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sequential_spike_experiment_test', HERE/'run.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class SpikeRealizationTests(unittest.TestCase):
    def setUp(self):
        time = np.arange(1001)/200.
        self.source = dict(time=np.array([0., 5.]), imu_time=time,
            imu_stamp_ns=np.rint(time*1e9).astype(np.int64),
            imu_source_index=np.arange(1001)+123,
            initial_quaternion_wxyz=np.array([1., 0., 0., 0.]),
            gyro=np.column_stack((time, 2*time, -time)))
        self.config = dict(rate_hz=.5, spike_seed=20261120, spike_count=1,
                           spike_magnitude_range_deg_s=[20., 60.])

    def test_white_realization_and_original_clock_are_preserved(self):
        before = {name: value.copy() for name, value in self.source.items()}
        selected, acquisitions, sensor = experiment.select_and_corrupt(self.source, self.config)
        for name in before:
            np.testing.assert_array_equal(self.source[name], before[name])
        np.testing.assert_array_equal(acquisitions['source_index'], [123, 523, 923])
        np.testing.assert_array_equal(acquisitions['stamp_ns'], [0, 2000000000, 4000000000])
        np.testing.assert_array_equal(selected['sparse_gyro'], self.source['gyro'][[0,400,800]])
        np.testing.assert_array_equal(sensor['gyro_white'], self.source['gyro'])

    def test_same_seed_reproduces_one_recorded_noninitial_impulse(self):
        first = experiment.select_and_corrupt(self.source, self.config)
        second = experiment.select_and_corrupt(self.source, self.config)
        for a, b in zip(first, second):
            for name in a:
                np.testing.assert_array_equal(a[name], b[name])
        _, acquisitions, sensor = first
        self.assertFalse(acquisitions['spike_mask'][0])
        self.assertEqual(int(acquisitions['spike_mask'].sum()), 1)
        self.assertEqual(int(sensor['spike_mask'].sum()), 1)
        np.testing.assert_array_equal(sensor['gyro_spike'][~sensor['spike_mask']],
                                      sensor['gyro_white'][~sensor['spike_mask']])
        np.testing.assert_allclose(acquisitions['gyro_spike']-acquisitions['gyro_white'],
                                   acquisitions['spike_vector_rad_s'], atol=1e-15)
        norm = np.linalg.norm(np.rad2deg(acquisitions['spike_vector_rad_s']), axis=1)
        self.assertTrue(np.all((norm[acquisitions['spike_mask']]>=20.)
                               & (norm[acquisitions['spike_mask']]<=60.)))


if __name__ == '__main__':
    unittest.main()
