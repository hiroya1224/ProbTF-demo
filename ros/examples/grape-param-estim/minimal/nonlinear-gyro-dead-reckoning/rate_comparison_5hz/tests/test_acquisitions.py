"""Original-clock selection and rate-independent comparison denominators."""
from pathlib import Path
import importlib.util
import json
import sys
import unittest
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('acquisition_rate_runner_checks',HERE/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class AcquisitionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archive=HERE.parent/'publication/bag8_50_55s'
        cls.data=dict(np.load(archive/'input.npz',allow_pickle=False))
        cls.metadata=json.loads((archive/'input.json').read_text())

    def test_old_half_hz_acquisitions_are_unchanged(self):
        selected,evidence=runner.acquisitions(self.data,self.metadata,.5)
        np.testing.assert_array_equal(selected['sparse_gyro_time'],self.data['sparse_gyro_time'])
        np.testing.assert_array_equal(selected['sparse_gyro'],self.data['sparse_gyro'])
        np.testing.assert_array_equal(evidence['header_stamp_ns'],self.data['sparse_gyro_stamp_ns'])

    def test_two_and_five_hz_select_distinct_original_measurements(self):
        for rate,count in ((2.,10),(5.,25)):
            selected,evidence=runner.acquisitions(self.data,self.metadata,rate)
            index=evidence['source_index']
            self.assertEqual(len(index),count)
            self.assertEqual(len(np.unique(index)),count)
            self.assertEqual(selected['sparse_gyro_time'][0],self.data['time'][0])
            expected=np.searchsorted(self.data['imu_time'],evidence['requested_time'],side='left')
            np.testing.assert_array_equal(index,expected)
            np.testing.assert_array_equal(evidence['gyro'],self.data['gyro'][index])
            np.testing.assert_array_equal(evidence['header_stamp_ns'],self.data['imu_stamp_ns'][index])
            np.testing.assert_array_equal(evidence['record_stamp_ns'],self.data['imu_record_ns'][index])

    def test_scoring_has_identical_time_weights_despite_different_acquisition_knots(self):
        start,end=self.data['time'];outputs=[]
        covariance=np.array([[.04,.005,-.003],[.005,.02,.001],[-.003,.001,.01]])
        for knots in (np.linspace(start,end,21),np.unique(np.r_[np.linspace(start,end,21),start+.123,start+1.987])):
            prediction=dict(time=knots,rpy_mean_rad=np.column_stack((.1+.01*(knots-start),
                -.02+.02*(knots-start),.03-.03*(knots-start))),
                rpy_covariance_rad2=np.broadcast_to(covariance,(len(knots),3,3)),
                core_vs_rotation_mean_max_abs=np.zeros(2))
            outputs.append(runner.common_statistics(self.data,prediction))
        np.testing.assert_array_equal(outputs[0][0]['time'],outputs[1][0]['time'])
        self.assertEqual(len(outputs[0][0]['time']),501)
        np.testing.assert_allclose(outputs[0][0]['rpy_proxy_rad'],outputs[1][0]['rpy_proxy_rad'],atol=1e-15)
        np.testing.assert_allclose(outputs[0][0]['rpy_covariance_delta_rad2'],np.broadcast_to(covariance,(501,3,3)),atol=1e-15)
        self.assertAlmostEqual(outputs[0][1]['geodesic_rmse_deg'],outputs[1][1]['geodesic_rmse_deg'],places=12)


if __name__=='__main__':unittest.main()
