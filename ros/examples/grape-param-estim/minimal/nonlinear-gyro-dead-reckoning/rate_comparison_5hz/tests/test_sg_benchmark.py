from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from sg_benchmark import compare_rate_curve,run_sg_benchmark


class SgBenchmarkTests(unittest.TestCase):
    def test_constant_rate_is_identical_at_both_acquisition_rates(self):
        source=np.linspace(-.2,5.2,541)
        omega=np.array([.12,-.08,.18])
        source_q=Rotation.from_rotvec(source[:,None]*omega).as_quat()
        data=dict(eval_interval=np.array([0.,5.]),mocap_time=source,
                  mocap_quaternion_xyzw=source_q)
        rate_time=np.linspace(0,5,2001)
        result=compare_rate_curve(data,rate_time,np.tile(omega,(len(rate_time),1)))
        self.assertEqual(result['metrics']['acquisition_counts'],{'0p5hz':3,'2hz':10,'5hz':25})
        for method in ('full_rate','0p5hz','2hz','5hz'):
            self.assertLess(result['metrics'][method]['geodesic_max_deg'],1e-9)
        self.assertFalse(result['metrics']['gyro_measurement_values_used'])

    def test_varying_rate_improves_with_more_frequent_samples(self):
        source=np.linspace(-.2,5.2,541)
        yaw=.1*source+.08*source**2
        data=dict(eval_interval=np.array([0.,5.]),mocap_time=source,
            mocap_quaternion_xyzw=Rotation.from_euler('z',yaw).as_quat())
        rate_time=np.linspace(0,5,2001)
        rate=np.column_stack((np.zeros((len(rate_time),2)),.1+.16*rate_time))
        result=compare_rate_curve(data,rate_time,rate)
        slow=result['metrics']['0p5hz']['geodesic_final_deg']
        middle=result['metrics']['2hz']['geodesic_final_deg']
        fast=result['metrics']['5hz']['geodesic_final_deg']
        self.assertLess(fast,middle)
        self.assertLess(middle,slow)
        self.assertLess(fast,slow/5.)
        self.assertLess(result['metrics']['full_rate']['geodesic_max_deg'],1e-9)

    def test_heldout_gyro_is_not_an_sg_diagnostic_input(self):
        source=np.linspace(-.2,1.2,141)
        data=dict(eval_interval=np.array([0.,1.]),mocap_time=source,
            mocap_quaternion_xyzw=Rotation.from_euler('z',.1*source+.08*source**2).as_quat(),
            gyro_rad_s=np.full((2,3),np.nan))
        result=run_sg_benchmark(data,.32)
        self.assertTrue(np.isfinite(result['arrays']['5hz_q_wxyz']).all())
        self.assertLess(result['metrics']['full_rate']['geodesic_max_deg'],.001)


if __name__=='__main__':unittest.main()
