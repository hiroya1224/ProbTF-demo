"""Requested I=1 is a separate real slice, including XY outside its volume."""
from pathlib import Path
import sys
import unittest
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from run_imu_pid_maps import gain_axes


class GridTests(unittest.TestCase):
    def test_i_one_independent_of_recorded_xy_i(self):
        gains=np.array([[3.,.1,1.],[5.,1.,2.5],[20.,1.,8.],[4.,1.,2.]])
        volume=gain_axes(33,gains,'xy','3d')
        plane=gain_axes(33,gains,'xy','plane')
        self.assertLess(volume[1][-1],1.)
        np.testing.assert_array_equal(plane[1],[1.])
        np.testing.assert_array_equal(plane[0],volume[0])
        np.testing.assert_array_equal(plane[2],volume[2])
        self.assertIn(gains[0,0],plane[0]);self.assertIn(gains[0,2],plane[2])

    def test_axes_nested_and_failure_only(self):
        gains=np.array([[3.,.1,1.],[5.,1.,2.5],[20.,1.,8.],[4.,1.,2.]])
        for group in ('xy','z','roll_pitch','yaw'):
            low=gain_axes(33,gains,group,'3d');high=gain_axes(65,gains,group,'3d')
            for a,b in zip(low,high):
                np.testing.assert_array_equal(b[np.searchsorted(b,a)],a)


if __name__=='__main__': unittest.main()
