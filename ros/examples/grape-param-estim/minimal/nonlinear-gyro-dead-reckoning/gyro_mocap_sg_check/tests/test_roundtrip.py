"""Kinematic right-product and SG differentiation/integration consistency."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from roundtrip import integrate_body_rate,quaternion_multiply,roundtrip_case,run_roundtrip


class RoundtripTests(unittest.TestCase):
    def test_constant_body_rate_right_multiplication_from_tilted_initial_pose(self):
        q0=Rotation.from_euler('xyz',[.3,-.2,.4]).as_quat()[[3,0,1,2]]
        times=np.linspace(0,1,101)
        omega=np.array([.2,-.1,.3])
        actual=integrate_body_rate(times,q0,[0.,1.],np.tile(omega,(2,1)),max_step=.004)
        delta=Rotation.from_rotvec(times[:,None]*omega).as_quat()[:,[3,0,1,2]]
        expected=quaternion_multiply(q0,delta)
        angle=(Rotation.from_quat(expected[:,[1,2,3,0]]).inv()*Rotation.from_quat(actual[:,[1,2,3,0]])).magnitude()
        self.assertLess(angle.max(),2e-12)

    def test_zoh_changes_at_acquisition_without_endpoint_leakage(self):
        times=np.array([0.,.4,1.,1.3,2.,3.])
        rate_time=np.array([0.,1.,2.])
        rate=np.array([[0.,0.,.1],[0.,0.,.3],[0.,0.,-.2]])
        actual=integrate_body_rate(times,[1.,0.,0.,0.],rate_time,rate,interpolation='zoh')
        yaw=.1*np.minimum(times,1.)+.3*np.clip(times-1.,0.,1.)-.2*np.maximum(times-2.,0.)
        expected=Rotation.from_euler('z',yaw).as_quat()
        error=(Rotation.from_quat(expected).inv()*Rotation.from_quat(actual[:,[1,2,3,0]])).magnitude()
        self.assertLess(error.max(),2e-15)

    def test_changing_single_axis_rate_matches_known_attitude(self):
        rate_time=np.linspace(0,2,401)
        rate=np.column_stack((np.zeros((401,2)),.1+.2*rate_time))
        times=np.linspace(0,2,121)
        actual=integrate_body_rate(times,[1.,0.,0.,0.],rate_time,rate,max_step=.002)
        expected=Rotation.from_euler('z',.1*times+.1*times**2).as_quat()
        error=(Rotation.from_quat(expected).inv()*Rotation.from_quat(actual[:,[1,2,3,0]])).magnitude()
        self.assertLess(error.max(),2e-12)

    def test_roundtrip_uses_only_initial_attitude_and_rate_no_later_reset(self):
        time=np.linspace(-.2,5.2,541)
        q=Rotation.from_euler('z',.2*time).as_quat()
        data=dict(eval_interval=np.array([0.,5.]),mocap_time=time,mocap_quaternion_xyzw=q)
        rate_time=np.linspace(0,5,1001)
        wrong_rate=np.tile([0.,0.,.1],(1001,1))
        result=roundtrip_case(data,dict(omega_body=wrong_rate,
            quaternion_xyzw=Rotation.from_euler('z',.2*rate_time).as_quat()),rate_time,.16)
        self.assertAlmostEqual(result['metrics']['full_rate']['geodesic_final_deg'],np.rad2deg(.5),places=8)
        self.assertAlmostEqual(result['metrics']['sparse_0p5hz']['geodesic_final_deg'],np.rad2deg(.5),places=8)
        self.assertFalse(result['metrics']['force_torque_model_used'])

    def test_sg_derivative_then_integration_recovers_a_changing_known_rotation(self):
        time=np.linspace(-.2,2.2,241)
        yaw=.1*time+.1*time**2
        data=dict(eval_interval=np.array([0.,2.]),mocap_time=time,
            mocap_quaternion_xyzw=Rotation.from_euler('z',yaw).as_quat())
        result=run_roundtrip(data,.16,rate_grid_dt=.005)
        # A smooth varying-rate trajectory tests the complete derivative /
        # integration seam, not just the constant-rate integrator identity.
        self.assertLess(result['metrics']['full_rate']['geodesic_max_deg'],.001)
        self.assertGreater(result['metrics']['sparse_0p5hz']['geodesic_final_deg'],20.)


if __name__=='__main__':unittest.main()
