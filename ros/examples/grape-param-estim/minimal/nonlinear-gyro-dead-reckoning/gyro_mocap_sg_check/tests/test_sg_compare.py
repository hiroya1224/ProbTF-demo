import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sg_compare import differentiate_quaternions, compare_lags


class QuaternionSgChecks(unittest.TestCase):
    def test_constant_body_rate_and_antipodal_representatives(self):
        t=np.linspace(-1.,2.,901)
        w=np.array([.3,-.2,.4])
        q=Rotation.from_rotvec(t[:,None]*w).as_quat()
        q[::7]*=-1.
        result=differentiate_quaternions(t,q,np.linspace(0.,1.,101),.16,3)
        np.testing.assert_allclose(result['omega_body'],np.tile(w,(101,1)),atol=2e-8)
        np.testing.assert_allclose(result['quaternion_tangency_residual'],0.,atol=1e-15)

    def test_nonidentity_initial_orientation_uses_body_not_world_rate(self):
        t=np.linspace(-1.,2.,901);w=np.array([.3,-.2,.4])
        q=(Rotation.from_euler('xyz',[.6,-.4,.2])*Rotation.from_rotvec(t[:,None]*w)).as_quat()
        result=differentiate_quaternions(t,q,np.linspace(0.,1.,101),.16,3)
        np.testing.assert_allclose(result['omega_body'],np.tile(w,(101,1)),atol=2e-8)

    def test_irregular_clock_is_not_replaced_by_constant_dt(self):
        base=np.linspace(-1.,2.,901);t=base+.0005*np.sin(7*base)
        angle=.2*t+.03*t**2
        q=Rotation.from_rotvec(np.column_stack((angle,np.zeros((len(t),2))))).as_quat()
        query=np.linspace(0.,1.,101)
        result=differentiate_quaternions(t,q,query,.08,3)
        np.testing.assert_allclose(result['omega_body'][:,0],.2+.06*query,atol=2e-8)
        np.testing.assert_allclose(result['omega_body'][:,1:],0.,atol=1e-15)

    def test_fixed_support_and_delay_sign(self):
        t=np.linspace(-1.,7.,4001)
        def signal(x):
            return np.column_stack((np.sin(3*x)+.2*np.cos(7*x),
                                    .7*np.cos(4*x),.4*np.sin(5*x)))
        lag=.0732
        comparison=compare_lags(t,signal(t-lag),t,signal(t),window=(0.,5.),max_lag=.25)
        self.assertAlmostEqual(comparison['best_lag_seconds'],lag,places=5)
        self.assertLess(comparison['after']['vector_rmse_rad_s'],2e-5)
        self.assertEqual(comparison['fixed_support_count'],2501)
        self.assertFalse(comparison['bias_or_scale_or_rotation_fitted'])

    def test_centered_window_requires_original_padding(self):
        t=np.linspace(0.,1.,121);q=Rotation.identity(len(t)).as_quat()
        with self.assertRaises(ValueError):
            differentiate_quaternions(t,q,[.01],.16)


if __name__=='__main__':unittest.main()
