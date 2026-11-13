"""Independent finite-law and Euler-proxy checks for long acquisition chains."""
from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parents[1]
PARENT=HERE.parent
for path in (HERE,PARENT):
    sys.path.insert(0,str(path))
from engine import predict_factorized,rotation_moment_update,euler_proxy
from propagation import quaternion_multiply

spec=importlib.util.spec_from_file_location('explicit_old_rotation_predictor_for_test',PARENT/'run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def discrete_rotation_moments(q,mass):
    rotation=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    vectors=rotation.transpose(0,2,1).reshape(-1,9)
    mean=mass@vectors
    centered=vectors-mean
    return mean.reshape(3,3,order='F'),np.einsum('s,si,sj->ij',mass,centered,centered)


def example_data(observations=(0.,.4,.8),end=1.):
    q=Rotation.from_euler('xyz',[.2,-.1,.3]).as_quat()[[3,0,1,2]]
    observations=np.asarray(observations)
    values=np.column_stack((.2*np.cos(observations),-.1*np.sin(2*observations),.15+0*observations))
    return dict(time=np.array([0.,end]),sparse_gyro_time=observations,sparse_gyro=values,
                initial_quaternion_wxyz=q)


def profile():
    return dict(inertia=np.diag([.06,.08,.12]),
                torque=lambda time:np.array([.002*np.sin(time),-.001,.0008]))


class FactorizedMomentTests(unittest.TestCase):
    def test_rotation_update_matches_explicit_products_and_keeps_cross_entries(self):
        parent=Rotation.from_euler('xyz',[[.2,-.1,.3],[-.1,.3,.2]]).as_quat()[:,[3,0,1,2]]
        delta=Rotation.from_rotvec([[.01,.02,-.03],[-.02,.01,.04],[.03,-.02,.01]]).as_quat()[:,[3,0,1,2]]
        wp=np.array([.3,.7]);wd=np.array([.2,.3,.5])
        m,c=discrete_rotation_moments(parent,wp)
        actual_m,actual_c=rotation_moment_update(m,c,delta,wd)
        products=quaternion_multiply(parent[:,None,:],delta[None,:,:]).reshape(-1,4)
        expected_m,expected_c=discrete_rotation_moments(products,(wp[:,None]*wd).reshape(-1))
        np.testing.assert_allclose(actual_m,expected_m,atol=8e-16)
        np.testing.assert_allclose(actual_c,expected_c,atol=2e-16)
        self.assertGreater(np.max(np.abs(actual_c-np.diag(np.diag(actual_c)))),1e-4)

    def test_analytic_euler_jacobian_matches_ambient_finite_differences(self):
        matrix=.97*Rotation.from_euler('xyz',[.4,-.3,.2]).as_matrix()
        _,_,analytic=euler_proxy(matrix,np.eye(9))
        epsilon=1e-6;numerical=np.empty((3,9))
        for column in range(9):
            change=np.zeros(9);change[column]=epsilon
            change=change.reshape(3,3,order='F')
            plus=euler_proxy(matrix+change,np.zeros((9,9)))[0]
            minus=euler_proxy(matrix-change,np.zeros((9,9)))[0]
            numerical[:,column]=(plus-minus)/(2*epsilon)
        np.testing.assert_allclose(analytic,numerical,atol=6e-11)

    def test_three_block_moments_match_all_216_old_products(self):
        data=example_data();p=profile()
        actual=predict_factorized(data,p,2.5,'nonlinear',output_dt=.04,max_step=.004)
        explicit=old.predict(data,p,2.5,'nonlinear',output_dt=.04,max_step=.004)
        np.testing.assert_array_equal(actual['time'],explicit['time'])
        np.testing.assert_allclose(actual['quaternion_second_moment'],explicit['quaternion_second_moment'],atol=4e-14)
        for row,time in enumerate(actual['time']):
            block=int(actual['block_index'][row])
            times=explicit[f'block_{block}_time']
            selected=int(np.argmin(np.abs(times-time)))
            parents=explicit[f'block_{block}_parent_wxyz'];weights=explicit[f'block_{block}_parent_weight']
            delta=explicit[f'block_{block}_delta_wxyz'][selected]
            q=quaternion_multiply(parents[:,None,:],delta[None,:,:]).reshape(-1,4)
            masses=(weights[:,None]*np.full((1,6),1/6)).reshape(-1)
            mean,covariance=discrete_rotation_moments(q,masses)
            np.testing.assert_allclose(actual['rotation_mean'][row],mean,atol=4e-15)
            np.testing.assert_allclose(actual['rotation_covariance_colmajor'][row],covariance,atol=4e-15)

    def test_many_acquisitions_keep_fixed_size_without_hidden_measurements(self):
        data=example_data(np.arange(0,5,.2),end=5.)
        actual=predict_factorized(data,profile(),2.5,'constant',output_dt=.05)
        data['dense_gyro']=np.full((100,3),np.nan)
        data['mocap_quaternion_xyzw']=np.full((100,4),np.nan)
        again=predict_factorized(data,profile(),2.5,'constant',output_dt=.05)
        for key in actual:np.testing.assert_array_equal(actual[key],again[key])
        self.assertEqual(actual['quaternion_second_moment'].shape[1:],(4,4))
        self.assertEqual(actual['rotation_covariance_colmajor'].shape[1:],(9,9))
        self.assertEqual(len([k for k in actual if k.endswith('_delta_wxyz')]),25)
        self.assertLess(float(np.max(actual['core_vs_rotation_mean_max_abs'])),3e-14)
        self.assertTrue(np.all(np.diagonal(actual['rpy_covariance_rad2'],axis1=1,axis2=2)>=0))

    def test_refresh_does_not_reset_or_reapply_attitude(self):
        data=example_data((0.,.5),end=1.)
        data['sparse_gyro']=np.array([[0,0,.2],[0,0,-.3]])
        actual=predict_factorized(data,profile(),0.,'constant',output_dt=.1)
        for row,time in enumerate(actual['time']):
            angle=.2*min(time,.5)-.3*max(time-.5,0)
            delta=np.array([np.cos(angle/2),0,0,np.sin(angle/2)])
            expected=quaternion_multiply(data['initial_quaternion_wxyz'],delta)
            np.testing.assert_allclose(actual['quaternion_second_moment'][row],np.outer(expected,expected),atol=2e-14)


if __name__=='__main__':unittest.main()
