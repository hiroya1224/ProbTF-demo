"""Exact full-covariance Gaussian marginalization and quaternion factors."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
CORE = str(ROOT/'ros/core/probtf_core/src')
if CORE not in sys.path:sys.path.insert(0,CORE)
SCRIPT = ROOT/'ros/examples/probtf_orientation_demo/scripts/translational_observation_factors.py'
spec = importlib.util.spec_from_file_location('translational_observation_factors', SCRIPT)
factors = importlib.util.module_from_spec(spec);spec.loader.exec_module(factors)


def quaternions_and_rotations():
    r = Rotation.from_euler('xyz', [[0.,0.,0.],[31.,-17.,53.],[-41.,23.,-19.],
                                  [17.,39.,-65.],[70.,-42.,31.]], degrees=True)
    return r.as_quat()[:,[3,0,1,2]], r


def factor_values(q, r, A, W, reference):
    h = r.inv().apply(np.broadcast_to(reference,(len(q),3)))
    return np.einsum('ni,ij,nj->n',q,A,q) - .5*np.einsum('ni,ij,nj->n',h,W,h)


class TranslationalObservationFactorTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([.7,-.3,9.3])
        self.m = np.array([.2,.1,-.15])
        self.F = np.array([[.1,.07,-.02],[-.03,.15,.09],[.05,-.04,.08]])
        self.S = np.array([[.7,.10,-.06],[.10,.4,.05],[-.06,.05,.6]])
        self.R = np.array([[.06,.01,-.005],[.01,.04,.008],[-.005,.008,.09]])
        self.g = np.array([0.,0.,-9.80665])

    def test_H_is_exact_core_rotation_quadratic_form(self):
        q,r = quaternions_and_rotations()
        x,z = np.array([.2,-.6,.8]),np.array([-.3,.1,.9])
        H = factors.quaternion_bilinear_form(x,z)
        np.testing.assert_array_equal(H,H.T)
        np.testing.assert_allclose(np.einsum('ni,ij,nj->n',q,H,q),r.apply(z)@x,atol=5e-16)
        np.testing.assert_allclose(np.einsum('ni,ij,nj->n',-q,H,-q),r.apply(z)@x,atol=5e-16)

    def test_acceleration_density_ratios_equal_closed_gaussian_integral(self):
        q,r = quaternions_and_rotations()
        f = factors.acceleration_factor(self.y,self.m,self.F,self.S,self.R,self.g)
        h = r.inv().apply(np.broadcast_to(self.g,(len(q),3)))
        residual = self.y - self.m - h@f['G'].T
        gaussian = -.5*np.einsum('ni,in->n',residual,np.linalg.solve(f['V'],residual.T))
        polynomial = factor_values(q,r,f['A'],f['Wg'],self.g)
        np.testing.assert_allclose(polynomial-polynomial[0],gaussian-gaussian[0],rtol=2e-14,atol=1e-13)
        np.testing.assert_array_equal(f['V'],self.S+self.R)

    def test_conditional_acceleration_update_equals_gaussian_conditioning(self):
        q,r = quaternions_and_rotations()
        f = factors.acceleration_factor(self.y,self.m,self.F,self.S,self.R,self.g)
        h = r.inv().apply(np.broadcast_to(self.g,(len(q),3)))
        prior_means = self.m + h@self.F.T
        expected = prior_means + (self.y+h-prior_means)@f['gain'].T
        actual = f['updated_m'] + h@f['updated_F'].T
        np.testing.assert_allclose(actual,expected,atol=4e-15)
        covariance = self.S-self.S@np.linalg.solve(self.S+self.R,self.S)
        np.testing.assert_allclose(f['updated_S'],covariance,atol=2e-16)
        np.testing.assert_array_equal(f['updated_S'],f['updated_S'].T)
        self.assertGreater(np.linalg.eigvalsh(f['updated_S']).min(),0.)

    def test_magnetic_density_ratios_equal_full_gaussian(self):
        q,r = quaternions_and_rotations()
        y,b = np.array([21.,3.,-7.]),np.array([18.,-4.,9.])
        f = factors.magnetic_factor(y,self.R,b)
        h = r.inv().apply(np.broadcast_to(b,(len(q),3)))
        residual=y-h
        gaussian=-.5*np.einsum('ni,in->n',residual,np.linalg.solve(self.R,residual.T))
        polynomial=factor_values(q,r,f['A'],f['Wb'],b)
        np.testing.assert_allclose(polynomial-polynomial[0],gaussian-gaussian[0],rtol=2e-14,atol=2e-12)
        np.testing.assert_allclose(f['Wb']@self.R,np.eye(3),atol=3e-16)
        self.assertGreater(np.max(np.abs(f['Wb']-np.diag(np.diag(f['Wb'])))),0.)

    def test_anisotropic_quartic_cannot_be_replaced_by_any_bingham_A(self):
        angle=np.linspace(0.,2*np.pi,65)
        r=Rotation.from_euler('y',angle)
        q=r.as_quat()[:,[3,0,1,2]]
        f=factors.magnetic_factor(np.zeros(3),np.diag([.4,.7,1.2]),np.array([0.,0.,1.]))
        values=factor_values(q,r,f['A'],f['Wb'],np.array([0.,0.,1.]))
        quadratic=(q[:,:,None]*q[:,None,:]).reshape(len(q),16)
        fitted=quadratic@np.linalg.lstsq(quadratic,values,rcond=None)[0]
        self.assertGreater(np.linalg.norm(values-fitted),1.)

    def test_isotropic_vector_factor_has_constant_quartic_term(self):
        q,r = quaternions_and_rotations()
        b = np.array([.2,-.4,.7])
        f=factors.magnetic_factor(np.array([.1,.5,-.3]),np.eye(3)*.3,b)
        values=factor_values(q,r,f['A'],f['Wb'],b)
        quadratic=np.einsum('ni,ij,nj->n',q,f['A'],q)
        np.testing.assert_allclose(values-quadratic,-.5*(b@b)/.3,atol=5e-16)

    def test_factors_do_not_include_a_prior_or_mutate_input_arrays(self):
        original=[x.copy() for x in (self.y,self.m,self.F,self.S,self.R,self.g)]
        with mock.patch.object(np.random,'default_rng',side_effect=AssertionError('no samples')):
            a=factors.acceleration_factor(self.y,self.m,self.F,self.S,self.R,self.g)
            m=factors.magnetic_factor(self.y,self.R,self.g)
        for old,current in zip(original,(self.y,self.m,self.F,self.S,self.R,self.g)):
            np.testing.assert_array_equal(old,current)
        self.assertNotIn('prior',a);self.assertNotIn('prior',m)
        self.assertNotIn('updated_m',m)
        np.testing.assert_array_equal(a['A'],a['A'].T)
        np.testing.assert_array_equal(a['Wg'],a['Wg'].T)

    def test_known_zero_conditional_variance_and_invalid_needed_solve(self):
        f=factors.acceleration_factor(self.y,self.m,self.F,np.zeros((3,3)),self.R,self.g)
        np.testing.assert_array_equal(f['updated_S'],np.zeros((3,3)))
        np.testing.assert_array_equal(f['updated_m'],self.m)
        np.testing.assert_array_equal(f['updated_F'],self.F)
        with self.assertRaises(np.linalg.LinAlgError):
            factors.magnetic_factor(self.y,np.zeros((3,3)),self.g)
        with self.assertRaises(ValueError):
            factors.acceleration_factor([np.nan,0.,0.],self.m,self.F,self.S,self.R,self.g)


if __name__=='__main__':unittest.main()
