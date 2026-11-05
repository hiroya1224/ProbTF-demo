"""Integration checks of the supplied joint attitude/acceleration recursion."""
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ros/examples/probtf_orientation_demo/scripts'))
from translational_joint_filter import JointTranslationAttitudeFilter
from translational_observation_factors import acceleration_factor,magnetic_factor
from translational_synthetic_data import generate_comparison_data,estimator_inputs


class TranslationalJointFilterTests(unittest.TestCase):
    def inputs_and_state(self, magnetic=True, translation=True):
        data=estimator_inputs(generate_comparison_data(duration=.05))
        state=JointTranslationAttitudeFilter(data['initial_wxyz'],data['gravity_world_m_s2'],
            data['magnetic_field_world'],data['gyro_covariance'],data['acceleration_covariance'],
            data['magnetic_covariance'],use_magnetic=magnetic,estimate_translation=translation)
        return data,state

    def test_observation_composition_counts_the_prior_once(self):
        data,state=self.inputs_and_state()
        priorA,priorG,priorB=state.law.A.copy(),state.law.Wg.copy(),state.law.Wb.copy()
        acc=acceleration_factor(data['acceleration_m_s2'][0],state.m,state.F,state.S,
            data['acceleration_covariance'],data['gravity_world_m_s2'])
        mag=magnetic_factor(data['magnetic_raw'][0],data['magnetic_covariance'],data['magnetic_field_world'])
        output=state.update(data['gyro_rad_s'][0],data['acceleration_m_s2'][0],0.,data['magnetic_raw'][0])
        np.testing.assert_allclose(output['A'],priorA+acc['A']+mag['A'])
        np.testing.assert_allclose(output['Wg'],priorG+acc['Wg'])
        np.testing.assert_allclose(output['Wb'],priorB+mag['Wb'])
        self.assertGreater(np.linalg.norm(output['conditional_F']),0.)

    def test_no_truth_or_random_candidates_enter_the_recursion(self):
        data,state=self.inputs_and_state()
        with mock.patch('numpy.random.default_rng',side_effect=AssertionError('No MC filter')):
            for k in range(len(data['measurement_stamps'])):
                output=state.update(data['gyro_rad_s'][k],data['acceleration_m_s2'][k],
                    0. if k==0 else .005,data['magnetic_raw'][k] if data['magnetic_observed'][k] else None)
        self.assertTrue(np.isfinite(output['quaternion_wxyz']).all())
        self.assertAlmostEqual(np.linalg.norm(output['quaternion_wxyz']),1.)
        np.linalg.cholesky(output['conditional_S'])
        self.assertEqual(state.law.parameters().shape,(19,))
        self.assertGreater(np.linalg.norm(output['Wb']),0.)
        self.assertGreater(np.linalg.norm(output['conditional_F']),0.)

    def test_gravity_control_keeps_translation_exactly_zero(self):
        data,state=self.inputs_and_state(magnetic=False,translation=False)
        for k in range(4):
            output=state.update(data['gyro_rad_s'][k],data['acceleration_m_s2'][k],0. if k==0 else .005)
        np.testing.assert_array_equal(output['conditional_m'],np.zeros(3))
        np.testing.assert_array_equal(output['conditional_F'],np.zeros((3,3)))
        np.testing.assert_array_equal(output['conditional_S'],np.zeros((3,3)))
        # A finite prediction projection may use either quartic basis to
        # represent gyro propagation; Wb is not a record of magnetic evidence.
        with mock.patch('translational_joint_filter.magnetic_factor',
                        side_effect=AssertionError('No magnetic observation')):
            state.update(data['gyro_rad_s'][4],data['acceleration_m_s2'][4],.005,
                         magnetic=data['magnetic_raw'][4])
        self.assertTrue(np.isfinite(output['pitch_variance_rad2']))


if __name__=='__main__':unittest.main()
