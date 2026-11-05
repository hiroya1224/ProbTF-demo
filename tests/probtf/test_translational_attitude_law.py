"""Deterministic checks of the exact quartic density and explicit projection."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'ros/examples/probtf_orientation_demo/scripts/translational_attitude_law.py'
spec = importlib.util.spec_from_file_location('translational_attitude_law_test_module', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from probtf.bingham import bingham_log_normalizer


G = np.array([0., 0., -9.81])
B = np.array([.4, .1, .2])
IDENTITY = np.array([1., 0., 0., 0.])


def law(A=None, Wg=None, Wb=None):
    return module.ExponentialAttitudeLaw(
        np.diag([0., -120., -90., -150.]) if A is None else A,
        np.diag([.02, .01, .03]) if Wg is None else Wg,
        np.diag([.4, .1, .2]) if Wb is None else Wb, G, B)


def recorded_missed_mode_law():
    # Immutable k641/t12.82s numerical regression from the rejected, historical
    # top-Haar-seeds-only gravity run. This is a saved density, not truth data.
    holder=module.ExponentialAttitudeLaw(
        [[9318016.89334093,3720157.4903837023,-1174502.2457590657,-4942014.892073779],
         [3720157.4903837023,2477643.8422939205,9708734.231085593,5826113.9366115965],
         [-1174502.2457590657,9708734.231085593,-8375612.976930423,2618136.3759225546],
         [-4942014.892073779,5826113.9366115965,2618136.3759225546,-3420047.758704427]],
        [[-1870.6865893398028,-1051.2008183152582,-44.45980894443667],
         [-1051.2008183152582,530.9605095315983,-256.1455402473018],
         [-44.45980894443667,-256.1455402473018,1339.7466588875304]],
        [[-381.61661719002177,2934.479083600393,-771.4264200322888],
         [2934.479083600393,2506.872327757067,4621.722742697941],
         [-771.4264200322888,4621.722742697941,-2125.2557105670453]],
        [0.,0.,-9.80665],[29.261626782550973,7.315406695637743,20.48313874778568])
    reference=np.array([.9827276687652308,-.1742468608296003,-.03481742058589605,-.05169243424360713])
    return holder,reference


class TranslationalAttitudeLawTests(unittest.TestCase):
    def test_density_matches_anisotropic_acceleration_marginal_without_tangent_fit(self):
        values = np.array([[1., 0., 0., 0.], [.8, .2, .3, .4], [.2, -.5, .1, .7]])
        values /= np.linalg.norm(values, axis=1)[:, None]
        mean = np.array([.2, -.1, .4])
        observed = np.array([.4, .7, 9.1])
        coupling = np.array([[.1, .03, 0.], [0., -.2, .01], [.02, 0., .15]])
        v = np.array([[.8, .1, .03], [.1, .3, -.02], [.03, -.02, .5]])
        transform = coupling - np.eye(3)
        information = np.linalg.inv(v)
        vector = transform.T @ information @ (observed - mean)
        holder = law(A=np.zeros((4, 4)), Wg=np.zeros((3, 3)), Wb=np.zeros((3, 3)))
        a = np.einsum('i,iab->ab', vector, holder.Qg)
        factor = law(A=a, Wg=transform.T @ information @ transform, Wb=np.zeros((3, 3)))
        rotation = module.rotation_matrices(values)
        h = np.einsum('nji,j->ni', rotation, G)
        residual = observed - mean - h @ transform.T
        expected = -.5 * np.einsum('ni,ij,nj->n', residual, information, residual)
        actual = factor.log_density(values)
        np.testing.assert_allclose(actual - actual[0], expected - expected[0], atol=1e-12)

    def test_antipodal_and_scalar_gauges_change_only_constant(self):
        holder = law()
        q = np.array([[1., 0., 0., 0.], [.9, .1, .2, -.3]])
        q /= np.linalg.norm(q, axis=1)[:, None]
        np.testing.assert_allclose(holder.log_density(q), holder.log_density(-q), atol=1e-13)
        changed = law(holder.A + 4 * np.eye(4), holder.Wg + 2*np.eye(3), holder.Wb - np.eye(3))
        expected = 4. - np.dot(G, G) + .5 * np.dot(B, B)
        np.testing.assert_allclose(changed.log_density(q)-holder.log_density(q), expected, atol=1e-12)

    def test_projective_measure_matches_existing_bingham_normalizer(self):
        holder = law(Wg=np.zeros((3, 3)), Wb=np.zeros((3, 3)))
        covariance = holder.proposal_covariance(IDENTITY)
        coarse = holder.integrate(IDENTITY,covariance,order=11)
        integrated = holder.integrate(IDENTITY,covariance,order=11,global_radial=12,global_angles=40)
        expected = bingham_log_normalizer(holder.A, integration_steps=200)
        self.assertAlmostEqual(integrated['logZ'], expected, places=7)
        self.assertLess(abs(integrated['logZ']-expected),abs(coarse['logZ']-expected))
        self.assertAlmostEqual(integrated['weights'].sum(), 1., places=13)
        np.testing.assert_allclose(np.trace(integrated['Mqq']), 1., atol=1e-14)
        np.testing.assert_allclose(integrated['Ehg'], integrated['mean_R'].T @ G, atol=1e-13)
        np.testing.assert_allclose(np.trace(integrated['Ngg']), np.dot(G, G), atol=1e-12)
        np.testing.assert_allclose(np.trace(integrated['Nbb']), np.dot(B, B), atol=1e-14)

    def test_exact_mode_curvature_is_only_an_integration_proposal(self):
        holder = law()
        reference, diagnostic = holder.find_mode(IDENTITY)
        covariance = holder.proposal_covariance(reference)
        basis = module.projective_basis(reference)
        def density(z):
            q = reference + basis @ z
            return float(holder.log_density(q / np.linalg.norm(q)))
        step = 1e-5
        hessian = np.empty((3, 3))
        for i in range(3):
            ei = np.eye(3)[i] * step
            for j in range(3):
                ej = np.eye(3)[j] * step
                hessian[i,j] = (density(ei+ej)-density(ei-ej)-density(-ei+ej)+density(-ei-ej))/(4*step*step)
        np.testing.assert_allclose(np.linalg.inv(covariance), -hessian, atol=2e-5, rtol=1e-6)
        self.assertTrue(diagnostic['finite_result_used'])
        uniform = law(np.zeros((4, 4)), np.zeros((3, 3)), np.zeros((3, 3)))
        with self.assertRaises(np.linalg.LinAlgError):
            uniform.proposal_covariance(IDENTITY)
        np.testing.assert_array_equal(uniform.proposal_covariance(IDENTITY, fallback=np.eye(3)), np.eye(3))
        np.testing.assert_array_equal(uniform.A, np.zeros((4, 4)))

    def test_all_nineteen_features_and_quartic_parameters_are_retained(self):
        initial = law()
        reference = initial.mode(IDENTITY)
        proposal = initial.proposal_covariance(reference)
        parameters = initial.parameters()
        adjustment = np.linspace(-.02, .03, 19)
        desired = initial.from_parameters(parameters + adjustment)
        target = desired.integrate(reference, proposal, order=7)
        fitted, diagnostic = module.forward_kl_projection(target['Mqq'], target['Ngg'], target['Nbb'],
            initial, reference, proposal, order=7, max_iterations=4)
        self.assertEqual(diagnostic['all_features_retained'], 19)
        self.assertTrue(diagnostic['finite_result_used'])
        self.assertLess(diagnostic['moment_residual'], diagnostic['initial_moment_residual'])
        self.assertTrue(np.any(fitted.Wg != 0.))
        self.assertTrue(np.any(fitted.Wb != 0.))
        self.assertEqual(target['features'].shape, (target['quadrature_nodes'], 19))
        self.assertGreater(target['global_quadrature_nodes'],0)
        np.testing.assert_allclose(target['feature_mean'], module.feature_target(target['Mqq'],target['Ngg'],target['Nbb'],G,B), atol=1e-15)

    def test_deterministic_transport_preserves_quartic_density(self):
        holder = law()
        delta = np.array([np.cos(.15),0.,np.sin(.15),0.])
        w,x,y,z=delta
        right=np.array([[w,-x,-y,-z],[x,w,z,-y],[y,-z,w,x],[z,y,-x,w]])
        d=module.rotation_matrices(delta)
        transported=law(right@holder.A@right.T,d.T@holder.Wg@d,d.T@holder.Wb@d)
        q=np.array([[1.,0.,0.,0.],[.6,-.2,.4,.3]])
        q/=np.linalg.norm(q,axis=1)[:,None]
        np.testing.assert_allclose(transported.log_density(q@right.T),holder.log_density(q),atol=1e-12)

    def test_integration_and_projection_do_not_sample(self):
        holder=law()
        reference=holder.mode(IDENTITY)
        proposal=holder.proposal_covariance(reference)
        with mock.patch.object(np.random,'default_rng',side_effect=AssertionError('No MC')):
            result=holder.integrate(reference,proposal,order=5)
            fitted,diagnostic=module.forward_kl_projection(result['Mqq'],result['Ngg'],result['Nbb'],holder,reference,proposal,order=5,max_iterations=1)
        self.assertTrue(np.isfinite(fitted.A).all())
        self.assertEqual(diagnostic['maximum_iterations'],1)


    def test_large_gauges_and_concentration_do_not_corrupt_probability_normalization(self):
        holder = law(np.diag([0., -1e12, -1e12, -1e12]),
                     np.diag([1e6, 2e6, 3e6]),np.diag([2e6, 3e6, 1e6]))
        shifted = law(holder.A + 1e14*np.eye(4),
                      holder.Wg + 1e12*np.eye(3),holder.Wb - 1e12*np.eye(3))
        covariance = holder.proposal_covariance(IDENTITY)
        result = holder.integrate(IDENTITY,covariance,order=7)
        other = shifted.integrate(IDENTITY,covariance,order=7)
        self.assertAlmostEqual(result['weights'].sum(),1.,places=14)
        self.assertAlmostEqual(np.trace(result['Mqq']),1.,places=14)
        np.testing.assert_allclose(other['weights'],result['weights'],atol=1e-15,rtol=1e-12)


    def test_damped_projection_transfers_small_gyro_variance_without_coefficient_explosion(self):
        from probtf.bingham import quaternion_product_second_moment
        sys.path.insert(0,str(SCRIPT.parent))
        from translational_prediction import gyro_increment
        holder=law(np.diag([0.,-8e5,-6e5,-1e6]),np.diag([.2,.1,.3]),np.diag([.2,.5,.3]))
        reference=holder.mode(IDENTITY)
        proposal=holder.proposal_covariance(reference)
        prior=holder.integrate(reference,proposal,7)
        increment=gyro_increment([0.,0.,0.],.005,np.diag([1e-4,2e-4,3e-4]))
        target=quaternion_product_second_moment(prior['Mqq'],increment['M2delta'])
        gravity=increment['average_transported_matrix'](prior['Ngg'])
        magnetic=increment['average_transported_matrix'](prior['Nbb'])
        result,diagnostic=module.forward_kl_projection(target,gravity,magnetic,holder,reference,proposal,7,3)
        posterior=result.integrate(reference,proposal,7)
        old_error=np.linalg.norm(prior['Mqq']-target)
        new_error=np.linalg.norm(posterior['Mqq']-target)
        self.assertLess(new_error,old_error*1e-3)
        self.assertGreater(np.trace(posterior['Mqq'][1:,1:]),np.trace(prior['Mqq'][1:,1:]))
        self.assertLess(diagnostic['coefficient_step_norm'],np.linalg.norm(holder.parameters()))
        self.assertTrue(diagnostic['damping_is_optimizer_only'])
        self.assertEqual(diagnostic['all_features_retained'],19)


    def test_recharted_mode_search_stays_finite_near_original_chart_boundary(self):
        angle=np.deg2rad(179.999)
        target=np.array([np.cos(angle/2),0.,0.,np.sin(angle/2)])
        parameter=1e8*(np.outer(target,target)-np.eye(4))
        holder=law(parameter,np.zeros((3,3)),np.zeros((3,3)))
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            mode,diagnostics=holder.find_mode(IDENTITY,max_iterations=12)
        self.assertTrue(np.isfinite(mode).all())
        self.assertAlmostEqual(float(mode@mode),1.,places=14)
        self.assertGreater(abs(float(mode@target)),1.-1e-10)
        self.assertEqual(diagnostics['optimizer'],'recharted_projective_Newton_trust_ascent')
        self.assertEqual(diagnostics['projective_trust_radius'],1.)
        self.assertTrue(diagnostics['finite_result_used'])

    def test_recharted_mode_can_leave_exact_minimum_without_changing_law(self):
        parameter=np.diag([-100.,-100.,-100.,0.])
        holder=law(parameter,np.zeros((3,3)),np.zeros((3,3)))
        mode,diagnostics=holder.find_mode(IDENTITY,max_iterations=12)
        self.assertGreater(abs(float(mode[3])),1.-1e-10)
        np.testing.assert_array_equal(holder.A,parameter)
        self.assertTrue(diagnostics['finite_result_used'])


    def test_target_width_seed_preserves_gyro_spread_beyond_old_quadrature_support(self):
        from probtf.bingham import quaternion_product_second_moment
        sys.path.insert(0,str(SCRIPT.parent))
        from translational_prediction import gyro_increment
        from translational_synthetic_data import ESTIMATION_COVARIANCES
        increment=gyro_increment([0.,0.,0.],.005,ESTIMATION_COVARIANCES['angular_velocity'])
        for concentration in (1e12,1e16):
            with self.subTest(concentration=concentration):
                holder=law(np.diag([0.,-concentration,-concentration,-concentration]),
                           np.zeros((3,3)),np.zeros((3,3)))
                old_proposal=holder.proposal_covariance(IDENTITY)
                prior=holder.integrate(IDENTITY,old_proposal,7)
                target=quaternion_product_second_moment(prior['Mqq'],increment['M2delta'])
                target_copy=target.copy()
                gravity=increment['average_transported_matrix'](prior['Ngg'])
                magnetic=increment['average_transported_matrix'](prior['Nbb'])
                fitted,diagnostics=module.forward_kl_projection(target,gravity,magnetic,
                    holder,IDENTITY,old_proposal,7,3)
                final=fitted.integrate(IDENTITY,fitted.proposal_covariance(IDENTITY),7)
                np.testing.assert_array_equal(target,target_copy)
                self.assertLess(np.linalg.norm(final['Mqq']-target),1e-13)
                self.assertAlmostEqual(np.trace(final['Mqq'][1:,1:])/np.trace(target[1:,1:]),1.,places=4)
                self.assertEqual(diagnostics['all_features_retained'],19)
                self.assertFalse(diagnostics['target_moments_changed'])
                self.assertFalse(diagnostics['proposal_is_exact_projective_covariance'])
                self.assertTrue(diagnostics['quartic_coefficients_seeded_unchanged'])


    def test_global_haar_nodes_contribute_remote_quartic_mass(self):
        concentration=1000.
        holder=module.ExponentialAttitudeLaw(np.diag([0.,-concentration,concentration,-concentration]),
            np.diag([-concentration,0.,0.]),np.zeros((3,3)),[1.,0.,0.],[0.,1.,0.])
        proposal=holder.proposal_covariance(IDENTITY)
        integral=holder.integrate(IDENTITY,proposal,7)
        local_count=integral['local_quadrature_nodes']
        self.assertGreater(float(integral['weights'][integral['qnodes'][:,2]**2>.5].sum()),.99)
        self.assertGreater(integral['global_quadrature_nodes'],0)
        self.assertLess(float(integral['Mqq'][0,0]),.1)
        self.assertEqual(integral['integration_method'],'deterministic_adaptive_multilocal_GH_and_global_Haar_MIS')

    def test_initialization_is_selected_by_same_global_rule_dual(self):
        holder=law()
        reference=holder.mode(IDENTITY)
        proposal=holder.proposal_covariance(reference)
        target=holder.integrate(reference,proposal,7)
        _,diagnostics=module.forward_kl_projection(target['Mqq'],target['Ngg'],target['Nbb'],
            holder,reference,proposal,7,1)
        values=diagnostics['initialization_dual_values']
        selected=diagnostics['selected_initialization']
        self.assertEqual(selected,min(values,key=values.get))
        self.assertEqual(diagnostics['all_features_retained'],19)


    def test_adaptive_multichart_normalizes_a_known_remote_bingham_peak(self):
        parameter=np.diag([-400.,0.,-400.,-400.])
        holder=law(parameter,np.zeros((3,3)),np.zeros((3,3)))
        integral=holder.integrate(IDENTITY,np.eye(3)*1e-4,7)
        expected=bingham_log_normalizer(parameter,integration_steps=200)
        self.assertAlmostEqual(integral['logZ'],expected,places=5)
        self.assertGreater(float(integral['Mqq'][1,1]),.99)
        self.assertGreater(len(integral['integration_centers']),1)

    def test_each_dual_candidate_recomputes_its_adaptive_normalizer(self):
        holder=law()
        reference=holder.mode(IDENTITY)
        proposal=holder.proposal_covariance(reference)
        desired=holder.from_parameters(holder.parameters()+np.linspace(-.02,.03,19))
        target=desired.integrate(reference,proposal,5)
        original=module.ExponentialAttitudeLaw.integrate
        called=[]
        def tracked(instance,*args,**kwargs):
            called.append(instance.parameters().copy())
            return original(instance,*args,**kwargs)
        with mock.patch.object(module.ExponentialAttitudeLaw,'integrate',autospec=True,side_effect=tracked):
            _,diagnostic=module.forward_kl_projection(target['Mqq'],target['Ngg'],target['Nbb'],
                holder,reference,proposal,5,1)
        self.assertEqual(len(called),diagnostic['adaptive_normalizer_evaluations'])
        self.assertGreater(len(called),2)
        self.assertTrue(any(not np.array_equal(called[0],value) for value in called[1:]))

    def test_spatial_starts_recover_recorded_peak_missed_by_top_Haar_scores(self):
        holder,reference=recorded_missed_mode_law()
        seeds,_=module._haar_grid(4,12)
        selected=np.argsort(holder.relative_log_density(seeds,reference))[-3:]
        old_modes=[holder.mode(seeds[i],max_iterations=20) for i in selected]
        self.assertLess(max(float(holder.relative_log_density(q,reference)) for q in old_modes),1.)
        integral=holder.integrate(reference,holder.proposal_covariance(reference),7)
        self.assertEqual(integral['spatial_projective_mode_seeds'],24)
        self.assertGreater(max(float(holder.relative_log_density(q,reference))
            for q,c in integral['integration_centers']),219900.)
        pitch=np.arcsin(np.clip(-integral['R'][:,2,0],-1.,1.))
        self.assertAlmostEqual(float(np.rad2deg(integral['weights']@pitch)),41.541391467,places=6)
        self.assertLess(len(integral['integration_centers']),10)
        self.assertTrue(any(d.get('duplicate_integration_proposal',False)
            for d in integral['integration_center_diagnostics']))

    def test_recorded_quartic_mass_is_consistent_under_deterministic_transport(self):
        holder,reference=recorded_missed_mode_law()
        before=holder.integrate(reference,holder.proposal_covariance(reference),7)
        delta=np.array([np.cos(.01),0.,np.sin(.01),0.])
        w,x,y,z=delta
        right=np.array([[w,-x,-y,-z],[x,w,z,-y],[y,-z,w,x],[z,y,-x,w]])
        rotation=module.rotation_matrices(delta)
        transported=module.ExponentialAttitudeLaw(right@holder.A@right.T,
            rotation.T@holder.Wg@rotation,rotation.T@holder.Wb@rotation,holder.g,holder.b)
        next_reference=right@reference
        after=transported.integrate(next_reference,transported.proposal_covariance(next_reference),7)
        np.testing.assert_allclose(after['Mqq'],right@before['Mqq']@right.T,atol=1e-10)
        np.testing.assert_allclose(after['Ngg'],rotation.T@before['Ngg']@rotation,atol=1e-8)
        np.testing.assert_allclose(after['Nbb'],rotation.T@before['Nbb']@rotation,atol=1e-7)
        self.assertAlmostEqual(after['logZ'],before['logZ'],places=6)

    def test_optional_global_grid_alignment_is_an_orthogonal_haar_change_only(self):
        holder=law()
        reference=np.array([.8,.1,.2,-.3]);reference/=np.linalg.norm(reference)
        covariance=np.array([[.03,.004,.001],[.004,.01,.002],[.001,.002,.05]])
        base,log_weights=module._haar_grid(3,8)
        saved_base=base.copy();saved_weights=log_weights.copy()
        integral=holder.integrate(reference,covariance,3,global_radial=3,global_angles=8,
            local_mixture_weight=.5,global_alignment='reference_covariance')
        vectors=np.linalg.eigh(covariance)[1]
        frame=np.column_stack((reference,module.projective_basis(reference)@vectors[:,::-1]))
        np.testing.assert_allclose(frame.T@frame,np.eye(4),atol=1e-15)
        self.assertAlmostEqual(abs(np.linalg.det(frame)),1.,places=14)
        np.testing.assert_allclose(integral['qnodes'][-len(base):],base@frame.T,atol=1e-15)
        np.testing.assert_array_equal(base,saved_base)
        np.testing.assert_array_equal(log_weights,saved_weights)
        self.assertAlmostEqual(float(np.exp(log_weights).sum()),1.,places=14)
        np.testing.assert_array_equal(holder.A,law().A)
        np.testing.assert_array_equal(holder.Wg,law().Wg)
        np.testing.assert_array_equal(holder.Wb,law().Wb)

    def test_omitted_alignment_preserves_the_original_native_default_exactly(self):
        holder=law();reference=holder.mode(IDENTITY);proposal=holder.proposal_covariance(reference)
        original=holder.integrate(reference,proposal,5)
        explicit=holder.integrate(reference,proposal,5,global_alignment='native')
        for key in ('qnodes','weights','R','Mqq','Ngg','Nbb','mean_R','features'):
            np.testing.assert_array_equal(original[key],explicit[key])
        self.assertEqual(original['logZ'],explicit['logZ'])
        self.assertEqual(original['relative_logZ'],explicit['relative_logZ'])

    def test_adaptive_quadrature_recovers_recorded_broad_law_haar_reference(self):
        fixture=json.loads((ROOT/'tests/probtf/fixtures/imu_broad_failure1_law1189.json').read_text())
        holder=module.ExponentialAttitudeLaw(fixture['A'],fixture['Wg'],fixture['Wb'],
            fixture['gravity_world'],fixture['magnetic_world'])
        parameters=holder.parameters().copy()
        reference=np.asarray(fixture['reference_wxyz'])
        covariance=np.asarray(fixture['proposal_covariance'])
        coarse=holder.integrate(reference,covariance,7)
        expected=fixture['reference_Haar']
        self.assertGreater(np.max(np.abs(coarse['mean_R']-expected['E_R'])),.3)
        improved=holder.integrate(reference,covariance,7,adaptive_refinement=True,
            refinement_tolerance=1e-5,global_radial=32,global_angles=32,
            local_mixture_weight=1e-5)
        np.testing.assert_allclose(improved['mean_R'],expected['E_R'],atol=1e-5,rtol=0.)
        np.testing.assert_allclose(improved['Mqq'],expected['Mqq'],atol=1e-5,rtol=0.)
        self.assertLess(abs(improved['logZ']-expected['logZ']),1e-5)
        np.testing.assert_array_equal(holder.parameters(),parameters)
        diagnostic=improved['numerical_refinement']
        self.assertTrue(diagnostic['dense_rule_used'])
        self.assertTrue(diagnostic['finite_result_retained'])
        self.assertFalse(diagnostic['state_or_observation_rejected'])
        self.assertFalse(diagnostic['error_estimate_is_global_bound'])
        self.assertGreater(max(diagnostic['final_discrepancy'].values()),0.)


if __name__=='__main__': unittest.main()
