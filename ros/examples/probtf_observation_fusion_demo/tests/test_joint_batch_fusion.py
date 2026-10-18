"""Exact batch products retain native branches and all shared cross blocks."""
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np
import pytest
import yaml
from scipy.special import logsumexp
from scipy.linalg import block_diag

from probtf.spherical_law.fusion import Gaussian, condition_gaussian, gaussian_reference_product
from probtf_observation_fusion_demo.batch_observation_factors import GaussianLikelihoodComponent
from probtf_observation_fusion_demo.joint_batch_fusion import (
    fuse_gaussian_likelihood_factors,fuse_gaussian_likelihood_factors_with_prefixes,
    log_evidence_parts_difference,
)


def component(label, h, y, noise, *, offset=None, log_weight=0.):
    h = np.atleast_2d(h)
    y = np.atleast_1d(y)
    return GaussianLikelihoodComponent(label, h, np.zeros(y.size) if offset is None else offset,
        y, np.atleast_2d(noise), log_weight, source_ids=('native:'+label,))


def factor(identifier, components, **kw):
    return SimpleNamespace(factor_id=identifier, observation_ids=('observation:'+identifier,),
        components=tuple(components), common_state_id=kw.get('state_id','common'),
        common_stamp=kw.get('stamp',1.), common_command=kw.get('command',(0.,)))


def correlated_prior():
    return Gaussian(np.array([.2,-.4,.7]), np.array([[2.,.4,.3],[.4,1.,.2],[.3,.2,.8]]))


def core_oracle(prior, components):
    locals_, local_masses = [], []
    for item in components:
        posterior, log_mass = condition_gaussian(prior, item.observed-item.offset,
                                                 item.covariance, item.affine)
        locals_.append(posterior)
        local_masses.append(item.log_weight+log_mass)
    posterior, correction = gaussian_reference_product(prior, locals_)
    return posterior, sum(local_masses)+correction


def test_correlated_joint_product_matches_core_reference_prior_correction():
    prior = correlated_prior()
    a = component('gravity', [[1.,0.,.5]], [.1], [[.2]], offset=[.3])
    b = component('marker', [[0.,1.,.3],[.2,0.,1.]], [-.7,.6], [[.1,.02],[.02,.3]])
    result = fuse_gaussian_likelihood_factors(prior, [factor('imu',[a]),factor('image',[b])])
    expected, mass = core_oracle(prior,[a,b])
    np.testing.assert_allclose(result.gaussian.mean,expected.mean,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(result.gaussian.covariance,expected.covariance,rtol=1e-12,atol=1e-12)
    assert result.log_evidence == pytest.approx(mass,abs=1e-12)
    assert result.gaussian.covariance[0,1] != 0.
    assert result.record['intermediate_projection_count'] == 0
    assert len(result.posterior_components) == 1


def test_all_native_mixture_combinations_and_exact_final_moments_match_oracle():
    prior = correlated_prior()
    factors = [factor('one',[component('a',[[1.,0.,0.]],[.1],[[.2]],log_weight=math.log(.3)),
                             component('b',[[1.,0.,0.]],[1.2],[[.4]],log_weight=math.log(.7))]),
               factor('two',[component('left',[[0.,1.,1.]],[.5],[[.3]],log_weight=math.log(.6)),
                             component('right',[[0.,1.,1.]],[-.9],[[.2]],log_weight=math.log(.4))])]
    result = fuse_gaussian_likelihood_factors(prior,factors,source_ids=('parent',))
    raw = [core_oracle(prior,parts) for parts in product(*(f.components for f in factors))]
    log_mass = logsumexp([entry[1] for entry in raw])
    weights = np.exp([entry[1]-log_mass for entry in raw])
    mean = sum(w*entry[0].mean for w,entry in zip(weights,raw))
    covariance = sum(w*(entry[0].covariance+np.outer(entry[0].mean-mean,entry[0].mean-mean))
                     for w,entry in zip(weights,raw))
    assert len(result.branches) == 4
    assert result.log_evidence == pytest.approx(log_mass,abs=1e-12)
    np.testing.assert_allclose(result.gaussian.mean,mean,atol=1e-12)
    np.testing.assert_allclose(result.gaussian.covariance,covariance,atol=1e-12)
    for branch,(expected,mass) in zip(result.branches,raw):
        np.testing.assert_allclose(branch.distribution.mean,expected.mean,atol=1e-12)
        np.testing.assert_allclose(branch.distribution.covariance,expected.covariance,atol=1e-12)
        assert branch.log_evidence == pytest.approx(mass,abs=1e-12)
        assert branch.source_ids[0] == 'parent'
    assert result.record['component_pruning_used'] is False
    assert result.record['laplace_used'] is False


def test_singular_prior_retains_dirac_coordinates_and_supported_full_cross():
    prior = Gaussian(np.array([.4,-.3,7.]),np.array([[1.,.6,0.],[.6,2.,0.],[0.,0.,0.]]))
    a = component('a',[[1.,.3,1.]],[7.6],[[.1]])
    b = component('b',[[0.,1.,-.2]],[-1.8],[[.4]])
    result = fuse_gaussian_likelihood_factors(prior,[factor('one',[a]),factor('two',[b])])
    expected,mass = core_oracle(prior,[a,b])
    np.testing.assert_allclose(result.gaussian.mean,expected.mean,atol=1e-12)
    np.testing.assert_allclose(result.gaussian.covariance,expected.covariance,atol=1e-12)
    assert result.gaussian.mean[2] == 7.
    np.testing.assert_array_equal(result.gaussian.covariance[2,:],np.zeros(3))
    assert result.log_evidence == pytest.approx(mass,abs=1e-12)


def test_deterministic_prior_evaluates_likelihood_without_added_variance():
    prior = Gaussian(np.array([2.,3.]),np.zeros((2,2)))
    a = component('a',[[1.,0.]],[2.5],[[.2]])
    result = fuse_gaussian_likelihood_factors(prior,[factor('one',[a])])
    expected_mass = -.5*(math.log(2*math.pi*.2)+.5**2/.2)
    assert result.gaussian is prior
    assert result.log_evidence == pytest.approx(expected_mass)


def test_underflow_native_mass_is_retained_with_its_finite_log_mass():
    prior = Gaussian(np.zeros(1),np.ones((1,1)))
    result = fuse_gaussian_likelihood_factors(prior,[factor('image',[
        component('strong',[[1.]],[0.],[[1.]]),
        component('tiny',[[1.]],[2.],[[1.]],log_weight=-1000.)])])
    assert len(result.branches) == 2
    assert result.branches[1].weight == 0.
    assert math.isfinite(result.branches[1].log_weight)
    assert len(result.posterior_components) == 2
    assert result.record['exact_branch_count'] == 2


def test_no_factor_and_ineffective_factor_preserve_exact_prior_and_provenance():
    prior = correlated_prior()
    result = fuse_gaussian_likelihood_factors(prior,[factor('missed-image',[])])
    assert result.gaussian is prior
    assert result.log_evidence == 0.
    assert result.record['factor_ids'] == ['missed-image']
    assert result.record['observation_ids'] == ['observation:missed-image']
    assert result.record['effective_factor_count'] == 0


def test_one_hundred_single_mode_factors_have_one_branch_and_order_independent_product():
    prior = Gaussian(np.zeros(2),np.array([[1.,.2],[.2,2.]]))
    factors = [factor(str(i),[component('mode',[[1.,.01*i]],[.1*i],[[.4]])]) for i in range(100)]
    forward = fuse_gaussian_likelihood_factors(prior,factors)
    reverse = fuse_gaussian_likelihood_factors(prior,reversed(factors))
    assert len(forward.branches) == 1
    np.testing.assert_allclose(forward.gaussian.mean,reverse.gaussian.mean,atol=1e-12)
    np.testing.assert_allclose(forward.gaussian.covariance,reverse.gaussian.covariance,atol=1e-12)
    assert forward.log_evidence == pytest.approx(reverse.log_evidence,abs=2e-12)
    assert forward.record['effective_factor_count'] == 100


def test_zero_noise_constraints_use_core_support_conditioning_not_regularization():
    prior = Gaussian(np.zeros(2),np.array([[1.,.4],[.4,2.]]))
    a = component('exact',[[1.,0.]],[.3],[[0.]])
    b = component('noisy',[[0.,1.]],[.8],[[.1]])
    result = fuse_gaussian_likelihood_factors(prior,[factor('one',[a]),factor('two',[b])])
    expected,first = condition_gaussian(prior,[.3],[[0.]],[[1.,0.]])
    expected,second = condition_gaussian(expected,[.8],[[.1]],[[0.,1.]])
    np.testing.assert_array_equal(result.gaussian.mean,expected.mean)
    np.testing.assert_array_equal(result.gaussian.covariance,expected.covariance)
    assert result.log_evidence == first+second
    assert result.record['product_algorithm'].startswith('existing core support')


@pytest.mark.parametrize('mismatch',[dict(state_id='other'),dict(stamp=2.),dict(command=(1.,))])
def test_different_frozen_common_chart_is_a_structural_error(mismatch):
    prior = Gaussian(np.zeros(1),np.ones((1,1)))
    a = component('mode',[[1.]],[0.],[[1.]])
    with pytest.raises(ValueError,match='one frozen common chart'):
        fuse_gaussian_likelihood_factors(prior,[factor('one',[a]),factor('two',[a],**mismatch)])


def test_duplicate_observation_is_not_silently_counted_as_new_information():
    prior = correlated_prior()
    a,b = factor('one',[]),factor('two',[])
    b.observation_ids = a.observation_ids
    with pytest.raises(ValueError,match='observation ID twice'):
        fuse_gaussian_likelihood_factors(prior,[a,b])


def test_repeated_factor_id_is_not_silently_dropped():
    prior = correlated_prior()
    with pytest.raises(ValueError,match='factor IDs'):
        fuse_gaussian_likelihood_factors(prior,[factor('same',[]),factor('same',[])])


def test_nonfinite_native_parameters_are_real_input_failure():
    prior = correlated_prior()
    invalid = SimpleNamespace(label='bad',affine=np.array([[1.,0.,0.]]),offset=np.zeros(1),
        observed=np.array([np.nan]),covariance=np.ones((1,1)),log_weight=0.,source_ids=())
    with pytest.raises(ValueError,match='must be finite'):
        fuse_gaussian_likelihood_factors(prior,[factor('broken',[invalid])])


def test_impossible_zero_noise_observation_does_not_become_zero_information():
    prior = Gaussian(np.array([0.]),np.zeros((1,1)))
    impossible = component('impossible',[[1.]],[1.],[[0.]])
    with pytest.raises(ValueError,match='support'):
        fuse_gaussian_likelihood_factors(prior,[factor('broken',[impossible])])


def test_actual_imu_and_native_marker_batch_matches_unprojected_core_products(tmp_path):
    from legacy_marker import legacy_marker_scene
    from probtf_observation_fusion_demo.batch_observation_factors import build_observation_factor
    from probtf_observation_fusion_demo.moving_inference import MovingInference

    scene = legacy_marker_scene(tmp_path,imu_target_link='tool0')
    inputs = scene.sensor_inputs(scene.step(0.,1000.))
    package = Path(__file__).resolve().parents[1]
    observations = yaml.safe_load((package/'config/imu_marker.yaml').read_text())['observations']
    inference = MovingInference(scene.robot_description,observations,config_path=scene.legacy_inference_config)
    state = inference.prior_at(inputs.command,inputs.stamp,0)
    factors = []
    for template in observations:
        observation = dict(template)
        if observation['sensor'] == 'imu':
            observation.update(sensor_id='imu_tool0',target_link='tool0',gravity_in_root=[0.,0.,-1.],
                gravity_in_sensor=np.asarray(inputs.gravity_in_sensor).tolist(),
                direction_sigma_rad=inputs.imu_sigma_rad)
        factors.append(build_observation_factor(state,state,observation,inference.paths,
                                                inference.common,inputs.image))
    prior = Gaussian(state.mean,state.covariance)
    result = fuse_gaussian_likelihood_factors(prior,factors)
    assert len(factors[1].components) >= 2
    assert len(result.branches) == len(factors[1].components)
    for branch,components in zip(result.branches,product(*(f.components for f in factors))):
        expected,mass = core_oracle(prior,components)
        np.testing.assert_allclose(branch.distribution.mean,expected.mean,rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(branch.distribution.covariance,expected.covariance,rtol=1e-8,atol=1e-10)
        # A single innovation solve independently checks the normalizing
        # constant without subtracting repeated large prior precisions.
        direct,direct_mass = condition_gaussian(prior,
            np.concatenate([c.observed-c.offset for c in components]),
            block_diag(*(c.covariance for c in components)),
            np.vstack([c.affine for c in components]))
        direct_mass += sum(c.log_weight for c in components)
        assert branch.log_evidence == pytest.approx(direct_mass,abs=1e-8)
        np.testing.assert_allclose(branch.distribution.mean,direct.mean,rtol=1e-8,atol=1e-10)
    assert result.record['common_state_id'] == state.state_id
    assert set(result.record['observation_ids']) == {o['observation_id'] for o in observations}


def test_exact_prefix_before_after_laws_and_conditional_masses_match_every_joint_prefix():
    prior = correlated_prior()
    factors = [factor('one',[component('a',[[1.,0.,0.]],[.1],[[.2]],log_weight=math.log(.3)),
                             component('b',[[1.,0.,0.]],[1.2],[[.4]],log_weight=math.log(.7))]),
               factor('two',[component('left',[[0.,1.,1.]],[.5],[[.3]],log_weight=math.log(.6)),
                             component('right',[[0.,1.,1.]],[-.9],[[.2]],log_weight=math.log(.4))]),
               factor('three',[component('last',[[.2,1.,.3]],[.7],[[.3]])])]
    result = fuse_gaussian_likelihood_factors_with_prefixes(prior,factors,source_ids=('batch-source',))
    previous_mass = 0.
    for index,prefix in enumerate(result.prefixes):
        expected = fuse_gaussian_likelihood_factors(prior,factors[:index+1])
        assert prefix.factor is factors[index]
        assert prefix.factor_id == factors[index].factor_id
        assert prefix.observation_ids == factors[index].observation_ids
        assert prefix.cumulative_log_evidence == pytest.approx(expected.log_evidence,abs=1e-12)
        assert prefix.log_evidence == pytest.approx(expected.log_evidence-previous_mass,abs=1e-12)
        assert prefix.prefix_total_log_evidence == prefix.cumulative_log_evidence
        assert len(prefix.posterior_components) == len(expected.branches)
        prior_labels = {item.label for item in prefix.prior_components}
        for child,reference in zip(prefix.posterior_components,expected.branches):
            np.testing.assert_allclose(child.distribution.mean,reference.distribution.mean,atol=1e-12)
            np.testing.assert_allclose(child.distribution.covariance,reference.distribution.covariance,atol=1e-12)
            assert child.log_weight == pytest.approx(reference.log_weight,abs=1e-12)
            assert child.parent_label == child.parent_branch_id
            assert child.parent_label in prior_labels
        if index:
            assert prefix.prior_components is result.prefixes[index-1].posterior_components
            assert prefix.previous_prefix_id == result.prefixes[index-1].prefix_id
        previous_mass = expected.log_evidence
    assert result.record['support_conditioning_call_count'] == 2+4+4
    assert result.record['prefix_approximation_count'] == 0
    assert result.record['final_one_shot_comparison']['mean_max_abs'] < 1e-12
    assert result.record['final_one_shot_comparison']['covariance_max_abs'] < 1e-12
    assert result.record['batch_source_ids'] == ['batch-source']


def test_prefixes_skip_ineffective_factors_and_preserve_bitwise_final_body_fusion():
    prior = correlated_prior()
    factors = [factor('missing',[]),factor('imu',[component('a',[[1.,0.,0.]],[.5],[[.2]])]),
               factor('missing-again',[])]
    expected = fuse_gaussian_likelihood_factors(prior,factors)
    actual = fuse_gaussian_likelihood_factors_with_prefixes(prior,factors)
    assert len(actual.prefixes) == 1
    assert actual.prefixes[0].factor_id == 'imu'
    assert [row['law_identity'] for row in actual.record['factor_links']] == [True,False,True]
    np.testing.assert_array_equal(actual.fusion.gaussian.mean,expected.gaussian.mean)
    np.testing.assert_array_equal(actual.fusion.gaussian.covariance,expected.gaussian.covariance)
    assert actual.fusion.log_evidence == expected.log_evidence
    assert actual.record['factor_links'][-1]['prefix_id'] == actual.prefixes[-1].prefix_id


def test_single_mode_prefix_work_is_linear_and_journal_contains_no_parameter_matrix_history():
    prior = Gaussian(np.zeros(2),np.array([[1.,.2],[.2,2.]]))
    factors = [factor(str(i),[component('mode',[[1.,.01*i]],[.1*i],[[.4]])]) for i in range(100)]
    result = fuse_gaussian_likelihood_factors_with_prefixes(prior,factors,source_ids=tuple(str(i) for i in range(100)))
    assert result.record['support_conditioning_call_count'] == 100
    assert len(result.prefixes) == 100
    assert len({step.prefix_id for step in result.prefixes}) == 100
    for prefix in result.prefixes:
        assert len(prefix.prior_components) == len(prefix.posterior_components) == 1
        assert 'covariance' not in prefix.record and 'mean' not in prefix.record
        assert 'batch_source_ids' not in prefix.record
        assert len(prefix.record['component_links']) == 1
        assert prefix.record['component_links'][0]['source_ids'] == ['native:mode']
    assert len(result.record['batch_source_ids']) == 100


def test_prefix_finite_underflowed_native_component_and_parent_are_never_pruned():
    prior = Gaussian(np.zeros(1),np.ones((1,1)))
    factors = [factor('image',[component('a',[[1.]],[0.],[[.1]]),
        component('tiny',[[1.]],[1.],[[.1]],log_weight=-1000.)]),
        factor('imu',[component('b',[[1.]],[.2],[[.1]])])]
    result = fuse_gaussian_likelihood_factors_with_prefixes(prior,factors)
    assert [len(step.posterior_components) for step in result.prefixes] == [2,2]
    tiny_before = result.prefixes[0].posterior_components[1]
    tiny_after = result.prefixes[1].posterior_components[1]
    assert tiny_before.weight == tiny_after.weight == 0.
    assert math.isfinite(tiny_before.log_weight) and math.isfinite(tiny_after.log_weight)
    assert tiny_after.parent_branch_id == tiny_before.label


def test_empty_prefix_result_preserves_initial_gaussian_and_reports_zero_work():
    prior = correlated_prior()
    result = fuse_gaussian_likelihood_factors_with_prefixes(prior,[factor('missing',[])])
    assert result.prefixes == ()
    assert result.fusion.gaussian is prior
    assert result.record['initial_prefix_id'] == result.record['final_prefix_id']
    assert result.record['support_conditioning_call_count'] == 0
    assert result.record['cumulative_log_evidence'] == 0.


def test_acquisition_prefix_ratios_restore_stiffness_information_lost_at_zero_load_endpoint():
    # theta(q,a)=q-a*tau(q), a=1/K. Reference a=.1; the
    # two acquisitions have (q,tau)=(1,1),(0,0), true a=.2.
    # Frozen command transport shares one body error coordinate, while K
    # must query that coordinate separately at each original command.
    prior = Gaussian(np.zeros(1),np.ones((1,1)))
    factors = [factor('loaded',[component('a',[[1.]],[-.1],[[.0001]])]),
               factor('unloaded',[component('b',[[1.]],[0.],[[.0001]])])]
    result = fuse_gaussian_likelihood_factors_with_prefixes(prior,factors)
    def log_density(distribution,x):
        return -.5*(math.log(2*math.pi*distribution.covariance[0,0])+
                    (x-distribution.mean[0])**2/distribution.covariance[0,0])
    def all_acquisition_ratios(a):
        values = (.1-a,0.)
        return sum(log_density(step.posterior_components[0].distribution,value)-
                   log_density(step.prior_components[0].distribution,value)
                   for step,value in zip(result.prefixes,values))
    assert all_acquisition_ratios(.2)-all_acquisition_ratios(.1) == pytest.approx(50.,abs=1e-10)
    # END alone is independent of a, even though the original loaded
    # observation strongly distinguishes the two stiffness candidates.
    end_ratio = log_density(result.fusion.gaussian,0.)-log_density(prior,0.)
    assert math.isfinite(end_ratio)
    assert end_ratio-end_ratio == 0.


def test_strong_observation_evidence_retains_finite_normalizer_beside_huge_residual_energy():
    prior = Gaussian(np.zeros(1),np.ones((1,1)))
    likelihood = component('strong',[[1.]],[1.],[[1e-20]])
    actual = fuse_gaussian_likelihood_factors(prior,[factor('precise',[likelihood])])
    # Marginal y~N(0,1+R). Completing a square via two O(1/R)
    # terms previously erased the finite log-normalizer, returning -23.0259.
    expected = -.5*(math.log(2*math.pi*(1.+1e-20))+1./(1.+1e-20))
    assert actual.log_evidence == pytest.approx(expected,abs=1e-12)
    assert actual.gaussian.mean[0] == pytest.approx(1.)
    assert actual.gaussian.covariance[0,0] == pytest.approx(1e-20,rel=1e-12,abs=0.)


def test_strong_native_mixture_keeps_correct_bayes_masses_instead_of_equalizing_by_cancellation():
    prior = Gaussian(np.zeros(1),np.ones((1,1)))
    factors = [factor('precise-mixture',[
        component('near',[[1.]],[1.],[[1e-20]],log_weight=math.log(.5)),
        component('far',[[1.]],[2.],[[1e-20]],log_weight=math.log(.5))])]
    actual = fuse_gaussian_likelihood_factors_with_prefixes(prior,factors)
    expected_logs = np.array([math.log(.5)-.5*(math.log(2*math.pi)+value*value)
                             for value in (1.,2.)])
    expected_logs -= logsumexp(expected_logs)
    np.testing.assert_allclose([b.log_weight for b in actual.fusion.branches],expected_logs,atol=1e-12)
    np.testing.assert_allclose([b.log_weight for b in actual.prefixes[0].posterior_components],expected_logs,atol=1e-12)
    assert actual.fusion.branches[0].weight > .8
    assert actual.fusion.branches[1].weight < .2


def test_scalar_likelihood_measure_coefficients_change_mass_without_changing_conditional_law():
    from dataclasses import replace
    prior = correlated_prior()
    mode = component('image',[[1.,.2,0.]],[.7],[[.001]],log_weight=-1000.)
    original = fuse_gaussian_likelihood_factors_with_prefixes(prior,[factor('original',[mode])])
    changed = fuse_gaussian_likelihood_factors_with_prefixes(prior,[factor('scaled',[
        replace(mode,log_weight=mode.log_weight+17.3)])])
    np.testing.assert_array_equal(original.fusion.gaussian.mean,changed.fusion.gaussian.mean)
    np.testing.assert_array_equal(original.fusion.gaussian.covariance,changed.fusion.gaussian.covariance)
    assert changed.fusion.log_evidence-original.fusion.log_evidence == pytest.approx(17.3,abs=1e-12)
    assert changed.prefixes[0].log_evidence-original.prefixes[0].log_evidence == pytest.approx(17.3,abs=1e-12)


def test_huge_equal_gaussian_log_evidence_preserves_native_weights_and_nonzero_body_mean():
    prior = Gaussian(np.zeros(1),np.ones((1,1)))
    first = factor('symmetric-remote-native',[
        component('positive',[[1.]],[1e10],[[1.]],log_weight=math.log(.17)),
        component('negative',[[1.]],[-1e10],[[1.]],log_weight=math.log(.83))])
    # Gaussian overlap logE is the same roughly -2.5e19 for both native
    # hypotheses, while their posterior means differ. Adding log weights
    # before removing that common scale used to turn .17/.83 into .5/.5.
    second = factor('common-negative-evidence',[
        component('constant',[[0.]],[1e10],[[.5]])])
    result = fuse_gaussian_likelihood_factors_with_prefixes(prior,[first,second])
    np.testing.assert_allclose([branch.weight for branch in result.fusion.branches],[.17,.83],atol=1e-14)
    for prefix in result.prefixes:
        np.testing.assert_allclose([branch.weight for branch in prefix.posterior_components],[.17,.83],atol=1e-14)
    assert result.fusion.gaussian.mean[0] == pytest.approx(-3.3e9,rel=1e-14)
    assert result.fusion.gaussian.covariance[0,0] == pytest.approx(.17*.83*1e20+.5,rel=1e-14)
    assert result.prefixes[1].prior_components is result.prefixes[0].posterior_components
    assert result.fusion.log_evidence < -1e20
    assert result.fusion.log_evidence_parts[1] != 0.


def test_compensated_prefix_evidence_transmits_finite_mass_differences_hidden_by_float_total():
    from dataclasses import replace
    prior = Gaussian(np.zeros(1),np.ones((1,1)))
    a = component('a',[[0.]],[1e10],[[.5]])
    b = component('b',[[0.]],[1e10],[[.5]])
    original = fuse_gaussian_likelihood_factors_with_prefixes(prior,[factor('one',[a]),factor('two',[b])])
    changed = fuse_gaussian_likelihood_factors_with_prefixes(prior,[
        factor('one-scaled',[replace(a,log_weight=3.)]),
        factor('two-scaled',[replace(b,log_weight=-2.)])])
    # The legacy scalar remains available, but one float cannot contain
    # these O(1) changes beside a -2e20 log scale.
    assert original.fusion.log_evidence == changed.fusion.log_evidence
    assert log_evidence_parts_difference(changed.fusion.log_evidence_parts,
        original.fusion.log_evidence_parts) == pytest.approx(1.,abs=1e-12)
    assert log_evidence_parts_difference(changed.prefixes[0].cumulative_log_evidence_parts,
        original.prefixes[0].cumulative_log_evidence_parts) == pytest.approx(3.,abs=1e-12)
    assert log_evidence_parts_difference(changed.prefixes[1].cumulative_log_evidence_parts,
        original.prefixes[1].cumulative_log_evidence_parts) == pytest.approx(1.,abs=1e-12)
