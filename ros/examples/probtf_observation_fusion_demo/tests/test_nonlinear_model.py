"""Fixed native charts, finite derivatives and direct joint result identity."""
from copy import deepcopy
from dataclasses import replace
import math
import pickle

import numpy as np
import pytest

from probtf.geometry import tangent_basis,quat_to_rotmat
from probtf_observation_fusion_demo.body_native import pose_query
from probtf_observation_fusion_demo.bayes_factor import transition_priors
from probtf_observation_fusion_demo.finite_modal import finite_modal_path
from probtf_observation_fusion_demo.model import WeightedState
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.nonlinear_model import run,JointPosteriorSensorView
from probtf_observation_fusion_demo.nonlinear_observation import prepare_likelihood,sphere_log_residual,SphereLogCut
from test_joint_batch_inference import acquired


@pytest.fixture
def problem(acquired):
    description,inputs,templates=acquired
    inference=MovingInference(description,templates)
    prior=inference.prior_at(inputs.command,inputs.stamp,0)
    observations=[]
    for template in templates:
        observation=deepcopy(template)
        observation['observation_id']='original:'+observation['sensor']
        if observation['sensor']=='imu':
            observation.update(target_link=inputs.imu_target_link,gravity_in_root=[0.,0.,-1.],
                gravity_in_sensor=inputs.gravity_in_sensor.tolist(),direction_sigma_rad=inputs.imu_sigma_rad)
        observations.append(observation)
    return inference,prior,inputs,observations


@pytest.mark.parametrize('angle',[0.,1e-8,1e-4,.3,1.8])
def test_sphere_log_is_angular_and_analytic_derivative_matches_deterministic_difference(angle):
    observed=np.array([0.,0.,1.]);basis=tangent_basis(observed)
    direction=np.array([math.sin(angle),0.,math.cos(angle)])
    residual,jacobian=sphere_log_residual(direction,observed,basis)
    assert np.linalg.norm(residual)==pytest.approx(angle,abs=2e-15)
    numerical=np.column_stack([(sphere_log_residual(direction+1e-7*unit,observed,basis)[0]
        -sphere_log_residual(direction-1e-7*unit,observed,basis)[0])/(2e-7)for unit in np.eye(3)])
    np.testing.assert_allclose(jacobian,numerical,atol=2e-8,rtol=2e-7)
    np.testing.assert_allclose(jacobian@direction,np.zeros(2),atol=2e-15)


def test_antipodal_axis_is_true_cut_and_near_antipodal_finite_candidates_are_not_rejected():
    observed=np.array([0.,0.,1.]);basis=tangent_basis(observed)
    with pytest.raises(SphereLogCut,match='scalar angular likelihood is finite'):
        sphere_log_residual(-observed,observed,basis)
    residual,jacobian=sphere_log_residual(np.array([1e-10,0.,-1.]),observed,basis)
    assert np.isfinite(residual).all() and np.isfinite(jacobian).all()
    assert np.linalg.norm(residual)==pytest.approx(math.pi-1e-10)



@pytest.mark.parametrize('observed',[
    np.array([.706013483277297,-.005795629970420448,-.7081746762656118]),
    np.array([.37,-.61,.70]),
])
def test_exact_nonaxis_antipode_is_a_cut_despite_rounded_tangent_basis(observed):
    observed=observed/np.linalg.norm(observed)
    basis=tangent_basis(observed)
    antipodal=-observed
    np.testing.assert_array_equal(np.cross(antipodal,observed),np.zeros(3))
    # Rounded basis construction need not make its projection exactly zero.
    with pytest.raises(SphereLogCut,match='scalar angular likelihood is finite'):
        sphere_log_residual(antipodal,observed,basis)
    nearby=antipodal+1e-10*basis[:,0]
    assert np.any(np.cross(nearby,observed)!=0.)
    residual,jacobian=sphere_log_residual(nearby,observed,basis)
    assert np.isfinite(residual).all() and np.isfinite(jacobian).all()
    assert np.linalg.norm(residual)==pytest.approx(math.pi-1e-10,abs=2e-12)


def test_native_marker_fixed_chart_preserves_covariance_and_finite_path_derivative(problem):
    inference,prior,inputs,observations=problem
    observation=next(item for item in observations if item['sensor']=='marker')
    prepared=prepare_likelihood(inference.common,observation,inference.paths,inputs.image,prior.stamp)
    assert prepared.components
    delta=np.zeros(prior.mean.size);delta[3:6]=[.02,-.01,.015];delta[9:12]=[-.008,.01,.005]
    for component in prepared.components:
        residual,jacobian=component.evaluate(prior,delta)
        numerical=np.column_stack([(component.evaluate(prior,delta+1e-7*unit)[0]
            -component.evaluate(prior,delta-1e-7*unit)[0])/(2e-7)for unit in np.eye(delta.size)])
        np.testing.assert_allclose(jacobian,numerical,atol=2e-8,rtol=3e-6)
        geometry=finite_modal_path(prior,component.path,delta)
        np.testing.assert_array_equal(residual[:3],geometry.translation-component.translation)
        assert component.record()['point_dependent_chart_volume_applied'] is False
    component=prepared.components[0]
    residual,_=component.evaluate(prior,np.zeros_like(delta))
    geometry=finite_modal_path(prior,component.path,np.zeros_like(delta))
    query=pose_query(prior,component.path)
    np.testing.assert_array_equal(residual[:3],geometry.translation-component.translation)
    assert np.linalg.norm(query.mean[:3]-geometry.translation)>1e-6


def test_imu_fixed_sphere_likelihood_analytic_joint_derivative(problem):
    inference,prior,inputs,observations=problem
    observation=next(item for item in observations if item['sensor']=='imu')
    likelihood=prepare_likelihood(inference.common,observation,inference.paths,inputs.image,prior.stamp).components[0]
    delta=np.zeros(prior.mean.size);delta[3:6]=[.02,-.01,.015]
    residual,jacobian=likelihood.evaluate(prior,delta)
    numerical=np.column_stack([(likelihood.evaluate(prior,delta+1e-7*unit)[0]
        -likelihood.evaluate(prior,delta-1e-7*unit)[0])/(2e-7)for unit in np.eye(delta.size)])
    np.testing.assert_allclose(jacobian,numerical,atol=2e-8,rtol=2e-6)
    geometry=finite_modal_path(prior,likelihood.path,delta)
    predicted=quat_to_rotmat(geometry.rotation_wxyz)@likelihood.root_direction
    angle=math.atan2(np.linalg.norm(np.cross(predicted,likelihood.observed_direction)),float(predicted@likelihood.observed_direction))
    assert np.linalg.norm(residual)==pytest.approx(angle,abs=1e-14)


def test_joint_update_prepares_image_once_and_never_re_lifts_local_posteriors(problem,monkeypatch):
    import probtf_observation_fusion_demo.model as affine
    import probtf_observation_fusion_demo.nonlinear_observation as sensor
    inference,prior,inputs,observations=problem
    parents,_=transition_priors(WeightedState(prior.component_id,1.,prior,0.),prior,.25)
    original=sensor.prepare_marker_observation;calls=[]
    def prepare(*args):
        calls.append(args);return original(*args)
    def forbidden(*args,**kwargs):
        raise AssertionError('Direct joint nonlinear posterior must not enter the old local Gaussian lift.')
    monkeypatch.setattr(sensor,'prepare_marker_observation',prepare)
    monkeypatch.setattr(affine,'lift_sensor_posterior',forbidden)
    result=run(inference.common,observations,inputs.image,prior_components=parents,paths=inference.paths,joint_evidence=True)
    assert len(calls)==1
    assert len(result.frozen_body_branches)==2
    assert result.records['observation_ids']==[item['observation_id']for item in observations]
    assert result.frozen_body_posterior.individuals==()
    assert result.records['local_gaussian_lift_applied'] is False
    marker=[item for item in result.local_estimates if item.detector_result is not None]
    assert len(marker)==2 and marker[0].detector_result is marker[1].detector_result
    assert all(isinstance(item.posterior,JointPosteriorSensorView)for item in result.local_estimates)
    assert sum(item.weight for item in result.frozen_body_posterior.components)==pytest.approx(1.,abs=2e-14)
    for branch,parent in zip(result.frozen_body_branches,parents):
        assert branch.reference_state is parent.state
        np.testing.assert_array_equal(branch.prior.covariance,parent.state.covariance)
        assert branch.reference_state.command==tuple(inputs.command) and branch.reference_state.stamp==inputs.stamp
        assert all(not child.distribution.covariance.flags.writeable for child in branch.posterior_components)
    for entry in result.records['nonlinear_observation_update']['branch_diagnostics']:
        assert entry['original_prior_held_fixed'] and not entry['repeated_observation_multiplication']
        assert entry['observation_dimension']==8
        assert entry['final_objective']<=entry['initial_objective']
    assert result.records['joint_observation_evidence']['original_joint_prior_used_once']
    diagnostics=result.records['nonlinear_observation_update']
    assert deepcopy(diagnostics) is diagnostics
    with pytest.raises(TypeError):
        diagnostics['branch_diagnostics'][0]['final_objective']=0.
    restored=pickle.loads(pickle.dumps(diagnostics))
    assert restored==diagnostics
    with pytest.raises(TypeError):
        restored['branch_diagnostics'].append({})
    # The caller still needs to attach the event's original acquisition IDs.
    result.records['caller_metadata']={'original_observation_ids':list(result.records['observation_ids'])}


def test_log_mass_underflow_does_not_remove_native_likelihood_branch(problem,monkeypatch):
    import probtf_observation_fusion_demo.nonlinear_model as nonlinear
    inference,prior,inputs,observations=problem
    observation=next(item for item in observations if item['sensor']=='marker')
    prepared=prepare_likelihood(inference.common,observation,inference.paths,inputs.image,prior.stamp)
    original=prepared.components[0]
    controlled=replace(prepared,components=(replace(original,label='dominant',log_weight=0.),replace(original,label='finite-tiny',log_weight=-1000.)))
    monkeypatch.setattr(nonlinear,'prepare_likelihood',lambda *args:controlled)
    result=run(inference.common,[observation],inputs.image,prior=prior,paths=inference.paths,joint_evidence=True)
    assert len(result.frozen_body_posterior.components)==2
    weak=next(item for item in result.frozen_body_posterior.components if 'finite-tiny'in item.label)
    assert weak.weight==0. and weak.log_weight==pytest.approx(-1000.,abs=2e-10)
    assert len(result.records['nonlinear_observation_update']['branch_diagnostics'])==2


def test_one_synchronous_product_is_exactly_order_invariant_without_reordering_parents(problem):
    inference,prior,inputs,observations=problem
    parents,_=transition_priors(WeightedState(prior.component_id,1.,prior,0.),prior,.25)
    parents=parents[::-1]
    first=run(inference.common,observations,inputs.image,prior_components=parents,
        paths=inference.paths,joint_evidence=True)
    reordered=run(inference.common,observations[::-1]+observations[:1],inputs.image,
        prior_components=parents,paths=inference.paths,joint_evidence=True)
    assert first.records==reordered.records
    expected_ids=tuple(sorted(item['observation_id']for item in observations))
    assert first.frozen_body_posterior.observation_ids==expected_ids
    assert tuple(branch.label for branch in first.frozen_body_branches)==tuple(parent.label for parent in parents)
    for original,permuted in zip(first.frozen_body_posterior.components,reordered.frozen_body_posterior.components):
        assert original.label==permuted.label and original.log_weight==permuted.log_weight
        assert original.lineage==expected_ids
        np.testing.assert_array_equal(original.distribution.mean,permuted.distribution.mean)
        np.testing.assert_array_equal(original.distribution.covariance,permuted.distribution.covariance)


def test_empty_or_missing_observation_preserves_the_same_source_law(problem):
    inference,prior,inputs,observations=problem
    empty=run(inference.common,(),inputs.image,prior=prior,paths=inference.paths)
    assert empty.stages[-1].components[0].state is prior and not empty.sensor_posteriors
    observation=next(item for item in observations if item['sensor']=='marker')
    missing=run(inference.common,[observation],np.full_like(inputs.image,255),prior=prior,paths=inference.paths)
    assert missing.stages[-1].components[0].state is prior
    assert missing.records['observation_ids']==[] and not missing.sensor_posteriors
    assert missing.local_estimates[0].posterior is None
    assert deepcopy(empty.records['nonlinear_observation_update']) is empty.records['nonlinear_observation_update']
    assert deepcopy(missing.records['nonlinear_observation_update']) is missing.records['nonlinear_observation_update']
