"""One-pass core propagation, fixed native charts, and common-source laws."""
from copy import deepcopy
from dataclasses import replace
import math
import pickle
import numpy as np
import pytest
from scipy.linalg import block_diag
from probtf.geometry import quat_mul,rotation_vector_to_quaternion,tangent_basis,quat_to_rotmat
from probtf.spherical_law.fusion import Gaussian,condition_gaussian
from probtf_observation_fusion_demo.bayes_factor import transition_priors
from probtf_observation_fusion_demo.model import WeightedState
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moment_model import run,MomentPosteriorSensorView
from probtf_observation_fusion_demo.moment_observation import prepare_observation
from test_joint_batch_inference import acquired


@pytest.fixture
def problem(acquired):
    description,inputs,templates=acquired
    inference=MovingInference(description,templates)
    prior=inference.prior_at(inputs.command,inputs.stamp,0)
    observations=[]
    for template in templates:
        observation=deepcopy(template);observation['observation_id']='original:'+observation['sensor']
        if observation['sensor']=='imu':
            observation.update(target_link=inputs.imu_target_link,gravity_in_root=[0.,0.,-1.],
                gravity_in_sensor=inputs.gravity_in_sensor.tolist(),direction_sigma_rad=inputs.imu_sigma_rad)
        observations.append(observation)
    return inference,prior,inputs,observations


def test_marker_transports_actual_core_corrected_mean_covariance_and_full_cross_to_native_chart(problem):
    inference,prior,inputs,observations=problem
    observation=next(x for x in observations if x['sensor']=='marker')
    prepared=prepare_observation(inference.common,observation,inference.paths,inputs.image,prior.stamp)
    assert prepared.components
    for item in prepared.propagate(prior):
        source=item.source_query;q=item.query;M=item.chart_matrix;offset=item.chart_offset
        np.testing.assert_array_equal(q.mean,M@source.mean+offset)
        np.testing.assert_array_equal(q.covariance,M@source.covariance@M.T)
        np.testing.assert_array_equal(q.cross_covariance,source.cross_covariance@M.T)
        np.testing.assert_array_equal(q.jacobian,M@source.jacobian)
        np.testing.assert_array_equal(q.within,M@source.within@M.T)
        np.testing.assert_array_equal(q.between,M@source.between@M.T)
        np.testing.assert_array_equal(q.rotation_wxyz,source.rotation_wxyz)
        assert np.linalg.norm(q.mean-q.mode)>1e-6
        assert q.state_id==prior.state_id and q.stamp==inputs.stamp and q.command==tuple(inputs.command)
        assert item.record()['additional_chart_Hessian_mean_correction_added'] is False
        assert item.likelihood.record()['point_dependent_chart_volume_applied'] is False


def test_imu_uses_core_direction_moments_in_one_shared_observed_fixed_plane(problem):
    inference,prior,inputs,observations=problem
    observation=next(x for x in observations if x['sensor']=='imu')
    prepared=prepare_observation(inference.common,observation,inference.paths,inputs.image,prior.stamp)
    item=prepared.propagate(prior)[0];native=item.likelihood
    B=tangent_basis(native.observed_direction)
    np.testing.assert_array_equal(native.basis,B)
    np.testing.assert_array_equal(item.query.mean,B.T@item.source_query.mean)
    np.testing.assert_array_equal(item.query.covariance,B.T@item.source_query.covariance@B)
    np.testing.assert_array_equal(item.query.cross_covariance,item.source_query.cross_covariance@B)
    np.testing.assert_array_equal(native.observed_mean,B.T@native.observed_direction)
    assert prepared.diagnostics['point_S2_Log_likelihood_used'] is False
    assert not prepared.diagnostics['heading_observation']
    assert np.linalg.norm(item.source_query.correction)>1e-6


def test_joint_update_is_one_condition_per_parent_native_combination_and_no_point_solver_or_old_lift(problem,monkeypatch):
    import probtf_observation_fusion_demo.moment_model as module
    import probtf_observation_fusion_demo.moment_observation as sensor
    import probtf_observation_fusion_demo.model as old_model
    import probtf_observation_fusion_demo.nonlinear_model as old_nonlinear
    import probtf_observation_fusion_demo.nonlinear_conditioning as old_solver
    import probtf_observation_fusion_demo.finite_modal as finite_geometry
    inference,prior,inputs,observations=problem
    parents,_=transition_priors(WeightedState(prior.component_id,1.,prior,0.),prior,.25)
    prepare=sensor.prepare_marker_observation;condition=module.condition_moments
    preparations=[];conditionings=[]
    def prepare_once(*args):preparations.append(args);return prepare(*args)
    def condition_once(*args):
        result=condition(*args);conditionings.append((args,result));return result
    def forbidden(*args,**kwargs):raise AssertionError('No point optimization or old local posterior lift is allowed.')
    monkeypatch.setattr(sensor,'prepare_marker_observation',prepare_once)
    monkeypatch.setattr(module,'condition_moments',condition_once)
    monkeypatch.setattr(old_model,'lift_sensor_posterior',forbidden)
    monkeypatch.setattr(old_nonlinear,'run',forbidden)
    monkeypatch.setattr(old_solver,'condition_nonlinear',forbidden)
    monkeypatch.setattr(finite_geometry,'finite_modal_path',forbidden)
    result=run(inference.common,observations,inputs.image,prior_components=parents,
        paths=inference.paths,joint_evidence=True)
    update=result.records['high_concentration_observation_update']
    assert len(preparations)==1
    assert len(conditionings)==len(result.frozen_body_posterior.components)
    assert not update['point_optimization_used'] and update['optimization_iterations']==0
    assert update['core_O_P_mean_used'] and not update['gauss_newton_used']
    assert not update['laplace_body_evidence_used']
    assert all(isinstance(x.posterior,MomentPosteriorSensorView)for x in result.local_estimates)
    markers=[x for x in result.local_estimates if x.detector_result is not None]
    assert markers[0].detector_result is markers[1].detector_result
    for (args,conditioned),raw in zip(conditionings,result.frozen_body_posterior.components):
        prior_G,mu,S,C,H,y,R=args
        expected,evidence=condition_gaussian(prior_G,y-mu+H@prior_G.mean,R,H)
        np.testing.assert_allclose(raw.distribution.mean,expected.mean,atol=3e-12,rtol=2e-12)
        np.testing.assert_allclose(raw.distribution.covariance,expected.covariance,atol=3e-12,rtol=2e-12)
        assert conditioned.log_evidence==pytest.approx(evidence,abs=3e-10)
        assert np.linalg.norm(S[:2,2:])>1e-6
        np.testing.assert_allclose(C,prior_G.covariance@H.T,atol=1e-15)
    for branch,parent in zip(result.frozen_body_branches,parents):
        assert branch.reference_state is parent.state
        np.testing.assert_array_equal(branch.prior.covariance,parent.state.covariance)
        assert branch.reference_state.stamp==inputs.stamp and branch.reference_state.command==tuple(inputs.command)
    assert math.fsum(x.weight for x in result.frozen_body_posterior.components)==pytest.approx(1.,abs=2e-14)
    assert deepcopy(update)is update
    assert pickle.loads(pickle.dumps(update))==update
    with pytest.raises(TypeError):update['branch_diagnostics'][0]['optimization_iterations']=1
    result.records['acquisition_metadata']={'original_stamp':inputs.stamp}


def test_distinct_native_modes_and_underflow_mass_all_survive_without_changing_fixed_likelihoods(problem,monkeypatch):
    import probtf_observation_fusion_demo.moment_model as module
    inference,prior,inputs,observations=problem
    observation=next(x for x in observations if x['sensor']=='marker')
    prepared=prepare_observation(inference.common,observation,inference.paths,inputs.image,prior.stamp)
    first=prepared.components[0]
    second=replace(first,label='different_pose',observed_mean=first.observed_mean+np.r_[.01,-.008,.005,0.,0.,0.],
        rotation_wxyz=quat_mul(first.rotation_wxyz,rotation_vector_to_quaternion([.02,-.01,.015])),log_weight=math.log(.3))
    controlled=replace(prepared,components=(replace(first,label='first',log_weight=math.log(.7)),second,
        replace(second,label='finite_tiny',log_weight=-1000.)))
    monkeypatch.setattr(module,'prepare_observation',lambda *args:controlled)
    result=run(inference.common,[observation],inputs.image,prior=prior,paths=inference.paths,joint_evidence=True)
    raw=result.frozen_body_posterior.components
    assert len(raw)==3
    assert not np.array_equal(raw[0].distribution.mean,raw[1].distribution.mean)
    tiny=next(x for x in raw if 'finite_tiny'in x.label)
    assert tiny.weight==0. and math.isfinite(tiny.log_weight)
    native_records=result.records['high_concentration_observation_update']['native_likelihoods']
    assert native_records==[x.record()for x in controlled.components]
    assert [x['native_component_labels'][0]for x in result.records['high_concentration_observation_update']['branch_diagnostics']]==['first','different_pose','finite_tiny']


def test_synchronous_product_is_bitwise_commutative_but_keeps_parent_and_native_order(problem):
    inference,prior,inputs,observations=problem
    parents,_=transition_priors(WeightedState(prior.component_id,1.,prior,0.),prior,.25);parents=parents[::-1]
    first=run(inference.common,observations,inputs.image,prior_components=parents,paths=inference.paths,joint_evidence=True)
    other=run(inference.common,observations[::-1]+observations[:1],inputs.image,prior_components=parents,paths=inference.paths,joint_evidence=True)
    assert first.records==other.records
    assert first.frozen_body_posterior.observation_ids==tuple(sorted(x['observation_id']for x in observations))
    assert tuple(x.label for x in first.frozen_body_branches)==tuple(x.label for x in parents)
    for a,b in zip(first.frozen_body_posterior.components,other.frozen_body_posterior.components):
        assert a.label==b.label and a.log_weight==b.log_weight
        np.testing.assert_array_equal(a.distribution.mean,b.distribution.mean)
        np.testing.assert_array_equal(a.distribution.covariance,b.distribution.covariance)


def test_empty_and_missing_image_preserve_exact_prior_and_no_observation_evidence(problem):
    inference,prior,inputs,observations=problem
    empty=run(inference.common,(),inputs.image,prior=prior,paths=inference.paths)
    observation=next(x for x in observations if x['sensor']=='marker')
    missing=run(inference.common,[observation],np.full_like(inputs.image,255),prior=prior,paths=inference.paths,joint_evidence=True)
    for result in (empty,missing):
        assert result.stages[-1].components[0].state is prior
        assert not result.sensor_posteriors and result.records['valid_observation_count']==0
        assert 'joint_observation_evidence'not in result.records
    assert missing.local_estimates[0].posterior is None
