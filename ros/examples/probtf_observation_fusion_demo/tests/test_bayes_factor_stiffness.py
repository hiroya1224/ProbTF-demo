"""Stiffness uses the displayed body law over the complete change prior."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
)
from probtf_observation_fusion_demo.body_native import reexpress_state
from probtf.spherical_law.fusion import Gaussian, PosteriorComponent
from probtf_observation_fusion_demo.model import FrozenBodyBranch, WeightedState, Stage
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.stiffness_identification import (
    ChartedBodyDensityRatio, StiffnessIdentifier,
)


def _state(vector, variance, component):
    xyzw = Rotation.from_rotvec(vector).as_quat()
    quaternion = np.r_[xyzw[3], xyzw[:3]]
    covariance = np.diag([.02, .03, .025, variance, variance*.8, variance*1.2,
                          .4, .35, .3, .08, .09, .07])
    covariance[0, 4] = covariance[4, 0] = .002
    covariance[1, 7] = covariance[7, 1] = .01
    edge = BinghamTransform('joint', np.zeros(3),
        bingham_from_rotation_coordinates(quaternion, covariance[3:6, 3:6]),
        covariance[:3, :3])
    marker = BinghamTransform('unknown_marker', np.array([.7, -.2, .3]),
        bingham_from_rotation_coordinates([1., 0., 0., 0.], covariance[9:, 9:]),
        covariance[6:9, 6:9])
    return JointTransformState((edge, marker), covariance, command=(.12,),
                               stamp=1234., version=4, component_id=component)


def _laws():
    keep = _state([.25, -.12, .08], .012, 'keep')
    reset = _state([-.32, .18, -.09], .055, 'reset')
    branches = []
    for index, (weight, state) in enumerate(((.75, keep), (.25, reset))):
        prior = Gaussian(state.mean, state.covariance)
        # Native observation children intentionally differ from the final
        # declared common-chart projection. They cannot become its numerator.
        mean = prior.mean.copy(); mean[3] = (.08, -.11)[index]
        child = PosteriorComponent('raw-'+str(index),
            Gaussian(mean, prior.covariance*.7), (.6, .4)[index])
        branches.append(FrozenBodyBranch(state.component_id, weight, state, prior, (child,)))
    mean = keep.mean.copy(); mean[:6] = [.01, -.02, .005, .09, -.03, .015]
    covariance = keep.covariance*.85
    covariance[2, 6] = covariance[6, 2] = .006
    child = PosteriorComponent('canonical-body', Gaussian(mean, covariance))
    projected = FrozenBodyBranch('projected-body-numerator', 1., keep,
        branches[0].prior, (child,))
    return tuple(branches), projected


def _physical_values(branches, x):
    physical = Rotation.from_rotvec([x[0], x[1], .1])
    values = []
    for branch in branches:
        quaternion = branch.reference_state.transforms[0].rotation_wxyz
        reference = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])
        values.append(np.r_[np.array([.01, -.004, .002]),
                            (reference.inv()*physical).as_rotvec()])
    return values


def _haar_log_density(value, distribution):
    # Independent coordinate-density conversion, including its nonzero
    # derivative. All rotation charts use the same physical Haar measure.
    angle = np.linalg.norm(value[3:6])
    log_volume = 2.*np.log(np.sin(angle/2.)/(angle/2.)) if angle else 0.
    return multivariate_normal.logpdf(value, distribution.mean[:6],
                                      distribution.covariance[:6, :6])-log_volume


def test_projected_body_ratio_uses_full_native_transition_prior_and_haar_gradient():
    branches, projected = _laws()
    ratio = ChartedBodyDensityRatio(branches, ('joint',), posterior_branch=projected)
    def expected(x, tight_only=False):
        values = _physical_values(branches, x)
        denominator = [np.log(branch.weight)+_haar_log_density(value, branch.prior)
                       for branch, value in zip(branches, values)]
        if tight_only:
            denominator = [_haar_log_density(values[0], branches[0].prior)]
        return (_haar_log_density(values[0], projected.posterior_components[0].distribution)
                -logsumexp(denominator))
    x = np.array([-.12, .13])
    values = _physical_values(branches, x)
    jacobians = [np.column_stack([(_physical_values(branches, x+1e-6*axis)[index]
                     -_physical_values(branches, x-1e-6*axis)[index])/(2e-6)
                    for axis in np.eye(2)]) for index in range(2)]
    actual, gradient = ratio.evaluate(tuple(zip(values, jacobians)))
    assert actual == pytest.approx(expected(x), abs=1e-12)
    numeric = [(expected(x+1e-6*axis)-expected(x-1e-6*axis))/(2e-6)
               for axis in np.eye(2)]
    np.testing.assert_allclose(gradient, numeric, atol=1e-8, rtol=2e-7)
    assert abs(actual-expected(x, tight_only=True)) > .1
    raw_ratio = ChartedBodyDensityRatio(branches, ('joint',))
    assert abs(actual-raw_ratio.evaluate(tuple(zip(values, jacobians)))[0]) > .1
    assert [branch.weight for branch in ratio.branches] == [.75, .25]
    assert [len(children) for _, children, _ in ratio.marginals] == [1, 0]
    record = ratio.record()
    assert 'transition-prior mixture' in record['denominator_semantics']
    assert 'not conditional keep-branch' in record['numerator_semantics']
    assert record['projected_numerator_source_state_id'] == branches[0].reference_state.state_id
    assert record['branches'][1]['components'] == []


def _result():
    branches, projected = _laws()
    reference = projected.reference_state
    child = projected.posterior_components[0]
    displayed = reexpress_state(reference, child.distribution.mean,
        child.distribution.covariance, version=5, component_id=child.label)
    return SimpleNamespace(prior=reference, frozen_body_prior=branches[0].prior,
        frozen_body_branches=branches, projected_body_numerator=projected,
        frozen_body_posterior=SimpleNamespace(components=(child,),
            prior_version=reference.version, stamp=reference.stamp,
            command=reference.command, observation_ids=('new-marker',)),
        stages=(Stage('fused', (WeightedState(child.label, 1., displayed),)),),
        records=dict(observation_ids=['new-marker'], internal_projection=dict(
            internal_state_id=displayed.state_id, source_prior_state_id=reference.state_id,
            approximation='declared full joint moment approximation')))


def _identifier():
    # Isolate the adapter and exact density law from auxiliary URDF numerical
    # continuation, which has its own physical integration tests.
    identifier = StiffnessIdentifier.__new__(StiffnessIdentifier)
    identifier.body_edge_ids = ('joint',)
    identifier.components = (PosteriorComponent('K-prior', Gaussian([0.], [[1.]])),)
    identifier.initial_components = identifier.components
    identifier.factors = ()
    identifier.factor_cache = {}
    identifier.observation_ids = frozenset()
    identifier.prior_source = {}
    identifier.model_source = {}
    identifier.last_stamp = identifier.last_key = identifier.last_record = None
    identifier._branch_anchor = None
    identifier._freeze_branch = lambda state, record: None
    identifier._properness_bounds = lambda ratio: dict(verified=True,
        log_factor_lower_bound=-100., log_factor_upper_bound=100.)
    return identifier


def test_adapter_records_canonical_display_body_but_keeps_both_native_denominators():
    result = _result()
    identifier = _identifier()
    record = identifier.update(result)
    assert record['status'] == 'updated_density_factor', record.get('failure_reason')
    assert record['body_posterior_state_ids'] == [result.stages[-1].components[0].state.state_id]
    assert record['body_posterior_labels'] == ['canonical-body']
    assert record['body_posterior_weights'] == [1.]
    assert record['source_prior_weights'] == [.75, .25]
    assert record['source_prior_state_ids'] == [item.reference_state.state_id
                                               for item in result.frozen_body_branches]
    factor = identifier.factors[0]
    assert factor.states == tuple(item.reference_state for item in result.frozen_body_branches)
    assert factor.ratio.posterior_branch is result.projected_body_numerator
    assert 'transition-prior mixture' in record['body_internal_approximation']['ratio_semantics']
    assert identifier.update(result) is record
    assert len(identifier.factors) == 1


@pytest.mark.parametrize('corruption', ['raw_posterior', 'projected_child', 'display',
                                       'reset_prior', 'projected_reference'])
def test_adapter_rejects_mismatched_display_or_transition_prior_sources(corruption):
    result = _result()
    if corruption in ('raw_posterior', 'projected_child'):
        child = result.projected_body_numerator.posterior_components[0]
        mean = child.distribution.mean.copy(); mean[3] += .001
        foreign = replace(child, distribution=Gaussian(mean, child.distribution.covariance))
        if corruption == 'raw_posterior':
            result.frozen_body_posterior.components = (foreign,)
        else:
            result.projected_body_numerator = replace(result.projected_body_numerator,
                                                      posterior_components=(foreign,))
    elif corruption == 'display':
        displayed = result.stages[-1].components[0]
        reference = result.prior
        foreign = replace(displayed, state=reexpress_state(reference,
            reference.mean, reference.covariance, version=5, component_id=displayed.label))
        result.stages = (replace(result.stages[-1], components=(foreign,)),)
    else:
        if corruption == 'reset_prior':
            branch = result.frozen_body_branches[1]
            foreign = replace(branch, prior=Gaussian(branch.prior.mean+.01,
                                                     branch.prior.covariance))
            result.frozen_body_branches = (result.frozen_body_branches[0], foreign)
        else:
            branch = result.projected_body_numerator
            result.projected_body_numerator = replace(branch,
                prior=Gaussian(branch.prior.mean+.01, branch.prior.covariance))
    with pytest.raises(ValueError, match='provenance differ|do not reexpress|does not match'):
        _identifier().update(result)


def test_no_observation_legacy_identity_never_queries_an_auxiliary_model():
    result = _result()
    reference = result.prior
    child = PosteriorComponent(reference.component_id, Gaussian(reference.mean,
                                                               reference.covariance))
    result.frozen_body_branches = (FrozenBodyBranch(reference.component_id, 1.,
        reference, child.distribution, (child,)),)
    result.projected_body_numerator = None
    result.frozen_body_posterior.components = (child,)
    result.frozen_body_posterior.observation_ids = ()
    result.records = dict(observation_ids=[])
    result.stages = (Stage('fused', (WeightedState(child.label, 1., reference),)),)
    identifier = _identifier()
    def forbidden(*args):
        raise AssertionError('An identity update must not query auxiliary mechanics.')
    identifier._freeze_branch = forbidden
    identifier._properness_bounds = forbidden
    record = identifier.update(result)
    assert record['status'] == 'identity_density_ratio'
    assert record['prior'] == record['posterior']
    assert record['identity_preserved_bitwise']
    assert identifier.factors == ()


def test_projected_numerator_cannot_silently_choose_a_new_reference_chart():
    branches, projected = _laws()
    foreign = _state([.5, -.1, .1], .02, 'foreign')
    projected = replace(projected, reference_state=foreign,
                        prior=Gaussian(foreign.mean, foreign.covariance))
    with pytest.raises(ValueError, match='actual transition-prior chart'):
        ChartedBodyDensityRatio(branches, ('joint',), posterior_branch=projected)


@pytest.mark.parametrize('case', ['imu', 'marker', 'imu_marker'])
def test_actual_sensor_update_produces_proper_ratio_for_the_same_canonical_body(tmp_path, case):
    package = Path(__file__).resolve().parents[1]
    repo = package.parents[2]
    scene = MovingSwitchScene(tmp_path/case)
    frame = scene.step(0., 1000.)
    observations = tuple(yaml.safe_load((package/'config'/f'{case}.yaml').read_text())['observations'])
    inference = MovingInference(scene.robot_description, observations)
    result = inference.update(scene.sensor_inputs(frame), 0)
    assert [branch.weight for branch in result.frozen_body_branches] == [.75, .25]
    assert result.projected_body_numerator is not None
    assert len(result.stages[-1].components) == 1
    identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, dict(
        prior_config_path=str(repo/'ros/examples/deflecomp/deflecomp_ros/config/estimator.yaml')))
    record = identifier.update(result)
    assert record['status'] == 'updated_density_factor', record.get('failure_reason')
    assert record['new_factor_properness']['verified']
    assert record['body_posterior_state_ids'] == [result.stages[-1].components[0].state.state_id]
    assert identifier.factors[0].ratio.posterior_branch is result.projected_body_numerator
    density, gradient = identifier.log_density(identifier.initial_components[0].distribution.mean)
    assert np.isfinite(density) and np.all(np.isfinite(gradient))
    assert abs(gradient[0]) < 1e-8
