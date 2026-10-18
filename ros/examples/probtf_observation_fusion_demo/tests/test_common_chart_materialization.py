"""A common geometric chart is not a change hypothesis's probability prior."""
from dataclasses import replace
import math

import numpy as np
import pytest

from probtf.geometry import quat_to_rotmat
from probtf.spherical_law.fusion import FusedMixture, Gaussian, PosteriorComponent
from probtf.spherical_law.support import covariance_support, require_in_support
from probtf_observation_fusion_demo.bayes_factor import (
    common_chart_gaussian, projection_input, transition_priors,
)
from probtf_observation_fusion_demo.body_projection import project_update
from probtf_observation_fusion_demo.immutable_body_cache import (
    materialize_state_in_chart, reexpress_state,
)
from probtf_observation_fusion_demo.model import DemoResult, FrozenBodyBranch, Stage, WeightedState
from probtf_observation_fusion_demo.stiffness_identification import ChartedBodyDensityRatio, StiffnessIdentifier
from probtf_observation_fusion_demo import stiffness_identification
from probtf_observation_fusion_demo.variational_conditioning import (
    ExpectedLikelihoodEvaluation, condition_variational,
)
from test_bayes_factor import _state


def _two_source_result():
    reference = _state()
    covariance = reference.covariance.copy()
    # A shared body/environment constraint, not a zero coordinate. Body
    # marginal is full rank; the retained joint prior is one rank smaller.
    scale = np.sqrt(covariance[18, 18]/covariance[3, 3])
    covariance[18, :] = scale*covariance[3, :]
    covariance[:, 18] = scale*covariance[:, 3]
    retained = replace(reference, covariance=covariance)
    direct = _state(variance=.008, angle=.08, component_id='direct')
    priors, _ = transition_priors(WeightedState(retained.component_id, 1., retained, 0.), direct)
    branches, children, displayed = [], [], []
    for index, parent in enumerate(priors):
        own = []
        logs = (math.log(.3),) if index == 0 else (math.log(.7), math.log(.7)-1000.)
        for native_index, log_mass in enumerate(logs):
            mean = np.zeros(24)
            mean[3], mean[9] = ((.01, .006) if index == 0 else (-.015, -.02))
            if index == 0:
                mean[18] = scale*mean[3]
            else:
                mean[18:21] = [.003+native_index*.002, -.001, .002]
            law = Gaussian(mean, parent.state.covariance*(.5 if index == 0 else .6))
            label = parent.label+'|image:native'+str(native_index)
            weight = math.exp(log_mass)
            child = PosteriorComponent(label, law, weight,
                lineage=('image:00000007',), log_weight=log_mass)
            state = reexpress_state(parent.state, mean, law.covariance,
                version=9, component_id=label)
            own.append(child);children.append(child)
            displayed.append(WeightedState(label, weight, state, log_mass))
        branches.append(FrozenBodyBranch(parent.label, parent.weight, parent.state,
            Gaussian(parent.state.mean, parent.state.covariance), tuple(own), parent.log_weight))
    fused = FusedMixture(tuple(children), (), ('image:00000007',), retained.version,
        retained.stamp, retained.command)
    path = (('joint_a', False), ('joint_b', False), ('tool', False))
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    return DemoResult(retained, (Stage('prior', priors), Stage('fused', tuple(displayed), ('marker',))),
        image, image, {'joint_a': path[:1], 'joint_b': path[:2]}, path, path,
        (('marker', False),), tuple((edge, True) for edge, _ in reversed(path))+(('marker', False),),
        'base', (), (), {'observation_ids': ['image:00000007'], 'observation_update_method': 'variational_moments'},
        tuple(branch.prior for branch in branches), fused, tuple(branches))


def test_reset_law_has_a_common_geometric_chart_without_belonging_to_keep_prior_support():
    result = _two_source_result()
    keep, reset = result.frozen_body_branches
    assert keep.prior.support.rank == 17 and reset.prior.support.rank == 18
    assert covariance_support(keep.prior.covariance[:12, :12]).rank == 12
    original = reset.posterior_components[0]
    transported = common_chart_gaussian(reset.reference_state, original.distribution, keep.reference_state)
    with pytest.raises(ValueError, match='support'):
        require_in_support(transported.mean, keep.prior.support)
    with pytest.raises(ValueError, match='support'):
        reexpress_state(keep.reference_state, transported.mean, transported.covariance,
            version=9, component_id=original.label)
    actual = materialize_state_in_chart(keep.reference_state, transported.mean, transported.covariance,
        version=9, component_id=original.label)
    source = result.stages[-1].components[1].state
    assert actual.stamp == source.stamp and actual.command == source.command
    assert actual.version == source.version and actual.component_id == source.component_id
    for left, right in zip(actual.transforms, source.transforms):
        np.testing.assert_allclose(left.translation, right.translation, atol=2e-15, rtol=0.)
        np.testing.assert_allclose(quat_to_rotmat(left.rotation_wxyz),
            quat_to_rotmat(right.rotation_wxyz), atol=3e-15, rtol=0.)
    np.testing.assert_allclose(actual.covariance, source.covariance, atol=3e-17, rtol=2e-12)
    # A cache hit in the geometric operation must not turn the strict
    # original-prior operation into a permissive update.
    with pytest.raises(ValueError, match='support'):
        reexpress_state(keep.reference_state, transported.mean, transported.covariance,
            version=9, component_id=original.label)


def test_common_chart_projection_preserves_all_global_masses_full_cross_and_original_timeline():
    result = _two_source_result()
    branches, stage = result.frozen_body_branches, result.stages[-1]
    original_bits = [(child.distribution.mean.tobytes(), child.distribution.covariance.tobytes())
        for branch in branches for child in branch.posterior_components]
    common = projection_input(result)
    assert result.frozen_body_branches is branches and result.stages[-1] is stage
    assert 'bayes_factor_transport' not in result.records
    transported = common.frozen_body_branches[0].posterior_components
    assert len(transported) == len(stage.components) == 3
    assert transported[-1].weight == 0. and transported[-1].log_weight < -1000.
    assert [child.label for child in transported] == [item.label for item in stage.components]
    np.testing.assert_array_equal([child.weight for child in transported], [item.weight for item in stage.components])
    np.testing.assert_array_equal([child.log_weight for child in transported], [item.log_weight for item in stage.components])
    provenance = common.records['bayes_factor_transport']
    assert provenance['common_chart_materialization']
    assert not provenance['common_chart_is_probability_prior']
    assert provenance['original_prior_state_ids'] == [branch.reference_state.state_id for branch in branches]
    assert provenance['source_posterior_state_ids'] == [item.state.state_id for item in stage.components]
    for item in common.stages[-1].components:
        assert item.state.stamp == result.prior.stamp
        assert item.state.command == result.prior.command
        np.testing.assert_array_equal(item.state.covariance[12:18], 0.)
    projection = project_update(common, construct_conditional_goals=False)
    weights = np.array([child.weight for child in transported])
    mean = sum(weight*child.distribution.mean for weight, child in zip(weights, transported))
    covariance = sum(weight*(child.distribution.covariance+np.outer(child.distribution.mean-mean,
        child.distribution.mean-mean)) for weight, child in zip(weights, transported))
    np.testing.assert_allclose(projection.posterior.mean, mean, atol=2e-17, rtol=0.)
    np.testing.assert_allclose(projection.posterior.covariance, covariance, atol=2e-17, rtol=2e-14)
    assert projection.posterior.covariance[3, 18] != 0.
    assert projection.posterior.covariance[3, 9] != 0.
    np.testing.assert_array_equal(projection.state.covariance[12:18], 0.)
    expected = materialize_state_in_chart(result.prior, projection.posterior.mean, projection.posterior.covariance,
        version=projection.state.version, component_id=projection.state.component_id)
    assert projection.state.state_id == expected.state_id
    assert projection.record['common_chart_materialization']
    assert not projection.record['reference_affine_support_used_as_prior_constraint']
    assert [item['log_weight'] for item in projection.record['sources']] == [child.log_weight for child in transported]
    assert original_bits == [(child.distribution.mean.tobytes(), child.distribution.covariance.tobytes())
        for branch in branches for child in branch.posterior_components]
    # Removing only the geometric provenance restores the strict prior
    # contract; this is not a global weakening of observation conditioning.
    without_transport = replace(common, records={})
    with pytest.raises(ValueError, match='support'):
        project_update(without_transport, construct_conditional_goals=False)


def test_canonical_common_chart_numerator_keeps_actual_prior_mixture_for_stiffness():
    result = _two_source_result()
    projection = project_update(projection_input(result), construct_conditional_goals=False)
    keep = result.frozen_body_branches[0]
    numerator = PosteriorComponent('canonical', projection.posterior, 1., log_weight=0.)
    branch = FrozenBodyBranch(keep.label, 1., keep.reference_state, keep.prior, (numerator,), 0.)
    ratio = ChartedBodyDensityRatio(result.frozen_body_branches, ('joint_a', 'joint_b'), posterior_branch=branch)
    assert ratio.posterior_branch is branch
    assert ratio.branches == result.frozen_body_branches
    assert all(support.rank == 12 for _, _, support in ratio.marginals)
    np.testing.assert_array_equal(ratio.marginals[0][1][0][1].covariance, projection.posterior.covariance[:12, :12])
    assert [item.log_weight for item in ratio.branches] == [math.log(.75), math.log(.25)]


def test_original_variational_prior_support_and_source_validation_remain_strict():
    result = _two_source_result()
    reference = result.prior
    null = np.zeros(24)
    null[3] = -np.sqrt(reference.covariance[18, 18]/reference.covariance[3, 3])
    null[18] = 1.
    def likelihood(mean, covariance):
        residual = float(null@mean)-.03
        terms = (.5*residual**2, .5*float(null@covariance@null))
        return ExpectedLikelihoodEvaluation(sum(terms), residual*null, .5*np.outer(null, null),
            expected_negative_log_likelihood_terms=terms)
    posterior = condition_variational(result.frozen_body_branches[0].prior, likelihood)
    assert abs(float(null@posterior.distribution.mean)) < 1e-14
    assert posterior.diagnostics['original_prior_held_fixed']
    assert posterior.diagnostics['prior_support_rank'] == 17
    require_in_support(posterior.distribution.mean, result.frozen_body_branches[0].prior.support)
    # The reset child's own original prior remains a genuine conditioning
    # constraint even though its target common geometry has no such role.
    reset = result.frozen_body_branches[1]
    child = reset.posterior_components[0]
    changed = child.distribution.mean.copy();changed[12] = .01
    invalid = replace(child, distribution=Gaussian(changed, child.distribution.covariance))
    branches = (result.frozen_body_branches[0], replace(reset,
        posterior_components=(invalid,)+reset.posterior_components[1:]))
    with pytest.raises(ValueError, match='support'):
        projection_input(replace(result, frozen_body_branches=branches))


def test_stiffness_freeze_validates_common_numerator_source_before_density_work(monkeypatch):
    result = _two_source_result()
    projection = project_update(projection_input(result), construct_conditional_goals=False)
    reference = result.frozen_body_branches[0]
    raw = PosteriorComponent(projection.state.component_id, projection.posterior, 1., log_weight=0.)
    numerator = FrozenBodyBranch(reference.label, 1., reference.reference_state, reference.prior, (raw,), 0.)
    final = replace(result, stages=result.stages[:-1]+(Stage('fused', (
        WeightedState(raw.label, 1., projection.state, 0.),), ('marker',)),),
        frozen_body_posterior=replace(result.frozen_body_posterior, components=(raw,)),
        projected_body_numerator=numerator,
        records=dict(result.records, internal_projection=projection.record))
    with pytest.raises(ValueError, match='support'):
        reexpress_state(reference.reference_state, raw.distribution.mean, raw.distribution.covariance)

    class SourceValidationReached(RuntimeError):
        pass
    identifier = StiffnessIdentifier.__new__(StiffnessIdentifier)
    identifier.last_key = None
    identifier.last_stamp = None
    # Stop exactly after actual frozen-law/provenance validation, before
    # density integration or auxiliary mechanics. No K evidence is mocked
    # or claimed successful by this isolated source-verification test.
    def density_boundary():
        raise SourceValidationReached('frozen numerator source validated')
    identifier._density_record = density_boundary
    calls = []
    actual_materialize = stiffness_identification.materialize_state_in_chart
    def record_materialize(*args, **kwargs):
        calls.append((args[0].state_id, kwargs))
        return actual_materialize(*args, **kwargs)
    monkeypatch.setattr(stiffness_identification, 'materialize_state_in_chart', record_materialize)
    with pytest.raises(SourceValidationReached, match='source validated'):
        identifier.update(final)
    assert len(calls) == 1 and calls[0][0] == result.prior.state_id
    assert final.frozen_body_branches is result.frozen_body_branches
    assert final.projected_body_numerator is numerator
    assert final.stages[-1].components[0].state is projection.state
