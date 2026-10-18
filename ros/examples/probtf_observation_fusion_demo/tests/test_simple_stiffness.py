"""One log-K Gaussian carries history; only the new rotation factor is queried."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from probtf.spherical_law.fusion import (
    FusedMixture, Gaussian, PosteriorComponent, condition_gaussian,
)
from probtf_observation_fusion_demo import simple_stiffness as simple
from probtf_observation_fusion_demo.simple_stiffness import (
    SimpleStiffnessIdentifier, GaussianProjectionUnavailable, project_gaussian_factor,
)
from probtf_observation_fusion_demo.body_native import reexpress_state
from probtf_observation_fusion_demo.model import FrozenBodyBranch, Stage, WeightedState
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.async_stiffness import (
    capture_stiffness_packet, PhysicalActionEngine,
)
from probtf_observation_fusion_demo.stiffness_command import StiffnessCommandMap
from legacy_marker import legacy_marker_scene


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]


def linear_factor(matrix, noise, observed):
    inverse = np.linalg.inv(noise)
    def evaluate(x):
        residual = matrix @ x-observed
        return -.5*float(residual @ inverse @ residual), -matrix.T @ inverse @ residual
    return evaluate


def test_linear_gaussian_factor_matches_exact_full_covariance_posterior():
    prior = Gaussian(np.array([.2, -.3]), np.array([[.8, .2], [.2, 1.1]]))
    matrix = np.array([[1., .2], [.3, -.4]])
    noise, observed = np.array([[.6, .1], [.1, .7]]), np.array([.8, -.2])
    posterior, diagnostics = project_gaussian_factor(prior,
        linear_factor(matrix, noise, observed), max_iterations=20)
    covariance = np.linalg.inv(np.linalg.inv(prior.covariance)+matrix.T @ np.linalg.solve(noise, matrix))
    mean = covariance @ (np.linalg.solve(prior.covariance, prior.mean)
                         +matrix.T @ np.linalg.solve(noise, observed))
    np.testing.assert_allclose(posterior.mean, mean, atol=2.e-7)
    np.testing.assert_allclose(posterior.covariance, covariance, atol=3.e-12)
    assert posterior.covariance[0, 1] != 0.
    assert not diagnostics['evidence_computed'] and not diagnostics['monte_carlo_used']
    assert diagnostics['prior_fixed_during_update']


def test_sequential_gaussian_projection_carries_old_information_without_recalling_old_factor():
    prior = Gaussian(np.zeros(2), np.array([[1., .2], [.2, .9]]))
    matrices = (np.array([[1., .4]]), np.array([[-.2, 1.]]))
    observations, noises = (np.array([.8]), np.array([-.3])), (np.array([[.3]]), np.array([[.4]]))
    calls = [0, 0]
    factors = []
    for index in range(2):
        base = linear_factor(matrices[index], noises[index], observations[index])
        def factor(x, i=index, evaluate=base):
            calls[i] += 1
            return evaluate(x)
        factors.append(factor)
    first, _ = project_gaussian_factor(prior, factors[0], max_iterations=20)
    first_count = calls[0]
    second, _ = project_gaussian_factor(first, factors[1], max_iterations=20)
    assert calls[0] == first_count and calls[1] > 0
    precision = np.linalg.inv(prior.covariance)
    information = precision @ prior.mean
    for matrix, noise, observed in zip(matrices, noises, observations):
        precision += matrix.T @ np.linalg.solve(noise, matrix)
        information += matrix.T @ np.linalg.solve(noise, observed)
    expected_covariance = np.linalg.inv(precision)
    np.testing.assert_allclose(second.mean, expected_covariance @ information, atol=2.e-7)
    np.testing.assert_allclose(second.covariance, expected_covariance, atol=3.e-12)


def test_finite_iteration_or_stationarity_shortfall_does_not_reject_available_gaussian():
    prior = Gaussian(np.zeros(2), np.eye(2))
    factor = linear_factor(np.eye(2), np.diag([.4, .7]), np.array([2., -1.]))
    posterior, diagnostic = project_gaussian_factor(prior, factor, max_iterations=0)
    np.testing.assert_array_equal(posterior.mean, prior.mean)
    np.testing.assert_allclose(posterior.covariance, np.diag([1./3.5, 1./(1.+1./.7)]), atol=2.e-12)
    assert diagnostic['projection_available'] and not diagnostic['stationarity_tolerance_met']
    assert 'zero mean-iteration budget' in diagnostic['optimizer_message']
    partial, finite = project_gaussian_factor(prior, factor, max_iterations=1)
    assert finite['projection_available'] and not finite['optimizer_success']
    assert np.all(np.isfinite(partial.mean))


def test_actual_nonpositive_required_posterior_curvature_is_not_floored():
    prior = Gaussian(np.zeros(2), np.eye(2))
    with pytest.raises(GaussianProjectionUnavailable) as raised:
        project_gaussian_factor(prior, lambda x: (float(x @ x), 2.*x), max_iterations=0)
    diagnostic = raised.value.diagnostics
    assert not diagnostic['projection_available'] and diagnostic['prior_retained']
    assert not diagnostic['covariance_regularization_added']
    np.testing.assert_allclose(diagnostic['posterior_curvature_whitened'], -np.eye(2), atol=1.e-12)


def test_nonfinite_factor_is_explicit_and_small_positive_variance_is_not_floored():
    prior = Gaussian(np.zeros(2), np.diag([1.e-16, 1.]))
    posterior, _ = project_gaussian_factor(prior, lambda x: (0., np.zeros(2)), max_iterations=0)
    np.testing.assert_allclose(posterior.covariance, prior.covariance, rtol=3.e-16, atol=0.)
    with pytest.raises(GaussianProjectionUnavailable, match='nonfinite'):
        project_gaussian_factor(prior, lambda x: (np.nan, np.zeros(2)))


@pytest.fixture(scope='module')
def model(tmp_path_factory):
    scene = legacy_marker_scene(tmp_path_factory.mktemp('simple_K_known_model'))
    inference = MovingInference(scene.robot_description, (), config_path=scene.legacy_inference_config)
    source = inference.prior_at(inference.reference_command, 12., 0)
    settings = dict(prior_config_path=str(REPO/
        'ros/examples/deflecomp/deflecomp_ros/config/estimator.yaml'),
        simple_gaussian_max_iterations=5,
        bayes_factor_surprise=dict(enabled=True, change_probability=.1))
    return scene, inference, source, settings


def result_for(model, source=None, event_id='simple_K:image:0', observed=True):
    _, inference, initial, _ = model
    source = initial if source is None else source
    prior = Gaussian(source.mean, source.covariance)
    if observed:
        observation = np.zeros((1, source.mean.size)); observation[0, 10] = 1.
        after = condition_gaussian(prior, np.array([.004]), np.array([[.0007]]), observation)[0]
        observation_ids = (event_id,)
    else:
        after, observation_ids = prior, ()
    child = PosteriorComponent(event_id+':posterior', after)
    displayed = source if not observed else reexpress_state(source, after.mean, after.covariance,
        version=source.version+1, component_id=child.label)
    branch = FrozenBodyBranch(source.component_id, 1., source, prior, (child,))
    return SimpleNamespace(prior=source, hand_path=inference.paths['hand'],
        frozen_body_prior=prior,
        frozen_body_posterior=FusedMixture((child,), (), observation_ids,
            source.version, source.stamp, source.command),
        frozen_body_branches=(branch,), projected_body_numerator=branch,
        stages=(Stage('estimated_joint_law', (WeightedState(child.label, 1., displayed),)),),
        records=dict(observation_ids=list(observation_ids)))


def make_identifier(model):
    scene, inference, _, settings = model
    return SimpleStiffnessIdentifier(scene.urdf, inference.body_ids, settings)


def test_initial_prior_has_explicit_finite_representative_and_compatible_packet(model):
    identifier = make_identifier(model)
    assert identifier.stiffness_bfs is None and len(identifier.components) == 1
    result = result_for(model, observed=False)
    record = identifier.update(result)
    result.records['stiffness'] = record
    proposal = identifier.command_proposal(result)
    assert proposal is not None and proposal['representative_from_prior']
    assert record['representative_from_prior'] and not record['update_applied']
    assert record['body_marginal']['edge_ids'] == list(identifier.body_edge_ids)
    assert record['posterior']['density_id'] == proposal['density_id']
    assert not record['evidence_computed'] and not record['bayes_factor_surprise_enabled']
    assert record['retained_factor_count'] == 0 and identifier.factors == ()
    np.testing.assert_array_equal(proposal['selected_logK'], identifier.components[0].distribution.mean)
    mapping = StiffnessCommandMap(identifier, result, proposal)
    mapping.validate_result(result)
    packet = capture_stiffness_packet(result)
    scene, inference, _, settings = model
    engine = PhysicalActionEngine(scene.urdf, inference.body_ids, identifier.settings)
    engine.set(packet)
    action = engine.increment(np.asarray(result.prior.command), np.asarray(result.prior.command))
    np.testing.assert_allclose(action['joint_increment'], 0., atol=1.e-13)
    assert not action['physical_truth_received']


def test_updates_keep_one_gaussian_and_never_requery_previous_acquisition_factor(model, monkeypatch):
    identifier = make_identifier(model)
    counts = {}
    original = identifier._factor_evaluation
    def tracked(factor, x):
        identity = factor.observation_ids[0]
        counts[identity] = counts.get(identity, 0)+1
        return original(factor, x)
    monkeypatch.setattr(identifier, '_factor_evaluation', tracked)
    first = result_for(model)
    before = identifier.components[0].distribution
    first_record = identifier.update(first)
    assert first_record['update_applied'], first_record.get('simple_projection_failures')
    assert len(first_record['simple_gaussian_projection']) == 1
    assert identifier.factors == () and not identifier.factor_cache
    retained = identifier.components[0].distribution
    assert retained is not before and not np.array_equal(retained.covariance, before.covariance)
    first_count = counts['simple_K:image:0']
    _, inference, source, _ = model
    command = np.asarray(source.command)+[0., .03, -.02, 0., 0., 0.]
    next_source = inference.prior_at(command, source.stamp+1.2, 1)
    second = result_for(model, next_source, 'simple_K:image:1')
    second_record = identifier.update(second)
    assert second_record['update_applied'], second_record.get('simple_projection_failures')
    assert counts['simple_K:image:0'] == first_count
    assert counts['simple_K:image:1'] > 0
    assert identifier.factors == () and not identifier.factor_cache
    assert len(identifier.components) == 1 and identifier.simple_projection_count == 2
    np.testing.assert_array_equal(second_record['prior']['covariance'], retained.covariance)
    projection = second_record['simple_gaussian_projection'][0]
    assert projection['source_stamp'] == next_source.stamp
    assert projection['source_command'] == list(next_source.command)
    assert projection['observation_ids'] == ['simple_K:image:1']
    assert not second_record['cumulative_factors_retained']
    assert second_record['posterior']['factor_ids'] == []


def test_duplicate_does_not_refuse_observation_and_changed_old_id_is_rejected(model):
    identifier = make_identifier(model); result = result_for(model)
    record = identifier.update(result)
    retained = identifier.components[0].distribution
    assert identifier.update(result) is record
    assert identifier.components[0].distribution is retained
    _, inference, source, _ = model
    changed = inference.prior_at(np.asarray(source.command)+[0., .01, 0., 0., 0., 0.], 13., 1)
    with pytest.raises(ValueError, match='old observation'):
        identifier.update(result_for(model, changed, 'simple_K:image:0'))


def test_real_projection_unavailability_retains_defined_prior_and_reports_unapplied_observation(model, monkeypatch):
    identifier = make_identifier(model)
    old = identifier.components[0].distribution
    def unavailable(*args, **kwargs):
        raise GaussianProjectionUnavailable('required posterior curvature is not positive definite',
            dict(projection_available=False, prior_retained=True, covariance_regularization_added=False))
    monkeypatch.setattr(simple, 'project_gaussian_factor', unavailable)
    result = result_for(model)
    record = identifier.update(result)
    assert not record['update_applied'] and record['prior_retained']
    assert record['status'] == 'single_gaussian_projection_unavailable'
    assert record['stiffness_posterior_defined']
    assert identifier.components[0].distribution is old
    assert len(record['simple_projection_failures']) == 1
    assert identifier.factors == () and not identifier.factor_cache
    proposal = identifier.command_proposal(result)
    assert proposal is not None and proposal['representative_from_prior']
    np.testing.assert_array_equal(proposal['selected_logK'], old.mean)


def test_prior_derived_seed_scan_uses_exact_columns_and_coherent_sum_without_changing_zero_budget():
    prior = Gaussian(np.array([.2, -.3]), np.array([[.8, .2], [.2, 1.1]]))
    factor = linear_factor(np.eye(2), np.eye(2), np.array([1., -1.]))
    _, diagnostic = project_gaussian_factor(prior, factor, max_iterations=1)
    scan = diagnostic['prior_seed_scan']
    scale = np.linalg.cholesky(prior.covariance)
    probes = {row['label']: prior.mean+scale @ row['whitened'] for row in scan['probes']}
    np.testing.assert_array_equal(probes['prior_mean'], prior.mean)
    for index in range(2):
        np.testing.assert_array_equal(probes[f'plus_Cholesky_column_{index}'], prior.mean+scale[:, index])
        np.testing.assert_array_equal(probes[f'minus_Cholesky_column_{index}'], prior.mean-scale[:, index])
    np.testing.assert_array_equal(probes['plus_Cholesky_column_sum'], prior.mean+np.sum(scale, axis=1))
    np.testing.assert_array_equal(probes['minus_Cholesky_column_sum'], prior.mean-np.sum(scale, axis=1))
    assert scan['original_prior_and_current_factor_fixed']
    posterior, zero = project_gaussian_factor(prior, factor, max_iterations=0)
    assert not zero['prior_seed_scan']['enabled']
    assert len(zero['prior_seed_scan']['probes']) == 1
    np.testing.assert_array_equal(posterior.mean, prior.mean)


def test_real_joint_stop_plateau_is_escaped_by_same_prior_and_rotation_ratio(model, monkeypatch, tmp_path):
    """A known-model test target is a prior-derived candidate, never true K.

    A centre-only local search can enter the actual URDF joint-stop plateau.
    The same normalized prior and the same full joint local
    rotation density ratio favor a finite coherent prior-coordinate start.
    """
    from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
    from probtf_observation_fusion_demo.simple_inference import SimpleInference
    # Use the current demo's supplied geometry and command, not the lighter
    # legacy test tool. Neither realized simulator poses nor its K are read.
    scene = MovingSwitchScene(tmp_path/'known_joint_stop_geometry')
    inference = SimpleInference(scene.robot_description, ())
    source = inference.prior_at(scene.initial_command, 12., 0)
    model = (scene, inference, source, model[3])
    probe = make_identifier(model)
    initial = probe.components[0].distribution
    declared = initial.mean+np.sum(np.linalg.cholesky(initial.covariance), axis=1)
    target, _, _, active = probe._prediction(declared, source)
    assert not active[2]
    indices = [6*source.edge_ids.index(name)+coordinate
               for name in inference.body_ids for coordinate in (3, 4, 5)]
    observation = np.eye(source.mean.size)[indices]
    after = condition_gaussian(Gaussian(source.mean, source.covariance),
        target[indices], np.eye(len(indices))*.0003, observation)[0]
    result = result_for(model, source=source, event_id='simple_K:declared_stop_plateau', observed=False)
    child = PosteriorComponent('declared_stop_plateau:posterior', after)
    displayed = reexpress_state(source, after.mean, after.covariance,
        version=source.version+1, component_id=child.label)
    branch = FrozenBodyBranch(source.component_id, 1., source,
        Gaussian(source.mean, source.covariance), (child,))
    result.frozen_body_posterior = replace(result.frozen_body_posterior,
        components=(child,), observation_ids=('simple_K:declared_stop_plateau',))
    result.frozen_body_branches = (branch,)
    result.projected_body_numerator = branch
    result.stages = (Stage('estimated_joint_law', (WeightedState(child.label, 1., displayed),)),)
    result.records['observation_ids'] = ['simple_K:declared_stop_plateau']

    old = SimpleStiffnessIdentifier(model[0].urdf,
        inference.body_ids, dict(model[3], simple_gaussian_prior_seed_scan=False))
    old_record = old.update(result)
    old_proposal = old.command_proposal(result)
    assert old_proposal['auxiliary_active_joint_limits'][2]
    old_physical = old._physical_prediction(np.asarray(old_proposal['selected_logK']), source)
    np.testing.assert_array_equal(old_physical['derivative'][2], np.zeros(6))

    new = make_identifier(model)
    frozen = []
    discard = new._discard_current_factors
    def keep_current_factor_metadata():
        frozen.extend(new.factors)
        discard()
    monkeypatch.setattr(new, '_discard_current_factors', keep_current_factor_metadata)
    new_record = new.update(result)
    new_proposal = new.command_proposal(result)
    assert new_record['update_applied'], new_record['simple_projection_failures']
    assert not new_proposal['auxiliary_active_joint_limits'][2]
    assert new_proposal['selected_K'][2] > old_proposal['selected_K'][2]
    assert new_record['simple_gaussian_projection'][0]['prior_seed_scan']['selected_seed_label'] != 'prior_mean'
    np.testing.assert_array_equal(new_record['prior']['mean'], old_record['prior']['mean'])
    np.testing.assert_array_equal(new_record['prior']['covariance'], old_record['prior']['covariance'])
    assert frozen[0].observation_ids == ('simple_K:declared_stop_plateau',)
    # Re-evaluate the factor's original fixed equilibrium branch at the
    # selected K; the command proposal's direct solve must agree in this case.
    new._branch_anchor = frozen[0].branch_anchor
    from_factor = new._physical_prediction(np.asarray(new_proposal['selected_logK']), source)
    np.testing.assert_allclose(new_proposal['auxiliary_equilibrium'], from_factor['theta'], atol=2.e-8, rtol=0.)
    new._branch_anchor = None
    _, factor_score = new._factor_evaluation(frozen[0], np.asarray(new_proposal['selected_logK']))
    assert factor_score[0] == pytest.approx(0., abs=1.e-8)
    diagnostic = new_record['simple_gaussian_projection'][0]
    whitened = np.linalg.solve(np.linalg.cholesky(initial.covariance),
        np.asarray(new_proposal['selected_logK'])-initial.mean)
    # Gravity yaw carries no K1 likelihood score. A remaining K1 displacement
    # after the finite budget is a reported prior-restoring stationarity
    # residual, not evidence that the observation identified that coordinate.
    assert diagnostic['selected_whitened_score'][0] == pytest.approx(whitened[0], abs=1.e-8)
    assert not new_record['evidence_computed'] and not new_record['bayes_factor_surprise_enabled']
