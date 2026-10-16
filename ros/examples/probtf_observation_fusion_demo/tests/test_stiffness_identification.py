"""Density-ratio stiffness inference uses full frozen laws without truth inputs."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

from probtf.spherical_law.fusion import Gaussian, PosteriorComponent
from probtf_observation_fusion_demo.model import run, FrozenBodyBranch
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.stiffness_identification import (
    BodyDensityRatio, ChartedBodyDensityRatio, StiffnessIdentifier, _rotation_chart_volume,
)
from probtf.spherical_law.first_order import BinghamTransform, JointTransformState, bingham_from_rotation_coordinates


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]


def analytic_update(cross):
    prior = Gaussian(np.zeros(2), np.eye(2))
    posterior = Gaussian(np.array([.3, -.2]), np.array([[.5, cross], [cross, .5]]))
    ratio = BodyDensityRatio(prior, (PosteriorComponent('observed', posterior),), [0, 1])
    identifier = StiffnessIdentifier.__new__(StiffnessIdentifier)
    identifier.initial_components = (PosteriorComponent('parameter_prior', Gaussian(np.zeros(1), np.eye(1))),)
    identifier.factors = (ratio,)
    def evaluate(factor, x):
        value, score, _ = factor.evaluate(np.repeat(x, 2))
        return value, np.array([sum(score)])
    identifier._factor_evaluation = evaluate
    zero = identifier.log_density([0.])[1][0]
    precision = zero-identifier.log_density([1.])[1][0]
    return zero/precision, 1./precision


def test_full_cross_covariance_matches_analytic_density_ratio_posterior():
    correct = analytic_update(.2)
    wrong_independent = analytic_update(0.)
    np.testing.assert_allclose(correct, [1./13., 7./13.], atol=1e-12)
    np.testing.assert_allclose(wrong_independent, [1./15., 1./3.], atol=1e-12)


def test_valid_broad_log_stiffness_posterior_needs_no_physical_second_moment_conversion():
    # A valid log-K Gaussian can have physical K moments outside float64.
    # Retaining its log density requires no physical second moments.
    prior = Gaussian(np.zeros(1), np.eye(1)*1000.)
    ratio = BodyDensityRatio(prior, (PosteriorComponent(
        'observed', Gaussian(np.zeros(1), np.eye(1)*600.)),), [0])
    identifier = StiffnessIdentifier.__new__(StiffnessIdentifier)
    identifier.initial_components = (PosteriorComponent('parameter_prior', prior),)
    identifier.factors = (ratio,)
    identifier._factor_evaluation = lambda factor, x: factor.evaluate(x)[:2]
    with np.errstate(over='raise', invalid='raise'):
        for value in (0., 3., 100.):
            density, score = identifier.log_density([value])
            assert density == pytest.approx(multivariate_normal.logpdf([value], [0.], [[600.]]))
            np.testing.assert_allclose(score, [-value/600.], atol=1e-12)


def test_all_body_mixture_weights_survive_and_unknown_marker_is_marginalized():
    prior = Gaussian(np.zeros(3), np.array([[1., .1, .3], [.1, 1., -.2], [.3, -.2, 2.]]))
    covariance = np.diag([.5, .6, 3.])
    components = (PosteriorComponent('a', Gaussian(np.array([.2, -.1, 100.]), covariance), .25),
                  PosteriorComponent('b', Gaussian(np.array([-.3, .2, -50.]), covariance), .75))
    ratio = BodyDensityRatio(prior, components, [0, 1])
    value = np.array([.4, -.2])
    actual, gradient, branches = ratio.evaluate(value)
    log_terms = []
    for item in ratio.components:
        d = value-item.distribution.mean
        log_terms.append(np.log(item.weight)-.5*(2*np.log(2*np.pi)
            +np.linalg.slogdet(item.distribution.covariance)[1]
            +d @ np.linalg.solve(item.distribution.covariance, d)))
    p0 = ratio.prior.covariance
    expected = logsumexp(log_terms)+.5*(2*np.log(2*np.pi)+np.linalg.slogdet(p0)[1]
                                      +value @ np.linalg.solve(p0, value))
    assert actual == pytest.approx(expected)
    assert [item.weight for item in ratio.components] == [.25, .75]
    assert [item.label for item in ratio.components] == ['a', 'b']
    assert len(branches) == 2
    assert not np.isclose(actual, max(branches))
    for j in range(2):
        step = 1e-6*np.eye(2)[j]
        numeric = (ratio.evaluate(value+step)[0]-ratio.evaluate(value-step)[0])/(2e-6)
        assert gradient[j] == pytest.approx(numeric, rel=1e-7, abs=1e-8)
    # Large changes in the unknown marker marginal do not condition the body.
    changed = tuple(PosteriorComponent(item.label, Gaussian(
        item.distribution.mean + [0., 0., 1000.], np.diag([.5, .6, 100.])), item.weight)
        for item in components)
    assert BodyDensityRatio(prior, changed, [0, 1]).evaluate(value)[0] == actual


def test_rank_deficient_support_has_no_jitter_or_silent_outside_evaluation():
    prior = Gaussian(np.zeros(2), np.diag([1., 0.]))
    posterior = PosteriorComponent('observed', Gaussian(np.array([.2, 0.]), np.diag([.5, 0.])))
    ratio = BodyDensityRatio(prior, (posterior,), [0, 1])
    assert ratio.prior_support.rank == 1
    assert np.isfinite(ratio.evaluate([.1, 0.])[0])
    with pytest.raises(ValueError):
        ratio.evaluate([.1, 1.])
    with pytest.raises(ValueError):
        BodyDensityRatio(prior, (PosteriorComponent('wrong', Gaussian(
            np.zeros(2), np.diag([.5, .1]))),), [0, 1])


def test_multimodal_parameter_density_is_not_replaced_by_one_smooth_local_mode():
    ratio = BodyDensityRatio(Gaussian(np.zeros(1), np.eye(1)*4.), (
        PosteriorComponent('left', Gaussian(np.array([-2.]), np.eye(1)*.1), .5),
        PosteriorComponent('right', Gaussian(np.array([2.]), np.eye(1)*.1), .5)), [0])
    identifier = StiffnessIdentifier.__new__(StiffnessIdentifier)
    identifier.initial_components = (PosteriorComponent('parameter_prior', Gaussian(np.zeros(1), np.eye(1))),)
    identifier.factors = (ratio,)
    identifier._factor_evaluation = lambda factor, x: factor.evaluate(x)[:2]
    modes = [-20./10.75, 20./10.75]
    log_values = [identifier.log_density([value]) for value in modes]
    assert log_values[0][0] == pytest.approx(log_values[1][0], abs=1e-12)
    for value, gradient in log_values:
        np.testing.assert_allclose(gradient, [0.], atol=1e-8)
    assert identifier.log_density([0.])[0] < log_values[0][0]-10.
    assert len(ratio.components) == 2
    assert [item.weight for item in ratio.components] == [.5, .5]


@pytest.fixture(scope='module')
def physical_inputs(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp('stiffness_known_model'))
    frame = scene.step(0., 1000.)
    templates = {name: tuple(yaml.safe_load((PACKAGE/'config'/f'{name}.yaml').read_text())['observations'])
                 for name in ('no_sensors', 'imu', 'marker', 'imu_marker')}
    settings = dict(prior_config_path=str(REPO/'ros/examples/deflecomp/deflecomp_ros/config/estimator.yaml'),
                    known_model_source='test explicitly declared shared URDF inertial model')
    return scene, frame, templates, settings


def test_no_information_is_bitwise_identity_and_duplicate_does_not_add_evidence(physical_inputs, monkeypatch):
    scene, frame, templates, settings = physical_inputs
    inference = MovingInference(scene.robot_description, templates['no_sensors'])
    result = inference.update(scene.sensor_inputs(frame), 0)
    identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings)
    before = identifier.components
    def forbidden(*args, **kwargs):
        raise AssertionError('No information must not even query an equilibrium candidate.')
    monkeypatch.setattr(identifier, '_prediction', forbidden)
    record = identifier.update(result)
    assert record['status'] == 'identity_density_ratio'
    assert record['prior'] == record['posterior']
    assert identifier.components is before
    assert identifier.update(result) is record
    foreign_state = replace(result.prior, version=result.prior.version+1)
    foreign_component = replace(result.stages[-1].components[0], state=foreign_state)
    foreign_stage = replace(result.stages[-1], components=(foreign_component,))
    with pytest.raises(ValueError):
        identifier.update(replace(result, stages=result.stages[:-1]+(foreign_stage,)))
    with pytest.raises(ValueError):
        identifier.update(replace(result, frozen_body_prior=Gaussian(
            result.frozen_body_prior.mean+1., result.frozen_body_prior.covariance)))
    assert identifier.update(result) is record
    assert record['information_rank'] == 0 and record['log_density_ratio'] == 0.
    assert record['source_prior_state_id'] == result.prior.state_id
    assert record['body_posterior_state_ids'] == [item.state.state_id for item in result.stages[-1].components]
    initial = record['prior']['initial_components'][0]
    np.testing.assert_array_equal(initial['x_est'], np.full(6, .5*np.log(500.)))
    np.testing.assert_array_equal(initial['P_est'], np.eye(6)*(np.log(500.)/4)**2)
    assert record['posterior']['factor_ids'] == []
    assert record['posterior']['normalization_constant'] is None
    assert record['posterior']['global_moments'] is None
    assert identifier.command_proposal(result) is None
    assert record['feedback_to_control'] is None
    assert not record['feedback_to_body_or_control']


def test_auxiliary_equilibrium_chart_jacobian_reuses_existing_physics_without_truth(physical_inputs):
    scene, frame, templates, settings = physical_inputs
    inference = MovingInference(scene.robot_description, templates['imu'])
    state = inference.prior_at(frame.command, frame.stamp, 0)
    identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings)
    x = identifier.components[0].distribution.mean
    value, jacobian, _, _ = identifier._prediction(x, state)
    for j in range(6):
        step = 1e-5*np.eye(6)[j]
        numeric = (identifier._prediction(x+step, state)[0]-identifier._prediction(x-step, state)[0])/(2e-5)
        np.testing.assert_allclose(jacobian[:, j], numeric, rtol=2e-4, atol=2e-5)
    # Gravity cannot identify rotation-axis stiffness about vertical root yaw.
    np.testing.assert_allclose(jacobian[:, 0], 0., atol=1e-12)
    assert value.shape == (36,)


def test_frozen_auxiliary_branch_is_independent_of_query_evaluation_order(physical_inputs):
    scene, frame, templates, settings = physical_inputs
    inference = MovingInference(scene.robot_description, templates['imu'])
    state = inference.prior_at(frame.command, frame.stamp, 0)
    identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings)
    record = {}
    identifier._freeze_branch(state, record)
    centre = identifier.components[0].distribution.mean
    queries = [centre+shift*np.eye(6)[joint]
               for joint, shift in ((1, .02), (4, -.015), (3, .005))]
    before = tuple(value.copy() if isinstance(value, np.ndarray) else value
                   for value in identifier._branch_anchor)
    forwards = [identifier._prediction(query, state) for query in queries]
    backwards = [identifier._prediction(query, state) for query in reversed(queries)]
    for left, right in zip(forwards, reversed(backwards)):
        for actual, repeated in zip(left, right):
            np.testing.assert_array_equal(actual, repeated)
    for actual, expected in zip(identifier._branch_anchor, before):
        np.testing.assert_array_equal(actual, expected)
    assert record['auxiliary_branch_selection']['source_state_id'] == state.state_id


def test_unavailable_equilibrium_query_never_discards_an_observation_factor(physical_inputs, monkeypatch):
    scene, frame, templates, settings = physical_inputs
    inference = MovingInference(scene.robot_description, templates['imu'])
    result = inference.update(scene.sensor_inputs(frame), 0)
    identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings)
    before = identifier.components
    def unsupported(*args):
        raise ValueError('No stable auxiliary branch in this numerical neighbourhood.')
    monkeypatch.setattr(identifier, '_prediction', unsupported)
    record = identifier.update(result)
    assert record['status'] == 'updated_density_factor'
    assert record['update_applied']
    assert len(record['posterior']['factor_ids']) == 1
    assert record['posterior']['global_moments'] is None
    assert record['posterior']['properness']['verified']
    assert record['information_rank'] is None
    assert record['information_eigenvalues'] is None
    assert identifier.components is before
    assert identifier.command_proposal(result) is None
    assert record['proposal_status'] == 'unavailable'
    assert len(record['proposal_selection']['unavailable_candidates']) == len(identifier._proposal_candidates)
    assert len(identifier.factors) == 1
    with pytest.raises(ValueError, match='No stable auxiliary branch'):
        identifier.log_density(identifier.initial_components[0].distribution.mean)
    assert record['source_prior_state_id'] == result.prior.state_id


def test_valid_sensor_cases_use_same_frozen_ratio_and_are_sensor_order_invariant(physical_inputs, monkeypatch):
    scene, frame, templates, settings = physical_inputs
    def forbidden(*args, **kwargs):
        raise AssertionError('Neither body propagation nor stiffness inference may use MC.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    records = {}
    for name in ('imu', 'marker', 'imu_marker'):
        inference = MovingInference(scene.robot_description, templates[name])
        result = inference.update(scene.sensor_inputs(frame), 0)
        raw = result.frozen_body_prior, result.frozen_body_posterior
        before = (raw[0].mean.tobytes(), raw[0].covariance.tobytes(), tuple(
            (item.weight, item.distribution.mean.tobytes(), item.distribution.covariance.tobytes())
            for item in raw[1].components), tuple(item.state.state_id for item in result.stages[-1].components))
        identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings)
        record = identifier.update(result)
        after = (raw[0].mean.tobytes(), raw[0].covariance.tobytes(), tuple(
            (item.weight, item.distribution.mean.tobytes(), item.distribution.covariance.tobytes())
            for item in raw[1].components), tuple(item.state.state_id for item in result.stages[-1].components))
        assert before == after
        assert record['update_applied'], record.get('failure_reason')
        assert not record['status'].startswith('failed')
        assert record['source_stamp'] == frame.stamp and record['source_command'] == frame.command.tolist()
        assert record['body_posterior_weights'] == [item.weight for item in result.stages[-1].components]
        marginal = record['body_marginal']['branches'][0]
        assert len(marginal['prior_mean']) == 36
        assert len(marginal['components']) == len(result.frozen_body_posterior.components)
        assert record['support_regularization_added'] is False
        assert record['feedback_to_body_or_control'] is False
        assert record['prior']['factor_ids'] == []
        assert len(record['posterior']['factor_ids']) == 1
        assert record['prior']['initial_components'] == record['posterior']['initial_components']
        density, gradient = identifier.log_density(identifier.initial_components[0].distribution.mean)
        assert np.isfinite(density) and np.all(np.isfinite(gradient))
        # Unobservable yaw stiffness is not invented by a local approximation.
        assert abs(gradient[0]) < 1e-8
        records[name] = record
    inference = MovingInference(scene.robot_description, templates['imu_marker'])
    original = inference.update(scene.sensor_inputs(frame), 0)
    swapped = run(inference.common, tuple(reversed(inference.last_observations)), frame.image,
                  prior=original.prior, paths=inference.paths)
    swapped = inference._project_internal_update(swapped, ())
    alternative = StiffnessIdentifier(scene.urdf, inference.body_ids, settings).update(swapped)
    assert alternative == records['imu_marker']
    # Source mismatches cannot be disguised by a new numerical fit.
    bad = replace(original, frozen_body_prior=Gaussian(
        original.frozen_body_prior.mean+1., original.frozen_body_prior.covariance))
    with pytest.raises(ValueError, match='does not match'):
        StiffnessIdentifier(scene.urdf, inference.body_ids, settings).update(bad)
    raw = original.frozen_body_posterior
    first = raw.components[0]
    displaced = first.distribution.mean.copy()
    displaced[3] += .001
    foreign = replace(first, distribution=Gaussian(displaced, first.distribution.covariance))
    bad = replace(original, frozen_body_posterior=replace(raw, components=(foreign,)+raw.components[1:]))
    with pytest.raises(ValueError, match='provenance differ|do not reexpress'):
        StiffnessIdentifier(scene.urdf, inference.body_ids, settings).update(bad)


def test_same_chart_density_ratio_uses_full_prior_and_posterior_mixtures():
    priors = (PosteriorComponent('old-a', Gaussian([-.2, .1], [[.7, .2], [.2, .8]]), .3),
              PosteriorComponent('old-b', Gaussian([.4, -.1], [[1., -.25], [-.25, .9]]), .7))
    posteriors = (PosteriorComponent('new-a', Gaussian([-.1, .2], [[.5, .1], [.1, .6]]), .6),
                  PosteriorComponent('new-b', Gaussian([.3, -.2], [[.6, -.15], [-.15, .7]]), .4))
    ratio = BodyDensityRatio(priors, posteriors, [0, 1])
    def expected(value):
        def density(items):
            return logsumexp([np.log(item.weight)+multivariate_normal.logpdf(
                value, item.distribution.mean, item.distribution.covariance) for item in items])
        return density(posteriors)-density(priors)
    value = np.array([.15, -.3])
    actual, score, _ = ratio.evaluate(value)
    assert actual == pytest.approx(expected(value), abs=1e-12)
    numeric = [(expected(value+1e-6*axis)-expected(value-1e-6*axis))/(2e-6)
               for axis in np.eye(2)]
    np.testing.assert_allclose(score, numeric, rtol=1e-8, atol=1e-9)
    # Dividing only by the first parent changes the observation information.
    first_only = BodyDensityRatio(priors[0].distribution, posteriors, [0, 1])
    assert abs(first_only.evaluate(value)[0]-actual) > .01


def test_a_float_underflow_branch_still_contributes_its_finite_log_mass():
    prior = Gaussian([0.], [[1000.]])
    children = (PosteriorComponent('visible', Gaussian([0.], [[1.]]), 1., log_weight=0.),
                PosteriorComponent('tiny', Gaussian([100.], [[1.]]), 0., log_weight=-1000.))
    ratio = BodyDensityRatio(prior, children, [0])
    log_ratio, score, _ = ratio.evaluate([100.])
    expected = logsumexp([multivariate_normal.logpdf([100.], [0.], [[1.]]),
        -1000.+multivariate_normal.logpdf([100.], [100.], [[1.]])])
    expected -= multivariate_normal.logpdf([100.], [0.], [[1000.]])
    assert log_ratio == pytest.approx(expected, abs=1e-12)
    np.testing.assert_allclose(score, [.1], atol=1e-12)
    assert ratio.components[1].weight == 0. and ratio.components[1].log_weight == -1000.


def test_different_body_charts_use_one_volume_measure_and_the_full_gradient():
    branches = []
    references = []
    for index, (weight, vector) in enumerate(((.3, [.2, -.1, .3]), (.7, [-.3, .25, -.1]))):
        rotation = Rotation.from_rotvec(vector)
        references.append(rotation)
        xyzw = rotation.as_quat(); quaternion = np.r_[xyzw[3], xyzw[:3]]
        covariance = np.diag([.02, .03, .04, .03, .04, .05])
        covariance[0, 4] = covariance[4, 0] = .005
        edge = BinghamTransform('joint', np.zeros(3),
            bingham_from_rotation_coordinates(quaternion, covariance[3:, 3:]), covariance[:3, :3])
        state = JointTransformState((edge,), covariance, command=(0.,), component_id=str(index))
        prior = Gaussian(np.zeros(6), covariance)
        mean = np.array([.01, -.02, .01, .04*(-1)**index, -.03, .02])
        child = PosteriorComponent('child-'+str(index), Gaussian(mean, covariance*.65), weight)
        branches.append(FrozenBodyBranch(str(index), weight, state, prior, (child,)))
    ratio = ChartedBodyDensityRatio(branches, ('joint',))
    def values(x):
        physical = Rotation.from_rotvec([x[0], x[1], .1])
        return [np.r_[np.zeros(3), (reference.inv()*physical).as_rotvec()]
                for reference in references]
    def expected(x, use_volume=True):
        minus, plus = [], []
        for branch, value in zip(branches, values(x)):
            angle = np.linalg.norm(value[3:])
            volume = 2*np.log(np.sin(angle/2)/(angle/2)) if use_volume else 0.
            minus.append(np.log(branch.weight)+multivariate_normal.logpdf(
                value, branch.prior.mean, branch.prior.covariance)-volume)
            child = branch.posterior_components[0]
            plus.append(np.log(child.weight)+multivariate_normal.logpdf(
                value, child.distribution.mean, child.distribution.covariance)-volume)
        return logsumexp(plus)-logsumexp(minus)
    x = np.array([.15, -.2])
    jacobians = [np.column_stack([(values(x+1e-6*axis)[j]-values(x-1e-6*axis)[j])/(2e-6)
                                  for axis in np.eye(2)]) for j in range(2)]
    actual, gradient = ratio.evaluate(tuple(zip(values(x), jacobians)))
    assert actual == pytest.approx(expected(x), abs=1e-12)
    numeric = [(expected(x+1e-6*axis)-expected(x-1e-6*axis))/(2e-6) for axis in np.eye(2)]
    np.testing.assert_allclose(gradient, numeric, rtol=1e-7, atol=1e-8)
    assert abs(expected(x)-expected(x, False)) > 1e-4
    # The chart volume derivative has a finite zero-angle limit.
    volume, derivative = _rotation_chart_volume(np.zeros(6))
    assert volume == 0. and np.array_equal(derivative, np.zeros(6))


def test_cumulative_memory_preserves_old_factors_on_marker_loss_and_adds_only_new_ids(physical_inputs):
    scene, frame, templates, settings = physical_inputs
    inference = MovingInference(scene.robot_description, templates['marker'])
    identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings)
    first = inference.update(scene.sensor_inputs(frame), 0)
    first_record = identifier.update(first)
    assert first_record['status'] == 'updated_density_factor'
    x = identifier.initial_components[0].distribution.mean
    old_density, old_gradient = identifier.log_density(x)
    missing_frame = scene.step(.05, frame.stamp+.05, command=frame.command)
    inputs = scene.sensor_inputs(missing_frame)
    inputs = replace(inputs, image=np.zeros_like(inputs.image))
    missing = inference.update(inputs, 1)
    missing_record = identifier.update(missing)
    assert missing_record['status'] == 'identity_density_ratio'
    assert missing_record['prior'] == missing_record['posterior'] == first_record['posterior']
    np.testing.assert_array_equal(identifier.log_density(x)[0], old_density)
    np.testing.assert_array_equal(identifier.log_density(x)[1], old_gradient)
    assert len(identifier.factors) == 1
    current_frame = scene.step(.10, frame.stamp+.10, command=frame.command)
    current = inference.update(scene.sensor_inputs(current_frame), 2)
    current_record = identifier.update(current)
    assert current_record['status'] == 'updated_density_factor', current_record.get('failure_reason')
    assert len(current.frozen_body_branches) == 1
    assert sum(len(branch.posterior_components) for branch in current.frozen_body_branches) == 1
    assert len(current.stages[-1].goal_components) == 2
    assert len(identifier.factors) == 2
    assert current_record['prior'] == first_record['posterior']
    assert len(current_record['body_marginal']['branches']) == 1
    new_contribution, new_derivative = identifier._factor_evaluation(identifier.factors[-1], x)
    np.testing.assert_allclose(identifier.log_density(x)[0], old_density+new_contribution, atol=1e-12)
    np.testing.assert_allclose(identifier.log_density(x)[1], old_gradient+new_derivative, atol=1e-12)
    # Check the cumulative parameter score through two physical equilibrium
    # maps and two different parent charts against an independent difference.
    for axis in np.eye(6):
        numerical = (identifier.log_density(x+1e-5*axis)[0]
            -identifier.log_density(x-1e-5*axis)[0])/(2e-5)
        np.testing.assert_allclose(identifier.log_density(x)[1] @ axis, numerical,
                                   rtol=3e-4, atol=2e-4)
    assert set(identifier.factors[0].observation_ids).isdisjoint(identifier.factors[1].observation_ids)
    assert identifier.update(current) is current_record
    assert len(identifier.factors) == 2
    # Reusing one old image ID at a new source time cannot count it again.
    repeated = replace(current, records=dict(current.records, observation_ids=first_record['observation_ids']))
    with pytest.raises(ValueError):
        identifier.update(repeated)


def test_command_proposal_ranks_the_continuous_density_and_does_not_replace_it(physical_inputs, monkeypatch):
    scene, frame, templates, settings = physical_inputs
    inference = MovingInference(scene.robot_description, templates['imu'])
    result = inference.update(scene.sensor_inputs(frame), 0)
    identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings)
    def forbidden(*args, **kwargs):
        raise AssertionError('Auxiliary proposal selection must not sample.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    record = identifier.update(result)
    density_record = record['posterior']
    body_ids = tuple(item.state.state_id for item in result.stages[-1].components)
    proposal = identifier.command_proposal(result)
    assert proposal is not None, record.get('proposal_failure_reason')
    assert record['feedback_to_body_or_control']
    assert not record['feedback_to_body']
    assert record['feedback_to_control'] == 'initial command proposal only'
    assert proposal['density_id'] == density_record['density_id']
    assert proposal['source_command'] == list(result.prior.command)
    assert proposal['source_stamp'] == result.prior.stamp
    scores = []
    for x in identifier._proposal_candidates:
        try:
            scores.append(identifier.log_density(x)[0])
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            scores.append(-np.inf)
    assert proposal['selected_candidate_index'] == int(np.argmax(scores))
    selection = record['proposal_selection']
    base = np.array(selection['base_logK'])
    exact_score = identifier.log_density(base)[1]
    np.testing.assert_allclose(selection['base_density_score'], exact_score, atol=1e-12)
    np.testing.assert_allclose(proposal['selected_logK'],
        base+np.asarray(selection['score_step']), atol=1e-12)
    selected_density, selected_score = identifier.log_density(np.asarray(proposal['selected_logK']))
    assert selected_density >= max(scores)
    assert proposal['selected_log_unnormalized_density'] == selected_density
    np.testing.assert_allclose(selection['selected_density_score'], selected_score, atol=1e-12)
    assert proposal['density_improvement_claimed'] == bool(selected_density > max(scores))
    assert not selection['global_mode_claimed']
    assert selection['selected_density_evaluated']
    np.testing.assert_allclose(proposal['command'],
        2*np.asarray(frame.command)-np.asarray(proposal['auxiliary_equilibrium']), atol=1e-12)
    assert record['posterior'] is density_record
    assert record['posterior']['global_moments'] is None
    assert record['proposal_approximation'] is None
    assert tuple(item.state.state_id for item in result.stages[-1].components) == body_ids
    assert identifier.command_proposal(result) == proposal
    foreign = replace(result.prior, version=result.prior.version+1)
    with pytest.raises(ValueError, match='current recorded body update'):
        identifier.command_proposal(replace(result, prior=foreign))
