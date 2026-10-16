"""Density-ratio stiffness inference uses full frozen laws without truth inputs."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.special import logsumexp

from probtf.spherical_law.fusion import Gaussian, PosteriorComponent
from probtf_observation_fusion_demo.model import run
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.stiffness_identification import (
    BodyDensityRatio, StiffnessIdentifier,
)


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]


def analytic_update(cross):
    prior = Gaussian(np.zeros(2), np.eye(2))
    posterior = Gaussian(np.array([.3, -.2]), np.array([[.5, cross], [cross, .5]]))
    ratio = BodyDensityRatio(prior, (PosteriorComponent('observed', posterior),), [0, 1])
    identifier = StiffnessIdentifier.__new__(StiffnessIdentifier)
    identifier.settings = {}
    identifier.components = (PosteriorComponent('parameter_prior', Gaussian(np.zeros(1), np.eye(1))),)
    identifier._prediction = lambda x, state: (np.repeat(x, 2), np.ones((2, 1)), x, np.zeros(1, bool))
    record = {}
    identifier._fit(ratio, None, record)
    return record


def test_full_cross_covariance_matches_analytic_density_ratio_posterior():
    correct = analytic_update(.2)['posterior']
    wrong_independent = analytic_update(0.)['posterior']
    np.testing.assert_allclose(correct['x_est'], [1./13.], atol=1e-8)
    np.testing.assert_allclose(correct['P_est'], [[7./13.]], atol=1e-8)
    np.testing.assert_allclose(wrong_independent['x_est'], [1./15.], atol=1e-8)
    np.testing.assert_allclose(wrong_independent['P_est'], [[1./3.]], atol=1e-8)


def test_valid_broad_log_stiffness_posterior_needs_no_physical_second_moment_conversion():
    # A valid log-K Gaussian can have physical K moments outside float64.
    # Only its required log-K law and finite median need to be represented.
    prior = Gaussian(np.zeros(1), np.eye(1)*1000.)
    ratio = BodyDensityRatio(prior, (PosteriorComponent(
        'observed', Gaussian(np.zeros(1), np.eye(1)*600.)),), [0])
    identifier = StiffnessIdentifier.__new__(StiffnessIdentifier)
    identifier.settings = {}
    identifier.components = (PosteriorComponent('parameter_prior', prior),)
    identifier._prediction = lambda x, state: (x, np.eye(1), x, np.zeros(1, bool))
    record = {}
    with np.errstate(over='raise', invalid='raise'):
        identifier._fit(ratio, None, record)
    assert record['status'] == 'updated_local_laplace'
    posterior = record['posterior']
    np.testing.assert_allclose(posterior['x_est'], [0.], atol=1e-12)
    np.testing.assert_allclose(posterior['P_est'], [[600.]], rtol=1e-12)
    np.testing.assert_allclose(posterior['K_median'], [1.], atol=1e-12)
    assert posterior['components'][0]['P_est'] == posterior['P_est']
    assert posterior['components'][0]['weight'] == 1.
    assert 'K_mean' not in posterior and 'K_covariance' not in posterior
    assert posterior['K_units'] == 'N m/rad'


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


def test_two_justified_parameter_modes_are_retained_with_their_weights():
    ratio = BodyDensityRatio(Gaussian(np.zeros(1), np.eye(1)*4.), (
        PosteriorComponent('left', Gaussian(np.array([-2.]), np.eye(1)*.1), .5),
        PosteriorComponent('right', Gaussian(np.array([2.]), np.eye(1)*.1), .5)), [0])
    identifier = StiffnessIdentifier.__new__(StiffnessIdentifier)
    identifier.settings = {}
    identifier.components = (PosteriorComponent('parameter_prior', Gaussian(np.zeros(1), np.eye(1))),)
    identifier._prediction = lambda x, state: (x, np.eye(1), x, np.zeros(1, bool))
    record = {}
    identifier._fit(ratio, None, record)
    assert record['status'] == 'updated_multimodal_laplace'
    assert len(identifier.components) == len(record['posterior']['components']) == 2
    np.testing.assert_allclose(sorted(item.distribution.mean[0] for item in identifier.components),
                               [-20./10.75, 20./10.75], atol=1e-6)
    np.testing.assert_allclose([item.weight for item in identifier.components], [.5, .5], atol=1e-7)
    assert record['rejected_mode_candidates']


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
    np.testing.assert_array_equal(record['prior']['x_est'], np.full(6, .5*np.log(500.)))
    np.testing.assert_array_equal(record['prior']['P_est'], np.eye(6)*(np.log(500.)/4)**2)


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


def test_unavailable_laplace_is_not_reported_as_a_posterior_or_zero_information(physical_inputs, monkeypatch):
    scene, frame, templates, settings = physical_inputs
    inference = MovingInference(scene.robot_description, templates['imu'])
    result = inference.update(scene.sensor_inputs(frame), 0)
    identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings)
    before = identifier.components
    def unsupported(*args):
        raise ValueError('No stable auxiliary branch in this numerical neighbourhood.')
    monkeypatch.setattr(identifier, '_prediction', unsupported)
    record = identifier.update(result)
    assert record['status'] == 'failed_support_or_laplace'
    assert record['posterior'] is None
    assert record['retained_prior'] == record['prior']
    assert record['prior_retained'] and not record['update_applied']
    assert record['information_rank'] is None
    assert record['information_eigenvalues'] is None
    assert identifier.components is before
    assert record['rejected_auxiliary_trials'] == dict(count=1, reasons=[
        'No stable auxiliary branch in this numerical neighbourhood.'])
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
        assert record['body_marginal']['indices'] == list(range(36))
        assert len(record['body_marginal']['components']) == len(result.frozen_body_posterior.components)
        assert record['support_regularization_added'] is False
        assert record['feedback_to_body_or_control'] is False
        assert record['prior']['P_est'][0][0] == pytest.approx(record['posterior']['P_est'][0][0], rel=1e-5)
        records[name] = record
    inference = MovingInference(scene.robot_description, templates['imu_marker'])
    original = inference.update(scene.sensor_inputs(frame), 0)
    swapped = run(inference.common, tuple(reversed(inference.last_observations)), frame.image,
                  prior=original.prior, paths=inference.paths)
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
    with pytest.raises(ValueError, match='do not reexpress'):
        StiffnessIdentifier(scene.urdf, inference.body_ids, settings).update(bad)
