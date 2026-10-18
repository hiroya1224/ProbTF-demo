"""Fixed-prior Gaussian distribution updates retain native laws and provenance.

The affine oracle below uses precision algebra, independently of the
variational solver.  Native/core moment derivatives are checked against
deterministic perturbations, including covariance cross blocks.
"""
from copy import deepcopy
from dataclasses import replace
import io
import math
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from visualization_msgs.msg import MarkerArray

from probtf_observation_fusion_demo import variational_model as module
from probtf_observation_fusion_demo.async_artifacts import capture_candidate_identity
from probtf_observation_fusion_demo.bayes_factor import transition_priors
from probtf_observation_fusion_demo.body_native import point_query
from probtf_observation_fusion_demo.display import make_markers, verify_wire_identity
from probtf_observation_fusion_demo.model import WeightedState
from probtf_observation_fusion_demo.moment_observation import prepare_observation
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.node import build_snapshot
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore
from test_joint_batch_inference import acquired


@pytest.fixture
def problem(acquired):
    description, inputs, templates = acquired
    inference = MovingInference(description, templates)
    prior = inference.prior_at(inputs.command, inputs.stamp, 0)
    observations = []
    for template in templates:
        observation = deepcopy(template)
        observation['observation_id'] = 'original:' + observation['sensor']
        if observation['sensor'] == 'imu':
            observation.update(target_link=inputs.imu_target_link,
                gravity_in_root=[0., 0., -1.],
                gravity_in_sensor=inputs.gravity_in_sensor.tolist(),
                direction_sigma_rad=inputs.imu_sigma_rad)
        observations.append(observation)
    return inference, prior, inputs, observations


def _observation(problem, kind):
    return next(item for item in problem[3] if item['sensor'] == kind)


def _parents(prior):
    return transition_priors(WeightedState(prior.component_id, 1., prior, 0.), prior, .25)[0]


def test_affine_likelihood_model_matches_independent_bayes_oracle_and_never_replaces_original_prior(problem, monkeypatch):
    """An affine special case must infer covariance as well as a best point."""
    inference, prior, inputs, _ = problem
    observed = _observation(problem, 'imu')
    n = prior.mean.size
    # The real full body prior includes uncertain and exactly fixed axes.
    # Couple every coordinate into two likelihood coordinates, rather than
    # using independent per-joint measurements or an isotropic covariance.
    H = .025*np.sin(np.arange(2*n).reshape(2, n) + .2)
    R = np.array([[.03, .009], [.009, .05]])
    precision = np.linalg.inv(R)
    residual = np.array([.018, -.025])
    normalizer = .5*(np.linalg.slogdet(R)[1]+2.*math.log(2.*math.pi))
    trials = []
    factory_sources = []

    class AffineExpectation:
        def __init__(self, reference, likelihoods):
            factory_sources.append((reference, tuple(likelihoods)))

        def __call__(self, mean, covariance):
            trials.append((mean.copy(), covariance.copy()))
            d = H@mean-residual
            return module.LikelihoodExpectation(
                .5*float(d@precision@d+np.trace(precision@H@covariance@H.T))+normalizer,
                H.T@precision@d, .5*H.T@precision@H,
                dict(affine_validation_oracle=True),
                (.5*float(d@precision@d), .5*float(np.trace(precision@H@covariance@H.T)), normalizer))

    monkeypatch.setattr(module, 'PropagatedLikelihoodExpectation', AffineExpectation)
    original_mean = prior.mean.copy()
    original_covariance = prior.covariance.copy()
    result = module.run(inference.common, [observed], inputs.image, prior=prior,
        paths=inference.paths, joint_evidence=True,
        update_options=dict(method='variational_moments', max_iterations=200,
                            gradient_tolerance=1e-9))
    # Kalman/Schur algebra is an independent exact oracle on singular P0.
    innovation = H@original_covariance@H.T+R
    gain = np.linalg.solve(innovation, H@original_covariance).T
    expected_mean = gain@residual
    expected_covariance = original_covariance-gain@H@original_covariance
    expected_log_evidence = -.5*(2.*math.log(2.*math.pi)+np.linalg.slogdet(innovation)[1]
        +residual@np.linalg.solve(innovation, residual))
    raw = result.frozen_body_posterior.components[0].distribution
    np.testing.assert_allclose(raw.mean, expected_mean, atol=1e-7, rtol=2e-5)
    np.testing.assert_allclose(raw.covariance, expected_covariance, atol=1e-7, rtol=2e-5)
    assert result.records['joint_observation_evidence']['log_evidence'] == pytest.approx(expected_log_evidence, abs=2e-8)
    assert factory_sources[0][0] is prior
    assert len(trials) > 1
    assert any(not np.array_equal(m, original_mean) for m, _ in trials)
    assert any(not np.array_equal(P, original_covariance) for _, P in trials)
    np.testing.assert_array_equal(prior.mean, original_mean)
    np.testing.assert_array_equal(prior.covariance, original_covariance)
    branch = result.frozen_body_branches[0]
    assert branch.reference_state is prior
    np.testing.assert_array_equal(branch.prior.covariance, original_covariance)
    seed_record = result.records['variational_observation_update']['branch_diagnostics'][0]['initial_distribution']
    assert seed_record['provided'] and seed_record['considered']
    assert seed_record['proposal_only'] and seed_record['used_as_new_prior'] is False


@pytest.mark.parametrize('sensor', ['imu', 'marker'])
def test_native_expectation_full_mean_and_covariance_adjoint_matches_independent_perturbations(problem, sensor):
    inference, prior, inputs, _ = problem
    prepared = prepare_observation(inference.common, _observation(problem, sensor),
        inference.paths, inputs.image, prior.stamp)
    native = prepared.components[0]
    evaluator = module.PropagatedLikelihoodExpectation(prior, (native,))
    # Perturb only the actual support. Nonzero m is essential: differentiating
    # only at a zero chart would miss reexpression/chart transport derivatives.
    values, vectors = np.linalg.eigh(prior.covariance)
    active = values > 1e-12
    root = vectors[:, active]*np.sqrt(values[active])
    direction_m = root@(.07*np.sin(np.arange(root.shape[1])+.3))
    mean = prior.mean + .3*direction_m
    covariance = .91*prior.covariance
    at = evaluator(mean, covariance)
    h = 2e-5
    plus = evaluator(mean+h*direction_m, covariance).expected_negative_log_likelihood
    minus = evaluator(mean-h*direction_m, covariance).expected_negative_log_likelihood
    assert (plus-minus)/(2.*h) == pytest.approx(float(at.gradient_mean@direction_m), rel=2e-5, abs=2e-5)
    # Dense support-coordinate perturbation explicitly mixes different links,
    # translation/rotation, and both symmetric halves of every cross block.
    a = np.cos(np.arange(root.shape[1])+.4)
    b = np.sin(1.7*np.arange(root.shape[1])+.2)
    D = root@(np.outer(a, b)+np.outer(b, a))@root.T/root.shape[1]
    assert np.linalg.norm(D[:6, 6:]) > 0.
    plus = evaluator(mean, covariance+h*D).expected_negative_log_likelihood
    minus = evaluator(mean, covariance-h*D).expected_negative_log_likelihood
    assert (plus-minus)/(2.*h) == pytest.approx(float(np.sum(at.gradient_covariance*D)), rel=3e-5, abs=3e-5)
    np.testing.assert_allclose(at.gradient_covariance, at.gradient_covariance.T, atol=1e-12)
    record = at.diagnostics['propagated_observations'][0]
    assert record['source_state_id'] == prior.state_id
    assert record['source_stamp'] == inputs.stamp
    assert record['source_command'] == list(inputs.command)
    assert record['observed_mean'] == native.observed_mean.tolist()
    predicted = np.asarray(record['predicted_mean'])
    S = np.asarray(record['predicted_covariance'])
    W = np.linalg.inv(native.covariance)
    normalizer = .5*(np.linalg.slogdet(native.covariance)[1]+len(predicted)*math.log(2.*math.pi))
    expected = .5*((predicted-native.observed_mean)@W@(predicted-native.observed_mean)+np.trace(W@S))+normalizer
    assert at.expected_negative_log_likelihood == pytest.approx(expected, abs=2e-9, rel=2e-12)


def test_every_native_and_parent_log_weight_survives_huge_common_evidence_without_pruning(problem, monkeypatch):
    inference, prior, inputs, _ = problem
    observation = _observation(problem, 'marker')
    prepared = prepare_observation(inference.common, observation, inference.paths, inputs.image, prior.stamp)
    first = prepared.components[0]
    controlled = replace(prepared, components=(
        replace(first, label='seventeen', log_weight=math.log(.17)),
        replace(first, label='eighty_three', log_weight=math.log(.83)),
        replace(first, label='finite_underflow', log_weight=-1000.)))
    monkeypatch.setattr(module, 'prepare_observation', lambda *args: controlled)
    calls = []

    def fixed_evidence(raw_prior, evaluator, **options):
        calls.append((raw_prior, evaluator, options))
        # The small term must not disappear when native/parent log weights
        # are combined with a common enormous finite negative log evidence.
        return SimpleNamespace(distribution=raw_prior, log_evidence=-1e20,
            log_evidence_parts=(-1e20, .031), diagnostics=dict(finite_budget_validation=True))

    monkeypatch.setattr(module, 'condition_variational', fixed_evidence)
    parents = _parents(prior)
    result = module.run(inference.common, [observation], inputs.image,
        prior_components=parents, paths=inference.paths, joint_evidence=True)
    assert len(calls) == 6
    raw = result.frozen_body_posterior.components
    assert len(raw) == 6
    for group, parent in zip((raw[:3], raw[3:]), parents):
        assert group[0].weight == pytest.approx(parent.weight*.17, abs=2e-15)
        assert group[1].weight == pytest.approx(parent.weight*.83, abs=2e-15)
        assert group[2].weight == 0. and math.isfinite(group[2].log_weight)
        assert group[1].log_weight-group[0].log_weight == pytest.approx(math.log(.83/.17), abs=2e-14)
    update = result.records['variational_observation_update']
    assert update['native_likelihoods'] == [item.record() for item in controlled.components]
    assert len(update['branch_diagnostics']) == 6
    assert math.fsum(item.weight for item in raw) == pytest.approx(1., abs=2e-14)
    for branch, parent in zip(result.frozen_body_branches, parents):
        assert branch.reference_state is parent.state
        assert len(branch.posterior_components) == 3
        assert branch.weight == parent.weight


def test_empty_and_missing_observations_are_identity_without_optimizer_or_evidence(problem, monkeypatch):
    inference, prior, inputs, _ = problem

    def forbidden(*args, **kwargs):
        raise AssertionError('Missing observations must not manufacture an optimization/evidence update.')

    monkeypatch.setattr(module, 'condition_variational', forbidden)
    monkeypatch.setattr(module, 'PropagatedLikelihoodExpectation', forbidden)
    empty = module.run(inference.common, (), inputs.image, prior=prior, paths=inference.paths)
    missing = module.run(inference.common, [_observation(problem, 'marker')],
        np.full_like(inputs.image, 255), prior=prior, paths=inference.paths, joint_evidence=True)
    for result in (empty, missing):
        assert result.stages[-1].components[0].state is prior
        assert result.records['valid_observation_count'] == 0
        assert not result.sensor_posteriors
        assert 'joint_observation_evidence' not in result.records
        assert result.records['variational_observation_update']['empty_observation_identity']
    assert missing.local_estimates[0].posterior is None


def test_two_acquired_events_accumulate_the_actual_previous_full_posterior_and_mass(problem, tmp_path):
    inference, _prior, inputs, _ = problem
    # Isolate ordinary Bayesian accumulation; Keep/Reset weighting is covered
    # separately above. Each event is still the actual acquisition interface.
    inference.surprise_enabled = False
    inference.observation_update_settings = dict(method='variational_moments', max_iterations=2)
    inference.observation_update_method = 'variational_moments'
    store = SensorEventStore(tmp_path/'accumulation.sqlite3')
    payload = dict(time=0., frame_index=0,
        gravity_in_sensor=inputs.gravity_in_sensor, imu_target_link=inputs.imu_target_link,
        imu_sigma_rad=inputs.imu_sigma_rad)
    first_event = store.append('imu:first', 'imu', inputs.stamp, inputs.command, payload)
    later_payload = dict(payload, time=.05, frame_index=1)
    later_event = store.append('imu:second', 'imu', inputs.stamp+.05, inputs.command, later_payload)
    try:
        first = inference.update_sensor_event(first_event)
        first_body = first.stages[-1].components[0].state
        second = inference.update_sensor_event(later_event)
    finally:
        store.close()
    prior = second.frozen_body_branches[0].reference_state
    assert prior.stamp == later_event.stamp and prior.command == later_event.command
    np.testing.assert_array_equal(prior.covariance, first_body.covariance)
    for actual, previous in zip(prior.transforms, first_body.transforms):
        np.testing.assert_array_equal(actual.translation, previous.translation)
        np.testing.assert_array_equal(actual.rotation_wxyz, previous.rotation_wxyz)
        if actual.bingham is not None:
            np.testing.assert_array_equal(actual.bingham, previous.bingham)
    assert not second.records['bayes_factor_surprise']['applied']
    assert first.records['observation_ids'] == [first_event.event_id]
    assert second.records['observation_ids'] == [later_event.event_id]
    assert second.records['candidate_measure']['previous_log_total_mass'] == first.records['candidate_measure']['log_total_mass']
    assert second.records['candidate_measure']['log_total_mass'] == pytest.approx(
        first.records['candidate_measure']['log_total_mass']+
        second.records['joint_observation_evidence']['log_evidence'], abs=2e-12)
    assert second.records['variational_observation_update']['branch_diagnostics'][0]['source_prior_state_id'] == prior.state_id


def test_real_posterior_canonical_source_and_full_cross_are_identical_in_control_snapshot_and_wire(problem, monkeypatch, tmp_path):
    inference, _prior, inputs, _ = problem
    import probtf_observation_fusion_demo.model as old_model
    import probtf_observation_fusion_demo.nonlinear_model as old_point_model
    import probtf_observation_fusion_demo.nonlinear_conditioning as old_point_solver
    import probtf_observation_fusion_demo.moment_model as old_conditional

    def forbidden(*args, **kwargs):
        raise AssertionError('Distribution optimization must not use point MAP/GN, old one-pass update, or Monte Carlo.')

    monkeypatch.setattr(old_model, 'lift_sensor_posterior', forbidden)
    monkeypatch.setattr(old_point_model, 'run', forbidden)
    monkeypatch.setattr(old_point_solver, 'condition_nonlinear', forbidden)
    monkeypatch.setattr(old_conditional, 'run', forbidden)
    # A finite, deliberately short budget is a valid Gaussian approximation;
    # it must reach canonicalization/control/display with honest diagnostics.
    inference.observation_update_settings = dict(method='variational_moments', max_iterations=1)
    inference.observation_update_method = 'variational_moments'
    patch = pytest.MonkeyPatch()
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        patch.setattr(np.random, name, forbidden)
    store = SensorEventStore(tmp_path/'events.sqlite3')
    event = store.append('image:original:00017', 'image', inputs.stamp, inputs.command,
        dict(time=inputs.time, frame_index=17, image=inputs.image, image_stamp=inputs.stamp))
    try:
        result = inference.update_sensor_event(event)
    finally:
        # Existing display particles legitimately sample the canonical query.
        patch.undo()
        store.close()
    stage = result.stages[-1]
    assert len(stage.components) == 1
    canonical = stage.components[0].state
    assert inference.candidate_measure.components[0].state is canonical
    assert canonical.stamp == inputs.stamp and canonical.command == tuple(inputs.command)
    assert result.records['sensor_event']['event_id'] == event.event_id
    assert result.records['sensor_event']['acquisition_stamp'] == event.stamp
    assert result.records['sensor_event']['sent_command'] == list(event.command)
    assert result.records['sensor_event']['sequence'] == event.sequence
    assert result.records['observation_ids'] == [event.event_id]
    offdiagonal = canonical.covariance.copy()
    for index in range(len(canonical.transforms)):
        offdiagonal[6*index:6*index+6, 6*index:6*index+6] = 0.
    assert np.linalg.norm(offdiagonal) > 1e-7
    identity_before = capture_candidate_identity(result)
    snapshot = build_snapshot(result, stage, inference.common)
    hand = next(item for item in snapshot.points if item.frame_label == 'hand' and item.axis == 'origin')
    expected = point_query(canonical, result.hand_path)
    assert hand.query.state_id == canonical.state_id
    np.testing.assert_array_equal(hand.query.mean, expected.mean)
    np.testing.assert_array_equal(hand.query.covariance, expected.covariance)
    np.testing.assert_array_equal(hand.query.cross_covariance, expected.cross_covariance)
    settings = inference.settings['display']
    stream = io.BytesIO()
    make_markers(snapshot, settings['seed'], settings['point_size']).serialize(stream)
    wire = MarkerArray().deserialize(stream.getvalue())
    assert verify_wire_identity(snapshot, wire, seed=settings['seed'], point_size=settings['point_size'])
    assert capture_candidate_identity(result) == identity_before
    update = result.records['variational_observation_update']
    assert update['original_prior_held_fixed'] and update['observation_added_once']
    assert update['distribution_parameter_optimization_used']
    assert not update['point_optimization_used'] and not update['gauss_newton_used']
    assert not update['laplace_body_evidence_used'] and not update['monte_carlo_used']
    assert result.records['local_gaussian_lift_applied'] is False
    assert deepcopy(update) is update
    assert pickle.loads(pickle.dumps(update)) == update
    with pytest.raises(TypeError):
        update['branch_diagnostics'][0]['source_prior_state_id'] = 'foreign-source'
