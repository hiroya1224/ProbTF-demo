"""The opt-in stiffness hazard uses the actual joint ratio and same K law."""

from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore
from probtf_observation_fusion_demo.stiffness_bayes_factor import (
    LocalLaplaceEvidence, LogIntegral, StiffnessEvidenceUnavailable,
)
from probtf_observation_fusion_demo.stiffness_identification import (
    StiffnessIdentifier, _stiffness_surprise_settings,
)


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]
UNIT = 'per_effective_sensor_event'


def configuration(**override):
    return dict(enabled=True, change_probability=.10,
        reset_measure='initial_log_stiffness', event_probability_unit=UNIT,
        evidence_method='laplace', **override)


def settings(surprise):
    return dict(prior_config_path=str(REPO/'ros/examples/deflecomp/deflecomp_ros/config/estimator.yaml'),
                bayes_factor_surprise=surprise)


@pytest.mark.parametrize('missing', ['reset_measure', 'event_probability_unit', 'evidence_method'])
def test_positive_hazard_requires_explicit_reset_unit_and_integrator(missing):
    supplied = configuration()
    supplied.pop(missing)
    with pytest.raises(ValueError, match='explicit configuration.*'+missing):
        StiffnessIdentifier('/does-not-exist.urdf', (), settings(supplied))


def test_disabled_and_zero_hazard_need_no_arbitrary_reset_or_evidence_choice():
    assert not _stiffness_surprise_settings({})['enabled']
    assert _stiffness_surprise_settings(dict(bayes_factor_surprise=dict(enabled=True,
        change_probability=0.)))['change_probability'] == 0.
    with pytest.raises(ValueError, match='Unsupported.*reset_measure'):
        _stiffness_surprise_settings(dict(bayes_factor_surprise=dict(configuration(), reset_measure='point')))


def test_local_laplace_integrates_a_correlated_gaussian_without_changing_the_density():
    covariance = np.array([[.7, .23], [.23, 1.1]])
    inverse = np.linalg.inv(covariance)
    mean = np.array([.6, -.4])
    log_mass = 4.2
    normalization = math.log(2.*math.pi)+.5*np.linalg.slogdet(covariance)[1]
    def density(x):
        error = x-mean
        return log_mass-normalization-.5*error@inverse@error, -inverse@error
    rule = LocalLaplaceEvidence(lambda:(np.zeros(2),))
    integral = rule(density)
    assert integral.log_value == pytest.approx(log_mass, abs=1e-9)
    assert not integral.exact
    assert integral.diagnostics['stationarity_verified']
    assert integral.diagnostics['positive_definiteness_verified']
    assert integral.diagnostics['two_step_stability_verified']
    assert not integral.diagnostics['density_projection_applied']
    assert not integral.diagnostics['all_modes_integrated']
    np.testing.assert_allclose(integral.diagnostics['Hessian'], inverse, atol=1e-9)


def test_local_laplace_cannot_use_a_nonstationary_one_score_representative_as_a_mode():
    def density(x):
        error = x-np.array([4., -2.])
        precision = np.diag([1., 5.])
        return -.5*error@precision@error, -precision@error
    rule = LocalLaplaceEvidence(lambda:(np.zeros(2),), max_iterations=1)
    with pytest.raises(StiffnessEvidenceUnavailable, match='full-score tolerance') as caught:
        rule(density)
    assert caught.value.diagnostics['mode_score_inf'] > rule.gradient_tolerance


def test_local_laplace_rejects_nonpositive_hessian_without_jitter_or_clipping():
    def saddle(x):
        return float(x[0]**2-x[0]**4), np.array([2*x[0]-4*x[0]**3])
    with pytest.raises(StiffnessEvidenceUnavailable, match='positive definite') as caught:
        LocalLaplaceEvidence(lambda:(np.zeros(1),))(saddle)
    assert caught.value.diagnostics['Hessian_eigenvalues'][0] < 0.


def test_local_laplace_rejects_unstable_score_difference_hessian():
    def unresolved(x):
        return -.5*x[0]**2, np.array([-x[0]+.001*np.sin(1e4*x[0])])
    with pytest.raises(StiffnessEvidenceUnavailable, match='unstable across'):
        LocalLaplaceEvidence(lambda:(np.zeros(1),))(unresolved)


def test_score_root_polish_is_not_rejected_by_cancelling_log_density_roundoff(monkeypatch):
    # Mimic BFGS's observed precision-loss exit at a nearly stationary point.
    # The exact log law is a Gaussian. Its computed value also contains a
    # mathematically zero difference of large, cancelling body-log terms.
    # The score remains resolved: the exact Newton root has *lower* computed
    # log density because the objective subtraction loses precision.
    target = .3
    near = np.array([target+5e-8])
    def density(x):
        residual = x[0]-target
        cancellation = ((1e4+math.sin(x[0]))-1e4)-math.sin(x[0])
        return -.5*100.*residual**2+cancellation, np.array([-100.*residual])
    assert density(np.array([target]))[0] < density(near)[0]-1e-12
    def precision_loss(*args, **kwargs):
        return SimpleNamespace(x=near.copy(), success=False, nit=10,
            message='Desired error not necessarily achieved due to precision loss.')
    monkeypatch.setattr('probtf_observation_fusion_demo.stiffness_bayes_factor.minimize', precision_loss)
    integral = LocalLaplaceEvidence(lambda:(near,))(density)
    assert integral.diagnostics['stationarity_polish'][0]['accepted']
    assert not integral.diagnostics['stationarity_polish'][0]['objective_monotonicity_required']
    assert integral.diagnostics['mode_score_inf'] <= 1e-6
    assert integral.diagnostics['stationarity_verified']
    assert integral.diagnostics['positive_definiteness_verified']
    assert integral.diagnostics['two_step_stability_verified']
    np.testing.assert_allclose(integral.diagnostics['Hessian'], [[100.]], atol=1e-9)
    assert integral.log_value == pytest.approx(.5*math.log(2.*math.pi/100.), abs=1e-11)


@pytest.fixture(scope='module')
def acquired(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp('stiffness_bfs_acquired'))
    inputs = scene.sensor_inputs(scene.step(0., 1000.))
    templates = tuple(yaml.safe_load((PACKAGE/'config/imu_marker.yaml').read_text())['observations'])
    return scene, inputs, templates


def event(store, inputs, identifier, kind, stamp):
    payload = dict(time=stamp-1000.)
    if kind == 'imu':
        payload.update(gravity_in_sensor=inputs.gravity_in_sensor,
            imu_target_link=inputs.imu_target_link, imu_sigma_rad=inputs.imu_sigma_rad)
    else:
        if kind == 'image':
            payload.update(image=inputs.image, image_stamp=stamp)
    return store.append(identifier, kind, stamp, inputs.command, payload)


def test_actual_separate_imu_image_same_stamp_use_full_body_ratio_and_same_continuous_law(acquired, tmp_path, monkeypatch):
    scene, inputs, templates = acquired
    inference = MovingInference(scene.robot_description, templates)
    identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings(configuration()))
    def forbidden(*args, **kwargs):
        raise AssertionError('Stiffness BFS/representatives must not use Monte Carlo.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        records = []
        for index, kind in enumerate(('imu', 'image')):
            acquired_event = event(store, inputs, kind+':0', kind, 1000.)
            result = inference.update_sensor_event(acquired_event, event_index=0, probability_unit=UNIT)
            sources = tuple(item.state.state_id for item in result.stages[-1].components)
            record = identifier.update(result)
            assert record['update_applied'], (record.get('failure_reason'), record.get('evidence_failure_diagnostics'))
            assert record['stiffness_posterior_defined']
            assert identifier.update(result) is record
            surprise = record['bayes_factor_surprise']
            assert surprise['rho0'] == .1
            assert surprise['metadata']['probability_unit'] == UNIT
            assert surprise['metadata']['source_stamp'] == 1000.
            assert surprise['observation_ids'] == [kind+':0']
            assert not surprise['keep_evidence']['exact']
            assert surprise['keep_evidence']['diagnostics']['stationarity_verified']
            assert not record['posterior']['normalization_estimate_is_exact']
            assert record['posterior']['global_moments'] is None
            assert record['source_prior_weights'] == [.75, .25]
            assert [len(item['components']) for item in record['body_marginal']['branches']] == [1, 0]
            factor = identifier.factors[-1]
            assert factor.ratio.posterior_branch is result.projected_body_numerator
            assert identifier.stiffness_bfs.updates[-1].factor.func == identifier._factor_evaluation
            assert identifier.stiffness_bfs.updates[-1].factor.args[0] is factor
            proposal = identifier.command_proposal(result)
            assert proposal is not None, record.get('proposal_failure_reason')
            assert record['proposal_selection']['maximum_iterations'] == 1
            selected = np.asarray(proposal['selected_logK'])
            density, score = identifier.stiffness_bfs.evaluate(selected)
            assert proposal['selected_log_unnormalized_density'] == density
            np.testing.assert_array_equal(record['proposal_selection']['selected_density_score'], score)
            assert proposal['density_id'] == record['posterior']['density_id']
            assert tuple(item.state.state_id for item in result.stages[-1].components) == sources
            records.append(record)
        assert len(identifier.factors) == len(identifier.stiffness_bfs.updates) == 2
        assert identifier.observation_ids == frozenset(('imu:0', 'image:0'))
        assert records[0]['bayes_factor_surprise']['identical_keep_reset_evidence_reused']


def test_zero_hazard_actual_factor_product_is_bitwise_legacy(acquired, tmp_path):
    scene, inputs, templates = acquired
    inference = MovingInference(scene.robot_description, templates)
    disabled = StiffnessIdentifier(scene.urdf, inference.body_ids, settings(dict(enabled=False)))
    zero = StiffnessIdentifier(scene.urdf, inference.body_ids,
                               settings(dict(enabled=True, change_probability=0.)))
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        initial = inference.update_sensor_event(event(store, inputs, 'joint:0', 'joint', 1000.),
                                                 0, probability_unit=UNIT)
        assert disabled.update(initial) == zero.update(initial)
        for index in range(2):
            acquired_event = event(store, inputs, 'imu:'+str(index), 'imu', 1000.+index*.01)
            result = inference.update_sensor_event(acquired_event, index, probability_unit=UNIT)
            left = disabled.update(result)
            right = zero.update(result)
            assert left == right
            for x in (disabled.initial_components[0].distribution.mean,
                      disabled.initial_components[0].distribution.mean+.2):
                left_value, left_score = disabled.log_density(x)
                right_value, right_score = zero.log_density(x)
                assert left_value == right_value
                np.testing.assert_array_equal(left_score, right_score)
            assert disabled.command_proposal(result) == zero.command_proposal(result)


def test_failed_evidence_retains_pending_factor_and_law_without_fabricating_a_posterior(acquired, tmp_path):
    scene, inputs, templates = acquired
    inference = MovingInference(scene.robot_description, templates)
    identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings(configuration()))
    prior = deepcopy(identifier._density_record())
    old_integrator = identifier.stiffness_bfs.integrator
    def unavailable(_):
        raise StiffnessEvidenceUnavailable('unresolved local evidence', dict(reason='test failure'))
    identifier.stiffness_bfs.integrator = unavailable
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        result = inference.update_sensor_event(event(store, inputs, 'imu:0', 'imu', 1000.),
                                                0, probability_unit=UNIT)
        record = identifier.update(result)
        assert record['posterior'] is None and not record['update_applied']
        assert not record['stiffness_posterior_defined']
        assert record['retained_prior'] == prior
        assert record['evidence_failure_diagnostics'] == dict(reason='test failure')
        assert not identifier.factors and not identifier.observation_ids
        assert len(identifier.pending_stiffness_events) == 1
        assert identifier.command_proposal(result) is None
        identifier.stiffness_bfs.integrator = old_integrator
        next_result = inference.update_sensor_event(event(store, inputs, 'imu:1', 'imu', 1000.01),
                                                    1, probability_unit=UNIT)
        recovered = identifier.update(next_result)
        assert recovered['update_applied'], (recovered.get('failure_reason'), recovered.get('evidence_failure_diagnostics'))
        assert recovered['posterior'] is not None
        assert len(recovered['recovered_pending_events']) == 1
        assert recovered['recovered_pending_events'][0]['observation_ids'] == ['imu:0']
        assert identifier.observation_ids == frozenset(('imu:0', 'imu:1'))
        assert len(identifier.factors) == 2 and not identifier.pending_stiffness_events


def test_no_observation_has_no_stiffness_hazard_or_mechanics_queries(acquired, tmp_path, monkeypatch):
    scene, inputs, _ = acquired
    inference = MovingInference(scene.robot_description, ())
    identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings(configuration()))
    old = deepcopy(identifier._density_record())
    def forbidden(*args, **kwargs):
        raise AssertionError('No observations must evaluate neither mechanics nor a BFS integral.')
    monkeypatch.setattr(identifier, '_prediction', forbidden)
    identifier.stiffness_bfs.integrator = forbidden
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        result = inference.update_sensor_event(event(store, inputs, 'disabled:0', 'imu', 1000.),
                                                0, probability_unit=UNIT)
        record = identifier.update(result)
        assert record['status'] == 'identity_density_ratio'
        assert record['prior'] == record['posterior'] == old
        assert identifier.stiffness_bfs.updates == ()
