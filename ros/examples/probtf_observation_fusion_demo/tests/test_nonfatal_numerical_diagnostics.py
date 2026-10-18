"""Finite numerical quality warnings cannot silently discard sensor updates."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import probtf_observation_fusion_demo.stiffness_bayes_factor as evidence
from probtf_observation_fusion_demo.async_workers import event_inference_worker
from probtf_observation_fusion_demo.stiffness_bayes_factor import (
    ContinuousStiffnessBFS, LocalLaplaceEvidence, ResetMeasure,
    StiffnessEvidenceUnavailable,
)

from test_async_execution import (
    _new_session, _complete_acquisitions, _advance_ticks, LocalQueue,
)


def quadratic_density(x):
    x = np.asarray(x, dtype=float)
    return -.5*float(x@x)-.5*x.size*np.log(2.*np.pi), -x


def finite_nonstationary_minimize(fun, x0, **kwargs):
    # A controlled finite optimizer endpoint on a strictly concave density.
    # The score is re-evaluated at that endpoint, not manufactured or zeroed.
    point = np.asarray(x0, dtype=float).copy()
    point[0] += .01
    value, derivative = fun(point)
    return OptimizeResult(x=point, fun=value, jac=derivative,
        success=False, status=2, message='test: finite result before convergence',
        nit=0, nfev=1, njev=1, hess_inv=np.eye(point.size))


def test_finite_nonstationary_evidence_updates_full_BFS_and_next_observation(monkeypatch):
    monkeypatch.setattr(evidence, 'minimize', finite_nonstationary_minimize)
    integrator = LocalLaplaceEvidence(lambda: (np.array([.2, -.3]),))
    reset = ResetMeasure('initial', quadratic_density)
    density = ContinuousStiffnessBFS(quadratic_density, initial_source_id='initial',
                                    rho0=.1, reset=reset, integrator=integrator)
    def factor(x):
        x = np.asarray(x)
        return -.1*float(x@x), -.2*x
    first = density.advance('factor_a', ('imu:a',), factor, event_id='imu:a')
    second = density.advance('factor_b', ('imu:b',), factor, event_id='imu:b')
    assert first['update_applied'] and second['update_applied']
    assert [item.factor_id for item in density.updates] == ['factor_a', 'factor_b']
    assert density.observation_ids == frozenset(('imu:a', 'imu:b'))
    for record in (first, second):
        for name in ('keep_evidence', 'reset_evidence'):
            integral = record[name]
            diagnostic = integral['diagnostics']
            assert np.isfinite(integral['log_value'])
            assert diagnostic['optimizer_success'] is False
            assert diagnostic['stationarity_verified'] is False
            assert diagnostic['finite_nonstationary_point_accepted']
            assert diagnostic['numerical_quality_checks_are_update_gates'] is False
            assert diagnostic['two_step_stability_checked'] is False
            assert diagnostic['two_step_stability_verified'] is None
    value, score = density.evaluate(np.array([.1, -.2]))
    assert np.isfinite(value) and np.all(np.isfinite(score))


@pytest.mark.parametrize('bad_value,bad_score', [(np.nan, [0., 0.]),
                                               (np.inf, [0., 0.]),
                                               (0., [np.nan, 0.]),
                                               (0., [np.inf, 0.])])
def test_actual_nonfinite_density_or_score_remains_an_explicit_failure(bad_value, bad_score):
    rule = LocalLaplaceEvidence(lambda: (np.array([0., 0.]),))
    with pytest.raises(StiffnessEvidenceUnavailable):
        rule(lambda x: (bad_value, np.asarray(bad_score)))


def worker_inputs(tmp_path):
    current, clock, _, _ = _new_session(tmp_path, case='imu', duration=.1,
                                      control_mode='running_gradient',
                                      visibility_factor_options=dict(enabled=False, weight=0.))
    _advance_ticks(current, clock, 4)
    _complete_acquisitions(current)
    events = current.store.events()
    requests = LocalQueue()
    # All acquired records remain acquisition ordered, including command
    # notifications and the unused image; the two IMUs are not paired to images.
    assert [event.kind for event in events].count('imu') == 2
    for event in events:
        index = int(event.payload['frame_index'])
        requests.put((event, index, ()))
    requests.put(None)
    return current, requests


def test_actual_inference_worker_continues_finite_unconverged_evidence_with_honest_flags(tmp_path, monkeypatch):
    current, requests = worker_inputs(tmp_path)
    try:
        original_minimize = evidence.minimize
        def near_mode_with_finite_residual(fun, x0, *args, **kwargs):
            optimum = original_minimize(fun, x0, *args, **kwargs)
            point = np.asarray(optimum.x).copy()
            # The first revolute axis is about gravity. Its log-K direction
            # retains the proper initial Gaussian curvature. A small move off
            # the numerical mode creates a finite nonzero score without an
            # unavailable equilibrium or a fictitious likelihood.
            point[0] += 1e-3
            value, derivative = fun(point)
            return OptimizeResult(optimum, x=point, fun=value, jac=derivative,
                success=False, status=2,
                message='test: finite nonstationary endpoint with re-evaluated exact score')
        monkeypatch.setattr(evidence, 'minimize', near_mode_with_finite_residual)
        replies = LocalQueue()
        event_inference_worker(requests, replies, current.scene.robot_description,
            str(current.scene.urdf), current.templates, str(current.output),
            'per_effective_sensor_event')
        messages = []
        while not replies.empty():
            messages.append(replies.get_nowait())
        assert messages and not any(item.get('failed') for item in messages), messages
        imu = [item for item in messages if item['sensor_kind'] == 'imu']
        assert len(imu) == 2
        previous_factor_ids = []
        for index, message in enumerate(imu, 1):
            record = message['result'].records['stiffness']
            assert record['update_applied']
            assert record['posterior'] is not None
            assert message['packet'] is not None
            posterior = record['posterior']
            factor_ids = posterior['factor_ids']
            assert len(factor_ids) == index
            assert factor_ids[:len(previous_factor_ids)] == previous_factor_ids
            assert len(set(factor_ids)) == len(factor_ids)
            previous_factor_ids = factor_ids
            for name in ('keep_evidence', 'reset_evidence'):
                diagnostic = record['bayes_factor_surprise'][name]['diagnostics']
                assert diagnostic['optimizer_success'] is False
                assert diagnostic['stationarity_verified'] is False
                assert diagnostic['finite_nonstationary_point_accepted']
                assert diagnostic['numerical_quality_checks_are_update_gates'] is False
                assert diagnostic['positive_definiteness_verified']
            assert np.all(np.isfinite(message['packet'].proposal['selected_K']))
        assert imu[0]['event_id'] != imu[1]['event_id']
        assert imu[0]['result'].prior.stamp < imu[1]['result'].prior.stamp
    finally:
        current.close()


def test_actual_inference_worker_stops_on_real_nonfinite_evidence_without_fake_posterior(tmp_path, monkeypatch):
    current, requests = worker_inputs(tmp_path)
    try:
        def actual_nonfinite(self, density):
            raise StiffnessEvidenceUnavailable('test: actual NaN in evaluated density',
                diagnostics=dict(nonfinite_value=True, numerical_quality_checks_are_update_gates=False))
        monkeypatch.setattr(LocalLaplaceEvidence, '__call__', actual_nonfinite)
        replies = LocalQueue()
        event_inference_worker(requests, replies, current.scene.robot_description,
            str(current.scene.urdf), current.templates, str(current.output),
            'per_effective_sensor_event')
        messages = []
        while not replies.empty():
            messages.append(replies.get_nowait())
        failures = [item for item in messages if item.get('failed')]
        assert len(failures) == 1
        assert failures[0]['worker'] == 'inference'
        assert not any(item.get('sensor_kind') == 'imu' for item in messages if not item.get('failed'))
        records = (current.output/'estimation_failure.jsonl').read_text()
        assert 'actual NaN' in records
        assert '"posterior":null' in records
    finally:
        current.close()
