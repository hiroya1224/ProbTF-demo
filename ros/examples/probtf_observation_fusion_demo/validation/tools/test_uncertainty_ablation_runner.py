"""Withdrawn truth runs fail before process or inference side effects."""
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

import uncertainty_ablation as runner
from uncertainty_ablation_policy import TruthInjectionRevoked

REVOKED = ('oracle_goal', 'oracle_marker', 'true_K', 'oracle_body',
           'all_oracle', 'known_landmark', 'oracle_body_goal', 'unknown')


def test_baseline_queue_wrappers_preserve_payload_objects_and_numbers(tmp_path):
    outgoing, incoming = Queue(), Queue()
    session = SimpleNamespace(control_requests=outgoing, control_replies=incoming,
                              parameter_packets={}, output=tmp_path)
    runner.install_coordinator_proxies(session, 'baseline')
    matrix = np.arange(36, dtype=float).reshape(6, 6)
    request = dict(packet=None, result=object(), covariance=matrix)
    session.control_requests.put(request)
    assert outgoing.get_nowait() is request
    incoming.put(request)
    assert session.control_replies.get_nowait() is request
    assert request['covariance'] is matrix
    assert sorted(request) == ['covariance', 'packet', 'result']


@pytest.mark.parametrize('mode', REVOKED)
def test_run_context_and_proxy_reject_before_creating_or_mutating_anything(mode, tmp_path):
    output = tmp_path/'never-created'
    session = SimpleNamespace(control_requests=Queue(), control_replies=Queue())
    original = session.control_requests, session.control_replies
    operations = (
        lambda: runner.run('marker', mode, output, duration=0.),
        lambda: runner.DiagnosticContext(mode, output, {}),
        lambda: runner.install_coordinator_proxies(session, mode),
    )
    for operation in operations:
        with pytest.raises(TruthInjectionRevoked, match='revoked.*2026-10-19'):
            operation()
        assert not output.exists()
        assert (session.control_requests, session.control_replies) == original


@pytest.mark.parametrize('mode', REVOKED)
def test_direct_worker_entrypoints_cannot_bypass_the_gate(mode):
    # Invalid arguments establish that rejection precedes imports, queue
    # access, source loading and monkeypatch installation.
    operations = (
        lambda: runner.inference_worker(mode),
        lambda: runner.acquisition_worker(mode, None, None, None, None, None),
        lambda: runner.control_worker(mode, None, None, None, None, None, None, None),
    )
    for operation in operations:
        with pytest.raises(TruthInjectionRevoked):
            operation()


def test_modified_context_mode_is_checked_before_process_creation(tmp_path):
    context = runner.DiagnosticContext('baseline', tmp_path/'never-created', {})
    context.mode = 'oracle_marker'
    with pytest.raises(TruthInjectionRevoked):
        context.Process(target=object(), args=())
    assert not (tmp_path/'never-created').exists()


def test_control_truth_for_injection_is_withdrawn_without_reading_physics():
    with pytest.raises(TruthInjectionRevoked):
        runner.control_truth(None, None)


def test_posthoc_physical_scoring_still_available():
    frames = [SimpleNamespace(tip=np.array([x, 0., 0.]), time=float(i), reached=i == 2)
              for i, x in enumerate((1., .5, 0.))]
    session = SimpleNamespace(frames=frames, scene=SimpleNamespace(goal=np.zeros(3)),
                              frame=frames[-1])
    score = runner.trajectory_summary(session)
    assert score['reached'] and score['final_error_m'] == 0.
    assert score['tip_arc_m'] == 1.
    assert score['truth_used_for_diagnostic_scoring']


def test_cli_advertises_baseline_only():
    assert runner.MODES == ('baseline',)


def test_baseline_inference_wrapper_passes_the_original_arguments(tmp_path, monkeypatch):
    import probtf_observation_fusion_demo.async_workers as workers
    supplied = (object(), object(), object(), object(), object(), tmp_path)
    completed = object()
    seen = []
    def actual(*args):
        seen.append(args)
        return completed
    monkeypatch.setattr(workers, 'event_inference_worker', actual)
    assert runner.inference_worker('baseline', *supplied) is completed
    assert len(seen) == 1
    assert all(left is right for left, right in zip(seen[0], supplied))


def test_baseline_acquisition_wrapper_preserves_physical_render_request(tmp_path, monkeypatch):
    import probtf_observation_fusion_demo.async_workers as workers
    incoming, outgoing = Queue(), Queue()
    request = dict(kind='imu', payload=dict(gravity_in_sensor=[0., 0., 1.]))
    incoming.put(request)
    specification, database = object(), object()
    old_board, old_planar = workers.BoardCamera, workers.PlanarCamera
    def actual(requests, replies, spec, db, output):
        assert requests.get() is request
        assert spec is specification and db is database and output is tmp_path
        replies.put(request)
        return 'rendering unchanged'
    monkeypatch.setattr(workers, 'acquisition_worker', actual)
    assert runner.acquisition_worker('baseline', incoming, outgoing,
        specification, database, tmp_path) == 'rendering unchanged'
    assert outgoing.get() is request
    assert workers.BoardCamera is old_board and workers.PlanarCamera is old_planar
    assert not list(tmp_path.iterdir())


def test_baseline_control_wrapper_passes_the_law_and_restores_helpers(tmp_path, monkeypatch):
    import probtf_observation_fusion_demo.async_workers as workers
    incoming, outgoing = Queue(), Queue()
    request = dict(result=object(), covariance=np.eye(3))
    incoming.put(request)
    prediction = object()
    original_predict = lambda *args, **kwargs: prediction
    original_freeze = lambda kind, value: value
    monkeypatch.setattr(workers, 'predict_result', original_predict)
    monkeypatch.setattr(workers, 'freeze_work_item', original_freeze)
    def actual(requests, replies, *args):
        assert requests.get() is request
        assert workers.predict_result() is prediction
        assert workers.freeze_work_item('control_artifact_job', request) is request
        replies.put(request)
    monkeypatch.setattr(workers, 'control_worker', actual)
    runner.control_worker('baseline', incoming, outgoing, None, None, None, tmp_path, .5)
    assert outgoing.get() is request
    assert workers.predict_result is original_predict
    assert workers.freeze_work_item is original_freeze
