"""Optional low-frequency stage timing never changes K density arithmetic."""
from contextlib import contextmanager

import numpy as np

from probtf_observation_fusion_demo.stiffness_bayes_factor import (
    ContinuousStiffnessBFS, LocalLaplaceEvidence, ResetMeasure,
)
from test_joint_batch_stiffness import arm, batch_inputs, identifier
from test_time_indexed_stiffness_batch import prefix_specifications


class RecordingMonitor:
    def __init__(self):
        self.records = []

    @contextmanager
    def stage(self, name, **context):
        self.records.append(('enter', name, context))
        try:
            yield
        except BaseException:
            self.records.append(('error', name, context))
            raise
        else:
            self.records.append(('exit', name, context))


def test_laplace_timers_preserve_mode_mass_score_and_query_sequence():
    queries = [[], []]
    def density(index):
        def evaluate(x):
            queries[index].append(x.copy())
            delta = x-np.array([.2, -.3])
            return -.5*float(delta@delta), -delta
        return evaluate
    seeds = lambda: (np.array([0., 0.]), np.array([1., -1.]))
    plain = LocalLaplaceEvidence(seeds)(density(0))
    observed_rule = LocalLaplaceEvidence(seeds)
    monitor = observed_rule.runtime_monitor = RecordingMonitor()
    observed = observed_rule(density(1))
    assert plain.log_value == observed.log_value
    for key in ('mode', 'mode_score', 'Hessian', 'evaluations', 'optimizer_iterations'):
        assert plain.diagnostics[key] == observed.diagnostics[key]
    assert len(queries[0]) == len(queries[1])
    for left, right in zip(*queries):
        np.testing.assert_array_equal(left, right)
    entered = [name for phase, name, _ in monitor.records if phase == 'enter']
    assert entered == ['stiffness.laplace.attempt', 'stiffness.laplace.seed_ranking',
        'stiffness.laplace.mode_optimization', 'stiffness.laplace.hessian',
        'stiffness.laplace.mass_factorization']
    assert [name for phase, name, _ in monitor.records if phase == 'exit'] == [
        'stiffness.laplace.seed_ranking', 'stiffness.laplace.mode_optimization',
        'stiffness.laplace.hessian', 'stiffness.laplace.mass_factorization',
        'stiffness.laplace.attempt']


def test_bfs_names_keep_and_reset_and_propagates_optional_monitor():
    def initial(x):
        return -.5*float(x@x), -x
    def reset(x):
        delta = x-.4
        return -.5*float(delta@delta), -delta
    rule = LocalLaplaceEvidence(lambda: (np.zeros(1), np.ones(1)))
    density = ContinuousStiffnessBFS(initial, initial_source_id='initial', rho0=.1,
        reset=ResetMeasure('reset', reset), integrator=rule)
    monitor = density.runtime_monitor = RecordingMonitor()
    record = density.advance('factor', ('imu:a',),
        lambda x: (-.5*float((x-.2)@(x-.2)), -(x-.2)), event_id='batch:a')
    assert record['update_applied']
    assert rule.runtime_monitor is monitor
    branches = [(name, context) for phase, name, context in monitor.records
                if phase == 'enter' and name.startswith('stiffness.evidence.')]
    assert [name for name, _ in branches] == ['stiffness.evidence.keep', 'stiffness.evidence.reset']
    assert all(context == dict(event_id='batch:a', factor_id='factor',
        observation_count=1, retained_event_count=0) for _, context in branches)
    assert len(density.updates) == 1 and density.observation_ids == {'imu:a'}


def test_identifier_propagates_monitor_and_retains_same_full_time_indexed_K_law(arm):
    plain, observed = identifier(arm), identifier(arm)
    monitor = observed.runtime_monitor = RecordingMonitor()
    result, events, ledger = batch_inputs(arm[2])
    specifications = prefix_specifications(arm[2], result, events)
    records = [engine.update_batch(result, events=events, command_ledger=ledger,
        batch_id='timing-only', event_body_ratios=specifications)
        for engine in (plain, observed)]
    assert all(record['update_applied'] for record in records)
    assert plain.stiffness_bfs.density_id == observed.stiffness_bfs.density_id
    assert plain.factors[0].factor_id == observed.factors[0].factor_id
    assert observed.stiffness_bfs.runtime_monitor is monitor
    assert observed.stiffness_bfs.integrator.runtime_monitor is monitor
    for x in (np.full(6, 3.), np.full(6, 4.)):
        left, right = plain.stiffness_bfs.evaluate(x), observed.stiffness_bfs.evaluate(x)
        assert left[0] == right[0]
        np.testing.assert_array_equal(left[1], right[1])
    names = {name for phase, name, _ in monitor.records if phase == 'enter'}
    assert {'stiffness.time_indexed_ratio_validation',
        'stiffness.endpoint_body_ratio_diagnostic', 'stiffness.factor_freeze',
        'stiffness.bfs_update', 'stiffness.evidence.keep',
        'stiffness.laplace.mode_optimization'} <= names
