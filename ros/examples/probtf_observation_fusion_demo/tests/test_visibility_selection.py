"""Independent visibility selection reuses the actual full task law and K.

The compatibility fixture includes real rendered-image/IMU inference and a
captured estimated-K equilibrium action. These tests isolate selection from
inference without replacing its Gaussian/Bingham objective by synthetic math.
"""

from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.special import logsumexp

from probtf_observation_fusion_demo import selection_gradient as selection_module
from probtf_observation_fusion_demo.running_gradient import GradientPacket, RunningGradientTracker
from probtf_observation_fusion_demo.selection_gradient import selection_gradient_at_command
from probtf_observation_fusion_demo.visibility_factor import VisibilityFactor

from test_visual_approach_integration import problems, controller, forbid_sampling


def _factor(*, enabled=True, weight=2e6, minimum_norm_m=1e-9):
    return VisibilityFactor(dict(enabled=enabled, weight=weight,
                                 minimum_norm_m=minimum_norm_m))


def _packet(record):
    return GradientPacket(**{key: record[key]
        for key in GradientPacket.__dataclass_fields__ if key in record})


@pytest.mark.parametrize('weight', [1e5, 2e6])
def test_positive_factor_preserves_task_J_and_gradient_and_combines_the_same_FD(problems, monkeypatch, weight):
    problem = problems['imu_marker']
    _, _, _, source, mapping, _, _ = problem
    task_controller = controller(problem)
    reference = task_controller.gradient_at_command(source, command_map=mapping).record
    forbid_sampling(monkeypatch)
    record = selection_gradient_at_command(controller(problem), source,
        _factor(weight=weight), command_map=mapping).record
    assert reference['status'] == record['status'] == 'ok', record['reason']
    assert record['objective_at_source'] == reference['objective_at_source']
    np.testing.assert_array_equal(record['task_gradient'], reference['gradient'])
    assert np.array(record['task_gradient']).tobytes() == np.array(reference['gradient']).tobytes()
    assert record['finite_difference_diagnostics'] == reference['finite_difference_diagnostics']
    assert record['source_branches'] == reference['source_branches']
    for key in ('body_state_ids', 'goal_state_ids', 'source_command', 'source_stamp',
                'selected_K', 'parameter_density_id', 'body_log_weights', 'goal_log_weights'):
        assert record[key] == reference[key]
    np.testing.assert_array_equal(record['weighted_view_gradient'],
                                  weight*np.asarray(record['view_gradient']))
    np.testing.assert_array_equal(record['gradient'],
        np.asarray(reference['gradient'])+weight*np.asarray(record['view_gradient']))
    assert record['gradient'] == record['selected_gradient']
    view = record['visibility_factor']
    assert view['status'] == 'ok' and view['penalty'] > 0.
    assert record['log_task_factor'] == -reference['objective_at_source']
    assert record['log_view_factor'] == -weight*view['penalty']
    assert record['log_select_factor'] == record['log_task_factor']+record['log_view_factor']
    assert record['selected_objective_at_source'] == reference['objective_at_source']+weight*view['penalty']
    assert record['view_factor_is_observation'] is False
    assert record['task_candidate_law_changed'] is False
    # Independently apply phi to each actual camera-relative query mean before
    # log-weight averaging; never replace them by one mean point first.
    modes = view['per_mode_record']
    logs = np.array([float(mode['input_log_weight']) for mode in modes])
    weights = np.exp(logs-logsumexp(logs))
    penalties = []
    for mode in modes:
        position = np.asarray(mode['mean'][:3])
        penalties.append(1.-position[2]/np.linalg.norm(position))
    np.testing.assert_allclose(view['penalty'], weights@penalties, atol=1e-15)
    # Visibility finite differences are reconstructed from their own logged
    # values at exactly the original complete-task stencil.
    probe_values = {tuple(item['command']): item['view_factor']['penalty']
                    for item in record['visibility_finite_difference_diagnostics']['evaluations']}
    source_command = np.array(record['source_command'])
    expected = []
    for coordinate in reference['finite_difference_diagnostics']['coordinates']:
        index = coordinate['joint_index']
        hp = coordinate['forward']['step']
        hm = -coordinate['backward']['step']
        plus, minus = source_command.copy(), source_command.copy()
        plus[index] += hp
        minus[index] -= hm
        fp, fm = probe_values[tuple(plus)], probe_values[tuple(minus)]
        f0 = probe_values[tuple(source_command)]
        expected.append((hm*(fp-f0)/hp+hp*(f0-fm)/hm)/(hp+hm))
    np.testing.assert_array_equal(record['view_gradient'], expected)


def test_task_and_view_receive_identical_trial_objects_and_do_not_push_twice(problems, monkeypatch):
    problem = problems['imu_marker']
    _, _, _, source, mapping, _, _ = problem
    current = controller(problem)
    task_states, pushes = {}, []
    original_task_query = current._query
    original_view_query = selection_module.pose_query
    original_push = mapping.push

    def task_query(state):
        task_states[(state.component_id, state.command)] = state
        return original_task_query(state)

    def view_query(state, path, **kwargs):
        assert task_states[(state.component_id, state.command)] is state
        assert path == tuple((edge, not inverse) for edge, inverse in reversed(source.camera_path))+source.marker_path
        assert kwargs['frame'] == 'camera_optical_frame'
        return original_view_query(state, path, **kwargs)

    def push(state, command, body_ids, actions):
        pushes.append((state.state_id, tuple(command)))
        return original_push(state, command, body_ids, actions)

    monkeypatch.setattr(current, '_query', task_query)
    monkeypatch.setattr(selection_module, 'pose_query', view_query)
    monkeypatch.setattr(mapping, 'push', push)
    record = selection_gradient_at_command(current, source, _factor(), command_map=mapping).record
    assert record['status'] == 'ok', record['reason']
    evaluations = record['finite_difference_diagnostics']['objective_evaluation_count']
    count = len(source.stages[-1].goal_components)
    assert len(pushes) == evaluations*count
    assert len(set(pushes)) == len(pushes)
    assert record['view_probe_count'] == evaluations
    for probe in record['visibility_finite_difference_diagnostics']['evaluations']:
        view = probe['view_factor']
        for key in ('body_state_ids', 'goal_state_ids', 'selected_K', 'parameter_density_id',
                    'source_stamp', 'source_command'):
            assert view['source_ids'][key] == record[key]


@pytest.mark.parametrize('enabled,weight', [(False, 2e6), (True, 0.)])
def test_disabled_or_zero_weight_is_bitwise_task_ablation_with_no_extra_pose_query(problems, monkeypatch, enabled, weight):
    problem = problems['imu_marker']
    _, _, _, source, mapping, _, _ = problem
    reference = controller(problem).gradient_at_command(source, command_map=mapping).record

    def forbidden(*args, **kwargs):
        raise AssertionError('Task ablation must not perform a visibility pose query.')

    monkeypatch.setattr(selection_module, 'pose_query', forbidden)
    record = selection_gradient_at_command(controller(problem), source,
        _factor(enabled=enabled, weight=weight), command_map=mapping).record
    assert record['status'] == 'ok', record['reason']
    assert record['objective_at_source'] == record['selected_objective_at_source'] == reference['objective_at_source']
    assert record['gradient'] == record['selected_gradient'] == record['task_gradient'] == reference['gradient']
    assert np.asarray(record['gradient']).tobytes() == np.asarray(reference['gradient']).tobytes()
    assert record['finite_difference_diagnostics'] == reference['finite_difference_diagnostics']
    assert record['source_branches'] == reference['source_branches']
    assert record['visibility_factor']['status'] == 'unit_factor'
    assert record['view_probe_count'] == 0
    assert record['view_gradient'] is None
    assert record['log_view_factor'] == 0.
    assert record['log_select_factor'] == -reference['objective_at_source']
    # The actual second-order command integrator must also be an exact
    # ablation, including nonzero persistent velocity and irregular real dt.
    current = controller(problem)
    trackers = [RunningGradientTracker(source.prior.command, eta=1e-6, gamma=8.,
        joint_lower=current.lower, joint_upper=current.upper,
        initial_velocity=.001*np.arange(1, len(source.prior.command)+1),
        initial_stamp=source.prior.stamp) for _ in range(2)]
    assert trackers[0].accept(_packet(reference), source.prior.stamp)['accepted']
    assert trackers[1].accept(_packet(record), source.prior.stamp)['accepted']
    for dt in (0., 1e-8, .0047, .013, .083, .019):
        first, second = [tracker.advance(dt) for tracker in trackers]
        assert not first.record['halted'] and not second.record['halted']
        assert np.asarray(first.command).tobytes() == np.asarray(second.command).tobytes()
        assert np.asarray(first.velocity).tobytes() == np.asarray(second.velocity).tobytes()


def test_unacquired_view_rejects_new_input_and_retains_previous_finite_gradient(problems):
    problem = problems['imu_marker']
    _, _, _, source, mapping, _, _ = problem
    current = controller(problem)
    reference = selection_gradient_at_command(current, source,
        _factor(weight=0.), command_map=mapping).record
    source = replace(source, records=dict(source.records,
        fixed_goal_memory=dict(source.records['fixed_goal_memory'], available=False)))
    factor = _factor()
    failed = selection_gradient_at_command(current, source, factor, command_map=mapping).record
    assert failed['task_status'] == 'ok'
    assert failed['task_gradient'] == reference['task_gradient']
    assert failed['objective_at_source'] == reference['objective_at_source']
    assert failed['status'] == 'failed'
    assert failed['gradient'] is failed['selected_gradient'] is None
    assert failed['visibility_factor']['status'] == 'unavailable'
    assert failed['reason'] and failed['visibility_factor']['reason']
    tracker = RunningGradientTracker(source.prior.command, eta=1e-6, gamma=8.,
        joint_lower=current.lower, joint_upper=current.upper, initial_stamp=source.prior.stamp)
    previous = _packet(reference)
    assert tracker.accept(previous, source.prior.stamp)['accepted']
    tracker.advance(.01)
    position, velocity = tracker.position, tracker.velocity
    rejected = tracker.accept(_packet(failed), source.prior.stamp+.01)
    assert not rejected['accepted']
    assert tracker.last_packet is previous
    np.testing.assert_array_equal(tracker.position, position)
    np.testing.assert_array_equal(tracker.velocity, velocity)


def test_view_radius_quality_threshold_does_not_reject_finite_selection_gradient(problems):
    problem = problems['imu_marker']
    _, _, _, source, mapping, _, _ = problem
    current = controller(problem)
    reference = selection_gradient_at_command(current, source, _factor(), command_map=mapping).record
    actual = selection_gradient_at_command(current, source,
        _factor(minimum_norm_m=10.), command_map=mapping).record
    assert actual['status'] == reference['status'] == 'ok'
    np.testing.assert_array_equal(actual['selected_gradient'], reference['selected_gradient'])
    np.testing.assert_array_equal(actual['task_gradient'], reference['task_gradient'])
    assert actual['visibility_factor']['penalty'] == reference['visibility_factor']['penalty']
    assert all(mode['position_norm_below_diagnostic_threshold']
               and mode['minimum_norm_is_acceptance_gate'] is False
               for mode in actual['visibility_factor']['per_mode_record'])


def test_factor_weights_do_not_mutate_any_body_goal_cross_K_or_inference_record(problems):
    problem = problems['imu_marker']
    _, inference, inferred, source, mapping, captured, _ = problem
    components = source.stages[-1].components+source.stages[-1].goal_components
    before = [(item.state, item.state.covariance.copy(), item.weight, item.log_weight)
              for item in components]
    source_record = json.dumps(source.records, sort_keys=True)
    inferred_record = json.dumps(inferred.records, sort_keys=True)
    action = mapping.record()
    original_packet = captured.record_json
    records = [selection_gradient_at_command(controller(problem), source,
        _factor(weight=weight), command_map=mapping).record for weight in (0., 1e5, 2e6)]
    assert all(record['status'] == 'ok' for record in records)
    assert records[0]['task_gradient'] == records[1]['task_gradient'] == records[2]['task_gradient']
    assert records[0]['gradient'] != records[-1]['gradient']
    assert json.dumps(source.records, sort_keys=True) == source_record
    assert json.dumps(inferred.records, sort_keys=True) == inferred_record
    assert mapping.record() == action
    assert captured.record_json == original_packet
    for (state, covariance, weight, log_weight), item in zip(before, components):
        assert item.state is state and item.weight == weight and item.log_weight == log_weight
        np.testing.assert_array_equal(item.state.covariance, covariance)
    # This checks a meaningful correlated fixture, not merely zero cross
    # blocks that would survive an accidental independent-marginal rebuild.
    state = source.stages[-1].goal_components[0].state
    body_rows = [6*state.edge_ids.index(edge)+offset
                 for edge in inference.body_ids for offset in range(6)]
    marker = state.edge_ids.index('marker')
    assert np.linalg.norm(state.covariance[np.ix_(body_rows, range(6*marker, 6*marker+6))]) > 0.
