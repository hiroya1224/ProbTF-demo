"""Overload selection covers the whole history and preserves acquisition order."""
from types import SimpleNamespace
from itertools import combinations

import numpy as np
import pytest

from probtf_observation_fusion_demo.observation_selection import select_observations


def events(times, kind='image', offset=0):
    return tuple(SimpleNamespace(event_id=f'{kind}:{i}', kind=kind, stamp=float(stamp),
                                 sequence=i+offset, command=(.001*i,))
                 for i, stamp in enumerate(times))


def test_default_uses_every_enabled_observation_and_does_not_read_payloads():
    image = events(np.linspace(0., 10., 101))
    value = select_observations(image, {'marker'})
    assert value['selected_event_ids'] == [event.event_id for event in image]
    assert not value['thinning_applied']
    assert value['original_raw_retained']
    assert not value['selection_uses_observed_values']


def test_capacity_covers_the_entire_pending_history_instead_of_a_prefix_or_suffix():
    image = events(np.linspace(0., 100., 1001))
    value = select_observations(image, {'marker'},
        {'mode': 'uniform_time', 'max_events_per_sensor': {'image': 5}})
    times = value['per_sensor']['image']['selected_stamps']
    np.testing.assert_allclose(times, [0., 25., 50., 75., 100.], atol=.051)
    assert value['per_sensor']['image']['target_times']==[0., 25., 50., 75., 100.]
    assert value['per_sensor']['image']['endpoint_times_included']
    assert len(value['unselected_event_ids']) == 996
    assert value['thinning_applied'] and value['unselected_reason'] == 'temporal_thinning'
    assert value['actual_command_ledger_omitted'] is False


def test_irregular_times_choose_each_sensor_separately_and_keep_cross_sensor_order():
    image = events([0., .01, .02, 3., 5., 7., 9.98, 9.99, 10.], offset=100)
    imu = events(np.linspace(0., 10., 101), kind='imu')
    combined = tuple(sorted((*image, *imu), key=lambda event: (event.stamp, event.sequence)))
    value = select_observations(combined, {'marker', 'imu'},
        {'mode': 'uniform_time', 'max_events_per_sensor': {'image': 3, 'imu': 5}})
    assert value['per_sensor']['image']['selected_stamps'] == [0., 5., 10.]
    chosen = set(value['selected_event_ids'])
    assert value['selected_event_ids'] == [event.event_id for event in combined if event.event_id in chosen]
    assert len(chosen) == 8


def test_same_timestamp_uses_sequence_coverage_and_sparse_sensor_is_not_removed():
    value = select_observations(events([1.] * 100), {'marker'},
        {'mode': 'uniform_time', 'max_events_per_sensor': {'image': 4}})
    assert value['selected_event_ids'] == ['image:12', 'image:37', 'image:62', 'image:87']
    assert value['per_sensor']['image']['identical_timestamp_sequence_quantiles']
    sparse = select_observations(events([1.]), {'marker'},
        {'mode': 'uniform_time', 'max_events_per_sensor': {'image': 4}})
    assert sparse['selected_event_ids'] == ['image:0'] and not sparse['thinning_applied']


def test_two_midpoint_adjacent_images_cannot_replace_both_frontier_ends():
    value = select_observations(events([0., 49.9, 50.1, 100.]), {'marker'},
        {'mode':'uniform_time', 'max_events_per_sensor':{'image':2}})
    assert value['per_sensor']['image']['selected_stamps']==[0., 100.]
    assert value['per_sensor']['image']['target_times']==[0., 100.]
    assert value['selection_algorithm']=='endpoint_constrained_ordered_least_squares_v2'
    assert value['unselected_event_ids']==['image:1', 'image:2']


def test_separated_time_clusters_are_not_consumed_by_forward_greedy_selection():
    value = select_observations(events([0., .01, .02, 100., 100.01, 100.02]), {'marker'},
        {'mode':'uniform_time', 'max_events_per_sensor':{'image':4}})
    assert value['per_sensor']['image']['selected_stamps']==[0., .02, 100., 100.02]


@pytest.mark.parametrize('times,count',[
    ([0., 1., 9., 14., 16.],4),
    ([0., .01, .02, 100., 100.01, 100.02],4),
    ([0., 1., 1., 1.1, 5., 5., 9., 10.],5),
    ([0., 1., 8., 9., 10.],3),
])
def test_ordered_assignment_minimizes_the_full_time_residual_not_each_greedy_residual(times,count):
    value = select_observations(events(times), {'marker'},
        {'mode':'uniform_time', 'max_events_per_sensor':{'image':count}})
    chosen = value['per_sensor']['image']['selected_stamps']
    target = np.asarray(value['per_sensor']['image']['target_times'])
    costs = [float(np.sum((np.asarray([times[0], *(times[index] for index in interior), times[-1]])-target)**2))
             for interior in combinations(range(1,len(times)-1),count-2)]
    assert float(np.sum((np.asarray(chosen)-target)**2))==pytest.approx(min(costs),abs=1e-12)
    assert chosen[0]==times[0] and chosen[-1]==times[-1]
    ids=value['selected_event_ids']
    assert len(ids)==count and len(set(ids))==count
    assert [int(identity.split(':')[1]) for identity in ids]==sorted(int(identity.split(':')[1]) for identity in ids)


def test_global_assignment_corrects_the_endpoint_greedy_counterexample():
    value = select_observations(events([0.,1.,9.,14.,16.]), {'marker'},
        {'mode':'uniform_time', 'max_events_per_sensor':{'image':4}})
    # Choosing 9 first is individually closer to 16/3, but forces 14 for
    # 32/3. The total squared time error is smaller for the ordered pair 1,9.
    assert value['per_sensor']['image']['selected_stamps']==[0.,1.,9.,16.]


def test_singleton_capacity_uses_the_central_time_and_sequence_breaks_ties():
    value = select_observations(events([0.,4.,6.,10.]), {'marker'},
        {'mode':'uniform_time', 'max_events_per_sensor':{'image':1}})
    assert value['selected_event_ids']==['image:1']
    assert value['per_sensor']['image']['target_times']==[5.]
    identical = select_observations(events([3.]*9), {'marker'},
        {'mode':'uniform_time', 'max_events_per_sensor':{'image':1}})
    assert identical['selected_event_ids']==['image:4']


def test_extreme_finite_timestamps_keep_the_same_scaled_assignment_without_overflow():
    value = select_observations(events([-1e308,-5e307,5e307,1e308]), {'marker'},
        {'mode':'uniform_time', 'max_events_per_sensor':{'image':3}})
    assert value['per_sensor']['image']['selected_stamps']==[-1e308,-5e307,1e308]
    assert np.isfinite(value['per_sensor']['image']['target_times']).all()


def test_selection_does_not_modify_original_commands_or_source_order():
    original=events([0.,1.,9.,14.,16.])
    before=tuple((item.event_id,item.stamp,item.sequence,item.command) for item in original)
    value=select_observations(original, {'marker'},
        {'mode':'uniform_time', 'max_events_per_sensor':{'image':4}})
    assert before==tuple((item.event_id,item.stamp,item.sequence,item.command) for item in original)
    assert value['original_time_command_sequence_retained']
    assert value['original_raw_retained']
    assert value['actual_command_ledger_omitted'] is False
    assert value['bfs_or_K_update_for_unselected_observations'] is False


@pytest.mark.parametrize('capacity', [0, -1, 1.5, True])
def test_invalid_capacity_is_a_configuration_error(capacity):
    with pytest.raises(ValueError, match='positive'):
        select_observations(events([0., 1.]), {'marker'},
            {'mode': 'uniform_time', 'max_events_per_sensor': {'image': capacity}})


def test_disabled_sensors_are_not_labelled_temporally_thinned():
    value = select_observations(events([0., 1.]), {'imu'},
        {'mode': 'uniform_time', 'max_events_per_sensor': {'image': 1}})
    assert value['selected_event_ids'] == value['unselected_event_ids'] == []
    assert not value['thinning_applied']
