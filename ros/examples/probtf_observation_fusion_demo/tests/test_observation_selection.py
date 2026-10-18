"""Overload selection covers the whole history and preserves acquisition order."""
from types import SimpleNamespace

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
    np.testing.assert_allclose(times, [10., 30., 50., 70., 90.], atol=.051)
    assert len(value['unselected_event_ids']) == 996
    assert value['thinning_applied'] and value['unselected_reason'] == 'temporal_thinning'
    assert value['actual_command_ledger_omitted'] is False


def test_irregular_times_choose_each_sensor_separately_and_keep_cross_sensor_order():
    image = events([0., .01, .02, 3., 5., 7., 9.98, 9.99, 10.], offset=100)
    imu = events(np.linspace(0., 10., 101), kind='imu')
    combined = tuple(sorted((*image, *imu), key=lambda event: (event.stamp, event.sequence)))
    value = select_observations(combined, {'marker', 'imu'},
        {'mode': 'uniform_time', 'max_events_per_sensor': {'image': 3, 'imu': 5}})
    assert value['per_sensor']['image']['selected_stamps'] == [3., 5., 7.]
    chosen = set(value['selected_event_ids'])
    assert value['selected_event_ids'] == [event.event_id for event in combined if event.event_id in chosen]
    assert len(chosen) == 8


def test_same_timestamp_uses_sequence_coverage_and_sparse_sensor_is_not_removed():
    value = select_observations(events([1.] * 100), {'marker'},
        {'mode': 'uniform_time', 'max_events_per_sensor': {'image': 4}})
    assert value['selected_event_ids'] == ['image:12', 'image:37', 'image:62', 'image:87']
    sparse = select_observations(events([1.]), {'marker'},
        {'mode': 'uniform_time', 'max_events_per_sensor': {'image': 4}})
    assert sparse['selected_event_ids'] == ['image:0'] and not sparse['thinning_applied']


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
