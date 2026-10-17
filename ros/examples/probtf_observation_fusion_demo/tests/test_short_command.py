"""A delayed one-update result keeps its own time and cannot run forever."""

from copy import deepcopy

import numpy as np
import pytest

from probtf_observation_fusion_demo.short_command import command_reference


@pytest.fixture
def selected():
    return dict(source_stamp=100., source_command=[0., 2.],
                next_command=[1., 1.], duration=.5)


def test_delayed_delivery_tracks_source_clock_and_has_no_future_updates(selected):
    reference = command_reference(selected, 100.3)
    np.testing.assert_allclose(reference['reference_command'], [.6, 1.4], atol=2e-14)
    np.testing.assert_array_equal(reference['reference_velocity'], [2., -2.])
    assert reference['nominal_end_stamp'] == 100.5
    assert reference['future_IK_updates'] == 0
    assert not reference['past_selected_duration']


@pytest.mark.parametrize('stamp', [100.5, 101., 10000.])
def test_after_duration_reference_holds_selected_position_without_extrapolation(selected, stamp):
    reference = command_reference(selected, stamp)
    np.testing.assert_array_equal(reference['reference_command'], selected['next_command'])
    np.testing.assert_array_equal(reference['reference_velocity'], [0., 0.])
    assert reference['past_selected_duration']


def test_before_start_and_zero_displacement_hold_the_given_position(selected):
    before = command_reference(selected, 99.)
    assert before['before_start']
    np.testing.assert_array_equal(before['reference_command'], selected['source_command'])
    np.testing.assert_array_equal(before['reference_velocity'], [0., 0.])
    selected['next_command'] = list(selected['source_command'])
    stationary = command_reference(selected, 100.25)
    np.testing.assert_array_equal(stationary['reference_command'], selected['source_command'])
    np.testing.assert_array_equal(stationary['reference_velocity'], [0., 0.])


def test_epoch_roundoff_does_not_change_the_held_target(selected):
    selected['source_stamp'] = 1792080000.
    selected['duration'] = .1
    reference = command_reference(selected, selected['source_stamp']+.1)
    assert reference['past_selected_duration']
    np.testing.assert_array_equal(reference['reference_command'], selected['next_command'])


def test_reference_has_no_input_alias_or_randomness(selected, monkeypatch):
    saved = deepcopy(selected)
    def forbidden(*args, **kwargs):
        raise AssertionError('One selected position reference has no random candidates.')
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    reference = command_reference(selected, 100.25)
    reference['reference_command'][0] = 999.
    reference['next_command'][0] = 999.
    assert selected == saved


@pytest.mark.parametrize('field,value', [
    ('source_stamp', np.nan), ('source_stamp', np.inf),
    ('source_command', []), ('source_command', [[0., 1.]]),
    ('source_command', [0., np.nan]), ('next_command', [0.]),
    ('next_command', [np.inf, 0.]), ('duration', 0.),
    ('duration', -1.), ('duration', np.nan), ('duration', np.inf),
])
def test_invalid_selected_command_is_explicit(selected, field, value):
    selected[field] = value
    with pytest.raises(ValueError, match='position command'):
        command_reference(selected, 100.1)


@pytest.mark.parametrize('stamp', [np.nan, np.inf, -np.inf, 'invalid'])
def test_invalid_execution_time_is_explicit(selected, stamp):
    with pytest.raises(ValueError, match='position command'):
        command_reference(selected, stamp)


def test_unrepresentable_nominal_end_is_rejected(selected):
    selected['source_stamp'] = 1e20
    with pytest.raises(ValueError, match='nominal end'):
        command_reference(selected, 1e20)
