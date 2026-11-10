"""Identifier ordering and arbitrary equality survive the string fast path."""
import numpy as np
import pytest

from probtf.dependency import GaussianLatentStore, GaussianObservationFactor
from probtf.dependency import store as store_module
from probtf.provenance import Provenance


def original_unique(values):
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return tuple(output)


def test_plain_string_sources_retain_first_occurrence_and_input_order():
    values = ("imu/2", "camera/1", "imu/2", "imu/1", "camera/1")
    assert store_module._unique(iter(values)) == ("imu/2", "camera/1", "imu/1")
    assert store_module._unique(()) == ()


def test_nonstring_equality_fallback_never_hashes_or_restarts_a_generator():
    class EqualsString:
        def __hash__(self):
            raise AssertionError("The original equality-only path never hashes this object")
        def __eq__(self, other):
            return isinstance(other, str) and other == "matching"
    marker = EqualsString()
    first = [1, 2]
    values = ("first", "matching", marker, first, [1, 2], {"a": 1}, {"a": 1}, "last", "first")
    expected = original_unique(values)
    consumed = []
    def once():
        for value in values:
            consumed.append(id(value))
            yield value
    actual = store_module._unique(once())
    assert tuple(map(id, actual)) == tuple(map(id, expected))
    assert consumed == list(map(id, values))


def test_ambiguous_general_equality_keeps_original_exception_and_consumption():
    first, second = np.array([1, 2]), np.array([1, 2])
    for unique in (original_unique, store_module._unique):
        consumed = []
        def values():
            for value in (first, second, "not_reached"):
                consumed.append(id(value))
                yield value
        with pytest.raises(ValueError, match="truth value"):
            unique(values())
        assert consumed == [id(first), id(second)]


def test_real_gaussian_update_keeps_numeric_state_and_complete_provenance(monkeypatch):
    identifiers = tuple("raw/imu/P@{}".format(i) for i in range(128))
    original = store_module._unique
    def run(unique):
        monkeypatch.setattr(store_module, "_unique", unique)
        store = GaussianLatentStore()
        store.put_factor("shared", [.1, -.2], [[.4, .05], [.05, .3]], 0.,
            Provenance(source_ids=("prior",) + identifiers[:100], derived_from_edge_ids=("parent",)))
        result = store.apply_observation(GaussianObservationFactor(
            "next", ("shared",), [.3], (np.array([[1., -.5]]),), np.array([[.2]]), 1.,
            Provenance(source_ids=identifiers[90:], derived_from_edge_ids=("parent", "child"))))
        return store.snapshot().factor("shared"), result
    expected, expected_result = run(original_unique)
    actual, result = run(original)
    np.testing.assert_array_equal(actual.mean, expected.mean)
    np.testing.assert_array_equal(actual.covariance, expected.covariance)
    np.testing.assert_array_equal(result.kalman_gain, expected_result.kalman_gain)
    assert actual.provenance == expected.provenance
    assert actual.provenance.source_ids == ("prior",) + identifiers
    assert actual.provenance.derived_from_edge_ids == ("parent", "child")
