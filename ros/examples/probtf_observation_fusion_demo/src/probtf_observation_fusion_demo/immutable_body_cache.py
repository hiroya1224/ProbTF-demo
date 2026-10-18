"""Memoize byte-identical core results; share only immutable JSON trees.

Eviction discards evaluated arithmetic, never candidate laws or history.
Every miss uses the unchanged core; failures are never cached.
"""
from collections import OrderedDict
from threading import RLock

import numpy as np

from probtf.spherical_law.first_order import reexpress_state as _core_reexpress_state


class ReexpressionCache:
    def __init__(self, maximum_entries=256):
        if isinstance(maximum_entries, bool) or not isinstance(maximum_entries, int) or maximum_entries < 1:
            raise ValueError('Reexpression memo capacity must be a positive integer.')
        self.maximum_entries = maximum_entries
        self._values = OrderedDict()
        self._lock = RLock()
        self._hits = self._misses = self._evictions = 0

    def clear(self):
        with self._lock:
            self._values.clear()
            self._hits = self._misses = self._evictions = 0

    def statistics(self):
        with self._lock:
            return dict(hits=self._hits, misses=self._misses, evictions=self._evictions,
                        entries=len(self._values), maximum_entries=self.maximum_entries)

    def reexpress(self, state, mean, covariance, version=None, component_id=None):
        mean = np.asarray(mean, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        # The core ID hashes the joint covariance, transform values and
        # metadata. Include separately stored local marginals and zero-chart
        # mean too, so even tolerated marginal roundoff cannot share a key.
        source_marginals = tuple(edge.translation_covariance.tobytes() for edge in state.transforms)
        key = (state.state_id, state.mean.tobytes(), source_marginals,
               mean.shape, mean.tobytes(), covariance.shape,
               covariance.tobytes(), state.version+1 if version is None else int(version),
               state.component_id if component_id is None else str(component_id))
        with self._lock:
            if key in self._values:
                self._hits += 1
                self._values.move_to_end(key)
                return self._values[key]
            self._misses += 1
        result = _core_reexpress_state(state, mean, covariance,
                                       version=version, component_id=component_id)
        with self._lock:
            self._values[key] = result
            self._values.move_to_end(key)
            while len(self._values) > self.maximum_entries:
                self._values.popitem(last=False)
                self._evictions += 1
        return result


_REEXPRESSIONS = ReexpressionCache()
reexpress_state = _REEXPRESSIONS.reexpress


def reexpression_cache_statistics():
    return _REEXPRESSIONS.statistics()


def clear_reexpression_cache():
    _REEXPRESSIONS.clear()


def _immutable(*args, **kwargs):
    raise TypeError('A retained JSON snapshot is immutable.')


class _FrozenDict(dict):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __reduce__(self):
        return freeze_json_record, (dict(self),)


class _FrozenList(list):
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = __iadd__ = __imul__ = _immutable

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __reduce__(self):
        return freeze_json_record, (list(self),)


def freeze_json_record(value):
    """Detach a JSON tree once, preserving dict/list equality and JSON output.

    Nested containers cannot be changed through another result's reference.
    Deepcopy and pickle preserve this immutability across async boundaries.
    """
    if isinstance(value, (_FrozenDict, _FrozenList)):
        return value
    if isinstance(value, dict):
        return _FrozenDict((key, freeze_json_record(item)) for key, item in value.items())
    if isinstance(value, list):
        return _FrozenList(freeze_json_record(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_json_record(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('A retained JSON snapshot contains a non-JSON mutable value: '+type(value).__name__)


def mutable_json_record(value):
    """Detach a public diagnostic from a retained immutable JSON snapshot."""
    if isinstance(value, dict):
        return {key: mutable_json_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [mutable_json_record(item) for item in value]
    return value
