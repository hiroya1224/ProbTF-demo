"""Immutable JSON transfer skips no candidate array and changes no source law."""
from dataclasses import replace
import hashlib
import json
import multiprocessing
import pickle

import numpy as np
import pytest

from probtf.spherical_law.first_order import BinghamTransform, JointTransformState, bingham_from_rotation_coordinates
from probtf_observation_fusion_demo import immutable_body_cache as cache
from probtf_observation_fusion_demo.async_workers import _freeze_arrays
from probtf_observation_fusion_demo.body_native import point_query


def state():
    edges = tuple(BinghamTransform('edge'+str(i), [.1, .2, .3],
        bingham_from_rotation_coordinates([1., 0., 0., 0.], .001*np.eye(3)),
        .01*np.eye(3)) for i in range(2))
    source = JointTransformState.from_independent(edges, stamp=2.5,
        command=(.2, -.1), version=8, component_id='retained-law')
    P = source.covariance.copy()
    P[3, 9] = P[9, 3] = .0002
    return replace(source, covariance=P)


def law_digest(source):
    h = hashlib.sha256()
    for array in (source.mean, source.covariance):
        h.update(array.tobytes())
    for edge in source.transforms:
        for array in (edge.translation, edge.translation_covariance,
                      edge.rotation_wxyz, edge.bingham):
            if array is not None:
                h.update(array.tobytes())
    return h.hexdigest()


def spawn_check(connection, source, record):
    try:
        payload = dict(source=source, record=record, normal=dict(array=np.arange(3.)))
        _freeze_arrays(payload)
        query = point_query(source, source.edge_ids)
        mutation_blocked = False
        try:
            record['nested'][0]['id'].append('changed')
        except TypeError:
            mutation_blocked = True
        connection.send(dict(state_id=source.state_id, numeric_sha256=law_digest(source),
            record=json.dumps(record, sort_keys=True), mutation_blocked=mutation_blocked,
            flags=(source.covariance.flags.writeable,
                   source.transforms[0].bingham.flags.writeable,
                   payload['normal']['array'].flags.writeable),
            query_state_id=query.state_id, query_sha256=hashlib.sha256(query.mean.tobytes()+query.covariance.tobytes()).hexdigest()))
    finally:
        connection.close()


def test_new_pickle_reducer_preserves_nested_values_aliases_and_immutability():
    nested = cache.freeze_json_record(dict(id=['event:1'], values=[1., float('nan'), float('inf'), -float('inf')]))
    original = cache.freeze_json_record(dict(nested=[nested, nested], tuple=(dict(a=2), [3, None])))
    restored = pickle.loads(pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL))
    assert restored['nested'][0] is restored['nested'][1]
    assert json.dumps(restored, sort_keys=True) == json.dumps(original, sort_keys=True)
    assert cache.is_frozen_json_record(restored)
    for value in (original, restored):
        with pytest.raises(TypeError):
            value['nested'][0]['id'].append('new')
        with pytest.raises(TypeError):
            value['tuple'][1][0] = 7
        encoded = json.dumps(value, sort_keys=True)
        value.__init__({'new': []})
        value['nested'].__init__(['new'])
        assert json.dumps(value, sort_keys=True) == encoded


@pytest.mark.parametrize('bad', [np.array([1.]), {'value': np.zeros(2)},
                                  ([{'array': np.zeros((1, 1))}],), {1, 2}, object()])
def test_first_construction_still_rejects_non_JSON_values_in_every_container(bad):
    for construct in (cache.freeze_json_record,
                      lambda value: cache._FrozenDict(dict(value=value)),
                      lambda value: cache._FrozenList([value])):
        with pytest.raises(TypeError, match='non-JSON'):
            construct(bad)


def test_old_pickles_calling_freeze_json_record_remain_compatible(monkeypatch):
    original = cache.freeze_json_record(dict(nested=[dict(raw_id='image:7', command=[.2, -.1])]))
    with monkeypatch.context() as context:
        context.setattr(cache._FrozenDict, '__reduce__',
                        lambda self: (cache.freeze_json_record, (dict(self),)))
        context.setattr(cache._FrozenList, '__reduce__',
                        lambda self: (cache.freeze_json_record, (list(self),)))
        old_encoded = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)
    assert b'freeze_json_record' in old_encoded
    restored = pickle.loads(old_encoded)
    assert json.dumps(restored, sort_keys=True) == json.dumps(original, sort_keys=True)
    assert cache.is_frozen_json_record(restored['nested'][0])
    with pytest.raises(TypeError):
        restored['nested'][0]['command'].append(.3)


def test_JSON_scalar_subclass_cannot_hide_mutable_array_attributes():
    class DiagnosticFloat(float):
        pass
    value = DiagnosticFloat(float('inf'))
    value.mutable = np.ones(2)
    frozen = cache.freeze_json_record(dict(value=value))
    assert type(frozen['value']) is float
    assert json.dumps(frozen) == json.dumps(dict(value=value))
    assert not hasattr(frozen['value'], 'mutable')


def test_array_restore_skips_only_validated_JSON_and_retains_normal_dict_and_core_arrays():
    source = state()
    record = cache.freeze_json_record(dict(ids=['imu:1'], nested=[dict(stamp=2.5)]))
    array = np.arange(4.)
    payload = pickle.loads(pickle.dumps(dict(source=source, record=record,
        ordinary=dict(array=array), shared=(array, array))))
    assert payload['source'].covariance.flags.writeable
    assert payload['ordinary']['array'].flags.writeable
    _freeze_arrays(payload)
    assert not payload['source'].covariance.flags.writeable
    assert not payload['source'].transforms[0].bingham.flags.writeable
    assert not payload['ordinary']['array'].flags.writeable
    assert payload['shared'][0] is payload['shared'][1] is payload['ordinary']['array']
    assert law_digest(payload['source']) == law_digest(source)
    assert payload['source'].state_id == source.state_id
    np.testing.assert_array_equal(payload['source'].covariance, source.covariance)
    assert payload['source'].covariance[3, 9] == .0002
    # An arbitrary subclass is not a validated JSON-only marker.
    class OrdinaryDictionary(dict):
        pass
    ordinary = OrdinaryDictionary(array=np.ones(2))
    assert not cache.is_frozen_json_record(ordinary)
    _freeze_arrays(ordinary)
    assert not ordinary['array'].flags.writeable


def test_spawn_roundtrip_preserves_law_source_JSON_and_all_array_readonly_flags():
    source = state()
    record = cache.freeze_json_record(dict(nested=[dict(id=['image:2'])], command=[.2, -.1]))
    query = point_query(source, source.edge_ids)
    context = multiprocessing.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=spawn_check, args=(send, source, record))
    process.start()
    send.close()
    try:
        assert receive.poll(20.), 'spawn worker did not return its immutable-source check'
        actual = receive.recv()
        process.join(20.)
        assert process.exitcode == 0
    finally:
        receive.close()
        if process.is_alive():
            process.terminate()
            process.join(2.)
    assert actual['state_id'] == actual['query_state_id'] == source.state_id
    assert actual['numeric_sha256'] == law_digest(source)
    assert actual['record'] == json.dumps(record, sort_keys=True)
    assert actual['mutation_blocked']
    assert actual['flags'] == (False, False, False)
    assert actual['query_sha256'] == hashlib.sha256(query.mean.tobytes()+query.covariance.tobytes()).hexdigest()
