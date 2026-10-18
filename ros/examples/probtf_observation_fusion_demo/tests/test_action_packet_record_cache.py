"""Private K record caching preserves numerical action and public ownership."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from probtf_observation_fusion_demo import async_stiffness as module
from probtf_observation_fusion_demo.async_stiffness import (
    FrozenKPacket, PhysicalActionEngine, capture_direct_stiffness_action,
    reanchor_retained_stiffness,
)
from test_async_stiffness import observed, caught_up_result


def new_engine(observed):
    scene,inference,_,_,settings,_ = observed
    return PhysicalActionEngine(scene.urdf,inference.body_ids,settings)


def test_one_full_history_decode_across_new_action_epochs_and_equal_ID_objects(observed,monkeypatch):
    original = observed[-1]
    packet = FrozenKPacket(original.record_json,original.packet_id)
    engine = new_engine(observed)
    loads = module.json.loads
    full_decodes = []

    def count(value,*args,**kwargs):
        if value == packet.record_json:
            full_decodes.append(value)
        return loads(value,*args,**kwargs)

    monkeypatch.setattr(module.json,'loads',count)
    engine.set(packet)
    current = caught_up_result(observed,engine)
    previous = None
    for _ in range(3):
        action = capture_direct_stiffness_action(engine,current,packet,previous)
        engine.set(action)
        previous = action
    # A separately transferred equal string must reuse the same validated
    # numeric record, while receiving its own independent proposal view.
    duplicate = FrozenKPacket((packet.record_json+' ')[:-1],packet.packet_id)
    engine.set(duplicate)
    assert duplicate.proposal == packet.proposal
    assert len(full_decodes) == 1
    assert len(engine._estimated_record_cache) == 1
    assert packet.record_json == original.record_json
    assert packet.packet_id == original.packet_id


def test_public_records_reanchor_density_and_provenance_do_not_alias_private_cache(observed):
    packet = FrozenKPacket(observed[-1].record_json,observed[-1].packet_id)
    engine = new_engine(observed)
    engine.set(packet)
    cached = engine._estimated_record_cache[packet.packet_id][1]
    original_text = json.dumps(cached,sort_keys=True)
    current = caught_up_result(observed,engine)
    action = capture_direct_stiffness_action(engine,current,packet)
    view,_ = reanchor_retained_stiffness(engine,current,action)
    external = view.records['stiffness']
    assert external['retained_prior'] is not cached['posterior']
    external['retained_prior']['density_id'] = 'external mutation'
    external['auxiliary_model']['gravity'][0] = 99.
    external['command_proposal']['selected_K'][0] = -1.
    public = packet.record
    public['body_posterior_state_ids'].append('external ID')
    public['command_proposal']['selected_K'][0] = -2.
    first,second = packet.record,packet.record
    assert first is not second and first['posterior'] is not second['posterior']
    provenance = engine.provenance()
    provenance['selected_K'][0] = -3.
    provenance['captured_body_posterior_state_ids'].append('external source')
    assert json.dumps(cached,sort_keys=True) == original_text
    assert packet.proposal['selected_K'][0] > 0.
    assert engine.provenance()['selected_K'][0] > 0.
    assert 'external source' not in engine.provenance()['captured_body_posterior_state_ids']


def test_reused_packet_ID_cannot_hide_changed_record_or_action_parent(observed):
    packet = observed[-1]
    engine = new_engine(observed)
    engine.set(packet)
    before = engine.provenance()
    changed = packet.record
    changed['command_proposal']['selected_K'][0] += 1.
    text = json.dumps(changed,sort_keys=True,separators=(',',':'),allow_nan=False)
    impostor = FrozenKPacket(text,packet.packet_id)
    with pytest.raises(ValueError,match='hash'):
        engine.set(impostor)
    current = caught_up_result(observed,engine)
    action = capture_direct_stiffness_action(engine,current,packet)
    with pytest.raises(ValueError,match='hash'):
        engine.set(replace(action,estimated_packet=impostor))
    assert engine.provenance() == before


def test_cached_and_uncached_record_reads_give_bitwise_same_action_and_full_body_law(observed):
    packet = observed[-1]
    cached,reference = new_engine(observed),new_engine(observed)

    def read_without_cache(estimated):
        assert hashlib.sha256(estimated.record_json.encode()).hexdigest() == estimated.packet_id
        return estimated.record

    reference._validated_estimated_record = read_without_cache
    cached.set(packet)
    reference.set(packet)
    current = caught_up_result(observed,cached)
    left = capture_direct_stiffness_action(cached,current,packet)
    right = capture_direct_stiffness_action(reference,current,packet)
    assert left.action_json == right.action_json and left.packet_id == right.packet_id
    cached_view,cached_map = reanchor_retained_stiffness(cached,current,left)
    reference_view,reference_map = reanchor_retained_stiffness(reference,current,right)
    q = np.asarray(current.prior.command)+.0002*np.arange(6)
    assert cached.increment(current.prior.command,q) == reference.increment(current.prior.command,q)
    inference = observed[1]
    for item in current.stages[-1].components+current.stages[-1].goal_components:
        first = cached_map.push(item.state,q,inference.body_ids,inference.actions)
        second = reference_map.push(item.state,q,inference.body_ids,inference.actions)
        assert first.state_id == second.state_id
        np.testing.assert_array_equal(first.mean,second.mean)
        np.testing.assert_array_equal(first.covariance,second.covariance)
    assert cached_view.records['stiffness'] == reference_view.records['stiffness']
