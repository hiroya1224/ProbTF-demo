"""Control reads retain exact values, source hash, and independent ownership."""
import hashlib
import json
import pickle

from probtf_observation_fusion_demo import async_stiffness as module


def test_small_proposal_reads_do_not_redecode_full_history_or_share_mutable_state(monkeypatch):
    record = dict(command_proposal=dict(selected_K=[22.36067977499789, 5.661],
        selected_logK=[3.1073040492110957, 1.733600554], density_id='same-law',
        source_command=[-.15, -1.85], source_stamp=1792080003.861111,
        log_weight=-1000000.0000000002),
        history=[dict(cross=[0., 1.e-250], source_id=str(i)) for i in range(1000)])
    encoded = json.dumps(record, separators=(',', ':'))
    identity = hashlib.sha256(encoded.encode()).hexdigest()
    packet = module.FrozenKPacket(encoded, identity)
    original_loads, full_decodes = json.loads, []

    def observed_loads(value, *args, **kwargs):
        if value == encoded:
            full_decodes.append(value)
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(module.json, 'loads', observed_loads)
    for _ in range(5):
        proposal = packet.proposal
        assert proposal == record['command_proposal']
        proposal['selected_K'][0] = -1.
    assert len(full_decodes) == 1
    assert packet.record_json == encoded and packet.packet_id == identity
    assert packet.record == record
    transferred = pickle.loads(pickle.dumps(packet))
    assert transferred.proposal == record['command_proposal']
    assert transferred.record_json == encoded and transferred.packet_id == identity
    transferred.proposal['source_command'][0] = 999.
    assert packet.proposal == record['command_proposal']


def test_absent_proposal_and_invalid_record_keep_existing_lazy_failure_behavior():
    import pytest
    packet = module.FrozenKPacket('{}', 'deliberately-invalid-hash')
    with pytest.raises(KeyError):
        _ = packet.proposal
    packet = module.FrozenKPacket('invalid json', 'deliberately-invalid-hash')
    with pytest.raises(ValueError):
        _ = packet.record
