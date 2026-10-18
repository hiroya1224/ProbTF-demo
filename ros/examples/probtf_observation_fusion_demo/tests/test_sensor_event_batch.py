"""Frontier batching preserves every raw event and chronological operator."""

from types import SimpleNamespace

import numpy as np
import pytest

from probtf_observation_fusion_demo.sensor_event_batch import event_requests, ledger_range_record
from probtf_observation_fusion_demo.sensor_event_store import LateSensorEventError, SensorEventStore


def append(store, identifier, stamp, *, kind='imu', index=0, command=(0.,), **payload):
    return store.append(identifier, kind, stamp, command,
                        dict(frame_index=index, **payload), received_wall_stamp=100.)


def test_batch_claims_all_ready_in_acquisition_order_not_insertion_or_latest_order(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        later = append(store, 'late-time', 2., index=2)
        first = append(store, 'first', 1., index=1)
        same = append(store, 'same-time', 1., index=1)
        future = append(store, 'future', 3., index=3)
        batch = store.claim_batch('worker', max_stamp=2., snapshot_id='snapshot', now_wall_stamp=101.)
        assert batch.event_ids == ('first', 'same-time', 'late-time')
        assert [event.sequence for event in batch.events] == [first.sequence, same.sequence, later.sequence]
        assert batch.frontier_sequence == future.sequence
        assert batch.frontier_stamp == 2.
        assert all(event.status == 'processing' and event.attempts == 1 for event in batch.events)
        assert store.get('future').status == 'pending'
        assert batch.record()['event_ids'] == list(batch.event_ids)
        assert not batch.record()['newest_only_selection']


def test_metadata_batch_reads_no_blob_and_raw_pixels_are_loaded_lazily_without_change(tmp_path):
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        original = append(store, 'image', 1., kind='image', image=image)
        statements = []
        store.connection.set_trace_callback(statements.append)
        batch = store.claim_batch('worker', max_stamp=1., snapshot_id='pixels', now_wall_stamp=101.)
        assert not any('payload_blobs' in statement for statement in statements)
        event = batch.events[0]
        assert not event.payload_loaded and not event.blobs
        assert event.payload_metadata['frame_index'] == 0
        assert event.payload_hash == original.payload_hash
        with pytest.raises(ValueError, match='metadata only'):
            _ = event.payload
        loaded = store.get(event.event_id)
        assert loaded.payload_loaded
        np.testing.assert_array_equal(loaded.payload['image'], image)
        assert not loaded.payload['image'].flags.writeable
        assert loaded.payload_hash == event.payload_hash
        assert loaded.sequence == event.sequence
        store.complete(event.event_id, 'worker', result_metadata={'batch_id':batch.batch_id}, now_wall_stamp=102.)
        np.testing.assert_array_equal(store.get('image').payload['image'], image)


def test_unready_reservation_blocks_the_prefix_and_new_arrivals_do_not_extend_it(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        append(store, 'first', 1.)
        store.reserve('reserved', 'image', 2., [0.], received_wall_stamp=100.)
        append(store, 'third', 3.)
        first = store.claim_batch('worker', max_stamp=3., snapshot_id='first', now_wall_stamp=101.)
        assert first.event_ids == ('first',)
        assert first.blocked_event_id == 'reserved'
        assert first.blocked_event_status == 'awaiting_payload'
        store.complete('first','worker',now_wall_stamp=102.)
        blocked = store.claim_batch('worker', max_stamp=3., snapshot_id='blocked', now_wall_stamp=103.)
        assert not blocked and blocked.reason == 'blocked_awaiting_payload'
        store.supply_payload('reserved', {'frame_index':2}, received_wall_stamp=104.)
        ready = store.claim_batch('worker', max_stamp=3., snapshot_id='ready', now_wall_stamp=105.)
        append(store, 'new-frontier', 4., index=4)
        assert ready.event_ids == ('reserved','third')
        assert ready.frontier_sequence < store.get('new-frontier').sequence
        assert store.get('new-frontier').status == 'pending'


def test_an_inflight_batch_cannot_be_passed_by_another_consumer_or_single_claim(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        for i in range(3):
            append(store,str(i),float(i),index=i)
        batch = store.claim_batch('one', max_stamp=1., snapshot_id='one', now_wall_stamp=101.)
        assert batch.event_ids == ('0','1')
        blocked = store.claim_batch('two', max_stamp=2., snapshot_id='two', now_wall_stamp=102.)
        assert not blocked and blocked.reason == 'blocked_processing'
        assert store.claim_next('two') is None
        with pytest.raises(ValueError, match='not claimed'):
            store.complete('0','two')
        store.complete('0','one',now_wall_stamp=103.)
        assert not store.claim_batch('two',max_stamp=2.,snapshot_id='still-blocked')
        store.complete('1','one',now_wall_stamp=104.)
        assert store.claim_batch('two',max_stamp=2.,snapshot_id='after').event_ids == ('2',)


def test_failed_event_and_late_acquisition_are_retained_and_require_explicit_recovery(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        append(store,'one',1.)
        batch = store.claim_batch('worker',max_stamp=1.,snapshot_id='claimed',now_wall_stamp=101.)
        store.fail('one','worker','real arithmetic failure',now_wall_stamp=102.)
        blocked = store.claim_batch('worker',max_stamp=2.,snapshot_id='failed')
        assert not blocked and blocked.reason == 'blocked_failed'
        store.retry('one')
        again = store.claim_batch('worker',max_stamp=1.,snapshot_id='explicit-retry',now_wall_stamp=103.)
        assert again.events[0].attempts == 2
        store.complete('one','worker',now_wall_stamp=104.)
        with pytest.raises(LateSensorEventError):
            append(store,'unreserved-old',.5,gravity=np.array([0.,0.,-1.]))
        late = store.get('unreserved-old')
        assert late.status == 'late_requires_replay'
        np.testing.assert_array_equal(late.payload['gravity'],[0.,0.,-1.])
        assert not store.claim_batch('worker',max_stamp=2.,snapshot_id='late')
        assert store.get('unreserved-old').stamp == .5


def step(index):
    return SimpleNamespace(index=index,start_stamp=float(index),end_stamp=float(index+1),
        start_command=(float(index),),end_command=(float(index+1),),density_id='K-'+str(index))


def test_batch_requests_keep_every_interval_and_same_frame_event_separate(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        append(store,'initial',0.,index=0,command=(0.,))
        append(store,'image',2.,kind='image',index=2,command=(2.,))
        append(store,'imu',2.,index=2,command=(2.,))
        append(store,'last',3.,index=3,command=(3.,))
        batch = store.claim_batch('worker',max_stamp=3.,snapshot_id='ledger')
        requests=event_requests(batch,tuple(step(i) for i in range(3)))
        assert [request[0].event_id for request in requests] == ['initial','image','imu','last']
        assert [[action.index for action in request[2]] for request in requests] == [[],[0,1],[],[2]]
        assert [ledger_range_record(request)['actual_interval_count'] for request in requests] == [0,2,0,1]
        assert ledger_range_record(requests[1])['selected_K_density_ids'] == ['K-0','K-1']


def test_existing_source_frame_batch_uses_all_intervals_and_rejects_missing_or_mismatched_ledger(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        append(store,'next',3.,index=3,command=(3.,))
        batch = store.claim_batch('worker',max_stamp=3.,snapshot_id='suffix')
        request, = event_requests(batch,(step(1),step(2)),previous_frame_index=1)
        assert [action.index for action in request[2]] == [1,2]
        with pytest.raises(ValueError,match='missing'):
            event_requests(batch,(step(2),),previous_frame_index=1)
        with pytest.raises(ValueError,match='duplicate'):
            event_requests(batch,(step(1),step(1),step(2)),previous_frame_index=1)
        bad = step(2);bad.end_command=(99.,)
        with pytest.raises(ValueError,match='endpoint disagree'):
            event_requests(batch,(step(1),bad),previous_frame_index=1)


def test_empty_database_has_an_explicit_frontier_without_fake_events(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        batch = store.claim_batch('worker',max_stamp=0.,snapshot_id='empty')
        assert not batch and len(batch)==0
        assert batch.frontier_sequence == 0
        assert batch.reason == 'no_events_through_frontier'
        assert event_requests(batch,()) == ()


def test_explicit_full_payload_batch_remains_compatible_with_old_event_requests(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        append(store,'imu',0.,gravity=np.array([0.,0.,-1.]))
        batch=store.claim_batch('worker',max_stamp=0.,snapshot_id='full',include_payload=True)
        assert batch.events[0].payload_loaded
        np.testing.assert_array_equal(batch.events[0].payload['gravity'],[0.,0.,-1.])


def test_batch_claim_ownership_is_visible_to_a_separate_database_connection(tmp_path):
    path=tmp_path/'events.sqlite3'
    with SensorEventStore(path) as producer, SensorEventStore(path) as other:
        append(producer,'one',1.)
        append(producer,'two',2.)
        first=producer.claim_batch('first',max_stamp=1.,snapshot_id='one')
        assert first.event_ids == ('one',)
        assert not other.claim_batch('other',max_stamp=2.,snapshot_id='blocked')
        producer.complete('one','first')
        assert other.claim_batch('other',max_stamp=2.,snapshot_id='after').event_ids == ('two',)


def test_batch_materialization_failure_rolls_back_every_claim_without_losing_raw_data(tmp_path, monkeypatch):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        append(store,'one',1.,gravity=np.array([0.,0.,-1.]))
        append(store,'two',2.,gravity=np.array([0.,1.,0.]))
        actual_event=store._event
        def interrupted(row, **kwargs):
            if row is not None and row['event_id']=='two' and kwargs.get('include_payload') is False:
                raise RuntimeError('synthetic interruption before transaction commit')
            return actual_event(row,**kwargs)
        monkeypatch.setattr(store,'_event',interrupted)
        with pytest.raises(RuntimeError,match='before transaction commit'):
            store.claim_batch('worker',max_stamp=2.,snapshot_id='atomic')
        for event_id in ('one','two'):
            restored=store.get(event_id)
            assert restored.status=='pending' and restored.attempts==0
            assert restored.worker_id is None
            assert restored.payload['gravity'].shape==(3,)


def test_initial_command_prior_claim_leaves_all_sensor_payloads_and_clocks_for_next_batch(tmp_path):
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    gravity = np.array([.01, .02, -.99])
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        initial = append(store, 'joint-zero', 0., kind='joint', index=0,
                         command=(.2, -.3), notification='sent_position')
        imu = append(store, 'imu-zero', 0., index=0, command=(.2, -.3), gravity=gravity)
        pixels = append(store, 'image-zero', 0., kind='image', index=0,
                        command=(.2, -.3), image=image)
        next_joint = append(store, 'joint-next', .025, kind='joint', index=1,
                            command=(.21, -.29))
        later = append(store, 'imu-next', .033, index=2,
                       command=(.22, -.28), gravity=-gravity)
        originals = (imu, pixels, next_joint, later)
        prior = store.claim_batch('worker', max_stamp=.033, snapshot_id='pure-prior',
            now_wall_stamp=101., initial_command_only=True)
        assert prior.event_ids == ('joint-zero',)
        assert prior.reason == prior.record()['reason'] == 'initial_command_prior_only'
        assert prior.events[0].stamp == initial.stamp
        assert prior.events[0].command == initial.command
        assert prior.events[0].sequence == initial.sequence
        assert prior.frontier_sequence == later.sequence
        for original in originals:
            remaining = store.get(original.event_id)
            assert remaining.status == 'pending' and remaining.attempts == 0
            assert remaining.worker_id is None
            assert (remaining.stamp, remaining.command, remaining.sequence,
                    remaining.payload_hash) == (original.stamp, original.command,
                        original.sequence, original.payload_hash)
        np.testing.assert_array_equal(store.get('image-zero').payload['image'], image)
        np.testing.assert_array_equal(store.get('imu-zero').payload['gravity'], gravity)
        store.complete(initial.event_id, 'worker', now_wall_stamp=102.)
        ordinary = store.claim_batch('worker', max_stamp=.033, snapshot_id='all-sensors',
                                    now_wall_stamp=103.)
        assert ordinary.reason == 'claimed_ready_prefix'
        assert ordinary.event_ids == tuple(event.event_id for event in originals)
        assert [(event.stamp, event.command, event.sequence, event.payload_hash)
                for event in ordinary.events] == [
                    (event.stamp, event.command, event.sequence, event.payload_hash)
                    for event in originals]


@pytest.mark.parametrize('kind,index', [('imu', 0), ('image', 0), ('joint', 1)])
def test_initial_command_prior_rejects_wrong_first_acquisition_without_claiming_it(tmp_path, kind, index):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        original = append(store, 'wrong-first', 0., kind=kind, index=index, command=(.3,))
        append(store, 'joint-zero', 0., kind='joint', index=0, command=(.3,))
        with pytest.raises(ValueError, match='initial command prior requires'):
            store.claim_batch('worker', max_stamp=0., snapshot_id='wrong-prior',
                              initial_command_only=True)
        remaining = store.get(original.event_id)
        assert remaining.status == 'pending' and remaining.attempts == 0
        assert remaining.payload_hash == original.payload_hash
        assert store.get('joint-zero').status == 'pending'


@pytest.mark.parametrize('completed', [False, True])
def test_initial_command_prior_cannot_be_requested_after_inference_has_claimed_data(tmp_path, completed):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        append(store, 'joint-zero', 0., kind='joint', index=0)
        append(store, 'later', 1., kind='joint', index=0)
        store.claim_batch('worker', max_stamp=0., snapshot_id='first',
                          now_wall_stamp=101., initial_command_only=True)
        if completed:
            store.complete('joint-zero', 'worker', now_wall_stamp=102.)
        with pytest.raises(ValueError, match='unclaimed initial database'):
            store.claim_batch('worker', max_stamp=1., snapshot_id='not-initial',
                              now_wall_stamp=103., initial_command_only=True)
        assert store.get('later').status == 'pending'
        assert store.get('later').attempts == 0


def test_initial_command_prior_waits_for_initial_raw_slot_without_advancing_any_sensor(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        empty = store.claim_batch('worker', max_stamp=0., snapshot_id='not-yet-acquired',
                                  initial_command_only=True)
        assert not empty and empty.reason == 'no_events_through_frontier'
        store.reserve('joint-zero', 'joint', 0., [.2], received_wall_stamp=100.)
        original_imu = append(store, 'imu-zero', 0., command=(.2,), gravity=np.array([0.,0.,-1.]))
        blocked = store.claim_batch('worker', max_stamp=0., snapshot_id='not-ready',
                                    initial_command_only=True)
        assert not blocked and blocked.reason == 'blocked_awaiting_payload'
        assert blocked.blocked_event_id == 'joint-zero'
        assert store.get('imu-zero').attempts == 0
        store.supply_payload('joint-zero', {'frame_index':0}, received_wall_stamp=101.)
        ready = store.claim_batch('worker', max_stamp=0., snapshot_id='ready',
                                  initial_command_only=True)
        assert ready.event_ids == ('joint-zero',)
        assert store.get('imu-zero').payload_hash == original_imu.payload_hash
        assert store.get('imu-zero').status == 'pending'
