"""Actual SQLite rows preserve every acquired event and its processing order."""

from dataclasses import FrozenInstanceError
from concurrent.futures import ThreadPoolExecutor
import pickle
import sqlite3

import numpy as np
import pytest

from probtf_observation_fusion_demo.sensor_event_store import (
    LateSensorEventError, SensorEventStore,
)


COMMAND = [0., .1, .2, .3, .4, .5]


def append(store, identifier, stamp, kind='imu', payload=None):
    return store.append(identifier, kind, stamp, COMMAND,
        {'gravity_in_sensor': [0., 0., -1.]} if payload is None else payload,
        received_wall_stamp=100.)


def test_roundtrip_arrays_bytes_nested_values_reopen_and_process_pickle(tmp_path):
    path = tmp_path/'events.sqlite3'
    pixels = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    with SensorEventStore(path) as store:
        event = append(store, 'image:1', 1., 'image',
            {'image': pixels[:, ::-1], 'time': 0., 'nested': {'bytes': b'raw\x00'}})
        pixels[:] = 0
        with pytest.raises(FrozenInstanceError):
            event.stamp = 4.
    with SensorEventStore(path) as store:
        actual = pickle.loads(pickle.dumps(store.get('image:1')))
        np.testing.assert_array_equal(actual.payload['image'],
            np.arange(60, dtype=np.uint8).reshape(4, 5, 3)[:, ::-1])
        assert actual.payload['nested']['bytes'] == b'raw\x00'
        assert not actual.payload['image'].flags.writeable
        actual.payload['nested']['bytes'] = b'changed'
        assert actual.payload['nested']['bytes'] == b'raw\x00'
        assert store.connection.execute('SELECT COUNT(*) FROM payload_blobs').fetchone()[0] == 2


def test_acquisition_order_watermark_and_one_global_inflight(tmp_path):
    path = tmp_path/'events.sqlite3'
    with SensorEventStore(path) as producer, SensorEventStore(path) as worker:
        append(producer, 'late_arrival', 3.)
        append(producer, 'first', 1.)
        append(producer, 'equal_second', 1., 'joint', {})
        append(producer, 'middle', 2.)
        assert worker.claim_next('w', through_stamp=.9) is None
        expected = ['first', 'equal_second', 'middle', 'late_arrival']
        for i, identifier in enumerate(expected):
            claimed = worker.claim_next('w', now_wall_stamp=101.+i, through_stamp=3.)
            assert claimed.event_id == identifier
            assert producer.claim_next('different worker') is None
            worker.complete(identifier, 'w', now_wall_stamp=101.5+i,
                            result_metadata={'index': i}, checkpoint=b'checkpoint')
        assert worker.claim_next('w') is None
        assert worker.stats()['counts'] == {'completed': 4}
        assert worker.checkpoint('first') == b'checkpoint'
        assert worker.get('middle').result_metadata == {'index': 2}
        assert worker.get('middle').timing['ready_to_claim_s'] == 3.
        assert worker.get('middle').timing['processing_s'] == .5
        assert worker.get('middle').timing['acquisition_to_wall_delay_s'] is None


def test_earliest_image_reservation_blocks_later_imu_without_packet_matching(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        store.reserve('image', 'image', 1., COMMAND, received_wall_stamp=100.)
        append(store, 'imu', 1.01)
        assert store.claim_next('w', through_stamp=2.) is None
        assert store.stats()['earliest_unfinished']['status'] == 'awaiting_payload'
        store.supply_payload('image', {'image': np.zeros((3, 3), np.uint8)},
                             received_wall_stamp=101.)
        assert store.claim_next('w', through_stamp=2.).event_id == 'image'
        store.complete('image', 'w')
        assert store.claim_next('w', through_stamp=2.).event_id == 'imu'


def test_one_ready_imu_proceeds_without_waiting_for_future_image(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        append(store, 'imu', 1.)
        store.reserve('future image', 'image', 1.1, COMMAND)
        assert store.claim_next('w', through_stamp=1.).event_id == 'imu'


def test_exact_duplicates_are_idempotent_and_mutations_rejected(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        expected = append(store, 'imu', 1.)
        assert append(store, 'imu', 1.) == expected
        assert store.stats()['total'] == 1
        with pytest.raises(ValueError, match='acquisition'):
            append(store, 'imu', 2.)
        with pytest.raises(ValueError, match='payload'):
            append(store, 'imu', 1., payload={'gravity_in_sensor': [0., 1., 0.]})
        assert store.stats()['total'] == 1
        assert store.get('imu').payload == expected.payload


def test_late_unreserved_acquisition_is_durable_and_never_applied_out_of_order(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        append(store, 'processed', 2.)
        store.claim_next('w')
        store.complete('processed', 'w')
        with pytest.raises(LateSensorEventError, match='predates'):
            append(store, 'late', 1.)
        assert store.get('late').status == 'late_requires_replay'
        assert store.get('late').payload['gravity_in_sensor'] == [0., 0., -1.]
        append(store, 'next', 3.)
        assert store.claim_next('w') is None
        assert store.stats()['total'] == 3


def test_transaction_rolls_back_invalid_new_payload(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        with pytest.raises(ValueError):
            append(store, 'bad', 1., payload={'bad': np.array([float('nan')])})
        assert store.get('bad') is None
        assert store.stats()['total'] == 0


@pytest.mark.parametrize('payload', [[], {'bad': object()}, {2: 3},
    {'bad': np.array([object()])}, {'bad': float('inf')}, {'__event_array__': 1}])
def test_invalid_payloads_rejected(tmp_path, payload):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        with pytest.raises(ValueError):
            append(store, 'bad', 1., payload=payload)
        assert store.stats()['total'] == 0


def test_worker_ownership_failed_event_blocks_and_explicit_retry(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        append(store, 'first', 1.)
        append(store, 'next', 2.)
        store.claim_next('owner', now_wall_stamp=101.)
        with pytest.raises(ValueError, match='worker'):
            store.complete('first', 'other', now_wall_stamp=102.)
        store.fail('first', 'owner', 'known failure', now_wall_stamp=102.)
        assert store.get('first').error == 'known failure'
        assert store.claim_next('owner') is None
        store.retry('first')
        assert store.claim_next('owner').attempts == 2
        store.complete('first', 'owner')
        assert store.claim_next('owner').event_id == 'next'


def test_sqlite_integrity_and_all_rates_retained_without_latest_drop(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        records = [(i/hz, kind, i) for kind, hz in
                   [('joint', 40), ('imu', 30), ('image', 12)] for i in range(hz)]
        # Arrival order intentionally differs from acquisition order.
        for stamp, kind, index in reversed(records):
            append(store, '{}:{}'.format(kind, index), stamp, kind, {'time': stamp})
        actual = []
        while True:
            event = store.claim_next('worker', through_stamp=1.)
            if event is None:
                break
            actual.append((event.stamp, event.sequence, event.kind))
            store.complete(event.event_id, 'worker')
        assert len(actual) == 82
        assert actual == sorted(actual)
        assert {kind: sum(item[2] == kind for item in actual) for kind in
                ('joint', 'imu', 'image')} == {'joint': 40, 'imu': 30, 'image': 12}
        assert store.connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'


def test_store_constructed_on_main_thread_accepts_serialized_ros_timer_threads(tmp_path):
    with SensorEventStore(tmp_path/'events.sqlite3') as store:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda index: append(store, 'imu:{}'.format(index), index/30.),
                          range(30)))
        def process():
            claimed = []
            while True:
                event = store.claim_next('timer-thread')
                if event is None:
                    return claimed
                claimed.append(event.stamp)
                store.complete(event.event_id, 'timer-thread')
        with ThreadPoolExecutor(max_workers=1) as pool:
            actual = pool.submit(process).result()
        assert actual == [index/30. for index in range(30)]
        assert store.stats()['counts'] == {'completed': 30}
