"""Durable acquisition-ordered sensor events, without observation resampling.

Image producers reserve their acquisition slot before asynchronous rendering.
An unfinished earlier reservation blocks later events. ``through_stamp`` is a
producer watermark, not a request to wait for a matching multi-sensor packet.
Each process opens its own store; SQLite transactions serialize producers and
the single inference consumer. A caller-owned checkpoint can be committed with
an acknowledgement, but acknowledging alone does not make external inference
state crash-atomic.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from threading import RLock
import time

import numpy as np


class LateSensorEventError(ValueError):
    """An unreserved event predates the inference cursor and requires replay."""


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + ' must be finite.')
    return value


def _encode(payload):
    blobs = []

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject or value.dtype.fields is not None:
                raise ValueError('Object and structured observation arrays are unsupported.')
            if value.dtype.kind not in 'buif':
                raise ValueError('Observation arrays must have real numeric or boolean dtype.')
            if not np.all(np.isfinite(value)):
                raise ValueError('Observation arrays must be finite.')
            array = np.ascontiguousarray(value)
            index = len(blobs)
            blobs.append(array.tobytes())
            return {'__event_array__': index, 'dtype': array.dtype.str,
                    'shape': list(array.shape)}
        if isinstance(value, (bytes, bytearray, memoryview)):
            index = len(blobs)
            blobs.append(bytes(value))
            return {'__event_bytes__': index}
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise ValueError('Observation payload keys must be strings.')
            if any(key.startswith('__event_') for key in value):
                raise ValueError('Reserved event encoding key in payload.')
            return {key: encode(value[key]) for key in sorted(value)}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise ValueError('Observation payload must contain finite JSON values, arrays or bytes.')

    if not isinstance(payload, dict):
        raise ValueError('Observation payload must be a dictionary.')
    encoded = json.dumps(encode(payload), sort_keys=True, separators=(',', ':'),
                         allow_nan=False)
    digest = hashlib.sha256(encoded.encode())
    for blob in blobs:
        digest.update(len(blob).to_bytes(8, 'big'))
        digest.update(blob)
    return encoded, tuple(blobs), digest.hexdigest()


def _decode(encoded, blobs):
    def decode(value):
        if isinstance(value, dict):
            if '__event_array__' in value:
                return np.frombuffer(blobs[value['__event_array__']],
                    dtype=np.dtype(value['dtype'])).reshape(value['shape'])
            if '__event_bytes__' in value:
                return blobs[value['__event_bytes__']]
            return {key: decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value
    return {} if encoded is None else decode(json.loads(encoded))


@dataclass(frozen=True)
class SensorEvent:
    sequence: int
    event_id: str
    kind: str
    stamp: float
    command: tuple
    status: str
    received_wall_stamp: float
    ready_wall_stamp: object
    worker_id: object
    claimed_wall_stamp: object
    completed_wall_stamp: object
    attempts: int
    payload_hash: object
    payload_json: object
    blobs: tuple
    result_metadata_json: object = None
    error: object = None

    @property
    def payload(self):
        """Decode a fresh mapping; ndarray storage is immutable DB bytes."""
        return _decode(self.payload_json, self.blobs)

    @property
    def result_metadata(self):
        return (None if self.result_metadata_json is None else
                json.loads(self.result_metadata_json))

    @property
    def timing(self):
        """Wall-clock delays; acquisition stamps may use a different clock."""
        return dict(receipt_to_ready_s=(None if self.ready_wall_stamp is None else
                    self.ready_wall_stamp-self.received_wall_stamp),
            ready_to_claim_s=(None if self.claimed_wall_stamp is None or self.ready_wall_stamp is None
                              else self.claimed_wall_stamp-self.ready_wall_stamp),
            processing_s=(None if self.completed_wall_stamp is None or self.claimed_wall_stamp is None
                          else self.completed_wall_stamp-self.claimed_wall_stamp),
            acquisition_stamp=self.stamp,
            acquisition_to_wall_delay_s=None,
            acquisition_and_wall_clocks_are_not_assumed_equal=True)


class SensorEventStore:
    def __init__(self, path):
        self.path = str(path)
        if self.path != ':memory:':
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.connection = sqlite3.connect(self.path, timeout=30., isolation_level=None,
                                         check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('PRAGMA foreign_keys=ON')
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
                stamp REAL NOT NULL, command_json TEXT NOT NULL,
                status TEXT NOT NULL, received_wall_stamp REAL NOT NULL,
                ready_wall_stamp REAL, payload_json TEXT, payload_hash TEXT,
                worker_id TEXT, claimed_wall_stamp REAL, completed_wall_stamp REAL,
                attempts INTEGER NOT NULL DEFAULT 0,
                result_metadata_json TEXT, checkpoint BLOB, error TEXT);
            CREATE TABLE IF NOT EXISTS payload_blobs (
                event_id TEXT NOT NULL REFERENCES events(event_id),
                blob_index INTEGER NOT NULL, value BLOB NOT NULL,
                PRIMARY KEY(event_id, blob_index));
            CREATE INDEX IF NOT EXISTS event_order ON events(stamp, sequence);
        ''')

    def close(self):
        with self._lock:
            self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @contextmanager
    def _transaction(self):
        with self._lock:
            self.connection.execute('BEGIN IMMEDIATE')
            try:
                yield
            except BaseException:
                self.connection.execute('ROLLBACK')
                raise
            else:
                self.connection.execute('COMMIT')

    def _event(self, row):
        if row is None:
            return None
        blobs = tuple(bytes(item[0]) for item in self.connection.execute(
            'SELECT value FROM payload_blobs WHERE event_id=? ORDER BY blob_index',
            (row['event_id'],)))
        return SensorEvent(row['sequence'], row['event_id'], row['kind'], row['stamp'],
            tuple(json.loads(row['command_json'])), row['status'],
            row['received_wall_stamp'], row['ready_wall_stamp'], row['worker_id'],
            row['claimed_wall_stamp'], row['completed_wall_stamp'], row['attempts'],
            row['payload_hash'], row['payload_json'], blobs,
            row['result_metadata_json'], row['error'])

    def get(self, event_id):
        with self._lock:
            return self._event(self.connection.execute(
                'SELECT * FROM events WHERE event_id=?', (str(event_id),)).fetchone())

    def _reserve(self, event_id, kind, stamp, command, wall):
        event_id = str(event_id)
        if not event_id or kind not in ('joint', 'imu', 'image'):
            raise ValueError('A nonempty event ID and joint/imu/image kind are required.')
        stamp = _finite(stamp, 'Acquisition stamp')
        command = np.asarray(command, dtype=float)
        if command.ndim != 1 or not command.size or not np.all(np.isfinite(command)):
            raise ValueError('The sent command must be a finite nonempty vector.')
        command_json = json.dumps(command.tolist(), separators=(',', ':'))
        existing = self.get(event_id)
        if existing is not None:
            if (existing.kind != kind or existing.stamp != stamp or
                    existing.command != tuple(command)):
                raise ValueError('A duplicate event ID has different acquisition data.')
            return existing, False
        cursor = self.connection.execute(
            "SELECT MAX(stamp) FROM events WHERE attempts>0").fetchone()[0]
        late = cursor is not None and stamp < cursor
        self.connection.execute('''INSERT INTO events
            (event_id,kind,stamp,command_json,status,received_wall_stamp,error)
            VALUES(?,?,?,?,?,?,?)''', (event_id, kind, stamp, command_json,
            'late_requires_replay' if late else 'awaiting_payload', wall,
            'Unreserved acquisition predates the claimed inference cursor.' if late else None))
        return self.get(event_id), late

    def reserve(self, event_id, kind, stamp, command, *, received_wall_stamp=None):
        wall = _finite(time.time() if received_wall_stamp is None else received_wall_stamp,
                       'Received wall stamp')
        with self._transaction():
            event, late = self._reserve(event_id, kind, stamp, command, wall)
        if late:
            raise LateSensorEventError(event.error)
        return event

    def _supply(self, event_id, payload, wall):
        event = self.get(event_id)
        if event is None:
            raise KeyError(event_id)
        encoded, blobs, digest = _encode(payload)
        if event.payload_hash is not None:
            if event.payload_hash != digest:
                raise ValueError('A duplicate event ID has different observation payload.')
            return event
        if event.status not in ('awaiting_payload', 'late_requires_replay'):
            raise ValueError('Only an unfilled acquisition slot accepts a payload.')
        self.connection.executemany(
            'INSERT INTO payload_blobs(event_id,blob_index,value) VALUES(?,?,?)',
            [(event_id, index, blob) for index, blob in enumerate(blobs)])
        self.connection.execute('''UPDATE events SET payload_json=?,payload_hash=?,
            ready_wall_stamp=?,status=? WHERE event_id=?''',
            (encoded, digest, wall, 'pending' if event.status == 'awaiting_payload'
             else event.status, event_id))
        return self.get(event_id)

    def supply_payload(self, event_id, payload, *, received_wall_stamp=None):
        wall = _finite(time.time() if received_wall_stamp is None else received_wall_stamp,
                       'Ready wall stamp')
        with self._transaction():
            return self._supply(str(event_id), payload, wall)

    def append(self, event_id, kind, stamp, command, payload, *, received_wall_stamp=None):
        wall = _finite(time.time() if received_wall_stamp is None else received_wall_stamp,
                       'Received wall stamp')
        with self._transaction():
            event, late = self._reserve(event_id, kind, stamp, command, wall)
            event = self._supply(event.event_id, payload, wall)
        if late:
            raise LateSensorEventError(event.error)
        return event

    def claim_next(self, worker_id, *, now_wall_stamp=None, through_stamp=None):
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError('A nonempty worker ID is required.')
        wall = _finite(time.time() if now_wall_stamp is None else now_wall_stamp,
                       'Claim wall stamp')
        cutoff = None if through_stamp is None else _finite(through_stamp, 'Watermark')
        with self._transaction():
            # No second consumer can pass an in-flight event, even if a late
            # producer inserts an earlier reservation after it was claimed.
            if self.connection.execute(
                    "SELECT 1 FROM events WHERE status='processing' LIMIT 1").fetchone():
                return None
            query = "SELECT * FROM events WHERE status!='completed'"
            params = ()
            if cutoff is not None:
                query += ' AND stamp<=?'
                params = (cutoff,)
            row = self.connection.execute(query+' ORDER BY stamp,sequence LIMIT 1', params).fetchone()
            if row is None or row['status'] != 'pending':
                return None
            self.connection.execute('''UPDATE events SET status='processing',worker_id=?,
                claimed_wall_stamp=?,attempts=attempts+1 WHERE event_id=?''',
                (worker_id, wall, row['event_id']))
            return self.get(row['event_id'])

    def _owned(self, event_id, worker_id):
        event = self.get(event_id)
        if event is None:
            raise KeyError(event_id)
        if event.status != 'processing' or event.worker_id != worker_id:
            raise ValueError('The event is not claimed by this inference worker.')
        return event

    def complete(self, event_id, worker_id, *, result_metadata=None, checkpoint=None,
                 now_wall_stamp=None):
        wall = _finite(time.time() if now_wall_stamp is None else now_wall_stamp,
                       'Completion wall stamp')
        metadata = json.dumps(result_metadata, sort_keys=True, allow_nan=False)
        if checkpoint is not None and not isinstance(checkpoint, bytes):
            raise ValueError('A caller-owned checkpoint must be bytes.')
        with self._transaction():
            event = self._owned(str(event_id), worker_id)
            if wall < event.claimed_wall_stamp:
                raise ValueError('Completion precedes the wall-clock claim.')
            self.connection.execute('''UPDATE events SET status='completed',
                completed_wall_stamp=?,result_metadata_json=?,checkpoint=? WHERE event_id=?''',
                (wall, metadata, checkpoint, event.event_id))
            return self.get(event.event_id)

    def fail(self, event_id, worker_id, error, *, now_wall_stamp=None):
        wall = _finite(time.time() if now_wall_stamp is None else now_wall_stamp,
                       'Failure wall stamp')
        with self._transaction():
            event = self._owned(str(event_id), worker_id)
            self.connection.execute('''UPDATE events SET status='failed',error=?,
                completed_wall_stamp=? WHERE event_id=?''', (str(error), wall, event.event_id))
            return self.get(event.event_id)

    def retry(self, event_id):
        """Explicit recovery only; caller restores its matching prior checkpoint."""
        with self._transaction():
            event = self.get(str(event_id))
            if event is None or event.status not in ('failed', 'processing'):
                raise ValueError('Only a failed or abandoned in-flight event can be retried.')
            self.connection.execute('''UPDATE events SET status='pending',worker_id=NULL,
                claimed_wall_stamp=NULL,completed_wall_stamp=NULL,error=NULL WHERE event_id=?''',
                (event.event_id,))
            return self.get(event.event_id)

    def checkpoint(self, event_id):
        with self._lock:
            row = self.connection.execute('SELECT checkpoint FROM events WHERE event_id=?',
                                          (str(event_id),)).fetchone()
            if row is None:
                raise KeyError(event_id)
            return None if row[0] is None else bytes(row[0])

    def events(self):
        with self._lock:
            return tuple(self._event(row) for row in self.connection.execute(
                'SELECT * FROM events ORDER BY stamp,sequence').fetchall())

    def stats(self):
        with self._lock:
            counts = dict(self.connection.execute(
                'SELECT status,COUNT(*) FROM events GROUP BY status').fetchall())
            row = self.connection.execute("SELECT * FROM events WHERE status!='completed' "
                'ORDER BY stamp,sequence LIMIT 1').fetchone()
            cursor = self.connection.execute(
                'SELECT MAX(stamp) FROM events WHERE attempts>0').fetchone()[0]
            return dict(counts=counts, total=sum(counts.values()), claimed_through_stamp=cursor,
                earliest_unfinished=(None if row is None else dict(event_id=row['event_id'],
                    stamp=row['stamp'], sequence=row['sequence'], status=row['status'])),
                ordering='acquisition stamp, then immutable insertion sequence',
                latest_only_selection=False)
