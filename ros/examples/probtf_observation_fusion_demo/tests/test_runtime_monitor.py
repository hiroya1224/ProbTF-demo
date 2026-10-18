import json
import pytest

from probtf_observation_fusion_demo.runtime_monitor import RuntimeMonitor


def test_pending_stage_is_visible_and_nested_completion_restores_parent(tmp_path):
    monitor = RuntimeMonitor(tmp_path, 'inference')
    with monitor.stage('batch', batch_id='original-id', source_stamp=12.):
        with monitor.stage('evidence', hypothesis='keep'):
            current = json.loads(monitor.current_path.read_text())
            assert current['active_stage']['name'] == 'evidence'
            assert current['event'] == 'stage_started'
        assert json.loads(monitor.current_path.read_text())['active_stage']['name'] == 'batch'
    current = json.loads(monitor.current_path.read_text())
    assert current['active_stage'] is None
    records = [json.loads(line) for line in monitor.path.read_text().splitlines()]
    assert [r['sequence'] for r in records] == list(range(1, len(records)+1))
    completed = [r for r in records if r['event'] == 'stage_finished']
    assert all(r['duration_s'] >= 0 and r['cpu_s'] >= 0 for r in completed)
    assert completed[-1]['context']['batch_id'] == 'original-id'


def test_failure_preserves_original_exception_and_records_stage(tmp_path):
    monitor = RuntimeMonitor(tmp_path, 'control')
    error = ValueError('actual operation failed')
    with pytest.raises(ValueError) as caught:
        with monitor.stage('gradient', request_id=7):
            raise error
    assert caught.value is error
    record = json.loads(monitor.current_path.read_text())
    assert record['outcome'] == 'failed'
    assert record['error'] == 'ValueError: actual operation failed'
    assert record['active_stage'] is None
