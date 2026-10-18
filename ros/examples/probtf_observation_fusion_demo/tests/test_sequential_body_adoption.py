"""Several sequential results in one frontier do not double-count events/K."""
from types import SimpleNamespace

from probtf_observation_fusion_demo.event_motion import AsyncMotionSession


def test_each_ready_adopts_only_its_original_event_and_preserves_selected_K(tmp_path):
    session = object.__new__(AsyncMotionSession)
    session.output = tmp_path
    session.clock = lambda: 10.
    session.frame = SimpleNamespace(stamp=20.)
    acquisitions = tuple(SimpleNamespace(event_id=identity, kind=kind,
        sequence=i, stamp=float(i), command=(.01*i,), payload_metadata={'frame_index': i})
        for i, (identity, kind) in enumerate((('imu:0', 'imu'), ('image:0', 'image'), ('imu:1', 'imu'))))
    session.pending_inference_batch = SimpleNamespace(events=acquisitions,
        event_ids=tuple(event.event_id for event in acquisitions))
    session.pending_inference_batch_id = 'batch'
    session.inference_busy = True
    session.joint_body_batch = False
    session.enabled_sensors = {'imu', 'marker'}
    session.assimilated, session.estimate_times = [], []
    session.stiffness_jobs_submitted = set()
    session.latest_packet = SimpleNamespace(packet_id='previous-completed-K')
    session.latest_marker_observed = False
    session.last_assimilated_event = None
    session.last_observation_selection = None

    def accept(offset, job, marker=False):
        event = acquisitions[offset]
        session.batch_progress_count = offset
        session._adopt_body_ready(dict(batch_id='batch',
            result=SimpleNamespace(prior=SimpleNamespace(stamp=event.stamp, command=event.command)),
            capture_index=offset, event_id=event.event_id,
            event_sequence=event.sequence, capture_stamp=event.stamp,
            sensor_kind=event.kind, marker_observed=marker,
            assimilated_event_ids=[event.event_id], body_candidate_id='body:'+str(offset),
            stiffness_job_sequence=job, body_update_seconds=.01))

    accept(0, 1)
    assert session.assimilated == ['imu:0']
    assert not session.latest_marker_observed
    accept(1, 2, marker=True)
    assert session.assimilated == ['imu:0', 'image:0']
    assert session.latest_marker_observed
    accept(2, None)
    assert session.assimilated == ['imu:0', 'image:0', 'imu:1']
    assert session.latest_marker_observed
    assert session.latest_packet.packet_id == 'previous-completed-K'
    assert session.stiffness_jobs_submitted == {1, 2}
    assert session.last_assimilated_event == 'imu:1'
