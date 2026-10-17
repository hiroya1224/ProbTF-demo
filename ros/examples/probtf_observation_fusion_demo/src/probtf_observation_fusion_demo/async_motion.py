"""Physical coordination with ROS rendering in an independent process."""
from pathlib import Path
from queue import Empty
from threading import RLock
import json
import time
import traceback

import cv2
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import String

# Retain the existing import surface used by headless integrations.
from .async_workers import _append, _freeze_arrays, control_worker
from .async_publication import (
    PublicationPacket, control_key, publication_specification, publication_worker,
)
from .event_motion import AsyncMotionSession


class AsyncMotionLive:
    """Keep physics on its timer; send immutable sources to the ROS process."""
    def __init__(self, observations_path, output, case, autostart, *, imu_target_link=None,
                 max_updates=0, duration=None, control_mode=None, running_gradient_options=None):
        if max_updates:
            raise ValueError('max_updates is a synchronous audit limit; use async_duration_s for async runs.')
        self.lock, self.publication_lock, self.case = RLock(), RLock(), case
        self.observations_path, self.output = Path(observations_path), Path(output)
        self.imu_target_link, self.duration = imu_target_link, duration
        self.control_mode,self.running_gradient_options=control_mode,running_gradient_options
        self.autostart, self.replays = bool(autostart), 0
        self.closed = False
        self.failure_publisher = None
        cv2.setNumThreads(1)
        self.session = AsyncMotionSession(observations_path, output,
            imu_target_link=imu_target_link, duration=duration,control_mode=control_mode,
            running_gradient_options=running_gradient_options)
        self._start_publication()
        self.service = rospy.Service('/observation_fusion/replay', Trigger, self.replay)
        self.physics_timer = rospy.Timer(rospy.Duration(self.session.period), self.advance)
        self.timer = rospy.Timer(rospy.Duration(1./self.session.rates['joint']), self.publish)
        rospy.on_shutdown(self.close)

    def _start_publication(self):
        context = self.session.context
        self.publication_requests, self.publication_replies = context.Queue(), context.Queue()
        self.publication_process = context.Process(target=publication_worker,
            args=(self.publication_requests, self.publication_replies,
                  publication_specification(self.session, self.case)), daemon=True)
        self.publication_process.start()
        self.publication_sequence = self.trajectory_transmitted = 0
        self._sent_control_key = self._sent_preview_index = None
        self.published_control = self.published_image = None
        self.publication_ready = False
        self.publication_status = self.publication_failure = self.publication_stopped = None

    def _read_publication_replies(self):
        while True:
            try:
                message = self.publication_replies.get_nowait()
            except Empty:
                break
            if message.get('failed'):
                self.publication_failure = message
            elif message['kind'] == 'ready':
                self.publication_ready = True
            elif message['kind'] == 'stopped':
                self.publication_stopped = message
            elif message['kind'] == 'published':
                self.publication_status = message
                self.published_control = message['published_control']
                self.published_image = message['published_image']
            else:
                raise ValueError('Unknown independent publication reply.')

    def _poll_publication(self):
        self._read_publication_replies()
        if self.publication_failure is not None:
            raise RuntimeError('Publication worker failed: '+self.publication_failure['traceback'])
        if (not self.closed and not self.publication_process.is_alive()
                and self.publication_process.exitcode is not None):
            raise RuntimeError('Publication worker exited unexpectedly: '+str(self.publication_process.exitcode))

    def _enqueue_publication(self):
        """Capture references only; serialization/drawing occurs in the child."""
        session = self.session
        current, frame = session.current_control, session.frame
        rows = tuple(session.scene.trajectory[self.trajectory_transmitted:])
        events = tuple(session.raw_publications)
        key = None if current is None else control_key(current)
        update = None
        if current is not None and key != self._sent_control_key:
            update = {name: current[name] for name in
                ('snapshot', 'markers_wire', 'capture_index', 'index')}
            update.update(request_id=current.get('request_id'), event_id=current.get('event_id'),
                diagnostics={name: current['diagnostics'][name] for name in
                             ('body_source_state_ids', 'goal_source_state_ids')})
        preview = None
        index = session.preview_capture_index
        if index is not None and index != self._sent_preview_index:
            preview = (index, session.preview_image, session.frames[index].stamp)
        metadata = session.manifest() if current is not None else None
        if metadata is not None:
            metadata['publication_dispatch'] = dict(send_sequence=self.publication_sequence,
                previously_acknowledged_sequence=(None if self.publication_status is None else
                                                 self.publication_status['sequence']),
                control_request_id=None if current is None else current.get('request_id'),
                inference_completion_is_not_publication_completion=True)
        packet = PublicationPacket(sequence=self.publication_sequence,
            enqueued_monotonic_s=time.monotonic(), frame=frame,
            frame_index=len(session.frames)-1,
            trajectory_start=self.trajectory_transmitted, trajectory_rows=rows,
            raw_events=events, metadata=metadata,
            control=update, preview=preview)
        self.publication_requests.put(packet)
        # Historical rows/arrays are never modified after this point. A failed
        # send leaves the acquisition and trajectory cursors intact.
        session.raw_publications = []
        self.publication_sequence += 1
        self.trajectory_transmitted += len(rows)
        self._sent_control_key = key
        if preview is not None:
            self._sent_preview_index = index

    def _stop_publication(self):
        self.publication_requests.put(None)
        deadline = time.monotonic()+5.
        while self.publication_process.is_alive() and time.monotonic() < deadline:
            # Drain ACKs while joining: a full reply pipe must not prevent the
            # child Queue feeder from flushing its terminal acknowledgement.
            self._read_publication_replies()
            self.publication_process.join(timeout=.05)
        self._read_publication_replies()
        if self.publication_process.is_alive():
            self.publication_process.terminate()
            self.publication_process.join(timeout=1.)
        _append(self.session.output/'publication_coordinator_shutdown.jsonl', dict(
            worker_exitcode=self.publication_process.exitcode,
            enqueued_packets=self.publication_sequence,
            acknowledged_packet=None if self.publication_status is None else self.publication_status['sequence'],
            graceful=self.publication_stopped is not None and self.publication_process.exitcode == 0,
            stopped=self.publication_stopped, failure=self.publication_failure,
            raw_observation_database_is_separately_retained=True))
        for queue in (self.publication_requests, self.publication_replies):
            queue.cancel_join_thread()
            queue.close()

    def _fail(self, error):
        for name in ('physics_timer', 'timer'):
            timer = getattr(self, name, None)
            if timer is not None:
                timer.shutdown()
        with self.publication_lock, self.lock:
            if self.closed:
                return
            self.session.failure = dict(failed=True, error=str(error), traceback=traceback.format_exc(),
                publication_failure=self.publication_failure)
            try:
                # Exception-only fallback: a dead publication process cannot
                # replace its last latched status with the actual stop reason.
                self.failure_publisher = rospy.Publisher('/observation_fusion/motion_metadata',
                    String, queue_size=1, latch=True)
                self.failure_publisher.publish(String(json.dumps(self.session.failure)))
            except Exception as publication_error:
                self.session.failure['failure_publication_error'] = str(publication_error)
            _append(self.session.output/'execution_failure.jsonl', self.session.failure)
            if self.publication_process.is_alive():
                self.publication_requests.put(dict(kind='failure', record=self.session.failure))
            self.closed = True
            self.session.close()
            self._stop_publication()
        rospy.logerr('Async execution stopped: %s', error)

    def advance(self, _event):
        try:
            with self.lock:
                if self.closed:
                    return
                self._poll_publication()
                self.session.advance()
                if self.autostart and self.session.current_control is not None:
                    self.session.start()
        except Exception as error:
            self._fail(error)

    def publish(self, _event):
        with self.publication_lock:
            try:
                with self.lock:
                    if self.closed:
                        return
                    self._poll_publication()
                    self._enqueue_publication()
            except Exception as error:
                self._fail(error)

    def replay(self, _request):
        with self.publication_lock, self.lock:
            if self.closed:
                return TriggerResponse(False, 'The asynchronous session is closed.')
            if self.session.started is None:
                self.autostart = True
                ready = self.session.current_control is not None and self.session.start()
                return TriggerResponse(True, 'Starting asynchronous motion.' if ready else 'Starting after initial estimate.')
            self._enqueue_publication()
            self.session.close()
            self._stop_publication()
            self.replays += 1
            path = self.output.with_name(self.output.name+'_replay_{:03d}'.format(self.replays))
            self.session = AsyncMotionSession(self.observations_path, path,
                imu_target_link=self.imu_target_link, duration=self.duration,
                control_mode=self.control_mode,running_gradient_options=self.running_gradient_options)
            self._start_publication()
            self.autostart = True
            return TriggerResponse(True, 'Starting a fresh asynchronous experiment.')

    def close(self):
        for name in ('physics_timer', 'timer'):
            timer = getattr(self, name, None)
            if timer is not None:
                timer.shutdown()
        with self.publication_lock, self.lock:
            if self.closed:
                return
            if self.publication_process.is_alive():
                self._enqueue_publication()
            self.closed = True
            self.session.close()
            self._stop_publication()
