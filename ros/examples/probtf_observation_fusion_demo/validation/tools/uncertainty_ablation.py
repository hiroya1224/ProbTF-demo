#!/usr/bin/env python3
"""Isolated, explicitly truth-assisted, closed-loop uncertainty experiments.

This diagnostic does not edit/import a replacement production controller. It
uses AsyncMotionSession, its five existing workers, native dynamics, complete
task/view likelihood, local quadratic tracker and the real asynchronous clock.
Only the named input is replaced at its immutable request boundary. Baseline
requests and numerical functions are passed through without transformation.

Run raw/DB/image archives belong in /tmp, never in the repository. Timing is
measured rather than claimed identical: different closed-loop observations and
wall-clock worker scheduling are an unavoidable second-order confound. Pair
these experiments with fixed-input probes before assigning sole causality.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import signal
import time
import traceback

import numpy as np


TOOLS = Path(__file__).resolve().parent
PACKAGE = TOOLS.parents[1]
if not (PACKAGE/'config/moving_estimation.yaml').is_file():
    import rospkg
    PACKAGE=Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
MODES = ('baseline', 'oracle_goal', 'oracle_marker', 'true_K', 'oracle_body',
         'all_oracle', 'known_landmark','oracle_body_goal')


def _frozen_helpers(output):
    path=Path(output)/'diagnostic_source'
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0,str(path))


def _append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, separators=(',', ':'))+'\n')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class QueueProxy:
    """Only this diagnostic's process boundary is wrapped; queue stays intact."""
    def __init__(self, queue, *, on_get=None, on_put=None):
        self.queue, self.on_get, self.on_put = queue, on_get, on_put

    def get(self, *args, **kwargs):
        value = self.queue.get(*args, **kwargs)
        return self.on_get(value) if self.on_get is not None else value

    def get_nowait(self):
        return self.get(False)

    def put(self, value, *args, **kwargs):
        if self.on_put is not None:
            value = self.on_put(value)
        return self.queue.put(value, *args, **kwargs)

    def __getattr__(self, key):
        return getattr(self.queue, key)


def control_worker(mode, requests, replies, description, urdf, templates, output,
                   horizon, control_mode=None, visibility_options=None,
                   artifact_requests=None):
    _frozen_helpers(output)
    import probtf_observation_fusion_demo.async_workers as workers
    from probtf_observation_fusion_demo.moving_inference import MovingInference
    from probtf_observation_fusion_demo.async_stiffness import PhysicalActionEngine
    import yaml

    original_predict = workers.predict_result
    original_freeze = workers.freeze_work_item
    context = {}
    original_packets = {}
    oracle_packets = {}
    engine = None
    inference = (MovingInference(description, templates) if mode in
        ('oracle_goal', 'oracle_body', 'oracle_body_goal','true_K', 'all_oracle') else None)
    if mode in ('true_K', 'all_oracle'):
        settings = yaml.safe_load((PACKAGE/'config/stiffness_identification.yaml').read_text())
        engine = PhysicalActionEngine(Path(urdf), tuple(inference.body_ids), settings)

    def receive(request):
        context.clear()
        if request is None:
            return None
        context['request'] = request
        if mode in ('true_K', 'all_oracle'):
            from uncertainty_ablation_states import oracle_K_packet
            incoming = request.get('packet')
            if incoming is not None:
                original_packets[incoming.packet_id] = incoming
            original_id = request.get('parameter_packet_id')
            original = original_packets.get(original_id)
            # Preserve availability: no invented K before the first completed
            # production K estimate, even though its value is known to us.
            if original is not None:
                if original.packet_id not in oracle_packets:
                    oracle_packets[original.packet_id] = oracle_K_packet(
                        engine, request['result'], original, request['_diagnostic_truth'])
                oracle = oracle_packets[original.packet_id]
                context['oracle_packet'] = oracle
                request = dict(request, packet=oracle,
                               parameter_packet_id=oracle.packet_id)
        return request

    def predicted(*args, **kwargs):
        value = original_predict(*args, **kwargs)
        if mode in ('oracle_goal', 'oracle_body', 'oracle_body_goal','all_oracle'):
            from uncertainty_ablation_states import intervene_control_result
            value = intervene_control_result(value, inference,
                context['request']['_diagnostic_truth'], mode)
        return value

    def send(value):
        if value is not None and isinstance(value, dict) and mode != 'baseline':
            value = dict(value)
            value['diagnostic_intervention'] = mode
            value['diagnostic_truth_assistance'] = True
            if context.get('oracle_packet') is not None:
                value['_diagnostic_estimated_packet'] = context['oracle_packet']
            for key in ('gradient_evaluation', 'diagnostics'):
                if isinstance(value.get(key), dict):
                    value[key] = dict(value[key], diagnostic_intervention=mode,
                                      physical_truth_received=True)
        return value

    def freeze(kind, value):
        if mode != 'baseline' and kind == 'control_artifact_job':
            result = value['result']
            result.records['diagnostic_intervention'] = dict(
                mode=mode, physical_truth_received=True,
                production_path=False, inference_truth_assisted=(mode in (
                    'oracle_marker', 'known_landmark','all_oracle')))
            for key in ('async_control', 'controller'):
                if isinstance(result.records.get(key), dict):
                    result.records[key] = dict(result.records[key],
                        physical_truth_received=True, diagnostic_intervention=mode)
        return original_freeze(kind, value)

    workers.predict_result = predicted
    workers.freeze_work_item = freeze
    try:
        workers.control_worker(QueueProxy(requests, on_get=receive),
            QueueProxy(replies, on_put=send), description, urdf, templates, output,
            horizon, control_mode, visibility_options, artifact_requests)
    finally:
        workers.predict_result = original_predict
        workers.freeze_work_item = original_freeze


def acquisition_worker(mode, requests, replies, specification, database, output):
    import probtf_observation_fusion_demo.async_workers as workers
    actual = workers.acquisition_worker
    original_board, original_planar = workers.BoardCamera, workers.PlanarCamera
    camera_context = {}

    def camera_factory(factory):
        def create(*args, **kwargs):
            value = factory(*args, **kwargs)
            camera_context['camera'] = value
            return value
        return create

    workers.BoardCamera = camera_factory(original_board)
    workers.PlanarCamera = camera_factory(original_planar)

    def checkpoint():
        # No per-image file I/O is added to the timed acquisition path.
        # Retain exact state in memory, flush only on normal/forced shutdown.
        if camera_context.get('last_completed') is not None:
            path = Path(output)/'diagnostic_camera_rng_checkpoint.json'
            temporary = path.with_suffix('.pending')
            temporary.write_text(json.dumps(camera_context['last_completed'])+'\n')
            temporary.replace(path)

    previous_term = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        checkpoint()
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, terminate)

    def receive(request):
        if request is not None and request.get('kind') == 'image':
            camera_context['current_request'] = request
            camera_context['before_image'] = camera_context['camera'].rng.bit_generator.state
        if (request is not None and request.get('kind') == 'image' and mode in
                ('oracle_marker', 'all_oracle')):
            path = Path(output)/'diagnostic_marker_truth'
            path.mkdir(exist_ok=True)
            value = {key:request[key] for key in ('event_id', 'stamp', 'command')}
            value.update(camera_pose=np.asarray(request['camera_pose']).tolist(),
                         marker_pose=np.asarray(request['marker_pose']).tolist())
            temporary = path/(request['event_id']+'.pending')
            temporary.write_text(json.dumps(value)+'\n')
            temporary.replace(path/(request['event_id']+'.json'))
        return request

    def send(reply):
        if isinstance(reply, dict) and reply.get('kind') == 'image_ready':
            request = camera_context['current_request']
            camera_context['last_completed'] = dict(event_id=request['event_id'],
                stamp=request['stamp'], command=list(request['command']),
                image_sha256=reply['image_sha256'],
                state_before_image=camera_context['before_image'],
                state_after_image=camera_context['camera'].rng.bit_generator.state,
                bit_generator='PCG64', physical_motion_changed=False,
                checkpoint_file_written_during_timed_motion=False)
        return reply

    try:
        return actual(QueueProxy(requests, on_get=receive), QueueProxy(replies,on_put=send),
                      specification, database, output)
    finally:
        checkpoint()
        workers.BoardCamera,workers.PlanarCamera = original_board,original_planar
        signal.signal(signal.SIGTERM,previous_term)


def inference_worker(mode, *args):
    _frozen_helpers(args[5])
    from probtf_observation_fusion_demo.async_workers import event_inference_worker
    restore = None
    if mode in ('oracle_marker', 'all_oracle'):
        from uncertainty_ablation_observations import install_oracle_marker_patch
        # args: requests,replies,description,urdf,templates,output,...
        restore = install_oracle_marker_patch(args[5],require_detected=True)
    elif mode == 'known_landmark':
        from uncertainty_ablation_landmark import install_known_landmark_patch
        from probtf_switch_press_demo.scene import transform, inverse
        import yaml
        config=yaml.safe_load((PACKAGE/'config/switch_press_scene.yaml').read_text())
        marker=config['marker']
        pose=inverse(transform(config['base_translation']))@transform(
            marker['world_translation'],marker['world_rpy'])
        restore=install_known_landmark_patch(pose)
    try:
        return event_inference_worker(*args)
    finally:
        if restore is not None:
            restore()


class DiagnosticContext:
    """Select wrapper process targets without changing the production module."""
    def __init__(self, mode, output=None, helper_sources=None):
        self.context, self.mode = mp.get_context('spawn'), mode
        self.output=output
        self.helper_sources=helper_sources
        self.copied=False

    def Queue(self):
        return self.context.Queue()

    def Process(self, *, target, args, **kwargs):
        if self.output is not None and not self.copied:
            path=Path(self.output)/'diagnostic_source'
            path.mkdir()
            for name,data in self.helper_sources.items():
                (path/name).write_bytes(data)
            self.copied=True
        replacement = {'control_worker':control_worker,
                       'acquisition_worker':acquisition_worker,
                       'event_inference_worker':inference_worker}.get(target.__name__)
        if replacement is not None:
            target, args = replacement, (self.mode,)+args
        return self.context.Process(target=target, args=args, **kwargs)


def control_truth(session, request):
    frame = session.frame
    if (float(request['stamp']) != float(frame.stamp) or
            int(request['index']) != len(session.frames)-1):
        raise ValueError('Diagnostic truth is not the exact control source frame.')
    return dict(stamp=float(frame.stamp), time=float(frame.time),
        command=np.asarray(frame.command).tolist(),
        physical_joint_position=np.asarray(frame.joints).tolist(),
        link_poses={key:np.asarray(value).tolist() for key,value in frame.link_poses.items()},
        marker_pose=np.asarray(session.scene.marker).tolist(),
        goal_position=np.asarray(session.scene.goal).tolist(),
        true_K=np.asarray(session.scene.config['plant']['stiffness']).tolist())


def install_coordinator_proxies(session, mode):
    def request(value):
        if value is not None and mode in ('oracle_goal', 'true_K', 'oracle_body',
                                          'oracle_body_goal','all_oracle'):
            truth = control_truth(session, value)
            value = dict(value, _diagnostic_truth=truth)
            _append(session.output/'diagnostic_control_truth.jsonl',
                    dict(request_id=value['request_id'], intervention=mode, **truth))
        return value

    def reply(value):
        if isinstance(value, dict) and value.get('_diagnostic_estimated_packet') is not None:
            packet = value['_diagnostic_estimated_packet']
            session.parameter_packets[packet.packet_id] = packet
        return value
    session.control_requests = QueueProxy(session.control_requests, on_put=request)
    session.control_replies = QueueProxy(session.control_replies, on_get=reply)


def trajectory_summary(session):
    positions = np.array([frame.tip for frame in session.frames])
    times = np.array([frame.time for frame in session.frames])
    goal = np.asarray(session.scene.goal)
    error = np.linalg.norm(positions-goal, axis=1)
    increments = np.diff(positions, axis=0)
    arc = np.linalg.norm(increments, axis=1)
    ray = (goal-positions[:-1])/np.maximum(error[:-1,None], np.finfo(float).tiny)
    radial = np.sum(ray*increments, axis=1)
    tangent = np.linalg.norm(increments-radial[:,None]*ray, axis=1)
    dt = np.diff(times)
    shortening = float(error[0]-error[-1])
    first_reach=next((index for index,frame in enumerate(session.frames)if frame.reached),None)
    return dict(initial_error_m=float(error[0]), final_error_m=float(error[-1]),
        minimum_error_m=float(error.min()), minimum_error_stamp_s=float(times[error.argmin()]),
        net_shortening_m=shortening, tip_arc_m=float(arc.sum()),
        net_shortening_per_arc=None if not arc.sum() else shortening/float(arc.sum()),
        tangent_arc_m=float(tangent.sum()),
        outward_time_fraction=float(dt[radial<0.].sum()/dt.sum()) if dt.sum() else None,
        reached=bool(any(frame.reached for frame in session.frames)),
        first_reach_time_s=None if first_reach is None else float(times[first_reach]),
        arc_to_first_reach_m=None if first_reach is None else float(arc[:first_reach].sum()),
        final_frame_reached=bool(session.frame.reached), frame_count=len(session.frames),
        physical_duration_s=float(session.frame.time),
        truth_used_for_diagnostic_scoring=True)


def run(case, mode, output, duration=180., drain=15.):
    from probtf_observation_fusion_demo.event_motion import AsyncMotionSession
    output = Path(output)
    root = PACKAGE.parents[2]
    source_files = list((PACKAGE/'src/probtf_observation_fusion_demo').glob('*.py'))
    source_files += list((root/'ros/core/probtf_core/src').rglob('*.py'))
    source_files += list((PACKAGE/'config').glob('*.yaml'))
    protected = list(root.rglob('*.rviz'))
    sources = {str(path):_sha(path) for path in source_files+protected if path.is_file()}
    helper_sources={path.name:path.read_bytes()for path in sorted(TOOLS.glob('uncertainty_ablation*.py'))}
    setup = dict(case=case, intervention=mode, duration_s=float(duration),
        drain_wall_s=float(drain), controller='production running_gradient/local_quadratic',
        physical_model='unchanged native dynamics and damping',
        physical_truth_received=(mode != 'baseline'),
        baseline_numeric_transformations=False,
        truth_K_availability='substitute value only after production first completed K packet',
        marker_oracle_preserves_detection=True,
        wall_clock_async=True, timing_identical_between_experiments=False,
        random_seed_policy='unchanged camera capture-order PCG64 seed; IMU source-time-keyed seed; inference has no Monte Carlo',
        source_sha256=sources,
        git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        diagnostic_sha256={str(TOOLS/name):hashlib.sha256(data).hexdigest()
                           for name,data in helper_sources.items()},
        diagnostic_source_copy='diagnostic_source; helper modules loaded from these frozen bytes')
    session = AsyncMotionSession(PACKAGE/('config/'+case+'.yaml'), output,
        duration=duration, context=DiagnosticContext(mode,output,helper_sources))
    install_coordinator_proxies(session, mode)
    (output/'diagnostic_setup.json').write_text(json.dumps(setup, indent=2)+'\n')
    failure = None
    try:
        # This is the production unobserved-prior startup barrier, not a
        # fabricated observed goal, convergence test, or numerical audit.
        while not session.start():
            session.poll()
            session.submit_estimate()
            session.submit_control()
            if session.failure is not None:
                raise RuntimeError(session.failure)
            time.sleep(.005)
        started = time.monotonic()
        last_report = -np.inf
        while not session.motion_finished:
            tick = time.monotonic()
            session.advance()
            elapsed = time.monotonic()-started
            if elapsed-last_report >= 10.:
                last_report = elapsed
                row = dict(elapsed_wall_s=elapsed, physical_stamp_s=session.frame.time,
                    physical_error_m=float(np.linalg.norm(session.frame.tip-session.scene.goal)),
                    gradient_evaluations=session.gradient_evaluations_completed,
                    acquired=session.acquisition_submitted, archived=session.acquisition_completed,
                    stiffness_done=len(session.stiffness_jobs_completed),
                    stiffness_submitted=len(session.stiffness_jobs_submitted))
                _append(output/'diagnostic_progress.jsonl', row)
                print(json.dumps(dict(case=case, intervention=mode, **row)), flush=True)
            if session.failure is not None:
                raise RuntimeError(session.failure)
            time.sleep(max(0., session.period-(time.monotonic()-tick)))
        physical_cutoff = session.manifest()
        summary = trajectory_summary(session)
        (output/'diagnostic_physical_cutoff.json').write_text(json.dumps(physical_cutoff, indent=2)+'\n')
        drain_started = time.monotonic()
        while time.monotonic()-drain_started < drain and not session.estimation_finished:
            session.poll()
            session.submit_estimate()
            time.sleep(.01)
        summary.update(case=case, intervention=mode,
            actual_motion_hz=physical_cutoff.get('actual_motion_hz'),
            actual_control_hz=physical_cutoff.get('actual_control_hz'),
            actual_camera_hz=physical_cutoff.get('actual_camera_hz'),
            source_changed=[name for name,digest in sources.items() if _sha(name)!=digest],
            body_drained=session.body_estimation_finished,
            pending_K_jobs=len(session.pending_stiffness_jobs),
            physical_cutoff_manifest='diagnostic_physical_cutoff.json')
        (output/'diagnostic_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        return summary
    except Exception as error:
        failure = traceback.format_exc()
        (output/'diagnostic_failure.txt').write_text(failure)
        summary=trajectory_summary(session)
        summary.update(case=case,intervention=mode,completed_requested_duration=False,
            requested_duration_s=float(duration),failure=failure,
            failure_type=type(error).__name__,failure_message=str(error),
            source_changed=[name for name,digest in sources.items() if _sha(name)!=digest])
        (output/'diagnostic_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        raise
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('marker', 'imu_marker'), required=True)
    parser.add_argument('--intervention', choices=MODES, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--duration', type=float, default=180.)
    parser.add_argument('--drain', type=float, default=15.)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.case, arguments.intervention, arguments.output,
                         arguments.duration, arguments.drain), indent=2), flush=True)


if __name__ == '__main__':
    main()
