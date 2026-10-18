"""Finite selection from an immutable latest body law and captured K.

This consumer performs no observation or K update. The inference owner can
continue its recurrence while this worker transports a frozen acquisition
through the actual command ledger and selects one short position segment.
"""

from dataclasses import replace
from io import BytesIO
from pathlib import Path
import time
import traceback

import rospkg
import yaml

from .async_state import predict_result
from .async_stiffness import PhysicalActionEngine, reanchor_retained_stiffness


def selection_worker(requests, replies, description, urdf, templates, output):
    """Reply with the finite plan first, then its identical-law display wire.

    Each request supplies ``result``, ``packet``, ``steps``, ``index``,
    ``stamp``, ``request_id``, ``capture_index`` and ``goal_observed``. The
    result is the original acquired law; steps are exactly the completed
    intervals from capture_index to index. No observation is fused again and
    no absolute mechanical pose replaces the supplied body law.
    """
    import cv2
    from .simple_inference import SimpleInference
    from .simple_control import SimpleSelector
    from .display import make_markers
    from .node import build_snapshot

    cv2.setNumThreads(1)
    try:
        inference = SimpleInference(description, templates)
        package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
        settings = yaml.safe_load((package/'config/stiffness_identification.yaml').read_text())
        settings.update(inference.settings['simple_stiffness'])
        settings['bayes_factor_surprise'] = dict(enabled=False)
        engine = PhysicalActionEngine(urdf, tuple(inference.body_ids), settings)
        selector = SimpleSelector(inference, description, inference.settings.get('simple_reference', {}))
        while True:
            request = requests.get()
            if request is None:
                return
            started = time.monotonic()
            source = request['result']
            steps = tuple(request['steps'])
            capture_index, index = request['capture_index'], request['index']
            if tuple(step.index for step in steps) != tuple(range(capture_index, index)):
                raise ValueError('Finite selection needs every actual interval from its source acquisition to the target frame.')
            sensor_event = source.records.get('sensor_event') or {}
            if sensor_event and (
                    sensor_event['acquisition_stamp'] != source.prior.stamp
                    or tuple(sensor_event['sent_command']) != tuple(source.prior.command)
                    or sensor_event['event_index'] != capture_index):
                raise ValueError('The frozen selection law lost its original acquisition time, command or frame index.')
            present = predict_result(source, steps, request['stamp'], index,
                                     inference.body_ids, inference.actions)
            present.records['simple_inference'] = dict(source.records.get('simple_inference', {}))
            packet = request['packet']
            command_map = None
            if packet is not None:
                present, command_map = reanchor_retained_stiffness(engine, present, packet)
            select = bool(request.get('select', True))
            if select:
                q_next, diagnostics = selector.select(present, command_map=command_map,
                                                      goal_observed=request['goal_observed'])
            else:
                state = present.stages[-1].components[0].state
                q_next = None
                diagnostics = dict(status='display_only_after_motion',
                    selection_performed=False, source_stamp=state.stamp,
                    source_command=list(state.command), source_version=state.version,
                    source_prior_state_id=present.prior.state_id,
                    goal_observed=bool(request['goal_observed']))
            observation_stamp = source.records.get('last_effective_observation_stamp')
            if observation_stamp is None:
                observation_stamp = source.prior.stamp
            observation_command = source.records.get('last_effective_observation_command')
            if observation_command is None and observation_stamp == source.prior.stamp:
                observation_command = source.prior.command
            diagnostics.update(
                body_source_state_ids=[item.state.state_id for item in present.stages[-1].components],
                goal_source_state_ids=[item.state.state_id for item in present.stages[-1].components],
                selected_at_source_stamp=present.prior.stamp,
                original_observation_stamp=observation_stamp,
                original_observation_command=(None if observation_command is None
                                              else list(observation_command)),
                body_acquisition_stamp=source.prior.stamp,
                body_acquisition_command=list(source.prior.command),
                acquisition_event_id=sensor_event.get('event_id'))
            base = dict(request_id=request['request_id'],
                event_id=sensor_event.get('event_id'), capture_index=capture_index,
                index=index, diagnostics=diagnostics)
            replies.put(dict(base, kind='plan', q_next=q_next, packet=packet,
                selection_performed=select, source_command=list(present.prior.command),
                duration=selector.duration, work_seconds=time.monotonic()-started))
            snapshot = replace(build_snapshot(present, present.stages[-1], inference.common),
                summary='', show_labels=False,
                stiffness_record=present.records.get('stiffness'))
            display = inference.common['display']
            markers = make_markers(snapshot, display['seed'], display['point_size'])
            wire = BytesIO()
            markers.serialize(wire)
            replies.put(dict(base, kind='display', snapshot=snapshot,
                             markers_wire=wire.getvalue()))
    except Exception:
        replies.put(dict(failed=True, worker='simple_selection', traceback=traceback.format_exc()))
