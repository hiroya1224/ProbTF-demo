"""Combine independent task/view log factors only at command selection.

Neither factor is an observation. The existing complete task J is evaluated
unchanged; its finite-difference probes expose their already transported joint
laws to the independent visibility factor. No second command action, fitted
visual Jacobian, endpoint IK or modified candidate distribution is used.
"""
import hashlib
import json
from time import perf_counter

import numpy as np

from probtf.spherical_law.first_order import pose_query
from .moving_control import GradientEvaluation


def selection_gradient_at_command(controller, result, visibility_factor,
                                  source_command=None, *, command_map=None):
    started=perf_counter()
    active=visibility_factor.enabled and visibility_factor.weight>0.
    action=None if command_map is None else command_map.record()
    body=tuple(result.stages[-1].components)
    observation=result.records.get('async_prediction',{}).get('observation_source')
    body_stamp=(observation['stamp'] if observation.get('observation_ids') else None) if observation is not None else (
        result.prior.stamp if result.records.get('observation_ids') else None)
    source_ids=dict(source_stamp=result.prior.stamp,source_command=list(result.prior.command),
        body_state_ids=[item.state.state_id for item in body],
        parameter_density_id=None if action is None else action['density_id'],
        selected_K=None if action is None else action['selected_K'],
        body_observation_source_stamp=body_stamp,
        goal_observation_source_stamp=result.records.get('fixed_goal_memory',{}).get('observation_stamp'))
    probes={}
    view_seconds=[]
    path=tuple((edge,not inverse) for edge,inverse in reversed(result.camera_path))+tuple(result.marker_path)

    def observe(command,components,states):
        view_started=perf_counter()
        identity=tuple(float(value) for value in command)
        ids=dict(source_ids,goal_state_ids=[item.state.state_id for item in components],
            evaluation_state_ids=[state.state_id for state in states],
            evaluation_command=list(identity),camera_relative_marker_path=list(path))
        try:
            if not result.records.get('fixed_goal_memory',{}).get('available',False):
                raise ValueError('No acquired root marker candidate is available for the view factor.')
            queries=tuple(pose_query(state,path,query_id='visibility:camera_relative_marker',
                frame='camera_optical_frame') for state in states)
            record=visibility_factor.evaluate(components,queries,source_ids=ids)
        except Exception as error:
            record=dict(status='unavailable',reason=type(error).__name__+': '+str(error),
                enabled=visibility_factor.enabled,weight=visibility_factor.weight,
                penalty=None,weighted_penalty=None,log_factor=None,
                source_ids=ids,per_mode_record=[],physical_truth_received=False)
        probes[identity]=record
        view_seconds.append(perf_counter()-view_started)

    task=controller.gradient_at_command(result,source_command,command_map=command_map,
        **({'evaluation_observer':observe} if active else {})).record
    source_ids.update(body_state_ids=task.get('body_state_ids',[]),
        goal_state_ids=task.get('goal_state_ids',[]))
    record=dict(task)
    task_gradient=task.get('gradient')
    task_objective=task.get('objective_at_source')
    source=tuple(task['source_command']) if task.get('source_command') is not None else None
    if active:
        view=probes.get(source,dict(status='unavailable',reason='No valid task source probe.',
            enabled=True,weight=visibility_factor.weight,penalty=None,weighted_penalty=None,
            log_factor=None,source_ids=source_ids,per_mode_record=[]))
    else:
        view=visibility_factor.evaluate((),(),source_ids=source_ids)
    record.update(task_status=task['status'],task_reason=task.get('reason'),
        task_gradient=task_gradient,view_gradient=None,weighted_view_gradient=None,
        selected_gradient=None,visibility_factor=view,
        log_task_factor=None if task_objective is None else -task_objective,
        log_view_factor=view.get('log_factor'),log_select_factor=None,
        selected_objective_at_source=None,selection_kind='independent task times view factors',
        view_gradient_definition='unweighted dV/dc; None when inactive or unavailable',
        selected_gradient_definition='dJ/dc + visibility_factor.weight * dV/dc',
        selection_failure_policy='retain last accepted gradient; hold initially if none',
        task_candidate_law_changed=False,view_factor_is_observation=False,
        visibility_finite_difference_diagnostics=dict(evaluations=[
            dict(command=list(command),view_factor=value) for command,value in probes.items()],coordinates=[]))
    try:
        if task['status']!='ok':
            raise ValueError('Task gradient unavailable: '+str(task.get('reason')))
        if not active:
            # Preserve the original floating values and stencil exactly. Do
            # not evaluate V and claim that its uncomputed value was zero.
            selected=list(task_gradient)
            weighted=np.zeros(len(selected)).tolist()
            record.update(weighted_view_gradient=weighted,
                log_select_factor=record['log_task_factor'],
                selected_objective_at_source=task_objective)
        else:
            if view['status']!='ok':
                raise ValueError('Source view factor unavailable: '+str(view.get('reason')))
            value=float(view['penalty'])
            view_gradient=[]
            for coordinate in task['finite_difference_diagnostics']['coordinates']:
                index=coordinate['joint_index']
                available={}
                for side in ('forward','backward'):
                    probe=coordinate[side]
                    if probe['status']!='ok':
                        continue
                    candidate=list(source)
                    candidate[index]+=probe['step']
                    candidate_view=probes.get(tuple(candidate))
                    if candidate_view is None or candidate_view['status']!='ok':
                        raise ValueError('View difference unavailable at joint %d %s: %s' %
                            (index,side,None if candidate_view is None else candidate_view.get('reason')))
                    available[side]=(abs(probe['step']),float(candidate_view['penalty']))
                if len(available)==2:
                    forward,plus=available['forward']; backward,minus=available['backward']
                    derivative=(backward*(plus-value)/forward+
                        forward*(value-minus)/backward)/(forward+backward)
                elif 'forward' in available:
                    step,plus=available['forward']; derivative=(plus-value)/step
                elif 'backward' in available:
                    step,minus=available['backward']; derivative=(value-minus)/step
                else:
                    raise ValueError('No common finite task/view stencil at joint '+str(index))
                if not np.isfinite(derivative):
                    raise ValueError('Nonfinite view derivative at joint '+str(index))
                view_gradient.append(float(derivative))
                record['visibility_finite_difference_diagnostics']['coordinates'].append(dict(
                    joint_index=index,method=coordinate['method'],derivative=float(derivative)))
            weighted=(visibility_factor.weight*np.asarray(view_gradient)).tolist()
            selected=(np.asarray(task_gradient)+np.asarray(weighted)).tolist()
            record.update(view_gradient=view_gradient,weighted_view_gradient=weighted,
                log_select_factor=record['log_task_factor']+view['log_factor'],
                selected_objective_at_source=task_objective+view['weighted_penalty'])
        if not np.all(np.isfinite(selected)):
            raise ValueError('Nonfinite selected task/view gradient.')
        record.update(gradient=selected,selected_gradient=selected,status='ok',reason=None)
    except Exception as error:
        record.update(gradient=None,selected_gradient=None,status='failed',
            reason=type(error).__name__+': '+str(error))
    record['selection_factors']=dict(
        task_gradient=record['task_gradient'],view_gradient=record['view_gradient'],
        weighted_view_gradient=record['weighted_view_gradient'],
        selected_gradient=record['selected_gradient'],
        log_task_factor=record['log_task_factor'],log_view_factor=record['log_view_factor'],
        log_select_factor=record['log_select_factor'],visibility_factor=record['visibility_factor'],
        task_gradient_definition='dJ/dc',view_gradient_definition=record['view_gradient_definition'],
        selected_gradient_definition=record['selected_gradient_definition'])
    record['task_computation_seconds_including_view_probes']=record['computation_seconds']
    record['view_probe_computation_seconds']=sum(view_seconds)
    record['view_probe_count']=len(view_seconds)
    record['task_and_FD_overhead_seconds_excluding_view_probes']=max(0.,
        record['computation_seconds']-sum(view_seconds))
    record['computation_seconds']=perf_counter()-started
    encoded=json.dumps(record,sort_keys=True,separators=(',',':'))
    return GradientEvaluation(encoded,hashlib.sha256(encoded.encode()).hexdigest())
