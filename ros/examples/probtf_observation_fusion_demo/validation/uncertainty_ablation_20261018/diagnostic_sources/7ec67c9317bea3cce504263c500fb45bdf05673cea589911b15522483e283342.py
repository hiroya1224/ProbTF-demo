"""Explicit truth interventions for the authorized diagnostic harness only.

Never imported by production. All truth is supplied by the caller at the exact
control capture time. Baseline is object-identical. Narrowing is a full
congruence, including cross covariance; no inference covariance is clipped.
"""
from dataclasses import replace
import hashlib
import json

import numpy as np

from probtf.geometry import rotmat_to_quat
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
)
from probtf_observation_fusion_demo.model import component_log_weight
from probtf_observation_fusion_demo.moving_control import control_components
from probtf_observation_fusion_demo.async_stiffness import FrozenKPacket


def _replace_state(state, replacements, *, scale_indices=(), scale=1.,
                   exact_environment=None):
    """Retain the declared P in replacement charts, optionally scale it."""
    covariance=np.array(state.covariance,copy=True)
    factors=np.ones(covariance.shape[0])
    for index in scale_indices:
        factors[6*index:6*index+6]=float(scale)
    covariance=factors[:,None]*covariance*factors[None,:]
    if exact_environment is not None:
        index,translation_sd,rotation_sd=exact_environment
        part=slice(6*index,6*index+6)
        covariance[part,:]=0.;covariance[:,part]=0.
        covariance[part,part]=np.diag([translation_sd**2]*3+[rotation_sd**2]*3)
    transforms=[]
    for index,edge in enumerate(state.transforms):
        pose=replacements.get(edge.edge_id)
        part=slice(6*index,6*index+6)
        if pose is None and np.array_equal(covariance[part,part],state.covariance[part,part]):
            transforms.append(edge)
            continue
        translation=edge.translation if pose is None else np.asarray(pose)[:3,3]
        quaternion=edge.rotation_wxyz if pose is None else rotmat_to_quat(np.asarray(pose)[:3,:3])
        block=covariance[6*index+3:6*index+6,6*index+3:6*index+6]
        parameter=(None if edge.bingham is None else
                   bingham_from_rotation_coordinates(quaternion,block))
        transforms.append(BinghamTransform(edge.edge_id,translation,parameter,
            covariance[6*index:6*index+3,6*index:6*index+3],
            quaternion if parameter is None else None))
    return JointTransformState(tuple(transforms),covariance,version=state.version,
        stamp=state.stamp,command=state.command,component_id=state.component_id)


def _validate_truth(result,truth):
    if (float(truth['stamp'])!=result.prior.stamp or
            not np.array_equal(truth['command'],result.prior.command)):
        raise ValueError('Diagnostic truth must have the exact control-capture stamp and command.')


def intervene_control_result(predicted,inference,truth,intervention):
    """Replace only the selected control-side candidate; recursive state stays.

    oracle_goal supplies true root-marker pose with 0.1mm/0.001rad finite
    width and independent body/goal coupling. oracle_body supplies the six
    true local centers and scales their entire existing joint SD by 0.01.
    The latter preserves all correlation coefficients by congruence. Goal
    centers/marginals and the internal unknown-marker law are not corrected.
    """
    if intervention in ('baseline','oracle_marker','true_K'):
        return predicted
    if intervention not in ('oracle_goal','oracle_body','oracle_body_goal','all_oracle'):
        raise ValueError('Unknown diagnostic intervention: '+str(intervention))
    _validate_truth(predicted,truth)
    body,goals=control_components(predicted)
    original_body_ids=[item.state.state_id for item in body]
    original_goal_ids=[item.state.state_id for item in goals]
    marker_id=inference.marker.edge_id
    replacement={}
    if intervention in ('oracle_body','oracle_body_goal','all_oracle'):
        poses=truth['link_poses']
        parent=inference.settings['root_frame']
        for edge_id in inference.body_ids:
            replacement[edge_id]=np.linalg.solve(np.asarray(poses[parent]),np.asarray(poses[edge_id]))
            parent=edge_id
        def body_state(state):
            indices=[state.edge_ids.index(name)for name in inference.body_ids]
            return _replace_state(state,replacement,scale_indices=indices,scale=.01)
        body=tuple(replace(item,state=body_state(item.state))for item in body)
        goals=tuple(replace(item,state=body_state(item.state))for item in goals)
    if intervention in ('oracle_goal','oracle_body_goal','all_oracle'):
        # Even before any image has been fused, this intervention supplies
        # the declared oracle target, rather than waiting for a detector.
        source_goals=goals or body
        updated=[]
        for item in source_goals:
            index=item.state.edge_ids.index(marker_id)
            state=_replace_state(item.state,{marker_id:np.asarray(truth['marker_pose'])},
                exact_environment=(index,1e-4,1e-3))
            updated.append(replace(item,state=state))
        goals=tuple(updated)
    stage=replace(predicted.stages[-1],components=body,goal_components=goals)
    records=dict(predicted.records)
    measure=records.get('candidate_measure')
    if measure is not None:
        records['candidate_measure']=dict(measure,
            shape_source_state_ids=[item.state.state_id for item in body],
            shape_source_versions=[item.state.version for item in body],
            shape_source_stamps=[item.state.stamp for item in body],
            shape_source_commands=[list(item.state.command)for item in body],
            shape_log_weights=[component_log_weight(item)for item in body],
            diagnostic_truth_intervention=intervention)
    if intervention in ('oracle_goal','oracle_body_goal','all_oracle'):
        records['fixed_goal_memory']=dict(records.get('fixed_goal_memory',{}),
            available=True,observation_stamp=truth['stamp'],
            observation_command=list(truth['command']),observation_version=predicted.prior.version,
            observation_ids=['diagnostic_oracle_root_marker'],
            source='authorized diagnostic physical root-marker pose; not an observation')
    records['diagnostic_truth_intervention']=dict(name=intervention,
        physical_truth_received=True,stamp=truth['stamp'],command=list(truth['command']),
        original_body_state_ids=original_body_ids,original_goal_state_ids=original_goal_ids,
        body_state_ids=[item.state.state_id for item in body],
        goal_state_ids=[item.state.state_id for item in goals],
        body_sd_scale=.01 if intervention in ('oracle_body','oracle_body_goal','all_oracle')else 1.,
        oracle_goal_translation_sd_m=1e-4,oracle_goal_rotation_sd_rad=1e-3,
        recursive_estimation_modified=False,objective_changed=False,monte_carlo_used=False)
    result=replace(predicted,stages=predicted.stages[:-1]+(stage,),records=records)
    control_components(result)
    return result


def oracle_K_packet(engine,result,existing_packet,truth):
    """A clearly labelled known-K diagnostic packet, never a learned density.

    Preserve the original estimator record in its own worker/archive. This
    replacement packet retains the same source timing and model, and solves
    an auxiliary equilibrium using the oracle K; it never uses physical
    dynamic joints as an auxiliary equilibrium. If existing_packet is None,
    the caller may preserve initial K-unavailability for value-only isolation.
    """
    if existing_packet is None:
        return None
    record=existing_packet.record
    proposal=dict(record['command_proposal'])
    stiffness=np.asarray(truth['true_K'],dtype=float)
    command=np.asarray(proposal['source_command'])
    theta,_,_,active=engine.identifier._refine_equilibrium(
        np.asarray(proposal['auxiliary_equilibrium']),command,stiffness)
    old_density=record.get('posterior')or record.get('retained_prior')
    density_id=hashlib.sha256(('diagnostic_known_K:'+json.dumps(stiffness.tolist())).encode()).hexdigest()
    density=dict(density_id=density_id,representation='diagnostic_known_K_Dirac',
        selected_logK=np.log(stiffness).tolist(),global_moments=None,
        original_estimated_density_id=old_density['density_id'],
        physical_truth_received=True,estimated_density_claimed=False)
    proposal.update(density_id=density_id,selected_K=stiffness.tolist(),
        selected_logK=np.log(stiffness).tolist(),auxiliary_equilibrium=theta.tolist(),
        auxiliary_active_joint_limits=np.asarray(active,dtype=bool).tolist(),
        method='authorized diagnostic true stiffness; not inferred',physical_truth_received=True)
    record.update(command_proposal=proposal,posterior=density,retained_prior=density,
        status='diagnostic_true_K',update_applied=False,
        diagnostic_truth_intervention=dict(name='true_K',
            original_packet_id=existing_packet.packet_id,original_density_id=old_density['density_id'],
            original_selected_K=existing_packet.proposal['selected_K'],
            physical_truth_received=True,estimated_density_claimed=False))
    encoded=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)
    return FrozenKPacket(encoded,hashlib.sha256(encoded.encode()).hexdigest())
