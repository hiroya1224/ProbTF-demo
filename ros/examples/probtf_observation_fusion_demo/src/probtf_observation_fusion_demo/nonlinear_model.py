"""Direct joint-prior nonlinear conditioning; no local posterior re-lifting."""
from dataclasses import dataclass, replace
from itertools import product
import math

import numpy as np
from scipy.linalg import block_diag

from probtf.spherical_law.fusion import Gaussian, PosteriorComponent, FusedMixture
from .body_native import pose_query, direction_query
from .immutable_body_cache import freeze_json_record, reexpress_state
from .joint_batch_fusion import add_log_evidence_parts, _normalize_log_evidence_parts
from .model import (run as identity_run, FrozenBodyBranch, WeightedState, Stage,
                    component_log_weight, geometry_paths, _joint_record)
from .nonlinear_conditioning import condition_nonlinear
from .nonlinear_observation import prepare_likelihood
from .sensors import LocalEstimate, camera_xy_queries


@dataclass(frozen=True)
class JointPosteriorSensorView:
    """Core push-forwards of joint posteriors, never an input to linear lift.

``query`` identifies the original sensor/prior source. Each component_query
is the actual posterior core query, including its own modal orientation chart.
It is not a FrozenGaussianQuery/common linear conditional reconstruction.
"""
    query: object
    sensor_id: str
    observation_id: str
    components: tuple
    component_queries: tuple
    role: str='joint nonlinear posterior core push-forward; not for lift_sensor_posterior'


def _solver_options(options):
    options=dict(options or {})
    unknown=set(options)-{'method','max_iterations','score_tolerance','backtrack_steps','armijo'}
    if unknown:
        raise ValueError('Unknown nonlinear solver options: '+repr(sorted(unknown)))
    if options.get('method','nonlinear_fixed_prior')!='nonlinear_fixed_prior':
        raise ValueError('This direct joint updater implements nonlinear_fixed_prior only.')
    return dict(max_iterations=options.get('max_iterations',40),
        gradient_tolerance=options.get('score_tolerance',1e-6),
        max_backtracks=options.get('backtrack_steps',16),armijo=options.get('armijo',1e-4))


def run(common,observations,image=None,*,prior=None,paths=None,prior_components=None,
        joint_evidence=False,shared_observation_frame=False,solver_options=None):
    """Apply all supplied raw likelihoods once to each ORIGINAL joint prior.

The normal chronological caller supplies one sensor event. Synchronous
diagnostics may supply several observations; their raw likelihoods are
multiplied once inside one joint objective, keeping every native combination.
The empty-observation helper below is the exact identity, not an affine
fallback after nonlinear conditioning failure.
"""
    options=_solver_options(solver_options)
    base=identity_run(common,(),image,prior=prior,paths=paths,prior_components=prior_components)
    prior=base.prior;parents=base.stages[0].components
    paths=geometry_paths(common) if paths is None else paths
    prepared=[];seen={}
    for observation in observations:
        identifier=observation['observation_id']
        if identifier in seen:
            if observation!=seen[identifier]:
                raise ValueError('A repeated nonlinear observation ID has different input data.')
            continue
        seen[identifier]=observation
    # A synchronous raw-likelihood product is commutative.  Canonicalize
    # only this one update's sensor factors, not chronological event replay,
    # prior-parent order, or each detector's original native branch order.
    for identifier in sorted(seen):
        prepared.append(prepare_likelihood(common,seen[identifier],paths,base.image,prior.stamp))
    available=tuple(item for item in prepared if item.components)
    detection=base.detection_image
    for item in prepared:
        if item.prepared_marker is not None:
            detection=item.prepared_marker.debug_image
    if not available:
        missing=tuple(LocalEstimate(None,dict(item.diagnostics),
            None if item.prepared_marker is None else item.prepared_marker.debug_image,
            board_fusion=None if item.prepared_marker is None else item.prepared_marker.board_fusion)
            for item in prepared for _parent in parents)
        base.records.update(nonlinear_observation_update=freeze_json_record(dict(method='nonlinear_fixed_prior',
            effective_observation_count=0,empty_observation_identity=True,
            original_prior_held_fixed=True,monte_carlo_used=False)),
            sensor_estimates=[estimate.diagnostics for estimate in missing])
        return replace(base,local_estimates=missing,detection_image=detection)

    observation_ids=tuple(item.observation['observation_id']for item in available)
    sensor_ids=tuple(dict.fromkeys(item.observation['sensor_id']for item in available))
    candidates=[];parent_evidence=[]
    for parent in parents:
        entries=[]
        raw_prior=Gaussian(parent.state.mean,parent.state.covariance)
        for selected in product(*(item.components for item in available)):
            covariance=block_diag(*(item.covariance for item in selected))
            def evaluate(value,selected=selected,state=parent.state):
                values=[item.evaluate(state,value)for item in selected]
                return np.concatenate([item[0]for item in values]),np.vstack([item[1]for item in values])
            conditioned=condition_nonlinear(raw_prior,evaluate,covariance,**options)
            native_log_mass=math.fsum(item.log_weight for item in selected)
            likelihood_parts=add_log_evidence_parts((0.,0.),conditioned.log_evidence_parts)
            conditional_parts=add_log_evidence_parts(likelihood_parts,(native_log_mass,))
            label=parent.label+'|'+ '|'.join(item.observation_id+':'+item.label for item in selected)
            entry=dict(parent=parent,label=label,selected=selected,conditioned=conditioned,
                conditional_parts=conditional_parts,
                global_parts=add_log_evidence_parts(conditional_parts,(component_log_weight(parent),)))
            entries.append(entry);candidates.append(entry)
        _,parts=_normalize_log_evidence_parts([entry['conditional_parts']for entry in entries])
        parent_evidence.append(dict(label=parent.label,log_evidence=math.fsum(parts),
            log_evidence_parts=list(parts),source_state_id=parent.state.state_id,
            prior_log_weight=component_log_weight(parent),combination_count=len(entries)))
    normalized,total_parts=_normalize_log_evidence_parts([entry['global_parts']for entry in candidates])
    components=[];raw_components=[];children={parent.label:[] for parent in parents}
    for entry,log_weight in zip(candidates,normalized):
        log_weight=float(log_weight);weight=float(math.exp(log_weight))
        gaussian=entry['conditioned'].distribution
        raw=PosteriorComponent(entry['label'],gaussian,weight,
            lineage=observation_ids,log_weight=log_weight)
        state=reexpress_state(entry['parent'].state,gaussian.mean,gaussian.covariance,
            version=prior.version+2,component_id=entry['label'])
        item=WeightedState(entry['label'],weight,state,log_weight)
        components.append(item);raw_components.append(raw)
        children[entry['parent'].label].append(raw)
        entry.update(raw=raw,weighted=item)
    components=tuple(components)
    stages=(base.stages[0],Stage('fused',components,sensor_ids))
    fused=FusedMixture(tuple(raw_components),(),observation_ids,prior.version,prior.stamp,prior.command)
    frozen=tuple(FrozenBodyBranch(parent.label,parent.weight,parent.state,
        Gaussian(parent.state.mean,parent.state.covariance),tuple(children[parent.label]),
        component_log_weight(parent))for parent in parents)

    # These views expose the actual core push-forwards of joint results.
    # They are not local Bayesian posteriors and never enter the old lift.
    estimates=[];views=[]
    for prepared_item in prepared:
        observation=prepared_item.observation
        for parent in parents:
            diagnostics=dict(prepared_item.diagnostics,
                prior_component_id=parent.label,source_prior_state_id=parent.state.state_id,
                posterior_role='actual joint nonlinear posterior core push-forward; not re-lifted')
            if not prepared_item.components:
                marker=prepared_item.prepared_marker
                estimates.append(LocalEstimate(None,diagnostics,marker.debug_image,
                    board_fusion=marker.board_fusion))
                continue
            likelihood=prepared_item.components[0]
            def query(state):
                if likelihood.sensor=='marker':
                    return pose_query(state,likelihood.path,query_id=likelihood.sensor_id+':joint_posterior_view',frame='camera')
                return direction_query(state,likelihood.path,likelihood.root_direction,
                    query_id=likelihood.sensor_id+':joint_posterior_view',frame=observation['target_link'])
            original=query(parent.state)
            own=[entry for entry in candidates if entry['parent'].label==parent.label]
            queries=tuple(query(entry['weighted'].state)for entry in own)
            local=tuple(PosteriorComponent(entry['label'],Gaussian(item.mean,item.covariance),
                entry['raw'].weight,log_weight=component_log_weight(entry['raw']))
                for entry,item in zip(own,queries))
            view=JointPosteriorSensorView(original,observation['sensor_id'],observation['observation_id'],local,queries)
            views.append(view)
            marker=prepared_item.prepared_marker
            evidence=next(item['log_evidence']for item in parent_evidence if item['label']==parent.label)
            diagnostics.update(log_evidence=evidence,
                log_evidence_scope='joint raw observation product for this fixed prior parent',
                joint_posterior_source_state_ids=[item.state_id for item in queries])
            image_queries=camera_xy_queries(view) if likelihood.sensor=='marker' else ()
            estimates.append(LocalEstimate(view,diagnostics,
                None if marker is None else marker.debug_image,
                None if marker is None else marker.detected,
                display_query=original,observed_direction=likelihood.observed_direction,
                image_queries=image_queries,log_evidence=evidence,
                board_fusion=None if marker is None else marker.board_fusion))
    records=dict(base.records)
    records.update(valid_observation_count=len(available),observation_ids=list(observation_ids),
        approximation='nonlinear finite modal likelihood; fixed-prior finite-iterate GN-Laplace joint Gaussian proxy; existing Bingham reexpression',
        stages=[dict(label=stage.label,sensor_ids=list(stage.sensor_ids),
            components=[_joint_record(item,paths,common['root_frame'])for item in stage.components])for stage in stages],
        sensor_estimates=[item.diagnostics for item in estimates],
        frozen_local_posteriors=[],lift_total_covariance=[],
        local_gaussian_lift_applied=False,
        mixture_policy='all Keep/Reset parents and every native observation combination; no pruning',
        frozen_body_charts=[dict(prior_label=branch.label,prior_weight=branch.weight,
            prior_log_weight=branch.log_weight,reference_state_id=branch.reference_state.state_id,
            posterior_labels=[item.label for item in branch.posterior_components])for branch in frozen],
        nonlinear_observation_update=dict(method='nonlinear_fixed_prior',
            original_prior_held_fixed=True,observation_added_once=True,solver_options=dict(solver_options or {}),
            native_likelihoods=[component.record()for item in available for component in item.components],
            branch_diagnostics=[dict(parent_label=entry['parent'].label,label=entry['label'],
                source_prior_state_id=entry['parent'].state.state_id,
                native_component_labels=[item.label for item in entry['selected']],
                native_log_weights=[item.log_weight for item in entry['selected']],
                posterior_log_weight=component_log_weight(entry['raw']),
                conditional_log_evidence_parts=list(entry['conditional_parts']),
                **entry['conditioned'].diagnostics)for entry in candidates],
            body_single_mode_approximation='one finite GN-Laplace Gaussian per explicit prior/native branch; not global posterior mode enumeration',
            strict_nonlinear_moment_matching_claimed=False,monte_carlo_used=False))
    if joint_evidence:
        records['joint_observation_evidence']=dict(observation_ids=list(observation_ids),
            parents=parent_evidence,log_evidence=math.fsum(total_parts),log_evidence_parts=list(total_parts),
            measure='same native fixed image-Laplace charts / observed-fixed S2 Log tangent Gaussian scalar likelihoods',
            approximation='finite-iterate GN-Laplace evidence, not exact nonlinear integration',
            shared_body_reference_product_correction=False,original_joint_prior_used_once=True)
    # Iteration diagnostics are immutable once the update returns.  Freeze
    # once so deepcopy/IPC can share them without retraversing their history;
    # the surrounding records remain mutable for acquisition metadata.
    records['nonlinear_observation_update']=freeze_json_record(records['nonlinear_observation_update'])
    return replace(base,stages=stages,detection_image=detection,sensor_posteriors=tuple(views),
        local_estimates=tuple(estimates),records=records,
        frozen_body_posterior=fused,frozen_body_branches=frozen)
