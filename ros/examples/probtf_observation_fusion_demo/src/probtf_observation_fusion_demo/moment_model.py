"""One core moment propagation and analytic joint conditional per observation."""
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
from .moment_conditioning import condition_moments
from .moment_observation import prepare_observation
from .sensors import LocalEstimate, camera_xy_queries


@dataclass(frozen=True)
class MomentPosteriorSensorView:
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
    role: str='joint propagated-moment posterior core push-forward; not for lift_sensor_posterior'


def _update_options(options):
    options=dict(options or {})
    if set(options)-{'method'}:
        raise ValueError('Moment propagation has no optimizer or iterative update options.')
    if options.get('method','propagated_moments')!='propagated_moments':
        raise ValueError('This updater implements propagated_moments only.')
    return options


def run(common,observations,image=None,*,prior=None,paths=None,prior_components=None,
        joint_evidence=False,shared_observation_frame=False,update_options=None):
    """Propagate each original prior once, then condition once analytically.

The chronological caller supplies one sensor event. Synchronous diagnostics
stack all moments from that SAME prior with their full cross covariance;
only the independent raw measurement noise is block diagonal. Every native
combination remains explicit. No point search, MAP, GN, or Laplace body
update is performed. The empty-observation helper is the exact identity.
"""
    options=_update_options(update_options)
    base=identity_run(common,(),image,prior=prior,paths=paths,prior_components=prior_components)
    prior=base.prior;parents=base.stages[0].components
    paths=geometry_paths(common) if paths is None else paths
    prepared=[];seen={}
    for observation in observations:
        identifier=observation['observation_id']
        if identifier in seen:
            if observation!=seen[identifier]:
                raise ValueError('A repeated moment observation ID has different input data.')
            continue
        seen[identifier]=observation
    # A synchronous raw-likelihood product is commutative.  Canonicalize
    # only this one update's sensor factors, not chronological event replay,
    # prior-parent order, or each detector's original native branch order.
    for identifier in sorted(seen):
        prepared.append(prepare_observation(common,seen[identifier],paths,base.image,prior.stamp))
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
        base.records.update(high_concentration_observation_update=freeze_json_record(dict(method='propagated_moments',
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
        propagated_groups=[item.propagate(parent.state)for item in available]
        for propagated in product(*propagated_groups):
            selected=tuple(item.likelihood for item in propagated)
            covariance=block_diag(*(item.covariance for item in selected))
            observed=np.concatenate([item.observed_mean for item in selected])
            queries=tuple(item.query for item in propagated)
            predicted=np.concatenate([item.mean for item in queries])
            H=np.vstack([item.jacobian for item in queries])
            C=np.hstack([item.cross_covariance for item in queries])
            # Off-diagonal sensor blocks use the SAME complete joint P;
            # diagonal blocks retain the actual propagated core S exactly.
            S=H@raw_prior.covariance@H.T
            start=0
            for query in queries:
                stop=start+query.mean.size
                S[start:stop,start:stop]=query.covariance
                start=stop
            conditioned=condition_moments(raw_prior,predicted,S,C,H,observed,covariance)
            native_log_mass=math.fsum(item.log_weight for item in selected)
            likelihood_parts=add_log_evidence_parts((0.,0.),conditioned.log_evidence_parts)
            conditional_parts=add_log_evidence_parts(likelihood_parts,(native_log_mass,))
            label=parent.label+'|'+ '|'.join(item.observation_id+':'+item.label for item in selected)
            entry=dict(parent=parent,label=label,selected=selected,propagated=propagated,conditioned=conditioned,
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
                posterior_role='actual joint propagated-moment posterior core push-forward; not re-lifted')
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
            view=MomentPosteriorSensorView(original,observation['sensor_id'],observation['observation_id'],local,queries)
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
        approximation='unchanged high-concentration core O(P) mean/full covariance/cross propagation; one analytic Gaussian conditional; existing Bingham reexpression',
        stages=[dict(label=stage.label,sensor_ids=list(stage.sensor_ids),
            components=[_joint_record(item,paths,common['root_frame'])for item in stage.components])for stage in stages],
        sensor_estimates=[item.diagnostics for item in estimates],
        frozen_local_posteriors=[],lift_total_covariance=[],
        local_gaussian_lift_applied=False,
        mixture_policy='all Keep/Reset parents and every native observation combination; no pruning',
        frozen_body_charts=[dict(prior_label=branch.label,prior_weight=branch.weight,
            prior_log_weight=branch.log_weight,reference_state_id=branch.reference_state.state_id,
            posterior_labels=[item.label for item in branch.posterior_components])for branch in frozen],
        high_concentration_observation_update=dict(method='propagated_moments',
            original_prior_held_fixed=True,observation_added_once=True,update_options=options,
            point_optimization_used=False,optimization_iterations=0,gauss_newton_used=False,
            laplace_body_evidence_used=False,core_O_P_mean_used=True,
            native_likelihoods=[component.record()for item in available for component in item.components],
            branch_diagnostics=[dict(parent_label=entry['parent'].label,label=entry['label'],
                source_prior_state_id=entry['parent'].state.state_id,
                native_component_labels=[item.label for item in entry['selected']],
                native_log_weights=[item.log_weight for item in entry['selected']],
                posterior_log_weight=component_log_weight(entry['raw']),
                conditional_log_evidence_parts=list(entry['conditional_parts']),
                propagated_observations=[item.record()for item in entry['propagated']],
                **entry['conditioned'].diagnostics)for entry in candidates],
            body_single_mode_approximation='one analytic Gaussian conditional per explicit prior/native branch, followed by unchanged canonical projection',
            strict_nonlinear_moment_matching_claimed=False,monte_carlo_used=False))
    if joint_evidence:
        records['joint_observation_evidence']=dict(observation_ids=list(observation_ids),
            parents=parent_evidence,log_evidence=math.fsum(total_parts),log_evidence_parts=list(total_parts),
            measure='same fixed native image-Laplace charts / observed-fixed S2 tangent Gaussian observation coordinates',
            approximation='normalized Gaussian overlap of propagated core moments; exact within the frozen Gaussian proxy, not exact nonlinear/raw-pixel integration',
            shared_body_reference_product_correction=False,original_joint_prior_used_once=True)
    # Propagation diagnostics are immutable once the update returns.  Freeze
    # once so deepcopy/IPC can share them without retraversing their arrays;
    # the surrounding records remain mutable for acquisition metadata.
    records['high_concentration_observation_update']=freeze_json_record(records['high_concentration_observation_update'])
    return replace(base,stages=stages,detection_image=detection,sensor_posteriors=tuple(views),
        local_estimates=tuple(estimates),records=records,
        frozen_body_posterior=fused,frozen_body_branches=frozen)
