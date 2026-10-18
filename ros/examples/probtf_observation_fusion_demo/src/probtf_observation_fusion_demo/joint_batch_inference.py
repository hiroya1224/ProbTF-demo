"""All acquired likelihoods in one frozen chart, one terminal body projection.

This is the explicitly authorized joint-batch approximation, not the old
per-event assumed-density recursion. The change clock is aggregated over
effective observations; a redraw is placed at the batch start. Every native
likelihood branch and its normalizing constant is retained.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib

import numpy as np

from probtf.spherical_law.fusion import Gaussian, FusedMixture, PosteriorComponent

from .async_state import _action_signature, _recorded_action
from .batch_observation_factors import (build_observation_factor,
    AcquisitionTransport, prepare_batch_images, transport_state_to_acquisition)
from .bayes_factor import transition_priors
from .fresh_goal import fresh_marker_goals
from .immutable_body_cache import reexpress_state
from .joint_batch_fusion import fuse_gaussian_likelihood_factors_with_prefixes
from .time_resolved_body_ratios import normalized_parent_evidence, time_resolved_body_ratios
from .model import DemoResult, FrozenBodyBranch, Stage, WeightedState, _joint_record, component_log_weight
from .sensor_event_store import _encode


def _event_record(event):
    payload = event.payload
    allowed = {'time', 'frame_index'} | ({'image', 'image_stamp', 'image_path', 'image_sha256'}
        if event.kind == 'image' else {'gravity_in_sensor', 'imu_target_link', 'imu_sigma_rad'}
        if event.kind == 'imu' else {'command_velocity'} if event.kind == 'joint' else set())
    if event.kind not in ('joint', 'imu', 'image') or set(payload)-allowed:
        raise ValueError('Joint-batch input contains an undeclared acquisition field or sensor.')
    if event.status not in ('pending', 'processing', 'completed'):
        raise ValueError('A joint batch requires ready raw observations.')
    index = payload.get('frame_index')
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError('A joint-batch acquisition needs its original physical frame index.')
    if not np.isfinite(event.stamp) or not np.all(np.isfinite(event.command)):
        raise ValueError('Acquisition command and timestamp must be finite.')
    if event.kind == 'image':
        image = np.asarray(payload['image'])
        if image.dtype != np.uint8 or image.ndim not in (2, 3):
            raise ValueError('An acquired batch image must contain uint8 pixels.')
        if payload.get('image_stamp', event.stamp) != event.stamp:
            raise ValueError('Image pixels and acquisition timestamps disagree.')
        if ('image_sha256' in payload and hashlib.sha256(image.tobytes()).hexdigest()
                != payload['image_sha256'].lower()):
            raise ValueError('Acquired image content differs from its source hash.')
    _, _, digest = _encode(payload)
    if digest != event.payload_hash:
        raise ValueError('A batch payload differs from its immutable acquisition hash.')
    return dict(event_id=event.event_id, sequence=event.sequence, kind=event.kind,
        event_index=index, acquisition_stamp=event.stamp, sent_command=list(event.command),
        observation_id=event.event_id, payload_hash=digest,
        physical_truth_received=False, monte_carlo_used=False)


def _observation(inference, event):
    sensor = {'imu': 'imu', 'image': 'marker'}.get(event.kind)
    templates = [item for item in inference.templates if item['sensor'] == sensor]
    if not templates:
        return None
    if len(templates) != 1:
        raise ValueError('A raw event must identify one enabled observation template.')
    observation = deepcopy(templates[0])
    observation['observation_id'] = event.event_id
    if sensor == 'imu':
        payload = event.payload
        target = str(payload.get('imu_target_link', 'link_3'))
        direction = np.asarray(payload['gravity_in_sensor'], dtype=float)
        sigma = float(payload.get('imu_sigma_rad', .018))
        if (target not in tuple(inference.body_ids)+('tool0',) or direction.shape != (3,)
                or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) == 0.
                or not np.isfinite(sigma) or sigma <= 0.):
            raise ValueError('Invalid acquired IMU direction, attachment or covariance.')
        observation.update(sensor_id='imu_'+target, target_link=target,
            gravity_in_root=[0., 0., -1.], gravity_in_sensor=direction.tolist(),
            direction_sigma_rad=sigma)
    return observation


def update_sensor_batch(inference, events, ledger_steps, *, batch_id):
    """Commit one joint posterior only after all events at this frontier fuse."""
    events, ledger_steps = tuple(events), tuple(ledger_steps)
    if not events:
        raise ValueError('An empty acquisition batch cannot create an update.')
    records = [_event_record(event) for event in events]
    identities = tuple(event.event_id for event in events)
    if len(set(identities)) != len(identities):
        raise ValueError('A joint batch repeats an acquisition identity.')
    if any((a.stamp, a.sequence) >= (b.stamp, b.sequence) for a, b in zip(events, events[1:])):
        raise ValueError('A joint batch must preserve acquisition timestamp and sequence order.')
    if any(identifier in inference._sensor_events for identifier in identities):
        raise ValueError('A consumed acquisition requires explicit checkpoint replay, not double fusion.')
    previous_event = (None if inference.last_result is None else
        inference.last_result.records.get('sensor_event'))
    if previous_event is not None and 'sequence' in previous_event:
        previous_key = (previous_event['acquisition_stamp'], previous_event['sequence'])
        if (events[0].stamp, events[0].sequence) <= previous_key:
            raise ValueError('An acquisition preceding the retained batch cursor requires explicit checkpoint replay.')
    last = events[-1]
    version = 3*int(last.payload['frame_index'])+1
    retained_source_ids = [item.state.state_id for item in inference.components]
    if inference.components:
        if len(inference.components) != 1:
            raise ValueError('Joint batch input must be the retained single internal law.')
        source = inference.components[0].state
        source = replace(source, version=version)
    else:
        first = events[0]
        source = inference.prior_at(first.command, first.stamp, int(first.payload['frame_index']))
        source = replace(source, version=version)
    by_index = {step.index: step for step in ledger_steps}
    if len(by_index) != len(ledger_steps):
        raise ValueError('Actual batch ledger has duplicate intervals.')
    # A later batch starts at the previous posterior's endpoint, not at its
    # first raw event. Its complete source-to-frontier suffix is supplied.
    start_index = (int(previous_event['event_index']) if previous_event is not None
        and 'event_index' in previous_event else ledger_steps[0].index if ledger_steps
        else int(events[0].payload['frame_index']))
    end_index = int(last.payload['frame_index'])
    if end_index < start_index or any(index not in by_index for index in range(start_index, end_index)):
        raise ValueError('Actual command intervals are missing before an acquisition.')
    end_steps = tuple(by_index[index] for index in range(start_index, end_index))
    for event in events:
        # Every event's frame index names its own executed endpoint. Looking
        # up only stamp/command later would miss a corrupted intermediate
        # frame index even when its actual likelihood clock is unchanged.
        end = int(event.payload['frame_index'])
        if not start_index <= end <= end_index:
            raise ValueError('An acquisition frame_index is outside its actual batch ledger.')
        stamp, command = ((source.stamp, source.command) if end == start_index else
                          (by_index[end-1].end_stamp, by_index[end-1].end_command))
        if stamp != event.stamp or tuple(command) != tuple(event.command):
            raise ValueError('An acquisition frame_index differs from its actual ledger endpoint stamp or command.')
    transports = {}
    def transport(state, event):
        if state.state_id not in transports:
            transports[state.state_id] = AcquisitionTransport(state, end_steps,
                inference.body_ids, inference.actions)
        return transports[state.state_id].at(event.stamp, event.command, version=version)
    prepared = prepare_batch_images([event for event in events if _observation(inference, event)
                                   is not None and event.kind == 'image'], inference.common)
    observations = [(event, _observation(inference, event)) for event in events]
    effective_count = sum(observation is not None and (event.kind != 'image'
        or prepared[event.event_id].detected is not None) for event, observation in observations)
    log_keep = effective_count*np.log1p(-inference.change_probability)
    rho = float(-np.expm1(log_keep))
    transition = inference.surprise_enabled and inference.change_probability > 0. and effective_count > 0
    retained = WeightedState(source.component_id, 1., source, 0.)
    transition_record = None
    if transition:
        direct = replace(inference.prior_at(source.command, source.stamp,
            int(last.payload['frame_index'])), version=version)
        parents, transition_record = transition_priors(retained, direct,
            inference.change_probability, marker_id=inference.marker.edge_id)
        parents = (replace(parents[0], weight=float(np.exp(log_keep)), log_weight=float(log_keep)),
                   replace(parents[1], weight=rho, log_weight=float(np.log(rho))))
        transition_record.update(change_probability=rho, per_event_change_probability=inference.change_probability,
            effective_event_count=effective_count, log_keep_prior_probability=float(log_keep),
            change_prior_odds=None, log_change_prior_odds=float(np.log(rho)-log_keep),
            reset_location='batch start; explicit approximation to unknown within-batch change time')
    else:
        parents = (retained,)
    image = (np.zeros((int(inference.common['camera']['height']),
                       int(inference.common['camera']['width']), 3), dtype=np.uint8)
             if inference._last_event_image is None else inference._last_event_image)
    if prepared:
        image = np.asarray(next(event for event in reversed(events)
            if event.event_id in prepared).payload['image'])
    parent_factors, fused_parents, endpoint_parents, prefix_fusions = [], [], [], []
    for parent in parents:
        factors = []
        for event_index, (event, observation) in enumerate(observations):
            if observation is None:
                continue
            state, chart, history = transport(parent.state, event)
            factor = build_observation_factor(parent.state, state, observation,
                inference.paths, inference.common,
                prepared[event.event_id].image if event.kind == 'image' else image,
                common_to_acquisition=chart, event_metadata=records[event_index],
                ledger_record=history, prepared_marker=prepared.get(event.event_id))
            factors.append(factor)
        history = fuse_gaussian_likelihood_factors_with_prefixes(Gaussian(parent.state.mean,
            parent.state.covariance), factors, source_ids=identities)
        fusion = history.fusion
        prefix_fusions.append(history)
        endpoint, chart, _ = transport(parent.state, last)
        parent_factors.append(tuple(factors)); fused_parents.append((fusion, chart))
        endpoint_parents.append(replace(parent, state=endpoint))
    parent_posterior_logs, evidence_parts = normalized_parent_evidence(
        [component_log_weight(parent) for parent in parents],
        [fusion.log_evidence for fusion, _ in fused_parents],
        [fusion.log_evidence_parts for fusion, _ in fused_parents])
    evidence = float(sum(evidence_parts))
    children, displayed, branches = [], [], []
    for parent_index, (parent, (fusion, chart)) in enumerate(zip(endpoint_parents, fused_parents)):
        native = []
        for child in fusion.posterior_components:
            label = (parent.label if effective_count == 0 else
                     parent.label+'|batch:'+str(batch_id)+':'+child.label)
            log_weight = parent_posterior_logs[parent_index]+child.log_weight
            gaussian = Gaussian(chart @ child.distribution.mean,
                chart @ child.distribution.covariance @ chart.T)
            raw = replace(child, label=label, distribution=gaussian,
                weight=float(np.exp(log_weight)), log_weight=float(log_weight))
            state = (parent.state if effective_count == 0 else
                reexpress_state(parent.state, gaussian.mean, gaussian.covariance,
                    version=version+2, component_id=label))
            native.append(raw); children.append(raw)
            displayed.append(WeightedState(label, raw.weight, state, raw.log_weight))
        branches.append(FrozenBodyBranch(parent.label, parent.weight, parent.state,
            Gaussian(parent.state.mean, parent.state.covariance), tuple(native), component_log_weight(parent)))
    endpoint = endpoint_parents[0].state
    effective_ids = tuple(factor.observation_ids[0] for factor in parent_factors[0] if factor.effective)
    sensor_ids = tuple(dict.fromkeys(factor.record['sensor_id'] for factor in parent_factors[0] if factor.effective))
    stage = Stage('fused', tuple(displayed), sensor_ids)
    surprise = dict(enabled=inference.surprise_enabled, applied=transition,
        change_probability=rho if transition else 0., per_event_change_probability=inference.change_probability,
        probability_unit='one joint batch with effective-event survival aggregation',
        effective_event_count=effective_count, transition=transition_record,
        log_keep_evidence=fused_parents[0][0].log_evidence,
        log_reset_evidence=None if not transition else fused_parents[1][0].log_evidence,
        log_observation_evidence=evidence, observation_ids=list(effective_ids),
        posterior_change_probability=None if not transition else float(np.exp(parent_posterior_logs[1])),
        approximation='one batch-start redraw; no enumeration of within-batch change times',
        monte_carlo_used=False, physical_truth_received=False)
    rec = dict(root_frame=inference.common['root_frame'], observation_ids=list(effective_ids),
        valid_observation_count=len(effective_ids), bayes_factor_surprise=surprise,
        joint_observation_evidence=dict(log_evidence=evidence, log_evidence_parts=list(evidence_parts), parents=[dict(
            label=parent.label, log_evidence=fusion.log_evidence,
            log_evidence_parts=list(fusion.log_evidence_parts))
            for parent, (fusion, _) in zip(parents, fused_parents)]),
        stages=[dict(label='fused', sensor_ids=list(sensor_ids), components=[
            _joint_record(item, inference.paths, inference.common['root_frame']) for item in displayed])],
        sensor_estimates=[factor.acquisition_estimate.diagnostics for factor in parent_factors[0]],
        joint_batch=dict(batch_id=str(batch_id), raw_events=records,
            retained_input_state_ids=retained_source_ids,
            raw_event_ids=list(identities), effective_event_ids=list(effective_ids),
            source_stamp=source.stamp, source_command=list(source.command),
            endpoint_stamp=last.stamp, endpoint_command=list(last.command),
            actual_ledger=[step.record() for step in end_steps],
            transport_ledgers=[item.ledger_record for item in transports.values()],
            likelihood_factors=[[factor.record for factor in factors] for factors in parent_factors],
            parent_fusions=[fusion.record for fusion, _ in fused_parents],
            sensor_posterior_views='original acquisition likelihood posteriors pulled into START factor charts; final frozen body branches use END charts',
            intermediate_projection_count=0, terminal_projection_count=1,
            approximation='frozen acquisition likelihoods in one common start chart; one terminal analytic moment/Bingham projection',
            old_per_event_recursion_equivalent=False, monte_carlo_used=False))
    frozen = FusedMixture(tuple(children), (), effective_ids, version, last.stamp, tuple(last.command))
    prior = branches[0].prior if len(branches) == 1 else tuple(PosteriorComponent(
        item.label, item.prior, item.weight, log_weight=item.log_weight) for item in branches)
    result = DemoResult(endpoint, (stage,), image, image, inference.paths['links'],
        inference.paths['hand'], inference.paths['camera'], inference.paths['marker'],
        inference.paths['camera_marker'], inference.common['root_frame'],
        tuple(factor.common_posterior for factors in parent_factors for factor in factors
              if factor.common_posterior is not None), (), rec, prior, frozen, tuple(branches))
    predicted_goals = tuple(replace(item, state=transport(replace(item.state, version=version), last)[0])
                            for item in inference.goal_components)
    # Empty local estimates deliberately defer fresh task construction until
    # the terminal canonical body is available at the image's true clock.
    result = inference._project_internal_update(result, predicted_goals)
    latest_factor = next((factor for factor in reversed(parent_factors[0])
        if factor.effective and factor.acquisition_estimate.detector_result is not None), None)
    if latest_factor is not None:
        capture = next(event for event in events if event.event_id == latest_factor.observation_ids[0])
        suffix = tuple(step for step in end_steps if step.index >= int(capture.payload['frame_index']))
        increment = -sum((np.asarray(step.joint_increment) for step in suffix), np.zeros(len(last.command)))
        canonical = result.stages[-1].components[0].state
        signature = _action_signature(inference.body_ids, inference.actions, len(last.command))
        captured = replace(_recorded_action(canonical, capture.command, increment, signature), stamp=capture.stamp)
        capture_result = replace(result, local_estimates=(latest_factor.acquisition_estimate,))
        goals, construction = fresh_marker_goals(capture_result, captured, marker_id=inference.marker.edge_id)
        capture_stage = replace(result.stages[-1], components=(WeightedState(captured.component_id, 1., captured, 0.),),
                                goal_components=goals)
        capture_result = replace(capture_result, stages=result.stages[:-1]+(capture_stage,))
        capture_result.records['fresh_goal_construction'] = construction
        construction['batch_body_smoothing'] = dict(
            terminal_canonical_state_id=canonical.state_id, terminal_stamp=canonical.stamp,
            terminal_command=list(canonical.command),
            captured_canonical_state_id=captured.state_id, original_image_stamp=capture.stamp,
            original_image_command=list(capture.command),
            posterior_raw_source_clock='batch END',
            future_within_batch_observations_may_inform_captured_body=True,
            inverse_actual_suffix_joint_increment=increment.tolist(),
            actual_suffix=[step.record() for step in suffix])
        inference._record_fixed_goal_memory(capture_result)
        end_goals = []
        for item in goals:
            state = transport_state_to_acquisition(item.state, suffix,
                inference.body_ids, inference.actions, acquisition_stamp=last.stamp,
                acquisition_command=last.command, version=canonical.version)[0]
            marker_index = state.edge_ids.index(inference.marker.edge_id)
            body = [i for j in range(len(state.transforms)) if j != marker_index
                    for i in range(6*j, 6*j+6)]
            covariance = state.covariance.copy()
            covariance[np.ix_(body, body)] = canonical.covariance[np.ix_(body, body)]
            # The body was shared before the inverse/forward action. Reuse
            # that very canonical object data rather than round it twice.
            state = replace(state, transforms=tuple(edge if j == marker_index else canonical.transforms[j]
                for j, edge in enumerate(state.transforms)), covariance=covariance)
            end_goals.append(replace(item, state=state))
        end_goals = tuple(end_goals)
        result = replace(result, stages=result.stages[:-1]+(replace(result.stages[-1], goal_components=end_goals),))
        rec['internal_projection'].update(goal_state_ids=[item.state.state_id for item in end_goals],
            goal_construction='fresh image-time task cut from terminal batch body, then actual ledger transport',
            goal_policy='all native fresh image modes; fixed-interval batch body proxy at original image time',
            conditional_goal_diagnostics_used_for_task=False)
    else:
        inference._record_fixed_goal_memory(result)
    final_stage = result.stages[-1]
    final_goals = tuple(final_stage.goal_components)
    rec['internal_projection'].update(
        goal_state_ids=[item.state.state_id for item in final_goals],
        goal_source_state_ids=[item.state.state_id for item in final_goals],
        goal_labels=[item.label for item in final_goals],
        goal_source_labels=[item.label for item in final_goals],
        goal_source_weights=[item.weight for item in final_goals],
        goal_log_weights=[component_log_weight(item) for item in final_goals],
        goal_source_log_weights=[component_log_weight(item) for item in final_goals],
        goal_parent_internal_state_id=final_stage.components[0].state.state_id)
    rec['stages'][-1] = dict(label=final_stage.label, sensor_ids=list(final_stage.sensor_ids),
        components=[_joint_record(item, inference.paths, result.root_frame) for item in final_stage.components],
        goal_components=[_joint_record(item, inference.paths, result.root_frame) for item in final_goals])
    rec['mixture_policy'] = ('all joint-batch native likelihood product branches and log masses; '
        'one terminal body moment proxy; latest native goal mixture retained without pruning')
    previous_effective_stamp = (None if inference.last_result is None else
        inference.last_result.records.get('joint_batch_source', {}).get('last_effective_observation_stamp'))
    effective_set = set(effective_ids)
    rec['joint_batch_source'] = dict(batch_id=str(batch_id), raw_event_ids=list(identities),
        effective_event_ids=list(effective_ids), retained_input_state_ids=retained_source_ids,
        factor_reference_stamp=source.stamp, endpoint_prediction_stamp=last.stamp,
        batch_last_effective_observation_stamp=max((event.stamp for event in events if event.event_id in effective_set), default=None),
        last_effective_observation_stamp=max((event.stamp for event in events if event.event_id in effective_set), default=previous_effective_stamp),
        terminal_body_state_id=final_stage.components[0].state.state_id,
        terminal_goal_state_ids=[item.state.state_id for item in final_goals])
    estimates = tuple(factor.acquisition_estimate for factor in parent_factors[0])
    detection = next((item.debug_image for item in reversed(estimates)
                      if getattr(item, 'debug_image', None) is not None), image)
    historical_ratios = time_resolved_body_ratios(parents, prefix_fusions, events,
        transport, batch_id=batch_id)
    rec['joint_batch']['time_resolved_stiffness'] = dict(
        historical_ratio_count=len(historical_ratios),
        observation_ids=[item['observation_id'] for item in historical_ratios],
        acquisition_stamps=[item['source_stamp'] for item in historical_ratios],
        acquisition_commands=[list(item['source_command']) for item in historical_ratios],
        comparison_policy='each original acquisition command q_i; FK(K,q_end) is not substituted',
        intermediate_body_projections=0,
        prefix_history=[history.record for history in prefix_fusions])
    result = replace(result, local_estimates=estimates, detection_image=detection,
                     time_resolved_body_ratios=historical_ratios)
    rec['sensor_event'] = dict(records[-1], probability_unit='joint_batch_effective_event_clock',
        batch_id=str(batch_id), chronological_separate_sensor_update=False,
        joint_batch_likelihood_fusion=True)
    rec['prediction_command_correction'] = dict(kind='recorded_executed_command_history',
        steps=[step.record() for step in end_steps])
    rec['moving_input_family'] = dict(law_id=inference.law['law_id'], revision=inference.law['revision'],
        initialization=not bool(inference.components), physical_truth_received=False,
        temporal_policy='joint batch likelihood product; one terminal common-chart moment projection')
    inference._record_candidate_measure(result)
    for event, record in zip(events, records):
        inference._sensor_events[event.event_id] = (event.kind, event.stamp, event.command, record['payload_hash'])
    inference._last_sensor_event_id = last.event_id
    inference._last_event_image = image
    inference.components = tuple(result.stages[-1].components)
    inference.goal_components = tuple(result.stages[-1].goal_components)
    inference.last_result = result
    return result
