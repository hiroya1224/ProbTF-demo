"""Historical body density ratios at each observation's actual command.

All-prefix Gaussian laws are computational intermediates in a frozen batch,
not separately installed/displayed body states. Only the terminal body proxy
is adopted. Each stiffness ratio nevertheless keeps its own original command
and is evaluated by FK(K, q_i), rather than FK(K, q_end).
"""
from dataclasses import replace

import numpy as np
from scipy.special import logsumexp

from probtf.spherical_law.fusion import Gaussian, PosteriorComponent

from .model import FrozenBodyBranch, component_log_weight
from .joint_batch_fusion import add_log_evidence_parts, log_evidence_parts_difference


def normalized_parent_evidence(parent_logs, evidences, evidence_parts=None):
    """Cancel the common likelihood scale before adding prior log masses.

    The two-term representation retains finite evidence differences smaller
    than one ULP of the shared large log likelihood. It describes the same
    scalar mass; no mixture branch or information is removed.
    """
    evidences = tuple(evidences)
    if not evidences or len(evidences) != len(parent_logs):
        raise ValueError('Each parent needs its own complete evidence.')
    supplied_parts = (() if evidence_parts is None else tuple(evidence_parts))
    if supplied_parts and len(supplied_parts) != len(evidences):
        raise ValueError('Each parent needs its own complete evidence parts.')
    parts = tuple(tuple(part) if part else (float(value), 0.) for value, part in
                  zip(evidences, supplied_parts or ((),)*len(evidences)))
    parent_logs = np.asarray(parent_logs)
    active = np.isfinite(parent_logs)
    if not active.any() or np.any(~active & (parent_logs != -np.inf)):
        raise ValueError('Joint parent log masses do not define a positive finite mixture.')
    reference = max((part for part, enabled in zip(parts, active) if enabled),
                    key=lambda part: (part[0], part[1]))
    shifted = np.array([log_evidence_parts_difference(part, reference) if enabled else -np.inf
                        for part, enabled in zip(parts, active)])
    logits = parent_logs+shifted
    normalizer = float(logsumexp(logits))
    if not np.isfinite(normalizer):
        raise ValueError('Joint parents have no finite positive normalizing mass.')
    return logits-normalizer, add_log_evidence_parts(reference, (normalizer,))


def time_resolved_body_ratios(parents, prefix_fusions, events, transport, *, batch_id):
    """Normalize all parent/native branches jointly at every causal prefix."""
    parents, prefix_fusions = tuple(parents), tuple(prefix_fusions)
    if len(parents) != len(prefix_fusions) or not parents:
        raise ValueError('Every frozen parent requires its complete observation-prefix laws.')
    sequences = [tuple(step.observation_ids for step in fusion.prefixes) for fusion in prefix_fusions]
    if any(sequence != sequences[0] for sequence in sequences[1:]):
        raise ValueError('Keep/Reset prefix laws disagree on effective acquisition order.')
    by_id = {event.event_id: event for event in events}
    specifications, previous_evidences = [], np.zeros(len(parents))
    previous_parts = ((0., 0.),)*len(parents)
    parent_logs = np.array([component_log_weight(parent) for parent in parents])
    for index, observation_ids in enumerate(sequences[0]):
        if len(observation_ids) != 1 or observation_ids[0] not in by_id:
            raise ValueError('One density-ratio step must name one original sensor acquisition.')
        event = by_id[observation_ids[0]]
        steps = tuple(fusion.prefixes[index] for fusion in prefix_fusions)
        evidences = np.array([step.cumulative_log_evidence for step in steps])
        evidence_parts = tuple(getattr(step, 'cumulative_log_evidence_parts', ())
                               or (float(value), 0.) for step, value in zip(steps, evidences))
        before_logs, _ = normalized_parent_evidence(parent_logs, previous_evidences, previous_parts)
        after_logs, _ = normalized_parent_evidence(parent_logs, evidences, evidence_parts)
        branches, source_records = [], []
        for parent_index, (parent, step) in enumerate(zip(parents, steps)):
            acquisition, chart, ledger = transport(parent.state, event)
            children_by_parent = {}
            for child in step.posterior_components:
                children_by_parent.setdefault(child.parent_label, []).append(child)
            for local_index, prior in enumerate(step.prior_components):
                label = 'time-ratio:{}:{}:{}:{}'.format(batch_id, index, parent_index, local_index)
                reference = replace(acquisition, component_id=label)
                prior_law = Gaussian(chart @ prior.distribution.mean,
                    chart @ prior.distribution.covariance @ chart.T)
                log_prior_weight = before_logs[parent_index]+prior.log_weight
                children = []
                for child in children_by_parent.pop(prior.label, ()):
                    log_weight = after_logs[parent_index]+child.log_weight
                    law = Gaussian(chart @ child.distribution.mean,
                        chart @ child.distribution.covariance @ chart.T)
                    children.append(PosteriorComponent(label+'|'+child.label, law,
                        float(np.exp(log_weight)), lineage=(step.prefix_id, event.event_id),
                        log_weight=float(log_weight)))
                if not children:
                    raise ValueError('A historical prior branch lost all native observation children.')
                branches.append(FrozenBodyBranch(label, float(np.exp(log_prior_weight)),
                    reference, prior_law, tuple(children), float(log_prior_weight)))
            if children_by_parent:
                raise ValueError('A historical posterior child has no actual prefix prior.')
            source_records.append(dict(parent_label=parent.label,
                common_start_state_id=parent.state.state_id,
                acquisition_state_id=acquisition.state_id,
                prefix_id=step.prefix_id, previous_prefix_id=step.previous_prefix_id,
                actual_ledger_prefix=ledger,
                cumulative_log_evidence_before=float(previous_evidences[parent_index]),
                cumulative_log_evidence_after=float(evidences[parent_index]),
                cumulative_log_evidence_parts_before=list(previous_parts[parent_index]),
                cumulative_log_evidence_parts_after=list(evidence_parts[parent_index])))
        specifications.append(dict(observation_id=event.event_id,
            source_stamp=event.stamp, source_command=tuple(event.command),
            branches=tuple(branches), posterior_branch=None,
            provenance=dict(batch_id=str(batch_id), prefix_index=index,
                event_sequence=event.sequence, payload_hash=event.payload_hash,
                acquisition_frame_index=int(event.payload['frame_index']),
                prefix_sources=source_records,
                source_clock='original acquisition', comparison_clock='original acquisition',
                intermediate_body_projection_applied=False,
                terminal_command_substituted=False,
                stiffness_prediction='FK(K, original acquisition q_i) for every continuous K query',
                approximation='full native prefix Gaussian mixtures in frozen Bingham-derived charts; not the terminal projected body ratio')))
        previous_evidences = evidences
        previous_parts = evidence_parts
    return tuple(specifications)
