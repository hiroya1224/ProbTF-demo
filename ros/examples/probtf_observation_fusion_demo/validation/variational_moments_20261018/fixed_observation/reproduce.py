"""Read original raw-native archive once; no truth, detector rerun or new prior."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from probtf.spherical_law.fusion import Gaussian, PosteriorComponent, condition_gaussian
from probtf_observation_fusion_demo.async_artifacts import load_stiffness_job
from probtf_observation_fusion_demo.bayes_factor import projection_input
from probtf_observation_fusion_demo.body_projection import project_update
from probtf_observation_fusion_demo.immutable_body_cache import reexpress_state
from probtf_observation_fusion_demo.joint_batch_fusion import (
    add_log_evidence_parts, _normalize_log_evidence_parts)
from probtf_observation_fusion_demo.model import FrozenBodyBranch, Stage, WeightedState
from probtf_observation_fusion_demo.moment_observation import (
    NativeMomentLikelihood, PreparedMomentObservation)
from probtf_observation_fusion_demo.moment_reverse import backend_record
from probtf_observation_fusion_demo.variational_conditioning import condition_variational
from probtf_observation_fusion_demo.variational_model import PropagatedLikelihoodExpectation


OUT = Path('/tmp/probtf_variational_job22_audit_warmstart_20261018')
OUT.mkdir(exist_ok=True)
REF = Path('/tmp/probtf_propagated_moments_180s_live_20261018/imu_marker/stiffness_jobs/00000022_4796ca64621e3006.json')
ROOT = Path('/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/probtf_observation_fusion_demo')
SOURCES = [ROOT/'src/probtf_observation_fusion_demo'/name for name in (
    'variational_model.py', 'variational_conditioning.py', 'moment_reverse.py',
    'moment_observation.py', 'body_native.py', 'body_projection.py', 'bayes_factor.py')]
SOURCES += [ROOT/'native/moment_reverse.cpp']


def hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES if p.exists()}


def numeric_hash(law):
    return hashlib.sha256(law.mean.tobytes()+law.covariance.tobytes()).hexdigest()


def summarize(state, native):
    prepared = PreparedMomentObservation(dict(sensor='marker'), (native,), {})
    propagated = prepared.propagate(state)[0]
    mu, S = propagated.query.mean, propagated.query.covariance
    residual = mu-native.observed_mean
    W = np.linalg.inv(native.covariance)
    return dict(state_id=state.state_id, stamp=state.stamp, command=list(state.command),
        predicted_mean=mu.tolist(), predicted_covariance=S.tolist(),
        native_position_residual_m=float(np.linalg.norm(residual[:3])),
        native_rotation_chart_residual_rad=float(np.linalg.norm(residual[3:])),
        whitened_squared_mean_residual=float(residual@W@residual),
        expected_whitened_squared_residual=float(residual@W@residual+np.trace(W@S)),
        full_joint_covariance_trace=float(np.trace(state.covariance)),
        maximum_local_rotation_variance=float(max(np.linalg.eigvalsh(
            state.covariance[6*i+3:6*i+6,6*i+3:6*i+6]).max() for i in range(len(state.transforms)))))


before = hashes()
reference = json.loads(REF.read_text())
job = load_stiffness_job(reference)
old = job['result']
event = job['events'][0]
record = old.records['high_concentration_observation_update']['native_likelihoods'][0]
keys = tuple(NativeMomentLikelihood.__dataclass_fields__)
native = NativeMomentLikelihood(**{key: record[key] for key in keys if key in record})
report = dict(schema='original-job22-fixed-prior-variational-audit-v1',
    original_archive=dict(reference_path=str(REF), **reference),
    original_event=event, native_likelihood=native.record(),
    command_ledger_count=len(job['command_ledger']),
    historical_prior_operator='propagated_moments', truth_received=False,
    raw_pixels_received=False, current_backend=backend_record(),
    approximation='Gaussian KL plus high-concentration propagated moment likelihood expectation; not exact nonlinear evidence/lower bound',
    original_prior_fixed=True, initialization='production affine Gaussian initial candidate, compared using the same F and original fixed prior', cases=[])

for budget in (24, 100):
    case = dict(max_iterations=budget, branches=[])
    outputs = []
    evidence = []
    for branch in old.frozen_body_branches:
        evaluator = PropagatedLikelihoodExpectation(branch.reference_state, (native,))
        source_hash = numeric_hash(branch.prior)
        started = time.perf_counter()
        propagated = PreparedMomentObservation(dict(sensor='marker'), (native,), {}).propagate(branch.reference_state)[0].query
        seed, _ = condition_gaussian(branch.prior, native.observed_mean-propagated.mean+propagated.jacobian@branch.prior.mean, native.covariance, propagated.jacobian)
        result = condition_variational(branch.prior, evaluator, max_iterations=budget, initial_distribution=seed)
        seconds = time.perf_counter()-started
        raw = result.distribution
        materialized = reexpress_state(branch.reference_state, raw.mean, raw.covariance,
            version=old.stages[-1].components[0].state.version,
            component_id=branch.posterior_components[0].label)
        at = evaluator(raw.mean, raw.covariance)
        core = summarize(materialized, native)
        optimizer_moment = at.diagnostics['propagated_observations'][0]
        mu = np.asarray(optimizer_moment['predicted_mean'])
        S = np.asarray(optimizer_moment['predicted_covariance'])
        b = dict(parent_label=branch.label, reference_state_id=branch.reference_state.state_id,
            prior_numeric_sha256=source_hash, prior_unchanged=source_hash==numeric_hash(branch.prior),
            prior=summarize(branch.reference_state, native),
            old_frozen_posterior=summarize(next(x.state for x in old.stages[-2].components
                if x.label==branch.posterior_components[0].label), native)
                if any(x.label==branch.posterior_components[0].label for x in old.stages[-2].components) else None,
            elapsed_seconds=seconds, raw_posterior_numeric_sha256=numeric_hash(raw),
            mean_shift_norm=float(np.linalg.norm(raw.mean-branch.prior.mean)),
            covariance_trace=float(np.trace(raw.covariance)),
            raw_posterior_local_rotation_shift_norms=[float(np.linalg.norm(raw.mean[6*i+3:6*i+6]))
                for i in range(len(branch.reference_state.transforms))],
            final_materialized=core,
            optimizer_vs_materialized_mean_max_abs=float(np.max(np.abs(mu-np.asarray(core['predicted_mean'])))),
            optimizer_vs_materialized_covariance_max_abs=float(np.max(np.abs(S-np.asarray(core['predicted_covariance'])))),
            log_evidence=result.log_evidence, log_evidence_parts=list(result.log_evidence_parts),
            diagnostics=result.diagnostics)
        # Evaluate the historical raw posterior in its correct original chart,
        # then reexpress once. Zero means in a different chart are never used.
        old_raw = branch.posterior_components[0].distribution
        old_state = reexpress_state(branch.reference_state, old_raw.mean, old_raw.covariance,
            version=materialized.version, component_id=branch.posterior_components[0].label)
        b['old_frozen_posterior'] = summarize(old_state, native)
        case['branches'].append(b)
        outputs.append((branch, raw, materialized))
        evidence.append(add_log_evidence_parts(result.log_evidence_parts,
            (branch.log_weight, native.log_weight)))
        print('budget', budget, 'parent', branch.label, 'seconds', round(seconds, 4),
            'residual_m', core['native_position_residual_m'], 'gradient', result.diagnostics['final_gradient_inf'], flush=True)
    logweights, total = _normalize_log_evidence_parts(evidence)
    frozen, displayed = [], []
    for (branch, raw, state), logweight, b in zip(outputs, logweights, case['branches']):
        logweight = float(logweight); weight = math.exp(logweight)
        label = state.component_id
        child = PosteriorComponent(label, raw, weight, log_weight=logweight)
        displayed.append(WeightedState(label, weight, state, logweight))
        frozen.append(FrozenBodyBranch(branch.label, branch.weight, branch.reference_state,
            branch.prior, (child,), branch.log_weight))
        b['posterior_weight'] = weight
        b['posterior_log_weight'] = logweight
    raw_result = replace(old, stages=(old.stages[0], Stage('fused', tuple(displayed), ('marker',))),
        frozen_body_branches=tuple(frozen))
    canonical = project_update(projection_input(raw_result))
    case['canonical'] = summarize(canonical.state, native)
    case['canonical_source_count'] = len(displayed)
    case['log_total_evidence_parts'] = list(total)
    case['log_total_evidence'] = math.fsum(total)
    report['cases'].append(case)

report['source_sha256_before'] = before
report['source_sha256_after'] = hashes()
report['source_unchanged'] = report['source_sha256_after']==before
(OUT/'report.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n')
print('report', OUT/'report.json', flush=True)
