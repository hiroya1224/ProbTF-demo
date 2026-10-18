#!/usr/bin/env python3
"""Frozen-law comparison only: no observations, K posterior, or control update.

Run after sourcing the catkin workspace, using /usr/bin/python3. Input jobs
must be trusted local pickle archives from ordinary, truth-free runs. The
saved simulation K and physical tip are read solely as posthoc evaluation
parameters; they are never installed in inference or an execution packet.
"""
import argparse
from copy import copy
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
import yaml

from probtf.bingham import bingham_log_normalizer
from probtf.geometry import quat_left_matrix, quat_to_rotmat, rotmat_to_quat
from probtf.spherical_law.fusion import Gaussian
from probtf_observation_fusion_demo.body_native import materialize_state_in_chart, pose_query
from probtf_observation_fusion_demo.hand_stiffness import HandPoseDensityRatio
from probtf_observation_fusion_demo.joint_rotation_stiffness import JointRotationDensityRatio
from probtf_observation_fusion_demo.stiffness_identification import (
    ChartedBodyDensityRatio, StiffnessIdentifier, _log_weight,
)


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bingham_value_score(parameter, quaternion, derivative, steps=120):
    """Normalized S3 density; same normalization convention in both laws."""
    parameter, quaternion = np.asarray(parameter), np.asarray(quaternion)
    return (float(quaternion @ parameter @ quaternion-bingham_log_normalizer(parameter, steps)),
            2.*np.asarray(derivative).T @ parameter @ quaternion)


class IndependentLocalBinghamRatio:
    """Diagnostic independence inside each retained global source branch.

    Every native prior/child mass remains. Only local Bingham marginals are
    multiplied, so within-branch link cross covariance is intentionally lost.
    The shared Keep/Reset branch itself is not independently redrawn per link.
    """
    def __init__(self, branches, body_ids, posterior_branch=None, steps=120):
        self.steps, self.body_ids = steps, tuple(body_ids)
        self.minus, self.plus = [], []
        for branch in branches:
            def local_laws(distribution):
                reference = branch.reference_state
                state = reference if (np.array_equal(distribution.mean, reference.mean)
                    and np.array_equal(distribution.covariance, reference.covariance)) else materialize_state_in_chart(
                        reference, distribution.mean, distribution.covariance)
                by_id = {item.edge_id: item for item in state.transforms}
                return tuple((np.array(by_id[name].bingham, copy=True),
                              bingham_log_normalizer(by_id[name].bingham, steps))
                             for name in body_ids)
            self.minus.append((_log_weight(branch), local_laws(branch.prior)))
            sources = branch.posterior_components if posterior_branch is None else (
                posterior_branch.posterior_components if branch.reference_state.state_id
                == posterior_branch.reference_state.state_id else ())
            self.plus.extend((_log_weight(child), local_laws(child.distribution)) for child in sources)

    def evaluate(self, quaternions, derivatives):
        def mixture(components):
            values, gradients = [], []
            for weight, laws in components:
                terms = [float(q @ parameter @ q-normalizer)
                         for q, (parameter, normalizer) in zip(quaternions, laws)]
                values.append(weight+sum(terms))
                gradients.append(sum((2.*dq.T @ parameter @ q for q, dq, (parameter, _)
                    in zip(quaternions, derivatives, laws)), np.zeros(derivatives[0].shape[1])))
            total = float(logsumexp(values))
            return total, np.exp(np.asarray(values)-total) @ np.asarray(gradients)
        minus, minus_score = mixture(self.minus)
        plus, plus_score = mixture(self.plus)
        return plus-minus, plus_score-minus_score


def block_diagonal_rotation_ratio(ratio):
    """Read-only cross ablation of rotational Gaussians, not a body update."""
    result = copy(ratio)
    def diagonal(distribution):
        covariance = np.zeros_like(distribution.covariance)
        for start in range(0, covariance.shape[0], 3):
            part = slice(start, start+3)
            covariance[part, part] = distribution.covariance[part, part]
        return Gaussian(distribution.mean, covariance)
    marginals = []
    for prior, children, _ in ratio.marginals:
        prior = diagonal(prior)
        children = tuple((child, diagonal(distribution)) for child, distribution in children)
        marginals.append((prior, children, prior.support))
    result.marginals = marginals
    result.child_supports = [tuple(distribution.support for _, distribution in children)
                             for _, children, _ in marginals]
    return result


def factor_set(result, body_ids):
    branches, posterior = result.frozen_body_branches, result.projected_body_numerator
    rotation = JointRotationDensityRatio(branches, body_ids, posterior_branch=posterior)
    return dict(joint_rotation=rotation,
        independent_rotation_gaussian=block_diagonal_rotation_ratio(rotation),
        independent_local_bingham=IndependentLocalBinghamRatio(branches, body_ids, posterior),
        full_body_legacy=ChartedBodyDensityRatio(branches, body_ids, posterior_branch=posterior),
        hand_pose_previous=HandPoseDensityRatio(branches, body_ids, result.hand_path, posterior))


def evaluate_factors(identifier, result, factors, logK):
    """One auxiliary equilibrium is shared by every competing factor."""
    physical = identifier._physical_prediction(np.asarray(logK), result.prior)
    predictions = tuple(identifier._body_chart(physical['theta'], physical['derivative'],
        branch.reference_state, local_poses=physical['local_poses'])
        for branch in result.frozen_body_branches)
    quaternions, derivatives = [], []
    for (_, _, _, joint, axis), (_, rotation) in zip(identifier.edges, physical['local_poses']):
        quaternion = rotmat_to_quat(rotation)
        omega = np.outer(axis, physical['derivative'][joint])
        quaternions.append(quaternion)
        derivatives.append(.5*quat_left_matrix(quaternion)[:, 1:] @ omega)
    hand = identifier._hand_prediction(physical, result.prior, result.hand_path)
    values = {}
    for name, factor in factors.items():
        try:
            if name == 'independent_local_bingham':
                value, score = factor.evaluate(quaternions, derivatives)
            elif name == 'hand_pose_previous':
                value, score = factor.evaluate_pose(*hand)
            else:
                value, score = factor.evaluate(predictions)
            values[name] = dict(log_factor=float(value), logK_score=np.asarray(score).tolist())
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            values[name] = dict(unavailable=type(error).__name__+': '+str(error))
    return values, physical, hand


def numeric_scores(evaluate, logK, step=1.e-5):
    output = {}
    for index in range(len(logK)):
        offset = step*np.eye(len(logK))[index]
        plus, minus = evaluate(logK+offset)[0], evaluate(logK-offset)[0]
        for name in plus.keys() & minus.keys():
            if 'log_factor' in plus[name] and 'log_factor' in minus[name]:
                output.setdefault(name, {})[index] = (plus[name]['log_factor']-minus[name]['log_factor'])/(2.*step)
    return {name: [columns[index] for index in range(len(logK))] for name, columns in output.items()
            if len(columns) == len(logK)}


def physical_local_rotation_errors(theta, actual_joints, edges):
    """Posthoc angles only, for the declared fixed-origin scalar revolutes.

    Both arrays use the same known URDF generalized-coordinate ordering.
    Each edge carries its actual Pinocchio idx_q; edge order is not assumed
    to equal joint order. Fixed origins cancel in the relative local angle.
    Saved actual joints are never passed to the auxiliary model calculation.
    """
    theta, actual_joints = np.asarray(theta), np.asarray(actual_joints)
    differences = np.array([theta[joint]-actual_joints[joint]
                            for _, _, _, joint, _ in edges])
    return np.abs(np.arctan2(np.sin(differences), np.cos(differences))).tolist()


def compare_job(run, message, physical_rows, settings, finite_difference=True):
    sequence = message['job_sequence']
    job_path = next((run/'stiffness_jobs').glob(str(sequence).zfill(8)+'_*.pickle'))
    job = pickle.loads(job_path.read_bytes())
    result = job['result']
    if (message['source_stamp'] != result.prior.stamp
            or not np.array_equal(message['source_command'], result.prior.command)
            or any(branch.reference_state.stamp != result.prior.stamp
                   or branch.reference_state.command != result.prior.command
                   for branch in result.frozen_body_branches)):
        raise ValueError('Frozen factor comparisons must retain the exact original acquisition command and time.')
    body_ids = tuple(name for name, _ in result.link_paths['link_6'] if name.startswith('link_'))
    identifier = StiffnessIdentifier(run/'robot.urdf', body_ids, settings)
    identifier._freeze_branch(result.prior, {})
    factors = factor_set(result, body_ids)
    frozen = {item.state.state_id: item.state for item in result.stages[-1].components}
    canonical = result.stages[-1].components[0].state
    hand_query = pose_query(canonical, result.hand_path, query_id='diagnostic:frozen_hand', frame=result.root_frame)
    by_id = {item.edge_id: item for item in canonical.transforms}
    actual = physical_rows.get(float(result.prior.stamp))
    if actual is not None and not np.array_equal(actual['command'], result.prior.command):
        raise ValueError('Physical posthoc record has a different original sent command.')
    initial = identifier.initial_components[0].distribution.mean
    queries = [('initial_prior_mode', initial, 'normal_parameter_query')]
    proposal = message['stiffness_record'].get('command_proposal') or {}
    if proposal.get('selected_logK') is not None:
        queries.append(('recorded_selected_K', np.asarray(proposal['selected_logK']), 'normal_parameter_query'))
    if actual is not None:
        queries.append(('simulation_K_posthoc_only', np.log(actual['physics']['stiffness']),
                        'evaluation_parameter_only_not_inference_input'))
    evaluations = []
    evaluate = lambda x: evaluate_factors(identifier, result, factors, x)
    for name, logK, role in queries:
        try:
            values, physical, hand = evaluate(np.asarray(logK))
            if finite_difference:
                numeric = numeric_scores(evaluate, np.asarray(logK))
                for factor_name, expected in numeric.items():
                    score = np.asarray(values[factor_name]['logK_score'])
                    error = np.asarray(expected)-score
                    values[factor_name].update(numeric_logK_score=expected,
                        score_FD_max_abs_error=float(np.max(np.abs(error))),
                        score_FD_relative_norm_error=float(np.linalg.norm(error)/max(1., np.linalg.norm(score))))
            angles = [float(np.linalg.norm(Rotation.from_matrix(
                quat_to_rotmat(by_id[edge[0]].rotation_wxyz).T @ pose[1]).as_rotvec()))
                for edge, pose in zip(identifier.edges, physical['local_poses'])]
            evaluations.append(dict(query=name, truth_role=role, logK=np.asarray(logK).tolist(),
                K=np.exp(logK).tolist(), factors=values,
                local_rotation_error_to_estimated_modal_rad=angles,
                local_rotation_error_to_physical_rad=None if actual is None else
                    physical_local_rotation_errors(physical['theta'], actual['actual_joints'], identifier.edges),
                model_hand_position=hand[0].tolist(), estimated_hand_position_mean=hand_query.mean[:3].tolist(),
                model_hand_to_estimated_mean_error_m=float(np.linalg.norm(hand[0]-hand_query.mean[:3])),
                model_hand_to_estimated_mode_error_m=float(np.linalg.norm(hand[0]-hand_query.mode[:3])),
                physical_tip_posthoc=None if actual is None else actual['tip'],
                model_hand_to_physical_tip_error_m=None if actual is None else float(np.linalg.norm(hand[0]-actual['tip'])),
                estimated_hand_mean_to_physical_tip_error_m=None if actual is None else float(np.linalg.norm(hand_query.mean[:3]-actual['tip'])),
                auxiliary_active_joint_limits=physical['active'].tolist()))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            evaluations.append(dict(query=name, truth_role=role, unavailable=type(error).__name__+': '+str(error)))
    diagonal = block_diagonal_rotation_ratio(factors['joint_rotation'])
    prior_cross, posterior_cross = [], []
    for (prior, children, _), (independent_prior, independent_children, _) in zip(
            factors['joint_rotation'].marginals, diagonal.marginals):
        prior_cross.append(float(np.max(np.abs(prior.covariance-independent_prior.covariance))))
        posterior_cross.extend(float(np.max(np.abs(child.covariance-independent_child.covariance)))
            for (_, child), (_, independent_child) in zip(children, independent_children))
    rankings = {}
    for factor_name in factors:
        usable = [item for item in evaluations if 'log_factor' in item.get('factors', {}).get(factor_name, {})]
        rankings[factor_name] = [item['query'] for item in sorted(usable,
            key=lambda item: item['factors'][factor_name]['log_factor'], reverse=True)]
    return dict(job_sequence=sequence, frozen_job_path=str(job_path), frozen_job_sha256=sha256(job_path),
        selected_K_message_sha256=hashlib.sha256(json.dumps(message, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
        source_stamp=result.prior.stamp, source_command=list(result.prior.command),
        observation_ids=list(result.records['observation_ids']), body_state_ids=list(frozen),
        hand_path=[list(item) for item in result.hand_path],
        physical_posthoc_exact_stamp_match=actual is not None,
        physical_orientation_diagnostic=dict(source='saved physical_trajectory.jsonl actual_joints at exact source t/q',
            truth_role='posthoc_comparison_only_not_auxiliary_model_input',
            joint_indices=[dict(edge_id=name, urdf_generalized_coordinate_index=joint)
                           for name, _, _, joint, _ in identifier.edges],
            formula='abs(atan2(sin(theta_model[idx_q]-theta_phys[idx_q]), cos(theta_model[idx_q]-theta_phys[idx_q]))) for fixed-origin single-axis revolute edges'),
        body_prior_branch_count=len(result.frozen_body_branches),
        rotation_cross_covariance_max_abs_prior=max(prior_cross, default=0.),
        rotation_cross_covariance_max_abs_posterior=max(posterior_cross, default=0.),
        finite_query_rankings_not_optimization=rankings,
        evaluations=evaluations)


def compare_run(run, count=3, sequences=None):
    messages = [message for message in rows(run/'stiffness.jsonl')
                if message.get('stiffness_record', {}).get('update_applied')]
    if sequences is None:
        indices = sorted(set(int(round(value)) for value in np.linspace(0., len(messages)-1, min(count, len(messages)))))
        selected = [messages[index] for index in indices]
    else:
        by_sequence = {message['job_sequence']: message for message in messages}
        selected = [by_sequence[sequence] for sequence in sequences]
    stamps = {message['source_stamp'] for message in selected}
    physical_rows = {item['stamp']: item for item in rows(run/'physical_trajectory.jsonl') if item['stamp'] in stamps}
    import rospkg
    package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
    settings = yaml.safe_load((package/'config/stiffness_identification.yaml').read_text())
    settings = dict(settings, matching_target='joint_rotation')
    return dict(case=run.name, input_directory=str(run), ordinary_frozen_source=True,
        original_physical_tip_used_for_posthoc_only=True,
        stiffness_posterior_updated=False, inference_or_control_truth_supplied=False,
        jobs=[compare_job(run, message, physical_rows, settings) for message in selected])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--jobs-per-run', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-from', type=Path,
        help='Reuse exact immutable job IDs from an earlier comparison, even while archive journals continue draining.')
    parser.add_argument('--protected-baseline', type=Path,
        default=Path('/tmp/probtf_no_concentration_authorized_baseline_20261018.json'))
    args = parser.parse_args()
    selection = ({} if args.selection_from is None else {
        item['input_directory']: [job['job_sequence'] for job in item['jobs']]
        for item in json.loads(args.selection_from.read_text())['runs']})
    import rospkg
    package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
    node = package/'src/probtf_observation_fusion_demo/node.py'
    before_node = sha256(node)
    protected = json.loads(args.protected_baseline.read_text())['protected_sha256']
    before_protected = {path: sha256(path) for path in protected}
    report = dict(method='same frozen ordinary before/after body laws, original sent command, one shared auxiliary equilibrium per logK query',
        independent_bingham='proper normalized S3 local Bingham product within each retained global branch; global branch masses retained',
        independent_gaussian='diagnostic rotational Gaussian block diagonalization only; original body law is not changed',
        K_coordinate='x=log(K); score is d log factor / d x; K in Nm/rad; rotations in rad; positions in m',
        finite_difference='deterministic central logK differences, step 1e-5; no Monte Carlo',
        bingham_normalizer_integration_steps=120,
        selection_reproduction='Pass --selection-from this JSON to reuse exact job IDs; default first/middle/last is relative to the completed-job journal at selection time.',
        limitations=[
            'Frozen factor values and derivatives do not measure a new closed-loop outcome.',
            'Joint rotational Gaussian and independent Bingham comparisons differ in both dependence and local density approximation; the block-diagonal Gaussian comparison isolates rotational cross blocks.',
            'All local orientation fit is not an equivalent guarantee of end-effector position fit.',
            'The physical simulator is dynamic, while the auxiliary model is quasistatic. Simulation K need not be the best effective approximation at a moving acquisition.',
            'Numerical K representatives are query points, not samples or a replacement stiffness distribution.',
            'Simulation K is read only from saved physical diagnostics after freezing normal inference laws; no estimate, K posterior, or execution packet receives truth.',
            'A finite-difference mismatch is diagnostic and is never used to stop a production update.'],
        runs=[compare_run(run, args.jobs_per_run, selection.get(str(run))) for run in args.runs])
    report['read_only_source_integrity'] = dict(
        tool_sha256=sha256(__file__), protected_baseline=str(args.protected_baseline),
        protected_baseline_sha256=sha256(args.protected_baseline),
        protected_file_count=len(protected), rviz_file_count=sum(Path(path).suffix == '.rviz' for path in protected),
        mismatched_against_baseline=[path for path, digest in protected.items() if before_protected[path] != digest],
        changed_during_comparison=[path for path in protected if sha256(path) != before_protected[path]],
        node_path=str(node), node_sha256_before=before_node, node_sha256_after=sha256(node))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), bytes=args.output.stat().st_size)))


if __name__ == '__main__':
    main()
