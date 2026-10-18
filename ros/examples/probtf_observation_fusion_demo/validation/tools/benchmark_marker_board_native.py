#!/usr/bin/env python3
"""Replay recorded native image mixtures through both board arithmetic paths.

The small fixture contains all individual image modes, covariance Schur
factors and log masses; it contains no pixels, simulator poses or true K.
This measures board fusion only, not ArUco detection or native image fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


PACKAGE = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = PACKAGE/'tests/data/marker_board_native_likelihoods_20261018.json'


def load_recorded_detections(record):
    from prob_artag_detector.models import EstimationDiagnostics, PoseMixtureResult
    from probtf.distributions.bingham_orientation import BinghamOrientation
    from probtf.distributions.conditional_translation import ConditionalGaussianTranslation
    from probtf.distributions.stamped import TransformDistributionStamped
    from probtf.distributions.transform_component import TransformComponent
    from probtf.distributions.transform_distribution import TransformDistribution
    from probtf.geometry import quat_to_rotmat
    output = {}
    for marker in record['raw_native_likelihoods']:
        components = tuple(TransformComponent(mode['native_component_id'], mode['weight'],
            BinghamOrientation.from_parameter_matrix(np.asarray(mode['native_bingham'])),
            ConditionalGaussianTranslation(np.asarray(mode['native_translation_mean']),
                np.asarray(mode['native_translation_residual_covariance']),
                np.asarray(mode['native_translation_rotation_coupling'])))
            for mode in marker['components'])
        stamped = TransformDistributionStamped(marker['parent_frame_id'], marker['child_frame_id'],
            marker['stamp'], marker['edge_id'], marker['authority'], TransformDistribution(components))
        count = len(components)
        output[marker['marker_id']] = PoseMixtureResult(stamped,
            EstimationDiagnostics(count, count, 0, 0, 0, 0, ()),
            tuple(quat_to_rotmat(item.orientation.mode_wxyz) for item in components),
            tuple(item.translation.mean_at_reference for item in components),
            tuple(item['native_log_mass'] for item in marker['components']),
            tuple(item['weight'] for item in marker['components']),
            tuple(item['native_seed_index'] for item in marker['components']))
    return output


def numerical_comparison(reference, actual):
    from probtf.bingham import bingham_second_moment
    a, b = reference.record, actual.record
    checks = dict(
        native_ID_order=a['fusion_order'] == b['fusion_order'],
        all_native_modes=[x['native_component_count'] for x in a['raw_observations']]
                        == [x['native_component_count'] for x in b['raw_observations']],
        all_product_branch_counts=[len(x['branch_products']) for x in a['steps']]
                                  == [len(x['branch_products']) for x in b['steps']],
        reduction_and_fallback=[x['projection']['applied'] for x in a['steps']]
                              == [x['projection']['applied'] for x in b['steps']],
        output_modes=a['central_component_count'] == b['central_component_count'])
    maximum = dict(translation=0., covariance=0., quaternion_second_moment=0., log_weight=0.,
                   projection_log_mass=0., native_center_query_mean=0., native_center_query_covariance=0.)
    for ra, rb in zip(a['raw_observations'], b['raw_observations']):
        for ca, cb in zip(ra['components'], rb['components']):
            maximum['native_center_query_mean'] = max(maximum['native_center_query_mean'],
                float(np.max(np.abs(np.asarray(ca['core_center_pose_query']['mean'])
                                    - cb['core_center_pose_query']['mean']))))
            maximum['native_center_query_covariance'] = max(maximum['native_center_query_covariance'],
                float(np.max(np.abs(np.asarray(ca['core_center_pose_query']['covariance'])
                                    - cb['core_center_pose_query']['covariance']))))
    for pa, pb in zip(a['steps'], b['steps']):
        maximum['projection_log_mass'] = max(maximum['projection_log_mass'],
            abs(pa['projection']['log_normalization']-pb['projection']['log_normalization']))
    for ca, cb, la, lb in zip(reference.detected.record.distribution.components,
            actual.detected.record.distribution.components, reference.detected.log_masses,
            actual.detected.log_masses):
        maximum['translation'] = max(maximum['translation'], float(np.max(np.abs(
            ca.translation.mean_at_reference-cb.translation.mean_at_reference))))
        maximum['covariance'] = max(maximum['covariance'], float(np.max(np.abs(
            ca.translation.residual_covariance-cb.translation.residual_covariance))))
        maximum['quaternion_second_moment'] = max(maximum['quaternion_second_moment'],
            float(np.max(np.abs(bingham_second_moment(ca.orientation.parameter_matrix())
                                - bingham_second_moment(cb.orientation.parameter_matrix())))))
        maximum['log_weight'] = max(maximum['log_weight'], abs(la-lb))
    return dict(structural_checks=checks, maximum_absolute_differences=maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from probtf_observation_fusion_demo.marker_board import fuse_marker_board
    fixture = json.loads(args.fixture.read_text())
    records = []
    for record in fixture['records']:
        detections = load_recorded_detections(record)
        results, timings = {}, {}
        # Alternating order reduces warm-up/order bias. No timing threshold is
        # used as a correctness criterion or a runtime update gate.
        for backend in ('python', 'native'):
            results[backend] = fuse_marker_board(detections, fixture['settings'],
                arithmetic_backend=backend)
            timings[backend] = []
        for iteration in range(args.repetitions):
            order = ('python', 'native') if iteration % 2 == 0 else ('native', 'python')
            for backend in order:
                start = time.perf_counter()
                fuse_marker_board(detections, fixture['settings'], arithmetic_backend=backend)
                timings[backend].append(time.perf_counter()-start)
        medians = {key: float(np.median(values)) for key, values in timings.items()}
        records.append(dict(observation_id=record['observation_id'], case=record['case'],
            timing_seconds=timings, median_seconds=medians,
            board_only_speedup=medians['python']/medians['native'],
            arithmetic_sources={key: value.record['arithmetic_backend_source']
                                for key, value in results.items()},
            comparison=numerical_comparison(results['python'], results['native'])))
    output = dict(schema='native_marker_board_same_law_replay_v1', records=records,
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        scope='board fusion only, all native modes; detector and image Hessian estimator excluded',
        monotonic_clock='time.perf_counter', monte_carlo_used=False, core_files_modified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
