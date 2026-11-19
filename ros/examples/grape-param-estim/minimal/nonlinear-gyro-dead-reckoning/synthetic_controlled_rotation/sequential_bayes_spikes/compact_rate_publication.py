#!/usr/bin/env python3
"""Keep every saved moment, retaining only the final redundant node weights.

All earlier weights are reconstructible from the saved original observations,
proposal parameters, quadrature order and physical-flow code. This selects
publication fields after inference; it does not change a candidate law.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def compact(output):
    records = {}
    summary_path = output/'summary.json'
    summary = json.loads(summary_path.read_text())
    order = summary['config']['quadrature_orders'][-1]
    refresh_diagnostics = {}
    for mode in ('nonlinear', 'linear_forced'):
        path = output/f'bayes_{mode}.npz'
        arrays = dict(np.load(path, allow_pickle=False))
        removed = ('refresh_log_weights', 'refresh_prior_log_weights')
        if all(key not in arrays for key in removed):
            continue
        before_bytes = path.stat().st_size
        before_hash = sha(path)
        retained = {key: array_sha(value) for key, value in arrays.items() if key not in removed}
        final_weights = arrays.pop(removed[0])[-1].copy()
        final_prior_weights = arrays.pop(removed[1])[-1].copy()
        arrays.update(last_refresh_log_weights=final_weights,
                      last_refresh_prior_log_weights=final_prior_weights)
        np.savez_compressed(path, **arrays)
        saved = dict(np.load(path, allow_pickle=False))
        unchanged = all(array_sha(saved[key])==digest for key, digest in retained.items())
        if not unchanged:
            raise RuntimeError('Publication field selection changed a retained array')
        records[mode] = dict(before_bytes=before_bytes, after_bytes=path.stat().st_size,
            before_sha256=before_hash, after_sha256=sha(path),
            every_retained_array_unchanged=True, retained_array_sha256=retained,
            removed_redundant_fields=list(removed),
            final_weights_preserved=True,
            reconstruction='All earlier node weights: original prior times acquired prefix likelihoods divided by saved proposal; deterministic GH nodes from saved refresh mean/covariance/order',
            inference_rerun=False, covariance_or_mean_changed=False)
        diagnostic_file = output/'cache'/f'order{order}_{mode}_refresh.json'
        if diagnostic_file.exists():
            refresh_diagnostics[mode] = json.loads(diagnostic_file.read_text())
    if records:
        value = dict(schema='grape/selected-rate-publication-storage/v1', records=records,
            compact_script_sha256=sha(Path(__file__)),
            every_observation_and_every_published_joint_moment_retained=True,
            omitted_fields_are_reconstructible_numerical_cache=True)
        (output/'publication_storage.json').write_text(json.dumps(value, indent=2)+'\n')
        if refresh_diagnostics:
            (output/'proposal_refresh_diagnostics.json').write_text(
                json.dumps(refresh_diagnostics, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        summary['publication_storage'] = dict(
            report='publication_storage.json',
            all_published_moments_unchanged=True,
            final_node_weights_keys=['last_refresh_log_weights', 'last_refresh_prior_log_weights'],
            intermediate_refresh_weights_reconstructible=True,
            finite_proposal_residuals_report='proposal_refresh_diagnostics.json',
            compact_script_sha256=value['compact_script_sha256'])
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    compact(parser.parse_args().output)
