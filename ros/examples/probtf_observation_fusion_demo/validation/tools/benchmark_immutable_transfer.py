#!/usr/bin/env python3
"""Measure transfer of the same archived body-result laws, without inference.

The existing pickle archive is read once outside timing. Each implementation
then encodes that identical candidate with its own reducer and measures decode
plus ndarray-readonly restoration. No raw archive is rewritten or copied into
the repository. Native arithmetic and sensor/K updates are not invoked.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np

from probtf_observation_fusion_demo.async_artifacts import capture_candidate_identity, identity_hash
from probtf_observation_fusion_demo.async_workers import _freeze_arrays
from probtf_observation_fusion_demo import immutable_body_cache


def measurement(values):
    return dict(median_s=float(np.median(values)), minimum_s=float(min(values)),
                maximum_s=float(max(values)), repetitions=len(values))


def main(args):
    records = []
    for path in args.input_job:
        original = path.read_bytes()
        job = pickle.loads(original)
        result = job['result']
        identity = capture_candidate_identity(result)
        record_json = json.dumps(result.records, sort_keys=True, separators=(',', ':'))
        started = time.perf_counter()
        encoded = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        encode_s = time.perf_counter()-started
        decode_wall, decode_cpu, freeze_wall, freeze_cpu = [], [], [], []
        for _ in range(args.repetitions):
            # Remove earlier replicas outside measurement; never disable GC.
            gc.collect()
            wall, cpu = time.perf_counter(), time.process_time()
            restored = pickle.loads(encoded)
            decode_wall.append(time.perf_counter()-wall)
            decode_cpu.append(time.process_time()-cpu)
            wall, cpu = time.perf_counter(), time.process_time()
            _freeze_arrays(restored)
            freeze_wall.append(time.perf_counter()-wall)
            freeze_cpu.append(time.process_time()-cpu)
            assert capture_candidate_identity(restored) == identity
            assert json.dumps(restored.records, sort_keys=True, separators=(',', ':')) == record_json
            assert not restored.prior.covariance.flags.writeable
            del restored
        records.append(dict(input_path=str(path.resolve()),
            original_archive_sha256=hashlib.sha256(original).hexdigest(),
            candidate_identity_sha256=identity_hash(identity),
            record_json_sha256=hashlib.sha256(record_json.encode()).hexdigest(),
            encoded_payload_bytes=len(encoded), encode_s=encode_s,
            decode_wall=measurement(decode_wall), decode_cpu=measurement(decode_cpu),
            ndarray_restore_wall=measurement(freeze_wall), ndarray_restore_cpu=measurement(freeze_cpu)))
    module_path = Path(immutable_body_cache.__file__)
    args.output.write_text(json.dumps(dict(schema='immutable_body_transfer_benchmark_v1',
        phase=args.phase, git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        immutable_module_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
        algorithm_changed=False, original_archive_unchanged=True, gc_disabled=False,
        measurement_scope='Single-process auxiliary transfer microbenchmark; original-file read, original decode, GC cleanup and identity/JSON checks excluded from reported decode/restore timings. The two phases may have different pickle representation, while the full source and law/record identities must match.',
        results=records), indent=2, allow_nan=False)+'\n')
    print(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-job', type=Path, nargs='+', required=True)
    parser.add_argument('--repetitions', type=int, default=7)
    parser.add_argument('--phase', choices=('before', 'after'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
