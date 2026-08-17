# ProbTF dependency-aware Gaussian smoother implementation

Status: first core milestone implemented

Date: 2026-08-18

This document records the implementation of
`PROBTF_DEPENDENCY_SMOOTHER_PLAN.md`.  The implementation is deliberately an
in-process, dense reference backend.  The sparse/incremental Phase 3 backend
remains a later optimization behind the same public contracts.

## Implemented contracts

- Python and C++ own versioned Gaussian latent factors, immutable snapshots,
  edge sensitivities, and path-local cross-factor covariance.
- The fixed pose chart is
  `translation_parent_rotation_right_local`, ordered as translation followed
  by a right-local rotation vector.
- Transform covariance is evaluated without sampling by aggregating physical
  edge residuals and latent sensitivities before covariance multiplication.
- Repeated dependency IDs resolve only when every participating edge has a
  matching factor version and binding.  Missing, stale, or mismatched data
  continues to fail closed.
- The local transform evaluator supports stochastic inverse traversal using an
  analytic mixed-chart Jacobian.
- Dense Joseph-form observation updates preserve cross covariance, update all
  correlated factors atomically, and increment their versions together.
- Python and C++ transform-moment results include factor versions,
  approximation metadata, provenance, and diagnostics.
- Python and C++ query caches are keyed by latent-store revision.  A posterior
  update is therefore visible on the next query without manual invalidation.
- Native concentrated components are reconstructed through the existing
  `component_from_pose_covariance()` path.  Uniform, ridge-like, and mixture
  global states are not silently projected into a local Gaussian chart.

## Primary APIs

Python:

```python
from probtf.dependency import (
    EdgeLatentBinding,
    GaussianLatentStore,
    GaussianObservationFactor,
)

store = GaussianLatentStore()
factor = store.put_factor("joint_bias", mean, covariance, stamp)
store.bind_edge(EdgeLatentBinding(..., factor_version=factor.version, ...))
graph = ProbTfGraph(latent_store=store)
moments = graph.lookup_transform_moments(target, source, query_stamp)
store.apply_observation(GaussianObservationFactor(...))
updated = graph.lookup_transform_moments(target, source, query_stamp)
```

C++:

```cpp
auto store = std::make_shared<probtf_core::GaussianLatentStore>();
store->putFactor(...);
store->bindEdge(...);
probtf_core::LatestSnapshot snapshot(dynamic_records, static_records, store);
probtf_core::TransformMomentObservation result;
snapshot.lookupTransformMoments(target, source, &result, &error);
store->applyObservation(...);
snapshot.lookupTransformMoments(target, source, &result, &error);
```

## Validation

The implementation is covered by the full Python suite and the catkin C++
suite, including finite-difference inverse checks, a Monte Carlo sibling-path
oracle, native covariance round-trip, a six-joint correlated chain, atomic
multi-factor conditioning, fail-closed dependency resolution, immediate cache
invalidation, finite-Bingham local moments, and Python/C++ metadata parity.

The Python command below assumes pytest 7.4.4 is available to the selected
interpreter.  On the measurement host it was supplied through an external
test-only site-packages directory; it is not a ProbTF runtime dependency.

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=ros/core/probtf_core/src \
python3 -m pytest -p no:cacheprovider tests/probtf -q

catkin build probtf_core --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin run_tests probtf_core
```

Observed result on 2026-08-18: 232 Python tests and 50 catkin tests passed,
with no build warnings.

## Performance

Run the Release benchmark with:

```bash
/home/leus/catkin_ws/devel/.private/probtf_core/lib/probtf_core/dependency_smoother_benchmark
```

The benchmark covers every requested path length, factor count, and factor
dimension, with one concentrated finite-Bingham edge.  Each uncached sample
uses a fresh `LatestSnapshot`; cached samples repeat the same frame pair and
unchanged latent revision.

| Case | Uncached p50/p95 | Cached p50/p95 |
|---|---:|---:|
| myCobot reference: path 8, 1 factor, dim 12 | 31.433 / 32.100 us | 0.240 / 0.241 us |
| path 8, 4 factors, dim 24 (96 path-local dimensions) | 170.394 / 179.270 us | 0.251 / 0.261 us |
| path 8, 4 factors, dim 48 (192 path-local dimensions) | 929.362 / 976.885 us | 0.260 / 0.271 us |
| largest case: path 32, 4 factors, dim 48 | 1169.640 / 1224.960 us | 0.561 / 0.571 us |

The myCobot reference is comfortably faster than a control/visualization
period even on an uncached query.  Dense uncached cost steepens between 96 and
192 total path-local dimensions, so 192 dimensions is the first measured scale
where a sparse backend is clearly worth evaluating.  Cached queries remain
below 0.6 us in all measured cases.

The complete machine-readable result is in
`PROBTF_DEPENDENCY_SMOOTHER_BENCHMARK_2026-08-18.json`.
