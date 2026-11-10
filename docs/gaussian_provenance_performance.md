# Gaussian provenance identifier-union performance

`GaussianLatentStore.apply_observation` retains all source and derived-edge IDs
from each factor and observation. Its private `_unique` previously searched
the growing output list once per identifier. A single accumulated union was
quadratic even when all IDs were ordinary strings.

`Provenance` already converts source/edge IDs to plain stripped strings and
checks uniqueness using a set. Gaussian factors and observations require this
existing `Provenance` type. The performance patch is confined to
`probtf/dependency/store.py::_unique`: while inputs are plain strings, a set
tracks membership and the existing list preserves first-occurrence order.

On the first non-plain-string input, the helper returns to its original
equality-only list path for that input and all later inputs. This preserves
unhashable inputs, cross-type equality, first-object identity, exceptions and
one-pass generator consumption without calling a custom object's `__hash__`.
No public type contract is narrowed. The provenance model, Gaussian equations,
source contents, ID formatting and dependency ownership are unchanged.

## Measured control

Python3.8.10, NumPy1.24.4, single-thread BLAS on this workstation; three repeats.
The input has 24,934 existing realistic-length raw-ID strings and 74 incoming
IDs, including eight duplicates, giving 25,000 unique sources. The comparison
patches **only** the identifier union in the same existing Gaussian store;
it does not implement a second Gaussian update.

| Timed operation | Original list union | String-set path | Ratio |
|---|---:|---:|---:|
| Union alone, median | 7.655 s | 10.47 ms | 731× |
| Same store's complete one-observation update, median | 7.636 s | 18.12 ms | 422× |

Posterior mean and covariance are bitwise identical, and all provenance IDs
and their order are identical. A separate actual 370-update run of the new
store accumulated all 25,000 IDs in **3.519 s**. The complete 370-update legacy
runtime was not measured, so the single-update ratio is not presented as a
full-run speedup.

This is a metadata benchmark with synthetic raw-ID strings, not a physical
force-bag replay or a sensor/ROS throughput result. Every update still copies
and validates its accumulated provenance. An individual ordinary-string union
is expected O(N); bounded history, constant per-step cost or fixed metadata
memory is not claimed. General non-string input can retain quadratic behavior.

Reproduce from the repository root with an environment containing the existing
ProbTF dependencies:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/benchmark_gaussian_provenance.py
```

Raw timings and environment: `docs/gaussian_provenance_25000.json`.
Focused tests cover ordering, general/unhashable equality, generator exception
semantics, the actual Gaussian update and existing provenance/wire composition:

```bash
python -m pytest -q tests/probtf/test_dependency_provenance.py \
  tests/probtf/test_dependency_smoother.py tests/probtf/test_ros_v2.py \
  tests/probtf/test_v2_composition.py
```
