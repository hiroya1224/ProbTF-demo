# JMAA kernel architecture: Phase A audit

Date: 2026-07-12

## Baseline

The worktree was clean before the migration.  The repository test suite passed
before any behavior change:

```text
PYTHONPATH=/tmp/probik-pytest:src:third_party/BinghamNLL/src \
  python3 -m pytest -q
120 passed, 1 warning in 1.04s
```

The base interpreter does not contain pytest.  `/tmp/probik-pytest` is the
existing local target installation used by the command above.

## Existing implementation

- `src/probtf` contains the ROS-independent models, Bingham moments, producer
  algorithms, and evidence fusion.  Its stored Bingham gauge is currently
  max-eigenvalue zero.
- `src/symaware_grasp/prob_tf` contains the prototype path expression, static
  tree, root-to-link moments, and tangent surrogate.  It has no timestamped
  buffer or lazy transform kernel.
- `ros/core/probtf_msgs` contains the v1 messages.  They must remain unchanged
  while additive v2 messages and adapters are introduced.
- `ros/core/probtf_core/src/probtf_ros` contains v1 conversions.  It is the
  existing ROS boundary and must not be imported by the new core.

## ISL search result

No exact induced-spherical-law evaluator exists in the Python source or Git
history.  `symaware_grasp.prob_tf.tangent_surrogate` is a local asymptotic
moment surrogate, not an exact density evaluator.  The JMAA manuscript in
`docs/additional_infos/jmaa_manuscript.pdf` is authoritative for the exact law,
but this migration does not transcribe that mathematical expression into new
code.  The initial exact backend therefore has to report
`UNAVAILABLE_BACKEND` explicitly.

## Scale and convention decisions

The JMAA manuscript, Section 6.1, fixes the normalization.  For trace-zero
`A`, order its eigenvalues as `lambda1 >= lambda2 >= lambda3 >= lambda4` and
define:

```text
kappa_A = lambda1 + lambda2
checked_A = A / kappa_A
inverse_concentration = 1 / kappa_A
```

This is the only shape normalization used by the new storage model.  It is not
a Frobenius normalization.  Quaternion storage is `wxyz`, `vec(R)` is
column-major, coupling is `3 x 9`, and ROS quaternion conversion remains in
`probtf_ros`.

## Compatibility risks

- The legacy tree's `lookup_path(source, target)` expresses a pose query and
  uses integer graph directions.  The new `lookup_kernel(target, source,
  stamp)` uses transform-action semantics.  A wrapper must swap the lookup
  roles and translate directions; the core path type cannot be re-exported as
  the legacy type.
- The existing `RotationMoment.kron_rot` is the operator for
  `E[R S R.T]`, not `E[vec(R) vec(R).T]`.  Coupling moments require a separate
  fixed-index helper.
- A v1 orientation-only message cannot be promoted to a complete stochastic
  transform by inventing a zero translation.  Such conversion must fail with
  a diagnostic unless translation is supplied by the caller.
- A reverse traversal must retain the original physical `edge_id`; it must not
  create an independent random transform.

## Missing safety coverage

The migration needs new tests for shape normalization, immutable arrays,
mixture weight diagnostics, coupling, forest and cycle validation, all temporal
policies, deterministic action order, repeated latent dependencies, unavailable
ISL evaluation, v2 matrix compression, and the ROS/core import boundary.
