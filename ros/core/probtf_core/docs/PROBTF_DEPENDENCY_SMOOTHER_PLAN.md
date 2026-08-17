# ProbTF dependency-aware Gaussian smoothing plan

Status: implementation plan, not an implementation

Date: 2026-08-18

Target package: `ros/core/probtf_core`

## 1. Purpose

ProbTF already represents a stochastic spatial edge with a global Bingham
orientation, conditional Gaussian translation, rotation--translation coupling,
mixtures, provenance, and explicit dependency IDs.  It also already has two
important pieces needed for correlated local uncertainty:

- local `6 x 6` pose covariance extraction/encoding in
  `src/probtf/temporal/backends.py` (`component_pose_covariance()` and
  `component_from_pose_covariance()`), and
- dependency detection in `src/probtf/kernels/*`; a stochastic path containing
  a repeated latent dependency currently fails closed with
  `DEPENDENCY_UNRESOLVED` instead of silently assuming independence.

The missing capability is numerical resolution of those dependencies.  The
first target is a set of correlated, locally Gaussian latent variables such as
joint zero offsets, hand-eye calibration errors, compliance parameters, or
other persistent small errors.  A spatial observation must be able to update
one joint posterior over those variables, after which every ProbTF query that
depends on them must immediately use the updated posterior.

The runtime path must remain matrix-based.  Sampling may be retained as an
offline validation oracle or an explicitly requested representation, but it is
not the propagation mechanism described here.

## 2. Mathematical contract

Use the same local pose convention already used by the temporal moment backend:

- translation perturbation expressed in the parent frame,
- rotation perturbation represented as a right-local rotation vector,
- pose perturbation ordered as `[translation(3), rotation(3)]`.

For a Gaussian latent factor `f`, let

```text
eta_f ~ N(mu_f, Sigma_f),      eta_f in R^{d_f}.
```

A physical edge `e` may depend on one or more factors.  Around the current
linearization point,

```text
delta_xi_e = sum_f G_{e,f} delta_eta_f + epsilon_e,
epsilon_e ~ N(0, Q_e).
```

`Q_e` is edge-local residual uncertainty.  `G_{e,f}` is a `6 x d_f`
sensitivity in the existing mixed pose convention.

For a path query, linearize the requested output `y` with respect to each edge
perturbation:

```text
delta_y = sum_e J_e delta_xi_e.
```

Collecting terms by latent factor gives

```text
A_f = sum_e J_e G_{e,f}.
```

The output covariance is then

```text
Sigma_y = sum_e J_e Q_e J_e^T
        + sum_f A_f Sigma_f A_f^T,
```

when factors are independent of each other.  If several logical factors are
jointly correlated, they must be represented by one joint Gaussian factor or by
a joint information system; do not drop cross-factor covariance.

This equation is the central dependency-aware replacement for the current
independent-edge accumulation.

## 3. State ownership

Do not put an ever-growing robot-wide dense covariance into every
`ProbabilisticTransformStamped` message.

Introduce a latent-state store owned by the ProbTF runtime/buffer.  Edges refer
to factors by stable IDs and carry only the sensitivity needed to map factor
perturbations into that edge.

Initial Python-side contracts should be conceptually equivalent to:

```python
@dataclass(frozen=True)
class GaussianLatentFactor:
    factor_id: str
    mean: np.ndarray          # (d,)
    covariance: np.ndarray    # (d,d)
    stamp: float
    version: int
    provenance: Provenance

@dataclass(frozen=True)
class EdgeLatentBinding:
    edge_id: str
    factor_id: str
    sensitivity: np.ndarray   # (6,d)
    linearization_stamp: float
    linearization_pose: DeterministicTransform
```

Names are provisional; semantics are not.

A factor version must change whenever its mean/covariance changes.  Query caches
must be keyed by factor versions so stale marginals cannot survive an update.

## 4. Phase 1: dependency-aware spatial moment evaluation

Implement this before adding observation updates.

### 4.1 Python latent store

Add a small module under a new package such as:

```text
src/probtf/dependency/
    __init__.py
    gaussian.py
    binding.py
    store.py
```

Required operations:

- insert/replace a Gaussian factor by `factor_id`,
- obtain an immutable snapshot of factors and versions,
- bind an edge to a factor with a `6 x d` sensitivity,
- obtain all bindings for a path,
- validate dimensions, PSD covariance, finite values, frame/perturbation
  convention, and linearization metadata.

The first implementation may use dense NumPy matrices inside each factor.  Do
not build one global dense covariance merely to answer a query.

### 4.2 Extend the kernel dependency contract

Current files:

```text
src/probtf/kernels/forward.py
src/probtf/kernels/inverse.py
src/probtf/kernels/composed.py
src/probtf/kernels/evaluation.py
```

keep their existing dependency-ID behavior.

Add a dependency-aware moment evaluator.  A repeated dependency is valid when:

1. every repeated ID resolves to a Gaussian latent factor,
2. every participating stochastic edge has a valid sensitivity binding, and
3. all bindings use the same factor version and perturbation convention.

If any of these fail, preserve the current fail-closed behavior and return
`DEPENDENCY_UNRESOLVED`.  Never fall back silently to independence.

For an independent path with no bindings, results must remain numerically
identical to the current evaluator.

### 4.3 Add transform-level moments

The C++ fast API currently exposes point moments through
`LatestSnapshot::lookupPointMoments()`.  Add a transform-level result whose
minimum contract is:

```cpp
struct TransformMoments {
  DeterministicTransformLike mean;
  Eigen::Matrix<double, 6, 6> covariance;
};
```

and a query analogous to:

```cpp
bool lookupTransformMoments(target_frame,
                            source_frame,
                            TransformMomentObservation*,
                            ...);
```

Python should expose the same semantic result.

Do not define a new perturbation convention.  Reuse the one already implemented
by `component_pose_covariance()` in `temporal/backends.py`.

### 4.4 Stochastic inverse

The current C++ point-moment path rejects a stochastic inverse.  Implement the
local-Gaussian inverse Jacobian in the transform-moment evaluator so a query can
cross two stochastic branches, as required by two-robot camera-to-camera
queries.

Tests must compare the analytic/implemented inverse Jacobian with finite
differences in the existing perturbation convention.

### 4.5 Native distribution reconstruction

Where a concentrated local transform marginal is requested as a native ProbTF
component, reuse:

```text
component_from_pose_covariance()
```

rather than creating a second Gaussian-to-Bingham conversion.

This reconstruction is explicitly a local/tangent approximation.  Preserve
`ApproximationInfo` and provenance.

Do not attempt to convert a uniform or ridge-like global Bingham to a finite
Gaussian merely to use this evaluator.

## 5. Phase 2: Gaussian observation factors and backward update

Once phase 1 can propagate a correlated latent posterior forward, add
conditioning.

For a local observation residual

```text
r(x) ~= r0 + H delta_eta,
noise ~ N(0, R),
```

support an update of the joint latent posterior.  The first implementation may
use either covariance or information form.  Information form is preferred for
the public factor abstraction:

```text
Lambda+ = Lambda + H^T R^-1 H
eta_info+ = eta_info + H^T R^-1 z_linearized.
```

For small dense factors, a Joseph-form covariance update is acceptable as the
reference implementation and is useful for tests.

### 5.1 Observation-factor API

Introduce a solver-facing object conceptually equivalent to:

```python
@dataclass(frozen=True)
class GaussianObservationFactor:
    observation_id: str
    latent_factor_ids: tuple[str, ...]
    residual: np.ndarray
    jacobian_blocks: tuple[np.ndarray, ...]
    noise_covariance: np.ndarray
    stamp: float
    provenance: Provenance
```

The production API should not require an estimator to expose its internal
algorithm.  It only provides the spatial residual/Jacobian/noise relation to the
latent variables it owns or references.

### 5.2 Atomic posterior update

An update touching multiple factors must be atomic:

- read one consistent prior snapshot,
- linearize against that snapshot,
- solve/update,
- increment all affected versions together,
- invalidate dependent query caches together.

A reader must never see half of a loop-closure update.

### 5.3 Persistent and instantaneous uncertainty

Separate persistent latent uncertainty from instantaneous measurement noise.
Good latent examples are:

- joint zero offsets,
- hand-eye extrinsics,
- compliance/deflection parameters,
- slowly varying calibration parameters.

Encoder white noise or image noise should normally remain residual/process
noise, not a persistent latent variable that is permanently "learned away".

## 6. Phase 3: smoothing graph / incremental solver

The first two-myCobot implementation has only about twelve joint-bias variables
plus a few calibration variables.  A dense factor is therefore small enough to
validate semantics first.

After the API and tests are stable, add sparse/incremental solving.  The desired
architecture is factor-graph-like:

```text
latent variables <---- observation factors
      |
      +---- edge sensitivities ----> ProbTF spatial graph
                                      |
                                      +---- transform queries
```

Important requirement: the spatial graph and the smoothing graph are related
but are not the same graph.

- The spatial graph keeps TF semantics: frames and physical transforms.
- The smoothing graph keeps statistical dependencies: variables and factors.

Do not force one graph topology to encode both concepts.

A sparse Cholesky/QR or Bayes-tree-style incremental backend can be added later.
The public ProbTF contract should not depend on a particular solver.

## 7. Hybrid global Bingham + local Gaussian state

The Gaussian smoother is not a replacement for ProbTF's global Bingham state.

Examples:

- unknown base yaw / one-direction ambiguity: keep the Bingham globally,
- joint zero offsets of roughly one degree: keep a local Gaussian latent factor,
- hand-eye residual around a calibrated pose: local Gaussian,
- discrete planar-tag alternatives: retain mixture components.

For vector-alignment evidence, Bingham natural parameters may continue to update
by addition exactly as today.

Do not create a tangent Gaussian for `UNIFORM` or a genuine `S1` Bingham ridge
solely to enter the Gaussian smoother.  The first hybrid milestone should allow
an edge/query to depend simultaneously on:

1. a native global Bingham spatial variable, and
2. one or more local Gaussian latent factors.

The output can remain a native ProbTF distribution when closure is available,
or an explicitly diagnosed moment summary when it is not.

A later research extension may support observation factors that jointly update
a broad Bingham variable and Gaussian latents.  That is not required for the
first dependency-aware Gaussian core milestone.

## 8. ROS/message layer

Do not add large covariance blocks directly to
`ProbabilisticTransformStamped` as the first design.

Preferred sequence:

1. implement the latent store and bindings as in-process core APIs,
2. validate semantics and performance,
3. only then add ROS messages/services if separate processes must own/update
   latent factors.

If wire support is required, use separate messages such as a latent-factor
array and edge-binding array.  A transform edge should reference stable factor
IDs; it should not duplicate the complete factor covariance on every publish.

The wire contract must include factor version/stamp so consumers can detect
inconsistent snapshots.

## 9. C++ runtime

Mirror the Python semantics after the Python reference tests pass.

Suggested additions under:

```text
include/probtf_core/
src/
```

are conceptually:

```text
gaussian_latent_store.hpp/.cpp
transform_moments.hpp/.cpp
dependency_aware_moments.hpp/.cpp
```

Use Eigen fixed-size matrices for `6 x 6` pose operations and dynamic matrices
only for latent dimensions.

For a path, aggregate sensitivities by factor before multiplying covariance:

```text
A_f = sum_e J_e G_{e,f}
Sigma += A_f Sigma_f A_f^T
```

Do not expand a full path-wide block covariance unless needed for debugging.

## 10. Complexity and caching

For the two-myCobot case, a single joint factor has `d = 12`.

- covariance storage: `12 x 12 = 144` doubles,
- one output sensitivity: `6 x 12`,
- one `A Sigma A^T` evaluation is negligible compared with Bingham moment
  integration.

The scalability rule is **path-local factor aggregation**.

Do not maintain one dense covariance over every uncertain edge in the robot.
Queries should touch only factors referenced by their path.

Cache candidates:

- Bingham second/fourth moments keyed by immutable orientation parameter,
- edge pose Jacobian/sensitivity keyed by edge record + factor version,
- path factor aggregation keyed by path signature + edge versions + factor
  versions.

Any posterior update must invalidate caches through version changes rather than
manual ad-hoc clearing.

## 11. Required correctness tests

Add Python tests first, then C++ parity tests.

### 11.1 Independence regression

With no shared latent factors, dependency-aware evaluation must match the
current moment evaluator to numerical tolerance.

### 11.2 Two-variable correlation toy problem

Use

```text
b1, b2 ~ independent N(0, sigma^2)
z = b1 + b2 + noise.
```

In the low-noise limit, the posterior must approach

```text
Sigma+ = sigma^2/2 [[ 1, -1],
                     [-1,  1]].
```

The propagated variance of `b1 + b2` must approach zero while the orthogonal
combination remains uncertain.  This test catches implementations that retain
only marginal variances and lose cross covariance.

### 11.3 Repeated dependency resolution

A path that currently produces `DEPENDENCY_UNRESOLVED` must:

- succeed when a complete matching factor/binding set is supplied,
- continue to fail closed when the factor or any binding is missing,
- fail on version/convention mismatch.

### 11.4 Forward/inverse round trip

Compare transform mean/covariance for paths containing stochastic forward and
inverse steps against finite-difference linearization and an offline Monte
Carlo oracle.

Monte Carlo is a test oracle only; it is not the runtime implementation.

### 11.5 Native round trip

For concentrated local distributions:

```text
component -> 6x6 covariance -> component
```

must preserve the local mean/covariance within documented approximation error.

### 11.6 Joint-chain test

Construct a six-revolute-joint synthetic arm with a correlated joint-bias
factor.  Compare

```text
J Sigma_joint J^T
```

with the dependency-aware ProbTF camera marginal.

### 11.7 Observation-update test

Add two camera/landmark factors from different arm configurations.  Verify that
the second configuration reduces a null direction left by the first and that
the updated covariance is immediately visible in subsequent transform queries.

### 11.8 Python/C++ parity

Use frozen fixtures for:

- mean transform,
- `6 x 6` covariance,
- factor IDs/versions used by the query,
- diagnostics/approximation metadata.

### 11.9 Performance benchmark

Benchmark at least:

- path lengths 4, 8, 16, 32,
- zero, one, and four Gaussian latent factors,
- latent dimensions 6, 12, 24, 48,
- concentrated finite Bingham on one global edge,
- repeated queries with unchanged versions to measure cache benefit.

Record p50/p95 query time in C++.

The acceptance target for the myCobot flagship is comfortably above its
control/visualization rate; the benchmark should also identify the dimension at
which a sparse backend becomes worthwhile.

## 12. Demo migration after core support lands

The accompanying `furisake_joint_uncertainty.launch` patch intentionally keeps
a temporary `12 x 12` Gaussian inside the demo node and publishes only the
resulting stochastic camera-edge marginals.

After this core plan is implemented, remove that workaround in this order:

1. create one core Gaussian factor containing the twelve joint zero offsets,
2. publish/bind the six physical joint edges of Tou and the six of Kasuga to
   the appropriate factor columns,
3. replace demo-side `J Sigma J^T` with a core
   `lookupTransformMoments(base, camera)` query,
4. convert common landmark observations into core observation factors,
5. let the core posterior update the joint factor atomically,
6. query the camera/handover distribution again without changing task code,
7. retain the demo-local implementation only as a regression oracle and then
   delete it once parity is proven.

At that point the demo should be changed so that no aggregate stochastic
`base -> tool` edge is fabricated by the demo.  The camera uncertainty must be
the result of traversing the real probabilistic joint chain.

## 13. Acceptance criteria

The core work is complete for the first milestone when all of the following are
true:

- both Tou and Kasuga may have stochastic per-joint edges,
- several edges may reference the same Gaussian latent factor,
- a transform query across those edges resolves the dependency without
  sampling,
- the same query still fails closed when dependency data is unavailable,
- stochastic inverse is supported in the local-moment path,
- a spatial observation can update a joint Gaussian posterior that touches
  several edges,
- the next transform query uses that updated posterior automatically,
- cross covariance is preserved (verified by the two-variable toy test),
- global Bingham states remain global and are not forcibly Gaussianized,
- Python and C++ results agree on frozen fixtures,
- C++ performance is measured and sufficient for the two-myCobot demo.

This milestone is intentionally narrower than a general non-Gaussian factor
solver.  It provides the missing bridge between ProbTF's existing global
orientation representation and the correlated local uncertainty needed by
real kinematic chains and calibration feedback.
