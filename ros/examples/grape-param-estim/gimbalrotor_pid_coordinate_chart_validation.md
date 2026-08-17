# Gimbalrotor PID sensitivity — coordinate-chart comparison

This experiment asks whether a substantial part of the finite-sigma curvature
seen in the PID postprocess is caused by the estimator chart itself.

It does **not** change the estimator, refit any bag, add a KKT constraint, or
assume an algebraically known ridge.  It is a downstream reparameterization
experiment.

The implementation compares all three current production bags:

```text
failure1
failure2
success
```

and does not special-case `failure2`.

---

## 1. Coordinate charts

### Existing chart: `estimator_quotient`

The existing sensitivity tool perturbs the 13-D common-scale quotient of the
14-D SI estimator chart,

```math
c = \hat c + B z.
```

The estimator covariance is already stored in this quotient coordinate.

### New chart: `centered_scale_free_spd`

Define the scale-free second moment

```math
K
=
\frac{\Sigma}{m}
=
\frac{1}{2}\operatorname{tr}\!\left(\frac{J}{m}\right)I
-
\frac{J}{m}.
```

At the fitted plant `(\hat K, \hat c, \hat g)` with
`g_i = f_i/m`, use

```math
U
=
\log\!\left(
\hat K^{-1/2} K \hat K^{-1/2}
\right),
```

```math
v = c - \hat c,
```

```math
w_i = \log(g_i / \hat g_i).
```

The 13-D coordinate is

```text
6 symmetric components of U
3 CoG offsets
4 log force-over-mass ratios
```

and the fitted plant is exactly the origin.

The inverse map is

```math
K
=
\hat K^{1/2}\exp(U)\hat K^{1/2},
```

```math
J/m = \operatorname{tr}(K)I-K,
```

```math
c = \hat c + v,
\qquad
g_i = \hat g_i e^{w_i}.
```

This removes the common-scale gauge before the finite excursion and keeps the
scale-free inertia representation inside the physical second-moment SPD cone.

---

## 2. Covariance transformation

No new uncertainty model is fitted.

Let `z` be the existing quotient coordinate and `y` the centered scale-free SPD
coordinate.  The implementation computes the analytic Jacobian

```math
T = \left.\frac{\partial y}{\partial z}\right|_{\hat x}
```

and uses

```math
C_y = T C_z T^\top.
```

At the center the SPD logarithm is differentiated at the identity, so the SPD
block is especially simple:

```math
dU
=
\hat K^{-1/2}\,dK\,\hat K^{-1/2}.
```

The test suite checks this analytic push-forward against a centered finite
difference through the actual `SiParameterChart`.

The infinitesimal propagated PID sensitivity should therefore agree between
the two charts up to numerical differentiation error.  A material difference
there is an implementation problem, not evidence that one chart is better.

---

## 3. Local derivative and finite excursion are separated

The old report obtained its `linearized_one_sigma` from the same finite
`k sigma` points used for the envelope.  That quantity therefore changed when
`k` changed in a nonlinear region.

The new implementation separates:

```text
infinitesimal 1-sigma
```

from

```text
finite secant 1-sigma at k = 0.5, 1, 2
```

The infinitesimal derivative uses a dedicated small fraction of one covariance
sigma.  Default:

```text
--derivative-sigma-fraction 1e-5
```

The originally proposed `1e-3` fraction was not infinitesimal for the current
`failure2` push-forward covariance: its largest centered-chart sigma is about
482, so that fraction still moves about 0.48 in log-SPD coordinates.  The
`1e-5` default reduces the worst observed chart-to-chart local-sigma mismatch
from about 2% to below 0.001% while remaining above the roundoff-dominated
regime observed at still smaller steps.

For a group `g`, the report includes

```math
\sigma_{g,\mathrm{local}}
```

and

```math
\sigma_{g,\mathrm{finite}}(k).
```

The main straightening diagnostic is

```math
R_g(k)
=
\frac{
\sigma_{g,\mathrm{finite}}(k)
}{
\sigma_{g,\mathrm{local}}
}.
```

A chart that makes the relevant map more nearly linear over the selected
finite excursion should keep `R_g(k)` closer to one as `k` increases.

Do not choose a chart from one scalar alone.  Also inspect the full
sigma-point envelope and the sampled physical plant.

---

## 4. No source-rank early termination

For sensitivity analysis, sampled `A_real` rank and condition number are
diagnostics.

The previous check

```python
if real.source_threshold_rank < 6:
    raise ...
```

is removed.

`A_real` is not pseudoinverted in

```math
H = A_\mathrm{real} A_\mathrm{cmd}^{+},
```

so loss of the controller source-threshold rank is not a reason to discard a
finite sample.

Likewise, a large or infinite reported condition number is retained.

A finite negative least-squares group scale is also retained as a diagnostic.

A sample is marked invalid only when the requested floating-point calculation
actually becomes non-finite or mathematically undefined, for example:

```text
matrix solve failure
matrix exponential overflow
exact zero denominator in the group-scale quotient
non-finite downstream effectiveness
```

Programming/input-contract errors are not converted into benign sample
invalidity.

---

## 5. Production three-bag command

From

```bash
cd /home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim
```

run:

```bash
SOURCE_COMMIT=$(git rev-parse HEAD)
OUT=minimal/outputs/${SOURCE_COMMIT}/gimbalrotor_pid_coordinate_chart_comparison

python3 minimal/three_bag_gimbalrotor_pid_coordinate_chart_comparison.py \
  --input failure1=minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json \
  --input failure2=minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json \
  --input success=minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json \
  --vehicle-model minimal/grape_vehicle_model.json \
  --output-dir "${OUT}"
```

Defaults:

```text
covariance modes:
  conservative_fusion
  overlap_corrected

coordinate modes:
  estimator_quotient
  centered_scale_free_spd

finite sigma:
  0.5
  1.0
  2.0

Monte Carlo:
  disabled
```

This produces the individual reports and:

```text
coordinate_chart_comparison.json
coordinate_chart_comparison.md
```

The output tree is organized by

```text
covariance / bag / coordinate / sigma
```

so every number remains traceable.

If desired, add:

```bash
--monte-carlo-samples 512 --seed 0
```

Monte Carlo is evaluated once per bag / covariance / coordinate chart and is
reused in the three finite-sigma reports.

---

## 6. Tests to run

At minimum:

```bash
PYTHONPATH=minimal/tests python3 -m unittest \
  minimal/tests/test_gimbalrotor_pid_postprocess_sensitivity.py \
  minimal/tests/test_gimbalrotor_pid_coordinate_chart.py
```

If the repository normally runs the complete minimal test suite, run that as
well.

The new tests check:

```text
centered SPD chart round trip
centered chart origin = fitted scale-free plant
analytic covariance push-forward Jacobian vs finite difference
first-order PID sensitivity invariance between charts
covariance push-forward equality
rank-deficient A_real is not rejected early
negative finite group scale is retained
```

---

## 7. What to inspect in the results

### First: infinitesimal coordinate invariance

For every bag and group, compare:

```text
local_one_sigma_estimator_quotient
local_one_sigma_centered_scale_free_spd
```

They should agree closely.

If they do not, inspect the analytic push-forward Jacobian or the derivative
step before interpreting finite-sigma results.

### Second: finite/local ratio

For every bag and PID group, inspect `R_g(k)` at:

```text
0.5 sigma
1.0 sigma
2.0 sigma
```

The new chart is useful if it systematically keeps the ratio nearer one and
reduces asymmetric envelope growth without merely hiding a genuine physical
variation.

The desired result is not restricted to `failure2`; inspect all three bags.

### Third: physical excursion

For directions with large changes, inspect:

```text
scale_free_vector
A_real_condition_number
H_dimensionless_diagonal
gain-group scales
```

The centered chart can reduce coordinate-induced curvature, but it should not
be used to erase a real nonlinear dependence of PID effectiveness on the plant.

### Fourth: sample validity

The old `rank deficient under source threshold` invalid samples should
disappear.

If a sample remains invalid, its exception should correspond to an actual
floating-point or mathematical failure.

---

## 8. Interpretation

There are three useful outcomes.

### The centered chart improves all or most bags

Then a meaningful fraction of the previous finite-sigma curvature was chart
curvature.  The centered scale-free SPD coordinate is a good candidate for
future downstream sensitivity analysis.

### It mainly improves `failure2`

That is still informative, but do not redesign the estimator solely to make one
bag look regular.  Keep both chart reports and continue to distinguish
bag-specific weak identifiability from general parameterization geometry.

### It does not materially improve the finite-sigma behavior

Then the curved structure is likely in the actual objective/output map rather
than primarily in the coordinate representation.  At that point, a profile
ridge calculation or another explicit ridge-tracing method is more appropriate
than adding more coordinate machinery.

No KKT constraint is introduced by this patch.
