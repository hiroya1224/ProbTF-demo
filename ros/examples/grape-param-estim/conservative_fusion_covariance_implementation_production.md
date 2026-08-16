# Conservative Fusion Covariance: Mandatory Implementation, Validation, Production, and Push Procedure

## 0. Purpose

This document specifies one narrowly scoped revision to the single-bag rigid-body parameter-estimation pipeline in:

```text
ros/examples/grape-param-estim/minimal/
```

Repository:

```text
hiroya1224/ProbTF-demo
```

Starting repository commit:

```text
9009582432d26b7e592797dde8d7b62721c23c8a
```

The immediately preceding residual-wrench source implementation commit is:

```text
197caa23da3b201bca68174c48f7f35424cf58a7
```

The goal of this revision is to add a **deliberately conservative covariance intended for later multi-bag distribution fusion**.

The central scientific requirement is:

> A single bag must not claim artificial certainty in parameter directions that the bag does not constrain well.

The later use case is to combine several bag-specific parameter distributions. Therefore each bag-specific distribution must communicate both:

```text
"this direction is well constrained"
```

and:

```text
"this direction is ambiguous"
```

without narrowing the latter merely because a convenient stochastic model treats samples as independent, centers away a systematic residual mean, or subtracts uncertainty components aggressively.

This task does **not** attempt to identify the exact generative stochastic process of the residual wrench.

This task explicitly prefers a conservative uncertainty construction over a narrowly calibrated noise model.

---

# 1. Scope restrictions

This task adds one new post-fit covariance product:

```text
conservative_fusion_covariance
```

and the diagnostics needed to validate and use it.

The following restrictions are mandatory.

> **DO NOT change the point-estimation objective.**

> **DO NOT change the point estimator.**

> **DO NOT change the SG window or degree.**

> **DO NOT change measured-gimbal handling.**

> **DO NOT change rotor-lag continuation.**

> **DO NOT change strict-ZOH cell refinement.**

> **DO NOT change the KKT common-scale gauge.**

> **DO NOT change the matrix-exponential inertia chart.**

> **DO NOT add priors or arbitrary physical bounds.**

> **DO NOT add a residual-wrench parameter or residual-wrench penalty to optimization.**

> **DO NOT add a block sandwich.**

> **DO NOT add an AR, sinusoidal, Gaussian-process, HAC-bandwidth, or other temporal-correlation model in this task.**

> **DO NOT remove or rename the existing covariance products.**

> **DO NOT remove the existing residual-wrench uncertainty implementation.**

> **DO NOT run an ablation suite for this task.**

The existing products must remain:

```text
parameter_covariance_naive
parameter_covariance_overlap_corrected
parameter_covariance_wrench_corrected
```

The new product is added alongside them:

```text
parameter_covariance_conservative_fusion
```

The existing `wrench_corrected` covariance retains its present interpretation:

```text
SG overlap
+
centered residual-wrench excess after subtracting the reference SG contribution.
```

The new `conservative_fusion` covariance has a deliberately different interpretation, defined below.

---

# 2. Motivation

The current residual-wrench implementation separates:

```math
S_w
=
\operatorname{Cov}(w)
```

from the mean:

```math
\bar w
=
E[w].
```

It then estimates an excess centered model-discrepancy covariance by subtracting the SG-predicted wrench covariance.

That construction is useful as a model-discrepancy diagnostic, and it must remain.

For distribution fusion, however, this can still be too narrow.

A systematic residual such as

```math
w_k \approx \bar w \neq 0
```

has almost zero centered covariance even though the bag is visibly inconsistent with the model in that direction.

Similarly, subtracting an estimated SG contribution can narrow the uncertainty even when the practical requirement is to avoid overstating certainty.

The new fusion covariance therefore follows a different principle:

> **Every observed post-fit residual contributes positive-semidefinite uncertainty. The residual mean is not removed. The existing SG-overlap uncertainty is retained and the residual contribution is added on top.**

This intentionally permits double counting of part of the SG contribution.

That is not an implementation bug.

It is the explicit conservative design of this covariance product.

The name `conservative_fusion_covariance` must be used so that it cannot be confused with a calibrated generative covariance estimate.

---

# 3. Existing final residual and Jacobian

At the frozen final strict-ZOH point, for each objective sample `k`, the estimator already has:

```math
r_k
=
\begin{bmatrix}
s_k^{obs}-\hat s_k\\
\alpha_k^{obs}-\hat\alpha_k
\end{bmatrix}
\in\mathbb R^6,
```

and the analytic physical-parameter Jacobian:

```math
J_k
=
\frac{\partial r_k}{\partial q}
\in\mathbb R^{6\times14}.
```

Use the **raw acceleration residual** and **raw analytic physical Jacobian**.

Do not finite-difference anything.

The optimization metric provides:

```math
W_k.
```

For the current production default:

```text
covariance_mode = identity
```

so:

```math
W_k=I_6.
```

The implementation must nevertheless remain correct for the existing covariance modes by using the already-defined `covariance.weight`.

---

# 4. The conservative residual contribution

For every time sample define the score-like parameter vector:

```math
g_k
=
J_k^\top W_k r_k
\in\mathbb R^{14}.
```

Do **not** center `r_k`.

Do **not** replace it by:

```math
r_k-\bar r.
```

Do **not** subtract a predicted SG covariance from this new residual term.

Define:

```math
M_{\mathrm{res}}
=
\sum_{k=1}^{N}g_kg_k^\top.
```

Equivalently:

```math
\boxed{
M_{\mathrm{res}}
=
\sum_{k=1}^{N}
J_k^\top W_k
r_kr_k^\top
W_kJ_k.
}
```

Every summand is positive semidefinite.

Therefore:

```math
M_{\mathrm{res}}\succeq0.
```

This term automatically retains:

- residual fluctuation;
- residual mean;
- systematic residual bias;
- any sinusoidal or structured residual amplitude at the sampled times;
- full six-dimensional acceleration-residual coupling as projected through the analytic Jacobian.

This task deliberately does not model temporal cross-covariance between different `k`.

The new conservative term is a **zero-lag, uncentered, empirical score second moment**.

---

# 5. Relationship to residual wrench

The same physical discrepancy is already represented by the nominal-mass-gauge residual wrench:

```math
w_k^{res}.
```

The exact Newton--Euler closure is:

```math
w_k^{res}=G r_k,
```

and:

```math
r_k=Bw_k^{res},
```

with:

```math
BG=I_6.
```

The new conservative covariance should be computed in acceleration-residual space because:

1. `r_k` is common-scale-gauge invariant;
2. the analytic parameter Jacobian is already defined there;
3. it avoids any artificial dependence on the displayed wrench gauge.

However, add a diagnostic cross-check showing that the same `r_k` is recovered from the nominal-mass-gauge residual wrench:

```math
r_k
\approx
B w_{k,\mathrm{nominal\ mass}}^{res}
```

to machine precision.

This confirms that the conservative fusion covariance is using the same unexplained physical motion that is visualized in `residual_wrench.pdf`.

---

# 6. Existing SG-overlap sandwich term

Preserve the current curvature:

```math
A
=
\sum_k
J_k^\top W_kJ_k.
```

Preserve the existing SG-overlap sandwich middle exactly:

```math
M_{\mathrm{SG}}
=
\sum_{k,\ell}
J_k^\top W_k
C^{SG}_{k\ell}
W_\ell J_\ell.
```

Do not modify the existing cross-time SG covariance model in this task.

Do not subtract any part of `M_SG` from the new residual term.

---

# 7. Definition of the new conservative fusion covariance

Define:

```math
\boxed{
M_{\mathrm{cons}}
=
M_{\mathrm{SG}}
+
M_{\mathrm{res}}.
}
```

Let `P` be the same exact common-scale gauge-section basis used by the existing parameter covariance implementation.

Define:

```math
A_P=P^\top AP,
```

```math
M_{\mathrm{cons},P}
=
P^\top M_{\mathrm{cons}}P.
```

Using the existing machine-precision symmetric pseudoinverse:

```math
A_P^\dagger,
```

define:

```math
\boxed{
C_q^{\mathrm{cons}}
=
P
A_P^\dagger
M_{\mathrm{cons},P}
A_P^\dagger
P^\top.
}
```

The output name is:

```text
parameter_covariance_conservative_fusion
```

The corresponding sandwich middle outputs are:

```text
parameter_sandwich_middle_residual_uncentered
parameter_sandwich_middle_conservative_fusion
```

The existing fields remain:

```text
parameter_sandwich_middle_sg
parameter_sandwich_middle_wrench
parameter_sandwich_middle_total
```

where the existing `parameter_sandwich_middle_total` continues to mean the existing SG + centered-excess-wrench construction.

Do not overload old names with the new definition.

---

# 8. Required positive-semidefinite ordering

Because:

```math
M_{\mathrm{res}}\succeq0,
```

the new covariance must satisfy, on the identified common-scale gauge section:

```math
\boxed{
C_q^{\mathrm{cons}}
-
C_q^{\mathrm{SG}}
\succeq0
}
```

up to machine roundoff.

Here:

```text
C_q^SG
```

means the existing:

```text
parameter_covariance_overlap_corrected.
```

This PSD-order monotonicity is a core acceptance condition.

The conservative covariance is not allowed to become narrower than the SG-overlap covariance in any identified direction.

Do not enforce this by arbitrary eigenvalue clipping of the final covariance.

It must follow algebraically from the construction.

If a materially negative eigenvalue appears in:

```math
C_q^{\mathrm{cons}}
-
C_q^{\mathrm{SG}},
```

treat it as an implementation error.

---

# 9. Why this is intentionally conservative

This covariance intentionally includes:

```math
M_{\mathrm{SG}}
```

and an uncentered empirical residual contribution that may itself contain SG-induced residual energy.

Therefore part of the measurement uncertainty may be represented twice.

This is deliberate.

The intended interpretation is not:

```text
"best calibrated estimate of the true sampling covariance"
```

but:

```text
"a conservative distribution for later fusion that should avoid
claiming unsupported certainty from one bag."
```

The output metadata and report must explicitly say this.

Required wording in JSON/report documentation:

```text
The conservative fusion covariance deliberately retains the existing
SG-overlap uncertainty and adds the uncentered empirical residual score
second moment without subtracting the SG contribution. It is intended
as a conservative fusion distribution, not as a calibrated generative
noise covariance.
```

---

# 10. Mean must be visibly retained

The existing residual-wrench diagnostics already save:

```text
residual_wrench_mean
residual_wrench_centered
residual_wrench_total_empirical_covariance
```

Preserve them.

Add the uncentered second moment explicitly:

```math
R_w
=
\frac1N\sum_kw_kw_k^\top.
```

Save:

```text
residual_wrench_uncentered_second_moment
```

and verify numerically:

```math
R_w
=
\frac{N-1}{N}S_w
+
\bar w\bar w^\top
```

because the existing empirical covariance uses `ddof=1`.

Also save the acceleration-space uncentered second moment:

```math
R_r
=
\frac1N\sum_kr_kr_k^\top
```

as:

```text
residual_acceleration_uncentered_second_moment
```

These are diagnostics.

The conservative parameter sandwich must still use the time-aligned per-sample score outer products:

```math
\sum_k
J_k^\top W_kr_kr_k^\top W_kJ_k,
```

not a global `R_r` substituted at every time.

This preserves the actual association between each residual and the local Jacobian.

---

# 11. Decomposition of the conservative residual contribution

For interpretation only, add a decomposition that makes the role of the mean visible.

Let:

```math
\bar r
=
\frac1N\sum_kr_k,
```

and:

```math
\tilde r_k=r_k-\bar r.
```

Compute:

```math
M_{\mathrm{centered,time-aligned}}
=
\sum_k
J_k^\top W_k
\tilde r_k\tilde r_k^\top
W_kJ_k,
```

and define:

```math
M_{\mathrm{mean/remainder}}
=
M_{\mathrm{res}}
-
M_{\mathrm{centered,time-aligned}}.
```

Because `J_k` varies with time, the second term is not simply one global rank-one matrix.

Do not force an interpretation stronger than the algebra allows.

Save:

```text
parameter_sandwich_middle_residual_centered_time_aligned
parameter_sandwich_middle_residual_mean_remainder
```

The main conservative covariance remains based on:

```text
parameter_sandwich_middle_residual_uncentered.
```

This decomposition exists only to answer:

```text
"how much did keeping the mean matter?"
```

---

# 12. Extend `ParameterCovarianceResult`

Extend the existing dataclass without removing fields.

Required existing fields:

```text
curvature
sandwich_middle
naive
overlap_corrected
gauge_basis
wrench_corrected
sandwich_middle_wrench
sandwich_middle_total
```

Add:

```text
conservative_fusion
sandwich_middle_residual_uncentered
sandwich_middle_conservative_fusion
sandwich_middle_residual_centered_time_aligned
sandwich_middle_residual_mean_remainder
```

Suggested internal field names may differ slightly, but JSON/NPZ external names in this document are mandatory.

---

# 13. Extend `parameter_covariances(...)`

The current function already accepts:

```python
additional_residual_covariance
```

for the existing centered excess-wrench covariance.

Preserve that behavior.

Extend the function so it additionally receives the final uncentered residual blocks, or directly receives the precomputed `M_res`.

Preferred interface:

```python
parameter_covariances(
    raw_parameter_jacobian,
    covariance,
    gauge_direction,
    additional_residual_covariance=...,
    uncentered_residual=final.acceleration_residual,
)
```

Inside the function use:

```python
weights = covariance.weight
```

and compute the new middle exactly as:

```python
middle_residual = np.zeros((dimension, dimension))
for k in range(count):
    g = jacobian[k].T @ weights[k] @ residual[k]
    middle_residual += np.outer(g, g)
```

This form is preferred because it makes positive-semidefiniteness explicit.

Do not flatten the residual and accidentally form one global outer product:

```python
(J.T @ r) @ (J.T @ r).T
```

That is **not** the required construction.

The required construction is the **sum of per-sample score outer products**.

---

# 14. Quotient-space output is the primary fusion representation

The common-scale gauge must not contaminate cross-bag fusion.

Let:

```math
Q\in\mathbb R^{14\times13},
```

with:

```math
Q^\top Q=I,
\qquad
Q^\top v_{\mathrm{scale}}=0.
```

Preserve the existing quotient coordinate:

```math
z=Q^\top q.
```

Add:

```math
\boxed{
C_z^{\mathrm{cons}}
=
Q^\top C_q^{\mathrm{cons}}Q.
}
```

Save as:

```text
quotient_covariance_conservative_fusion
```

This is the **primary covariance for cross-bag fusion**.

Do not fuse the arbitrary 14-D KKT representatives directly.

---

# 15. Required ambiguous-direction diagnostics

The main scientific purpose is to communicate which directions one bag cannot constrain.

The existing local ridge/right-singular basis is already available from the final physical Jacobian.

For every ridge/right-singular direction `v_i`, compute:

```math
\sigma^2_{SG,i}
=
v_i^\top C_q^{SG}v_i,
```

```math
\sigma^2_{wrench,i}
=
v_i^\top C_q^{wrench}v_i,
```

```math
\sigma^2_{cons,i}
=
v_i^\top C_q^{cons}v_i.
```

Save:

```text
uncertainty_variance_conservative_fusion_in_ridge_basis
```

and:

```text
conservative_to_overlap_variance_ratio_in_ridge_basis
```

For a machine-zero denominator, save/report:

```text
undefined
```

rather than a huge numerical ratio.

Also report the top three non-gauge ridge directions with the largest conservative variance.

For each top direction, show its 14-D physical-chart components with labels:

```text
log_mass
second_moment_diag_1
second_moment_diag_2
second_moment_diag_3
second_moment_offdiag_12
second_moment_offdiag_13
second_moment_offdiag_23
cog_x
cog_y
cog_z
log_force_eff_1
log_force_eff_2
log_force_eff_3
log_force_eff_4
```

The exact common-scale gauge direction must be separately identified and not presented as a physical ambiguity.

---

# 16. Nominal-mass-gauge physical uncertainty

For experimental interpretation, additionally transform the new covariance to:

```text
mass = nominal vehicle-model mass
```

using the exact common-scale gauge transformation.

If the original chart coordinate is `q`, define the mass-fixed chart coordinate:

```math
q^{(m_0)}
=
q-q_0v_{\mathrm{scale}},
```

where `q_0` is the log-mass coordinate relative to nominal mass.

The covariance transformation is exactly linear:

```math
T_m
=
I-v_{\mathrm{scale}}e_0^\top,
```

```math
C_q^{(m_0)}
=
T_m C_q T_m^\top.
```

Do this for at least:

```text
overlap_corrected
wrench_corrected
conservative_fusion
```

Save:

```text
nominal_mass_gauge_covariance_overlap_corrected
nominal_mass_gauge_covariance_wrench_corrected
nominal_mass_gauge_covariance_conservative_fusion
```

Then use the existing analytic chart decoder Jacobian to propagate the mass-fixed conservative covariance to physical quantities.

At minimum save/report conservative-fusion one-sigma values for:

```text
force_effectiveness[0:4]
cog_offset[0:3]
principal inertia moments
```

The mass coordinate itself is fixed and should have zero variance in this representation, up to machine roundoff.

The primary fusion remains the 13-D quotient representation.

This nominal-mass output is for experimental readability.

---

# 17. Extend result JSON and NPZ

## `result.json`

Preserve all current outputs.

Add:

```text
uncertainty.parameter_covariance_conservative_fusion
```

Add under quotient diagnostics:

```text
diagnostics.quotient.covariance_conservative_fusion
```

Add a new diagnostic object:

```text
diagnostics.conservative_fusion
```

containing at least:

```text
interpretation
residual_mean
residual_uncentered_second_moment
sandwich_middle_residual_trace
sandwich_middle_sg_trace
sandwich_middle_conservative_trace
covariance_psd_order_min_eigenvalue
variance_conservative_fusion_in_ridge_basis
conservative_to_overlap_variance_ratio_in_ridge_basis
top_ambiguous_non_gauge_directions
```

Also include nominal-mass physical one-sigma summaries.

## `arrays.npz`

Add at minimum:

```text
residual_wrench_uncentered_second_moment
residual_acceleration_uncentered_second_moment

parameter_sandwich_middle_residual_uncentered
parameter_sandwich_middle_residual_centered_time_aligned
parameter_sandwich_middle_residual_mean_remainder
parameter_sandwich_middle_conservative_fusion

parameter_covariance_conservative_fusion
quotient_covariance_conservative_fusion

uncertainty_variance_conservative_fusion_in_ridge_basis
conservative_to_overlap_variance_ratio_in_ridge_basis

nominal_mass_gauge_covariance_overlap_corrected
nominal_mass_gauge_covariance_wrench_corrected
nominal_mass_gauge_covariance_conservative_fusion

nominal_mass_gauge_force_effectiveness
nominal_mass_gauge_force_effectiveness_std_overlap_corrected
nominal_mass_gauge_force_effectiveness_std_wrench_corrected
nominal_mass_gauge_force_effectiveness_std_conservative_fusion
```

Also save CoG and principal-inertia uncertainty summaries if practical.

Do not remove any existing NPZ key.

---

# 18. Reporting changes

Do not delete any existing report page.

The current residual-wrench pages remain.

Add one new page to `report.pdf` titled approximately:

```text
Conservative fusion uncertainty
```

The page must contain:

1. variance along the local ridge basis for:
   - SG overlap;
   - centered excess-wrench corrected;
   - conservative fusion;
2. conservative / SG-overlap variance ratio;
3. top three non-gauge ambiguous directions;
4. nominal-mass-gauge force-effectiveness estimate with one-sigma bars for:
   - SG overlap;
   - existing wrench-corrected;
   - conservative fusion.

Do not make a separate large ablation-style report.

The purpose is to see immediately:

```text
which directions became wider
```

and:

```text
whether the conservative covariance now represents bag-specific ambiguity.
```

---

# 19. Cross-bag consensus script must be extended for validation

Current script:

```text
single_bag_cross_bag_consensus.py
```

currently reads only:

```text
quotient_covariance_overlap_corrected
```

Extend it to load all three:

```text
quotient_covariance_overlap_corrected
quotient_covariance_wrench_corrected
quotient_covariance_conservative_fusion
```

Do not remove the existing cross-evaluation calculation.

## 19.1 Fix the pairwise-distance pseudoinverse while touching this code

The current implementation uses:

```python
np.linalg.pinv(combined)
```

and:

```python
max(0.0, value)
```

for pairwise squared distance.

Replace this with the repository's machine-precision symmetric pseudoinverse policy.

Before inversion:

```math
C=C_i+C_j
```

must be symmetrized.

Check its eigenvalues with a machine-precision tolerance.

If there is a materially negative eigenvalue, fail/report the covariance as invalid.

Do not hide a negative quadratic form with:

```python
max(0, value).
```

The squared distance must be produced directly by a valid PSD covariance.

This change is allowed because reliable conservative-fusion validation requires the distance calculation itself to be trustworthy.

---

# 20. Pairwise distance validation

For each covariance type:

```text
overlap_corrected
wrench_corrected
conservative_fusion
```

compute:

```math
d_{ij}^2
=
(z_i-z_j)^\top
(C_i+C_j)^\dagger
(z_i-z_j).
```

Save matrices:

```text
pairwise_distance_squared_overlap_corrected
pairwise_distance_overlap_corrected

pairwise_distance_squared_wrench_corrected
pairwise_distance_wrench_corrected

pairwise_distance_squared_conservative_fusion
pairwise_distance_conservative_fusion
```

The report must display the three distance matrices side by side or on clearly comparable pages.

Because:

```math
C_{\mathrm{cons},b}
\succeq
C_{\mathrm{overlap},b},
```

the conservative pairwise distance should not increase relative to the overlap-only distance, modulo machine-rank/support effects.

If a conservative distance materially increases, inspect covariance support/rank before accepting the implementation.

Do **not** impose a scientific threshold such as:

```text
distance must be < 1
```

or:

```text
distance must be < 3
```

The purpose is to measure whether the previous incompatibility was caused by underrepresented uncertainty, not to force agreement.

---

# 21. Actual three-distribution Gaussian fusion diagnostic

The intended downstream use is to combine the three bag-specific distributions.

Therefore perform the actual quotient-space Gaussian product as a diagnostic.

For each bag `b`:

```math
z_b\in\mathbb R^{13},
\qquad
C_b=C_{z,b}^{cons}.
```

Use the machine-precision symmetric pseudoinverse:

```math
\Lambda_b=C_b^\dagger.
```

Define:

```math
\Lambda_F
=
\sum_b\Lambda_b.
```

Then:

```math
C_F
=
\Lambda_F^\dagger,
```

and:

```math
z_F
=
C_F
\sum_b\Lambda_bz_b.
```

Save:

```text
fused_quotient_precision_conservative_fusion
fused_quotient_covariance_conservative_fusion
fused_quotient_coordinate_conservative_fusion
```

Report the machine rank of:

```math
\Lambda_F.
```

If it is less than 13, explicitly report unresolved fused directions rather than inventing precision.

Do not silently regularize with arbitrary diagonal jitter.

---

# 22. Bag-to-fused consistency

For each bag compute:

```math
d_{b,F}^2
=
(z_b-z_F)^\top
(C_b+C_F)^\dagger
(z_b-z_F).
```

Save/report:

```text
bag_to_fused_distance_squared
bag_to_fused_distance
```

Again, do not impose an arbitrary pass/fail scientific cutoff.

The diagnostic exists to show whether the three broad distributions occupy a mutually compatible region.

---

# 23. Required implementation tests

Tests are implementation-correctness tests.

Do not assert a desired real-bag scientific result.

## 23.1 Per-sample score outer-product identity

Generate synthetic:

```math
J_k,W_k,r_k.
```

Verify exactly that the implementation gives:

```math
M_{\mathrm{res}}
=
\sum_k
g_kg_k^\top
```

with:

```math
g_k=J_k^\top W_kr_k.
```

Also verify equivalence to:

```math
\sum_k
J_k^\top W_kr_kr_k^\top W_kJ_k.
```

## 23.2 PSD of residual middle

Verify:

```math
\lambda_{\min}(M_{\mathrm{res}})
```

is nonnegative up to machine precision.

## 23.3 Zero residual null case

If:

```math
r_k=0
```

for every sample, verify:

```text
parameter_covariance_conservative_fusion
==
parameter_covariance_overlap_corrected
```

to machine precision.

## 23.4 Constant nonzero residual retains mean

Use:

```math
r_k=c\neq0
```

for all `k`.

Centered residual covariance is zero.

Verify:

```math
M_{res}\neq0
```

and:

```text
conservative_fusion
```

is wider than:

```text
overlap_corrected
```

in at least one identified direction.

This test is mandatory.

It proves that the mean is not discarded.

## 23.5 No SG subtraction in conservative term

Construct a case where residuals are numerically consistent with the SG covariance.

Verify that the conservative residual term is still the observed uncentered score second moment.

Do not reduce it by the SG covariance.

The existing `wrench_corrected` path may subtract SG as before.

## 23.6 Existing wrench-corrected path unchanged

For the same synthetic input, compare the old existing `wrench_corrected` output before and after this revision.

It must be identical.

## 23.7 Conservative PSD ordering

On the common-scale gauge section verify:

```math
C^{cons}-C^{overlap}
```

is PSD up to machine precision.

## 23.8 Common-scale quotient invariance

Regauge an equivalent synthetic physical point by:

```math
q\mapsto q+c v_{\mathrm{scale}}
```

for several non-clean values such as:

```text
c = -1.17, 0.43, 1.91
```

Verify that:

```text
quotient coordinate
quotient covariance conservative fusion
```

remain invariant to numerical precision.

## 23.9 Nominal-mass gauge transform

Verify:

```math
T_m C T_m^\top
```

has machine-zero variance in the fixed mass coordinate and gives the same physical force-effectiveness uncertainty as a directly regauged chart point.

## 23.10 Weak-direction behavior

Construct a synthetic Jacobian with one deliberately weak non-gauge singular direction.

Verify that the conservative covariance represents a larger variance in that direction than in a strongly constrained direction when residual energy excites both.

Do not assert a hard ratio.

## 23.11 Cross-bag distance monotonicity

For synthetic full-rank quotient covariances with:

```math
C^{cons}_b\succeq C^{overlap}_b,
```

verify:

```math
d_{ij}^{cons}
\le
d_{ij}^{overlap}
```

up to numerical precision.

## 23.12 Pairwise covariance PSD failure

Feed a materially indefinite combined covariance to the new pairwise-distance helper.

Verify it fails/reports invalid covariance.

Do not clamp the squared distance to zero.

## 23.13 Gaussian product sanity

For three synthetic 13-D Gaussians, compare the implemented fused mean/covariance with the direct full-rank precision-sum formula.

## 23.14 Point-estimator non-regression

The new path is post-fit only.

Verify the point-estimator outputs remain unchanged:

```text
physical_coordinate
strict rotor lag
objective
acceleration residual
raw residual wrench
```

## 23.15 Report/output smoke test

A synthetic completed case must contain all new JSON/NPZ keys and generate the updated PDF.

---

# 24. Production bag JSON files

After implementation and tests, run the default estimator on exactly these three JSON files.

Paths are relative to:

```text
ros/examples/grape-param-estim/minimal/
```

## Failure bag 1

```text
bag_jsons/single_rosbag_1.json
```

Resolved input:

```text
/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/20260612_grape_hovering_4_2026-06-12-17-33-59.bag
```

Interval:

```text
19.0 s -- 25.0 s
```

## Failure bag 2

```text
bag_jsons/single_rosbag_2.json
```

Resolved input:

```text
/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/20260612_grape_hovering_6_2026-06-12-17-40-34.bag
```

Interval:

```text
25.5 s -- 31.0 s
```

## Successful bag

```text
bag_jsons/single_rosbag_succeeded.json
```

Resolved input:

```text
/home/leus/catkin_ws/bags/grape-drone/20260613_grape_hovering/20260613_grape_hovering_1_2026-06-13-13-44-01.bag
```

Interval:

```text
65.0 s -- 75.0 s
```

Do not substitute direct `--bag` arguments in the production commands.

---

# 25. Production defaults must remain unchanged

Use the default estimator settings.

In particular:

```text
SG window               = 1.0 s
SG degree               = 5
optimization covariance = identity
gimbal source           = measured_sg
rotor lag               = estimated
initial lag             = one median publication period
continuation            = epsilon_k = 2^-k, k=0,...,9
final lag               = exact strict-ZOH cell refinement
SO(3)                   = geometric
solver                  = custom KKT LM
common-scale KKT        = enabled
```

Do not run a covariance-mode sweep.

Do not run an SG-window sweep.

Do not run ablations.

---

# 26. Source commit before production

Production output paths are namespaced by the source commit.

Therefore source changes must be committed first.

## 26.1 Starting state

Run:

```bash
git status --short
git rev-parse HEAD
```

Expected starting commit:

```text
9009582432d26b7e592797dde8d7b62721c23c8a
```

Preserve unrelated user changes.

Do not reset or discard unrelated work.

## 26.2 Implement and test

Run the complete relevant single-bag test suite.

At minimum:

```bash
cd "$(git rev-parse --show-toplevel)/ros/examples/grape-param-estim/minimal"
python3 -m unittest discover -s tests -p 'test_*.py'
```

If the repository has an established alternative command, use it as well.

## 26.3 Commit source

Suggested source commit message:

```text
Add conservative fusion covariance
```

Then record:

```bash
SOURCE_COMMIT="$(git rev-parse HEAD)"
echo "$SOURCE_COMMIT"
```

All production outputs must be under:

```text
outputs/$SOURCE_COMMIT/
```

---

# 27. Production commands

From:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/ros/examples/grape-param-estim/minimal"
```

Source the existing ROS/catkin environment if required by the production machine.

Do not change Python/dependency versions for this task.

Use:

```text
grape_vehicle_model.json
```

## 27.1 Failure bag 1

```bash
python3 single_bag_savgol_estimator.py \
  --bag-json bag_jsons/single_rosbag_1.json \
  --vehicle-model grape_vehicle_model.json \
  --run-id single_rosbag_1_conservative_fusion_production_20260816
```

## 27.2 Failure bag 2

```bash
python3 single_bag_savgol_estimator.py \
  --bag-json bag_jsons/single_rosbag_2.json \
  --vehicle-model grape_vehicle_model.json \
  --run-id single_rosbag_2_conservative_fusion_production_20260816
```

## 27.3 Successful bag

```bash
python3 single_bag_savgol_estimator.py \
  --bag-json bag_jsons/single_rosbag_succeeded.json \
  --vehicle-model grape_vehicle_model.json \
  --run-id single_rosbag_succeeded_conservative_fusion_production_20260816
```

Do not pass:

```text
--skip-bag-sha256
```

Do not run the ablation runner.

---

# 28. Mandatory point-estimator non-regression against previous production

The current valid point-estimator production results are under:

```text
197caa23da3b201bca68174c48f7f35424cf58a7
```

Use these exact references.

## Failure bag 1

```text
outputs/197caa23da3b201bca68174c48f7f35424cf58a7/default/single_rosbag_1_residual_wrench_uncertainty_production_20260816/result.json
```

## Failure bag 2

```text
outputs/197caa23da3b201bca68174c48f7f35424cf58a7/default/single_rosbag_2_residual_wrench_uncertainty_production_20260816/result.json
```

## Successful bag

```text
outputs/197caa23da3b201bca68174c48f7f35424cf58a7/default/single_rosbag_succeeded_residual_wrench_uncertainty_production_20260816/result.json
```

For each new production run compare:

```text
parameters.chart_coordinate
parameters.rotor_lag_seconds
parameters.rotor_lag_strict_cell_seconds
common_evaluation.identity_objective_sum
common_evaluation.specific_acceleration_rmse_m_per_s2
common_evaluation.angular_acceleration_rmse_rad_per_s2
```

Recommended deterministic tolerance:

```python
np.allclose(new, old, rtol=1e-9, atol=1e-12)
```

Also compare `raw_residual_wrench` arrays if convenient.

If a material difference appears:

> **STOP. Do not accept or commit the production outputs until the point-estimator regression is understood.**

---

# 29. Mandatory per-bag production checks

Each production result must satisfy:

```text
status == completed
success == true
strict_final_evaluation == true
```

Each output directory must contain at least:

```text
status.json
result.json
arguments.json
timing.json
arrays.npz
report.pdf
residual_wrench.pdf
```

Also verify the new covariance products exist.

For every bag check:

1. `parameter_covariance_conservative_fusion` is finite and symmetric.
2. `quotient_covariance_conservative_fusion` is finite and symmetric.
3. the quotient covariance is PSD to machine precision.
4. `C_cons - C_overlap` is PSD on the quotient space.
5. the exact scale gauge remains the only machine null direction in the 14-D physical Jacobian.
6. the residual mean is non-destructively retained.
7. the uncentered residual middle is PSD.
8. no SG subtraction appears in the conservative residual term.
9. the existing `wrench_corrected` covariance is still produced unchanged.
10. the conservative covariance is not silently substituted for the point-estimation metric.

---

# 30. Per-bag production summary

For each bag print/save a compact summary containing:

```text
bag JSON path
resolved bag path
time interval
strict lag cell
identity objective

residual wrench mean
residual wrench component std
residual wrench uncentered second-moment eigenvalues

trace(M_SG)
trace(M_existing_wrench)
trace(M_residual_uncentered)
trace(M_conservative)

min eig(C_cons - C_overlap) in quotient space

ridge-direction variance:
    overlap
    wrench corrected
    conservative fusion
    conservative / overlap ratio

nominal-mass force effectiveness:
    point
    sigma overlap
    sigma wrench corrected
    sigma conservative fusion
```

For machine-zero denominator ratios, print:

```text
undefined
```

---

# 31. Run the updated cross-bag consensus on the three new production outputs

After the three production cases complete, run the updated:

```text
single_bag_cross_bag_consensus.py
```

with exactly the three new case directories.

Assuming the source commit was stored in:

```bash
SOURCE_COMMIT="$(git rev-parse HEAD)"
```

the command should be structurally:

```bash
python3 single_bag_cross_bag_consensus.py \
  --case-directory \
    "outputs/$SOURCE_COMMIT/default/single_rosbag_1_conservative_fusion_production_20260816" \
    "outputs/$SOURCE_COMMIT/default/single_rosbag_2_conservative_fusion_production_20260816" \
    "outputs/$SOURCE_COMMIT/default/single_rosbag_succeeded_conservative_fusion_production_20260816" \
  --vehicle-model grape_vehicle_model.json \
  --run-id three_bag_conservative_fusion_production_20260816
```

If the actual CLI accepts repeated `--case-directory` rather than one `nargs` list, use the actual parser contract.

Do not change the scientific input set.

The consensus result must be generated under the existing commit-namespaced consensus location.

---

# 32. Cross-bag validation output

The consensus report/result must contain:

```text
pairwise distance matrix: overlap corrected
pairwise distance matrix: existing wrench corrected
pairwise distance matrix: conservative fusion

fused quotient mean: conservative fusion
fused quotient covariance: conservative fusion
bag-to-fused distances
```

Also preserve the existing strict cross-evaluation matrix:

```text
Delta L_ij.
```

The strict cross-evaluation matrix is not expected to change merely because covariance broadened.

It remains a point-estimate/model-compatibility diagnostic.

The covariance distance is the diagnostic expected to reflect the new ambiguity representation.

---

# 33. Scientific interpretation rules

The implementation report must distinguish the following clearly.

## Point disagreement

```math
\hat z_i-\hat z_j
```

This is unchanged by the new covariance.

## Existing SG-overlap uncertainty

This is the measurement/SG overlap working covariance.

## Existing centered excess-wrench uncertainty

This subtracts the SG-predicted wrench contribution and uses centered residual fluctuation.

## Conservative fusion uncertainty

This deliberately retains:

```text
SG-overlap uncertainty
+
uncentered per-sample residual score second moment.
```

It is intentionally broader and intended for distribution fusion.

Do not describe it as:

```text
the statistically exact covariance
```

or:

```text
the true posterior covariance.
```

Describe it as:

```text
a conservative local Gaussian uncertainty representation for fusion.
```

---

# 34. What constitutes a successful scientific result

Do not force numerical agreement.

A successful implementation means:

1. each bag's conservative covariance is algebraically valid;
2. each bag preserves weakly constrained directions as broad covariance directions;
3. pairwise conservative distances do not become larger through an implementation artifact;
4. the three-distribution product can be computed without arbitrary regularization;
5. the resulting fused distribution reports unresolved directions if they remain.

It is acceptable if the three bags remain significantly separated even after conservative broadening.

That would be a scientific result.

Do not tune multipliers until the bags agree.

No arbitrary scalar covariance inflation factor is authorized in this task.

The conservative broadening comes only from the explicit residual second-moment construction.

---

# 35. No arbitrary safety multiplier

Do not introduce:

```text
x2 covariance
x10 covariance
confidence multiplier
temperature
manual inflation factor
minimum variance floor
ridge floor
```

in this task.

The broadening must come from observed residual information and exact algebra.

If the result remains too narrow, report that outcome.

Do not hide it by adding an arbitrary constant.

---

# 36. Recommended non-tuning diagnostic

Without changing the covariance definition, report:

```math
\operatorname{tr}(M_{\mathrm{res}})
/
\operatorname{tr}(M_{\mathrm{SG}})
```

and the corresponding ridge-direction variance ratios.

This helps quantify whether the conservative covariance is dominated by residual evidence or SG uncertainty.

This diagnostic must not become a tuning parameter.

---

# 37. Production results commit and push

Only after:

- tests pass;
- all three default production runs succeed;
- point-estimator non-regression passes;
- conservative covariance PSD-order checks pass;
- cross-bag consensus completes;

run:

```bash
git status --short
```

Add only the intended production outputs and consensus outputs.

Suggested results commit message:

```text
Add conservative fusion production results
```

Then push the current branch:

```bash
git push origin HEAD
```

Finally print:

```bash
git log -2 --oneline
git status --short
```

The working tree should be clean except for deliberately preserved unrelated pre-existing user changes.

---

# 38. Mandatory final implementation report

After push, report all of the following.

## Provenance

```text
starting commit:
source implementation commit:
production results commit:
pushed branch:
```

## Source files changed

List exact source/test files.

## Tests

List exact commands and pass/fail counts.

## Production cases

For each bag:

```text
JSON path
resolved bag path
time interval
output directory
status
strict lag cell
identity objective
```

## Conservative covariance per bag

For each bag:

```text
min/max eigenvalue in quotient space
min eigenvalue of C_cons-C_overlap
trace overlap covariance
trace existing wrench-corrected covariance
trace conservative covariance
top 3 ambiguous non-gauge directions
nominal-mass force-effectiveness one-sigma values
```

## Cross-bag comparison

Provide all three pairwise distance matrices:

```text
overlap corrected
existing wrench corrected
conservative fusion
```

and the ratios:

```math
d_{ij}^{cons}/d_{ij}^{overlap}.
```

## Fused distribution

Report:

```text
fused quotient rank
fused quotient coordinate
fused quotient covariance eigenvalues
bag-to-fused distances
```

## Interpretation

State whether:

```text
the previously large cross-bag distances were materially reduced
when each bag's unresolved directions were represented more conservatively.
```

Do not overstate agreement if it is not present.

---

# 39. Suggested source files to modify

Expected files include:

```text
single_bag_savgol_covariance.py
single_bag_savgol_core.py
single_bag_savgol_reports.py
single_bag_cross_bag_consensus.py
tests/test_residual_wrench_uncertainty.py
```

A dedicated new test file such as:

```text
tests/test_conservative_fusion_covariance.py
```

is encouraged.

Do not perform unrelated refactors.

---

# 40. Acceptance checklist

The task is complete only if every item is satisfied.

- [ ] Starting commit recorded.
- [ ] Existing point estimator unchanged.
- [ ] Existing covariance products preserved.
- [ ] Existing residual-wrench products preserved.
- [ ] New uncentered residual score middle implemented.
- [ ] Residual mean is not removed from the conservative term.
- [ ] No SG subtraction is applied to the conservative residual term.
- [ ] `M_res` is implemented as a sum of per-sample score outer products.
- [ ] `M_res` is PSD to machine precision.
- [ ] `M_cons = M_SG + M_res`.
- [ ] Conservative covariance uses the same exact common-scale gauge reduction.
- [ ] `C_cons-C_overlap` is PSD on identified quotient space.
- [ ] `parameter_covariance_conservative_fusion` saved.
- [ ] `quotient_covariance_conservative_fusion` saved.
- [ ] Nominal-mass-gauge conservative covariance saved.
- [ ] Conservative force-effectiveness one-sigma values saved/reported.
- [ ] Ridge-direction conservative variances saved.
- [ ] Top ambiguous non-gauge directions reported.
- [ ] Residual wrench uncentered second moment saved.
- [ ] Residual acceleration uncentered second moment saved.
- [ ] Existing `wrench_corrected` calculation unchanged.
- [ ] No block sandwich added.
- [ ] No temporal stochastic model added.
- [ ] No arbitrary inflation multiplier added.
- [ ] No finite differences added.
- [ ] Pairwise-distance pseudoinverse made machine-precision symmetric/PSD-aware.
- [ ] Negative distance is not hidden with `max(0, value)`.
- [ ] Cross-bag consensus loads all three covariance types.
- [ ] Three pairwise distance matrices generated.
- [ ] Conservative quotient Gaussian product generated.
- [ ] Bag-to-fused distances generated.
- [ ] Full relevant tests pass.
- [ ] Source implementation committed before production.
- [ ] Failure bag 1 run from `bag_jsons/single_rosbag_1.json`.
- [ ] Failure bag 2 run from `bag_jsons/single_rosbag_2.json`.
- [ ] Successful bag run from `bag_jsons/single_rosbag_succeeded.json`.
- [ ] All three production runs completed.
- [ ] New point estimates match previous `197caa23...` production results.
- [ ] Updated three-bag consensus production run completed.
- [ ] Production and consensus outputs committed.
- [ ] Both commits pushed.
- [ ] Final report includes both commit SHAs and exact output paths.

No checklist item is optional.
