# CoG-Prior-Conditioned Reseeding and Prior-Free Full Refinement
## Mandatory Implementation, Validation, Production, and Push Procedure

## 0. Purpose and exact repository baseline

Repository:

```text
hiroya1224/ProbTF-demo
```

This task is defined **relative to the current repository HEAD**:

```text
b2daa3066ed1838b213d88f5a5f10abba1d3ab35
```

Commit message:

```text
Add conservative fusion production results
```

The immediately preceding source implementation commit is:

```text
d575adf7062789188e10e3356b0d1a1f9dfb725a
```

Commit message:

```text
Add conservative fusion covariance
```

The production outputs at the baseline HEAD were generated from source commit
`d575adf7062789188e10e3356b0d1a1f9dfb725a`.

This task tests one specific hypothesis:

> The current data-only estimator may converge to a locally valid parameter
> solution in which CoG absorbs physical effects that could also be explained
> by inertia and/or rotor force effectiveness. The existing full joint
> covariance contains local cross-covariance information describing that
> compensation direction. A physically plausible CoG prior can therefore be
> used **only to construct a new initial point**. Starting from that point, the
> complete original prior-free nonlinear estimator is run again with every
> presently estimated quantity free. If the failure bags then converge to
> success-like physical parameters while retaining essentially the same
> data-only fit quality, the data support a distinct nonlinear solution branch
> that the original initialization did not reach.

The operational workflow is exactly:

```text
existing data-only estimate
        |
        |  use saved full joint covariance
        v
Gaussian conditioning with CoG prior
        |
        |  conditioned mean is ONLY a new initial coordinate
        v
complete original prior-free estimator
        |
        |  all current estimated quantities remain free
        v
refined data-only estimate
```

The CoG prior is **not** a prior in the refinement objective.

The CoG prior is **not** retained after reseeding.

The conditioned Gaussian is **not** itself treated as the final nonlinear
answer.

The full nonlinear refinement is mandatory.

---

# 1. Scientific question

The current single-bag estimator produces substantially different physical
parameter point estimates across the two failure bags and the successful bag.

At the same time, the full saved joint covariance shows strong coupling among
CoG, inertia-chart coordinates, and rotor force-effectiveness coordinates.

The current vehicle-model nominal CoG is:

```math
c_0
=
\begin{bmatrix}
-0.002024708562282\\
-0.000030526578941\\
+0.009509749599446
\end{bmatrix}
\ {\rm m}.
```

Equivalently:

```text
(-2.0247086, -0.0305266, +9.5097496) mm
```

This task uses the fixed conditioning prior:

```math
c
\sim
\mathcal N
\left(
c_0,\,
\sigma_c^2 I_3
\right),
\qquad
\sigma_c=0.001\ {\rm m}.
```

Thus the production prior standard deviation is exactly:

```text
1 mm independently on x, y, and z.
```

The primary experimental question is:

> Can the two failure bags, after CoG-covariance-guided reseeding followed by a
> completely prior-free full nonlinear refinement, converge to the same
> physical parameter region as the successful bag without materially worsening
> their own original data-only objective?

If yes, the earlier bag-to-bag disagreement is explained at least in substantial
part by nonlinear parameter compensation rather than by genuinely incompatible
physical parameters.

---

# 2. Mandatory scope restrictions

The following restrictions are part of the scientific definition of the task.

> **DO NOT change the existing data-only point-estimation objective.**

> **DO NOT add a CoG prior residual to the refinement objective.**

> **DO NOT perform MAP refinement.**

> **DO NOT keep the CoG prior active after the conditioned seed has been
> constructed.**

> **DO NOT fix CoG during refinement.**

> **DO NOT fix inertia during refinement.**

> **DO NOT fix force effectiveness during refinement.**

> **DO NOT fix the rotor lag during refinement.**

> **DO NOT fix any presently estimated physical degree of freedom, except for
> the existing exact common-scale gauge treatment already enforced by the
> current KKT formulation.**

> **DO NOT alter the existing common-scale gauge.**

> **DO NOT alter the matrix-exponential second-moment inertia chart.**

> **DO NOT alter the pose SG window or degree.**

> **DO NOT alter the measured-gimbal default.**

> **DO NOT alter the current identity acceleration weighting used by the
> production default.**

> **DO NOT alter the power-of-two continuation.**

> **DO NOT alter exact strict-ZOH cell refinement.**

> **DO NOT alter the residual-wrench uncertainty implementation.**

> **DO NOT alter the conservative-fusion covariance implementation in this
> task.**

> **DO NOT use the successful bag's parameter values when constructing either
> failure bag's conditioned seed.**

> **DO NOT tune the 1 mm CoG prior by looking at which value makes the failure
> bags agree best with the successful bag.**

> **DO NOT clip, project, or otherwise heuristically repair an extreme
> conditioned inertia-chart coordinate.**

> **DO NOT introduce arbitrary bounds, priors, regularizers, eigenvalue
> thresholds, covariance inflation factors, or temporal-correlation models.**

> **DO NOT use code or results from `minimal/legacies/` for this task.**

> **DO NOT overwrite or mutate the baseline production outputs.**

The point of the experiment is precisely to test whether the **existing**
nonlinear model, when reinitialized using information already present in the
existing joint distribution, reaches a different data-supported solution.

---

# 3. Existing estimator that must remain the source of truth

The current estimator entry point is:

```text
ros/examples/grape-param-estim/minimal/single_bag_savgol_estimator.py
```

Its reusable execution function is:

```python
run_estimator(arguments, case_name="default", output_directory=None)
```

The current production scientific path must remain unchanged:

```text
pose SG window          = 1.0 s
pose SG degree          = 5
gimbal source           = measured_sg
gimbal SG window        = same centered irregular SG construction
covariance_mode         = identity
rotor lag               = estimated
continuation            = epsilon_k = 2^-k, k = 0,...,9
strict refinement       = exact strict-ZOH cell refinement
solver                  = custom_kkt_lm
common-scale gauge      = existing exact KKT gauge
physical chart          = existing 14-D SI matrix-exponential chart
```

The 14 physical chart coordinates remain:

```text
q[0]       log mass
q[1:7]     symmetric second-moment matrix-exponential coordinates
q[7:10]    additive CoG displacement from the vehicle-model reference CoG
q[10:14]   log force-effectiveness scales
```

The exact common-scale gauge direction remains:

```math
v_{\rm scale}
=
(1,1,1,1,0,0,0,0,0,0,1,1,1,1)^T.
```

No additional gauge is introduced.

---

# 4. Exact baseline production cases

Use the existing three completed production cases as the immutable input
distributions.

Base directory:

```text
ros/examples/grape-param-estim/minimal/outputs/
d575adf7062789188e10e3356b0d1a1f9dfb725a/default/
```

Failure bag 1:

```text
single_rosbag_1_conservative_fusion_production_20260816
```

Failure bag 2:

```text
single_rosbag_2_conservative_fusion_production_20260816
```

Successful bag:

```text
single_rosbag_succeeded_conservative_fusion_production_20260816
```

For every input case, require:

```text
result.json
arguments.json
arrays.npz
status.json
```

and require:

```text
result.json["status"] == "completed"
result.json["strict_final_evaluation"] == true
```

The source commit recorded by those cases must be:

```text
d575adf7062789188e10e3356b0d1a1f9dfb725a
```

Do not silently accept another case directory with the same bag name.

---

# 5. Distribution used only for reseeding

The reseeding distribution is the existing full chart-space output:

```text
parameter_covariance_conservative_fusion
```

Use:

```text
result.json["parameters"]["chart_coordinate"]
```

as the original chart mean:

```math
\mu\in\mathbb R^{14}.
```

Use the full saved matrix:

```text
arrays.npz["parameter_covariance_conservative_fusion"]
```

as:

```math
C\in\mathbb R^{14\times14}.
```

Cross-check that the same covariance stored in `result.json["uncertainty"]`
agrees with the NPZ matrix to machine-level numerical precision.

The role of this covariance in this task is deliberately limited:

> It is used as a **local joint parameter geometry for generating a new
> initialization**.

This task does **not** require interpreting it as an exact generative posterior.

This task does **not** use its conditioned covariance inside the nonlinear
refinement.

---

# 6. CoG prior in the current chart

The `SiParameterChart` uses additive CoG coordinates:

```math
c(q)
=
c_0
+
\begin{bmatrix}
q_7\\q_8\\q_9
\end{bmatrix}.
```

Therefore the nominal CoG prior is especially simple in chart coordinates:

```math
H_c q
\sim
\mathcal N(0,R_c),
```

where:

```math
H_c
=
\begin{bmatrix}
0_{3\times7} & I_3 & 0_{3\times4}
\end{bmatrix},
```

and:

```math
R_c
=
(0.001)^2 I_3
=
10^{-6} I_3\ {\rm m}^2.
```

Do not convert the prior mean through another coordinate system.

Do not set the physical CoG prior mean to `(0,0,0)`.

The prior mean is the vehicle-model nominal CoG, which is exactly `q[7:10]=0`
in the current chart.

---

# 7. Gaussian conditioning used to generate the new seed

Given:

```math
q\sim\mathcal N(\mu,C)
```

and:

```math
H_cq
\sim
\mathcal N(0,R_c),
```

compute:

```math
S_c
=
H_c C H_c^T + R_c.
```

Because `R_c` is strictly positive definite, `S_c` is a finite positive-definite
`3 x 3` matrix.

Use a direct symmetric solve of this `3 x 3` system.

Do **not** pseudoinvert the full singular 14-D covariance.

Define:

```math
K_c
=
C H_c^T S_c^{-1}.
```

Then:

```math
\boxed{
\mu_{\rm cond}
=
\mu
-
K_c H_c\mu
}
```

and:

```math
\boxed{
C_{\rm cond}
=
C
-
K_c H_c C.
}
```

Numerically symmetrize only for floating-point roundoff:

```math
C_{\rm cond}
\leftarrow
\frac12
(C_{\rm cond}+C_{\rm cond}^T).
```

No eigenvalue clipping is allowed.

No covariance inflation is allowed.

No artificial floor is allowed beyond the explicitly specified CoG prior
covariance `R_c`.

---

# 8. Gauge consistency of the conditioning step

The original covariance lives on the current KKT common-scale gauge section.

Conditioning on CoG must not inject a scale-gauge displacement because CoG is
invariant under the exact common scaling.

Verify:

```math
v_{\rm scale}^T
(\mu_{\rm cond}-\mu)
\approx 0
```

to machine precision.

Also verify that the conditioned covariance retains the same exact gauge null
direction:

```math
C_{\rm cond}v_{\rm scale}
\approx0.
```

Do not force these conditions by projection after the fact.

They must follow from the existing covariance and the conditioning algebra.

If they fail materially, treat the run as an implementation error.

---

# 9. Conditioning diagnostics that must be saved

For every bag, save the following before any nonlinear refinement.

## 9.1 Original values

```text
original_chart_coordinate
original_parameter_covariance_conservative_fusion
original_rotor_lag_seconds
original_strict_identity_objective
original_nominal_mass_gauge_parameters
```

## 9.2 Prior specification

```text
cog_prior_mean_physical_m
cog_prior_mean_chart
cog_prior_std_m
cog_prior_covariance_m2
```

with production values:

```text
cog_prior_mean_chart = [0, 0, 0]
cog_prior_std_m      = [0.001, 0.001, 0.001]
```

## 9.3 Conditioning matrices

```text
cog_selector_H
innovation_covariance
conditioning_gain
conditioned_chart_coordinate
conditioned_parameter_covariance
```

## 9.4 Physical decoded conditioned point

Decode `conditioned_chart_coordinate` through the current `SiParameterChart`
without modification and save:

```text
conditioned_mass_kg
conditioned_inertia_kg_m2
conditioned_principal_inertia_moments_kg_m2
conditioned_cog_position_body_m
conditioned_force_effectiveness
conditioned_scale_free_inertia_over_mass_m2
conditioned_scale_free_force_effectiveness_over_mass
```

The conditioned physical point may be extreme.

Do not clip it.

Do not replace it by a manually selected "reasonable" value.

The purpose of the next nonlinear stage is precisely to test whether this
local-Gaussian extrapolation leads to another true nonlinear solution basin.

## 9.5 Numerical validation fields

Save:

```text
conditioning_covariance_symmetry_error
conditioning_covariance_min_eigenvalue
conditioning_update_scale_gauge_dot
conditioning_covariance_scale_gauge_norm
conditioning_innovation_condition_number
```

---

# 10. Conditioned result is an output, not the final answer

The direct conditioning result must be emitted before refinement.

This is required because it distinguishes:

```text
what the saved local Gaussian predicts
```

from:

```text
what the actual nonlinear rigid-body estimator supports.
```

The direct conditioned result is scientifically useful even if the subsequent
nonlinear refinement returns to the original solution.

Do not silently replace the direct conditioned point by the refined result.

---

# 11. Construction of the nonlinear-refinement initialization

For every bag, clone the exact original `arguments.json`.

Preserve every scientific setting.

Override only the initialization/output fields required for reseeding.

Set:

```text
initial_coordinate = conditioned_chart_coordinate
```

Set the initial rotor lag to the original completed data-only result:

```text
initial_rotor_lag =
    result.json["parameters"]["rotor_lag_seconds"]
```

Set:

```text
scale_initial_offset = 0.0
```

because the conditioned coordinate is already a complete chart coordinate on
the same gauge section.

The original rotor lag is only an initial seed.

It is **not fixed**.

The original physical parameters are only used indirectly through the
conditioned seed.

They are **not fixed**.

---

# 12. Full nonlinear refinement is prior-free

Call the existing estimator pipeline again from the conditioned seed.

The refinement must use the unchanged existing call path:

```python
run_estimator(...)
```

or the exact same internal estimator function used by it.

The refinement objective must be exactly the existing data-only objective.

There must be no additional term of the form:

```math
\|c-c_0\|^2/\sigma_c^2.
```

There must be no CoG-prior residual.

There must be no prior Hessian.

There must be no prior precision added to the parameter information matrix.

There must be no conditioning covariance passed into the optimizer.

The refinement must again estimate all current quantities exactly as the normal
pipeline does.

In particular, during refinement:

```text
mass / common-scale-section coordinate    free as in current KKT estimator
inertia-chart coordinates                 free
CoG                                       free
force effectiveness                       free
rotor lag                                 free
```

The existing KKT common-scale gauge remains enforced.

The existing rotor-lag continuation runs normally.

The existing exact strict-ZOH cell refinement runs normally.

The existing residual-wrench diagnostics run normally.

The existing conservative-fusion covariance is recomputed normally at the new
refined solution.

---

# 13. Important interpretation of the refinement

The refinement is intentionally allowed to do any of the following.

## Outcome A: return to the original solution

```math
q_{\rm refine}\approx q_{\rm original}.
```

This means the conditioned tangent direction did not lead to a separate
data-supported nonlinear solution basin.

## Outcome B: converge to a different solution with essentially the same loss

```math
q_{\rm refine}\not\approx q_{\rm original},
```

while:

```math
L_{\rm refine}\approx L_{\rm original}.
```

This is the primary scientifically interesting outcome.

It means the same pose trajectory admits a distinct nonlinear parameter
solution with comparable explanatory power.

## Outcome C: converge to a different solution with lower loss

```math
L_{\rm refine}<L_{\rm original}.
```

This indicates that the original initialization missed a better data-only
solution.

## Outcome D: conditioned point is poor and refinement remains substantially
worse

This means the local Gaussian compensation direction did not continue to a
useful nonlinear branch.

All four outcomes are valid experimental results.

Do not modify the algorithm to force Outcome B or C.

---

# 14. New implementation boundary

Prefer a new orchestration module rather than modifying the scientific core.

Recommended new file:

```text
ros/examples/grape-param-estim/minimal/
single_bag_cog_prior_reseed_refinement.py
```

Its responsibilities are:

1. load one immutable completed baseline case;
2. validate the baseline case;
3. load the saved full joint conservative-fusion covariance;
4. apply the fixed 1 mm nominal-CoG Gaussian conditioning;
5. save the direct conditioning result;
6. clone the original estimator arguments;
7. replace only the initialization fields described above;
8. call the existing prior-free `run_estimator(...)`;
9. save original-vs-conditioned-vs-refined comparisons;
10. never modify the baseline case.

Do not duplicate the estimator implementation in the new file.

Do not fork the dynamics model.

Do not copy/paste the LM solver.

The existing estimator remains the single source of truth.

---

# 15. Suggested reusable pure function

Add a small pure function with no estimator side effects, either in the new
module or in an appropriately narrow helper:

```python
condition_chart_gaussian_on_cog_prior(
    mean,
    covariance,
    *,
    cog_std_m,
) -> CogConditioningResult
```

The production call uses:

```python
cog_std_m = np.array([0.001, 0.001, 0.001])
```

The prior mean is not a free production tuning argument.

It is always the current chart reference CoG, i.e. zero in `q[7:10]`.

A scalar CLI convenience value may be accepted:

```text
--cog-prior-std-m 0.001
```

but the exact value used must be persisted in every output.

The default production value is exactly `0.001`.

---

# 16. Output structure

Do not place new files inside the immutable baseline directories.

After the implementation source commit is created, let:

```bash
SOURCE_COMMIT=$(git rev-parse HEAD)
```

Use commit-namespaced output beneath:

```text
ros/examples/grape-param-estim/minimal/outputs/
<SOURCE_COMMIT>/cog_prior_reseed_refinement/
```

For each bag:

```text
<run-id>/
    status.json
    arguments.json
    conditioning.json
    conditioning_arrays.npz
    comparison.json
    comparison.pdf
    refined/
        result.json
        arguments.json
        arrays.npz
        report.pdf
        residual_wrench.pdf
        status.json
        timing.json
        ...
```

The `refined/` directory must be a standard completed estimator output created
through the existing report pipeline.

Recommended production run IDs:

```text
single_rosbag_1_cog1mm_reseed_refinement_production_20260817
single_rosbag_2_cog1mm_reseed_refinement_production_20260817
single_rosbag_succeeded_cog1mm_reseed_refinement_production_20260817
```

---

# 17. Required `comparison.json`

For each bag, save one compact comparison object containing all three stages:

```text
original
conditioned
refined
```

For `original` and `refined`, include:

```text
chart_coordinate
rotor_lag_seconds
strict_identity_objective_sum
specific_acceleration_rmse_m_per_s2
angular_acceleration_rmse_rad_per_s2
mass_kg
inertia_kg_m2
principal_inertia_moments_kg_m2
cog_position_body_m
force_effectiveness
inertia_over_mass_m2
force_effectiveness_over_mass
```

For `conditioned`, include the corresponding decoded physical quantities but
do not claim a strict nonlinear objective unless it is explicitly evaluated by
the nonlinear model.

Also save:

```math
\Delta L
=
L_{\rm refine}-L_{\rm original},
```

and:

```math
\frac{\Delta L}{\max(L_{\rm original},\epsilon_{\rm machine})}.
```

Save parameter differences:

```text
refined_minus_original_chart
conditioned_minus_original_chart
refined_minus_original_cog_m
refined_minus_original_force_effectiveness
refined_minus_original_inertia_over_mass_m2
```

---

# 18. Required direct evaluation of the conditioned seed

Before starting the full refinement, evaluate the conditioned physical
coordinate once with the current nonlinear model at the original rotor-lag
seed.

This evaluation is diagnostic only.

Save:

```text
conditioned_seed_identity_objective_at_original_lag
conditioned_seed_specific_acceleration_rmse
conditioned_seed_angular_acceleration_rmse
conditioned_seed_is_finite
```

This tells us how far the local Gaussian conditioning moved away from the true
nonlinear loss surface before refinement.

Do not optimize during this diagnostic evaluation.

---

# 19. Three-bag comparison required after all refinements

Run the procedure on all three production bags.

The two failure bags are the primary test cases.

The successful bag must also be run through the same procedure as a stability
control.

After all three refined outputs exist, compare at least:

```text
failure1 original
failure1 refined
failure2 original
failure2 refined
success original
success refined
```

in the exact scale-free physical quantities:

```math
J/m,
\qquad
f_i/m,
\qquad
c_{\rm CoG}.
```

The primary scientific question is whether:

```math
q_{{\rm failure1},{\rm refine}}
```

and:

```math
q_{{\rm failure2},{\rm refine}}
```

move into the same physical region as the successful-bag solution while
retaining comparable failure-bag data-only objective values.

---

# 20. Required cross-evaluation

Use the current cross-bag evaluation machinery rather than inventing a new
metric.

For physical-parameter cross-evaluation, continue to profile the target bag's
own rotor lag as the current consensus code does.

At minimum compute:

1. original success physical parameters evaluated on failure bag 1;
2. original success physical parameters evaluated on failure bag 2;
3. refined failure bag 1 physical parameters evaluated on the success bag;
4. refined failure bag 2 physical parameters evaluated on the success bag;
5. all three refined physical parameters cross-evaluated on all three bags.

Do not force one bag's rotor-lag value onto another bag.

The lag is a bag-specific nuisance variable in the cross-evaluation.

Save both absolute costs and delta costs relative to each target bag's own
data-only optimum.

---

# 21. Required distribution comparison after refinement

The existing estimator will produce a new full conservative-fusion covariance
at every refined solution.

Retain it.

For every refined bag, save the existing scale-quotient covariance and compare:

```text
original conservative-fusion covariance
refined conservative-fusion covariance
```

Do not reuse the conditioned Gaussian covariance as the refined covariance.

The conditioned covariance is only a local reseeding diagnostic.

For the refined outputs, use the covariance recomputed by the existing
post-fit pipeline at the refined nonlinear solution.

---

# 22. Do not use success information to guide the failure refinements

This is a mandatory anti-cheating rule.

For failure bag 1, the seed may depend only on:

```text
failure bag 1 original chart coordinate
failure bag 1 original full conservative-fusion covariance
vehicle-model nominal CoG
fixed 1 mm CoG prior std
failure bag 1 original rotor-lag seed
```

For failure bag 2, analogously use only failure bag 2 information.

The successful bag's point estimate, covariance, cost, force effectiveness,
inertia, or CoG must not enter either conditioning calculation.

Success is used only after the failure refinements are complete, for
evaluation.

---

# 23. Unit tests: Gaussian conditioning

Add dedicated tests, recommended file:

```text
minimal/tests/test_cog_prior_reseed_refinement.py
```

The tests must include all of the following.

## 23.1 Nominal CoG chart identity

Verify directly through `SiParameterChart` that:

```text
q[7:10] = [0,0,0]
```

decodes to the exact current vehicle-model nominal CoG.

## 23.2 Zero cross-covariance test

Construct a covariance in which CoG is independent of all non-CoG parameters.

Applying the CoG prior must change only:

```text
q[7:10]
```

and their covariance block.

## 23.3 Cross-covariance propagation test

Construct a known correlated Gaussian.

Verify analytically that conditioning CoG changes the correlated non-CoG
parameter means and covariances exactly as expected.

## 23.4 Prior already centered test

If the input mean already has:

```text
q[7:10] = 0
```

then the conditioned mean must remain unchanged while the appropriate
covariance decreases.

## 23.5 Weak-prior limit

For a very large explicitly supplied test standard deviation, the conditioned
mean/covariance must approach the input mean/covariance.

This is a unit test only.

Do not add a large-prior production ablation.

## 23.6 Tight-prior limit

For a very small explicitly supplied test standard deviation, the conditioned
CoG mean must approach zero.

This is a unit test only.

The production standard deviation remains exactly 1 mm.

## 23.7 PSD and symmetry

Verify:

```math
C_{\rm cond}=C_{\rm cond}^T
```

to floating-point tolerance and:

```math
C-C_{\rm cond}\succeq0.
```

Do not enforce this by clipping.

## 23.8 Gauge preservation

Verify:

```math
v_{\rm scale}^T(\mu_{\rm cond}-\mu)\approx0
```

and:

```math
C_{\rm cond}v_{\rm scale}\approx0.
```

## 23.9 No full covariance pseudoinverse

The conditioning code must not require a pseudoinverse of the singular 14-D
covariance.

Only the strictly positive-definite `3 x 3` innovation covariance is solved.

---

# 24. Unit/integration tests: refinement boundary

The tests must make it difficult to accidentally turn this experiment into a
MAP estimator.

## 24.1 Prior never enters estimator objective

Verify that the wrapper calls the existing estimator with no prior residual,
prior precision, prior weight, or modified loss.

## 24.2 All physical coordinates are passed as an initial coordinate

Verify that all 14 values of `conditioned_chart_coordinate` are passed to:

```text
initial_coordinate
```

without manually replacing individual inertia/CoG/force-effectiveness entries.

## 24.3 CoG is not fixed

Verify the refinement configuration contains no CoG-fixing branch.

## 24.4 Rotor lag is not fixed

For the production default, verify:

```text
lag_mode == "estimated"
```

is preserved.

The original lag is passed only through:

```text
initial_rotor_lag
```

and not through:

```text
fixed_rotor_lag
```

unless the original baseline case itself was a fixed-lag case.

The three specified production inputs are estimated-lag default cases.

## 24.5 Scientific arguments are preserved

Compare baseline and refinement arguments.

Except for:

```text
initial_coordinate
initial_rotor_lag
scale_initial_offset
output location / run identifier
```

all estimator scientific settings must be exactly preserved.

## 24.6 Baseline files remain unchanged

The wrapper must never open baseline `result.json`, `arrays.npz`, or report files
for writing.

## 24.7 Extreme conditioned seed is not clipped

Use a synthetic correlated covariance that produces a large inertia-chart shift.

Verify that the exact conditioned chart coordinate is passed to the refinement
initializer.

---

# 25. Existing regression tests

Run the complete currently relevant test suite, not only the new test file.

At minimum run:

```bash
python3 -m pytest minimal/tests
```

from:

```text
ros/examples/grape-param-estim/
```

If the repository uses another already-established invocation for the same
suite, use that exact invocation instead.

All existing tests must continue to pass.

Do not delete or weaken tests in order to make this task pass.

---

# 26. Production procedure

## Phase A: implementation commit

Start from a clean checkout at:

```text
b2daa3066ed1838b213d88f5a5f10abba1d3ab35
```

Implement:

```text
single_bag_cog_prior_reseed_refinement.py
```

the conditioning helper/result structure, tests, and this implementation plan.

Do not include production outputs in the source implementation commit.

Run the tests.

Then create one source commit.

Recommended commit message:

```text
Add CoG-prior conditioned reseed refinement
```

Record:

```bash
SOURCE_COMMIT=$(git rev-parse HEAD)
```

Production outputs must be generated from this committed source revision so
their source commit is auditable.

---

# 27. Production runs

Run exactly the three baseline inputs listed in Section 4.

Use production CoG prior:

```text
mean = vehicle-model nominal CoG
std  = 0.001 m on x/y/z
```

Generate the three run IDs from Section 16.

Every production run must complete:

1. baseline validation;
2. direct Gaussian conditioning;
3. one nonlinear diagnostic evaluation at the conditioned seed;
4. complete prior-free full refinement;
5. ordinary post-fit residual-wrench diagnostics;
6. ordinary conservative-fusion covariance recomputation;
7. original/conditioned/refined comparison report.

Do not skip the successful bag.

---

# 28. Production three-bag consensus

After all three refinements complete, run a three-bag comparison using the
standard refined output directories.

Produce a consensus output under the same source-commit namespace.

Recommended run ID:

```text
three_bag_cog1mm_reseed_refinement_consensus_production_20260817
```

The consensus report must show at least:

- original three-bag scale-free parameters;
- refined three-bag scale-free parameters;
- original pairwise distances;
- refined pairwise distances;
- original cross-evaluation cost table;
- refined cross-evaluation cost table;
- original-vs-refined objective change for each bag;
- failure-refined vs original-success comparisons;
- failure-refined vs refined-success comparisons.

Do not hide a result merely because the failure refinements return to their
original solutions.

---

# 29. Required production summary table

At the end of production, produce one machine-readable and one human-readable
summary with rows:

```text
failure1
failure2
success
```

and columns at least:

```text
L_original
L_conditioned_seed_at_original_lag
L_refined
Delta_L_refined_minus_original

rotor_lag_original
rotor_lag_refined

CoG_original_x/y/z
CoG_conditioned_x/y/z
CoG_refined_x/y/z

force_effectiveness_original_1..4
force_effectiveness_conditioned_1..4
force_effectiveness_refined_1..4

J_over_m_original
J_over_m_conditioned
J_over_m_refined
```

Also report distances from each failure result to the original successful-bag
physical parameter point before and after refinement.

The comparison must use scale-invariant quantities where appropriate.

---

# 30. Primary scientific success criterion

There is intentionally no arbitrary numerical pass/fail threshold such as
"within 2 sigma".

The scientific evidence is strongest if, for both failure bags:

```math
L_{\rm refine}
\approx
L_{\rm original}
```

or lower,

while:

```math
(J/m,\ f/m,\ c)_{\rm failure,refine}
```

moves substantially toward the successful-bag physical solution.

A particularly strong result is:

```text
failure1 refined  ~ success solution
failure2 refined  ~ success solution
```

with little or no degradation of each failure bag's own strict data-only
objective.

The report must present the numerical values directly.

Do not convert this into a binary assertion unless the numbers themselves make
the conclusion clear.

---

# 31. Secondary scientific interpretation

If direct Gaussian conditioning moves the failure parameters toward the success
region but prior-free nonlinear refinement returns to the original failure
solution, report that explicitly.

That outcome means:

```text
the local covariance contains a compensation tangent,
but the nonlinear loss does not support a separate branch far enough along it.
```

If conditioning produces an extreme inertia point but nonlinear refinement
moves to a finite distinct solution with comparable loss, that is evidence that:

```text
the extreme conditioned inertia was caused by local linear extrapolation,
while the nonlinear compensation manifold bends into another physically
meaningful solution.
```

This distinction is one of the main reasons the refinement stage is mandatory.

---

# 32. Reporting requirements

Create a dedicated comparison PDF for every bag.

It should contain, at minimum:

1. CoG original / conditioned / refined against the nominal CoG and 1 mm prior;
2. four force-effectiveness values original / conditioned / refined;
3. principal inertia moments or a clear scale-free inertia summary;
4. strict objective original / conditioned-seed diagnostic / refined;
5. original-vs-refined acceleration residual diagnostics;
6. original-vs-refined rotor lag;
7. a text page stating explicitly:

```text
The CoG Gaussian prior is used only to condition the saved joint distribution
and generate a nonlinear refinement initialization. The refinement objective is
the unchanged prior-free pose-derived data objective. CoG, inertia, force
effectiveness, and rotor lag remain free during refinement.
```

Do not label the refined result "MAP".

Do not label the conditioned point "refined".

---

# 33. Metadata required in every top-level refinement output

Persist:

```text
baseline_repository_commit
baseline_case_source_commit
conditioning_source_covariance_name
conditioning_prior_role
refinement_prior_role
refinement_all_parameters_free
refinement_lag_reestimated
refinement_existing_estimator_reused
```

The production values must communicate:

```text
baseline_repository_commit =
    b2daa3066ed1838b213d88f5a5f10abba1d3ab35

conditioning_source_covariance_name =
    parameter_covariance_conservative_fusion

conditioning_prior_role =
    initialization_only

refinement_prior_role =
    none

refinement_all_parameters_free =
    true

refinement_lag_reestimated =
    true

refinement_existing_estimator_reused =
    true
```

The implementation's actual source commit is separately recorded by the normal
commit-namespaced reporting machinery.

---

# 34. Failure handling

Each bag run must be failure-isolated.

A failure in one bag must not suppress output for the other bags.

If conditioning succeeds but refinement fails, persist:

```text
conditioning.json
conditioning_arrays.npz
failure status
refinement failure stage
exception type
message
traceback
```

Do not silently fall back to the original solution.

Do not silently retry with a weaker prior.

Do not silently clip the conditioned point.

---

# 35. Source-code review checks before production

Before the production run, manually verify in the diff that:

- no prior residual was added to `single_bag_savgol_core.py`;
- no prior precision was added to `parameter_covariances`;
- no CoG fixing was added;
- no lag fixing was added;
- no successful-bag parameter value appears in failure seed construction;
- no arbitrary bounds/clipping were added;
- the baseline estimator function is called rather than duplicated;
- the conditioned covariance is not reused as the refined covariance;
- original case directories are read-only inputs.

---

# 36. Commit and push procedure

The task is not complete until the implementation and production results are
committed and pushed.

## 36.1 Source implementation commit

After implementation and tests pass:

```bash
git status
git diff --check
git add <implementation files> <tests> <this plan>
git commit -m "Add CoG-prior conditioned reseed refinement"
```

Do not include generated production outputs in this source commit.

Record the resulting source commit SHA.

## 36.2 Generate production from the committed source

Run the three production refinements and the three-bag consensus only after the
source implementation commit exists.

Verify every output records that source commit.

## 36.3 Production-results commit

After checking all production outputs:

```bash
git status
git diff --check
git add ros/examples/grape-param-estim/minimal/outputs/<SOURCE_COMMIT>/
git commit -m "Add CoG-prior reseed refinement production results"
```

Do not amend the source implementation commit after production has been
generated.

If source changes become necessary, create a new source commit and regenerate
production from the new source commit.

## 36.4 Push

Push the completed branch to the configured origin.

Use a normal non-force push:

```bash
git push origin HEAD
```

Do **not** force-push.

If the push fails, report the failure and do not claim completion.

---

# 37. Final completion report required from the implementer

After the push, report:

```text
baseline HEAD
source implementation commit
production-results commit
pushed branch/ref
test command
test result / pass count
three per-bag production directories
three-bag consensus directory
```

Then give one compact numerical table containing:

```text
original -> conditioned -> refined
```

for:

```text
objective
CoG
force effectiveness
J/m
rotor lag
```

for all three bags.

Also report:

```text
failure1 refined vs original success
failure2 refined vs original success
failure1 refined vs refined success
failure2 refined vs refined success
```

using the scale-free physical comparison and cross-evaluation metrics.

Do not make a qualitative conclusion without showing the actual numbers.

---

# 38. Definition of done

This task is complete only when all of the following are true:

- the baseline data-only results remain untouched;
- the 1 mm nominal-CoG conditioning is implemented exactly;
- the conditioned point and conditioned covariance are saved;
- the conditioned point is used only as a new initial coordinate;
- the original rotor lag is used only as an initial lag seed;
- the full ordinary prior-free estimator runs again;
- CoG is free during refinement;
- inertia is free during refinement;
- force effectiveness is free during refinement;
- rotor lag is free during refinement;
- the existing KKT common-scale gauge remains the only gauge treatment;
- the refinement result is separately saved;
- all three production bags are run;
- success is not used to construct either failure seed;
- cross-bag comparison is generated;
- existing and new tests pass;
- source implementation is committed;
- production results are committed;
- the branch is pushed to origin.

The experiment must answer the scientific question by computation rather than
by modifying the estimator until the bags agree.
