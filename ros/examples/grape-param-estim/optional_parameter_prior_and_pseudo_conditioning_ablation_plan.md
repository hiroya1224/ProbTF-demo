# Optional Gaussian Parameter Priors and Nominal Pseudo-Conditioning Ablation
## Implementation, Validation, Production, and Push Plan

## 0. Repository baseline and purpose

Repository:

```text
hiroya1224/ProbTF-demo
```

This task is defined relative to the current pushed HEAD:

```text
475bff634decae3449d10fc90738b9bfc0be0bd7
```

Commit message:

```text
Add CoG-prior reseed refinement production results
```

The preceding source commit for the reseed experiment is:

```text
9cbb734dbb5501a3b74777c5ba6aa44fbf44520f
```

The current repository already established two useful facts:

1. the normal estimator works without any parameter prior;
2. for the completed failure-1 and success reseed experiments, even a very
   different covariance-conditioned initialization returned to essentially the
   same prior-free physical solution.

The new task changes the role of prior information.

The default estimator remains:

```text
prior-free
```

and this remains the primary baseline.

When a prior JSON is explicitly supplied, the estimator adds the corresponding
Gaussian parameter factor **from the beginning of the same nonlinear
optimization and keeps that factor active throughout the optimization**.

There is no two-stage estimation in the new normal prior path.

The intended architecture is:

```text
raw pose trajectory + commands
          |
          v
existing prior-free dynamics objective
          |
          +-----------------------------+
          |                             |
          | no --prior-json             | --prior-json supplied
          |                             |
          v                             v
current estimator                current estimator
unchanged                        + Gaussian parameter factors
          |                             |
          v                             v
data-only estimate               prior-informed estimate
```

The old `single_bag_cog_prior_reseed_refinement.py` experiment is retained as a
historical/diagnostic initialization-robustness experiment. It is **not** the
new default prior implementation and it is **not** used as an input to the new
ablation.

---

# 1. Scientific objective

The central question is:

> When one physically meaningful parameter quantity is constrained very tightly
> to its vehicle-model nominal value, which other inferred quantities move to
> preserve the pose-derived rigid-body explanation?

This is deliberately similar to an ablation study.

For each identifiable physical quantity, construct a very narrow Gaussian prior
at the nominal vehicle-model value. The prior remains finite, so the variable is
still mathematically free, but it behaves approximately as if conditioned on
the nominal value.

For one pseudo-conditioned quantity \(x_j\), compare

```math
\hat x_{\rm free}
```

with

```math
\hat x_{j,\rm pseudo}
```

and record

```math
\Delta x^{(j)}
=
\hat x_{j,\rm pseudo}-\hat x_{\rm free}.
```

The resulting collection of shifts is a direct empirical map of parameter
compensation in the nonlinear estimator.

The especially important observations are:

- whether a nominal CoG component causes inertia or force effectiveness to move;
- whether nominal inertia components cause CoG or rotor effectiveness to move;
- whether nominal rotor effectiveness causes CoG or inertia to move;
- whether the same pseudo-conditioning makes the two failure bags and the
  successful bag converge toward the same scale-free physical parameter region;
- how much **data-only pose objective** must be sacrificed to enforce each
  nominal pseudo-condition.

The prior-ablation output must therefore keep the prior penalty and the
pose-derived data loss separate.

---

# 2. Fundamental interpretation

There are two distinct uses of the same implementation.

## 2.1 Ordinary optional external prior

A future user may supply a realistically calibrated Gaussian prior, for example
a CoG distribution from weighing/CAD/calibration.

That prior is ordinary external information.

## 2.2 Pseudo-conditioning ablation

The production configs introduced in this task deliberately use **very small
standard deviations**.

These standard deviations are not estimates of physical uncertainty.

They are an experimental device:

> "Hold this identifiable physical quantity extremely close to nominal, while
> leaving every parameter mathematically free, and observe how the remaining
> nonlinear estimate reorganizes."

Every pseudo-conditioning JSON must state this explicitly.

Do not report the tiny standard deviations as measured sensor/calibration
uncertainties.

---

# 3. The prior-free estimator remains the default and source of truth

The normal entry point remains:

```text
ros/examples/grape-param-estim/minimal/single_bag_savgol_estimator.py
```

The current production default remains unchanged when no prior is supplied:

```text
pose SG window                 = 1.0 s
pose SG degree                 = 5
geometric SO(3)                = enabled
gimbal source                  = measured_sg
acceleration covariance mode   = identity
rotor lag                      = estimated
continuation                   = epsilon_k = 2^-k, k=0,...,9
strict final command model     = exact strict ZOH
solver                         = custom_kkt_lm
inertia chart                  = matrix-exponential second moment
common-scale gauge             = existing exact KKT gauge
parameter prior                = none
```

The new CLI option is optional:

```text
--prior-json PATH
```

When omitted, the estimator must follow the existing prior-free numerical path.

A no-prior regression test is mandatory.

---

# 4. Do not resurrect "prior required for estimation"

The implementation must not make the prior a hidden regularizer.

The following are forbidden:

> **DO NOT add a default prior.**

> **DO NOT silently load a prior config.**

> **DO NOT add a weak numerical prior when `--prior-json` is absent.**

> **DO NOT use the prior to make the KKT solve invertible.**

> **DO NOT use the prior as an optimizer-stability patch.**

> **DO NOT change the prior-free point estimate by default.**

The conceptual order is:

```text
the data-only estimator is already valid
+
optional physical external information
```

not:

```text
the estimator needs a prior in order to work
```

This distinction must be stated in the README/report metadata.

---

# 5. Prior v1 lives on the identifiable physical quotient

The current data objective has the exact common scaling:

```math
(m,J,f_1,\ldots,f_4)
\mapsto
(\lambda m,\lambda J,\lambda f_1,\ldots,\lambda f_4).
```

The current 14-D chart common-scale direction is:

```math
v_{\rm scale}
=
(1,1,1,1,0,0,0,0,0,0,1,1,1,1)^T.
```

Therefore v1 of the prior system must operate only on scale-invariant physical
quantities:

```math
\boxed{
J/m,\qquad
c_{\rm CoG},\qquad
f_i/m
}
```

These contain exactly 13 independent named physical quantities:

```text
Jxx/m
Jyy/m
Jzz/m
Jxy/m
Jxz/m
Jyz/m

CoG_x
CoG_y
CoG_z

f1/m
f2/m
f3/m
f4/m
```

This matches the 13-D physical quotient already used for cross-bag comparison.

This choice is deliberate.

Every prior factor in v1 must satisfy:

```math
J_{\rm prior}v_{\rm scale}=0
```

up to machine precision.

The existing KKT gauge therefore remains unchanged.

---

# 6. Absolute-scale priors are explicitly out of v1 scope

Do not accept these quantities in the first schema:

```text
mass_kg
absolute inertia_kg_m2
absolute force_effectiveness
```

A prior on one of those quantities identifies the otherwise exact global scale
and therefore changes the role of the current KKT gauge.

That is a separate design problem.

If absolute-scale priors are needed later, introduce a new schema/version with
an explicit gauge policy.

Do not silently combine an absolute mass/inertia/thrust prior with the existing
fixed gauge section.

---

# 7. Physical quotient quantity definitions

Let the current decoded physical parameters be:

```math
m(q),\qquad
J(q),\qquad
c(q),\qquad
f(q).
```

Define:

```math
g_J(q)=\operatorname{symvec}\!\left(J(q)/m(q)\right)\in\mathbb R^6,
```

with fixed component order:

```text
xx, yy, zz, xy, xz, yz
```

Define:

```math
g_c(q)=c(q)\in\mathbb R^3
```

with order:

```text
x, y, z
```

and:

```math
g_f(q)=f(q)/m(q)\in\mathbb R^4
```

with order:

```text
rotor_1, rotor_2, rotor_3, rotor_4.
```

The combined reporting vector is:

```math
x(q)=
\begin{bmatrix}
g_J(q)\\
g_c(q)\\
g_f(q)
\end{bmatrix}
\in\mathbb R^{13}.
```

Use this same order everywhere in prior-ablation machine-readable output.

---

# 8. Nominal quotient values

The target `"vehicle_model_nominal"` is resolved by decoding the vehicle-model
reference parameter set, not by hard-coding values in Python.

For the current `grape_vehicle_model.json`, the nominal CoG is:

```math
c_{\rm nom}
=
(-0.002024708562282,\,
 -0.000030526578941,\,
 +0.009509749599446)\ {\rm m}.
```

The nominal force-effectiveness values are all one and the nominal mass is
approximately \(2.3516\,{\rm kg}\), so the current nominal values satisfy:

```math
(f_i/m)_{\rm nom}
\approx 0.4252428238\ {\rm kg}^{-1}.
```

The current nominal \(J/m\) is approximately:

```math
\begin{bmatrix}
2.76408097\times10^{-2} &
-3.09533934\times10^{-7} &
8.08602633\times10^{-6}\\
-3.09533934\times10^{-7} &
2.76206510\times10^{-2} &
2.51604719\times10^{-8}\\
8.08602633\times10^{-6} &
2.51604719\times10^{-8} &
5.48529694\times10^{-2}
\end{bmatrix}
\ {\rm m}^2.
```

These numerical values are useful for audit/reporting only.

The implementation must resolve nominal targets from the loaded vehicle model so
that the prior system is not tied to these literal numbers.

---

# 9. New prior module

Add a narrow dedicated module, recommended:

```text
ros/examples/grape-param-estim/minimal/single_bag_parameter_prior.py
```

Recommended responsibilities:

```text
JSON parsing / schema validation
target resolution from vehicle model
physical quotient value evaluation
exact analytic Jacobian evaluation
Gaussian whitening
prior residual/Jacobian evaluation
diagnostic serialization
```

Do not put JSON parsing into the rigid-body dynamics core.

Do not duplicate `SiParameterChart`.

Use the existing physical decoder and physical parameter Jacobian.

---

# 10. Recommended data structures

Use explicit immutable structures, for example:

```python
@dataclass(frozen=True)
class GaussianPriorFactorSpec:
    quantity: str
    components: tuple[str, ...]
    target: ...
    covariance: ...

@dataclass(frozen=True)
class ResolvedGaussianPriorFactor:
    name: str
    quantity: str
    components: tuple[str, ...]
    target: np.ndarray
    covariance: np.ndarray
    whitening: np.ndarray

@dataclass(frozen=True)
class ParameterPrior:
    name: str
    source_path: Path
    source_sha256: str
    factors: tuple[ResolvedGaussianPriorFactor, ...]

@dataclass(frozen=True)
class PriorEvaluation:
    residual: np.ndarray
    jacobian: np.ndarray
    value: np.ndarray
    target: np.ndarray
    factor_slices: ...
```

Names may differ, but the separation between JSON specification, resolved
physical target, and runtime evaluation must remain clear.

---

# 11. Prior JSON schema v1

Use a stable explicit schema marker:

```json
{
  "schema": "grape-param-estim/parameter-prior/v1",
  "name": "pseudo_condition_cog_x_nominal",
  "description": "Ablation-only tight nominal pseudo-conditioning factor.",
  "role": "pseudo_conditioning_ablation",
  "factors": [
    {
      "name": "cog_x_nominal",
      "quantity": "cog_position_body_m",
      "components": ["x"],
      "target": {
        "source": "vehicle_model_nominal"
      },
      "std": [1e-5]
    }
  ]
}
```

Supported `quantity` values in v1:

```text
cog_position_body_m
inertia_over_mass_m2
force_effectiveness_over_mass
```

Supported component names:

```text
cog_position_body_m:
    x, y, z

inertia_over_mass_m2:
    xx, yy, zz, xy, xz, yz

force_effectiveness_over_mass:
    rotor_1, rotor_2, rotor_3, rotor_4
```

Unknown quantity/component names are hard errors.

---

# 12. Prior target specification

Support two target forms.

## 12.1 Vehicle-model nominal

```json
"target": {
  "source": "vehicle_model_nominal"
}
```

The selected components are extracted from the nominal scale-free physical
quantity.

## 12.2 Explicit physical target

```json
"target": {
  "value": [ ... ]
}
```

The value must have exactly the same length as `components`.

Exactly one of:

```text
source
value
```

must be present.

No implicit unit conversion is performed.

The JSON uses SI physical units.

---

# 13. Prior covariance specification

Support either diagonal standard deviations:

```json
"std": [ ... ]
```

or a full covariance:

```json
"covariance": [
  [...],
  [...]
]
```

Exactly one must be supplied.

Requirements:

- finite entries;
- positive standard deviations;
- full covariance symmetric;
- full covariance strictly positive definite;
- dimensions exactly match selected components.

Use Cholesky/triangular solves for whitening.

Do not explicitly invert the covariance matrix.

---

# 14. Gaussian factor definition

For one resolved factor with selected physical quantity:

```math
g(q)\in\mathbb R^d,
```

target:

```math
\mu_p,
```

and covariance:

```math
R_p,
```

the factor is:

```math
r_p(q)
=
L_p^{-1}(g(q)-\mu_p),
\qquad
R_p=L_pL_p^T.
```

Its Jacobian is:

```math
J_p(q)
=
L_p^{-1}Dg(q).
```

The prior contribution to the optimization objective is:

```math
L_p(q)
=
\frac12 r_p(q)^T r_p(q).
```

For multiple factors, concatenate all prior residual rows.

---

# 15. Exact analytic Jacobians

Do not use numerical differentiation for prior factors.

The current `SiParameterChart.decode_with_jacobian()` already exposes:

```text
Dm
DJ
Dc
Df
```

in the 14-D chart.

Use them directly.

For CoG:

```math
D(c)=D c.
```

For each force-effectiveness-over-mass component:

```math
D(f_i/m)
=
\frac{Df_i}{m}
-
\frac{f_i}{m^2}Dm.
```

For each symmetric inertia-over-mass component:

```math
D(J_{ab}/m)
=
\frac{DJ_{ab}}{m}
-
\frac{J_{ab}}{m^2}Dm.
```

Add finite-difference tests only as verification of the analytic implementation.

---

# 16. Gauge-invariance test is mandatory

For every resolved v1 factor and for random valid chart coordinates, verify:

```math
\|J_p(q)v_{\rm scale}\|
```

is at machine-level numerical noise.

If a factor materially responds to the exact common-scale direction, treat that
as an implementation error.

This property is the reason v1 can reuse the current KKT gauge unchanged.

---

# 17. New CLI option

Extend:

```text
single_bag_savgol_estimator.py
```

with:

```text
--prior-json PATH
```

Default:

```text
None
```

When absent:

```text
no prior factors are created
```

and the existing estimator behavior is preserved.

When present:

1. load the vehicle model;
2. parse/resolve the JSON against that vehicle model;
3. construct the prior factor set;
4. pass it to the existing nonlinear estimator;
5. persist the path and SHA256 of the prior JSON.

The bag JSON remains minimal and unchanged.

Do not put prior content into `bag_jsons/`.

---

# 18. One estimation only

The new prior-enabled estimator must not perform:

```text
data-only fit
-> Gaussian conditioning
-> second fit
```

The new normal prior path is:

```text
normal initial coordinate
+
normal initial lag
+
data residual
+
optional prior residual
-> one full nonlinear estimation
```

The existing reseed-refinement code can remain in the repository as a diagnostic
experiment, but the prior-enabled estimator must not call it.

---

# 19. Prior must be active in every optimization stage

The factor must be present in:

```text
smooth lag continuation
strict-ZOH cell physical refinement
final strict physical refinement
```

If the objective is evaluated in 15-D `[q, rotor_lag]` coordinates, append a
zero lag column:

```math
J_{p,\rm global}
=
\begin{bmatrix}
J_p & 0
\end{bmatrix}.
```

The prior therefore does not directly constrain rotor lag.

Rotor lag remains freely estimated and can move indirectly because the physical
solution changes.

---

# 20. Objective composition

For the ordinary data objective residual:

```math
r_d(q,\delta),
```

and optional prior residual:

```math
r_p(q),
```

the optimizer receives:

```math
r_{\rm total}
=
\begin{bmatrix}
r_d\\
r_p
\end{bmatrix},
```

and:

```math
J_{\rm total}
=
\begin{bmatrix}
J_d\\
J_p
\end{bmatrix}.
```

Thus:

```math
L_{\rm total}
=
L_{\rm data}
+
L_{\rm prior}.
```

Do not scale the prior by sample count.

Do not divide the prior residual by \(\sqrt N\).

Do not duplicate the prior once per pose sample.

The prior is one external factor; the pose samples are the data factors.

---

# 21. Data loss and prior loss must always be reported separately

This is mandatory for the pseudo-conditioning experiment.

At the final solution save:

```text
data_objective_sum
prior_objective_sum
total_objective_sum
```

with:

```math
L_{\rm total}=L_{\rm data}+L_{\rm prior}.
```

Also save per-factor:

```text
factor_name
physical_value
physical_target
physical_error
standardized_residual
factor_objective
```

For comparing different pseudo-conditioning cases, the main scientific loss is:

```math
\Delta L_{\rm data}
=
L_{\rm data,prior}
-
L_{\rm data,free}.
```

Do not compare pseudo-conditioning cases only by total loss because the tiny
prior standard deviations deliberately dominate violations of their target.

---

# 22. Default prior-free metrics remain comparable

`common_evaluation` must continue to report the same pose-derived quantities as
before.

The presence of a prior must not redefine:

```text
identity_objective_sum
specific acceleration RMSE
angular acceleration RMSE
raw residual wrench
```

Those quantities remain data/model diagnostics evaluated at the final
prior-informed parameter point.

Add separate prior/total-optimization sections rather than overloading old keys.

---

# 23. Prior and uncertainty outputs

Do not silently replace the semantics of the existing covariance fields.

At a prior-informed solution:

```text
parameter_covariance_naive
parameter_covariance_overlap_corrected
parameter_covariance_wrench_corrected
parameter_covariance_conservative_fusion
```

must remain the existing **data-derived post-fit covariance constructions
evaluated at that final parameter point** unless explicitly renamed.

Add a new prior-information diagnostic:

```math
A_{\rm prior}=J_p^TJ_p.
```

Add:

```math
A_{\rm total}=A_{\rm data}+A_{\rm prior}.
```

On the existing exact gauge section with basis \(P\), additionally report:

```math
C_{\rm local,prior}
=
P
\left[
P^T A_{\rm total}P
\right]^\dagger
P^T.
```

Recommended output name:

```text
parameter_covariance_prior_augmented_local_curvature
```

This is a local quadratic covariance implied by the current optimization metric
plus the Gaussian factor.

It must be labeled as:

```text
local prior-augmented curvature / pseudo-posterior diagnostic
```

and not silently substituted for the existing robust/conservative covariance.

No prior-aware sandwich construction is required in this task.

---

# 24. Prior metadata in `result.json`

When no prior is supplied:

```json
"prior": {
  "active": false
}
```

When a prior is supplied, include at least:

```text
active
schema
name
role
source_path
source_sha256
resolved_factors
prior_residual
prior_objective_sum
prior_information_matrix
total_local_curvature
parameter_covariance_prior_augmented_local_curvature
```

Persist all resolved targets and covariance/std values.

A report must be reproducible from outputs without rereading the source JSON.

---

# 25. Prior arrays in `arrays.npz`

When a prior is active, save:

```text
prior_residual
prior_jacobian
prior_information_matrix
prior_augmented_local_curvature
parameter_covariance_prior_augmented_local_curvature
```

Also save a stable factor-offset table in JSON so residual rows can be mapped
back to factor names.

No prior arrays are required in no-prior output beyond an optional empty marker.

---

# 26. New config directory

Create:

```text
ros/examples/grape-param-estim/minimal/config/
```

Use:

```text
config/
  priors/
    README.md
    pseudo_conditioning/
      scalar/
      group/
  prior_ablation/
    nominal_pseudo_conditioning.json
```

Do not scatter prior JSON files under the repository root or `bag_jsons/`.

---

# 27. Tight pseudo-conditioning standard deviations

Use deliberately small finite values.

These are ablation strengths, not physical uncertainty estimates.

For CoG:

```text
std = 1.0e-5 m
```

i.e. 10 micrometres.

For each \(J/m\) component:

```text
std = 1.0e-6 m^2
```

For each \(f_i/m\) component:

```text
std = 1.0e-5 kg^-1
```

These values are intentionally strong enough to behave approximately like
conditioning while avoiding exact hard constraints.

Do not call them "measured standard deviations".

Every pseudo-conditioning config must contain:

```json
"role": "pseudo_conditioning_ablation"
```

and a description stating that the std is intentionally artificial/tight.

---

# 28. Scalar pseudo-conditioning configs

Create exactly these 13 primary scalar configs.

## CoG

```text
config/priors/pseudo_conditioning/scalar/
  cog_x_nominal.json
  cog_y_nominal.json
  cog_z_nominal.json
```

Each constrains one selected component of:

```text
cog_position_body_m
```

to `vehicle_model_nominal` with `1e-5 m` std.

## Inertia over mass

```text
config/priors/pseudo_conditioning/scalar/
  inertia_over_mass_xx_nominal.json
  inertia_over_mass_yy_nominal.json
  inertia_over_mass_zz_nominal.json
  inertia_over_mass_xy_nominal.json
  inertia_over_mass_xz_nominal.json
  inertia_over_mass_yz_nominal.json
```

Each constrains one selected component of:

```text
inertia_over_mass_m2
```

to nominal with `1e-6 m^2` std.

## Force effectiveness over mass

```text
config/priors/pseudo_conditioning/scalar/
  force_over_mass_rotor_1_nominal.json
  force_over_mass_rotor_2_nominal.json
  force_over_mass_rotor_3_nominal.json
  force_over_mass_rotor_4_nominal.json
```

Each constrains one selected component of:

```text
force_effectiveness_over_mass
```

to nominal with `1e-5 kg^-1` std.

---

# 29. Group pseudo-conditioning configs

Also create:

```text
config/priors/pseudo_conditioning/group/
  cog_all_nominal.json
  inertia_over_mass_all_nominal.json
  force_over_mass_all_nominal.json
```

These constrain all members of one physical family simultaneously, with
diagonal covariance using the same pseudo-conditioning stds.

Optionally create:

```text
all_quotient_nominal.json
```

as a diagnostic config that contains all three groups.

Do not include `all_quotient_nominal.json` in the primary scientific matrix
unless runtime is acceptable; it is mainly a nominal-model consistency check.

---

# 30. Prior config README

`config/priors/README.md` must explain:

- supported schema;
- supported quantities/components;
- SI units;
- nominal target resolution;
- std vs covariance;
- exact common-scale gauge restriction;
- the distinction between real external priors and pseudo-conditioning configs;
- the artificial tight std values used by the ablation;
- an example command.

---

# 31. Ablation manifest

Create:

```text
config/prior_ablation/nominal_pseudo_conditioning.json
```

It explicitly lists the cases in a fixed order.

Recommended structure:

```json
{
  "schema": "grape-param-estim/prior-ablation/v1",
  "name": "nominal_pseudo_conditioning",
  "include_prior_free_baseline": true,
  "cases": [
    {
      "case_name": "cog_x_nominal",
      "prior_json": "../priors/pseudo_conditioning/scalar/cog_x_nominal.json"
    }
  ]
}
```

Do not discover scientific cases by an uncontrolled filesystem glob.

The manifest is the experiment definition.

---

# 32. New ablation runner

Add:

```text
minimal/single_bag_prior_ablation.py
```

Responsibilities:

1. accept one ordinary bag JSON;
2. accept the vehicle model;
3. accept the prior-ablation manifest;
4. run the no-prior baseline;
5. run every prior case independently;
6. use the exact same normal estimator entry path;
7. isolate failures per case;
8. generate one per-bag summary.

Every case must start from the same ordinary default initialization.

Do not warm-start one pseudo-conditioning case from another case.

Do not use the prior-free solution as an initialization unless the normal
estimator itself already does so.

The scientific difference between cases must be only the supplied prior JSON.

---

# 33. Reuse `run_estimator`

The ablation runner must call the same:

```python
run_estimator(...)
```

used by ordinary single-bag estimation.

Do not duplicate the nonlinear solver.

Do not duplicate the dynamics equations.

Do not implement a special "conditioning solver".

A pseudo-conditioning case is simply:

```text
normal estimator + a very tight optional prior factor
```

---

# 34. Ablation failure isolation

Each case must have its own output directory and top-level exception boundary.

One failed prior case must not stop later cases.

Save:

```text
status
failure_stage
exception_type
message
traceback
```

for failed cases.

This matters because tight pseudo-conditioning may deliberately drive the
remaining free parameters into difficult nonlinear regions.

---

# 35. Preserve a completed point estimate if post-fit uncertainty fails

The current pushed reseed experiment demonstrated that a completed strict
optimization can be lost because a later residual-wrench covariance closure
check fails.

The new ablation must not lose the optimized point for that reason.

Refactor the post-fit boundary minimally so that:

```text
final strict point estimate
final rotor lag
data objective
prior objective
total objective
ridge / raw Jacobian information if already available
```

are preserved even if a later post-fit uncertainty diagnostic fails.

Requirements:

- do not relax the existing closure-map tolerance;
- do not silently fabricate covariance;
- mark unavailable post-fit products explicitly;
- preserve the optimized point and objective;
- continue the ablation summary using point-estimate fields.

Recommended status separation:

```text
optimization_status
postfit_uncertainty_status
overall_case_status
```

A post-fit uncertainty failure is not an optimizer failure.

---

# 36. Per-case outputs

Each prior case should retain the normal estimator files where available:

```text
result.json
arguments.json
arrays.npz
report.pdf
residual_wrench.pdf
status.json
timing.json
```

The case output must additionally expose the resolved prior diagnostics.

The no-prior baseline should use the same reporting code.

---

# 37. Per-bag ablation summary vector

For every completed/point-estimate-valid case, compute:

```math
x=
[
J_{xx}/m,
J_{yy}/m,
J_{zz}/m,
J_{xy}/m,
J_{xz}/m,
J_{yz}/m,
c_x,c_y,c_z,
f_1/m,f_2/m,f_3/m,f_4/m
]^T.
```

Let:

```math
x_0
```

be the prior-free estimate.

For case \(j\), save:

```math
\Delta x^{(j)}=x_j-x_0.
```

This is the primary compensation-response output.

---

# 38. Per-bag machine-readable summary

Create:

```text
prior_ablation.json
prior_ablation.npz
```

with at least:

```text
case_names
prior_config_paths
prior_config_sha256

x_prior_free
x_per_case
delta_x_per_case

data_objective_prior_free
data_objective_per_case
delta_data_objective_per_case

prior_objective_per_case
total_objective_per_case

rotor_lag_prior_free
rotor_lag_per_case

prior_target_error_per_case
prior_standardized_residual_per_case

optimization_status_per_case
postfit_uncertainty_status_per_case
```

Preserve units/component labels.

---

# 39. Per-bag PDF report

Create:

```text
prior_ablation.pdf
```

Do not put mixed-unit quantities on one unlabeled heatmap.

Recommended pages:

## Page 1: case overview

For every case:

```text
prior target
achieved value
target error / std
data loss increase
prior loss
rotor lag
status
```

## Page 2: CoG response

Rows = prior cases.

Columns:

```text
Delta CoG_x [mm]
Delta CoG_y [mm]
Delta CoG_z [mm]
```

## Page 3: inertia-over-mass response

Rows = prior cases.

Columns:

```text
Delta Jxx/m
Delta Jyy/m
Delta Jzz/m
Delta Jxy/m
Delta Jxz/m
Delta Jyz/m
```

Units:

```text
m^2
```

## Page 4: force-over-mass response

Rows = prior cases.

Columns:

```text
Delta f1/m
Delta f2/m
Delta f3/m
Delta f4/m
```

Units:

```text
kg^-1
```

## Page 5: data-fit price of conditioning

Plot:

```math
\Delta L_{\rm data}
```

for all cases.

This page is crucial.

A large parameter rearrangement with almost no increase in data loss is strong
evidence of compensation/ambiguity.

## Page 6: prior target satisfaction

Plot standardized final prior residuals to verify that each tight factor really
acted as a pseudo-condition.

---

# 40. Three-bag orchestration

Add a simple orchestrator, recommended:

```text
minimal/run_prior_ablation.sh
```

or a Python equivalent.

Run the same manifest independently on:

```text
single_rosbag_1.json
single_rosbag_2.json
single_rosbag_succeeded.json
```

The three bag jobs may run concurrently.

Case-level concurrency may reuse the existing failure-isolated ablation pattern.

Do not allow numerical-thread oversubscription.

---

# 41. Three-bag prior-ablation aggregation

Add:

```text
minimal/three_bag_prior_ablation_summary.py
```

It consumes only completed outputs from the three per-bag prior ablations.

For every prior case, compare the three physical point estimates in:

```math
J/m,\quad c,\quad f/m.
```

The primary question is:

> Does pseudo-conditioning the same nominal physical quantity reduce the
> disagreement among failure1, failure2, and success?

---

# 42. Cross-bag point-spread metrics

For each prior case compute simple transparent point-spread summaries.

For CoG:

```math
S_c
=
\sqrt{
\frac{1}{3}
\sum_b
\|c_b-\bar c\|^2
}.
```

For inertia-over-mass:

```math
S_J
=
\sqrt{
\frac{1}{3}
\sum_b
\|J_b/m_b-\overline{J/m}\|_F^2
}.
```

For force-over-mass:

```math
S_f
=
\sqrt{
\frac{1}{3}
\sum_b
\|f_b/m_b-\overline{f/m}\|^2
}.
```

Also save all pairwise Euclidean family distances.

These metrics are intentionally simple and do not depend on the current
controversial covariance calibration.

Do not reduce the entire experiment to one scalar score.

---

# 43. Cross-evaluation remains valuable

Where computationally practical, use the existing cross-bag physical
cross-evaluation machinery.

For each prior case:

- take the physical solution from each source bag;
- evaluate it on each target bag;
- profile only the target bag's rotor lag, exactly as current cross-evaluation
  does;
- compare target-bag data loss with that target bag's own optimum for the same
  prior configuration.

This reveals whether a prior causes the three bags to move to parameters that
actually explain one another's pose trajectories.

If runtime is too large for every case, make cross-evaluation a second explicit
phase after the primary scalar-ablation results identify the most informative
conditioning factors.

Do not silently choose cases after viewing success unless the selection rule is
reported.

---

# 44. Three-bag summary report

Create:

```text
prior_ablation_three_bag.json
prior_ablation_three_bag.pdf
```

For each case show:

```text
Delta L_data for failure1/failure2/success
CoG spread S_c
J/m spread S_J
f/m spread S_f
```

Also show the three actual parameter vectors, not only spread metrics.

Recommended key visualization:

```text
case on y-axis
bag on line/color/group
estimated parameter family on x-axis
```

with separate panels for CoG, J/m, and f/m.

---

# 45. Primary ablation cases and count

Primary scalar cases:

```text
13
```

Primary group cases:

```text
3
```

Prior-free baseline:

```text
1
```

Total primary cases per bag:

```text
17
```

Across three bags:

```text
51 estimator runs
```

The optional all-quotient nominal diagnostic is not counted above.

Do not add unrelated solver/covariance/SG ablations to this experiment.

---

# 46. Interpretation of a pseudo-conditioning case

For one prior target \(x_j=x_{j,\rm nom}\), inspect two things.

## 46.1 Compensation magnitude

```math
\Delta x^{(j)}.
```

Large movement in another component \(x_i\) means the nonlinear estimator can
transfer explanatory responsibility between \(x_j\) and \(x_i\).

## 46.2 Data-fit price

```math
\Delta L_{\rm data}^{(j)}.
```

If:

```text
large parameter movement
+
small data-loss increase
```

then the compensation is strongly supported by the pose trajectory.

If:

```text
large data-loss increase
```

then the nominal pseudo-condition is genuinely inconsistent with the pose
trajectory under the current model.

Do not infer ambiguity from parameter movement alone.

---

# 47. Meaning of the group cases

The group priors answer different questions.

`cog_all_nominal.json` asks:

> If CoG is essentially known, where do inertia and rotor effectiveness go?

`inertia_over_mass_all_nominal.json` asks:

> If scale-free inertia is essentially known, where do CoG and rotor
> effectiveness go?

`force_over_mass_all_nominal.json` asks:

> If rotor force-effectiveness ratios are essentially known, where do CoG and
> inertia go?

These are particularly useful for explaining model-compensation mechanisms in
the dissertation/paper.

---

# 48. No success-bag information enters a prior

Every pseudo-conditioning target comes from:

```text
vehicle_model_nominal
```

only.

The successful bag is not used to define:

```text
target mean
std
factor selection
initial value
```

for failure bags.

This is mandatory.

The successful bag is only another dataset on which the same predefined
ablation manifest is run.

---

# 49. The old reseed experiment remains separate

Do not delete:

```text
single_bag_cog_prior_reseed_refinement.py
three_bag_cog_prior_reseed_consensus.py
```

or their committed outputs.

They now serve as evidence that the prior-free optimizer can recover its usual
solution from a substantially displaced initialization.

Do not call those scripts from the new prior implementation.

Do not mix their conditioned covariance with the new prior factor.

---

# 50. Tests: JSON/schema

Add a dedicated test module, recommended:

```text
minimal/tests/test_parameter_prior.py
```

Test:

- valid scalar CoG factor;
- valid scalar J/m factor;
- valid scalar f/m factor;
- valid group factor;
- explicit target value;
- nominal target resolution;
- diagonal std;
- full covariance;
- invalid schema;
- unknown quantity;
- unknown component;
- duplicate component;
- dimension mismatch;
- non-positive std;
- non-SPD covariance;
- both std and covariance supplied;
- neither supplied.

---

# 51. Tests: analytic Jacobians

For random moderate 14-D coordinates, compare the analytic prior Jacobian with
central finite differences for:

```text
CoG x/y/z
all six J/m components
all four f/m components
```

Use a numerical test tolerance appropriate to the matrix exponential chart.

The production implementation remains analytic.

---

# 52. Tests: exact gauge invariance

For every supported v1 quantity verify:

```math
J_pv_{\rm scale}\approx0.
```

Also verify that adding any v1 prior does not change the exact scale-gauge
direction of the combined objective.

No projection should be needed to make the prior gauge-invariant.

---

# 53. Tests: objective composition

For a synthetic/current problem verify:

```math
r_{\rm total}
=
[r_d;r_p]
```

and:

```math
J_{\rm total}
=
[J_d;J_p].
```

Verify the global lag column of the prior block is exactly zero.

Verify the prior is present in both:

```text
smooth global objective
strict physical objective
```

paths.

---

# 54. Tests: prior-free regression

This is mandatory.

With:

```text
prior_json = None
```

verify:

- residual length is unchanged;
- residual values are unchanged;
- Jacobian shape/values are unchanged;
- no prior objective is added;
- the same initial coordinate is used;
- the same KKT gauge is used;
- the same lag search path is used.

Where feasible, reproduce a known deterministic small test result exactly or to
machine-level tolerance.

Do not weaken existing tests.

---

# 55. Tests: pseudo-conditioning behavior

For a small controlled synthetic problem:

1. run without prior;
2. apply a tight nominal prior to one quotient quantity;
3. verify the selected quantity moves toward the target;
4. verify other parameters remain free to move;
5. verify the factor is finite, not a hard fixed-coordinate implementation.

Do not test by checking for exact equality to the target.

---

# 56. Tests: config inventory

Load every JSON under:

```text
config/priors/pseudo_conditioning/
```

and verify:

- schema validity;
- unique case name;
- `role == pseudo_conditioning_ablation`;
- nominal target source;
- expected quantity/component;
- expected tight std;
- gauge invariance after resolution.

Load the ablation manifest and verify it references every required primary case
exactly once.

---

# 57. Tests: ablation case independence

Verify:

- every case starts from the same baseline estimator initialization;
- a previous case result is never copied into the next case initial coordinate;
- only `prior_json` differs between prior cases;
- the prior-free case has no prior.

---

# 58. Tests: post-fit diagnostic failure isolation

Create/mock a case in which:

```text
optimization completes
post-fit uncertainty raises
```

and verify that the output still contains:

```text
final parameter point
final lag
data objective
prior objective
total objective
optimization_status = completed
postfit_uncertainty_status = failed
```

Do not suppress the diagnostic exception from metadata.

---

# 59. Reporting additions to ordinary estimator

When prior is active, add one prior page to the ordinary report showing:

```text
factor targets
final values
target error
std
standardized residual
factor objective
data objective
prior objective
total objective
```

Also state:

```text
The parameter prior is optional external information. The default estimator is
prior-free.
```

For a pseudo-conditioning config, additionally state:

```text
This configuration uses an intentionally tight artificial standard deviation
for an ablation-style pseudo-conditioning experiment; it is not a calibrated
physical uncertainty.
```

---

# 60. README update

Update:

```text
minimal/README.md
```

with:

- no-prior default command;
- prior-enabled command;
- config directory;
- v1 gauge-invariant supported quantities;
- pseudo-conditioning ablation command;
- the difference between ordinary prior and pseudo-conditioning.

Do not describe prior as required for convergence.

---

# 61. Recommended commands

Normal prior-free use remains:

```bash
python3 minimal/single_bag_savgol_estimator.py \
  --bag-json minimal/bag_jsons/single_rosbag_1.json \
  --vehicle-model minimal/grape_vehicle_model.json
```

Prior-enabled example:

```bash
python3 minimal/single_bag_savgol_estimator.py \
  --bag-json minimal/bag_jsons/single_rosbag_1.json \
  --vehicle-model minimal/grape_vehicle_model.json \
  --prior-json minimal/config/priors/pseudo_conditioning/group/cog_all_nominal.json
```

Ablation example:

```bash
python3 minimal/single_bag_prior_ablation.py \
  --bag-json minimal/bag_jsons/single_rosbag_1.json \
  --vehicle-model minimal/grape_vehicle_model.json \
  --manifest minimal/config/prior_ablation/nominal_pseudo_conditioning.json
```

---

# 62. Production input bags

Run the primary 17-case ablation on:

```text
minimal/bag_jsons/single_rosbag_1.json
minimal/bag_jsons/single_rosbag_2.json
minimal/bag_jsons/single_rosbag_succeeded.json
```

Use the same production defaults as the current prior-free estimator.

Do not change bag intervals.

Do not choose different prior configurations by bag.

---

# 63. Output namespace

After the source implementation commit is created, use:

```bash
SOURCE_COMMIT=$(git rev-parse HEAD)
```

and write production outputs under:

```text
minimal/outputs/<SOURCE_COMMIT>/prior_ablation/
```

Recommended per-bag run IDs:

```text
single_rosbag_1_nominal_pseudo_conditioning_production_20260817
single_rosbag_2_nominal_pseudo_conditioning_production_20260817
single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817
```

Recommended aggregate run ID:

```text
three_bag_nominal_pseudo_conditioning_production_20260817
```

---

# 64. Source implementation commit

Implement:

```text
single_bag_parameter_prior.py
single_bag_prior_ablation.py
three_bag_prior_ablation_summary.py
config/...
tests/...
README updates
this plan
```

and only the minimal necessary modifications to:

```text
single_bag_savgol_estimator.py
single_bag_savgol_core.py
single_bag_savgol_reports.py
single_bag_savgol_covariance.py
```

Do not include production outputs in the source commit.

Recommended commit message:

```text
Add optional physical parameter prior factors
```

Run:

```bash
git diff --check
python3 -m pytest minimal/tests
```

from:

```text
ros/examples/grape-param-estim/
```

or use the repository's exact established equivalent if different.

Commit only after tests pass.

---

# 65. Production execution

After the source commit exists:

1. run all three per-bag primary prior ablations;
2. verify the prior-free case reproduces the current normal estimator behavior;
3. aggregate the three bag results;
4. inspect failure-isolated cases;
5. retain point estimates even when a post-fit uncertainty diagnostic fails;
6. generate per-bag and three-bag PDFs/JSON summaries.

Do not tune tight std values after looking at which config gives the desired
cross-bag agreement.

If the predefined std values expose numerical problems, report those problems
and make an implementation-level numerical correction only if it does not
change the scientific prior definition.

---

# 66. Production-results commit

After production outputs are inspected:

```bash
git status
git diff --check
git add ros/examples/grape-param-estim/minimal/outputs/<SOURCE_COMMIT>/
git commit -m "Add nominal prior pseudo-conditioning results"
```

Do not amend the source implementation commit after generating production.

If source changes are needed, create a new source commit and regenerate outputs
under the new commit namespace.

---

# 67. Push

Push the completed branch normally:

```bash
git push origin HEAD
```

Do not force-push.

The task is not complete until both the implementation and production-result
commits are pushed.

---

# 68. Final implementer report

After push, report:

```text
baseline commit:
    475bff634decae3449d10fc90738b9bfc0be0bd7

source implementation commit:
    <sha>

production results commit:
    <sha>

pushed branch/ref:
    <branch>

tests:
    <command and pass count>
```

Also report the three per-bag output directories and aggregate output directory.

---

# 69. Final scientific summary required

Produce one compact table for all primary cases with:

```text
case
conditioned quantity/component
target
achieved value
target error / pseudo std

failure1 Delta L_data
failure2 Delta L_data
success Delta L_data

failure1/2/success CoG
failure1/2/success J/m
failure1/2/success f/m

S_c
S_J
S_f
```

Then identify, numerically:

- which nominal pseudo-condition produces the largest change in CoG;
- which produces the largest change in J/m;
- which produces the largest change in f/m;
- which most reduces failure-vs-success point disagreement;
- what data-loss increase that reduction costs.

Do not make a qualitative claim without the actual numbers.

---

# 70. Definition of done

The task is complete only when:

- prior-free remains the default;
- `--prior-json` is optional;
- no hidden prior exists;
- v1 prior quantities are exactly the gauge-invariant physical quotient
  quantities;
- prior JSON supports nominal and explicit targets;
- prior JSON supports diagonal std and full covariance;
- prior Jacobians are analytic;
- prior factors are active through smooth and strict optimization;
- all currently estimated physical quantities remain free;
- rotor lag remains free;
- the existing KKT gauge remains unchanged;
- data/prior/total objectives are separately reported;
- existing covariance semantics are not silently changed;
- prior-augmented local curvature is separately reported;
- all 13 scalar pseudo-conditioning configs exist;
- all 3 group configs exist;
- the ablation manifest is explicit;
- every ablation case starts independently from the same normal initialization;
- all three bags use the identical predefined case set;
- completed point estimates survive post-fit uncertainty failures;
- per-bag compensation-response summaries are produced;
- three-bag aggregation is produced;
- existing and new tests pass;
- source implementation is committed;
- production outputs are committed;
- the branch is pushed.

The experiment should reveal nonlinear explanatory substitution among the
identifiable physical parameters without making prior information a prerequisite
for the estimator itself.
