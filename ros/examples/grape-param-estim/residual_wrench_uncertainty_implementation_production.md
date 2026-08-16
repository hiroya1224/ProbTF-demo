# Residual-Wrench Uncertainty: Mandatory Implementation and Production Procedure

## 0. Purpose and scope

This document is a **mandatory implementation specification** for the single-bag Savitzky--Golay rigid-body parameter estimator in:

```text
ros/examples/grape-param-estim/minimal/
```

Repository:

```text
hiroya1224/ProbTF-demo
```

Starting point for this work:

```text
1cedae934fa5f8521f85fdba68615c3b37a4f9d3
```

The purpose of this revision is **only** to restore and promote residual-wrench diagnostics and to propagate the observed residual-wrench fluctuation into post-fit parameter uncertainty.

The residual wrench is scientifically more important here than another ablation sweep. Therefore:

> **DO NOT implement or run a new ablation study in this task.**

> **DO NOT redesign the point estimator.**

> **DO NOT change the default SG window, gimbal handling, rotor-lag continuation, strict-ZOH cell refinement, KKT gauge, physical parameter chart, objective, or solver.**

> **DO NOT add a residual-wrench penalty to the optimization objective.**

> **DO NOT fit an additional free wrench trajectory inside the parameter optimizer.**

> **DO NOT remove any existing outputs or plots.**

The required revision is a post-fit uncertainty and reporting revision around the already-computed raw residual wrench.

After implementation, the default estimator must be run in the production environment for exactly the three single-bag JSON inputs listed in Section 13, the outputs must be inspected, and both source changes and production outputs must be committed and pushed.

---

# 1. Existing quantity that is the center of this task

The current estimator already computes, at every final strict-ZOH evaluation time,

```python
raw_residual_wrench = required_wrench - modeled_wrench
```

with components

```text
[Fx, Fy, Fz, Tx, Ty, Tz].
```

The current Newton--Euler closure is

```math
F_k^{res}
=
m\left(
r_{s,k}
+
\ell \times r_{\alpha,k}
\right),
```

```math
\tau_k^{res}
=
Jr_{\alpha,k},
```

where

```math
r_k
=
\begin{bmatrix}
r_{s,k}\\
r_{\alpha,k}
\end{bmatrix}
=
\begin{bmatrix}
s_k^{obs}-\hat s_k\\
\alpha_k^{obs}-\hat\alpha_k
\end{bmatrix},
```

and

```math
\ell
=
p_{\mathrm{pose\ sensor}/B}-c_{\mathrm{CoG}}.
```

This exact closure is already tested numerically to machine precision and **must remain unchanged**.

The existing `raw_residual_wrench` array must remain available in `arrays.npz`.

---

# 2. Critical gauge issue: the scientifically reported wrench must use a fixed mass gauge

The common-scale gauge is exact:

```math
(m,J,f_1,\ldots,f_4)
\mapsto
(a m,aJ,af_1,\ldots,af_4).
```

Acceleration predictions are invariant under this transformation, but wrench values scale by `a`.

Therefore a residual wrench expressed in N / Nm is **not scientifically comparable across bags unless the common scale is fixed**.

For all new residual-wrench scientific diagnostics in this task, fix the scale by setting mass to the vehicle-model nominal mass:

```math
m_{\mathrm{fixed}}
=
m_{\mathrm{nominal}}.
```

Let the final estimator-gauge mass be `m_hat`. Define

```math
a_m
=
\frac{m_{\mathrm{nominal}}}{m_{\mathrm{hat}}}.
```

The nominal-mass-gauge residual wrench is

```math
w_{k,\mathrm{massfix}}^{res}
=
a_m\,w_k^{res}.
```

The same scale factor applies to `required_wrench` and `modeled_wrench`.

The implementation may obtain this either by:

1. exact analytic scaling of the final wrench arrays, or
2. regauging the chart coordinate by the exact common-scale direction and re-evaluating at the same strict lag.

Whichever implementation is used, add a test that the two constructions agree at machine precision.

## Required naming

Do **not** silently replace the existing estimator-gauge quantity.

Preserve:

```text
raw_residual_wrench
```

and add explicit nominal-mass-gauge arrays, e.g.

```text
residual_wrench_nominal_mass_gauge
modeled_wrench_nominal_mass_gauge
required_wrench_nominal_mass_gauge
residual_wrench_mass_gauge_scale
residual_wrench_fixed_mass_kg
```

The main scientific report must use `residual_wrench_nominal_mass_gauge`.

---

# 3. Separate the mean wrench from the wrench fluctuation

For one bag, let

```math
w_k
=
w_{k,\mathrm{massfix}}^{res}
\in\mathbb R^6,
\qquad
k=1,\ldots,N.
```

Compute the empirical mean

```math
\bar w
=
\frac1N\sum_{k=1}^N w_k.
```

Define centered residual wrench

```math
\tilde w_k
=
w_k-\bar w.
```

Compute the full empirical covariance

```math
S_w
=
\frac1{N-1}
\sum_{k=1}^N
\tilde w_k\tilde w_k^\top.
```

`S_w` is full `6 x 6`.

No diagonal assumption is allowed.

Force--force, torque--torque, and force--torque coupling must be retained.

The mean and fluctuation have different interpretations:

- `bar_w`: systematic mean residual / mean unmodelled wrench diagnostic;
- `S_w`: observed time fluctuation around that mean.

The mean must **not** be inserted into parameter covariance as if it were zero-mean noise.

---

# 4. Do not double-count SG uncertainty

This is mandatory.

The raw residual wrench fluctuation contains both:

1. fluctuation produced by the SG-derived observation uncertainty, and
2. additional model discrepancy / unmodelled physical wrench fluctuation.

The current parameter covariance already propagates the SG observation covariance, including SG-window overlap.

Therefore:

> **DO NOT simply add the full empirical residual-wrench covariance `S_w` to the existing SG sandwich covariance.**

That would double-count the SG contribution.

The implementation must explicitly estimate the SG-predicted contribution in wrench space and subtract it before constructing the additional model-discrepancy covariance.

---

# 5. Map SG covariance into wrench space

Use the **reference full SG covariance**, not the optimization covariance mode.

This is essential because the default optimization mode is `identity`.

For each time index `k`, use

```text
dataset.reference_covariance.local_sigma_z[k]
```

as

```math
\Sigma^{SG}_{z,k}
=
\operatorname{Cov}
\begin{bmatrix}
s_k\\
\alpha_k
\end{bmatrix}.
```

For the nominal-mass-gauge final physical parameters define

```math
G
=
\begin{bmatrix}
mI_3 & m[\ell]_\times\\
0 & J
\end{bmatrix},
```

so that the exact closure can be written

```math
w_k^{res}
=
G r_k.
```

Here `m` and `J` are the **nominal-mass-gauge** mass and inertia.

Then the SG-predicted marginal wrench covariance is

```math
\Sigma^{SG}_{w,k}
=
G\Sigma^{SG}_{z,k}G^\top.
```

Save the full per-time array

```text
residual_wrench_sg_covariance_per_time
```

with shape

```text
(N, 6, 6).
```

Also save its arithmetic mean

```math
\bar\Sigma_w^{SG}
=
\frac1N
\sum_k
\Sigma^{SG}_{w,k}.
```

as

```text
residual_wrench_sg_covariance_mean
```

with shape `(6, 6)`.

---

# 6. Estimate the additional residual-wrench model-discrepancy covariance

Define the raw method-of-moments excess covariance

```math
Q_{w,\mathrm{raw}}
=
S_w
-
\bar\Sigma_w^{SG}.
```

Save this matrix **before any PSD correction**:

```text
residual_wrench_excess_covariance_raw
```

and save its symmetric eigenvalues:

```text
residual_wrench_excess_covariance_raw_eigenvalues
```

This pre-projection quantity is scientifically important.

If it has substantially negative eigenvalues, that indicates that the working SG covariance and the empirical residual fluctuation are inconsistent in those directions. That information must not be hidden.

## PSD projection

The actual additional model-discrepancy covariance used for uncertainty propagation must be the nearest PSD matrix under eigenvalue truncation:

```math
Q_w
=
V\,\operatorname{diag}(\max(\lambda_i,0))\,V^\top
```

for the eigendecomposition of the symmetrized `Q_w_raw`.

This is a constrained covariance estimate, not a hand-selected ridge threshold.

Use zero as the physical boundary.

Machine-scale negative eigenvalues may be treated as zero.

If a negative eigenvalue is appreciably larger than machine roundoff, record it explicitly in diagnostics.

Save:

```text
residual_wrench_model_discrepancy_covariance
residual_wrench_model_discrepancy_eigenvalues
residual_wrench_model_discrepancy_std
residual_wrench_model_discrepancy_correlation
```

Also save:

```text
residual_wrench_total_empirical_covariance
residual_wrench_total_empirical_std
residual_wrench_total_empirical_correlation
residual_wrench_mean
residual_wrench_centered
```

All of these scientific wrench quantities are in the **nominal-mass gauge**.

---

# 7. Map the additional wrench fluctuation back to acceleration-residual space

The inverse closure map is

```math
r_k
=
B w_k^{res},
```

with

```math
B
=
\begin{bmatrix}
m^{-1}I_3 & -[\ell]_\times J^{-1}\\
0 & J^{-1}
\end{bmatrix}.
```

Verify numerically that

```math
BG=I_6
```

to machine precision.

The additional acceleration-residual covariance implied by residual-wrench fluctuation is

```math
Q_r
=
BQ_wB^\top.
```

Save:

```text
residual_acceleration_model_discrepancy_covariance
```

and its eigenvalues/std/correlation.

Because the common-scale transformation multiplies `w` by `a` and divides `B` by `a`, `Q_r` must be common-scale-gauge invariant.

Add an explicit test for this invariance.

---

# 8. Propagate residual-wrench fluctuation into parameter covariance

The point estimator is unchanged.

Let the current optimization residual Jacobian block be

```math
J_k
=
\frac{\partial r_k}{\partial q}
\in\mathbb R^{6\times14},
```

and let the current optimization weight be

```math
W_k.
```

The current curvature is

```math
A
=
\sum_k
J_k^\top W_kJ_k.
```

The current SG-overlap sandwich middle is already implemented:

```math
M_{SG}
=
\sum_{k,\ell}
J_k^\top W_k
C^{SG}_{k\ell}
W_\ell J_\ell.
```

Keep this exactly as it is.

For the additional residual-wrench model discrepancy, use the working temporal model

```math
\operatorname{Cov}_{wrench}(r_k,r_\ell)
=
\delta_{k\ell}Q_r.
```

This task does **not** introduce a hand-selected HAC bandwidth or arbitrary temporal kernel.

Therefore the additional sandwich middle is

```math
M_w
=
\sum_k
J_k^\top W_k
Q_r
W_kJ_k.
```

The total middle is

```math
M_{total}
=
M_{SG}+M_w.
```

Perform the same exact common-scale gauge reduction already used by `parameter_covariances()`.

If `P` is the existing gauge-section basis, define

```math
A_r=P^\top AP,
```

```math
M_{SG,r}=P^\top M_{SG}P,
```

```math
M_{total,r}=P^\top M_{total}P.
```

Use the same machine-precision symmetric pseudoinverse policy already used by the current implementation.

The three covariance outputs are:

```math
C_q^{naive}
=
P A_r^\dagger P^\top,
```

```math
C_q^{SG}
=
P A_r^\dagger
M_{SG,r}
A_r^\dagger
P^\top,
```

```math
C_q^{SG+wrench}
=
P A_r^\dagger
M_{total,r}
A_r^\dagger
P^\top.
```

The existing names must be preserved:

```text
parameter_covariance_naive
parameter_covariance_overlap_corrected
```

Add:

```text
parameter_covariance_wrench_corrected
```

where `wrench_corrected` means:

```text
SG overlap + excess residual-wrench model discrepancy.
```

Do not overwrite `overlap_corrected`.

Also expose the additional middle matrix, at least in `arrays.npz`:

```text
parameter_sandwich_middle_sg
parameter_sandwich_middle_wrench
parameter_sandwich_middle_total
```

This makes the uncertainty decomposition auditable.

---

# 9. Required code structure

Keep this revision compact and explicit.

A suggested implementation is:

## `single_bag_savgol_covariance.py`

Add a frozen result dataclass, e.g.

```python
@dataclass(frozen=True)
class ResidualWrenchUncertainty:
    mass_gauge_scale: float
    fixed_mass_kg: float
    wrench: np.ndarray
    centered_wrench: np.ndarray
    mean: np.ndarray
    empirical_covariance: np.ndarray
    sg_covariance_per_time: np.ndarray
    sg_covariance_mean: np.ndarray
    excess_covariance_raw: np.ndarray
    excess_covariance_raw_eigenvalues: np.ndarray
    model_discrepancy_covariance: np.ndarray
    model_discrepancy_eigenvalues: np.ndarray
    acceleration_model_discrepancy_covariance: np.ndarray
```

Add a function with a name such as:

```python
residual_wrench_uncertainty(...)
```

It must receive enough information to use:

- final strict evaluation,
- nominal vehicle mass,
- reference full SG covariance,
- pose-sensor lever arm.

Do not let this function depend on the optimization covariance mode for the SG subtraction.

## `ParameterCovarianceResult`

Extend the existing result to include, at minimum:

```python
wrench_corrected
sandwich_middle_wrench
sandwich_middle_total
```

The current fields must remain.

## `parameter_covariances(...)`

Extend it to receive the additional acceleration-space covariance, for example:

```python
additional_residual_covariance: Optional[np.ndarray] = None
```

with accepted shape `(6,6)`.

The existing SG-overlap calculation remains exactly as-is.

Add only the `M_w` term described above.

## `single_bag_savgol_core.py`

The final estimator sequence must remain:

```text
final strict evaluation
-> ridge analysis
-> post-fit uncertainty
-> diagnostics
```

Compute residual-wrench uncertainty **after** the point estimate is frozen.

Pass `Q_r` into parameter covariance propagation.

Do not feed `Q_r` back into `estimate_single_bag()` optimization iterations.

## `single_bag_savgol_reports.py`

Restore residual wrench as a first-class output as described below.

---

# 10. Required residual-wrench report restoration

The residual-wrench time history must be easy to find.

Do not leave it only on the final closure page.

## Main `report.pdf`

Insert a dedicated residual-wrench page immediately after the first SG trajectory page.

No existing page may be deleted.

The new page must show six subplots:

```text
Fx Fy Fz
Tx Ty Tz
```

using

```text
residual_wrench_nominal_mass_gauge.
```

For every component plot:

- full time series;
- empirical mean `bar_w`;
- `bar_w +/- 1 sigma_total`, where `sigma_total = sqrt(diag(S_w))`.

The title must include:

```text
mass gauge fixed to nominal mass = <value> kg
```

and must state that the plotted wrench is the **raw Newton--Euler residual wrench**, not the trajectory-fitted external wrench.

The page must also show, in text or title:

- force vector RMS about zero;
- torque vector RMS about zero;
- force-component standard deviations about the mean;
- torque-component standard deviations about the mean.

## Add a second dedicated covariance page

Add a page containing at minimum:

1. empirical total residual-wrench correlation matrix;
2. mean SG-predicted wrench covariance/correlation;
3. excess model-discrepancy wrench covariance/correlation.

Label axes:

```text
Fx Fy Fz Tx Ty Tz.
```

Use scientific units and explicit titles.

Do not remove the existing final wrench / closure page.

## Standalone file

Every completed default run must additionally write:

```text
residual_wrench.pdf
```

This standalone PDF must contain the two dedicated pages above.

The user must not need to search a 10+ page report just to find the residual-wrench result.

---

# 11. Required JSON and NPZ outputs

## `result.json`

Add a top-level or diagnostic object:

```text
diagnostics.residual_wrench
```

containing JSON-serializable summaries:

```text
mass_gauge_scale
fixed_mass_kg
mean
empirical_covariance
empirical_std
empirical_correlation
sg_covariance_mean
excess_covariance_raw
excess_covariance_raw_eigenvalues
model_discrepancy_covariance
model_discrepancy_eigenvalues
model_discrepancy_std
model_discrepancy_correlation
acceleration_model_discrepancy_covariance
acceleration_model_discrepancy_eigenvalues
```

Also add uncertainty metadata:

```text
uncertainty.parameter_covariance_wrench_corrected
```

Do not remove the existing covariance fields.

## `arrays.npz`

At minimum include:

```text
raw_residual_wrench
residual_wrench_nominal_mass_gauge
residual_wrench_centered
residual_wrench_mean
residual_wrench_total_empirical_covariance
residual_wrench_total_empirical_std
residual_wrench_total_empirical_correlation
residual_wrench_sg_covariance_per_time
residual_wrench_sg_covariance_mean
residual_wrench_excess_covariance_raw
residual_wrench_excess_covariance_raw_eigenvalues
residual_wrench_model_discrepancy_covariance
residual_wrench_model_discrepancy_eigenvalues
residual_wrench_model_discrepancy_std
residual_wrench_model_discrepancy_correlation
residual_acceleration_model_discrepancy_covariance
parameter_covariance_naive
parameter_covariance_overlap_corrected
parameter_covariance_wrench_corrected
parameter_sandwich_middle_sg
parameter_sandwich_middle_wrench
parameter_sandwich_middle_total
```

Do not rename or delete existing arrays.

---

# 12. Mandatory tests

Tests are implementation-correctness tests only.

Do not introduce a test requiring a particular real-bag scientific result.

The following tests are mandatory.

## 12.1 Closure map inverse

For nontrivial `m`, SPD `J`, and nonzero `ell`, verify:

```math
BG=I_6
```

to machine precision.

## 12.2 Wrench covariance propagation

For a synthetic SPD `Sigma_z`, verify:

```math
Sigma_w = G Sigma_z G^T
```

and

```math
B Sigma_w B^T = Sigma_z
```

to numerical precision.

## 12.3 Common-scale invariance

For several non-clean scale factors, e.g.

```text
a = 0.37, 1.83, 4.71
```

verify:

```math
m' = am,
J' = aJ,
w' = aw
```

and that the recovered

```math
Q_r = BQ_wB^T
```

is invariant.

## 12.4 No-double-counting null case

Construct an exact algebraic case in which

```math
S_w = mean(Sigma_w_SG).
```

Then:

```math
Q_w = 0
```

and:

```math
parameter_covariance_wrench_corrected
==
parameter_covariance_overlap_corrected
```

to machine precision.

## 12.5 Known excess covariance

Construct

```math
S_w
=
mean(Sigma_w_SG)+Q_known
```

for a known full SPD `Q_known`.

Verify that the residual-wrench uncertainty calculation recovers `Q_known`.

## 12.6 Full coupling retention

Use a synthetic `Q_known` with nonzero force--torque off-diagonal terms.

Verify that they remain nonzero after the full pipeline.

No diagonalization is allowed.

## 12.7 PSD behavior

Give an excess raw covariance with one negative eigenvalue.

Verify:

- raw matrix/eigenvalue is preserved in diagnostics;
- propagated `Q_w` is PSD;
- no silent `abs()` or arbitrary epsilon shift is applied.

## 12.8 Correct SG covariance source

For default `covariance_mode=identity`, verify that the residual-wrench SG subtraction uses `reference_full_covariance`, not the identity optimization covariance.

This test is mandatory because otherwise the default implementation would subtract the wrong quantity.

## 12.9 Corrected parameter covariance symmetry / PSD

Verify that:

```text
parameter_covariance_wrench_corrected
```

is symmetric to machine precision.

Any materially negative eigenvalue must fail the test.

Machine-roundoff negative eigenvalues may be reported and treated as numerical zero.

## 12.10 Point-estimator non-regression

The new residual-wrench uncertainty calculation is post-fit only.

A deterministic synthetic estimator case must produce identical:

```text
physical_coordinate
rotor_lag
objective
raw residual
```

before and after enabling the post-fit residual-wrench uncertainty/reporting path.

## 12.11 Report smoke test

A completed synthetic case must create:

```text
report.pdf
residual_wrench.pdf
arrays.npz
result.json
```

and the NPZ/JSON keys listed above must exist.

---

# 13. Production inputs: use exactly these JSON files

After source implementation and tests are complete, run the default estimator on exactly these three inputs.

Paths are relative to the repository root.

## Failure bag 1

JSON:

```text
ros/examples/grape-param-estim/minimal/bag_jsons/single_rosbag_1.json
```

It currently resolves to:

```text
bag:
/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/20260612_grape_hovering_4_2026-06-12-17-33-59.bag

start_seconds: 19.0
end_seconds:   25.0
```

## Failure bag 2

JSON:

```text
ros/examples/grape-param-estim/minimal/bag_jsons/single_rosbag_2.json
```

It currently resolves to:

```text
bag:
/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/20260612_grape_hovering_6_2026-06-12-17-40-34.bag

start_seconds: 25.5
end_seconds:   31.0
```

## Successful bag

JSON:

```text
ros/examples/grape-param-estim/minimal/bag_jsons/single_rosbag_succeeded.json
```

It currently resolves to:

```text
bag:
/home/leus/catkin_ws/bags/grape-drone/20260613_grape_hovering/20260613_grape_hovering_1_2026-06-13-13-44-01.bag

start_seconds: 65.0
end_seconds:   75.0
```

These JSON paths must be used explicitly in the production commands.

Do not substitute direct `--bag` arguments for the production run.

---

# 14. Default production configuration: do not override defaults

The production runs in this task are **not ablations**.

Use the default estimator configuration.

As of the starting commit, that means in particular:

```text
SG window                 = 1.0 s
SG degree                 = 5
covariance mode           = identity
gimbal source             = measured_sg
rotor lag mode            = estimated
initial lag multiplier    = 1.0 median command period
lag continuation depth    = 9, epsilon_k = 2^-k
solver                     = custom_kkt_lm
KKT                        = enabled
geometric SO(3)            = enabled
```

Do not explicitly pass flags that merely restate defaults unless needed for debugging.

The final production command should be minimal and auditable.

---

# 15. Mandatory source-commit-before-production workflow

Production outputs are namespaced by the current source commit.

Therefore **the source implementation must be committed before the production runs**.

Do not run production from an uncommitted implementation and then accidentally namespace the results under the previous source commit.

## 15.1 Before editing

From the repository:

```bash
git status --short
git rev-parse HEAD
```

The intended starting revision is:

```text
1cedae934fa5f8521f85fdba68615c3b37a4f9d3
```

If there are unrelated user changes, preserve them.

Do not reset or discard unrelated work.

## 15.2 Implement and test

Run the complete relevant test suite under:

```text
ros/examples/grape-param-estim/minimal/tests/
```

At minimum run all single-bag SG tests.

No finite-difference implementation may be introduced.

## 15.3 Commit source changes

Commit only the intended source/test changes.

Suggested commit message:

```text
Propagate residual-wrench fluctuation into uncertainty
```

Then record:

```bash
SOURCE_COMMIT="$(git rev-parse HEAD)"
echo "$SOURCE_COMMIT"
```

This exact `SOURCE_COMMIT` must be the namespace under which the production outputs are generated.

---

# 16. Mandatory production commands

Assume the repository is already the production checkout.

Use:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/ros/examples/grape-param-estim/minimal"
```

If the production ROS environment requires it, source the existing catkin environment first, e.g.

```bash
source ~/catkin_ws/devel/setup.bash
```

Do not create a new environment or change dependency versions for this task.

Use the existing vehicle model:

```text
grape_vehicle_model.json
```

Run exactly the following three default cases.

## 16.1 Failure bag 1

```bash
python3 single_bag_savgol_estimator.py \
  --bag-json bag_jsons/single_rosbag_1.json \
  --vehicle-model grape_vehicle_model.json \
  --run-id single_rosbag_1_residual_wrench_uncertainty_production_20260816
```

## 16.2 Failure bag 2

```bash
python3 single_bag_savgol_estimator.py \
  --bag-json bag_jsons/single_rosbag_2.json \
  --vehicle-model grape_vehicle_model.json \
  --run-id single_rosbag_2_residual_wrench_uncertainty_production_20260816
```

## 16.3 Successful bag

```bash
python3 single_bag_savgol_estimator.py \
  --bag-json bag_jsons/single_rosbag_succeeded.json \
  --vehicle-model grape_vehicle_model.json \
  --run-id single_rosbag_succeeded_residual_wrench_uncertainty_production_20260816
```

Do not pass:

```text
--skip-bag-sha256
```

for production.

Do not run `single_bag_savgol_ablation.py`.

---

# 17. Mandatory production non-regression check against the previous default results

This revision must not change the point estimator.

The previous production default results are stored under source commit:

```text
1264f5c376537941a7cfe78e8a2946bf34939f07
```

Use these as the point-estimator reference.

## Failure bag 1 reference

```text
ros/examples/grape-param-estim/minimal/outputs/1264f5c376537941a7cfe78e8a2946bf34939f07/ablation/single_rosbag_1_measured_gimbal_single_lag_production_20260816/cases/default/result.json
```

## Failure bag 2 reference

```text
ros/examples/grape-param-estim/minimal/outputs/1264f5c376537941a7cfe78e8a2946bf34939f07/ablation/single_rosbag_2_measured_gimbal_single_lag_production_20260816/cases/default/result.json
```

## Successful bag reference

```text
ros/examples/grape-param-estim/minimal/outputs/1264f5c376537941a7cfe78e8a2946bf34939f07/ablation/single_rosbag_succeeded_measured_gimbal_single_lag_production_20260816/cases/default/result.json
```

For each new production run, compare at least:

```text
parameters.chart_coordinate
parameters.rotor_lag_seconds
common_evaluation.identity_objective_sum
common_evaluation.specific_acceleration_rmse_m_per_s2
common_evaluation.angular_acceleration_rmse_rad_per_s2
```

The result must be identical up to ordinary deterministic floating-point reproduction.

Recommended automated tolerance:

```python
np.allclose(new, old, rtol=1e-9, atol=1e-12)
```

If this check fails materially:

> **STOP. Do not commit production outputs as valid results.**

Find why the point estimator changed.

This task is post-fit residual-wrench uncertainty only.

---

# 18. Mandatory production output checks

For each of the three runs, require:

```text
status == completed
success == true
strict_final_evaluation == true
```

The output directory must be:

```text
ros/examples/grape-param-estim/minimal/outputs/<SOURCE_COMMIT>/default/<RUN_ID>/
```

Each directory must contain at minimum:

```text
status.json
result.json
arguments.json
timing.json
arrays.npz
report.pdf
residual_wrench.pdf
```

## Validate the residual-wrench data

For each bag verify:

1. `residual_wrench_nominal_mass_gauge` is finite.
2. `residual_wrench_mean` is finite.
3. empirical covariance is symmetric.
4. empirical covariance is PSD to machine precision.
5. SG mean wrench covariance is finite and symmetric.
6. raw excess covariance is saved before PSD projection.
7. model-discrepancy covariance is PSD.
8. `Q_r` is finite, symmetric, and PSD.
9. `parameter_covariance_wrench_corrected` is finite and symmetric.
10. the corrected covariance is not silently identical to the old overlap covariance unless `Q_w` is genuinely numerically zero.

Also print a compact production summary for each bag:

```text
nominal fixed mass
mean residual force [N]
std residual force [N]
mean residual torque [Nm]
std residual torque [Nm]
eigenvalues of empirical S_w
eigenvalues of SG wrench covariance mean
eigenvalues of raw excess covariance
eigenvalues of final Q_w
parameter uncertainty inflation summary:
    diag(C_wrench_corrected) / diag(C_overlap_corrected)
```

For directions whose denominator is machine-zero, print `undefined` rather than a meaningless ratio.

---

# 19. Mandatory scientific sanity checks on the production outputs

The following must be checked before committing results.

## 19.1 Gauge consistency

For each production solution, independently rescale the final estimator gauge to the nominal-mass gauge and verify:

```math
w_{\mathrm{massfix}}
=
a_m w_{\mathrm{estimator\ gauge}}
```

to machine precision.

## 19.2 Closure after mass gauge fixing

The Newton--Euler closure error must remain at machine precision after the common scale transformation.

## 19.3 SG subtraction audit

Report:

```math
S_w,
\qquad
\bar\Sigma_w^{SG},
\qquad
Q_{w,\mathrm{raw}},
\qquad
Q_w.
```

Do not hide if one or more raw excess eigenvalues are negative.

## 19.4 Uncertainty monotonicity in PSD order

Because

```math
M_w >= 0,
```

the wrench-corrected covariance should add uncertainty relative to SG-overlap covariance on the identified gauge section.

Numerically inspect the eigenvalues of:

```math
C_q^{SG+wrench}-C_q^{SG}.
```

Materially negative eigenvalues indicate an implementation error.

Machine-roundoff negatives may be reported as numerical zero.

## 19.5 No point-estimate feedback

Confirm again that the point estimate is unchanged from the previous production default.

---

# 20. Production results commit and push

Only after all three production runs pass the checks above:

```bash
git status --short
```

Add only the intended production output directories under:

```text
ros/examples/grape-param-estim/minimal/outputs/<SOURCE_COMMIT>/default/
```

Do not add unrelated temporary files or caches.

Suggested production-results commit message:

```text
Add residual-wrench uncertainty production results
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

The working tree should be clean except for any unrelated pre-existing user changes that were deliberately preserved.

---

# 21. Required final implementation report

After push, report all of the following.

## Source provenance

```text
starting commit:
source implementation commit:
production results commit:
pushed branch:
```

## Files changed

List the exact source/test files changed.

## Tests

State the exact test commands and their result.

## Production runs

For each bag give:

```text
JSON path
resolved bag path
time interval
output directory
status
final strict lag cell
identity objective
```

## Residual-wrench results

For each bag give, in the nominal-mass gauge:

```text
mean wrench [Fx,Fy,Fz,Tx,Ty,Tz]
empirical std [Fx,Fy,Fz,Tx,Ty,Tz]
empirical covariance
mean SG-predicted wrench covariance
raw excess covariance eigenvalues
final Q_w eigenvalues
```

## Uncertainty effect

For each bag give a concise comparison of:

```text
diag(parameter_covariance_overlap_corrected)
diag(parameter_covariance_wrench_corrected)
```

and a useful inflation summary.

Do not claim that this covariance captures every source of uncertainty.

It specifically adds the empirically observed residual-wrench fluctuation, after subtracting the SG-predicted marginal contribution, to the existing SG-overlap sandwich uncertainty.

---

# 22. Explicit non-goals

This task does **not** authorize any of the following:

- changing the point-estimation objective;
- changing identity weighting to full covariance;
- changing SG window or degree;
- changing gimbal handling;
- changing lag initialization;
- changing `2^-k` continuation;
- changing strict-ZOH cell refinement;
- changing KKT gauge handling;
- changing matrix-exponential inertia parameterization;
- adding priors;
- adding arbitrary parameter bounds;
- adding residual-wrench penalties;
- adding a free residual-wrench trajectory to the optimizer;
- running new ablations;
- removing existing diagnostics;
- replacing raw residual wrench with trajectory-fitted external wrench;
- using trajectory-fitted external wrench to estimate `Q_w`;
- silently clipping or hiding scientifically relevant negative eigenvalues before reporting the raw excess covariance;
- using the identity optimization covariance as the SG noise contribution in the default case.

The source of the residual-wrench fluctuation estimate is specifically:

```text
final strict raw Newton--Euler residual wrench
```

after exact common-scale regauging to:

```text
mass = nominal vehicle-model mass.
```

---

# 23. Acceptance checklist

The task is complete only if every item below is true.

- [ ] Point estimator unchanged.
- [ ] Raw residual wrench remains saved.
- [ ] Residual wrench is additionally reported in nominal-mass gauge.
- [ ] Residual-wrench mean is saved.
- [ ] Full `6x6` empirical wrench covariance is saved.
- [ ] Force--torque coupling is retained.
- [ ] Full reference SG covariance is mapped into wrench space.
- [ ] SG contribution is subtracted before constructing added model discrepancy.
- [ ] Raw excess covariance is preserved and reported.
- [ ] PSD model-discrepancy covariance `Q_w` is produced.
- [ ] `Q_w` is mapped back to acceleration space as `Q_r`.
- [ ] `Q_r` is added to the post-fit sandwich uncertainty.
- [ ] Existing naive covariance remains.
- [ ] Existing SG-overlap covariance remains.
- [ ] New wrench-corrected parameter covariance is added.
- [ ] Residual-wrench time-history page is restored near the front of `report.pdf`.
- [ ] Residual-wrench covariance page is added.
- [ ] Standalone `residual_wrench.pdf` is generated.
- [ ] No existing report page is deleted.
- [ ] No ablation is run for this task.
- [ ] All relevant tests pass.
- [ ] Source changes are committed before production runs.
- [ ] Failure bag 1 is run from `bag_jsons/single_rosbag_1.json`.
- [ ] Failure bag 2 is run from `bag_jsons/single_rosbag_2.json`.
- [ ] Successful bag is run from `bag_jsons/single_rosbag_succeeded.json`.
- [ ] All three production runs complete successfully.
- [ ] New point estimates match the previous production defaults.
- [ ] All three output directories contain `residual_wrench.pdf`.
- [ ] Production outputs are committed.
- [ ] Both commits are pushed.
- [ ] Final report includes source SHA, results SHA, and production output paths.

No item on this checklist is optional.
