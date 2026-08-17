# Prologue: 以下の話が出てくる経緯

今回の実装は、推定器が実際に保存している `physical_coordinate`, `quotient_basis`, `quotient_covariance_*` を利用します。推定器自身も covariance を common-scale quotient basis へ射影して保存しているので、その定義に沿っています。

配置は以下です。

```text
ros/examples/grape-param-estim/
├── gimbalrotor_pid_postprocess_sensitivity_validation.md
└── minimal/
    ├── gimbalrotor_pid_postprocess_sensitivity.py
    ├── three_bag_gimbalrotor_pid_postprocess_sensitivity_summary.py
    └── tests/
        └── test_gimbalrotor_pid_postprocess_sensitivity.py
```

実装の主処理は **bag 不要**です。`result.json` の隣の `arrays.npz` を自動的に読み、

\[
c_i^\pm = \hat c \pm k\sqrt{\lambda_i}Bv_i
\]

という 13 covariance eigen-direction の ± 点と中心、計 **27点**を `SiParameterChart.decode()` に通します。物理量へ直接 Gaussian 加算する実装にはしていません。そのため inertia の物理的 parameterization、正の force effectiveness、common-scale gauge が維持されます。

各 group について

\[
\Delta s_{g,i} =
\frac{s_g(+\sigma_i)-s_g(-\sigma_i)}{2},
\qquad
\sigma_{s_g,\mathrm{local}} =
\sqrt{\sum_i\Delta s_{g,i}^2}
\]

を計算し、

```text
center scale
local linearized one-sigma
relative one-sigma
27点での min / max
各 eigen-direction の寄与率
±方向の非対称性（二階差分）
```

を JSON / md に出します。これでまさに、

```text
xy, z は 1.18 ± 0.03 程度
roll_pitch は 3.5 ± 2.0 程度
```

なのか、

```text
各 bag 内では roll_pitch も非常に狭い
しかし bag 間で 1.7, 3.5, 4.2 と違う
```

なのかを切り分けられます。

さらに optional に、

```bash
--monte-carlo-samples 512 --seed 0
```

を付けると同じ quotient covariance から nonlinear stress test もできます。ただしコード中でも出力 md でも、これを posterior / credible interval とは呼ばないようにしてあります。

3 bag の結果が出たら、

```bash
python3 minimal/three_bag_gimbalrotor_pid_postprocess_sensitivity_summary.py \
  --input failure1=.../pid_gain_sensitivity.json \
  --input failure2=.../pid_gain_sensitivity.json \
  --input success=.../pid_gain_sensitivity.json \
  --output-dir ...
```

で、

\[
\sigma_{\rm within}
\quad\text{vs.}\quad
\sigma_{\rm between}
\]

を直接比較します。これが今回知りたかった「回転側の correction 自体の信頼度が低いのか、それとも各 bag は鋭く決まっているのに bag 間で実効プラントが変わってしまうのか」の判定になります。

validation md にはさらに、`overlap_corrected` と `conservative_fusion` の両方を試すこと、0.5σ/1σ/2σ で結論が反転しないか、27点の invalid sample を黙って捨てないこと、中心点が既存 failure1 の `1.1520 / 1.1698 / 3.5288 / 3.3789` を再現すること、などをチェック項目として残してあります。現在の estimator が `arrays.npz` に quotient covariance を保存していることも既存実装と整合しています。


# Gimbalrotor PID postprocess sensitivity — validation and adjustment notes

This document accompanies:

```text
minimal/gimbalrotor_pid_postprocess_sensitivity.py
minimal/three_bag_gimbalrotor_pid_postprocess_sensitivity_summary.py
minimal/tests/test_gimbalrotor_pid_postprocess_sensitivity.py
```

The sensitivity tool is deliberately usable **without reopening the ROS bag**.

It consumes the physical estimator's existing artifacts:

```text
result.json
arrays.npz
grape_vehicle_model.json
```

and propagates local estimator uncertainty through the already-implemented
static PID postprocess:

```text
14-D SI parameter chart
    -> 13-D common-scale quotient perturbation
    -> scale-free plant (J/m, CoG, f/m)
    -> A_real
    -> H = A_real A_cmd^+
    -> H_bar
    -> s_xy, s_z, s_roll_pitch, s_yaw
```

The physical estimator is not rerun and its loss is not changed.

---

## 1. Why sampling is performed in the estimator chart

Do not sample the 13 physical quantities

```text
J/m
CoG
f/m
```

by adding an unconstrained Gaussian vector directly.

The committed estimator covariance is defined in the local quotient of the
estimator's 14-D SI chart. The tool therefore reads:

```text
physical_coordinate
quotient_basis
quotient_covariance_<mode>
```

from `arrays.npz`, forms

```math
c = \hat c + B \delta z,
```

and decodes every sample using `SiParameterChart.decode()`.

New estimator artifacts always store `physical_coordinate`. The three current
production archives predate that field but do store `quotient_coordinate`.
For those archives only, the loader reconstructs the unique gauge-orthogonal
representative `quotient_basis @ quotient_coordinate`; this differs from the
original 14-D coordinate at most by the exact common-scale gauge and is checked
against the scale-free plant in `result.json` before any sensitivity result is
written.

This preserves the estimator's physical parameterization, including positive
mass, positive force effectiveness, and the positive second-moment
parameterization underlying the inertia tensor.

It also keeps the exact common-scale gauge out of the perturbations.

---

## 2. Primary result: deterministic 27-point eigen-direction sensitivity

For the selected 13x13 quotient covariance,

```math
\Sigma_z = V \Lambda V^\top,
```

the primary run evaluates:

```math
\hat c
```

and, for all 13 covariance eigen-directions,

```math
\hat c \pm k \sqrt{\lambda_i} B v_i.
```

At the default `k=1`, this is:

```text
1 center + 2 * 13 directions = 27 evaluations
```

No random sampling is required for the primary result.

For each PID group the tool reports:

```text
center scale
local linearized one-sigma scale sensitivity
relative one-sigma sensitivity
minimum and maximum among the 27 points
dominant covariance directions
```

The local linear one-sigma value is reconstructed from the centered
plus/minus evaluations:

```math
\Delta s_{g,i}
=
\frac{s_g(+\sigma_i)-s_g(-\sigma_i)}{2},
```

```math
\sigma_{s_g,\mathrm{local}}
=
\sqrt{\sum_i \Delta s_{g,i}^2}.
```

This is a deterministic sensitivity summary. It is useful even if the project
does not use a probabilistic plant estimate operationally.

---

## 3. Covariance modes

Supported modes are:

```text
naive
overlap_corrected
wrench_corrected
conservative_fusion
```

Default:

```text
conservative_fusion
```

The estimator explicitly treats `conservative_fusion` as conservative and not
as a calibrated generative noise covariance. Therefore:

- call the output a **local sensitivity envelope**;
- do not call the eigen-point range a credible interval;
- do not call optional Monte Carlo quantiles posterior credible intervals.

For scientific interpretation, run at least:

```text
overlap_corrected
conservative_fusion
```

and compare them.

If the conclusion "rotation gain correction is weakly determined" appears only
under `conservative_fusion`, say so explicitly.

If both modes give a large roll/pitch or yaw spread while xy/z remain narrow,
the evidence is much stronger.

---

## 4. Commands for the current three production results

From:

```bash
cd /home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim
```

set:

```bash
SOURCE_COMMIT=$(git rev-parse HEAD)
OUT=minimal/outputs/${SOURCE_COMMIT}/gimbalrotor_pid_sensitivity
```

### failure1

```bash
python3 minimal/gimbalrotor_pid_postprocess_sensitivity.py \
  --result minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json \
  --vehicle-model minimal/grape_vehicle_model.json \
  --covariance-mode conservative_fusion \
  --output-dir ${OUT}/failure1_conservative
```

### failure2

```bash
python3 minimal/gimbalrotor_pid_postprocess_sensitivity.py \
  --result minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json \
  --vehicle-model minimal/grape_vehicle_model.json \
  --covariance-mode conservative_fusion \
  --output-dir ${OUT}/failure2_conservative
```

### success

```bash
python3 minimal/gimbalrotor_pid_postprocess_sensitivity.py \
  --result minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json \
  --vehicle-model minimal/grape_vehicle_model.json \
  --covariance-mode conservative_fusion \
  --output-dir ${OUT}/success_conservative
```

`--arrays` is omitted above because the CLI defaults to `arrays.npz` next to
the supplied `result.json`.

Repeat with:

```text
--covariance-mode overlap_corrected
```

before drawing a scientific conclusion from the spread.

---

## 5. Optional nonlinear Monte Carlo stress test

The deterministic 27 points are the primary analysis.

If the eigen-direction result shows large rotation sensitivity, additionally
run, for example:

```bash
python3 minimal/gimbalrotor_pid_postprocess_sensitivity.py \
  ... \
  --monte-carlo-samples 512 \
  --seed 0
```

The samples are generated in the 13-D quotient chart covariance and decoded
through `SiParameterChart`.

Use Monte Carlo to answer:

```text
Is the map from plant configuration to PID scale strongly nonlinear over the
selected local covariance size?
```

Do not use it to assert a Bayesian posterior unless the estimator is separately
given that interpretation.

---

## 6. Three-bag within-vs-between comparison

After generating the three sensitivity reports:

```bash
python3 minimal/three_bag_gimbalrotor_pid_postprocess_sensitivity_summary.py \
  --input failure1=${OUT}/failure1_conservative/pid_gain_sensitivity.json \
  --input failure2=${OUT}/failure2_conservative/pid_gain_sensitivity.json \
  --input success=${OUT}/success_conservative/pid_gain_sensitivity.json \
  --output-dir ${OUT}/three_bag_conservative
```

The important comparison is:

```text
within-bag local sigma
vs
between-bag standard deviation of center scales
```

Interpretation:

### Large within-bag spread, modest between-bag spread

Example pattern:

```text
s_xy:          narrow
s_z:           narrow
s_roll_pitch:  broad
s_yaw:         broad
```

This supports:

```text
the translational correction is locally well determined;
the rotational correction is sensitive to the identified-plant ridge.
```

### Small within-bag spread, large between-bag spread

This supports a different diagnosis:

```text
each bag gives a locally sharp effective plant,
but different bags identify mutually inconsistent effective plants.
```

Then inspect model discrepancy, excitation, SG differentiation, local minima,
and unmodelled dynamics before describing the issue as parameter uncertainty.

### Both are large

Both local identifiability and bag-to-bag model consistency matter.

---

## 7. What must match the existing static postprocess

For every bag, the sensitivity center must reproduce the existing point
postprocess scale to numerical precision.

Current failure1 target:

```text
xy          1.15204476
z           1.16976482
roll_pitch  3.52877431
yaw         3.37886790
```

If the center differs, stop. Do not interpret the sensitivity output.

Likely causes are:

1. `physical_coordinate` and `result.json` are from different runs;
2. the wrong `arrays.npz` was supplied;
3. the vehicle-model JSON differs;
4. the static postprocess implementation has changed;
5. BODY-origin vs CoG-relative geometry handling diverged.

The sensitivity script explicitly decodes the center and compares its
scale-free plant with `result.json`; a mismatch is an input error.

---

## 8. What to inspect in `pid_gain_sensitivity.json`

### 8.1 Covariance spectrum

Inspect:

```text
center_and_eigen_sensitivity.covariance.eigenvalues_descending
center_and_eigen_sensitivity.covariance.numerical_rank
center_and_eigen_sensitivity.covariance.retained_condition_number
```

Tiny negative eigenvalues caused by floating-point symmetrization are clipped.

A materially negative eigenvalue is rejected. Do not enlarge the clipping
tolerance merely to make a run succeed.

### 8.2 Validity of all 27 samples

Inspect:

```text
eigen_sampling.valid_sample_count
eigen_sampling.invalid_sample_count
```

Expected under ordinary local use:

```text
27 valid
0 invalid
```

If +/-1 sigma already produces invalid static allocations or nonpositive gain
scales, that itself is evidence that the static correction is unstable over the
chosen local uncertainty scale.

Do not silently discard invalid directions.

### 8.3 Relative gain-scale sensitivity

Inspect:

```text
group_summary.<group>.relative_linearized_one_sigma
```

The most informative pattern for the present question would be:

```text
xy, z             small
roll_pitch, yaw   large
```

There is no hard universal percentage threshold in this code. Report the
actual values.

### 8.4 Dominant directions

Inspect:

```text
group_summary.<group>.top_eigen_directions
eigen_sampling.directions[i].scale_free_one_sigma_effect
```

This shows which covariance eigen-directions actually drive each PID scale.

The full 13-component physical scale-free change is retained so a direction can
be related back to:

```text
J/m
CoG
f_i/m
```

without pretending that mixed-unit component magnitudes can be ranked by one
universal norm.

### 8.5 Nonlinearity

Inspect:

```text
eigen_sampling.directions[i].scale_second_difference_per_sigma2
```

A large second difference relative to the first-order scale effect indicates
that one-sigma linearization is already weak.

Then compare:

```text
--sigma-multiple 0.5
--sigma-multiple 1.0
--sigma-multiple 2.0
```

A robust local conclusion should not reverse merely because the diagnostic
radius is changed modestly.

---

## 9. Tests that do not require a ROS bag

Run:

```bash
python3 -m pytest minimal/tests/test_gimbalrotor_pid_postprocess_sensitivity.py
```

The tests cover:

```text
zero-covariance identity behavior
27-point construction
finite local spread
common-scale gauge invariance
PSD covariance validation
Monte Carlo reproducibility
within-vs-between summary logic
current failure1 center-scale smoke check
```

The current failure1 smoke check uses only the committed estimator
`result.json`, `arrays.npz`, and vehicle model. It does not open a `.bag`.

Also run the existing suite:

```bash
python3 -m pytest minimal/tests
```

The sensitivity code must not alter the physical estimator or existing static
postprocessor behavior.

---

## 10. If the real bag cannot be used

That is acceptable for this sensitivity task.

The following are sufficient to validate the new calculation:

1. the committed `result.json` exists;
2. the sibling `arrays.npz` exists;
3. `physical_coordinate` decodes back to the exact scale-free plant in
   `result.json`;
4. the sensitivity center reproduces the already-committed static PID scales;
5. the quotient basis is orthonormal and orthogonal to the common-scale gauge;
6. the selected covariance is PSD up to numerical roundoff;
7. all 27 one-sigma samples can be evaluated, or invalid directions are
   reported explicitly;
8. zero covariance gives exactly zero sensitivity;
9. shifting the center along the exact common-scale gauge does not change the
   result;
10. the three-bag summary clearly separates within-bag sensitivity from
    between-bag variation.

No rosbag signal is required for any of these checks.

The rosbag remains necessary for the separate task of reconstructing the
flight-time PID baseline and for later closed-loop replay. That is already
handled by the static postprocessor / existing rosbag adapter.

---

## 11. If `arrays.npz` is unavailable

Do not manufacture a covariance from the three bag point estimates.

The correct fixes are:

1. locate the `arrays.npz` generated beside the corresponding estimator
   `result.json`;
2. if the run predates post-fit covariance output, rerun that estimator result
   with post-fit uncertainty enabled;
3. if post-fit uncertainty failed, fix that failure first or restrict the
   analysis to deterministic point estimates.

The script can project a stored 14x14 `parameter_covariance_<mode>` through
`quotient_basis` if a direct `quotient_covariance_<mode>` field is absent.

It must not invent a diagonal covariance.

---

## 12. Likely follow-up adjustments

### 12.1 Covariance choice

Run both `overlap_corrected` and `conservative_fusion`.

If they disagree strongly about rotational sensitivity, document that as part
of the uncertainty-model dependence.

### 12.2 Sampling radius

Use 0.5, 1, and 2 sigma only as a sensitivity-radius study.

Do not tune the radius to obtain a desired conclusion.

### 12.3 Characteristic length

The static PID mapping uses the same nominal radius-of-gyration characteristic
length as the existing point postprocessor.

If `--characteristic-length` is changed, rerun the center and verify that the
change is understood as a different translation/rotation weighting convention.

### 12.4 Strongly nonlinear directions

If plus/minus asymmetry is large:

- retain the explicit plus/minus samples;
- use optional Monte Carlo;
- avoid summarizing that direction by only a linear standard deviation.

### 12.5 Large rotational sensitivity with narrow translation sensitivity

This is the result most directly relevant to the current hypothesis.

The next step would be to inspect the dominant physical quotient changes and
then perform the already-planned closed-loop replay for representative
low/center/high rotational-correction samples.

Do not immediately deploy an extreme `roll_pitch` or `yaw` gain merely because
the center static inverse is large.

---

## 13. Scientific wording

Preferred wording if the result is broad:

> Under the estimator's local quotient-space sensitivity model, the
> translational PID correction is stable while the rotational correction varies
> substantially along weakly constrained plant directions.

Preferred wording if within-bag sensitivity is narrow but bags disagree:

> Each fitted plant gives a locally stable rotational correction, while the
> correction changes substantially across flight records; the dominant issue is
> therefore cross-record model consistency rather than local parameter
> sensitivity.

Avoid calling the selected covariance a posterior unless the estimator is
deliberately restored to a probabilistic interpretation.
