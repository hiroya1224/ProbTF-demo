# Gimbalrotor PID Monte Carlo postprocess — production validation

## 1. Role in the pipeline

The intended production pipeline is

```text
flight data
  -> physical parameter estimator
  -> Gaussian approximation in the estimator's native common-scale quotient
  -> Monte Carlo plant samples
  -> nonlinear static PID map
  -> empirical PID gain-scale distribution
  -> median PID proposal + quantile ranges for the next experiment
```

The PID postprocessor does **not** force the output distribution back to a
Gaussian.  The estimator Gaussian is only the input uncertainty model.

The production Monte Carlo sampler stays in `estimator_quotient`.  The previous
`centered_scale_free_spd` experiment remains a diagnostic experiment and is not
used here.  A Gaussian transformed through a nonlinear coordinate map would no
longer be Gaussian, so constructing another Gaussian after such a transform
would silently change the uncertainty model.

## 2. New files

```text
minimal/gimbalrotor_pid_monte_carlo_postprocess.py
minimal/three_bag_gimbalrotor_pid_monte_carlo_postprocess.py
minimal/tests/test_gimbalrotor_pid_monte_carlo_postprocess.py
gimbalrotor_pid_monte_carlo_postprocess_validation.md
```

No estimator code is changed by this patch.

## 3. Inputs

One production case consumes

```text
result.json
arrays.npz
minimal/grape_vehicle_model.json
pid_gain_postprocess.json
```

`pid_gain_postprocess.json` is the already-generated static postprocess result.
It contains the PID gains actually reconstructed from the ROS bag's recorded
dynamic-reconfigure state.  The new Monte Carlo postprocessor reuses that
snapshot instead of reopening the bag.

Before sampling, the tool checks that

```text
estimator source commit
estimator case name
static point PID scales
```

match the supplied estimator result and vehicle model.  A mismatched static
postprocess result is therefore not silently used as the gain baseline.

## 4. Monte Carlo calculation

For the selected quotient covariance

```math
C_q = V \Lambda V^T,
```

the tool draws

```math
\delta q^{(n)} = V\Lambda^{1/2}\xi^{(n)},
\qquad
\xi^{(n)}\sim N(0,I),
```

and decodes

```math
q^{(n)} = \hat q + \delta q^{(n)}
```

through the estimator's `SiParameterChart`.

Each decoded plant is propagated through

```text
scale-free plant
  -> A_real
  -> H = A_real A_cmd^+
  -> H_dimensionless
  -> s_xy, s_z, s_roll_pitch, s_yaw
```

using the current sensitivity evaluator.  In particular:

- loss of the source `1e-4` SVD-threshold rank is **not** a rejection condition;
- a large or infinite `A_real` condition number is retained as a diagnostic;
- a finite negative gain-group scale is retained and counted;
- no gain scale is clipped;
- only a genuinely undefined or non-finite floating-point evaluation is marked
  invalid.

This matches the project's numerical policy: do not stop merely because a
sample looks physically surprising or ill-conditioned.

## 5. Output contract

For each PID group the report contains the empirical scale distribution

```text
mean
standard deviation
minimum / maximum
q2.5
q16
q50 (median)
q84
q97.5
nonpositive sample fraction
```

and maps the same scale quantiles onto the recorded P/I/D gains.  Since P, I,
and D use one common group scale in this v1 postprocessor, their ranges are
perfectly dependent within one group; they are not three independent gain
uncertainties.

The primary next-experiment proposal is

```text
median
16--84 % range
2.5--97.5 % range
```

of the Monte Carlo push-forward.

The report also retains the joint distribution of

```text
(s_xy, s_z, s_roll_pitch, s_yaw)
```

through its 4x4 sample covariance and correlation matrices.

Raw samples are written to

```text
pid_gain_monte_carlo_samples.npz
```

with

```text
quotient_delta_samples
scale_free_samples
gain_scale_samples
valid_mask
A_real_source_threshold_rank
A_real_condition_number
```

so the tail can be inspected later without rerunning the bag or estimator.

## 6. Output files

One case writes

```text
pid_gain_monte_carlo_postprocess.json
pid_gain_monte_carlo_postprocess.md
pid_gain_monte_carlo_samples.npz
pid_gain_median_overlay.yaml
pid_gain_proposal_ranges.yaml
status.json
```

`pid_gain_median_overlay.yaml` is a directly readable controller gain overlay
containing the median proposal only.  `pid_gain_proposal_ranges.yaml` retains
both 68% and 95% ranges.

No file is automatically deployed to the controller.

## 7. Current three-bag production command

Run from

```bash
cd /home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim
```

The current estimator artifacts are

```text
failure1:
minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json

failure2:
minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json

success:
minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json
```

The corresponding recorded-gain static postprocess reports are

```text
failure1:
minimal/outputs/585db5ba8a236232d85f2097615cf64b7eb76ff0/gimbalrotor_pid_postprocess/single_rosbag_1_prior_free_static_pid_production_20260817/pid_gain_postprocess.json

failure2:
minimal/outputs/585db5ba8a236232d85f2097615cf64b7eb76ff0/gimbalrotor_pid_postprocess/single_rosbag_2_prior_free_static_pid_production_20260817/pid_gain_postprocess.json

success:
minimal/outputs/585db5ba8a236232d85f2097615cf64b7eb76ff0/gimbalrotor_pid_postprocess/single_rosbag_succeeded_prior_free_static_pid_production_20260817/pid_gain_postprocess.json
```

Recommended production run:

```bash
SOURCE_COMMIT=$(git rev-parse HEAD)
OUT=minimal/outputs/${SOURCE_COMMIT}/gimbalrotor_pid_monte_carlo_postprocess

python3 minimal/three_bag_gimbalrotor_pid_monte_carlo_postprocess.py \
  --result failure1=minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json \
  --result failure2=minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json \
  --result success=minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json \
  --static-postprocess failure1=minimal/outputs/585db5ba8a236232d85f2097615cf64b7eb76ff0/gimbalrotor_pid_postprocess/single_rosbag_1_prior_free_static_pid_production_20260817/pid_gain_postprocess.json \
  --static-postprocess failure2=minimal/outputs/585db5ba8a236232d85f2097615cf64b7eb76ff0/gimbalrotor_pid_postprocess/single_rosbag_2_prior_free_static_pid_production_20260817/pid_gain_postprocess.json \
  --static-postprocess success=minimal/outputs/585db5ba8a236232d85f2097615cf64b7eb76ff0/gimbalrotor_pid_postprocess/single_rosbag_succeeded_prior_free_static_pid_production_20260817/pid_gain_postprocess.json \
  --vehicle-model minimal/grape_vehicle_model.json \
  --samples 10000 \
  --seed 0 \
  --output-dir "${OUT}"
```

By default this runs both

```text
conservative_fusion
overlap_corrected
```

so the effect of the covariance construction remains visible.

To run only one distribution, for example:

```bash
--covariance-mode conservative_fusion
```

## 8. What to check first

### 8.1 Reproduce the point estimate

For every case the static center scale in the new report must reproduce the
existing point postprocess exactly enough to pass the built-in consistency
check.

### 8.2 Sample validity

Inspect

```text
requested_count
valid_count
invalid_count
invalid_examples
```

The earlier `rank deficient under source threshold` rejection must not reappear.
If invalid samples remain, inspect their actual numerical failure.

### 8.3 Bag2 roll/pitch

The key current question is whether the nonlinear push-forward gives bag2 a
substantially wider roll/pitch proposal than bag1/success and whether its upper
tail moves toward the success-case scale.

Do not compare only standard deviations.  Compare

```text
median
q16--q84
q2.5--q97.5
```

because the push-forward can be strongly skewed.

### 8.4 Gain-scale correlation

Inspect whether `roll_pitch` and `yaw` move together in the same physical plant
samples.  If their correlation is strong, selecting independent marginal
endpoints for the next experiment would describe combinations that have little
or no support under the fitted plant distribution.

### 8.5 Nonpositive scale mass

A negative finite least-squares scale is not deleted.  The report contains
`nonpositive_fraction`.  If it is nonzero, the raw output distribution itself
is warning that a simple positive multiplicative PID correction is inadequate
for part of the estimated plant distribution.

## 9. Tests

Run

```bash
PYTHONPATH=minimal/tests python3 -m unittest \
  minimal/tests/test_gimbalrotor_pid_postprocess_sensitivity.py \
  minimal/tests/test_gimbalrotor_pid_coordinate_chart.py \
  minimal/tests/test_gimbalrotor_pid_monte_carlo_postprocess.py
```

The new tests cover

```text
nonpositive samples are retained
common scale maps consistently to P/I/D ranges
static baseline mismatch is rejected
zero covariance produces an exact point-mass proposal
fixed seed reproducibility
three-bag summary retains each distribution separately
```

## 10. Interpretation of the range

The resulting range is the empirical push-forward of the chosen estimator
Gaussian approximation.  It should be described as such.  If the estimator
covariance is later given a calibrated confidence/posterior interpretation,
the same Monte Carlo machinery can inherit that interpretation; the
postprocessor itself does not manufacture one.
