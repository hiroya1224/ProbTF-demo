# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `overlap_corrected`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- valid eigen samples: `27/27`

## Gain-scale sensitivity

| group | center | local linear 1-sigma | relative | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|
| xy | 1.1520448 | 0.0003850356 | 0.033% | 1.1518121 | 1.1528091 |
| z | 1.1697648 | 9.4808533e-05 | 0.008% | 1.1696965 | 1.1698261 |
| roll_pitch | 3.5287743 | 0.034451441 | 0.976% | 3.5039769 | 3.5939827 |
| yaw | 3.3788679 | 0.079024678 | 2.339% | 3.3309613 | 3.4571853 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 4 | -0.00023029591 | 35.77% |
| 0 | +0.00019342885 | 25.24% |
| 3 | +0.0001458469 | 14.35% |
| 1 | +0.00011849939 | 9.47% |
| 2 | +0.0001176613 | 9.34% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 1 | +6.475824e-05 | 46.65% |
| 2 | -4.4138148e-05 | 21.67% |
| 4 | -3.1400596e-05 | 10.97% |
| 8 | -2.5759354e-05 | 7.38% |
| 3 | -2.06272e-05 | 4.73% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | +0.023930546 | 48.25% |
| 3 | -0.019501082 | 32.04% |
| 1 | +0.012903507 | 14.03% |
| 0 | +0.0069814014 | 4.11% |
| 4 | +0.0042799426 | 1.54% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | -0.063111977 | 63.78% |
| 2 | +0.041482382 | 27.56% |
| 3 | -0.016470216 | 4.34% |
| 1 | +0.016301194 | 4.26% |
| 4 | -0.0017501959 | 0.05% |

## Optional Monte Carlo stress test

| group | mean | std | q16 | median | q84 | q2.5 | q97.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.1526051 | 0.00078394774 | 1.1519289 | 1.1524343 | 1.153326 | 1.1515479 | 1.1546759 |
| z | 1.1697614 | 9.653939e-05 | 1.1696681 | 1.169772 | 1.1698503 | 1.169553 | 1.1699272 |
| roll_pitch | 3.5899434 | 0.083580381 | 3.5213908 | 3.5659894 | 3.6632221 | 3.486534 | 3.8037621 |
| yaw | 3.4009254 | 0.092137754 | 3.3156673 | 3.387492 | 3.4764553 | 3.2669534 | 3.6350899 |

Monte Carlo here is a nonlinear stress test of the selected local
covariance model. Do not relabel these quantiles as posterior
credible intervals without an independent probabilistic
calibration argument.

