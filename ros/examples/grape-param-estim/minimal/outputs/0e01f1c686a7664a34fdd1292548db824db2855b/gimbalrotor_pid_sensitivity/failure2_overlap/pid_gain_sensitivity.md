# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `overlap_corrected`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- valid eigen samples: `27/27`

## Gain-scale sensitivity

| group | center | local linear 1-sigma | relative | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|
| xy | 1.1787878 | 0.00014118274 | 0.012% | 1.1787319 | 1.1799563 |
| z | 1.1789924 | 0.00015273498 | 0.013% | 1.1788928 | 1.1793054 |
| roll_pitch | 1.7244864 | 0.28171073 | 16.336% | 1.7107749 | 3.2615882 |
| yaw | 1.9977971 | 0.20591642 | 10.307% | 1.9659625 | 8.2366858 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | +0.00012528308 | 78.74% |
| 2 | -5.1851241e-05 | 13.49% |
| 3 | +2.4634664e-05 | 3.04% |
| 4 | +2.2492366e-05 | 2.54% |
| 9 | +1.1926938e-05 | 0.71% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | +8.9937391e-05 | 34.67% |
| 1 | +8.9633003e-05 | 34.44% |
| 2 | -6.4628594e-05 | 17.90% |
| 4 | +4.5823317e-05 | 9.00% |
| 5 | +2.4237967e-05 | 2.52% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | +0.28130666 | 99.71% |
| 3 | +0.013745332 | 0.24% |
| 2 | -0.0047340086 | 0.03% |
| 1 | -0.0031860615 | 0.01% |
| 4 | -0.0024279112 | 0.01% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | +0.20086662 | 95.16% |
| 2 | -0.031931989 | 2.40% |
| 1 | -0.030765134 | 2.23% |
| 3 | +0.0090252126 | 0.19% |
| 4 | -0.0023195024 | 0.01% |

## Optional Monte Carlo stress test

| group | mean | std | q16 | median | q84 | q2.5 | q97.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.1794614 | 0.00040619264 | 1.1789452 | 1.1795088 | 1.1799232 | 1.1787688 | 1.180092 |
| z | 1.179145 | 0.00016572318 | 1.1789737 | 1.1791664 | 1.1793132 | 1.1787752 | 1.179398 |
| roll_pitch | 2.6879931 | 0.55930052 | 2.0494899 | 2.7383031 | 3.310511 | 1.7423666 | 3.6580553 |
| yaw | 3.370438 | 2.177361 | 1.761657 | 2.6643463 | 6.0659059 | 0.2442371 | 8.0624893 |

Monte Carlo here is a nonlinear stress test of the selected local
covariance model. Do not relabel these quantiles as posterior
credible intervals without an independent probabilistic
calibration argument.

