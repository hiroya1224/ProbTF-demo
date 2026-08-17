# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `conservative_fusion`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- valid eigen samples: `27/27`

## Gain-scale sensitivity

| group | center | local linear 1-sigma | relative | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|
| xy | 1.1520448 | 0.0021686875 | 0.188% | 1.150362 | 1.154915 |
| z | 1.1697648 | 0.0020643579 | 0.176% | 1.1679529 | 1.1715796 |
| roll_pitch | 3.5287743 | 0.080799539 | 2.290% | 3.4681521 | 3.7996002 |
| yaw | 3.3788679 | 0.24205699 | 7.164% | 3.2123158 | 3.6494622 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 8 | +0.0016837071 | 60.28% |
| 5 | +0.00086534731 | 15.92% |
| 0 | +0.00058118946 | 7.18% |
| 3 | +0.00047777154 | 4.85% |
| 6 | +0.00047345829 | 4.77% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 8 | +0.0018133449 | 77.16% |
| 5 | +0.00079282252 | 14.75% |
| 6 | +0.00036348025 | 3.10% |
| 7 | +0.00027619064 | 1.79% |
| 0 | +0.00022353778 | 1.17% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.057549786 | 50.73% |
| 4 | +0.038790666 | 23.05% |
| 0 | -0.029176155 | 13.04% |
| 3 | +0.028394213 | 12.35% |
| 1 | +0.0061031964 | 0.57% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | -0.21857322 | 81.54% |
| 2 | -0.09225577 | 14.53% |
| 3 | +0.042437885 | 3.07% |
| 4 | +0.021461057 | 0.79% |
| 5 | +0.0058729945 | 0.06% |

## Optional Monte Carlo stress test

| group | mean | std | q16 | median | q84 | q2.5 | q97.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.1539878 | 0.0030018098 | 1.151179 | 1.1534891 | 1.1567976 | 1.1493384 | 1.1612135 |
| z | 1.16968 | 0.001959368 | 1.1678102 | 1.1695937 | 1.171637 | 1.1660937 | 1.1739381 |
| roll_pitch | 3.7485483 | 0.2743425 | 3.5324205 | 3.6571967 | 3.9761959 | 3.4517694 | 4.5471886 |
| yaw | 3.4556525 | 0.3675398 | 3.1840461 | 3.4152632 | 3.7200541 | 2.8323059 | 4.3564986 |

Monte Carlo here is a nonlinear stress test of the selected local
covariance model. Do not relabel these quantiles as posterior
credible intervals without an independent probabilistic
calibration argument.

