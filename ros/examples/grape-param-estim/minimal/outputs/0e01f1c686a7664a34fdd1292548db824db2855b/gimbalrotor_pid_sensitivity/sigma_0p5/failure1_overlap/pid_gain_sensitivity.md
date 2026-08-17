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
| xy | 1.1520448 | 0.00038830531 | 0.034% | 1.151929 | 1.15229 |
| z | 1.1697648 | 9.4766054e-05 | 0.008% | 1.1697316 | 1.1697963 |
| roll_pitch | 3.5287743 | 0.034627836 | 0.981% | 3.5165915 | 3.5473211 |
| yaw | 3.3788679 | 0.075888276 | 2.246% | 3.3531736 | 3.4123209 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 4 | -0.00023029763 | 35.17% |
| 0 | +0.00019968506 | 26.45% |
| 3 | +0.00014585142 | 14.11% |
| 1 | +0.00011887075 | 9.37% |
| 2 | +0.00011757175 | 9.17% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 1 | +6.4696434e-05 | 46.61% |
| 2 | -4.4147506e-05 | 21.70% |
| 4 | -3.1400975e-05 | 10.98% |
| 8 | -2.5759354e-05 | 7.39% |
| 3 | -2.0627513e-05 | 4.74% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | +0.023932158 | 47.77% |
| 3 | -0.019501233 | 31.72% |
| 1 | +0.012912804 | 13.91% |
| 0 | +0.007784664 | 5.05% |
| 4 | +0.0042800739 | 1.53% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | -0.059147308 | 60.75% |
| 2 | +0.04149099 | 29.89% |
| 3 | -0.016470439 | 4.71% |
| 1 | +0.016243639 | 4.58% |
| 4 | -0.0017501782 | 0.05% |

## Monte Carlo

Disabled for this run. The deterministic 27-point eigen-direction
analysis is the primary result.

