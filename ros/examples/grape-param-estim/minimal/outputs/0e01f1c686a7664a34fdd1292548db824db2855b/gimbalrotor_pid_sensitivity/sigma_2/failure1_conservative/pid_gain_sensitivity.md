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
| xy | 1.1520448 | 0.0021922597 | 0.190% | 1.1486812 | 1.1604772 |
| z | 1.1697648 | 0.002064806 | 0.177% | 1.1661437 | 1.1733971 |
| roll_pitch | 3.5287743 | 0.081287027 | 2.304% | 3.4015842 | 4.4578924 |
| yaw | 3.3788679 | 0.39003972 | 11.544% | 2.7676319 | 4.2715695 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 8 | +0.0016837066 | 58.99% |
| 5 | +0.00086530919 | 15.58% |
| 0 | +0.00066364197 | 9.16% |
| 3 | +0.00047748256 | 4.74% |
| 6 | +0.00047344353 | 4.66% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 8 | +0.0018133471 | 77.13% |
| 5 | +0.00079280077 | 14.74% |
| 6 | +0.00036347735 | 3.10% |
| 7 | +0.00027619037 | 1.79% |
| 0 | +0.00022870802 | 1.23% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.057469882 | 49.98% |
| 4 | +0.03877969 | 22.76% |
| 0 | -0.030207859 | 13.81% |
| 3 | +0.028345564 | 12.16% |
| 1 | +0.008234272 | 1.03% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | -0.37598441 | 92.92% |
| 2 | -0.092021095 | 5.57% |
| 3 | +0.042294371 | 1.18% |
| 4 | +0.021460937 | 0.30% |
| 5 | +0.005871718 | 0.02% |

## Monte Carlo

Disabled for this run. The deterministic 27-point eigen-direction
analysis is the primary result.

