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
| xy | 1.1520448 | 0.0021797801 | 0.189% | 1.1512032 | 1.1529724 |
| z | 1.1697648 | 0.0020644029 | 0.176% | 1.1688585 | 1.1706718 |
| roll_pitch | 3.5287743 | 0.077682027 | 2.201% | 3.4992207 | 3.6002626 |
| yaw | 3.3788679 | 0.2084419 | 6.169% | 3.3031828 | 3.4837857 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 8 | +0.0016837072 | 59.66% |
| 5 | +0.00086535684 | 15.76% |
| 0 | +0.00062132439 | 8.12% |
| 3 | +0.0004778438 | 4.81% |
| 6 | +0.00047346198 | 4.72% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 8 | +0.0018133444 | 77.16% |
| 5 | +0.00079282796 | 14.75% |
| 6 | +0.00036348097 | 3.10% |
| 7 | +0.00027619071 | 1.79% |
| 0 | +0.00022365903 | 1.17% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.057569748 | 54.92% |
| 4 | +0.038793412 | 24.94% |
| 3 | +0.028406386 | 13.37% |
| 0 | -0.018982716 | 5.97% |
| 1 | +0.005557143 | 0.51% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | -0.18060283 | 75.07% |
| 2 | -0.092314268 | 19.61% |
| 3 | +0.042473841 | 4.15% |
| 4 | +0.021461087 | 1.06% |
| 5 | +0.0058733136 | 0.08% |

## Monte Carlo

Disabled for this run. The deterministic 27-point eigen-direction
analysis is the primary result.

