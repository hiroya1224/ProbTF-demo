# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `conservative_fusion`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- valid eigen samples: `25/27`

## Gain-scale sensitivity

| group | center | local linear 1-sigma | relative | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|
| xy | 1.1787878 | 0.0013557747 | 0.115% | 1.1765904 | 1.1809868 |
| z | 1.1789924 | 0.0014334394 | 0.122% | 1.1768091 | 1.1811795 |
| roll_pitch | 1.7244864 | 0.058066733 | 3.367% | 1.6139664 | 1.8399415 |
| yaw | 1.9977971 | 0.21978972 | 11.002% | 1.8127675 | 2.6224101 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 8 | +0.0010991203 | 65.72% |
| 7 | +0.00067354857 | 24.68% |
| 3 | +0.00027916172 | 4.24% |
| 2 | -0.00018455234 | 1.85% |
| 1 | +0.00018031008 | 1.77% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 8 | +0.0010926093 | 58.10% |
| 7 | +0.00066741687 | 21.68% |
| 1 | +0.00048858799 | 11.62% |
| 2 | -0.00025916167 | 3.27% |
| 3 | +0.00021288504 | 2.21% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 3 | +0.056493772 | 94.66% |
| 2 | -0.01285758 | 4.90% |
| 4 | +0.0034815285 | 0.36% |
| 5 | -0.0012463703 | 0.05% |
| 1 | -0.00075649184 | 0.02% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 1 | -0.19500254 | 78.72% |
| 2 | -0.093622816 | 18.14% |
| 3 | +0.037893315 | 2.97% |
| 5 | +0.0071806711 | 0.11% |
| 4 | +0.0053252751 | 0.06% |

## Monte Carlo

Disabled for this run. The deterministic 27-point eigen-direction
analysis is the primary result.

