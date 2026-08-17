# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `conservative_fusion`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- valid eigen samples: `27/27`

## Gain-scale sensitivity

| group | center | local linear 1-sigma | relative | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|
| xy | 1.1787878 | 0.0013716906 | 0.116% | 1.1782383 | 1.1799361 |
| z | 1.1789924 | 0.0014423464 | 0.122% | 1.1784462 | 1.1795388 |
| roll_pitch | 1.7244864 | 0.51483683 | 29.855% | 1.6963833 | 3.2915217 |
| yaw | 1.9977971 | 0.2435945 | 12.193% | 1.9497639 | 8.1898497 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 8 | +0.0010991224 | 64.21% |
| 7 | +0.00067355032 | 24.11% |
| 3 | +0.00027174829 | 3.92% |
| 0 | +0.00024773448 | 3.26% |
| 2 | -0.00018290573 | 1.78% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 8 | +0.0010926089 | 57.38% |
| 7 | +0.00066741891 | 21.41% |
| 1 | +0.00048580206 | 11.34% |
| 2 | -0.0002587272 | 3.22% |
| 3 | +0.00021290319 | 2.18% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | +0.5115534 | 98.73% |
| 3 | +0.056515192 | 1.21% |
| 2 | -0.012322917 | 0.06% |
| 4 | +0.0034827543 | 0.00% |
| 1 | -0.0031509144 | 0.00% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | -0.18096544 | 55.19% |
| 1 | -0.12703261 | 27.20% |
| 2 | -0.094536768 | 15.06% |
| 3 | +0.03787937 | 2.42% |
| 5 | +0.0071822671 | 0.09% |

## Monte Carlo

Disabled for this run. The deterministic 27-point eigen-direction
analysis is the primary result.

