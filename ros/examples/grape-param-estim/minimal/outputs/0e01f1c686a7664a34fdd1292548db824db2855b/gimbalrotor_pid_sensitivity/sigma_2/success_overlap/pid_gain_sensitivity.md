# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `overlap_corrected`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- valid eigen samples: `27/27`

## Gain-scale sensitivity

| group | center | local linear 1-sigma | relative | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|
| xy | 1.2145134 | 0.00010638753 | 0.009% | 1.2143269 | 1.2146939 |
| z | 1.2249249 | 4.4777104e-05 | 0.004% | 1.2248477 | 1.2249999 |
| roll_pitch | 4.1559028 | 0.015521834 | 0.373% | 4.1282171 | 4.1831218 |
| yaw | 2.4909143 | 0.0075630458 | 0.304% | 2.4788283 | 2.5028713 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 5 | -9.1761213e-05 | 74.39% |
| 1 | -3.0222397e-05 | 8.07% |
| 6 | +2.5246347e-05 | 5.63% |
| 2 | -2.4658309e-05 | 5.37% |
| 3 | -1.9611059e-05 | 3.40% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 1 | -3.8045305e-05 | 72.19% |
| 6 | +1.560103e-05 | 12.14% |
| 2 | -1.4139243e-05 | 9.97% |
| 5 | -9.7389504e-06 | 4.73% |
| 9 | -2.3884613e-06 | 0.28% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.013726165 | 78.20% |
| 0 | -0.0053762136 | 12.00% |
| 5 | -0.0043598548 | 7.89% |
| 4 | +0.0013468045 | 0.75% |
| 1 | +0.0012771254 | 0.68% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 3 | -0.0060107517 | 63.16% |
| 1 | +0.0034470148 | 20.77% |
| 4 | -0.0021392283 | 8.00% |
| 5 | -0.0020302892 | 7.21% |
| 2 | -0.00063358095 | 0.70% |

## Monte Carlo

Disabled for this run. The deterministic 27-point eigen-direction
analysis is the primary result.

