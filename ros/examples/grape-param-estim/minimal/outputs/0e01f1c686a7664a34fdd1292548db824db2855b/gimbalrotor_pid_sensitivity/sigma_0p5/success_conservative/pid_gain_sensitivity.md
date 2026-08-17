# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `conservative_fusion`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- valid eigen samples: `27/27`

## Gain-scale sensitivity

| group | center | local linear 1-sigma | relative | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|
| xy | 1.2145134 | 0.00089748883 | 0.074% | 1.214224 | 1.2147899 |
| z | 1.2249249 | 0.00051430831 | 0.042% | 1.2246977 | 1.2251412 |
| roll_pitch | 4.1559028 | 0.059441489 | 1.430% | 4.1327689 | 4.1784791 |
| yaw | 2.4909143 | 0.065158181 | 2.616% | 2.4609205 | 2.5206696 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 4 | -0.00056587681 | 39.75% |
| 2 | -0.00048587008 | 29.31% |
| 6 | -0.00028325794 | 9.96% |
| 1 | +0.00028284062 | 9.93% |
| 9 | -0.0002048202 | 5.21% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 4 | -0.00044354858 | 74.38% |
| 9 | -0.00020581499 | 16.01% |
| 1 | +9.578445e-05 | 3.47% |
| 2 | -8.0865147e-05 | 2.47% |
| 7 | -6.0635033e-05 | 1.39% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 3 | +0.045710172 | 59.14% |
| 6 | -0.028706526 | 23.32% |
| 2 | -0.017628249 | 8.80% |
| 0 | -0.011694865 | 3.87% |
| 5 | -0.0077813236 | 1.71% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.059749165 | 84.09% |
| 1 | +0.018319419 | 7.90% |
| 0 | +0.012713067 | 3.81% |
| 5 | -0.0095453724 | 2.15% |
| 6 | -0.0066476704 | 1.04% |

## Monte Carlo

Disabled for this run. The deterministic 27-point eigen-direction
analysis is the primary result.

