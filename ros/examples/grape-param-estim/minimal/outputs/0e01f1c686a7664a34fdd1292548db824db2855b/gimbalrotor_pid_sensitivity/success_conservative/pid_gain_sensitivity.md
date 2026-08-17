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
| xy | 1.2145134 | 0.00089748692 | 0.074% | 1.2139217 | 1.2150535 |
| z | 1.2249249 | 0.0005142981 | 0.042% | 1.2244594 | 1.2253465 |
| roll_pitch | 4.1559028 | 0.059425232 | 1.430% | 4.1090928 | 4.2004827 |
| yaw | 2.4909143 | 0.065129247 | 2.615% | 2.430711 | 2.5501634 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 4 | -0.00056586262 | 39.75% |
| 2 | -0.00048638275 | 29.37% |
| 6 | -0.00028324522 | 9.96% |
| 1 | +0.00028269262 | 9.92% |
| 9 | -0.0002048202 | 5.21% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 4 | -0.0004435394 | 74.38% |
| 9 | -0.00020581499 | 16.01% |
| 1 | +9.5779718e-05 | 3.47% |
| 2 | -8.0868302e-05 | 2.47% |
| 7 | -6.0635132e-05 | 1.39% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 3 | +0.045694968 | 59.13% |
| 6 | -0.028705924 | 23.33% |
| 2 | -0.017645542 | 8.82% |
| 0 | -0.011644582 | 3.84% |
| 5 | -0.0077810514 | 1.71% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.059726227 | 84.10% |
| 1 | +0.018314093 | 7.91% |
| 0 | +0.01268181 | 3.79% |
| 5 | -0.0095439852 | 2.15% |
| 6 | -0.0066475047 | 1.04% |

## Optional Monte Carlo stress test

| group | mean | std | q16 | median | q84 | q2.5 | q97.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.2146723 | 0.0009244841 | 1.2137777 | 1.2147091 | 1.2155369 | 1.2127237 | 1.2164777 |
| z | 1.2248831 | 0.00052061037 | 1.2243553 | 1.2249172 | 1.2253679 | 1.2237824 | 1.2258408 |
| roll_pitch | 4.1556926 | 0.054462283 | 4.1011625 | 4.1548746 | 4.205554 | 4.0493292 | 4.273106 |
| yaw | 2.5204098 | 0.07096977 | 2.4476325 | 2.5170185 | 2.593547 | 2.3850569 | 2.6609364 |

Monte Carlo here is a nonlinear stress test of the selected local
covariance model. Do not relabel these quantiles as posterior
credible intervals without an independent probabilistic
calibration argument.

