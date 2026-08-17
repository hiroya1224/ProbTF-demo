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
| xy | 1.2145134 | 0.00010639014 | 0.009% | 1.2144209 | 1.2146044 |
| z | 1.2249249 | 4.4777575e-05 | 0.004% | 1.2248866 | 1.2249627 |
| roll_pitch | 4.1559028 | 0.015521961 | 0.373% | 4.1421173 | 4.1695717 |
| yaw | 2.4909143 | 0.0075630501 | 0.304% | 2.4848872 | 2.4969091 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 5 | -9.176201e-05 | 74.39% |
| 1 | -3.0225515e-05 | 8.07% |
| 6 | +2.5246353e-05 | 5.63% |
| 2 | -2.4659048e-05 | 5.37% |
| 3 | -1.9609878e-05 | 3.40% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 1 | -3.8045903e-05 | 72.19% |
| 6 | +1.5601036e-05 | 12.14% |
| 2 | -1.4139061e-05 | 9.97% |
| 5 | -9.7391892e-06 | 4.73% |
| 9 | -2.3884613e-06 | 0.28% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.013727216 | 78.21% |
| 0 | -0.0053746156 | 11.99% |
| 5 | -0.0043598688 | 7.89% |
| 4 | +0.0013469937 | 0.75% |
| 1 | +0.0012738162 | 0.67% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 3 | -0.0060109558 | 63.17% |
| 1 | +0.0034466359 | 20.77% |
| 4 | -0.0021393234 | 8.00% |
| 5 | -0.0020302954 | 7.21% |
| 2 | -0.00063351847 | 0.70% |

## Optional Monte Carlo stress test

| group | mean | std | q16 | median | q84 | q2.5 | q97.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.2145314 | 0.00010924842 | 1.2144186 | 1.2145345 | 1.2146396 | 1.2143339 | 1.214753 |
| z | 1.2249248 | 4.4743646e-05 | 1.2248809 | 1.2249246 | 1.2249671 | 1.2248322 | 1.2250074 |
| roll_pitch | 4.1561322 | 0.015117396 | 4.1400602 | 4.1563213 | 4.1720422 | 4.1275597 | 4.1840866 |
| yaw | 2.4937766 | 0.0080139811 | 2.4861281 | 2.49333 | 2.5020151 | 2.4788809 | 2.5091524 |

Monte Carlo here is a nonlinear stress test of the selected local
covariance model. Do not relabel these quantiles as posterior
credible intervals without an independent probabilistic
calibration argument.

