# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `overlap_corrected`
- coordinate mode: `estimator_quotient`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- finite sigma multiple: `1`
- local derivative sigma fraction: `1e-05`
- valid finite eigen samples: `27/27`
- valid local-derivative directions: `13/13`

## Gain-scale sensitivity

| group | center | infinitesimal 1-sigma | relative | finite secant 1-sigma | finite/local | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.2145134 | 0.00010639102 | 0.009% | 0.00010639014 | 0.99999178 | 1.2144209 | 1.2146044 |
| z | 1.2249249 | 4.4777733e-05 | 0.004% | 4.4777575e-05 | 0.99999648 | 1.2248866 | 1.2249627 |
| roll_pitch | 4.1559028 | 0.015522 | 0.373% | 0.015521961 | 0.99999745 | 4.1421173 | 4.1695717 |
| yaw | 2.4909143 | 0.0075630516 | 0.304% | 0.0075630501 | 0.9999998 | 2.4848872 | 2.4969091 |

The infinitesimal value uses a dedicated small centered finite
difference and is independent of the requested finite sigma
excursion up to numerical differentiation error. The finite
secant value uses the requested +/- k-sigma points. A
`finite/local` ratio far from one therefore measures chart/output
nonlinearity over that finite excursion rather than changing the
definition of the local covariance.

Sampled `A_real` rank loss or a large condition number is retained
as a diagnostic. A sample is only marked invalid after the
floating-point calculation itself becomes non-finite or otherwise
mathematically undefined.

## Dominant local covariance directions

### xy

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 5 | -9.1762287e-05 | 74.39% |
| 1 | -3.0226521e-05 | 8.07% |
| 6 | +2.5246383e-05 | 5.63% |
| 2 | -2.4659275e-05 | 5.37% |
| 3 | -1.960947e-05 | 3.40% |

### z

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 1 | -3.8046111e-05 | 72.19% |
| 6 | +1.5601043e-05 | 12.14% |
| 2 | -1.413899e-05 | 9.97% |
| 5 | -9.7392427e-06 | 4.73% |
| 9 | -2.388445e-06 | 0.28% |

### roll_pitch

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.013727566 | 78.22% |
| 0 | -0.0053740757 | 11.99% |
| 5 | -0.0043598729 | 7.89% |
| 4 | +0.0013470564 | 0.75% |
| 1 | +0.0012727121 | 0.67% |

### yaw

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 3 | -0.0060110238 | 63.17% |
| 1 | +0.0034465095 | 20.77% |
| 4 | -0.0021393552 | 8.00% |
| 5 | -0.0020302974 | 7.21% |
| 2 | -0.00063349754 | 0.70% |

## Monte Carlo

Disabled for this run. The deterministic eigen-direction
analysis remains available.

