# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `overlap_corrected`
- coordinate mode: `estimator_quotient`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- finite sigma multiple: `2`
- local derivative sigma fraction: `1e-05`
- valid finite eigen samples: `27/27`
- valid local-derivative directions: `13/13`

## Gain-scale sensitivity

| group | center | infinitesimal 1-sigma | relative | finite secant 1-sigma | finite/local | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.1520448 | 0.00038954053 | 0.034% | 0.0003772969 | 0.96856905 | 1.1515747 | 1.154526 |
| z | 1.1697648 | 9.4750939e-05 | 0.008% | 9.4940763e-05 | 1.0020034 | 1.1696206 | 1.1698806 |
| roll_pitch | 3.5287743 | 0.03469435 | 0.983% | 0.034023237 | 0.98065643 | 3.4774616 | 3.76433 |
| yaw | 3.3788679 | 0.074874367 | 2.216% | 0.092844752 | 1.2400072 | 3.2753771 | 3.594216 |

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
| 4 | -0.00023029821 | 34.95% |
| 0 | +0.00020201905 | 26.90% |
| 3 | +0.00014585294 | 14.02% |
| 1 | +0.00011899497 | 9.33% |
| 2 | +0.00011754191 | 9.11% |

### z

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 1 | +6.467572e-05 | 46.59% |
| 2 | -4.4150639e-05 | 21.71% |
| 4 | -3.1401093e-05 | 10.98% |
| 8 | -2.5759339e-05 | 7.39% |
| 3 | -2.0627611e-05 | 4.74% |

### roll_pitch

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | +0.023932695 | 47.58% |
| 3 | -0.019501284 | 31.59% |
| 1 | +0.012915887 | 13.86% |
| 0 | +0.0080687212 | 5.41% |
| 4 | +0.0042801174 | 1.52% |

### yaw

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | -0.057843976 | 59.68% |
| 2 | +0.041493858 | 30.71% |
| 3 | -0.016470514 | 4.84% |
| 1 | +0.016224492 | 4.70% |
| 4 | -0.0017501724 | 0.05% |

## Monte Carlo

Disabled for this run. The deterministic eigen-direction
analysis remains available.

