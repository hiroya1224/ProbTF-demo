# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `conservative_fusion`
- coordinate mode: `centered_scale_free_spd`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- finite sigma multiple: `2`
- local derivative sigma fraction: `1e-05`
- valid finite eigen samples: `23/27`
- valid local-derivative directions: `13/13`

## Gain-scale sensitivity

| group | center | infinitesimal 1-sigma | relative | finite secant 1-sigma | finite/local | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.1787878 | 0.0013486732 | 0.114% | incomplete | incomplete | 1.1768537 | 1.180706 |
| z | 1.1789924 | 0.0014498283 | 0.123% | incomplete | incomplete | 1.1770856 | 1.180893 |
| roll_pitch | 1.7244864 | 0.059730388 | 3.464% | incomplete | incomplete | 1.6152322 | 1.8377067 |
| yaw | 1.9977971 | 0.16111232 | 8.064% | incomplete | incomplete | 1.8103116 | 2.1897794 |

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
| 7 | +0.0009630781 | 50.99% |
| 8 | -0.00086536526 | 41.17% |
| 3 | -0.00021571481 | 2.56% |
| 2 | +0.00018607449 | 1.90% |
| 6 | +0.00014907586 | 1.22% |

### z

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 7 | +0.00095186172 | 43.10% |
| 8 | -0.00085840034 | 35.05% |
| 1 | -0.00048346165 | 11.12% |
| 5 | -0.00025873976 | 3.18% |
| 2 | +0.00025847563 | 3.18% |

### roll_pitch

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 3 | -0.055666276 | 86.85% |
| 2 | +0.015730411 | 6.94% |
| 0 | -0.014037175 | 5.52% |
| 1 | +0.0034584366 | 0.34% |
| 5 | -0.0028628324 | 0.23% |

### yaw

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 1 | +0.12384111 | 59.08% |
| 2 | +0.095835804 | 35.38% |
| 3 | -0.032346755 | 4.03% |
| 0 | -0.018763007 | 1.36% |
| 5 | +0.0050748897 | 0.10% |

## Monte Carlo

Disabled for this run. The deterministic eigen-direction
analysis remains available.

