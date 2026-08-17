# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `conservative_fusion`
- coordinate mode: `centered_scale_free_spd`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- finite sigma multiple: `0.5`
- local derivative sigma fraction: `1e-05`
- valid finite eigen samples: `25/27`
- valid local-derivative directions: `13/13`

## Gain-scale sensitivity

| group | center | infinitesimal 1-sigma | relative | finite secant 1-sigma | finite/local | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.2145134 | 0.0008974921 | 0.074% | incomplete | incomplete | 1.214229 | 1.2181889 |
| z | 1.2249249 | 0.00051431196 | 0.042% | incomplete | incomplete | 1.2246984 | 1.2251404 |
| roll_pitch | 4.1559028 | 0.059446997 | 1.430% | incomplete | incomplete | 3.084786 | 4.1744213 |
| yaw | 2.4909143 | 0.065167701 | 2.616% | incomplete | incomplete | 1.3927758 | 2.5171649 |

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
| 5 | +0.000555555 | 38.32% |
| 2 | +0.00040705199 | 20.57% |
| 6 | +0.00027862107 | 9.64% |
| 0 | -0.00024713819 | 7.58% |
| 4 | -0.00024307143 | 7.34% |

### z

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 5 | +0.00044202381 | 73.86% |
| 9 | +0.00020624665 | 16.08% |
| 4 | -7.6922502e-05 | 2.24% |
| 0 | -7.4887618e-05 | 2.12% |
| 2 | +6.9445338e-05 | 1.82% |

### roll_pitch

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 3 | -0.037476835 | 39.74% |
| 2 | +0.028196308 | 22.50% |
| 6 | +0.027294995 | 21.08% |
| 4 | -0.018893748 | 10.10% |
| 0 | +0.012373117 | 4.33% |

### yaw

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | +0.053076305 | 66.33% |
| 3 | +0.025773903 | 15.64% |
| 0 | -0.020170513 | 9.58% |
| 4 | -0.01712472 | 6.91% |
| 1 | +0.0057181621 | 0.77% |

## Monte Carlo

Disabled for this run. The deterministic eigen-direction
analysis remains available.

