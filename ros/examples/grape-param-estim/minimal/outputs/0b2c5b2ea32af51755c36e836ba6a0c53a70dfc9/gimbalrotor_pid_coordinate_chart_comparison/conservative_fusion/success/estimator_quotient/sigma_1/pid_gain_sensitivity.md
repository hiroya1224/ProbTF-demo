# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `conservative_fusion`
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
| xy | 1.2145134 | 0.00089749197 | 0.074% | 0.00089748692 | 0.99999437 | 1.2139217 | 1.2150535 |
| z | 1.2249249 | 0.00051431172 | 0.042% | 0.0005142981 | 0.99997351 | 1.2244594 | 1.2253465 |
| roll_pitch | 4.1559028 | 0.059446995 | 1.430% | 0.059425232 | 0.99963391 | 4.1090928 | 4.2004827 |
| yaw | 2.4909143 | 0.065167698 | 2.616% | 0.065129247 | 0.99940996 | 2.430711 | 2.5501634 |

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
| 4 | -0.00056588153 | 39.75% |
| 2 | -0.00048569928 | 29.29% |
| 6 | -0.00028326219 | 9.96% |
| 1 | +0.00028289 | 9.94% |
| 9 | -0.00020482019 | 5.21% |

### z

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 4 | -0.00044355163 | 74.38% |
| 9 | -0.000205815 | 16.01% |
| 1 | +9.5786024e-05 | 3.47% |
| 2 | -8.0864104e-05 | 2.47% |
| 7 | -6.0634997e-05 | 1.39% |

### roll_pitch

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 3 | +0.045715242 | 59.14% |
| 6 | -0.028706727 | 23.32% |
| 2 | -0.017622483 | 8.79% |
| 0 | -0.011712043 | 3.88% |
| 5 | -0.0077814143 | 1.71% |

### yaw

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.059756814 | 84.08% |
| 1 | +0.018321196 | 7.90% |
| 0 | +0.012722802 | 3.81% |
| 5 | -0.0095458349 | 2.15% |
| 6 | -0.0066477256 | 1.04% |

## Monte Carlo

Disabled for this run. The deterministic eigen-direction
analysis remains available.

