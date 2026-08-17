# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `overlap_corrected`
- coordinate mode: `estimator_quotient`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- finite sigma multiple: `0.5`
- local derivative sigma fraction: `1e-05`
- valid finite eigen samples: `27/27`
- valid local-derivative directions: `13/13`

## Gain-scale sensitivity

| group | center | infinitesimal 1-sigma | relative | finite secant 1-sigma | finite/local | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.1787878 | 8.4205977e-05 | 0.007% | 0.00010225762 | 1.2143748 | 1.1787608 | 1.1794632 |
| z | 1.1789924 | 0.00017125447 | 0.015% | 0.00015293091 | 0.89300391 | 1.1789451 | 1.1792373 |
| roll_pitch | 1.7244864 | 0.015122414 | 0.877% | 0.20588133 | 13.614316 | 1.7176222 | 3.2522643 |
| yaw | 1.9977971 | 0.049890971 | 2.497% | 0.052337987 | 1.0490473 | 1.9818527 | 3.1410974 |

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
| 0 | +5.3350624e-05 | 40.14% |
| 2 | -5.1847782e-05 | 37.91% |
| 3 | +2.4607127e-05 | 8.54% |
| 4 | +2.2494751e-05 | 7.14% |
| 9 | +1.1926959e-05 | 2.01% |

### z

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | +0.00011871505 | 48.05% |
| 1 | +8.9609831e-05 | 27.38% |
| 2 | -6.4626071e-05 | 14.24% |
| 4 | +4.5824133e-05 | 7.16% |
| 5 | +2.4237845e-05 | 2.00% |

### roll_pitch

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 3 | +0.013745453 | 82.62% |
| 2 | -0.0047293711 | 9.78% |
| 1 | -0.0031543265 | 4.35% |
| 4 | -0.0024279379 | 2.58% |
| 0 | -0.0011968267 | 0.63% |

### yaw

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.031939506 | 40.98% |
| 1 | -0.030275028 | 36.82% |
| 0 | +0.021549266 | 18.66% |
| 3 | +0.0090251655 | 3.27% |
| 4 | -0.0023195189 | 0.22% |

## Monte Carlo

Disabled for this run. The deterministic eigen-direction
analysis remains available.

