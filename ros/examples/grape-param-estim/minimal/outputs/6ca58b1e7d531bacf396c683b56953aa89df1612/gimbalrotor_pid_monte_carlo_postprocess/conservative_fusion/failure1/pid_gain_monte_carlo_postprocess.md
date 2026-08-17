# Gimbalrotor PID Monte Carlo proposal

The estimator Gaussian is sampled in its native common-scale quotient
coordinate and propagated through the nonlinear static PID map. The PID
output is retained as an empirical distribution rather than re-Gaussianized.

- covariance mode: `conservative_fusion`
- coordinate mode: `estimator_quotient`
- samples: `10000/10000` valid
- seed: `0`

## Gain-scale distribution

| group | point | median | 16–84% | 2.5–97.5% | std | nonpositive |
|---|---:|---:|---:|---:|---:|---:|
| xy | 1.1520448 | 1.1535933 | [1.1510652, 1.1567788] | [1.1487391, 1.1609327] | 0.0030843038 | 0.000% |
| z | 1.1697648 | 1.1696583 | [1.1676269, 1.1716758] | [1.1655895, 1.1736699] | 0.0020566869 | 0.000% |
| roll_pitch | 3.5287743 | 3.6574536 | [3.5284331, 3.9730963] | [3.4329636, 4.5438843] | 0.30275729 | 0.000% |
| yaw | 3.3788679 | 3.4196526 | [3.1880616, 3.7136191] | [2.7179577, 4.3402056] | 0.37264932 | 0.000% |

## Proposed PID gains

### xy

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 3 | 3.46078 | [3.4531957, 3.4703365] | [3.4462174, 3.4827982] |
| i_gain | 0.1 | 0.11535933 | [0.11510652, 0.11567788] | [0.11487391, 0.11609327] |
| d_gain | 1 | 1.1535933 | [1.1510652, 1.1567788] | [1.1487391, 1.1609327] |

### z

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 5 | 5.8482917 | [5.8381346, 5.8583791] | [5.8279475, 5.8683497] |
| i_gain | 1 | 1.1696583 | [1.1676269, 1.1716758] | [1.1655895, 1.1736699] |
| d_gain | 2.5 | 2.9241458 | [2.9190673, 2.9291896] | [2.9139737, 2.9341748] |

### roll_pitch

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 20 | 73.149072 | [70.568662, 79.461927] | [68.659271, 90.877686] |
| i_gain | 1 | 3.6574536 | [3.5284331, 3.9730963] | [3.4329636, 4.5438843] |
| d_gain | 8 | 29.259629 | [28.227465, 31.784771] | [27.463709, 36.351075] |

### yaw

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 4 | 13.67861 | [12.752246, 14.854476] | [10.871831, 17.360822] |
| i_gain | 1 | 3.4196526 | [3.1880616, 3.7136191] | [2.7179577, 4.3402056] |
| d_gain | 2 | 6.8393052 | [6.3761231, 7.4272382] | [5.4359153, 8.6804111] |

## Joint gain-scale correlation

| | xy | z | roll_pitch | yaw |
|---|---:|---:|---:|---:|
| xy | 1 | 0.66676 | 0.63624 | -0.034928 |
| z | 0.66676 | 1 | 0.04056 | -0.057692 |
| roll_pitch | 0.63624 | 0.04056 | 1 | 0.30796 |
| yaw | -0.034928 | -0.057692 | 0.30796 | 1 |

## Numerical diagnostics

- source-threshold rank histogram: `{'6': 10000}`
- infinite allocation-condition fraction: `0`
- max finite allocation condition number: `11.174050827009152`
- warnings: `[]`

The ranges above are empirical quantiles of the selected estimator
Gaussian approximation after nonlinear PID postprocessing. No sample is
discarded merely because `A_real` loses the source SVD-threshold rank or
has a large condition number; only a genuinely undefined/non-finite
floating-point evaluation is marked invalid.
