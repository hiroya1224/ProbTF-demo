# Gimbalrotor PID Monte Carlo proposal

The estimator Gaussian is sampled in its native common-scale quotient
coordinate and propagated through the nonlinear static PID map. The PID
output is retained as an empirical distribution rather than re-Gaussianized.

- covariance mode: `overlap_corrected`
- coordinate mode: `estimator_quotient`
- samples: `10000/10000` valid
- seed: `0`

## Gain-scale distribution

| group | point | median | 16–84% | 2.5–97.5% | std | nonpositive |
|---|---:|---:|---:|---:|---:|---:|
| xy | 1.1520448 | 1.1524299 | [1.1519562, 1.1532894] | [1.1515701, 1.1548052] | 0.00081540981 | 0.000% |
| z | 1.1697648 | 1.1697693 | [1.169667, 1.1698553] | [1.1695407, 1.1699294] | 9.8898184e-05 | 0.000% |
| roll_pitch | 3.5287743 | 3.5676111 | [3.5206373, 3.6546298] | [3.482771, 3.8199102] | 0.085709876 | 0.000% |
| yaw | 3.3788679 | 3.3917213 | [3.3172318, 3.4813114] | [3.2480234, 3.613748] | 0.092576487 | 0.000% |

## Proposed PID gains

### xy

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 3 | 3.4572897 | [3.4558685, 3.4598683] | [3.4547102, 3.4644157] |
| i_gain | 0.1 | 0.11524299 | [0.11519562, 0.11532894] | [0.11515701, 0.11548052] |
| d_gain | 1 | 1.1524299 | [1.1519562, 1.1532894] | [1.1515701, 1.1548052] |

### z

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 5 | 5.8488464 | [5.8483348, 5.8492763] | [5.8477036, 5.849647] |
| i_gain | 1 | 1.1697693 | [1.169667, 1.1698553] | [1.1695407, 1.1699294] |
| d_gain | 2.5 | 2.9244232 | [2.9241674, 2.9246381] | [2.9238518, 2.9248235] |

### roll_pitch

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 20 | 71.352221 | [70.412746, 73.092595] | [69.655419, 76.398205] |
| i_gain | 1 | 3.5676111 | [3.5206373, 3.6546298] | [3.482771, 3.8199102] |
| d_gain | 8 | 28.540888 | [28.165098, 29.237038] | [27.862168, 30.559282] |

### yaw

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 4 | 13.566885 | [13.268927, 13.925246] | [12.992093, 14.454992] |
| i_gain | 1 | 3.3917213 | [3.3172318, 3.4813114] | [3.2480234, 3.613748] |
| d_gain | 2 | 6.7834426 | [6.6344637, 6.9626228] | [6.4960467, 7.2274959] |

## Joint gain-scale correlation

| | xy | z | roll_pitch | yaw |
|---|---:|---:|---:|---:|
| xy | 1 | 0.16745 | 0.83015 | 0.061057 |
| z | 0.16745 | 1 | 0.132 | 0.11386 |
| roll_pitch | 0.83015 | 0.132 | 1 | 0.34913 |
| yaw | 0.061057 | 0.11386 | 0.34913 | 1 |

## Numerical diagnostics

- source-threshold rank histogram: `{'6': 10000}`
- infinite allocation-condition fraction: `0`
- max finite allocation condition number: `4.729437148975091`
- warnings: `[]`

The ranges above are empirical quantiles of the selected estimator
Gaussian approximation after nonlinear PID postprocessing. No sample is
discarded merely because `A_real` loses the source SVD-threshold rank or
has a large condition number; only a genuinely undefined/non-finite
floating-point evaluation is marked invalid.
