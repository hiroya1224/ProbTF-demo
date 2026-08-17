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
| xy | 1.1787878 | 1.1795183 | [1.1789152, 1.1799298] | [1.1787362, 1.1800932] | 0.00042338955 | 0.000% |
| z | 1.1789924 | 1.1791531 | [1.1789592, 1.1793251] | [1.178739, 1.1794051] | 0.00017813188 | 0.000% |
| roll_pitch | 1.7244864 | 2.7301467 | [1.9646942, 3.272132] | [1.7305513, 3.660578] | 0.57187904 | 0.000% |
| yaw | 1.9977971 | 2.6282863 | [1.5341099, 6.3020452] | [0.13185361, 8.1825896] | 2.2638882 | 0.000% |

## Proposed PID gains

### xy

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 3 | 3.5385549 | [3.5367457, 3.5397893] | [3.5362087, 3.5402795] |
| i_gain | 0.1 | 0.11795183 | [0.11789152, 0.11799298] | [0.11787362, 0.11800932] |
| d_gain | 1 | 1.1795183 | [1.1789152, 1.1799298] | [1.1787362, 1.1800932] |

### z

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 5 | 5.8957656 | [5.8947958, 5.8966255] | [5.893695, 5.8970253] |
| i_gain | 1 | 1.1791531 | [1.1789592, 1.1793251] | [1.178739, 1.1794051] |
| d_gain | 2.5 | 2.9478828 | [2.9473979, 2.9483128] | [2.9468475, 2.9485126] |

### roll_pitch

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 10 | 27.301467 | [19.646942, 32.72132] | [17.305513, 36.60578] |
| i_gain | 1 | 2.7301467 | [1.9646942, 3.272132] | [1.7305513, 3.660578] |
| d_gain | 8 | 21.841174 | [15.717554, 26.177056] | [13.84441, 29.284624] |

### yaw

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 4 | 10.513145 | [6.1364395, 25.208181] | [0.52741445, 32.730358] |
| i_gain | 1 | 2.6282863 | [1.5341099, 6.3020452] | [0.13185361, 8.1825896] |
| d_gain | 2 | 5.2565727 | [3.0682198, 12.60409] | [0.26370722, 16.365179] |

## Joint gain-scale correlation

| | xy | z | roll_pitch | yaw |
|---|---:|---:|---:|---:|
| xy | 1 | 0.68743 | 0.74249 | 0.34522 |
| z | 0.68743 | 1 | 0.60678 | 0.18933 |
| roll_pitch | 0.74249 | 0.60678 | 1 | 0.53971 |
| yaw | 0.34522 | 0.18933 | 0.53971 | 1 |

## Numerical diagnostics

- source-threshold rank histogram: `{'4': 212, '5': 33, '6': 9755}`
- infinite allocation-condition fraction: `0`
- max finite allocation condition number: `2218054796.301993`
- warnings: `['source_threshold_rank_loss_present']`

The ranges above are empirical quantiles of the selected estimator
Gaussian approximation after nonlinear PID postprocessing. No sample is
discarded merely because `A_real` loses the source SVD-threshold rank or
has a large condition number; only a genuinely undefined/non-finite
floating-point evaluation is marked invalid.
