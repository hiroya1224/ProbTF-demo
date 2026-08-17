# Gimbalrotor PID sensitivity: within-bag vs between-bag

This comparison separates local sensitivity around each identified plant
from variation of the point estimates across bags. It is descriptive;
with only a few bags it is not a random-effects inference.

## Per-bag local sensitivity

| bag | group | center scale | within local 1-sigma | relative | eigen min | eigen max |
|---|---|---:|---:|---:|---:|---:|
| failure1 | xy | 1.1520448 | 0.00038830531 | 0.034% | 1.151929 | 1.15229 |
| failure1 | z | 1.1697648 | 9.4766054e-05 | 0.008% | 1.1697316 | 1.1697963 |
| failure1 | roll_pitch | 3.5287743 | 0.034627836 | 0.981% | 3.5165915 | 3.5473211 |
| failure1 | yaw | 3.3788679 | 0.075888276 | 2.246% | 3.3531736 | 3.4123209 |
| failure2 | xy | 1.1787878 | 0.00010225762 | 0.009% | 1.1787608 | 1.1794632 |
| failure2 | z | 1.1789924 | 0.00015293091 | 0.013% | 1.1789451 | 1.1792373 |
| failure2 | roll_pitch | 1.7244864 | 0.20588133 | 11.939% | 1.7176222 | 3.2522643 |
| failure2 | yaw | 1.9977971 | 0.052337987 | 2.620% | 1.9818527 | 3.1410974 |
| success | xy | 1.2145134 | 0.0001063908 | 0.009% | 1.2144673 | 1.2145591 |
| success | z | 1.2249249 | 4.4777693e-05 | 0.004% | 1.2249058 | 1.2249439 |
| success | roll_pitch | 4.1559028 | 0.015521991 | 0.373% | 4.1490245 | 4.162752 |
| success | yaw | 2.4909143 | 0.0075630512 | 0.304% | 2.4879048 | 2.4939158 |

## Within-vs-between scale

| group | center min | center max | between-bag std | within-bag sigma RMS | between/within |
|---|---:|---:|---:|---:|---:|
| xy | 1.1520448 | 1.2145134 | 0.025590436 | 0.00023983094 | 106.702 |
| z | 1.1697648 | 1.2249249 | 0.024123723 | 0.00010704128 | 225.368 |
| roll_pitch | 1.7244864 | 4.1559028 | 1.0306703 | 0.12086789 | 8.52725 |
| yaw | 1.9977971 | 3.3788679 | 0.57144871 | 0.053402544 | 10.7008 |

Interpretation guide:

- `within-bag sigma RMS >> between-bag std`: the gain correction is
  locally weakly determined by each fitted plant.
- `between-bag std >> within-bag sigma RMS`: each fit may be locally
  sharp, while different bags identify different effective plants.
- Comparable values indicate that both effects matter.

Do not average the bag-specific gain proposals from this table.
