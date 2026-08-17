# Gimbalrotor PID sensitivity: within-bag vs between-bag

This comparison separates local sensitivity around each identified plant
from variation of the point estimates across bags. It is descriptive;
with only a few bags it is not a random-effects inference.

## Per-bag local sensitivity

| bag | group | center scale | within local 1-sigma | relative | eigen min | eigen max |
|---|---|---:|---:|---:|---:|---:|
| failure1 | xy | 1.1520448 | 0.0003850356 | 0.033% | 1.1518121 | 1.1528091 |
| failure1 | z | 1.1697648 | 9.4808533e-05 | 0.008% | 1.1696965 | 1.1698261 |
| failure1 | roll_pitch | 3.5287743 | 0.034451441 | 0.976% | 3.5039769 | 3.5939827 |
| failure1 | yaw | 3.3788679 | 0.079024678 | 2.339% | 3.3309613 | 3.4571853 |
| failure2 | xy | 1.1787878 | 0.00014118274 | 0.012% | 1.1787319 | 1.1799563 |
| failure2 | z | 1.1789924 | 0.00015273498 | 0.013% | 1.1788928 | 1.1793054 |
| failure2 | roll_pitch | 1.7244864 | 0.28171073 | 16.336% | 1.7107749 | 3.2615882 |
| failure2 | yaw | 1.9977971 | 0.20591642 | 10.307% | 1.9659625 | 8.2366858 |
| success | xy | 1.2145134 | 0.00010639014 | 0.009% | 1.2144209 | 1.2146044 |
| success | z | 1.2249249 | 4.4777575e-05 | 0.004% | 1.2248866 | 1.2249627 |
| success | roll_pitch | 4.1559028 | 0.015521961 | 0.373% | 4.1421173 | 4.1695717 |
| success | yaw | 2.4909143 | 0.0075630501 | 0.304% | 2.4848872 | 2.4969091 |

## Within-vs-between scale

| group | center min | center max | between-bag std | within-bag sigma RMS | between/within |
|---|---:|---:|---:|---:|---:|
| xy | 1.1520448 | 1.2145134 | 0.025590436 | 0.00024461115 | 104.617 |
| z | 1.1697648 | 1.2249249 | 0.024123723 | 0.00010696053 | 225.539 |
| roll_pitch | 1.7244864 | 4.1559028 | 1.0306703 | 0.16410237 | 6.28065 |
| yaw | 1.9977971 | 3.3788679 | 0.57144871 | 0.1274149 | 4.48494 |

Interpretation guide:

- `within-bag sigma RMS >> between-bag std`: the gain correction is
  locally weakly determined by each fitted plant.
- `between-bag std >> within-bag sigma RMS`: each fit may be locally
  sharp, while different bags identify different effective plants.
- Comparable values indicate that both effects matter.

Do not average the bag-specific gain proposals from this table.
