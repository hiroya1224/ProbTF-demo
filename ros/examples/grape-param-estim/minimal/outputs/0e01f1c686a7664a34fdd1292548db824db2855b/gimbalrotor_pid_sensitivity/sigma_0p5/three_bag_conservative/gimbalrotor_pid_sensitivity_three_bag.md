# Gimbalrotor PID sensitivity: within-bag vs between-bag

This comparison separates local sensitivity around each identified plant
from variation of the point estimates across bags. It is descriptive;
with only a few bags it is not a random-effects inference.

## Per-bag local sensitivity

| bag | group | center scale | within local 1-sigma | relative | eigen min | eigen max |
|---|---|---:|---:|---:|---:|---:|
| failure1 | xy | 1.1520448 | 0.0021797801 | 0.189% | 1.1512032 | 1.1529724 |
| failure1 | z | 1.1697648 | 0.0020644029 | 0.176% | 1.1688585 | 1.1706718 |
| failure1 | roll_pitch | 3.5287743 | 0.077682027 | 2.201% | 3.4992207 | 3.6002626 |
| failure1 | yaw | 3.3788679 | 0.2084419 | 6.169% | 3.3031828 | 3.4837857 |
| failure2 | xy | 1.1787878 | 0.0013716906 | 0.116% | 1.1782383 | 1.1799361 |
| failure2 | z | 1.1789924 | 0.0014423464 | 0.122% | 1.1784462 | 1.1795388 |
| failure2 | roll_pitch | 1.7244864 | 0.51483683 | 29.855% | 1.6963833 | 3.2915217 |
| failure2 | yaw | 1.9977971 | 0.2435945 | 12.193% | 1.9497639 | 8.1898497 |
| success | xy | 1.2145134 | 0.00089748883 | 0.074% | 1.214224 | 1.2147899 |
| success | z | 1.2249249 | 0.00051430831 | 0.042% | 1.2246977 | 1.2251412 |
| success | roll_pitch | 4.1559028 | 0.059441489 | 1.430% | 4.1327689 | 4.1784791 |
| success | yaw | 2.4909143 | 0.065158181 | 2.616% | 2.4609205 | 2.5206696 |

## Within-vs-between scale

| group | center min | center max | between-bag std | within-bag sigma RMS | between/within |
|---|---:|---:|---:|---:|---:|
| xy | 1.1520448 | 1.2145134 | 0.025590436 | 0.0015746389 | 16.2516 |
| z | 1.1697648 | 1.2249249 | 0.024123723 | 0.0014839851 | 16.256 |
| roll_pitch | 1.7244864 | 4.1559028 | 1.0306703 | 0.3025584 | 3.40652 |
| yaw | 1.9977971 | 3.3788679 | 0.57144871 | 0.18888435 | 3.02539 |

Interpretation guide:

- `within-bag sigma RMS >> between-bag std`: the gain correction is
  locally weakly determined by each fitted plant.
- `between-bag std >> within-bag sigma RMS`: each fit may be locally
  sharp, while different bags identify different effective plants.
- Comparable values indicate that both effects matter.

Do not average the bag-specific gain proposals from this table.
