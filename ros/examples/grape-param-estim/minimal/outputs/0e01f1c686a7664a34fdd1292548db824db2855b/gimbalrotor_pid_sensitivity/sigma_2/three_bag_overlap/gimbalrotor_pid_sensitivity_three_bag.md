# Gimbalrotor PID sensitivity: within-bag vs between-bag

This comparison separates local sensitivity around each identified plant
from variation of the point estimates across bags. It is descriptive;
with only a few bags it is not a random-effects inference.

## Per-bag local sensitivity

| bag | group | center scale | within local 1-sigma | relative | eigen min | eigen max |
|---|---|---:|---:|---:|---:|---:|
| failure1 | xy | 1.1520448 | 0.0003772969 | 0.033% | 1.1515747 | 1.154526 |
| failure1 | z | 1.1697648 | 9.4940763e-05 | 0.008% | 1.1696206 | 1.1698806 |
| failure1 | roll_pitch | 3.5287743 | 0.034023237 | 0.964% | 3.4774616 | 3.76433 |
| failure1 | yaw | 3.3788679 | 0.092844752 | 2.748% | 3.2753771 | 3.594216 |
| failure2 | xy | 1.1787878 | 0.00014364362 | 0.012% | 1.178668 | 1.1800572 |
| failure2 | z | 1.1789924 | 0.00015592236 | 0.013% | 1.1787732 | 1.1793553 |
| failure2 | roll_pitch | 1.7244864 | 0.17479472 | 10.136% | 1.6971316 | 2.8368088 |
| failure2 | yaw | 1.9977971 | 0.075274336 | 3.768% | 0.12295845 | 2.0839516 |
| success | xy | 1.2145134 | 0.00010638753 | 0.009% | 1.2143269 | 1.2146939 |
| success | z | 1.2249249 | 4.4777104e-05 | 0.004% | 1.2248477 | 1.2249999 |
| success | roll_pitch | 4.1559028 | 0.015521834 | 0.373% | 4.1282171 | 4.1831218 |
| success | yaw | 2.4909143 | 0.0075630458 | 0.304% | 2.4788283 | 2.5028713 |

## Within-vs-between scale

| group | center min | center max | between-bag std | within-bag sigma RMS | between/within |
|---|---:|---:|---:|---:|---:|
| xy | 1.1520448 | 1.2145134 | 0.025590436 | 0.0002410427 | 106.166 |
| z | 1.1697648 | 1.2249249 | 0.024123723 | 0.00010852115 | 222.295 |
| roll_pitch | 1.7244864 | 4.1559028 | 1.0306703 | 0.10320158 | 9.98696 |
| yaw | 1.9977971 | 3.3788679 | 0.57144871 | 0.069146158 | 8.26436 |

Interpretation guide:

- `within-bag sigma RMS >> between-bag std`: the gain correction is
  locally weakly determined by each fitted plant.
- `between-bag std >> within-bag sigma RMS`: each fit may be locally
  sharp, while different bags identify different effective plants.
- Comparable values indicate that both effects matter.

Do not average the bag-specific gain proposals from this table.
