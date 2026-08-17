# Gimbalrotor PID sensitivity: within-bag vs between-bag

This comparison separates local sensitivity around each identified plant
from variation of the point estimates across bags. It is descriptive;
with only a few bags it is not a random-effects inference.

## Per-bag local sensitivity

| bag | group | center scale | within local 1-sigma | relative | eigen min | eigen max |
|---|---|---:|---:|---:|---:|---:|
| failure1 | xy | 1.1520448 | 0.0021686875 | 0.188% | 1.150362 | 1.154915 |
| failure1 | z | 1.1697648 | 0.0020643579 | 0.176% | 1.1679529 | 1.1715796 |
| failure1 | roll_pitch | 3.5287743 | 0.080799539 | 2.290% | 3.4681521 | 3.7996002 |
| failure1 | yaw | 3.3788679 | 0.24205699 | 7.164% | 3.2123158 | 3.6494622 |
| failure2 | xy | 1.1787878 | 0.001374904 | 0.117% | 1.1776889 | 1.1800089 |
| failure2 | z | 1.1789924 | 0.0014436748 | 0.122% | 1.1779002 | 1.1800855 |
| failure2 | roll_pitch | 1.7244864 | 0.31216716 | 18.102% | 1.6685932 | 2.8067449 |
| failure2 | yaw | 1.9977971 | 0.20962051 | 10.493% | 0.15142721 | 2.1985612 |
| success | xy | 1.2145134 | 0.00089748692 | 0.074% | 1.2139217 | 1.2150535 |
| success | z | 1.2249249 | 0.0005142981 | 0.042% | 1.2244594 | 1.2253465 |
| success | roll_pitch | 4.1559028 | 0.059425232 | 1.430% | 4.1090928 | 4.2004827 |
| success | yaw | 2.4909143 | 0.065129247 | 2.615% | 2.430711 | 2.5501634 |

## Within-vs-between scale

| group | center min | center max | between-bag std | within-bag sigma RMS | between/within |
|---|---:|---:|---:|---:|---:|
| xy | 1.1520448 | 1.2145134 | 0.025590436 | 0.0015704617 | 16.2948 |
| z | 1.1697648 | 1.2249249 | 0.024123723 | 0.0014843936 | 16.2516 |
| roll_pitch | 1.7244864 | 4.1559028 | 1.0306703 | 0.18930422 | 5.44452 |
| yaw | 1.9977971 | 3.3788679 | 0.57144871 | 0.1886568 | 3.02904 |

Interpretation guide:

- `within-bag sigma RMS >> between-bag std`: the gain correction is
  locally weakly determined by each fitted plant.
- `between-bag std >> within-bag sigma RMS`: each fit may be locally
  sharp, while different bags identify different effective plants.
- Comparable values indicate that both effects matter.

Do not average the bag-specific gain proposals from this table.
