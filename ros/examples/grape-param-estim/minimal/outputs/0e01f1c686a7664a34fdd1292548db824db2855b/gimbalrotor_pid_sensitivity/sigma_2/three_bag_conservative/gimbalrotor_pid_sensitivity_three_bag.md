# Gimbalrotor PID sensitivity: within-bag vs between-bag

This comparison separates local sensitivity around each identified plant
from variation of the point estimates across bags. It is descriptive;
with only a few bags it is not a random-effects inference.

## Per-bag local sensitivity

| bag | group | center scale | within local 1-sigma | relative | eigen min | eigen max |
|---|---|---:|---:|---:|---:|---:|
| failure1 | xy | 1.1520448 | 0.0021922597 | 0.190% | 1.1486812 | 1.1604772 |
| failure1 | z | 1.1697648 | 0.002064806 | 0.177% | 1.1661437 | 1.1733971 |
| failure1 | roll_pitch | 3.5287743 | 0.081287027 | 2.304% | 3.4015842 | 4.4578924 |
| failure1 | yaw | 3.3788679 | 0.39003972 | 11.544% | 2.7676319 | 4.2715695 |
| failure2 | xy | 1.1787878 | 0.0013557747 | 0.115% | 1.1765904 | 1.1809868 |
| failure2 | z | 1.1789924 | 0.0014334394 | 0.122% | 1.1768091 | 1.1811795 |
| failure2 | roll_pitch | 1.7244864 | 0.058066733 | 3.367% | 1.6139664 | 1.8399415 |
| failure2 | yaw | 1.9977971 | 0.21978972 | 11.002% | 1.8127675 | 2.6224101 |
| success | xy | 1.2145134 | 0.00089758283 | 0.074% | 1.2132787 | 1.2155419 |
| success | z | 1.2249249 | 0.00051425782 | 0.042% | 1.2239501 | 1.2257241 |
| success | roll_pitch | 4.1559028 | 0.0593639 | 1.428% | 4.0601768 | 4.2427137 |
| success | yaw | 2.4909143 | 0.065007757 | 2.610% | 2.3697361 | 2.6237748 |

## Within-vs-between scale

| group | center min | center max | between-bag std | within-bag sigma RMS | between/within |
|---|---:|---:|---:|---:|---:|
| xy | 1.1520448 | 1.2145134 | 0.025590436 | 0.0015758366 | 16.2393 |
| z | 1.1697648 | 1.2249249 | 0.024123723 | 0.001481287 | 16.2857 |
| roll_pitch | 1.7244864 | 4.1559028 | 1.0306703 | 0.067090483 | 15.3624 |
| yaw | 1.9977971 | 3.3788679 | 0.57144871 | 0.26119247 | 2.18785 |

Interpretation guide:

- `within-bag sigma RMS >> between-bag std`: the gain correction is
  locally weakly determined by each fitted plant.
- `between-bag std >> within-bag sigma RMS`: each fit may be locally
  sharp, while different bags identify different effective plants.
- Comparable values indicate that both effects matter.

Do not average the bag-specific gain proposals from this table.
