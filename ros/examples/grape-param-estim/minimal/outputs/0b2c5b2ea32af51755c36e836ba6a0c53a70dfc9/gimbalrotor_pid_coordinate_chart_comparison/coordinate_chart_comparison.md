# Gimbalrotor PID coordinate-chart comparison

The same fitted plants and local estimator covariance are propagated
through two coordinate charts:

- `estimator_quotient`: the existing common-scale quotient chart;
- `centered_scale_free_spd`: an estimate-centered chart on
  `SPD(3) x R^3 x R_+^4` using the scale-free second moment, CoG,
  and log force-over-mass ratios.

The infinitesimal one-sigma result is a coordinate-invariance sanity
check. The finite 0.5/1/2-sigma secants are the chart-curvature test.
No threshold automatically declares either chart superior.

## Covariance: `conservative_fusion`

### Infinitesimal coordinate-invariance check

| bag | group | center | estimator quotient sigma | centered SPD sigma | relative difference |
|---|---|---:|---:|---:|---:|
| failure1 | xy | 1.15204 | 0.00218626 | 0.00218626 | 0.000% |
| failure1 | z | 1.16976 | 0.00206444 | 0.00206444 | 0.000% |
| failure1 | roll_pitch | 3.52877 | 0.0767794 | 0.0767794 | 0.000% |
| failure1 | yaw | 3.37887 | 0.198081 | 0.198081 | 0.000% |
| failure2 | xy | 1.17879 | 0.00134867 | 0.00134867 | 0.000% |
| failure2 | z | 1.17899 | 0.00144983 | 0.00144983 | 0.000% |
| failure2 | roll_pitch | 1.72449 | 0.0597305 | 0.0597304 | 0.000% |
| failure2 | yaw | 1.9978 | 0.161112 | 0.161112 | 0.000% |
| success | xy | 1.21451 | 0.000897492 | 0.000897492 | 0.000% |
| success | z | 1.22492 | 0.000514312 | 0.000514312 | 0.000% |
| success | roll_pitch | 4.1559 | 0.059447 | 0.059447 | 0.000% |
| success | yaw | 2.49091 | 0.0651677 | 0.0651677 | 0.000% |

### Finite-sigma nonlinearity

| bag | group | sigma | estimator finite/local | centered SPD finite/local | estimator envelope | centered SPD envelope |
|---|---|---:|---:|---:|---|---|
| failure1 | xy | 0.5 | 0.997038 | 0.958566 | [1.1512, 1.15297] | [1.15127, 1.16388] |
| failure1 | xy | 1.0 | 0.991965 | incomplete | [1.15036, 1.15491] | [1.1505, 1.15797] |
| failure1 | xy | 2.0 | 1.00275 | 0.99285 | [1.14868, 1.16048] | [1.14895, 1.16873] |
| failure1 | z | 0.5 | 0.999983 | 0.999306 | [1.16886, 1.17067] | [1.16891, 1.17062] |
| failure1 | z | 1.0 | 0.999961 | incomplete | [1.16795, 1.17158] | [1.16805, 1.17148] |
| failure1 | z | 2.0 | 1.00018 | 1.0006 | [1.16614, 1.1734] | [1.16634, 1.1732] |
| failure1 | roll_pitch | 0.5 | 1.01176 | 1.32542 | [3.49922, 3.60026] | [3.49765, 3.85273] |
| failure1 | roll_pitch | 1.0 | 1.05236 | incomplete | [3.46815, 3.7996] | [3.41653, 3.58666] |
| failure1 | roll_pitch | 2.0 | 1.05871 | 5.63164 | [3.40158, 4.45789] | [3.12334, 7.3822] |
| failure1 | yaw | 0.5 | 1.0523 | 0.613864 | [3.30318, 3.48379] | [1.53024, 5.08586] |
| failure1 | yaw | 1.0 | 1.22201 | incomplete | [3.21232, 3.64946] | [3.28373, 7.74855] |
| failure1 | yaw | 2.0 | 1.96909 | 0.874416 | [2.76763, 4.27157] | [0.760564, 3.56971] |
| failure2 | xy | 0.5 | 1.01707 | incomplete | [1.17824, 1.17994] | [1.17831, 1.17927] |
| failure2 | xy | 1.0 | 1.01945 | incomplete | [1.17769, 1.18001] | [1.17782, 1.17975] |
| failure2 | xy | 2.0 | 1.02339 | incomplete | [1.17659, 1.18099] | [1.17685, 1.18071] |
| failure2 | z | 0.5 | 0.994839 | incomplete | [1.17845, 1.17954] | [1.17852, 1.17947] |
| failure2 | z | 1.0 | 0.995756 | incomplete | [1.1779, 1.18009] | [1.17804, 1.17994] |
| failure2 | z | 2.0 | 0.997744 | incomplete | [1.17681, 1.18118] | [1.17709, 1.18089] |
| failure2 | roll_pitch | 0.5 | 8.61932 | incomplete | [1.69638, 3.29152] | [1.69678, 1.75244] |
| failure2 | roll_pitch | 1.0 | 5.22626 | incomplete | [1.66859, 2.80674] | [6.57558e-129, 1.78064] |
| failure2 | roll_pitch | 2.0 | 3.60975 | incomplete | [1.61397, 2.722] | [1.61523, 1.83771] |
| failure2 | yaw | 0.5 | 1.51195 | incomplete | [1.94976, 8.18985] | [1.95006, 2.04584] |
| failure2 | yaw | 1.0 | 1.30108 | incomplete | [0.151427, 2.19856] | [8.50161e-133, 2.094] |
| failure2 | yaw | 2.0 | 1.4603 | incomplete | [0.0736477, 2.62241] | [1.81031, 2.18978] |
| success | xy | 0.5 | 0.999996 | incomplete | [1.21422, 1.21479] | [1.21423, 1.21819] |
| success | xy | 1.0 | 0.999994 | incomplete | [1.21392, 1.21505] | [1.21393, 1.22087] |
| success | xy | 2.0 | 1.0001 | incomplete | [1.21328, 1.21554] | [1.2133, 1.2212] |
| success | z | 0.5 | 0.999993 | incomplete | [1.2247, 1.22514] | [1.2247, 1.22514] |
| success | z | 1.0 | 0.999974 | incomplete | [1.22446, 1.22535] | [1.22446, 1.22534] |
| success | z | 2.0 | 0.999895 | incomplete | [1.22395, 1.22572] | [1.22395, 1.22572] |
| success | roll_pitch | 0.5 | 0.999907 | incomplete | [4.13277, 4.17848] | [3.08479, 4.17442] |
| success | roll_pitch | 1.0 | 0.999634 | incomplete | [4.10909, 4.20048] | [1.01649e-47, 4.19249] |
| success | roll_pitch | 2.0 | 0.998602 | incomplete | [4.06018, 4.24271] | [1.08458e-110, 4.22727] |
| success | yaw | 0.5 | 0.999854 | incomplete | [2.46092, 2.52067] | [1.39278, 2.51716] |
| success | yaw | 1.0 | 0.99941 | incomplete | [2.43071, 2.55016] | [7.06468e-47, 2.54282] |
| success | yaw | 2.0 | 0.997546 | incomplete | [2.36974, 2.62377] | [7.51879e-110, 2.59224] |

### Finite-sample validity

| bag | sigma | coordinate | valid / 27 | invalid |
|---|---:|---|---:|---:|
| failure1 | 0.5 | estimator_quotient | 27 / 27 | 0 |
| failure1 | 0.5 | centered_scale_free_spd | 27 / 27 | 0 |
| failure1 | 1.0 | estimator_quotient | 27 / 27 | 0 |
| failure1 | 1.0 | centered_scale_free_spd | 25 / 27 | 2 |
| failure1 | 2.0 | estimator_quotient | 27 / 27 | 0 |
| failure1 | 2.0 | centered_scale_free_spd | 27 / 27 | 0 |
| failure2 | 0.5 | estimator_quotient | 27 / 27 | 0 |
| failure2 | 0.5 | centered_scale_free_spd | 23 / 27 | 4 |
| failure2 | 1.0 | estimator_quotient | 27 / 27 | 0 |
| failure2 | 1.0 | centered_scale_free_spd | 24 / 27 | 3 |
| failure2 | 2.0 | estimator_quotient | 27 / 27 | 0 |
| failure2 | 2.0 | centered_scale_free_spd | 23 / 27 | 4 |
| success | 0.5 | estimator_quotient | 27 / 27 | 0 |
| success | 0.5 | centered_scale_free_spd | 25 / 27 | 2 |
| success | 1.0 | estimator_quotient | 27 / 27 | 0 |
| success | 1.0 | centered_scale_free_spd | 25 / 27 | 2 |
| success | 2.0 | estimator_quotient | 27 / 27 | 0 |
| success | 2.0 | centered_scale_free_spd | 24 / 27 | 3 |

A large allocation condition number or loss of the source
threshold rank is not an invalidation criterion in these runs.
Only an actually non-finite or mathematically undefined
floating-point calculation is marked invalid.

## Covariance: `overlap_corrected`

### Infinitesimal coordinate-invariance check

| bag | group | center | estimator quotient sigma | centered SPD sigma | relative difference |
|---|---|---:|---:|---:|---:|
| failure1 | xy | 1.15204 | 0.000389541 | 0.000389541 | 0.000% |
| failure1 | z | 1.16976 | 9.47509e-05 | 9.47509e-05 | 0.000% |
| failure1 | roll_pitch | 3.52877 | 0.0346944 | 0.0346944 | 0.000% |
| failure1 | yaw | 3.37887 | 0.0748744 | 0.0748744 | 0.000% |
| failure2 | xy | 1.17879 | 8.4206e-05 | 8.42061e-05 | 0.000% |
| failure2 | z | 1.17899 | 0.000171254 | 0.000171255 | 0.000% |
| failure2 | roll_pitch | 1.72449 | 0.0151224 | 0.0151224 | 0.000% |
| failure2 | yaw | 1.9978 | 0.049891 | 0.049891 | 0.000% |
| success | xy | 1.21451 | 0.000106391 | 0.000106391 | 0.000% |
| success | z | 1.22492 | 4.47777e-05 | 4.47777e-05 | 0.000% |
| success | roll_pitch | 4.1559 | 0.015522 | 0.015522 | 0.000% |
| success | yaw | 2.49091 | 0.00756305 | 0.00756305 | 0.000% |

### Finite-sigma nonlinearity

| bag | group | sigma | estimator finite/local | centered SPD finite/local | estimator envelope | centered SPD envelope |
|---|---|---:|---:|---:|---|---|
| failure1 | xy | 0.5 | 0.996829 | 0.853565 | [1.15193, 1.15229] | [1.15194, 1.16381] |
| failure1 | xy | 1.0 | 0.988435 | 0.829638 | [1.15181, 1.15281] | [1.15183, 1.1638] |
| failure1 | xy | 2.0 | 0.968569 | incomplete | [1.15157, 1.15453] | [1.15161, 1.16378] |
| failure1 | z | 0.5 | 1.00016 | 1.01732 | [1.16973, 1.1698] | [1.16974, 1.16997] |
| failure1 | z | 1.0 | 1.00061 | 1.02947 | [1.1697, 1.16983] | [1.16972, 1.17002] |
| failure1 | z | 2.0 | 1.002 | incomplete | [1.16962, 1.16988] | [1.16965, 1.17013] |
| failure1 | roll_pitch | 0.5 | 0.998083 | 0.990441 | [3.51659, 3.54732] | [3.51606, 3.92218] |
| failure1 | roll_pitch | 1.0 | 0.992999 | 0.963361 | [3.50398, 3.59398] | [3.50292, 3.82874] |
| failure1 | roll_pitch | 2.0 | 0.980656 | incomplete | [3.47746, 3.76433] | [3.43134, 3.83436] |
| failure1 | yaw | 0.5 | 1.01354 | 1.82443 | [3.35317, 3.41232] | [2.27167, 3.8003] |
| failure1 | yaw | 1.0 | 1.05543 | 0.781565 | [3.33096, 3.45719] | [1.54122, 5.0385] |
| failure1 | yaw | 2.0 | 1.24001 | incomplete | [3.27538, 3.59422] | [1.55827, 7.60666] |
| failure2 | xy | 0.5 | 1.21437 | incomplete | [1.17876, 1.17946] | [1.17876, 1.17964] |
| failure2 | xy | 1.0 | 1.67664 | incomplete | [1.17873, 1.17996] | [1.17873, 1.17967] |
| failure2 | xy | 2.0 | 1.70586 | incomplete | [1.17867, 1.18006] | [1.17867, 1.17963] |
| failure2 | z | 0.5 | 0.893004 | incomplete | [1.17895, 1.17924] | [1.17896, 1.17906] |
| failure2 | z | 1.0 | 0.89186 | incomplete | [1.17889, 1.17931] | [1.17892, 1.17909] |
| failure2 | z | 2.0 | 0.910472 | incomplete | [1.17877, 1.17936] | [1.17884, 1.1791] |
| failure2 | roll_pitch | 0.5 | 13.6143 | incomplete | [1.71762, 3.25226] | [1.42338, 1.73135] |
| failure2 | roll_pitch | 1.0 | 18.6287 | incomplete | [1.71077, 3.26159] | [1.42125, 1.73823] |
| failure2 | roll_pitch | 2.0 | 11.5587 | incomplete | [1.69713, 2.83681] | [0.00378335, 1.75204] |
| failure2 | yaw | 0.5 | 1.04905 | incomplete | [1.98185, 3.1411] | [0.682673, 2.01385] |
| failure2 | yaw | 1.0 | 4.12733 | incomplete | [1.96596, 8.23669] | [0.680451, 2.02993] |
| failure2 | yaw | 2.0 | 1.50878 | incomplete | [0.122958, 2.08395] | [0.000171754, 2.0622] |
| success | xy | 0.5 | 0.999998 | incomplete | [1.21447, 1.21456] | [1.21447, 1.22058] |
| success | xy | 1.0 | 0.999992 | incomplete | [1.21442, 1.2146] | [1.21442, 1.22058] |
| success | xy | 2.0 | 0.999967 | incomplete | [1.21433, 1.21469] | [1.21433, 1.21469] |
| success | z | 0.5 | 0.999999 | incomplete | [1.22491, 1.22494] | [1.22492, 1.22519] |
| success | z | 1.0 | 0.999996 | incomplete | [1.22489, 1.22496] | [1.22491, 1.22517] |
| success | z | 2.0 | 0.999986 | incomplete | [1.22485, 1.225] | [1.22489, 1.22496] |
| success | roll_pitch | 0.5 | 0.999999 | incomplete | [4.14902, 4.16275] | [0.236094, 4.16232] |
| success | roll_pitch | 1.0 | 0.999997 | incomplete | [4.14212, 4.16957] | [0.235716, 4.1687] |
| success | roll_pitch | 2.0 | 0.999989 | incomplete | [4.12822, 4.18312] | [4.12992, 4.18136] |
| success | yaw | 0.5 | 1 | incomplete | [2.4879, 2.49392] | [-0.00491622, 2.49365] |
| success | yaw | 1.0 | 1 | incomplete | [2.48489, 2.49691] | [-0.00504403, 2.49638] |
| success | yaw | 2.0 | 0.999999 | incomplete | [2.47883, 2.50287] | [2.47996, 2.50184] |

### Finite-sample validity

| bag | sigma | coordinate | valid / 27 | invalid |
|---|---:|---|---:|---:|
| failure1 | 0.5 | estimator_quotient | 27 / 27 | 0 |
| failure1 | 0.5 | centered_scale_free_spd | 27 / 27 | 0 |
| failure1 | 1.0 | estimator_quotient | 27 / 27 | 0 |
| failure1 | 1.0 | centered_scale_free_spd | 27 / 27 | 0 |
| failure1 | 2.0 | estimator_quotient | 27 / 27 | 0 |
| failure1 | 2.0 | centered_scale_free_spd | 26 / 27 | 1 |
| failure2 | 0.5 | estimator_quotient | 27 / 27 | 0 |
| failure2 | 0.5 | centered_scale_free_spd | 25 / 27 | 2 |
| failure2 | 1.0 | estimator_quotient | 27 / 27 | 0 |
| failure2 | 1.0 | centered_scale_free_spd | 24 / 27 | 3 |
| failure2 | 2.0 | estimator_quotient | 27 / 27 | 0 |
| failure2 | 2.0 | centered_scale_free_spd | 24 / 27 | 3 |
| success | 0.5 | estimator_quotient | 27 / 27 | 0 |
| success | 0.5 | centered_scale_free_spd | 25 / 27 | 2 |
| success | 1.0 | estimator_quotient | 27 / 27 | 0 |
| success | 1.0 | centered_scale_free_spd | 25 / 27 | 2 |
| success | 2.0 | estimator_quotient | 27 / 27 | 0 |
| success | 2.0 | centered_scale_free_spd | 23 / 27 | 4 |

A large allocation condition number or loss of the source
threshold rank is not an invalidation criterion in these runs.
Only an actually non-finite or mathematically undefined
floating-point calculation is marked invalid.

