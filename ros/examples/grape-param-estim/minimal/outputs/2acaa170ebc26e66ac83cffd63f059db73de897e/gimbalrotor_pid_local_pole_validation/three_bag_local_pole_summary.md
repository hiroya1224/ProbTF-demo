# Three-bag local sampled-data pole validation

Each row is an independent bag distribution; bags are never averaged.

| case | outcome | covariance | delay | roll/pitch P/I/D | fitted lag [s] | dt [s] | pole valid | stable fraction | center radius | median radius | radius 16–84% | radius 2.5–97.5% | median unstable poles |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| failure1 | crashed | conservative_fusion | fitted_thrust_delay | 20/1/8 | 0.19719923 | 0.0049953461 | 512/512 | 0.3671875 | 1.0002715 | 1.0001348 | [0.99988308, 1.0002587] | [0.99987944, 1.0002895] | 2 |
| failure1 | crashed | conservative_fusion | zero_thrust_delay | 20/1/8 | 0.19719923 | 0.0049953461 | 512/512 | 1 | 0.99988257 | 0.99988265 | [0.99988039, 0.99988463] | [0.99987835, 0.99988675] | 0 |
| failure1 | crashed | overlap_corrected | fitted_thrust_delay | 20/1/8 | 0.19719923 | 0.0049953461 | 512/512 | 0.033203125 | 1.0002715 | 1.0002446 | [1.0001453, 1.0002689] | [0.99998225, 1.0002778] | 2 |
| failure1 | crashed | overlap_corrected | zero_thrust_delay | 20/1/8 | 0.19719923 | 0.0049953461 | 512/512 | 1 | 0.99988257 | 0.99988262 | [0.99988177, 0.99988351] | [0.99988081, 0.99988421] | 0 |
| failure2 | crashed | conservative_fusion | fitted_thrust_delay | 10/1/8 | 0.2710892 | 0.0050077438 | 512/512 | 0.22265625 | 1.0030123 | 1.0000818 | [0.99983128, 1.0003598] | [0.99983128, 1.0026061] | 4 |
| failure2 | crashed | conservative_fusion | zero_thrust_delay | 10/1/8 | 0.2710892 | 0.0050077438 | 512/512 | 0.31640625 | 0.99983128 | 1.0000292 | [0.99983128, 1.0003015] | [0.99983128, 1.0003669] | 4 |
| failure2 | crashed | overlap_corrected | fitted_thrust_delay | 10/1/8 | 0.2710892 | 0.0050077438 | 512/512 | 0.32226562 | 1.0030123 | 1.0002207 | [0.99983128, 1.0012546] | [0.99983128, 1.0028965] | 2 |
| failure2 | crashed | overlap_corrected | zero_thrust_delay | 10/1/8 | 0.2710892 | 0.0050077438 | 512/512 | 0.5703125 | 0.99983128 | 0.99983128 | [0.99983128, 1.0003122] | [0.99983128, 1.0003655] | 0 |
| success | successful | conservative_fusion | fitted_thrust_delay | 13/1/20 | 0.17559731 | 0.0099983215 | 512/512 | 1 | 0.99974696 | 0.99974696 | [0.99974696, 0.99974696] | [0.99974696, 0.99974696] | 0 |
| success | successful | conservative_fusion | zero_thrust_delay | 13/1/20 | 0.17559731 | 0.0099983215 | 512/512 | 1 | 0.99974696 | 0.99974696 | [0.99974696, 0.99974696] | [0.99974696, 0.99974696] | 0 |
| success | successful | overlap_corrected | fitted_thrust_delay | 13/1/20 | 0.17559731 | 0.0099983215 | 512/512 | 1 | 0.99974696 | 0.99974696 | [0.99974696, 0.99974696] | [0.99974696, 0.99974696] | 0 |
| success | successful | overlap_corrected | zero_thrust_delay | 13/1/20 | 0.17559731 | 0.0099983215 | 512/512 | 1 | 0.99974696 | 0.99974696 | [0.99974696, 0.99974696] | [0.99974696, 0.99974696] | 0 |

## Fitted thrust delay versus zero delay

| case | covariance | stable fraction fitted | stable fraction zero | difference | median radius fitted | median radius zero | difference |
|---|---|---:|---:|---:|---:|---:|---:|
| failure1 | conservative_fusion | 0.3671875 | 1 | -0.6328125 | 1.0001348 | 0.99988265 | 0.00025212007 |
| failure1 | overlap_corrected | 0.033203125 | 1 | -0.96679688 | 1.0002446 | 0.99988262 | 0.00036193724 |
| failure2 | conservative_fusion | 0.22265625 | 0.31640625 | -0.09375 | 1.0000818 | 1.0000292 | 5.2594075e-05 |
| failure2 | overlap_corrected | 0.32226562 | 0.5703125 | -0.24804688 | 1.0002207 | 0.99983128 | 0.00038942111 |
| success | conservative_fusion | 1 | 1 | 0 | 0.99974696 | 0.99974696 | 1.7230661e-12 |
| success | overlap_corrected | 1 | 1 | 0 | 0.99974696 | 0.99974696 | 1.4424018e-12 |
