# 真値介入試行の実行時監査

現在の保存資料は完了13試行・途中終了3試行。未完了の動作を180秒試行へ付け替えない。既存compact監査・終端manifest・raw回収記録を読むだけで、推定や制御を再実行しない。

| 試行 | 実時間 s | 状態 | 終端 mm | 軌跡 m | 終端身体観測age s | 勾配age中央値 s | K source age中央値 s |
|---|---:|---|---:|---:|---:|---:|---:|
| imu_marker_all_oracle | 98.925 | 途中終了 | 1101.63 | 41.223 | — | — | — |
| imu_marker_baseline | 180.000 | 完了 | 99.93 | 1.407 | 70.49 | 0.263 | 79.525 |
| imu_marker_known_landmark | 180.000 | 完了 | 81.63 | 1.369 | 59.00 | 0.266 | 78.807 |
| imu_marker_oracle_body | 180.000 | 完了 | 363.97 | 0.007 | 70.23 | 0.277 | 71.669 |
| imu_marker_oracle_body_goal | 68.829 | 途中終了 | 243.01 | 3.291 | — | — | — |
| imu_marker_oracle_goal | 180.000 | 完了 | 363.61 | 0.055 | 64.32 | 0.268 | 81.666 |
| imu_marker_oracle_marker | 0.911 | 途中終了 | 671.50 | 0.431 | — | — | — |
| imu_marker_true_K | 180.000 | 完了 | 128.44 | 1.305 | 70.25 | 0.245 | 69.382 |
| marker_all_oracle | 180.000 | 完了 | 2.74 | 2.880 | 63.00 | 0.268 | 31.616 |
| marker_baseline | 180.000 | 完了 | 178.10 | 1.479 | 83.41 | 0.257 | 39.046 |
| marker_known_landmark | 180.000 | 完了 | 147.11 | 0.868 | 87.33 | 0.251 | 37.511 |
| marker_oracle_body | 180.000 | 完了 | 363.47 | 0.011 | 51.08 | 0.259 | 28.388 |
| marker_oracle_body_goal | 180.000 | 完了 | 2.86 | 2.320 | 80.66 | 0.257 | 38.660 |
| marker_oracle_goal | 180.000 | 完了 | 433.48 | 0.150 | 63.83 | 0.262 | 32.106 |
| marker_oracle_marker | 180.000 | 完了 | 727.54 | 3.759 | 68.33 | 0.229 | 35.850 |
| marker_true_K | 180.000 | 完了 | 147.12 | 1.004 | 75.83 | 0.239 | 40.832 |

K source ageは取得情報の出自であり、true_Kの真値が古くなったという意味ではない。oracle_body・oracle_body_goal・all_oracleでは制御身体の中心を各captureで真値へ置換する。その再帰推定のageを、そのまま実行身体中心の古さと呼ばない。

| 試行 | 物理終端raw保存/取得 | DB処理済み/取得 | K完了/提出 | 保存VI非定常/全枝 | raw欠落/保存preview |
|---|---:|---:|---:|---:|---:|
| imu_marker_all_oracle | — | — | — | 70/110 | 261/877 |
| imu_marker_baseline | 10120/12861 | 8278/12861 | 74/155 | 95/154 | 0/1621 |
| imu_marker_known_landmark | 9329/12764 | 8625/12764 | 69/113 | 88/144 | 0/1545 |
| imu_marker_oracle_body | 10119/12850 | 7867/12850 | 81/113 | 104/170 | 0/1666 |
| imu_marker_oracle_body_goal | — | — | — | 57/88 | 63/607 |
| imu_marker_oracle_goal | 10054/12763 | 8156/12763 | 67/163 | 84/138 | 0/1627 |
| imu_marker_oracle_marker | — | — | — | 4/4 | 1/11 |
| imu_marker_true_K | 11155/13957 | 8516/13957 | 87/113 | 109/178 | 0/1780 |
| marker_all_oracle | 10738/13785 | 8986/13785 | 18/18 | 34/34 | 93/1773 |
| marker_baseline | 9679/12985 | 7038/12985 | 36/36 | 72/72 | 37/1533 |
| marker_known_landmark | 9450/13063 | 6752/13063 | 24/24 | 48/48 | 8/1495 |
| marker_oracle_body | 9942/12977 | 9352/12977 | 30/30 | 58/58 | 0/1657 |
| marker_oracle_body_goal | 10010/12961 | 7230/12961 | 16/17 | 34/34 | 80/1632 |
| marker_oracle_goal | 9670/12956 | 8385/12956 | 42/42 | 82/82 | 0/1615 |
| marker_oracle_marker | 9865/13269 | 8179/13269 | 18/18 | 34/34 | 355/1938 |
| marker_true_K | 10313/14154 | 8208/14154 | 22/22 | 46/46 | 10/1989 |

DB処理済みには無効センサや時間被覆選択で融合しなかった行も含む。VIは独立K workerが保存したprefixであり、全身体更新数とは限らない。raw previewの保存窓は試行間で異なるため、180秒全体の欠落率ではない。IMUの2次元重力方向残差とマーカーの位置残差を混ぜていない。

raw回収は動作終了後であり、回収結果を終了済みの制御へ使っていない。回収件数はdrain後の不足数なので、物理終端の取得−保存の差とも一致しない。

imu_marker_all_oracle：最初の到達は37.413863秒、そこまでの軌跡は4.194149 m。その後の観測済み61.512秒の移動量は37.029271 m。最終frameは到達条件外。
imu_marker_all_oracleの途中終了理由：`Running-gradient explicit fail-safe: joint_position_out_of_range`。終端manifestがないため、終端age・Hz・処理数は不明としている。
imu_marker_oracle_body_goal：最初の到達は38.662417秒、そこまでの軌跡は2.260374 m。その後の観測済み30.167秒の移動量は1.030366 m。最終frameは到達条件外。
imu_marker_oracle_body_goalの途中終了理由：`Running-gradient explicit fail-safe: joint_position_out_of_range`。終端manifestがないため、終端age・Hz・処理数は不明としている。
imu_marker_oracle_markerの途中終了理由：`Running-gradient explicit fail-safe: joint_position_out_of_range`。終端manifestがないため、終端age・Hz・処理数は不明としている。
marker_all_oracle：最初の到達は38.015127秒、そこまでの軌跡は2.869145 m。その後の観測済み141.985秒の移動量は0.010687 m。最終frameも到達条件内。
marker_oracle_body_goal：最初の到達は45.291068秒、そこまでの軌跡は2.305360 m。その後の観測済み134.709秒の移動量は0.014990 m。最終frameも到達条件内。

各一回の非同期試行では軌跡・検出継続・処理負荷・観測frontierも変わる。身体と目標の共通位置ずれを片側だけ訂正する介入もあるため、終端距離だけで不定性を順位付けしない。未完了baselineは比較に含めない。

詳細は `CROSS_CASE_RUNTIME_SUMMARY.json`。
