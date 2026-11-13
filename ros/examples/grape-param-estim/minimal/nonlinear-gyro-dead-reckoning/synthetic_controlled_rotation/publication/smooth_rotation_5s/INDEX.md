# 合成gyro：線形・非線形回転動力学の比較

[sigma=2.5 deg/sの一覧図](rate_overview_sigma_2p5.pdf) / [生成した運動と入力](generated_motion.pdf) / [CSV](comparison_table.csv) / [方法](../../README.md)

共通の既知J・既知入力トルク・同じgyroを使用。線形側はEuler式のジャイロ項だけを省く。
生成ノイズと仮定する候補分散を分離し、真値を予測へ戻さない。帯は同じ回転モーメントからの局所±2sigma表示。

| 仮定するsigma [deg/s] | 0.5 Hz | 2 Hz | 5 Hz |
|---:|---|---|---|
| 0.1 | [PDF](rate_0p5_hz/sigma_0p1_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_0p1_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_0p1_deg_s/comparison.pdf) |
| 0.5 | [PDF](rate_0p5_hz/sigma_0p5_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_0p5_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_0p5_deg_s/comparison.pdf) |
| 1 | [PDF](rate_0p5_hz/sigma_1_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_1_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_1_deg_s/comparison.pdf) |
| 2 | [PDF](rate_0p5_hz/sigma_2_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_2_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_2_deg_s/comparison.pdf) |
| 2.5 | [PDF](rate_0p5_hz/sigma_2p5_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_2p5_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_2p5_deg_s/comparison.pdf) |
| 5 | [PDF](rate_0p5_hz/sigma_5_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_5_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_5_deg_s/comparison.pdf) |

sigma=2.5 deg/s、501共通時刻での姿勢中心のSO(3) RMSE [deg]：

| gyro Hz | 取得数 | 非線形Euler | ジャイロ項なし |
|---:|---:|---:|---:|
| 0.5 | 3 | 0.6715 | 102.8546 |
| 2 | 10 | 0.4224 | 30.3497 |
| 5 | 25 | 0.2459 | 12.5940 |

ノイズなし・biasなしの理想再生と数値整合は[audit.json](audit.json)に別保存。主結果の測定列は変更しない。

## 角速度の途中経過

[平均・±2σと幅の一覧PDF](gyro_progress_sigma_2p5.pdf) / [幅の一覧PNG](gyro_width_sigma_2p5.png)

観測の直前・直後を同じ時刻に両方表示。現在は新しい独立Gaussianへの置換であり、Bayes条件付けではない。非線形では狭まる場合と広がる場合があり、線形側の角速度分散は一定。

| 仮定sigma [deg/s] | 0.5 Hz | 2 Hz | 5 Hz |
|---:|---|---|---|
| 0.1 | [gyro PDF](rate_0p5_hz/sigma_0p1_deg_s/gyro_progress.pdf) | [gyro PDF](rate_2_hz/sigma_0p1_deg_s/gyro_progress.pdf) | [gyro PDF](rate_5_hz/sigma_0p1_deg_s/gyro_progress.pdf) |
| 0.5 | [gyro PDF](rate_0p5_hz/sigma_0p5_deg_s/gyro_progress.pdf) | [gyro PDF](rate_2_hz/sigma_0p5_deg_s/gyro_progress.pdf) | [gyro PDF](rate_5_hz/sigma_0p5_deg_s/gyro_progress.pdf) |
| 1 | [gyro PDF](rate_0p5_hz/sigma_1_deg_s/gyro_progress.pdf) | [gyro PDF](rate_2_hz/sigma_1_deg_s/gyro_progress.pdf) | [gyro PDF](rate_5_hz/sigma_1_deg_s/gyro_progress.pdf) |
| 2 | [gyro PDF](rate_0p5_hz/sigma_2_deg_s/gyro_progress.pdf) | [gyro PDF](rate_2_hz/sigma_2_deg_s/gyro_progress.pdf) | [gyro PDF](rate_5_hz/sigma_2_deg_s/gyro_progress.pdf) |
| 2.5 | [gyro PDF](rate_0p5_hz/sigma_2p5_deg_s/gyro_progress.pdf) | [gyro PDF](rate_2_hz/sigma_2p5_deg_s/gyro_progress.pdf) | [gyro PDF](rate_5_hz/sigma_2p5_deg_s/gyro_progress.pdf) |
| 5 | [gyro PDF](rate_0p5_hz/sigma_5_deg_s/gyro_progress.pdf) | [gyro PDF](rate_2_hz/sigma_5_deg_s/gyro_progress.pdf) | [gyro PDF](rate_5_hz/sigma_5_deg_s/gyro_progress.pdf) |
