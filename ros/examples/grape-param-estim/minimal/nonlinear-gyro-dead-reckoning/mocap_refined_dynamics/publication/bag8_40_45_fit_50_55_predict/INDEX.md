# mocapでrefineした非線形回転モデル

公称値を初期値にした最小二乗。調整は40〜45秒、予測評価は50〜55秒。
元の推定・旧図を変更せず、非線形Euler項と指令トルクを含めて比較した。

[周波数一覧図（sigma=2.5 deg/s）](rate_overview_sigma_2p5.pdf) / [動力学の当てはまり](dynamics_fit.pdf) / [パラメータ表](fit_report.pdf) / [全条件CSV](comparison_table.csv) / [結果と制約](RESULTS.md) / [方法](../../README.md)

| sigma [deg/s] | 0.5 Hz | 2 Hz | 5 Hz |
|---:|---|---|---|
| 0.1 | [PDF](rate_0p5_hz/sigma_0p1_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_0p1_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_0p1_deg_s/comparison.pdf) |
| 0.5 | [PDF](rate_0p5_hz/sigma_0p5_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_0p5_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_0p5_deg_s/comparison.pdf) |
| 1 | [PDF](rate_0p5_hz/sigma_1_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_1_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_1_deg_s/comparison.pdf) |
| 2 | [PDF](rate_0p5_hz/sigma_2_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_2_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_2_deg_s/comparison.pdf) |
| 2.5 | [PDF](rate_0p5_hz/sigma_2p5_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_2p5_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_2p5_deg_s/comparison.pdf) |
| 5 | [PDF](rate_0p5_hz/sigma_5_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_5_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_5_deg_s/comparison.pdf) |

sigma=2.5 deg/sでのSO(3) RMSE [deg]。全条件で同じ501時刻を評価。

| Hz | 取得数 | 公称初期モデル | refine後 |
|---:|---:|---:|---:|
| 0.5 | 3 | 103.534 | 57.992 |
| 2.0 | 10 | 51.467 | 14.814 |
| 5.0 | 25 | 22.349 | 6.787 |

帯は固定したモデル・固定した過去余剰平均を条件とする角速度不確かさだけ。
全回転モーメントを保存し、RPY中心・帯だけにdelta-method表示近似を使う。
評価中のmocap・高密度gyroから余剰トルクを算出して入力へ戻していない。
