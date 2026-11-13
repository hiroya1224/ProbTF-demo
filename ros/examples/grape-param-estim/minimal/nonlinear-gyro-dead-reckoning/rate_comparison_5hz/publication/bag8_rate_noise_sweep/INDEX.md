# 取得頻度・角速度ノイズの比較

各周波数・各ノイズは別フォルダ／別PDF。sigmaは各軸の1観測あたりの標準偏差deg/s。
初期姿勢・5秒区間・制御入力・公称モデル・固定した過去余剰トルク平均は共通。

| sigma [deg/s] | 0.5 Hz | 2 Hz | 5 Hz |
|---:|---|---|---|
| 0.1 | [PDF](rate_0p5_hz/sigma_0p1_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_0p1_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_0p1_deg_s/comparison.pdf) |
| 0.5 | [PDF](rate_0p5_hz/sigma_0p5_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_0p5_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_0p5_deg_s/comparison.pdf) |
| 1 | [PDF](rate_0p5_hz/sigma_1_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_1_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_1_deg_s/comparison.pdf) |
| 2 | [PDF](rate_0p5_hz/sigma_2_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_2_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_2_deg_s/comparison.pdf) |
| 2.5 | [PDF](rate_0p5_hz/sigma_2p5_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_2p5_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_2p5_deg_s/comparison.pdf) |
| 5 | [PDF](rate_0p5_hz/sigma_5_deg_s/comparison.pdf) | [PDF](rate_2_hz/sigma_5_deg_s/comparison.pdf) | [PDF](rate_5_hz/sigma_5_deg_s/comparison.pdf) |

## sigma=2.5 deg/sのSO(3) RMSE [deg]

| 取得頻度 | 観測数 | 一定角速度 | 線形強制 | 非線形 |
|---:|---:|---:|---:|---:|
| 0.500 | 3.000 | 12.507 | 108.729 | 108.589 |
| 2.000 | 10.000 | 4.650 | 50.981 | 50.980 |
| 5.000 | 25.000 | 1.768 | 21.707 | 21.660 |

[sigma=2.5の一覧図](rate_overview_sigma_2p5.pdf) / [全条件のCSV](comparison_table.csv)

[SG由来角速度の周波数対照](sg_diagnostic/sg_benchmark.pdf)は、力学モデル・ノイズ伝播を使わない別の運動学診断。
状態の候補法則は同じ6点増分因子の積であり、回転モーメントは全交差を含めて伝播。RPYの中心・帯は同じ回転モーメントからのdelta-method表示近似。
[方法・仮定・検証](../../README.md)
