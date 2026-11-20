# 実ドローン：Student-t観測による二モデル比較

[主図：5 Hz・σ=2.5 deg/s](rate_5_hz/sigma_2p5_deg_s/attitude_and_gyro.pdf) ／ [同条件の誤差](rate_5_hz/sigma_2p5_deg_s/reconstruction_errors.pdf) ／ [方法](../../README.md)

一日目#8の同じ5秒区間、bag時刻50.000988619〜55.000988619秒。40〜45秒でrefineした慣性・アクチュエータモデルと過去余剰トルク平均を凍結した。新たなnoiseや人工スパイクは加えず、原gyroの取得時刻・値を使う。Student-tはν=4、Q=0。ジャイロ項あり／なしだけを比較する。

初期姿勢は保存済みの一度だけのmocap整合。評価中のmocap・未取得gyroを推定やトルクの更新へ渡さない。モデルの再同定、時刻ずれ・bias補正はしない。

| Hz | 周辺σ [deg/s] | gyro数 | 項あり姿勢RMSE [deg] | 項なし姿勢RMSE [deg] | 図 |
|---:|---:|---:|---:|---:|---|
| 0.5 | 2.5 | 3 | 99.021416 | 95.851993 | [PDF](rate_0p5_hz/sigma_2p5_deg_s/attitude_and_gyro.pdf) |
| 2 | 2.5 | 10 | 94.552952 | 92.359745 | [PDF](rate_2_hz/sigma_2p5_deg_s/attitude_and_gyro.pdf) |
| 5 | 2.5 | 25 | 93.598478 | 91.572359 | [PDF](rate_5_hz/sigma_2p5_deg_s/attitude_and_gyro.pdf) |
| 10 | 2.5 | 50 | 82.853780 | 83.999094 | [PDF](rate_10_hz/sigma_2p5_deg_s/attitude_and_gyro.pdf) |
| 0.5 | 5 | 3 | 98.958126 | 95.465892 | [PDF](rate_0p5_hz/sigma_5_deg_s/attitude_and_gyro.pdf) |
| 2 | 5 | 10 | 77.059886 | 76.354228 | [PDF](rate_2_hz/sigma_5_deg_s/attitude_and_gyro.pdf) |
| 5 | 5 | 25 | 54.583684 | 55.612521 | [PDF](rate_5_hz/sigma_5_deg_s/attitude_and_gyro.pdf) |
| 10 | 5 | 50 | 27.740895 | 27.729353 | [PDF](rate_10_hz/sigma_5_deg_s/attitude_and_gyro.pdf) |

すべて同じ501時刻で評価した。最後の取得後も固定終端まで同じ事後候補を物理伝播し、水平補間で埋めない。四元数・角速度の全共同モーメントと観測直前／直後を保存する。RPY中心と帯は回転モーメントに対する局所表示近似である。

角速度の誤差欄は生gyroとの比較であり、角速度真値との誤差ではない。図のSG角速度は過去に保存したmocap微分の読取専用参照。固定モデルと過去余剰平均に残る実機との不一致をStudent-tが解消すると保証しない。

[全結果CSV](comparison_table.csv) ／ [条件・source hash](summary.json) ／ [独立検証](verification.json) ／ [結果の解釈](RESULTS.md) ／ [合成結果と旧実測結果の保存確認](preserved_sources.json)。
