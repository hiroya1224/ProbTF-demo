# 実ドローン：原gyro全件によるStudent-t比較

同じbag #8の50.000988619〜55.000988619秒に含まれる原gyroを、間引きも補間もせず全件使用する。最初の一件を初期角速度候補へ一度だけ使い、その後の各観測を取得順にStudent-t尤度として一度ずつ融合する。

慣性・実指令／gimbal由来トルク・40〜45秒の余剰トルク平均は前の実測比較と同じ凍結値。Q=0、ν=4、周辺σ=2.5／5 deg/sを維持する。途中mocap、noise追加、biasや時計の補正、再同定は使わない。合成結果と0.5／2／5／10 Hz結果は変更しない。

| 周辺σ [deg/s] | 原gyro数 | 項あり姿勢RMSE [deg] | 項なし姿勢RMSE [deg] | 図 |
|---:|---:|---:|---:|---|
| 2.5 | 1001 | 38.514552 | 37.033467 | [姿勢・gyro](rate_full_native/sigma_2p5_deg_s/attitude_and_gyro.pdf) ／ [誤差](rate_full_native/sigma_2p5_deg_s/reconstruction_errors.pdf) |
| 5 | 1001 | 34.064375 | 32.026786 | [姿勢・gyro](rate_full_native/sigma_5_deg_s/attitude_and_gyro.pdf) ／ [誤差](rate_full_native/sigma_5_deg_s/reconstruction_errors.pdf) |

観測直前／直後と、同じ501時刻の予測を保存する。最後の取得後も固定終端まで同じ事後候補を伝播する。角速度の全共分散と姿勢との交差を保持し、帯は同じ保存量の±2σ。RPYは回転モーメントからの局所表示近似。生gyroは雑音を含む測定参照であり真値ではない。

[結果の解釈](RESULTS.md) ／ [全結果CSV](comparison_table.csv) ／ [取得時計・条件・hash](summary.json) ／ [独立検証](verification.json) ／ [旧結果の保存確認](preserved_sources.json)
