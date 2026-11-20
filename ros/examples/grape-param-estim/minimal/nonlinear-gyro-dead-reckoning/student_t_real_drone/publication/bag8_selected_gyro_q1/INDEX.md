# 原gyroを間引いたStudent-t＋Qσ=1 deg/s比較

以前と同じ原gyroの0.5／2／5／10 Hz取得で、非線形Euler／ジャイロ項なしを比較した。同じbag #8の50.000988619〜55.000988619秒、同じ初期姿勢・J・実指令／gimbal由来トルク・過去余剰トルク平均を使用。ν=4、観測Rの周辺σ=2.5／5 deg/sを維持した。

角速度プロセス増分は、取得した観測間隔ごとに $Q_\omega=(\pi/180)^2I_3$ [rad²/s²] を一度加える。
間引いて使わない原gyro時刻にはQも観測因子も追加しない。初回と最後の観測後の尾部にはQを追加しない。取得頻度に応じたdt倍率は掛けていない。したがって一秒あたりの累積過程分散は取得頻度によって異なり、同一の連続時間拡散モデルで観測数だけを変えた比較とは区別する。

## 同じ501時刻での姿勢RMSE [deg]

| 観測σ [deg/s] | 取得 | 観測数 | Q1：非線形 | Q1：項なし | Q0：非線形 | Q0：項なし |
|---:|---|---:|---:|---:|---:|---:|
| 2.5 | 0.5 Hz | 3 | 98.706115 | 95.813268 | 99.021416 | 95.851993 |
| 2.5 | 2 Hz | 10 | 91.769938 | 90.015818 | 94.552952 | 92.359745 |
| 2.5 | 5 Hz | 25 | 43.497027 | 40.117372 | 93.598478 | 91.572359 |
| 2.5 | 10 Hz | 50 | 8.675655 | 9.278557 | 82.853780 | 83.999094 |
| 2.5 | 全原レート（約200 Hz） | 1001 | 0.569133 | 0.567951 | 38.514552 | 37.033467 |
| 5 | 0.5 Hz | 3 | 99.271982 | 96.851376 | 98.958126 | 95.465892 |
| 5 | 2 Hz | 10 | 74.672727 | 73.672378 | 77.059886 | 76.354228 |
| 5 | 5 Hz | 25 | 32.745641 | 26.864949 | 54.583684 | 55.612521 |
| 5 | 10 Hz | 50 | 6.548147 | 6.769319 | 27.740895 | 27.729353 |
| 5 | 全原レート（約200 Hz） | 1001 | 0.574762 | 0.573818 | 34.064375 | 32.026786 |

**間引き条件では7→11点でも大きな求積差が残る箇所がある。以下の結果は有限求積の実行結果として保存し、収束済みの復元精度や非線形モデルの優位とは主張しない。** 各条件の差は末尾の表とquadrature_refinement.jsonへ記録する。

Q1はσ=1 deg/sを意味し、単位行列Qそのものではない。Q0と全原レートの旧保存値は変更せず併記した。Q0には以前記録した未収束箇所があるため、収束済みの真値とは扱わない。途中mocapは同じ501時刻での評価・表示専用。角速度の誤差は雑音を含む生gyroとの差である。

## 図と検証

- 0.5 Hz、観測σ=2.5 deg/s：[姿勢・gyro・±2σ](rate_0p5_hz/sigma_2p5_deg_s/attitude_and_gyro.pdf) ／ [誤差](rate_0p5_hz/sigma_2p5_deg_s/reconstruction_errors.pdf)
- 2 Hz、観測σ=2.5 deg/s：[姿勢・gyro・±2σ](rate_2_hz/sigma_2p5_deg_s/attitude_and_gyro.pdf) ／ [誤差](rate_2_hz/sigma_2p5_deg_s/reconstruction_errors.pdf)
- 5 Hz、観測σ=2.5 deg/s：[姿勢・gyro・±2σ](rate_5_hz/sigma_2p5_deg_s/attitude_and_gyro.pdf) ／ [誤差](rate_5_hz/sigma_2p5_deg_s/reconstruction_errors.pdf)
- 10 Hz、観測σ=2.5 deg/s：[姿勢・gyro・±2σ](rate_10_hz/sigma_2p5_deg_s/attitude_and_gyro.pdf) ／ [誤差](rate_10_hz/sigma_2p5_deg_s/reconstruction_errors.pdf)
- 0.5 Hz、観測σ=5 deg/s：[姿勢・gyro・±2σ](rate_0p5_hz/sigma_5_deg_s/attitude_and_gyro.pdf) ／ [誤差](rate_0p5_hz/sigma_5_deg_s/reconstruction_errors.pdf)
- 2 Hz、観測σ=5 deg/s：[姿勢・gyro・±2σ](rate_2_hz/sigma_5_deg_s/attitude_and_gyro.pdf) ／ [誤差](rate_2_hz/sigma_5_deg_s/reconstruction_errors.pdf)
- 5 Hz、観測σ=5 deg/s：[姿勢・gyro・±2σ](rate_5_hz/sigma_5_deg_s/attitude_and_gyro.pdf) ／ [誤差](rate_5_hz/sigma_5_deg_s/reconstruction_errors.pdf)
- 10 Hz、観測σ=5 deg/s：[姿勢・gyro・±2σ](rate_10_hz/sigma_5_deg_s/attitude_and_gyro.pdf) ／ [誤差](rate_10_hz/sigma_5_deg_s/reconstruction_errors.pdf)

[全原レートQ1の図と更新式](../bag8_full_gyro_q1/INDEX.md)。今回も現在の共同候補をMarkov予測・Student-t条件付けで持ち越す同じengineを使い、初期角速度候補だけのQ0法則へ共分散を加算する処理や、毎回独立Gaussianへ置換する処理は行わない。

時刻間は連続ターゲットに対する決定論的Nyström求積と、角速度を条件にした四元数二次モーメント・回転の全共分散を保持する有限近似。Gaussian proposalは配置用で密度比を重みに残す。Monte Carlo、共分散floor、外れ値棄却、有限診断による更新停止は使わない。

| 観測σ | Hz | モデル | 求積変更 | 最大ω平均差 [deg/s] | 最大RPY中心差 [deg] |
|---:|---:|---|---|---:|---:|
| 2.5 | 0.5 | nonlinear | 7 -> 11 | 0.566626 | 1.1948 |
| 2.5 | 0.5 | linear_forced | 7 -> 11 | 0.331299 | 1.47446 |
| 2.5 | 2 | nonlinear | 7 -> 11 | 5.27366 | 5.8838 |
| 2.5 | 2 | linear_forced | 7 -> 11 | 1.5778 | 5.34858 |
| 2.5 | 5 | nonlinear | 7 -> 11 | 14.2178 | 30.4891 |
| 2.5 | 5 | linear_forced | 7 -> 11 | 4.54389 | 16.075 |
| 2.5 | 10 | nonlinear | 7 -> 11 | 2.99999 | 2.43315 |
| 2.5 | 10 | linear_forced | 7 -> 11 | 3.04664 | 1.8195 |
| 5 | 0.5 | nonlinear | 7 -> 11 | 4.69368 | 17.8008 |
| 5 | 0.5 | linear_forced | 7 -> 11 | 2.17456 | 23.1099 |
| 5 | 2 | nonlinear | 7 -> 11 | 5.10596 | 14.2848 |
| 5 | 2 | linear_forced | 7 -> 11 | 1.51156 | 18.8961 |
| 5 | 5 | nonlinear | 7 -> 11 | 6.49112 | 17.8689 |
| 5 | 5 | linear_forced | 7 -> 11 | 9.12301 | 25.7075 |
| 5 | 10 | nonlinear | 7 -> 11 | 1.83726 | 2.86759 |
| 5 | 10 | linear_forced | 7 -> 11 | 1.79747 | 3.09081 |

図の帯は同じ保存共同モーメントの±2σで、RPYは局所表示近似。推定値の補間・平滑化や表示用の分散変更はしない。有限求積差は診断であり精度保証ではない。原値・header/record ns・元indexを保持し、未来の観測・未取得gyroを予測へ入れない。

[条件・数値](summary.json) ／ [独立検証](verification.json) ／ [全原レートと旧結果の保存hash](preserved_sources.json)
