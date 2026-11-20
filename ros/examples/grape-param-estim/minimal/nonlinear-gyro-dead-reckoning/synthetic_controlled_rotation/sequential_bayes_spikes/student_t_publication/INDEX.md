# Student-t観測：ジャイロ項あり／なし

[姿勢・角速度と±2σ](attitude_and_gyro.pdf) ／ [復元誤差](reconstruction_errors.pdf) ／ [定式化と再現手順](../STUDENT_T.md)

両方の観測尤度を3変量Student-t、自由度ν=4とした。Student-tを使わない方式は今回の図に含めない。元のQ=0・body角速度3倍・スパイク振幅0.5倍・20秒・10 Hzの測定列を保存値のまま使い、途中の真値・スパイクフラグ・生成biasを推定へ渡さない。

観測の周辺共分散は $R=(5\pi/180)^2I_3$ のまま。Student-tの尺度行列は $\Sigma=(\nu-2)R/\nu$ とし、ν=4ではR/2。初期候補は従来と同じ $\mathcal N(W_0;y_0,R)$ で、初回gyroを一度だけ初期化に使う。その後の200観測は、すべてStudent-t尤度で取得順に一度ずつ更新する。

Gamma精度を解析的に周辺化したStudent-t密度を直接評価し、既存の決定論的求積で全候補と全交差を伝播する。PichéらのStudent-t観測モデルを使うが、同論文の変分Gaussian状態近似へ置換する実装ではない。Gaussianは初期候補と数値積分用proposalであり、各観測のStudent-t尤度をGaussianに戻していない。

| Student-t方式 | 姿勢中心RMSE [deg] | 終端姿勢誤差 [deg] | ω RMSE xyz [deg/s] |
|---|---:|---:|---|
| 非線形Euler | 4.970454 | 5.897187 | 0.541280, 1.013903, 0.298351 |
| ジャイロ項なし | 9.460661 | 25.961528 | 7.801798, 6.906859, 0.642886 |

共通する201個の保存済み観測直後で評価。推定列の補間や平滑化はしない。RPYの中心と帯は同じ回転モーメントからの局所表示近似。観測前後401行、四元数二次モーメント、全回転共分散、角速度共分散と交差を保存する。

[全評価値](student_t_metrics.json)、[求積の有限差](quadrature_refinement.json)、[独立検証](verification.json)、[実行条件・入力とsource hash](summary.json)。有限な求積差やproposal未収束を観測棄却・更新停止へ使わない。

参照：Piché, Särkkä, Hartikainen (2012), [Recursive Outlier-Robust Filtering and Smoothing for Nonlinear Systems Using the Multivariate Student-t Distribution](https://users.aalto.fi/~ssarkka/pub/studentFiltering_MLSP.pdf), §2.1–2.3。ν=4は実行前に同論文の推奨値に固定した。
