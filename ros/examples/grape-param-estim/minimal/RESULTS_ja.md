# 最小推定器の比較結果

## 目的

この実験は、記録された制御入力を機体モデルへ入れた open-loop 軌道が rosbag の位置、姿勢、速度、gyro、accelerometer を再現する物理パラメータを求めることを目的とする。

動作している deterministic direct-shooting を基準として固定し、GUI 版の根幹である sparse full-trajectory MAP と diagonal-Q Laplace-EM を GUI、project、worker process から切り離して段階的に比較した。

対象は同梱された `20260612_grape_hovering_4_2026-06-12-17-33-59.bag` の 19--24 秒である。

## 比較した段階

Deterministic baseline は最初の観測状態だけから 5 秒間を一度も reset せず、記録制御入力による単一の open-loop rollout 誤差を SciPy `least_squares` で最小化する。

Fixed-Q sparse MAP は deterministic の 18 次元座標を初期値かつ prior mean とし、lag を 0.01 秒、Q を `[25, 25, 25, 1, 1, 1]` に固定したまま、99 knot の latent trajectory と静的パラメータを同時に最適化する。

Laplace-EM は同じ fixed-lag graph から始め、Laplace covariance correction を含む Q target と log-Q damping の受理判定を 1 iteration だけ実行する。

両 probabilistic 段階とも GUI backend と同じ解析 Jacobian、sparse MAP、Laplace factorization、dynamics residual moment、Q update 実装を直接使用する。

時刻ごとの residual wrench は未知変数として追加していない。

## 実測結果

| method | position RMSE [m] | orientation RMSE [deg] | velocity RMSE [m/s] | terminal position error [m] |
|---|---:|---:|---:|---:|
| nominal rollout | 52.111 | 134.092 | 31.292 | 123.424 |
| deterministic baseline | 1.500 | 34.456 | 1.908 | 0.524 |
| fixed-Q sparse MAP | 49.687 | 125.903 | 30.748 | 119.862 |
| one-step Laplace-EM | 48.394 | 106.639 | 27.417 | 109.749 |

Fixed-Q sparse MAP は 2598 次元、99 knots、5547 factors の問題を 8 nonlinear iterations、約 38 秒で数値収束した。

その MAP objective は 469.579 から 9.611 へ低下した。

しかし、その静的 MAP parameter だけを同じ記録制御入力で open-loop replay すると、deterministic baseline より大幅に悪化した。

Fixed-Q MAP の objective 9.611 のうち、pose position factor は 0.489、dynamics residual factor は 0.917、static prior は 0.337 だった。

Latent trajectory が観測へ沿うことで観測 factor を小さくできる一方、Q が許す dynamics discrepancy の下では静的パラメータ単独の open-loop 再現能力が目的関数に十分反映されていない。

1 iteration の Laplace-EM は約 109 秒を要し、Q target への α=0.5 の更新を受理した。

受理後の Q は約 `[3.847, 3.780, 3.638, 0.445, 0.322, 0.429]` だった。

位置 RMSE は 49.687 m から 48.394 mへわずかに改善したが、deterministic baseline の 1.500 mには遠く及ばなかった。

## 現時点の判断

GUI backend の sparse MAP と Laplace-EM の配管は、GUI を通さない最小経路でも最後まで動作した。

一方で、現在の暫定 observation covariance、fixed-factor covariance、initial Q を用いた物理パラメータ推定は成功していない。

小さい latent MAP objective を、記録制御入力から観測軌道を再現できる静的物理パラメータが得られた証拠として扱ってはならない。

Laplace-EM iteration を増やすだけでは計算量を消費し、同定の混同を解消できる根拠がないため、今回は 1 iteration で止めた。

## 次に行う TODO

1. Position、orientation、direct velocity、gyro、accelerometer、controller、actuator、kinematics の covariance を一律の単位対角から分離し、各値の測定または仕様上の根拠を保存する。
2. Observation から作る初期 latent trajectoryと deterministic static parameter から valid interval の body-wrench dynamics residual を計算し、可変 dt 対応の robust per-axis scale を initial Q として保存する。
3. 校正した fixed Q で sparse MAP を解き、latent factor residual と static-parameter open-loop rollout の両方が改善する領域があるかを限定的に確認する。
4. Prior covariance を強くして baseline を固定する実験と、likelihood が静的パラメータを識別する実験を分離し、prior による維持を同定成功と呼ばない。
5. Fixed-Q MAP が deterministic baseline の再現性を大きく壊さないことを確認してから、Laplace-EM を複数 iteration 実行する。
6. Fixed-lag の同定が成立してから外側 lag profile を追加する。
7. MAP、Q、lag の科学的受入が済むまで MCMC と PID posterior evaluation を追加しない。

成果物は `output/result.json`、`output/trajectory.pdf`、`output/probabilistic/fixed_q/`、`output/probabilistic/laplace_em/` に保存している。
