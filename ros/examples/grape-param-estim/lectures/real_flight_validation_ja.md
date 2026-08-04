# 実飛行 rosbag による sparse batch validation

## 1. この報告の結論

新しい sparse batch estimator は、指定された失敗飛行の `18.0--24.0 s` を読み、pose、velocity、gyro、controller/actuator observation、kinematics、dynamics を一つの全軌道問題として解き、strict v1 artifact を complete 状態で出力した。
MAP は relative objective tolerance で収束し、Laplace 幾何も完了した。
一方、Laplace-EM は検証時間を抑えるため最大一 iteration に制限したので非収束であり、MCMC と PID evaluation はこの run では実行していない。
したがってこの結果は end-to-end 数値経路の成立を示す smoke/validation であり、実機 parameter、Q、delay、PID の科学的な校正結果ではない。

## 2. data split

### 2.1 estimator validation に使用した失敗 bag

- repository sample: `samples/rosbags/20260612_grape_hovering_4_2026-06-12-17-33-59.bag`。
- original: `/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/20260612_grape_hovering_4_2026-06-12-17-33-59.bag`。
- SHA256: `bd3fc7f71797c0f5cb665acc50832da93c590e540fa170f9977182ecedf93bf8`。
- selected local record-time interval: `18.0--24.0 s`。
- use: rosbag adapter、sensor contract、initialization、sparse MAP、delay profile、Q update、Laplace、artifact、performance の開発・検証。

### 2.2 成功 bag の現在の位置づけ

- repository sample: `samples/rosbags/20260613_grape_hovering_3_2026-06-13-15-12-51.bag`。
- original: `/home/leus/catkin_ws/bags/grape-drone/20260613_grape_hovering/20260613_grape_hovering_3_2026-06-13-15-12-51.bag`。
- SHA256: `a1569a48bf9a1d4d3f10a40bfc0e2c3c0cba192660b32204eeb37d1416425071`。
- inspection result: complete episode `7.2259--65.6365 s`、推奨 state=5 区間 `41.8469--62.9066 s`、topic contract valid。
- current use: GUI demo と未実施の data split 候補で、短い tuning run なら例えば `45--51 s` を明示的に選べる。

成功 bag の flight outcome と inspection contract は既に確認しているため、本報告では完全 hold-out と呼ばない。
payload、rotor/propeller、geometry、robot model revision、actuator wiring、hardware revision の provenance は bag だけでは不足し、configuration group の manual confirmation が必要である。
今後この bag を tuning や candidate selection に使った場合は、その事実を manifest/report に残し、別の未閲覧 bag を外部 validation 用に確保する。

## 3. request 設定

validation run ID は `failure-04-real-c-18.0-24.0` で、artifact は `/tmp/grape-sparse-real-18-24-run-20260804-c` に生成した。
run mode は `estimate_only`、knot period は `0.05 s`、solver maximum iteration は 30、delay bounds は `0.0--0.08 s` である。
delay profile は coarse 3 点、refinement 最大 2 evaluation、tolerance `0.01 s` の短縮設定である。
EM は minimum/maximum ともに 1 iteration とした。
MCMC は無効である。

Q definition は `body_wrench/continuous_spectral_density` である。
初期対角値は force が各 `25`、torque が各 `1`、floor は各 `1e-8` とした。
静的 18 次元 prior は chart origin、unit covariance の暫定設定であり、飛行データから校正した prior ではない。

actuator contract は thrust time constant `0.01 s`、gimbal time constant `0.02 s`、thrust bounds `1.5--27.6145 N`、gimbal angle limit `3.14 rad`、gimbal rate limit `6.0 rad/s` とした。
thrust time constant と limits は `gimbalrotor` の `MotorInfo.yaml` / URDF 情報に基づく。
gimbal time constant `0.02 s` は repository に校正値が見つからなかったため暫定値であり、step response 等による system identification が必要である。

## 4. sensor contract

主な direct observation/input は次である。

| quantity | topic | status |
|---|---|---|
| FC pose in world | `/gimbalrotor/mocap/pose` | pose factor に使用 |
| world linear velocity | `/gimbalrotor/uav/baselink/odom` | velocity factor に使用 |
| FC angular velocity | `/gimbalrotor/sensor_plugin/imu1/ros_converted` | gyro factor に使用 |
| specific force | 同上 | sensor origin が独立校正されていないため disabled |
| actual gimbal position | `/gimbalrotor/joint_states` | observation に使用 |
| issued gimbal command | `/gimbalrotor/gimbals_ctrl` | causal input/factor に使用 |
| issued rotor command | `/gimbalrotor/four_axes/command` | causal input/factor に使用 |
| PID debug/integral/reference | `/gimbalrotor/debug/pose/pid` | controller reconstruction/factor に使用 |
| flight state | `/gimbalrotor/flight_state` | recorded mode schedule に使用 |

body frame は `gimbalrotor/main_body`、pose/velocity/gyro sensor frame は `gimbalrotor/fc` とした。
main-body から FC への transform は bag の `/tf_static` を読み、Grape URDF の `fc_joint` と照合した。
odometry pose と angular twist、derived CoG odometry、duplicate native IMU、duplicate acceleration topic は直接 observation として重複使用していない。
message covariance が zero または存在しない quantity が多いため、今回の request covariance は暫定 project configuration である。

## 5. problem size と実行時間

| quantity | value |
|---|---:|
| wall time | `251.75 s` |
| knot count | `119` |
| inner MAP dimension | `3115` |
| factor count | `5495` |
| residual dimension | `21859` |
| Jacobian nnz | `149367` |
| peak memory | `680,513,536 B` |
| representative assembly | `0.1856 s` |
| representative factorization | `0.0463 s` |
| representative Schur solve | `0.0203 s` |
| nonlinear iteration | 約 `2.5 s` |
| EM iteration | `248.36 s` |

inner MAP dimension は shared static 18、119 knots の各 26、bag-local gyro bias 3 の `18 + 119*26 + 3 = 3115` である。
accelerometer factorを disabled にしたため accelerometer bias 3 は存在しない。
delay 1 と Q 6 は inner vector ではなく外側 parameter/hyperparameter なので、この数に含めない。

表示上 `forecast` と見える時間の大半は単発の forward time evolution ではない。
この run は delay coarse/refinement 点と Q input/candidate について sparse MAP を何度も解き、各 solve が複数 LM iteration を持つ。
一 LM iteration の assembly/factorization/Schur 自体は秒未満だが、factor evaluation と trial objective を含む iteration 全体が約 2.5 秒で、それを 71 回実行している。
したがって単一 rollout の C++ 化や独立 rollout の並列化を先に行う問題ではなく、profile evaluation 数、warm-start、symbolic structure reuse、factor kernel を測って順に最適化する必要がある。

## 6. convergence と MAP

| substage | converged | termination reason |
|---|---|---|
| MAP | true | `relative_objective_tolerance` |
| Laplace-EM | false | `maximum_iterations` |
| Laplace | true | `completed` |

selected delay は lower boundary の `0.0 s` だった。
delay profile objective は `0.0 s` が最小で、保存された grid は `[0.0, 0.00764, 0.01236, 0.02, 0.04] s` である。
local quadratic curvature からの nominal uncertainty は約 `0.0131 s` だが、最小が boundary にあるため symmetric Gaussian uncertainty として強く解釈しない。

主な static MAP は次である。

| quantity | MAP |
|---|---:|
| mass | `1.19122 kg` |
| CoG | `[0.00203, -0.00397, -0.00240] m` |
| force effectiveness | `[0.42345, 0.48039, 0.41941, 0.42325]` |
| torque effectiveness | `[0.96836, 0.99159, 0.95191, 0.98079]` |

これらは暫定 unit prior、暫定 observation covariance、暫定 actuator dynamics、単一短区間の条件付き MAP である。
parameter marginal や trajectory の見栄えだけから実機物理値として採用しない。

## 7. Q update

初期 Q `[25, 25, 25, 1, 1, 1]` に対する Laplace-EM target は `[14.4779, 14.5221, 13.4510, 0.0527430, 0.0568025, 0.184592]` だった。
candidate は `alpha=1` で marginal objective 非悪化条件を満たし、そのまま accepted Q になった。
floor activation は全成分 false である。

MAP residual second moment は `[0.02925, 0.00250, 0.07976, 0.000201, 0.007242, 0.000111]` だった。
covariance correction は `[14.4486, 14.5196, 13.3712, 0.052542, 0.049561, 0.184480]` で、expected second moment の大部分を占めた。
これは Q が MAP residual の二乗だけで更新されていないことを確認する実装診断である。

ただし一回の update は収束ではない。
minimum/maximum iteration を増やした run、covariance calibration、interval/knot sensitivity、actuator time constant sensitivity、複数 bag joint fit が必要である。

## 8. Laplace と ridge

Laplace artifact は reduced likelihood/posterior Hessian、static covariance、eigensystem、exact ridge direction、delay profile を strict shape で保存した。
likelihood eigenvalue は約 `9.92e-5` から `1.31e4` に広がり、condition number は約 `1.32e8` だった。
一方で effective-rank threshold 上は 18 と判定され、exact common-scale direction と最小 numerical direction の alignment は低かった。
この条件では「期待した exact ridge を実 bag で確認した」とは言えず、factor contract と covariance 校正後に synthetic/real の両方で再評価する。

## 9. trajectory 表示の読み方

GUI の observed は sensor measurement、nominal は推定前の model trajectory、MAP は全 factor と prior を同時に最適化した latent trajectory である。
selected posterior-sample conditional trajectory は MCMC 付き run でだけ存在する。
observed と MAP が近いことは pose fit の診断だが、dynamics residual や他 sensor residual が大きいままでも起こりうる。
`Dynamics residual` panel では force/torque の MAP residual、normalized residual、`sqrt(Q/dt)` band を比較し、特定軸だけが Q に吸収されていないか確認する。

## 10. 次の validation

1. pose、velocity、gyro、command、fixed-factor covariance を実測または再現性試験から校正する。
2. gimbal time constant と delay を別実験で識別し、両者の競合を減らす。
3. EM を複数 iteration 実行し、Q、lag、MAP/marginal objective の停止条件を満たすか確認する。
4. knot period、interval、prior、Q initial value、mode hypothesis の sensitivity を記録する。
5. additional failure bag を同じ configuration group の joint problem に加える。
6. ridge geometry が synthetic expectation と整合した後で multiple-chain MCMC を実行する。
7. tuning に使わなかった別 bag を外部 validation として確保する。
8. MCMC convergence と predictive diagnostics を満たした後にだけ PID candidate 評価へ進む。

現時点の結論は「新 backend が実 bag で finite に完走し、strict artifact と性能診断を生成した」である。
「Q、delay、mass、effectiveness が実機の真値として同定された」または「次回 PID を変更できる」とは結論していない。
