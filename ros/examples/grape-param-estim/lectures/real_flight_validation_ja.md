# 実飛行 rosbag による sparse batch validation

## 1. この報告の結論

新しい sparse batch estimator は、指定された失敗飛行の `18.0--24.0 s` を読み、pose、velocity、gyro、controller/actuator observation、kinematics、dynamics を一つの全軌道問題として解き、strict v1 artifact を complete 状態で出力した。
MAP は relative objective tolerance で収束し、Laplace 幾何も完了した。
一方、Laplace-EM は検証時間を抑えるため最大一 iteration に制限したので非収束であり、この `18.0--24.0 s` run では MCMC と PID evaluation を実行していない。
別の clean `18.0--18.3 s` run では estimate→MCMC→selected conditional trajectory→PID evaluation→成功飛行 tuning evaluation までを実行し、artifact 間の配管と監査契約を確認した。
どちらも実機 parameter、Q、delay、MCMC、PID の科学的な校正結果ではない。

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
- current use: GUI demo と tuning evaluation で、短区間 `45.0--45.3 s` を実際に使用した。

成功 bag の flight outcome と inspection contract は既に確認しているため、本報告では完全 hold-out と呼ばない。
payload、rotor/propeller、geometry、robot model revision、actuator wiring、hardware revision の provenance は bag だけでは不足し、configuration group の manual confirmation が必要である。
この bag を tuning に使った事実は manifest/report に記録し、別の未閲覧 bag を外部 validation 用に確保する。

## 3. request 設定

validation run ID は `failure-04-real-c-18.0-24.0` で、artifact は `/tmp/grape-sparse-real-18-24-run-20260804-c` に生成した。
artifact の `estimator_revision` は当時の `f2d5984...-dirty` であり、現在の clean HEAD による最終科学 run ではない。
以下の数値は実データ経路と表示の診断には使えるが、現 HEAD の再現性証明には別の clean run が必要である。
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

### 6.1 observation と dynamics の定量的な再現精度

artifact に保存された observation time へ nominal/MAP knot state を補間し、position/velocity/gyro/gimbal/controller integral は全 component の RMS、orientation は quaternion geodesic RMS を計算した。
ここで nominal は observation から作った smoothing initialization であり、物理モデルの open-loop forecast ではない。
したがって nominal が observation に近いこと自体はモデル予測性能を意味せず、比較は MAP が initialization からどれだけ動いたかを示す診断である。

| quantity | nominal RMS | MAP RMS | nominal maximum | MAP maximum |
|---|---:|---:|---:|---:|
| position component [m] | `0.001364` | `0.030819` | norm `0.00980` | norm `0.06435` |
| orientation geodesic [rad] | `0.019117` | `0.017722` | `0.09083` | `0.04974` |
| linear velocity component [m/s] | `0.002337` | `0.040992` | norm `0.01228` | norm `0.14401` |
| gyro component [rad/s] | `0.015901` | `0.025188` | norm `0.16659` | norm `0.13026` |
| actual gimbal component [rad] | `0.007276` | `0.014852` | norm `0.07387` | norm `0.08000` |
| controller integral component | `0.000180` | `0.000658` | norm `0.00410` | norm `0.00368` |

MAP は orientation RMS と一部 maximum error を改善したが、position、velocity、gyro、gimbal、integral の RMS は悪化した。
これは全 factor、prior、dynamics の妥協点が observation-anchored initialization から離れたことを示し、「観測軌道をよく再現した」とは評価できない。

MAP dynamics residual の軸別 RMS は force が `[0.44874, 0.15035, 0.73794]`、torque が `[0.040756, 0.201882, 0.040814]` だった。
全 force component RMS は `0.50614`、全 torque component RMS は `0.12122` である。
一方、最終 Q の `sqrt(Q/dt)` で規格化した軸別 RMS は `[0.02637, 0.00882, 0.04499, 0.03968, 0.18941, 0.02124]` で、全 708 scalar residual が one-Q band 内に入った。
factor-whitened dynamics residual も RMS `0.08236`、maximum absolute `0.42698` と小さい。
これは dynamics が十分に説明できた証拠というより、暫定 covariance と一回だけ更新した Q band が raw residual に対して広いことを示すため、Q が parameter/controller mismatch を吸収している可能性を否定できない。

nominal から MAP への correction-transform は translation component RMS `0.03073 m`、rotation-vector component RMS `0.01209 rad`、maximum norm がそれぞれ `0.06545 m` と `0.09422 rad` だった。
GUI で見えた軌道の分離は plotting artifact ではなく、この保存値に対応する。

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

GUI の observed は sensor measurement、nominal は観測を補間・平滑化して作る推定初期軌道、MAP は全 factor と prior を同時に最適化した latent trajectory である。
nominal は open-loop model forecast ではないため、observed との近さをモデルの予測性能として読まない。
selected posterior-sample conditional trajectory は MCMC 付き run でだけ存在する。
observed と MAP が近いことは pose fit の診断だが、dynamics residual や他 sensor residual が大きいままでも起こりうる。
`Dynamics residual` panel では force/torque の MAP residual、normalized residual、`sqrt(Q/dt)` band を比較し、特定軸だけが Q に吸収されていないか確認する。

2026 年 8 月 4 日に package 内 Python 3.10 environment の PySide6、pyqtgraph、PyVista、PyVistaQt、VTK を使い、`rosrun` で sample bag の inspection と実 3D 描画を確認した。
同日にこの実 artifact を production widget へロードし、world trajectory、correction transform、dynamics residual、Master の MAP/Q/EM/Laplace 表示を実画面で確認した。
画像は [rewrite 後の GUI visual acceptance](figures/gui_after_batch_rewrite/README.md) に保存している。
画面上でも observed、nominal、MAP の大きな乖離が見えたため、数値的に artifact が完走したことと科学的に推定が成功したことを区別している。

## 10. clean 短区間 E2E evidence

### 10.1 estimate と posterior sampling

clean source `run b` は失敗 bag の `18.0--18.3 s` を 5 knots で推定し、後段 worker で同じ complete estimate-only artifact へ MCMC を追加した。
estimate と sampling の `estimator_revision` / `sampler_revision` は `5b08e5c290925d7585024f3c5350a7f88a7f1fe9` である。

| substage | result |
|---|---|
| estimate-only wall time | `9.04 s` |
| MAP | converged、`gradient_tolerance` |
| Laplace | complete |
| Laplace-EM | non-converged、`maximum_iterations` |
| posterior sampling wall time | `11.71 s`、progress elapsed `11.316 s` |
| retained draws | 2 chains × 4 draws = 8 |
| selected conditional trajectories | 8/8 retained draws × 1 bag |
| conditional objective audit | strict all-close、maximum absolute error `8.88e-15` |
| MCMC convergence | configured R-hat/ESS thresholds not satisfied |

各 selected trajectory は retained static coordinate と delay を固定した fresh conditional sparse MAP であり、保存した conditional objective を同じ sample の MCMC target breakdown と照合した。
現行 policy では selected conditional trajectories は posterior sample の局所状態診断と可視化のための保存物であり、PID forecast の sample 別初期状態には使わない。
後述する旧 PID `run c` だけは `608decf` より前の初期条件 policy で実行したため、この原則の validation evidence には数えない。

最初の試行では proposal 間の trajectory warm start を引き継ぐと、非線形 solver の停止点を介して同じ proposal の target が chain history に依存する不具合を検出した。
revision `5b08e5c` は全 exact target evaluation を共通の selected-mode MAP warm start から開始し、target を proposal point だけの関数に戻した修正版である。
上表の `run b` はこの修正後に最初から作り直した結果である。

4 draws/chain は配管確認用の極小設定であり、warning `MCMC completed without satisfying convergence thresholds` が正しい判定である。
この sample を converged posterior と呼ばず、parameter や ridge の科学的推論に使わない。

### 10.2 PID cross-evaluation

PID `run c` は controller snapshot fingerprint を estimation と同じ bag order で計算する修正 `f56142b` の後に作り直した v2 artifact である。
修正前の試行は、flight input の順序契約を正しく fingerprint できず拒否されたため、validation evidence に数えない。

| quantity | result |
|---|---|
| wall time | `2.78 s` |
| candidates | current + sample-derived = 2 |
| plant population | explicit 2-sample subset |
| forecasts | `2 × 2 = 4` |
| completion mean | current `1.0`、sample-derived `1.0` |
| numerical failures | 0 |
| actuator saturation duration/rate | `0 s` / `0` |
| recommendation | unavailable |

current と sample-derived はどちらも non-dominated だったが、sample-derived は current に対する全 performance component の Pareto 改善を満たさなかった。
そのため rejection reason は `recommendation unavailable: no Pareto candidate improves current` である。
さらに explicit 2-sample subset は smoke/exploratory evaluation であり、現行 revision `608decf` は subset 上で改善が見えても retained posterior 全体の finalist reevaluation なしに推薦を出さない。
PID `run c` は `608decf` より前の revision で実行したため、明示した 2 plant samples の保存済み conditional trajectory initial state を使用した。
`608decf` 以降の現行 policy は全 plant sample を共通の `shared_selected_mode_map_initial` から始め、保存済み selected conditional trajectory を診断と可視化にだけ使用する。
したがって上表の metric は旧初期条件 policy による配管 smoke evidence であり、最終 HEAD の現行 policy で再実行するまで性能比較へ使わない。

### 10.3 成功飛行の tuning evaluation

成功 bag の `45.0--45.3 s` に 2 retained samples を連続 forecast する `tuning run b` は wall `3.83 s` で complete になった。
data split role は `tuning_evaluation`、semantic label は `tuning evaluation (not held-out)` であり、`strict_hold_out` ではない。
future discrepancy は `zero_model_discrepancy` とした。

| metric | mean |
|---|---:|
| observed position RMSE | `0.0201115 m` |
| observed orientation RMSE | `0.0563624 rad` |
| observed maximum position error | `0.0382226 m` |
| observed maximum orientation error | `0.0988568 rad` |
| reference position RMSE | `0.0612683 m` |
| reference orientation RMSE | `0.147179 rad` |
| reference maximum position error | `0.0634088 m` |
| reference maximum orientation error | `0.154781 rad` |
| forecast completion | `1.0` |
| numerical failure count | `0` |
| actuator saturation duration/rate | `0 s` / `0` |

2 retained samples が同じ値を返したため mean、configured quantile、upper CVaR はこの短区間では同じだった。
失敗飛行と翌日の成功飛行が同一 campaign であること以外に exact hardware/configuration identity は独立確認できず、artifact の compatibility status は `unconfirmed` である。
したがってこの誤差を parameter mismatch だけに帰属できず、成功飛行への predictive generalization を確認したとも言えない。

この clean E2E は estimate、posterior append、conditional trajectory、PID Cartesian product、別飛行 tuning artifact が整合することを検証した plumbing smoke test である。
短い区間、EM 一回、4 draws/chain、2-sample PID subset、zero discrepancy、未確認の configuration compatibility という条件なので、parameter、Q、MCMC、PID の科学的成功を主張しない。

## 11. 次の validation

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
