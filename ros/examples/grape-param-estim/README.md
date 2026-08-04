# grape_param_estim

Grape の実 rosbag を一つの疎な全軌道問題として解き、静的物理パラメータ、一定 command delay、対角モデル誤差共分散、局所事後幾何、MCMC sample、PID 候補の posterior predictive 評価を一続きに扱う ROS package です。
各 knot の状態を同時に推定する batch smoothing を使い、観測時刻ごとの reset や時刻ごとの residual-wrench 未知状態は使いません。

## 収録 rosbag で GUI を起動する

最初に package を build し、後述の GUI 用 Python 3.10 以上の環境を用意してください。
次の例は失敗飛行を repository 内の sample から新規 project へコピーし、inspection を自動開始します。
推定区間は record local time の `18.000 s` から `24.000 s` を推奨します。

```bash
cd /home/leus/catkin_ws
catkin build grape_param_estim --no-deps
source devel/setup.bash
rosrun grape_param_estim run_gui.py \
  --projects-root /tmp/grape-param-estim-failed-demo \
  --bag "$(rospack find grape_param_estim)/samples/rosbags/20260612_grape_hovering_4_2026-06-12-17-33-59.bag"
```

次の例は飛行に成功した 2026 年 6 月 13 日の bag を同じ経路で開きます。
inspection では complete episode `7.2259--65.6365 s` と state=5 区間 `41.8469--62.9066 s` を確認しており、短い tuning run には例えば `45--51 s` を明示的な候補にできます。
この成功 bag は GUI inspection と tuning 区間の選定に使用済みなので、本 repository の検証では完全な hold-out と呼びません。
この bag を forecast に使う場合は `tuning_evaluation` と明記し、strict hold-out には未閲覧かつ tuning 未使用の別 bag を割り当てます。

```bash
cd /home/leus/catkin_ws
source devel/setup.bash
rosrun grape_param_estim run_gui.py \
  --projects-root /tmp/grape-param-estim-success-demo \
  --bag "$(rospack find grape_param_estim)/samples/rosbags/20260613_grape_hovering_3_2026-06-13-15-12-51.bag"
```

二つの bag を一つの project に追加する場合は `--bag` を二回指定できますが、同じ configuration group にまとめるのは payload、rotor、geometry、wiring、robot model、hardware revision が同一だと確認できる場合だけです。
成功 sample の topic contract は valid ですが、この六つの configuration provenance 項目は bag だけでは不足するため manual confirmation が必要です。
sample の原本 SHA256 は失敗 bag が `bd3fc7f71797c0f5cb665acc50832da93c590e540fa170f9977182ecedf93bf8`、成功 bag が `a1569a48bf9a1d4d3f10a40bfc0e2c3c0cba192660b32204eeb37d1416425071` です。

## GUI の Python 環境

GUI は Python 3.10 以上、PySide6、pyqtgraph、PyVista、PyVistaQt、VTK を使います。
推定 worker は ROS Python 環境で別 process として動くため、GUI の virtual environment と ROS の Python を同一にする必要はありません。

```bash
/home/leus/.pyenv/versions/3.10.18/bin/python -m venv \
  /home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/gui/.venv
/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/gui/.venv/bin/python -m pip install -e \
  /home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/gui
```

`run_gui.py` の shebang 自体に host 固有の virtual environment を埋め込んでいません。
`rosrun` で起動された launcher は PySide6 を import する前に、`GRAPE_PARAM_ESTIM_GUI_PYTHON`、active な `VIRTUAL_ENV`、package 内 `gui/.venv`、現在の Python 3.10 以上、の順で interpreter を選び、必要なら `execve` で一度だけ再実行します。
ROS worker の interpreter を明示する場合は `GRAPE_PARAM_ESTIM_WORKER_PYTHON` に ROS package を import できる Python の絶対 path を設定します。

## GUI workflow

bag inspection 後に `Bag browser` で使用する区間と sensor contract を確認し、`Run estimation…` から `STEP` または `ALL` を選びます。

- `STEP` は `estimate_only` を実行し、疎な MAP、delay profile、Laplace-EM の Q 更新、Laplace 幾何を保存したところで止まるため、中間結果の点検に向きます。
- `ALL` は `estimate_and_sample` を実行し、同じ推定に ridge-aware MCMC を続けて posterior sample まで生成します。

表示する軌道は observed、nominal、MAP、保存された selected posterior-sample conditional trajectory です。
observed は測定値、nominal は推定前モデル、MAP は全 factor を同時に満たす最尤点ではなく prior を含む最大事後点、conditional trajectory は選択した static sample と delay に条件づけた局所 trajectory MAP です。
観測と推定軌道が近いことだけでは十分でなく、normalized sensor residual、dynamics residual、Q band、ridge、MCMC 診断も一緒に確認してください。
`Dynamics residual` は状態と物理パラメータから決まる interval residual の表示であり、推定された外力時系列ではありません。

worker は JSON Lines の progress を標準出力へ、診断を標準エラーへ出し、GUI は nonlinear iteration、lag profile point、EM iteration、MCMC proposal、PID forecast の境界で停止要求を処理します。
不完全な directory は complete result として読み込みません。
同じ request identity と run directory に対する `resume=true` は、完了済み MAP/EM/Laplace と proposal-boundary MCMC checkpoint を検証して再利用します。
疎 factorization の途中は保存せず、保存済み MAP 点から再生成します。

## 推定対象と次元

全 selected bag で共有する静的 chart は 18 次元です。

| 共有静的量 | 次元 |
|---|---:|
| log mass | 1 |
| relative full-SPD inertia | 6 |
| CoG offset | 3 |
| log force effectiveness | 4 |
| log torque effectiveness | 4 |
| 合計 | 18 |

科学的には continuous constant delay 1 次元を加えた 19 個が共有未知量ですが、ZOH command に対する目的関数が区分的に滑らかなため、delay は 18 次元 Gauss--Newton block へ入れず、外側の一次元 profile optimization で求めます。
各 knot の local state は position 3、SO(3) tangent 3、world linear velocity 3、body angular velocity 3、PID integral 6、actual rotor thrust 4、actual gimbal angle 4、の合計 26 次元です。
bag ごとに gyro bias 3 次元を持ち、calibrated accelerometer factor を有効にした場合だけ accelerometer bias 3 次元も持ちます。
したがって inner MAP の次元は `18 + sum_b(26 N_b + 3 + accelerometer_bias_b)` であり、`N_b` は bag `b` の knot 数、`accelerometer_bias_b` は accelerometer 使用時だけ 3 です。
18--24 秒の検証 run は 119 knots、gyro bias 有効、accelerometer 無効なので、inner MAP は `18 + 26*119 + 3 = 3115` 次元です。
Q の 6 対角成分は Laplace-EM の hyperparameter、delay は外側 profile parameter であり、この 3115 次元には含めません。
時刻ごとの residual wrench を未知変数として積み上げないため、軌道が長くなると増えるのは物理的な knot state だけです。

## 数理の要点

各 interval で required body wrench と actuator、drag から得る modeled body wrench の差を 6 次元 dynamics residual とします。

```math
\xi_k = w_{\mathrm{required},k} - w_{\mathrm{modeled},k},\qquad
Q=\operatorname{diag}(q_{F_x},q_{F_y},q_{F_z},q_{\tau_x},q_{\tau_y},q_{\tau_z}).
```

`body_wrench/continuous_spectral_density` の場合は interval-average residual の covariance を `Q / dt_k` と定義し、可変 sampling interval を明示的に扱います。
Q は MAP residual の二乗だけではなく、Laplace covariance correction を加えた期待二乗から 6 成分を別々に更新します。
production factor は解析 Jacobian block を返し、finite difference は derivative test の oracle にだけ使います。
bag-local sparse block を消去して 18 次元 Schur complement を解くため、full dense Hessian や full covariance は作りません。

詳しい導出と実装対応は次を参照してください。

- [batch_estimator_formulation_ja.md](lectures/batch_estimator_formulation_ja.md)
- [analytic_jacobian_implementation_ja.md](lectures/analytic_jacobian_implementation_ja.md)
- [laplace_em_q_estimation_ja.md](lectures/laplace_em_q_estimation_ja.md)
- [ridge_and_mcmc_diagnostics_ja.md](lectures/ridge_and_mcmc_diagnostics_ja.md)
- [real_flight_validation_ja.md](lectures/real_flight_validation_ja.md)
- [pid_particle_evaluation_ja.md](lectures/pid_particle_evaluation_ja.md)

## one-command worker

GUI が保存する strict request JSON は CLI からも一回の command で実行できます。
推定 request schema は `grape-param-estim/batch-estimation-request/v1` で、`run_mode` は `estimate_only` または `estimate_and_sample` です。

```bash
cd /home/leus/catkin_ws
source devel/setup.bash
rosrun grape_param_estim grape_estimate_flights.py \
  --request /absolute/path/to/batch-estimation-request.json
```

request は output directory、bag の絶対 path と SHA256、interval、全 observation/fixed/prior covariance、Q の quantity・単位・interval model、18 次元 prior、delay profile、actuator model、knot/interpolation/controller policy、mode、solver、EM、MCMC の全設定を明示します。
covariance を message に記録されていない値から暗黙に補う default はなく、使用しない factor は理由付きで disabled にします。
actuator model も request の必須情報であり、hidden default はありません。

`estimate_only` の complete run へ後から MCMC を追加する場合は、同じ run directory と元の estimation request を束縛した `grape-param-estim/posterior-sampling-request/v1` を使います。
この worker は MAP、delay profile、Laplace-EM を再実行せず、MCMC 完了時だけ run directory を原子的に置き換えます。

```bash
rosrun grape_param_estim grape_sample_parameter_posterior.py \
  --request /absolute/path/to/posterior-sampling-request.json
```

cancel 時は元の complete estimate-only artifact を変更せず、同一 sampling request fingerprint の chain checkpoint だけを保持します。
再実行時は同じ request の `resume` だけを `true` にし、upstream run ID、bag SHA256/interval、configuration、controller、estimator revision、元 request fingerprint のいずれかが変われば拒否します。

PID evaluation request schema は `grape-param-estim/pid-proposal-evaluation-request/v2` です。
完了した MCMC 付き estimation run、元 bag、current・user candidate、MCMC-derived candidate population の全件または deterministic k-medoids 上限、必ず残す source sample、plant sample subset、model discrepancy policy、replicate seed、tail level を明示して実行します。

```bash
rosrun grape_param_estim grape_evaluate_pid_proposals.py \
  --request /absolute/path/to/pid-proposal-evaluation-request.json
```

PID worker は candidate × retained plant sample × selected bag × discrepancy replicate の full closed-loop simulation を行います。
まず全 retained MCMC sample から exact correlated PID gain と response を導出し、sample 数が多い場合だけ request に記録した deterministic k-medoids で評価候補数を制限します。
raw proposal population は sample ID と整列して artifact に保存され、component-wise mean を代表 gain として捏造しません。
request の `forecast_workers` は `auto` または `1--32` の明示値で、独立 forecast を deterministic process pool へ割り当てます。
完了 forecast は content-addressed checkpoint に保存され、同一 request fingerprint の `resume=true` では未完了の Cartesian-product record だけを再計算します。
current PID の正本は baseline rosbag の controller snapshot であり、repository の YAML を現在値として代用しません。
出力 YAML は提案ファイルであり、controller や `dynamic_reconfigure` を自動変更しません。

保持済み MCMC sample を別飛行へ連続 forecast する request schema は `grape-param-estim/held-out-validation-request/v2` です。
`strict_hold_out` は source estimation に含まれる bag SHA256、estimator tuning 使用、PID tuning 使用を拒否し、`tuning_evaluation` は少なくとも一方の tuning 使用を明示させます。
今回の成功 sample は既に GUI inspection と tuning 区間確認に使ったため、次の command を実行する場合も結果の意味は必ず `tuning evaluation (not held-out)` です。

```bash
rosrun grape_param_estim grape_validate_held_out_flight.py \
  --request /absolute/path/to/held-out-or-tuning-evaluation-request.json
```

worker は retained MCMC draw だけを plant sample とし、各 sample を先頭状態から区間末尾まで state replacement なしで連続 closed-loop forecast します。
観測軌道に対する誤差と reference 追従誤差は別 metric として保存し、future model discrepancy は zero または推定済み対角 Q からの新規 sample のどちらかを request で明示します。
過去の residual path は再生せず、source actuator model、固定 drag、forecast 設定、configuration compatibility の status と根拠も artifact に固定します。

## Artifact schema

推定 artifact は `grape-param-estim/batch-estimation-run/v1` の strict directory bundle です。

```text
estimation_run/
  manifest.json
  map_static.npz
  q_em.npz
  laplace.npz
  diagnostics.npz
  mcmc_samples.npz                 # estimate_and_sample のときだけ
  bags/<bag_id>.npz
  trajectories/<bag_id>/selected_samples.npz  # 保存対象があるときだけ
```

`manifest.json` は status、run/request/configuration/controller fingerprint、bag SHA256 と interval、sensor contract、factor on/off、prior、delay、明示的 actuator model、Q 定義、solver/EM/MCMC 設定、substage convergence、warning、各 file SHA256 を持ちます。
`map_static.npz` は 18 次元 MAP の物理値、delay、最終 Q、objective decomposition を持ち、`q_em.npz` は input/target/accepted Q、alpha、MAP と marginal objective、expected residual moment、MAP moment、covariance correction を持ちます。
`laplace.npz` は prior を分離した reduced likelihood/posterior Hessian、covariance、eigensystem、rank、ridge、delay profile を持ち、`mcmc_samples.npz` は equal-weight retained draw を `sample_id` で保持します。
`bags/<bag_id>.npz` は raw observation、nominal/MAP trajectory、dynamics residual、normalized factor residual、covariance、数値診断を持ち、latent residual-wrench path は持ちません。
NPZ は `allow_pickle=False` で読み、object dtype、unknown key、shape/単位不一致、SHA256 不一致、`status != complete` を拒否します。

PID artifact は `grape-param-estim/pid-proposal-evaluation/v2` です。

```text
pid_proposal_evaluation/
  manifest.json
  source_samples.npz
  candidate_particles.npz
  summary.npz
  proposed_GimbalrotorControl.yaml
  proposed_GimbalrotorControl.diff.yaml
  bags/<bag_id>.npz
```

`source_samples.npz` は physical posterior に加えて、全 retained sample の exact group scale、gain、acceleration response を同じ sample order で持ちます。
manifest は raw/evaluated derived candidate 件数、all-raw または k-medoids policy、上限、必須 source sample を持ち、評価 candidate が raw proposal と一致しなければ loader が拒否します。

推薦条件を満たさない場合、`recommendation_available=false` と理由を保存し、提案 YAML を実機へ自動適用しません。

## 18--24 秒の実 bag 検証

2026 年 8 月 4 日に失敗 sample の `18.0--24.0 s` を 0.05 s knot、MCMC 無効、EM 最大 1 iteration の短縮設定で end-to-end 実行し、251.75 秒で complete artifact を生成しました。
MAP は `relative_objective_tolerance` で収束し、Laplace 幾何は完了しましたが、Laplace-EM は意図的に最大 1 iteration としたため `maximum_iterations` で非収束です。
選択 delay は境界の `0.0 s`、mass MAP は `1.19122 kg`、Q は `[14.4779, 14.5221, 13.4510, 0.0527430, 0.0568025, 0.184592]` でした。
初期 Q `[25, 25, 25, 1, 1, 1]` からの更新は `alpha=1` で受理されましたが、sensor/factor covariance と prior に暫定値を使い、EM を一回しか回していないため、これらを同定済みの実機値とは解釈しません。
thrust time constant `0.01 s` は `MotorInfo.yaml` の情報に基づきますが、gimbal time constant `0.02 s` は暫定値であり、別途 system identification が必要です。

この run は 119 knots、5495 factors、residual dimension 21859、Jacobian nnz 149367 でした。
assembly は 0.186 秒、bag-local factorization は 0.046 秒、Schur solve は 0.020 秒程度ですが、LM 一回は約 2.5 秒、EM 一回は 248.36 秒でした。
時間の大半は単発の forward propagation ではなく、複数の delay 候補と Q 候補について sparse MAP を繰り返す lag profile と Laplace-EM に使われます。
したがってこの処理を「ただの順方向 forecast」とみなして rollout だけを並列化しても支配時間は解消しません。
詳細は [real_flight_validation_ja.md](lectures/real_flight_validation_ja.md) に記録しています。

## Build と test

```bash
cd /home/leus/catkin_ws
catkin build grape_param_estim --no-deps
catkin run_tests grape_param_estim --no-deps
source devel/setup.bash
```

backend の synthetic generator は次で起動できます。

```bash
rosrun grape_param_estim grape_generate_synthetic_flight.py \
  --output /tmp/grape_synthetic_closed_loop.npz
```

GUI test は package-local environment から実行します。

```bash
cd /home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/gui
MPLCONFIGDIR=/tmp/grape-mpl-cache \
QT_QPA_PLATFORM=offscreen \
GRAPE_PARAM_ESTIM_DISABLE_3D=1 \
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

## 安全上の境界

この package は parameter posterior と counterfactual PID 評価を研究用に生成しますが、飛行安全を保証しません。
MAP の見栄え、狭い Laplace marginal、MCMC の sample 数、in-sample trajectory の一致だけで推定成功と判断しないでください。
sensor frame、covariance provenance、actuator dynamics、delay 境界、Q 収束、likelihood ridge、R-hat/ESS、複数 bag の data split、PID の completion と tail metric を確認し、実機 controller の変更は人間が別手順で判断してください。
