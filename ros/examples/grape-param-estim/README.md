# Grape plant assimilation and replay audit

Grape の rosbag を直接読み、飛行時 controller の再現可能性監査、記録済み
command による実機応答の Bayesian 同定、failure/success posterior predictive
validation、controller candidate の安全 gate、および従来の逆動力学 baseline
を扱うオフライン解析パッケージです。実機 parameter を書き換える経路はなく、
出力は human review 用の artifact または元 bag と別の解析用 ROS 1 bag
です。bag の読み出しに `rosbag play` や ROS master は必要ありません。

数式、座標系、filter、bag merge、ProbTF 表現、既知の限界は [`lectures/grape_parameter_estimation.md`](lectures/grape_parameter_estimation.md) にまとめています。
再設計の契約、段階、禁止事項は
[`plans/grape_param_estim_redesign_plan.md`](plans/grape_param_estim_redesign_plan.md)
が正本です。

## Controller、actuator、plant の境界

同定では次の三者を別 object として扱います。

- `ControllerSnapshot` は、飛行時 controller が信じていた nominal mass、
  inertia、CoG、geometry、gain、limit、mode option の固定 snapshot です。
  plant particle を変更しても、この snapshot と teacher-forced controller
  output は変化しません。
- actuator は `FourAxisCommand` や gimbal target を受け、delay、lag、
  scale、bias、saturation を経た realized wrench を生成します。
  `FourAxisCommand.base_thrust` を無条件に N とみなしません。
- plant は actuator wrench で時間発展する実機候補です。現在の未校正 bag
  では mass と thrust scale を分離せず `effective_closed_loop_v1` として
  報告します。校正 gate がない結果を physical mass/inertia と表示しません。

PID gain や controller nominal model を plant parameter と同時推定しません。
controller tuning は plant posterior を得た後の別段階です。

## 四つの実行モード

| mode | 入力と目的 | 現在の状態 |
|---|---|---|
| `factual_controller_replay` | bag に記録された feedback/target/state を固定 controller へ与え、記録 command と比較 | audit、fixture、conformance contract は実装済み。current bag は必要 snapshot が欠けるため `pc_exact` 不可 |
| `open_loop_plant_identification` | 記録済み command を actuator + plant particle へ与えて plant posterior を推定 | current bag 用 production path。controller backend を呼ばない |
| `closed_loop_plant_identification` | particle が生成した feedback を固定 controller へ戻して command を毎 tick 再計算 | forward path はあるが、passing `pc_exact` factual replay gate がない run は開始しない |
| `posterior_controller_evaluation` | weighted plant posterior 全体で別の `ControllerCandidate` を評価 | validation/design API はあるが、exactness、calibration、support、failure/success gate 通過前は recommendation を生成しない |

Python controller surrogate は smoke test、proposal、screening 専用であり、
`pc_exact` と表示されることはありません。PC-side exactness と spinal/PWM
exactness も `pc_exact` / `pc_mcu_exact` として分離します。

## ビルド

```bash
cd /home/leus/catkin_ws
catkin build grape_param_estim
source devel/setup.bash
```

## 最小実行サンプル

外部 bag や ROS master を使わず、インストール済み CLI と frozen artifact
の整合性だけを確認する最小スモークは次の一行です。

```bash
rosrun grape_param_estim verify_inverse_dynamics_baseline.py
```

成功時は `verified_runs` に `20260612-04`、`20260612-07`、
`20260612-08` が表示され、combined payload listing SHA-256 は
`76575356f084d99e5d1a2d1af9a3282d4208afb6b61c697ca06da1652773131c`
になります。

実 bag を読み、再設計 pipeline の入力、hash、区間、smoother までを最小構成で
確認する場合は、posterior 計算と artifact 書き込みを行わない
`--prepare-only` を使います。

```bash
GRAPE_BAG_ROOT=/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering

rosrun grape_param_estim estimate_grape_plant.py \
  --config "$(rospack find grape_param_estim)/config/plant_assimilation.yaml" \
  --bag-root "$GRAPE_BAG_ROOT" \
  --output-root /tmp/grape_plant_prepare \
  --prepare-only
```

この command は bag 4 を inference failure、bag 3 を held-out failure、
bag 7/8 を held-out success として準備し、run directory は作りません。

plant assimilation の既定設定は `config/plant_assimilation.yaml`、従来の
inverse-dynamics estimator は `config/estimator.yaml` です。prior は URDF
nominal 値を中心とする Gaussian ではなく、物理制約を満たす有限範囲の
bounded distribution です。乱数 seed、parameter bounds、入力 bag、使用区間、
設定内容は再現性と provenance の一部です。

## Controller replay sufficiency audit

current bag 4、7、8 を監査し、plan §8.3 の各入力を `AVAILABLE`、
`DERIVABLE`、`MISSING` に分類します。

```bash
GRAPE_BAG_ROOT=/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering

rosrun grape_param_estim audit_grape_controller_replay.py \
  --bag-root "$GRAPE_BAG_ROOT" \
  --output /tmp/controller_replay_audit.json
```

同一名の artifact は暗黙に上書きしません。明示的に置換する場合だけ
`--force` を指定します。2026-07-28 時点の canonical audit は
[`results/controller_replay_audit/2026-07-28/INDEX.md`](results/controller_replay_audit/2026-07-28/INDEX.md)
にあり、結果は次のとおりです。

| bag | AVAILABLE | DERIVABLE | MISSING | exact replay |
|---|---:|---:|---:|---|
| 4 | 8 | 3 | 4 | blocked |
| 7 | 9 | 4 | 2 | blocked |
| 8 | 9 | 4 | 2 | blocked |

bag 4 は full navigator target と control mode、完全な nominal
model/geometry snapshot、torque allocation matrix がありません。bag 7/8
にも後二者がありません。controller tick、controller が参照した onboard
state、integration/reset timeline などは候補 topic から導出可能でも、
materialize と conformance が済むまでは exact input としません。そのため
current bag の `exact_replay_ready` は全て `false` です。

## PC-side exact controller と closed-loop 入力

`jsk_aerial_robot` 側では `PoseLinearControllerCore` と
`GimbalrotorAllocationCore` を ROS 非依存の single source of truth とし、
live wrapper と `gimbalrotor_controller_replay` が同じ core を呼びます。
offline executable は batch request に加え、JSON-lines の `--server` mode
を持ちます。Python 側は一つの persistent process を全 rollout で共有しつつ、
particle ごとに独立した controller state を export/import します。
episode/particle likelihood は決定的な順序を保つ bounded worker pool で評価し、
同じ cache key の同時 miss は single-flight で直列化します。closed-loop では
互換な one-tick request を短い FIFO window で一つの multi-job C++ request
へまとめ、応答と controller final state を各 particle の stateful adapter
へ戻します。worker 数、chain worker 数、exact batch size、batch wait は
`inference` config と posterior provenance の両方に記録されます。

closed-loop CLI は次の五つを全て明示した場合だけ有効です。

- `--exact-replay-executable`
- `--controller-fixture-bundle`
- `--controller-snapshot-bundle`
- `--controller-state-bundle`（各 episode 共通、または nuisance sample ごと）
- `--factual-conformance-report`（executable/source/fixture/request/channel metric を束ねる）

どれか一つでも欠ける場合、bare boolean、surrogate、artifact/source
identity 不一致、command timestamp 不一致、未知の PID/allocation state は
SMC 開始前に拒否されます。current config は audit 結果に合わせて
`unavailable_from_current_bag` のままなので、五入力を与えても policy gate を
通りません。future replay recording から完全な fixture を作った run だけが
`closed_loop_plant_identification` へ進めます。

### Future bag から exact input を materialize する

future bag が `ReplayMetadata` / `ReplayFrame` を持つ場合は、一つの command
で bag を直接読み、同じ config と source-bag hash から
`estimate_grape_plant.py` と同じ episode preparation を行い、C++ exact
executable による factual conformance まで実行します。

```bash
rosrun grape_param_estim materialize_grape_controller_replay.py \
  --config /path/to/closed_loop_plant_assimilation.yaml \
  --bag-root "$GRAPE_BAG_ROOT" \
  --exact-replay-executable \
    /home/leus/catkin_ws/devel/.private/gimbalrotor/lib/gimbalrotor/gimbalrotor_controller_replay \
  --output-root /tmp/grape_exact_inputs \
  --run-id future-flight-001 \
  --write-stream /tmp/future-flight-001.replay-stream.json
```

`--write-stream` は任意で、bag から抽出した content-addressed canonical
JSON を新規 file として保存します。ROS/rosbag のない test・offline
environment では、同じ JSON を `--stream` へ渡せます。`--stream` と
`--write-stream` は同時指定できません。

controller tick の event time は `message.header.stamp` だけを採用し、bag
record time を fallback にしません。両時刻と bag record-time origin は
canonical stream に別々に保存され、Header time は ROS nanosecond 精度で
bag start からの offset に正規化されます。observation / likelihood /
report grid は prepared bag episode から保持し、plant integration grid
だけをその grid と全 `ReplayFrame` tick の和集合にします。controller
input、state before/after、output、event は `ReplayFrame` 以外から補完
しません。

出力 directory は atomic に公開され、同じ run ID を上書きしません。
全 JSON は content hash を持ち、次の六 file を生成します。

| materialized file | 用途 |
|---|---|
| `controller_replay_fixture_bundle.json` | bag/window/normalized episode hash と exact tick/input |
| `controller_snapshot_bundle.json` | episode ごとの frozen controller snapshot |
| `controller_state_bundle.json` | recorded initial PID/allocation/controller state |
| `exact_replay_request_bundle.json` | executable へ渡した hash-bound request |
| `exact_conformance_fixture_bundle.json` | recorded output/event と extraction provenance |
| `exact_episode_conformance_bundle.json` | episode ごとの passing typed report と全入力 hash linkage |

CLI の JSON stdout は、下流 `estimate_grape_plant.py` に必要な五つの exact
option path を返します。手動で渡す場合は次の形です。

```bash
EXACT_RUN=/tmp/grape_exact_inputs/future-flight-001

rosrun grape_param_estim estimate_grape_plant.py \
  --config /path/to/closed_loop_plant_assimilation.yaml \
  --bag-root "$GRAPE_BAG_ROOT" \
  --output-root /tmp/grape_plant_runs \
  --run-id future-flight-001 \
  --exact-replay-executable \
    /home/leus/catkin_ws/devel/.private/gimbalrotor/lib/gimbalrotor/gimbalrotor_controller_replay \
  --controller-fixture-bundle \
    "$EXACT_RUN/controller_replay_fixture_bundle.json" \
  --controller-snapshot-bundle \
    "$EXACT_RUN/controller_snapshot_bundle.json" \
  --controller-state-bundle \
    "$EXACT_RUN/controller_state_bundle.json" \
  --factual-conformance-report \
    "$EXACT_RUN/exact_episode_conformance_bundle.json"
```

dynamic reconfigure は gain だけでなく、各 axis の
`p/i/d, limit_sum, limit_p/i/d, limit_err_p/i/d` という完全な 6×10
`pid_config` event として、その Metadata が有効になる最初の tick に
materialize します。static option、nominal model、geometry、rate の途中変更
は episode 分割なしには拒否します。`nominal_geometry_sha256` は
`{"schema":"grape.gimbalrotor-allocation-geometry/v1","geometry":...}`
の canonical content hash と一致しなければなりません。conformance の
command timestamp tolerance は 0、continuous RMSE / maximum error threshold
はそれぞれ frozen `0.01 / 0.03`、event agreement は `1.0` です。
passing conformance fixture の recorded event mask は、同じ fixture、snapshot、
source bag に hash-bound された場合だけ closed-loop likelihood へ接続されます。
全 factual timestamp の完全一致を確認した後、score interval 内の controller
tick だけを saturation、mode transition、reset/other event の離散 likelihood
として評価します。pre-roll は controller state 再構成専用で、failure 後の
event は削除せず censored count として残します。caller が別の event mask を
差し替えることはできません。

## Recorded-command open-loop plant posterior

新しい production CLI は `config/plant_assimilation.yaml` の role を使います。
failure bag 4 を inference、bag 3 を held-out failure validation、bag 7/8 を
success validation とし、success bag を初期 posterior update へ混ぜません。

まず bag hash、区間、command、観測、trajectory smoother を確認できます。

```bash
GRAPE_BAG_ROOT=/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering

rosrun grape_param_estim estimate_grape_plant.py \
  --config "$(rospack find grape_param_estim)/config/plant_assimilation.yaml" \
  --bag-root "$GRAPE_BAG_ROOT" \
  --output-root /tmp/grape_plant_runs \
  --prepare-only
```

posterior と artifact を生成する場合は `--prepare-only` を外します。artifact
run は source commit を provenance に固定するため clean Git worktree を
要求し、既存 run directory を上書きしません。

```bash
rosrun grape_param_estim estimate_grape_plant.py \
  --config "$(rospack find grape_param_estim)/config/plant_assimilation.yaml" \
  --bag-root "$GRAPE_BAG_ROOT" \
  --output-root /tmp/grape_plant_runs \
  --run-id open_loop_effective_v1
```

一つの run は次の完全な bundle を持ちます。

| artifact | 内容 |
|---|---|
| `run_manifest.json` | 全 payload hash/size、source commit、bag/episode/controller/plant/prior/likelihood/seed provenance |
| `controller_snapshot.json` | controller-owned frozen snapshot。open-loop current run では unavailable/not-used を明記 |
| `controller_replay_audit.json` | factual replay sufficiency と blocking fields |
| `factual_replay_report.json` | conformance gate。current open-loop run は exact 未接続を明記 |
| `posterior_particles.npz` | 全 weighted particle、weight、log likelihood、raw/derived quantity、model ID の正本 |
| `posterior_summary.json` | joint covariance/correlation、ESS、multimodality と解釈 |
| `posterior_hpd95.csv` | joint 95% highest-posterior-mass particle subset |
| `identifiability_report.json` | rank、gauge/null direction、prior/bound dominance |
| `likelihood_components.csv` | episode・particle ごとの residual component |
| `posterior_predictive.npz` | weighted trajectory/failure predictive arrays |
| `failure_validation.json` | failure 時刻まで censor した held-out trajectory coverage と、別成分の occurrence/type/time validation |
| `success_validation.json` | success coverage と false-failure gate |
| `REPORT.md` | human-review summary と recommendation gate |

`posterior_particles.npz` が完全 posterior の正本です。
`PlantPosteriorSummary.msg` は可視化・transport 用 summary であり、粒子集合を
置き換えません。

episode nuisance は initial plant/actuator state に加え、明示した
`effective_constant_acceleration_disturbance_v1` の world-frame linear
acceleration と body-frame angular acceleration を持ちます。各 episode の
bounded prior から seed 固定で sample し、inference failure では particle
likelihood に条件づけた nuisance posterior weight、held-out episode では prior
weight で posterior predictive を作ります。sample、weight、disturbance model/
vector は fixture identity、likelihood row、posterior predictive NPZ に保存
されます。

trajectory likelihood と failure/controller-event likelihood は別 total として
`likelihood_components.csv` に出力されます。current recorded-command bag は
ReplayFrame event evidence を持たないため、open-loop では event component を
明示的に `not_scored_no_evidence` とし、空 event を捏造しません。future exact
closed-loop config は hash-bound controller event evidence を必須にします。
`identifiability_report.json` は全 inference-failure episode と全 nuisance sample
の重み付き finite-difference Jacobian を使い、global rank/gauge だけでなく、
episode ごとの singular value、parameter-direction coefficient、null direction
と nuisance ID/weight を報告します。

## Posterior controller evaluation

controller tuning は plant assimilation と別 API/bundle です。
`ControllerCandidate` は versioned allowlist で controller-only field だけを
受け入れ、plant/actuator parameter 名を再帰的に拒否します。
`evaluate_and_write_controller_candidate` は完全な `PlantPosterior` の各 weighted
particle を評価し、success/failure/saturation probability と trajectory /
trajectory-tube evidence を保持します。最終 evaluation では全 particle に
finite trajectory、target-tube measurement、明示的な saturation measurement
が必須で、bool-only outcome や既定 `saturated=false` は拒否されます。各出力は
candidate、full posterior、evaluator identity、particle index と content hash
で結ばれます。recommendation threshold は `(0, 1]` の明示値が必須です。

recommendation は hash-bound evidence から導出した exactness、actuator
calibration、support、probability calibration、held-out failure、held-out
success の六 gate が全て通り、weighted success probability が threshold 以上の
場合だけ許可されます。v3 binding は candidate、完全 plant posterior、
exact controller artifact、particle evaluator/config、actuator
model/backend/calibration、support reference、probability-calibration dataset、
held-out failure/success dataset、および六つの canonical report hash を一つの
content hash に固定します。binding のない raw callable、別 candidate/posterior/
evaluator/data からの evidence 再利用、または publication provenance の
controller artifact/config 不一致は fail closed です。evaluator artifact hash は
実行する callable の bytecode/source/default/closure state と、参照する
global/module/class attribute から測定し、実行時に再検証します。support、
failure、success の evidence identity は caller label ではなく、実際の support
reference / held-out observations と計算済み gate result を不可分に hash した
各 typed report から導出します。さらに promotion
には strict `verify_run_manifest` を再実行できる元の 13-file plant bundle が必須で、
その manifest、posterior NPZ、artifact provenance も同じ binding に含まれます。
直接渡した boolean でも gate を通せません。production writer は evaluation
context hash を独立に再計算し、
出力先の publication は concurrent writer が存在しても置換しません。出力は既存の
13-file plant bundleを変更せず、`controller_evaluation.json`、
`particle_outcomes.json`、`artifact_manifest.json` の独立した atomic /
non-overwriting bundle になります。

## 将来収録する replay debug message

future bag では controller wrapper から次を publish します。debug publisher
は制御計算へ feedback せず、必要なら compile option または ROS parameter
で有効化します。

| 推奨 topic | type | publish timing |
|---|---|---|
| `/gimbalrotor/controller_replay/metadata` | `grape_param_estim/GimbalrotorControllerReplayMetadata` | 起動時と controller model/config 変更時 |
| `/gimbalrotor/controller_replay/frame` | `grape_param_estim/GimbalrotorControllerReplayFrame` | controller tick ごと |
| `/analysis/grape/plant_posterior_summary` | `grape_param_estim/PlantPosteriorSummary` | offline run の summary materialization 時だけ |

metadata は source/artifact/model/geometry/config hash と controller nominal
mass/CoG/inertia、gain、limit、static option を保持します。frame は exact
controller input、state before/after、PID term、vectoring force、
`FourAxisCommand`/gimbal target、allocation matrix、mode/reset/saturation
event を一 tick にまとめます。metadata は `base_thrust` と allocation
array の明示的な motor index order も保持します。current bag にこれらの
message は存在せず、
checkout の URDF や default parameter で暗黙に補完しません。

live publisher は既定で無効です。`/gimbalrotor/controller_replay/enabled`
を `true` にする場合は、同じ namespace に `source_commit`、`backend_id`
（`fidelity: pc_exact`）、`controller_artifact_sha256`、
`controller_snapshot_sha256`、`nominal_model_sha256`、
`parameter_dump_sha256`、`nominal_geometry_sha256`、`input_frame_id` を
明示してください。一つでも欠ける、または lowercase SHA-256 でない場合は
publish を拒否し、hash を live wrapper 内で生成しません。gain、limit、
controller option が変わった後は snapshot と parameter dump の両 hash を
更新するまで frame publish を停止します。

## Frozen inverse-dynamics baseline の検証

`config/inverse_dynamics_baseline.json` は従来結果の三つの run anchor と、
各 run の `artifact_manifest.json` が束ねる全 payload を固定します。

```bash
rosrun grape_param_estim verify_inverse_dynamics_baseline.py
```

combined hash は shell の current directory や repository path に依存する
`sha256sum` 出力を直接 hash しません。canonical preimage は
`grape_inverse_dynamics_baseline_payload_listing/v1` とし、episode ID と
`*_sha256` artifact key をそれぞれ辞書順に並べた次の UTF-8 line の連結です。

```text
episode_id<TAB>artifact_key<TAB>sha256<LF>
```

verifier は combined listing hash に加え、三つの anchor file、run-local
artifact manifest 内の全 file hash/size、run ID を実データに対して検証します。
manifest の `model_id=inverse_dynamics_baseline_v1` は Phase 0 の legacy
baseline family label です。既存 real-bag payload は変更せず、実際の
`summary.json` の model label
`low_dimensional_effective_response/v1` を
`frozen_payload_model_ids` として別に固定・検証します。

## Legacy Bayesian real-bag vertical slice

`config/counterfactual.yaml` は bag 4、7、8 の固定区間、topic、source bag
SHA-256、ENU/FLU/SI 規約、20 Hz の共通 pipeline、seed、27 candidate の
共通 grid を定義します。実行は clean な Git checkout と完全一致する
source commit を必須にし、dirty tree、bag hash 不一致、frame/unit
不一致を出力前に拒否します。

```bash
GRAPE_BAG_ROOT=/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering
GRAPE_SLICE_OUT=/tmp/grape_real_bag_slice

rosrun grape_param_estim analyze_grape_counterfactual.py \
  --config "$(rospack find grape_param_estim)/config/counterfactual.yaml" \
  --bag-root "$GRAPE_BAG_ROOT" \
  --output-root "$GRAPE_SLICE_OUT"
```

各 run directory には次のファイルを保存します。

| file | 内容 |
|---|---|
| `summary.json` | source/config/commit/input/trajectory hash、frame、diagnostic、全 hard gate |
| `trajectory.csv` | desired、diagnostic nominal、actual posterior mean/std、residual の時系列 |
| `trajectory_particles.npz` | coherent RTS actual sample と、同じ sample 初期状態から積分した nominal sample、ID、weight |
| `candidate_grid.csv` | 共通 27 candidate。exact oracle 不在時は確率欄を空にした未評価 grid |
| `REPORT.md` | RMS、effective-response posterior、識別性、gate の人向け要約 |
| `artifact_manifest.json` | 上記 payload の SHA-256 と byte size |

必要な場合だけ `--analysis-bag-root` を追加すると、元 bag を変更せず、
解析 message を record-time 順に merge した派生 bag と
`*.analysis.json` SHA/count sidecar を生成します。派生 bag は元 bag
全体を含むため、repository へ commit せず容量に余裕のある出力先を
指定してください。

```bash
GRAPE_ANALYSIS_BAG_ROOT=/tmp/grape_analysis_bags

rosrun grape_param_estim analyze_grape_counterfactual.py \
  --config "$(rospack find grape_param_estim)/config/counterfactual.yaml" \
  --bag-root "$GRAPE_BAG_ROOT" \
  --output-root "$GRAPE_SLICE_OUT" \
  --analysis-bag-root "$GRAPE_ANALYSIS_BAG_ROOT"
```

派生 bag の application topic は次のとおりです。

| topic | type | 内容 |
|---|---|---|
| `/analysis/grape_param_estim/trajectory/desired` | `TrajectoryParticleSet` | 記録済み controller target |
| `/analysis/grape_param_estim/trajectory/nominal` | `TrajectoryParticleSet` | sample ごとの診断用 local nominal。exact PC/MCU replay ではない |
| `/analysis/grape_param_estim/trajectory/actual_posterior` | `TrajectoryParticleSet` | offline EKF/RTS の coherent trajectory samples |
| `/analysis/grape_param_estim/model_mismatch` | `ModelMismatch` | matched sample の SE(3) log tracking/model residual、区間、covariance |

現在の実 bag 結果は
[`results/real_bag_vertical_slice/2026-07-24/INDEX.md`](results/real_bag_vertical_slice/2026-07-24/INDEX.md)
にあり、frozen backend 判定と post-hardening 再生成は
[`SELECTION_RESULTS.md`](SELECTION_RESULTS.md) にあります。exact PC/MCU
replay、bag-derived exact fixture、controller
integrator state、joint state/parameter inference、12-fold calibration が
未接続なので、全 run は `EXPERIMENTAL`、
`recommendation_available=false` です。candidate CSV は推奨ではなく
未評価 grid です。また per-time の
`world→{desired, nominal, actual}` ProbTF edge materialization は未接続で、
現時点の trajectory/mismatch 可視化契約は application message 側です。

## Legacy inverse-dynamics synthetic sanity check

まず、真値を既知にした synthetic bag で生成、推定、評価を一続きに確認します。

```bash
rosrun grape_param_estim generate_sanity_bag.py \
  --output-bag /tmp/grape_sanity_input.bag \
  --seed 7

rosrun grape_param_estim estimate_grape_bag.py \
  --input-bag /tmp/grape_sanity_input.bag \
  --output-bag /tmp/grape_sanity_analysis.bag \
  --config "$(rospack find grape_param_estim)/config/estimator.yaml" \
  --seed 7

rosrun grape_param_estim evaluate_sanity.py \
  --analysis-bag /tmp/grape_sanity_analysis.bag
```

generator が書く ground truth は評価専用です。estimator は ground-truth topic、URDF nominal parameter、未来の観測を読んではいけません。評価は推定終了後に別 process で行います。

この三つの legacy CLI と入力 config の再現性は
`config/synthetic_sanity_baseline.json` に固定しています。次の verifier
は path だけを placeholder へ正規化し、generator summary、estimator
summary、最終 evaluator report の SHA-256 を再計算します。

```bash
rosrun grape_param_estim verify_synthetic_sanity_baseline.py
```

固定 run は CI 用の 6 秒・32 particle・`--report-only` reproducibility
smoke です。既定 threshold での parameter recovery 合否を緩めるものではなく、
上の full command と Phase 4 calibrated recovery test は別の acceptance
gate です。

この sanity bag では、1 cm / 1 deg（接空間）の mocap noise を加えた pose と、既知の剛体運動から生成した校正済み actuator wrench を別 topic に記録します。後者を使うことで mass と共通 thrust scale の gauge をいったん切り離し、10 次元の剛体慣性 parameter を broad bounded-uniform 初期粒子から回収できるかだけを検証します。推定器が読む topic 一覧に ground truth は含まれません。

## Legacy inverse-dynamics real-bag estimator

主要 topic がそろう短い bag での smoke test 例です。

```bash
INPUT=/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/20260612_grape_hovering_7_2026-06-12-17-41-34.bag

rosrun grape_param_estim estimate_grape_bag.py \
  --input-bag "$INPUT" \
  --output-bag /tmp/grape_hovering_7_analysis.bag \
  --config "$(rospack find grape_param_estim)/config/estimator.yaml" \
  --seed 7 \
  --start-offset 45 \
  --duration 8

rosbag info /tmp/grape_hovering_7_analysis.bag
```

同じ処理は launch file からも起動できます。

```bash
roslaunch grape_param_estim offline_estimator.launch \
  input_bag:="/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/20260612_grape_hovering_7_2026-06-12-17-41-34.bag" \
  output_bag:=/tmp/grape_hovering_7_analysis.bag \
  start_offset:=45 \
  duration:=8
```

HOVER へ遷移する前に失敗した bag 4 も、同じ launch で解析できます。

```bash
roslaunch grape_param_estim offline_estimator.launch \
  input_bag:="/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/20260612_grape_hovering_4_2026-06-12-17-33-59.bag" \
  output_bag:=/tmp/grape_hovering_failure.bag \
  start_offset:=18 \
  duration:=7
```

この区間は `TAKEOFF_STATE=3` のため、bag 冒頭 3 秒の mocap 高さを床面基準とし、
そこから `0.05 m` 以上上昇した sample だけを剛体 likelihood に使います。
地上での thrust ramp は除外されます。command から復元した wrench は校正済み
実推力ではないため、結果は引き続き `command_as_force_effective` な有効
parameter として解釈してください。

既定では mocap 運動学を 50 Hz で生成した後、5 sample ごとの 10 Hz を
尤度の evidence として使い、その各 sample で particle filter を更新して
posterior と診断値を出力します。したがって、候補 sample と診断値は 10 Hz、
gate を通った区間の posterior も原則 10 Hz です。gate された sample は
posterior を更新せず、理由を持つ診断だけを残します。50 Hz の全点を独立な
evidence とみなさないのは、隣接点の微分値が重なった Savitzky--Golay window
から生成され、強く相関するためです。
`estimation_stride` を小さくすると message 数は増えますが、相関をモデル化せずに
尤度を重複計上して過信を招くため、既定値では行いません。

入力 bag は変更しません。出力先には別 path を指定してください。`--start-offset` と `--duration` は推定に使う区間だけを選び、元 bag の message は区間外も含めて出力へ保存します。推定器は original message の型・内容・record timestamp と connection metadata（`/tf_static` の latch を含む）を保ったまま、解析 message と record-time 順に merge して新しい bag を作ります。Header がある sensor はその event time、Header がない `/gimbalrotor/four_axes/command` などは bag record time を使います。

## 解析 bag の再生

端末 1 で ROS master を起動します。

```bash
roscore
```

端末 2 で解析 bag を再生します。

```bash
source /home/leus/catkin_ws/devel/setup.bash
rosbag play --clock /tmp/grape_hovering_7_analysis.bag
```

端末 3 では、例えば posterior summary を確認できます。

```bash
source /home/leus/catkin_ws/devel/setup.bash
rostopic echo /grape_param_estim/estimate
```

## 出力 topic

| topic | type | 内容 |
|---|---|---|
| `/grape_param_estim/estimate` | `grape_param_estim/InertialParameterEstimate` | parameter 名、posterior mean/MAP/95% 区間/covariance、ESS、尤度、provenance |
| `/grape_param_estim/particles` | `grape_param_estim/ParameterParticleSet` | weighted systematic decimation した等重み particle。完全な全粒子 dump ではない。`values` は particle-major の row-major 配列 |
| `/grape_param_estim/diagnostics` | `grape_param_estim/EstimatorDiagnostics` | resampling、NIS、force/torque residual、MCMC、励起 rank、gate 理由 |
| `/grape_param_estim/predicted_wrench` | `geometry_msgs/WrenchStamped` | posterior mean による actuator wrench 予測 |
| `/grape_param_estim/wrench_residual` | `geometry_msgs/WrenchStamped` | 観測 wrench - posterior mean 予測 |
| `/probtf/grape_param_estim/cog` | `probtf_msgs/ProbabilisticTransformStamped` | parameter 粒子から誘導した `fc`→推定 CoG の moment summary。mass/inertia 自体は Prob-TF edge に格納しない |
| `/grape_param_estim/ground_truth` | `grape_param_estim/InertialParameterEstimate` | synthetic generator が評価専用に書く真値。実 bag には存在しない |

`covariance` は `parameter_names` と同じ順番の正方行列を row-major で格納します。`ParameterParticleSet.particle_count` は出力に残した particle 数、`stride` は一 particle あたりの値数です。

## Foxglove Studio

Foxglove Studio の **Open local file** から解析 bag を直接開けます。ROS bridge や `rosbag play` は不要です。

確認項目は次のとおりです。

- Raw Messages で `parameter_names` と配列 index の対応を確認する。
- Plot で `mean`、`lower_95`、`upper_95` を同じ parameter index について重ねる。
- `effective_sample_size`、`ess_before`、`ess_after` と `resampled` を並べ、particle collapse の有無を見る。
- force/torque residual、NIS、`excitation_rank`、`gate_reason` を見て、rank 不足を伴う更新と gate された区間を区別する。
- particle snapshot を使い、平均と covariance だけでは見えない多峰性や境界への集中を確認する。

## 解釈上の制約

- 「事前情報を使わない」は、URDF nominal 値を中心にした informative prior を使わないという意味です。無限範囲の一様分布は proper distribution にならないため、bounded uniform の上下限と物理制約は必要です。
- Synthetic sanity check でも ground truth を estimator の初期値、proposal、gate、入力 topic に使いません。真値は生成と事後評価だけに使います。
- `/gimbalrotor/four_axes/command` は nominal controller が出した目標推力であり、実推力 sensor ではありません。質量と共通 thrust scale は同時に観測できない場合があります。
- `/gimbalrotor/uav/cog/odom` は nominal CoG を用いた派生量であり、CoG を推定するときの独立観測として扱いません。
- bag の `/tf_static` にある `motor_arm*` と現 checkout の `rotor_arm*` には版差があります。現段階の command-to-wrench 経路は `plant.geometry_profile` の ID/content hash を config と全 artifact provenance に固定し、`ASSUMED_SOURCE_NOT_BAG_VERIFIED` と明記します。current bag に hash-bound URDF/geometry snapshot がないため物理 geometry の一致は主張せず、将来 bag では replay metadata または episode TF 由来の profile へ置き換えます。
- zero covariance は zero noise ではなく、未設定の場合があります。設定した noise floor と使用 policy を解析結果へ残します。
- hovering data だけでは full inertia、CoG、actuator scale が十分に励起される保証はありません。励起 rank が不足する parameter は nominal 値へ強制せず、未観測または prior/bounds 支配として報告します。
- 収録済み hovering bag の command を N とみなす単純釣り合いでは見かけ質量が概ね 2.85--2.95 kg となり、q=0 URDF composite の約 2.3516 kg とは一致しません。実 bag で nominal に強制収束させることはせず、出力を effective parameter として扱います。
- 実 bag 上で全慣性 parameter が URDF 値へ一致することは、この sanity check だけでは保証しません。episode を分けた held-out validation と actuator calibration が必要です。
