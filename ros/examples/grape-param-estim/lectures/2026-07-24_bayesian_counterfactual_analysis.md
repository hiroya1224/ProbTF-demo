# 2026-07-24: Grape Bayesian counterfactual analysis 実装報告

## 結論

失敗・成功 rosbag から desired / nominal / actual の確率的な差を抽出し、
effective response posterior と closed-loop counterfactual の安全 gate へ渡す
オフライン解析基盤を実装した。bag 4、7、8 には共通 pipeline を適用し、
content-addressed な trajectory diagnostic artifact と派生 analysis bag を
生成できる。

ただし、deployed PC+MCU controller の exact replay oracle と 12-fold held-out
calibration は成立していない。したがって、実 bag の candidate grid は未評価で、
全 run は `EXPERIMENTAL`、`recommendation_available=false` である。現在の成果は
model-mismatch diagnosis であり、次回 flight parameter の確率的推奨ではない。

TODO の削除は未達 gate を完了扱いにするものではない。実装結果と未達の検証条件を
この報告へ移し、fail-closed の状態を固定する。TODO/doc 移行直前の repository
HEAD は `70b551778383639a41da8924daf060c5e53ad20c`、Grape の実装・安全 gate
を real-bag run に結び付けた source commit は
`9b83e4de3eca38e59a65957c2387a5d2c6750bdc` である。

## 実装した解析 pipeline

### manifest、event time、episode

[`../config/bag_manifest.yaml`](../config/bag_manifest.yaml) に 12 bag の
SHA-256、record/event interval、topic/type/count、clock namespace、既知 label、
復元値・仮定値・`UNKNOWN` を保存した。

`episode.py` と `manifest.py` は mocap、IMU、controller debug、four-axis command、
PWM/ESC、gimbal、gain、flight state を event time で扱う。Header time と bag
record time の採用規則、clock offset/drift、mode、mocap gap、timestamp jump、
sensor/actuator 欠測、frame/unit/motor order を明示する。normalized input は
source bag、topic、absolute interval、config、array content の hash に結び付け、
入力 bag を変更しない。

Savitzky–Golay などの pose 微分は初期診断・可視化だけに残し、mocap 由来の
acceleration を独立 observation として主 likelihood に二重投入しない。

### actual trajectory posterior

`state_smoother.py` に共通 observation/posterior contract と
error-state EKF + RTS smoother を実装した。state は pose、body velocity、
angular velocity、IMU bias を持ち、mocap/IMU の timestamp と mask を使う。
retrospective RTS trajectory sample は同じ `sample_id` を全時刻で保持する。
causal prefix mode は filter output だけを使い、未来 evidence を拒否する。

`alternative_backends.py` には IMU preintegration を使う batch/factor-graph
vertical slice を同じ比較境界で実装した。これは候補実装であり、held-out
比較による `OPTIONAL`/`DEFAULT` 昇格はしていない。

### controller replay

`controller_replay.py` に `teacher_forced` と `free_run`、PID state、
saturation、anti-windup、mode reset、allocation/thrust scale、delay を扱う
vectorized Python surrogate を実装した。candidate gain では記録済み command
を固定せず、controller state と command を最初から閉ループ再計算する。

exact backend には subprocess/ctypes oracle、binary/source/config identity、
bag-derived fixture、conformance report の契約を設けた。fixture は source bag
SHA、topic、time/frame/unit/motor order、config/request/array content hash へ
結び付け、利用時にも再検証する。capability と `is_exact` は built-in `bool`
だけを受け付ける。

この契約を通る実 oracle は現在の workspace/bag から構成できなかった。そのため
Python nominal trajectory は明示的な diagnostic approximation であり、exact
counterfactual の代替とは扱わない。

### effective response と inference

`effective_response.py` に、軸別 effectiveness、cross-coupling、delay、
一次遅れ、damping、bias を持つ低次元 response model を実装した。trajectory
sample を point estimate に潰さず mixture likelihood で marginalize し、
heavy-tail residual、episode random effect、posterior correlation/rank による
非識別診断を返す。

`inference.py` は bounded transform/prior、tempered resample-move SMC、ESS、
systematic resampling、rejuvenation、chain diagnostics、predictive interval
coverage を提供する。比較候補として particle marginal Metropolis-Hastings、
structured 6-DoF mechanics、conditional backend gate も vertical slice として
実装した。これらの存在は held-out superiority を意味せず、全候補の状態は
選定 artifact に従う。

### counterfactual と safety gate

`counterfactual.py` に target trajectory/tube、joint posterior sample、
closed-loop rollout、support distance、importance-weight ESS、credible interval、
constraint violation、connected candidate region を実装した。

`CounterfactualResult` は次が同時に満たされない限り recommendation を返さない。

- exact controller backend と frozen replay metric
- bag-derived exact fixture と identity/conformance provenance
- target-tube probability calibration
- support と ESS
- joint state/parameter dependence
- controller integrator/internal state の復元または推定

gate 値、workflow status、run/content hash、candidate/support/credible bound の
数値整合を再検証し、metadata は deep-freeze する。現在は全 recommendation gate
が false なので `CounterfactualCandidate` ROS message も生成しない。

artifact の content hash は再現性・改変検出用であり、外部署名や暗号学的な
実行証明ではない。信頼境界には evaluator process と human review が残る。

## application message と artifact

次の ROS message を追加した。

- `TrajectoryParticleSet.msg`
- `ModelMismatch.msg`
- `CounterfactualCandidate.msg`

派生 analysis bag には次を materialize できる。

| topic | 内容 |
|---|---|
| `/analysis/grape_param_estim/trajectory/desired` | 記録された target |
| `/analysis/grape_param_estim/trajectory/nominal` | 同じ sample 初期状態から積分した diagnostic nominal |
| `/analysis/grape_param_estim/trajectory/actual_posterior` | coherent RTS trajectory samples |
| `/analysis/grape_param_estim/model_mismatch` | matched-sample SE(3) residual と covariance |

元 bag の message、record timestamp、connection metadata を保ち、解析 record を
record-time 順に別 bag へ merge する。sidecar には source/output SHA、record
count/type、run ID を保存する。bag 4 smoke では 3 個の
`TrajectoryParticleSet` と 151 個の `ModelMismatch`、計 154 analysis record を
確認し、生成前後で source bag SHA が不変であることを確認した。

run directory は `summary.json`、`trajectory.csv`、
`trajectory_particles.npz`、`candidate_grid.csv`、`REPORT.md`、
`artifact_manifest.json` を同じ run ID で保存する。CSV は LF に正規化し、
manifest は payload の SHA-256 と byte size、NPZ は trajectory evidence
preimage を保持する。

## 実 bag 4 / 7 / 8 の diagnostic result

共通設定は [`../config/counterfactual.yaml`](../config/counterfactual.yaml)、
canonical loaded-config SHA-256 は
`aebedd9b52b0702a050b7725e8197c5aa89908b3d8c3cabef2d5a1dd3d890faa`
である。20 Hz、trajectory sample 4、seed 7、共通 27-candidate
P/D/allocation grid を使った。grid の probability/credible interval は空欄で、
分類は `NOT_CLASSIFIED`、理由は `ORACLE_UNAVAILABLE` である。

| episode | offset window (s) | run ID | desired→actual position / attitude RMS | nominal→actual position / attitude RMS |
|---|---:|---|---:|---:|
| bag 4 | 18–26 | `b1b32ee30d43c05fe357` | `0.332293 m` / `0.244735 rad` | `2.815148 m` / `0.857021 rad` |
| bag 7 | 45–53 | `c63e4ebe570b6943f7dd` | `0.037049 m` / `0.068621 rad` | `0.210854 m` / `1.088083 rad` |
| bag 8 | 294–302 | `bcb0786294cad8f7f565` | `0.034606 m` / `0.054510 rad` | `0.495223 m` / `1.062087 rad` |

大きい nominal→actual error は、現在の nominal が exact deployed controller
replay ではないことも表す。この値から candidate の成功確率を推定してはいけない。
source bag/input slice/trajectory/file の全 hash は
[`../results/real_bag_vertical_slice/2026-07-24/INDEX.md`](../results/real_bag_vertical_slice/2026-07-24/INDEX.md)
に記録した。

## frozen selection

[`../config/selection_protocol.yaml`](../config/selection_protocol.yaml) は 12 bag の
leave-one-bag-out fold、stratum、seed、metric、hard gate、candidate を固定する。
baseline の machine result は
[`../config/selection_results.json`](../config/selection_results.json)
（result hash
`9b2725232b94cbd289243e31fca560e0e2b334698112f8610a45fed6596db059`）
である。

strict safety gate 後の Grape commit `9b83e4d` に対する再生成結果は
[`../config/selection_results_2026-07-24_post_hardening.json`](../config/selection_results_2026-07-24_post_hardening.json)
（result hash
`ef2d1837aa33b1d87c88dea5607f0c739f9b1a73fd81a97754f84c57c66bc52c`）
である。

両方とも observation は `0/12 folds`、`selection_complete=false` で、
13 candidate はすべて `EXPERIMENTAL`、selected default はない。real-bag
vertical slice は diagnostic evidence であり、held-out selection observation
として水増ししていない。詳細は
[`../SELECTION_RESULTS.md`](../SELECTION_RESULTS.md) にある。

## 検証

- ROS-aware test suite: `110/110` pass
- `catkin build grape_param_estim`: success
- selection baseline/post-hardening の self hash、全 candidate の
  `EXPERIMENTAL`、`0/12`、default 不在を再検証
- 3 run の INDEX、manifest、NPZ trajectory evidence、全 payload SHA を再計算
- 81 candidate row がすべて fail closed、6 CSV が CR なし・LF 終端
- bag 4 derived-bag smoke で型、絶対 record time、順序、sidecar SHA、
  source bag 不変性を確認
- `git diff --check`: pass

## exact oracle が成立しない具体的理由

- MCU 側の `libspinal_flight_controller.so` と `AttitudeController` 系譜はあるが、
  当時の PC model/controller を同じ identity で再現する library がない。
- PC 側には `kalman_filter`、geodesy/geographic message など不足 dependency が
  あり、現在 checkout の controller をそのまま oracle 化できない。
- counterfactual が必要とする deterministic step、state snapshot/restore、
  candidate injection の ABI がない。
- bag には controller integrator、内部 delay/filter/mixer state、flashed
  binary/source hash が完全には残っていない。
- 記録 gain/config と現在 source の対応を証明する immutable bag-derived
  factual fixture がない。

このため exact boundary と conformance gate は実装したが、exact replay の
empirical gate は通過していない。

## 昇格・実証待ち ledger

1. exact PC+MCU factual replay と P/D/allocation candidate の閉ループ grid、
   target-tube contour は未評価である。
2. V1–V7 の held-out 12-fold observation、50/80/95% calibration、
   Brier/log score、gain-sweep ranking は未収集である。
3. joint state/parameter dependence は trajectory-mixture による modular
   marginalization までで、完全な joint posterior ではない。
4. controller integrator/internal state は復元・latent inference されていない。
5. per-time の `world→desired/nominal/actual/counterfactual` ProbTF edge と
   `nominal→actual` transform topic は未 materialize である。現在は application
   trajectory/mismatch message が正本である。
6. bag 3/4/7/8 を含む operator outcome label の一部は一次資料による確定待ちである。
7. sparse GP/BayesSim/PMMH/factor graph/structured mechanics は比較候補または
   vertical slice であり、優位性を実証していない。
8. support 外 candidate は `UNSUPPORTED` のままで、成功確率を表示しない。
9. 実機 parameter の自動書換え、自動 flight、safe exploration は範囲外であり、
   human-reviewed proposal より先へ進めない。

利用方法、topic、artifact の詳細は [`../README.md`](../README.md) を参照する。
