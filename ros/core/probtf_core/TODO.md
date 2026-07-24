# ProbTF core TODO: model-based temporal evaluation

最終更新: 2026-07-24

この TODO は、Grape rosbag の議論から抽出された **ProbTF core に一般化できる仕事だけ**を扱う。目的は、現在
`UNSUPPORTED_TEMPORAL_POLICY` になる `INTERPOLATE_WITH_MODEL` と
`PREDICT_WITH_MODEL` を、因果性、不確実性、provenance を失わない core 機能として実装することである。
Grape 固有の controller、機体応答モデル、成功条件、PID 候補探索は
[`../../examples/grape-param-estim/TODO.md`](../../examples/grape-param-estim/TODO.md) 側に置く。

## 完成時に満たす契約

- 時刻 \(t\) の transform は、観測 sample の単純な穴埋めではなく、明示的に登録された時間モデルから評価する。
- interpolation、prediction、単なる sample 選択を診断情報から区別できる。
- prediction は query 時刻より未来の観測を読まず、最大予測 horizon と stale 制限を必須にする。
- process noise と初期分布を伝播し、horizon が延びたことによる uncertainty を隠さない。
- 同じ source uncertainty を再利用した場合、その依存関係を独立と誤認して二重加算しない。
- static uncertain edge は時刻によらず同じ分布であり、時間発展させない。
- 既存の `EXACT`、`NEAREST_WITHIN_TOLERANCE`、`LATEST`、`LATEST_COMMON` の挙動を変えない。

core が提供するのは「時間モデルを評価する共通契約」であって、万能な運動モデルではない。モデルを登録して
いない edge に対して暗黙の外挿は行わず、fail closed とする。

## 候補の状態と選び方

各候補には次の状態だけを用いる。実験前の `DEFAULT_CANDIDATE` は、正式な default を意味しない。

- `DEFAULT_CANDIDATE`: 最初に推奨する候補。Phase 2 を通過したら `DEFAULT` にする。
- `OPTIONAL_CANDIDATE`: 同じ API の後ろに実装し、同一条件で比較する候補。
- `OPTIONAL`: 特定の利用領域で明確な利点があるため残す。
- `PRUNE`: correctness gate に失敗したか、全評価領域で他候補に支配されているため公開選択肢から外す。
- `EXPERIMENTAL`: 結果が不十分で default にはできないが、追加データで再判定する価値がある。

閾値、評価データ、seed、比較 commit は、結果を見る前に `test/temporal_selection.yaml` に固定する。
結果を見て閾値を変更した場合は新しい評価 run として扱い、旧 run を上書きしない。

---

## Phase 1: 候補を実装する TODO

### C0. 現行仕様を固定する

- [ ] 現行 temporal policy の成功・失敗・error code を characterization test にする。
- [ ] static edge、同一 stamp、authority conflict、parent change、path-level `LATEST_COMMON` を回帰 test に含める。
- [ ] model-based policy が未登録モデルに対して明示的に失敗する test を先に追加する。
- [ ] benchmark 前に CPU、Python/C++ version、依存 package、乱数 seed を記録する共通 harness を作る。

### C1. 時間モデルの公開契約を定義する

- [ ] `TemporalEvaluationRequest` を追加する。少なくとも requested stamp、policy、利用可能な anchor、
  model selector、最大 horizon、乱数 stream/seed、query mode を持たせる。
- [ ] `TemporalEvaluationResult` を追加する。transform distribution に加え、source stamp 群、model ID/version、
  interpolation/prediction の別、horizon、dependency ID、近似、警告を返す。
- [ ] `TemporalModel` protocol/ABC を追加する。
  - 対応する distribution kind と必要な履歴数
  - `interpolate(left, right, request)`
  - `predict(history_at_or_before_t, request)`
  - 有効 support、最大 horizon、process-noise 単位
  - model/config fingerprint
- [ ] process noise の正規契約は連続時間 spectral density \(Q_c\) とする。
  sampling rate に依存する離散 \(Q_d\) は明示的な compatibility adapter のみにする。
- [ ] model instance は edge と authority に明示的に bind する。query 側の override は model ID を必須とし、
  結果の provenance に残す。名前なしの global default model は作らない。

### C2. buffer/query に因果的な評価を実装する

- [ ] `INTERPOLATE_WITH_MODEL` は query stamp を挟む二つの sample がある場合だけ呼ぶ。
- [ ] `PREDICT_WITH_MODEL` に渡す履歴は requested stamp 以下に制限する。
- [ ] `max_prediction_horizon`、`max_age`、model support のどれかを越えたら error にする。
- [ ] path query では各 edge の evaluation stamp、horizon、model を保持し、path 全体の診断に集約する。
- [ ] `LATEST_COMMON` と model evaluation を組み合わせる場合の anchor stamp を一意に定義する。
- [ ] query 中に buffer が更新されても、一つの snapshot 内で source record が変化しないようにする。
- [ ] static edge は requested stamp へ同じ分布を返し、process noise を加えない。

### C3. uncertainty 伝播 backend を同一 API で実装する

次の二つは非線形・非 Gaussian 領域で優劣が自明でないため、切替可能にして Phase 2 で選ぶ。

| Backend | Phase 1 の位置づけ | 用途 |
|---|---|---|
| tangent-space moment propagation | `DEFAULT_CANDIDATE` | online query 向けの高速な近似 |
| sample-wise propagation | `OPTIONAL_CANDIDATE` かつ参照 oracle | mixture、強い非線形、共有 latent の保持 |

- [ ] backend selector を public option にせず、まず model/config の明示フィールドとして実装する。
- [ ] moment backend は平均 pose、接空間 covariance、適用した線形化/近似を返す。
- [ ] sample backend は同じ `sample_id` を path と時系列で再利用し、重みを保持する。
- [ ] sample backend の固定 seed で bitwise または許容誤差内の再現性を保証する。
- [ ] moment backend が共有依存性を表現できない場合、独立近似を黙って行わず、
  `DEPENDENCE_APPROXIMATED` を付けるか query を拒否する。

### C4. reference temporal model を実装する

| Model | Phase 1 の位置づけ | 備考 |
|---|---|---|
| SE(3) constant-body-twist + \(Q_c\) | `DEFAULT_CANDIDATE` | 最小履歴で使え、基準モデルにしやすい |
| SE(3) constant-acceleration + \(Q_c\) | `OPTIONAL_CANDIDATE` | IMU や十分な履歴がある短時間予測向け |
| endpoint-conditioned sample interpolation | `OPTIONAL_CANDIDATE` | 非 Gaussian 分布の二端点補間用 |

- [ ] translation と quaternion を別々に外挿せず、明記した SE(3) convention で twist を扱う。
- [ ] quaternion sign、\(\pi\) 近傍、zero-motion、非常に短い \(\Delta t\) を test する。
- [ ] constant-acceleration model は acceleration source と frame を必須 metadata にし、欠落時は
  constant-twist へ暗黙 fallback しない。
- [ ] interpolation は両端点へ厳密に一致し、prediction とは別 diagnostic code を返す。
- [ ] Dirac transform 用の deterministic interpolation は conformance baseline として残すが、
  stochastic edge の uncertainty をゼロにする fallback には使わない。

### C5. 依存関係と provenance を保持する

- [ ] source record ごとに安定した `dependency_id` を持たせる。
- [ ] inverse、compose、同一 edge 再利用、時間予測後の compose に dependency 情報を伝播する。
- [ ] sample backend では共通乱数を再利用する。
- [ ] moment backend では表現可能な cross-covariance を利用し、表現不能なら近似を診断する。
- [ ] model ID/version、config hash、source stamps、authority、seed/stream、backend を結果から追跡可能にする。
- [ ] future sample を利用した smoothing/interpolation は `offline_smoothing` と明記された別 mode だけに許し、
  online prediction と同じ provenance を名乗らせない。

### C6. 診断と安全制限を公開する

- [ ] 少なくとも `MODEL_NOT_REGISTERED`、`MODEL_SUPPORT_EXCEEDED`、
  `PREDICTION_HORIZON_EXCEEDED`、`INSUFFICIENT_HISTORY`、`DEPENDENCE_APPROXIMATED`、
  `NON_CAUSAL_INPUT_REJECTED` を追加する。
- [ ] requested/evaluated/source stamp、horizon、uncertainty 増加量を query diagnostic に出す。
- [ ] uncertainty が model の上限を越えた場合は数値を返し続けず、設定に従って reject または
  `DEGRADED` とする。
- [ ] ROS bridge が新しい診断を落とさず、従来 consumer には後方互換な failure を返すようにする。

### C7. conformance test と benchmark corpus を作る

- [ ] synthetic: 静止、constant twist、constant acceleration、急な mode change、timestamp jitter/dropout。
- [ ] distribution: Dirac、局所 Gaussian、bimodal orientation、強い translation-rotation coupling。
- [ ] invariants: endpoint 一致、zero-\(Q_c\) の deterministic limit、正の \(Q_c\) で horizon と共に
  uncertainty が減らないこと、frame/inverse consistency。
- [ ] causality: query より未来の record を挿入しても online prediction 結果が変わらないこと。
- [ ] Monte Carlo oracle と moment result の平均、covariance、energy distance を比較する。
- [ ] orientation demo と Grape から匿名化・固定した小さな fixture を作る。
  core test は rosbag 本体や Grape package に依存させない。
- [ ] p50/p95 latency、peak memory、sample 数に対する scaling を保存する。

### C8. API 文書と移行例を追加する

- [ ] 「sample selection」「model interpolation」「causal prediction」「offline smoothing」の違いを説明する。
- [ ] model registration、process noise、最大 horizon、failure handling の最小例を追加する。
- [ ] 標準 TF/TF2 へ落とすと分布、model provenance、予測 horizon が失われることを明記する。
- [ ] Grape integration は application plugin の例として示し、Grape dynamics を core に import しない。

---

## Phase 2: 比較して default / optional / prune を決める TODO

### S0. 評価を凍結する

- [ ] `test/temporal_selection.yaml` に corpus hash、split、seed 群、metric、初期閾値を保存する。
- [ ] 全候補を同じ input snapshot、乱数 sample、hardware 条件で実行する。
- [ ] 1 回の平均だけでなく、episode/seed 単位 bootstrap 95% confidence interval を出す。
- [ ] correctness gate に失敗した run は性能比較へ進めない。

### S1. hard correctness gate

以下は一つでも失敗した候補を `PRUNE` または修正後の再評価にする。

- [ ] online prediction が未来の sample に依存しない。
- [ ] interpolation endpoint、frame、inverse、static edge の invariant を満たす。
- [ ] nominal 95% interval の empirical coverage について、binomial 95% CI が 0.95 を含む。
- [ ] NaN、非正規化 quaternion、非 PSD covariance、無限 weight を返さない。
- [ ] horizon/support 超過を成功扱いにしない。
- [ ] provenance から全 source record、model/config、近似の有無を復元できる。
- [ ] 現行 temporal policy の回帰 test を全て通す。

### S2. moment backend と sample backend を選ぶ

- [ ] Monte Carlo oracle に対する pose error、covariance relative error、energy distance、coverage を比較する。
- [ ] p50/p95 latency、peak memory、結果の再現性を比較する。
- [ ] **default 規則:** correctness gate を通った候補のうち、held-out score が最良候補の
  1 standard error 以内にある最も単純かつ高速なものを default にする。
- [ ] moment backend が calibration を保ち、sample backend より実用上十分速ければ moment を default にする。
- [ ] sample backend が bimodal/coupled corpus で held-out NLL または energy distance を
  10% 以上改善し、その bootstrap CI が 0 を跨がなければ optional に残す。
- [ ] sample backend に固有の改善がなく、runtime または memory が 2 倍以上なら公開 option から prune し、
  test oracle としてだけ残す。
- [ ] moment backend が非 Gaussian 領域で gate を満たさない場合、その distribution kind では自動拒否し、
  sample backend を default にする。

### S3. constant-twist と constant-acceleration を選ぶ

- [ ] 1-step と複数 horizon の held-out log predictive density、pose RMSE、coverage を比較する。
- [ ] 通常運動、IMU 利用可能、dropout、急運動を別 strata として報告する。
- [ ] constant-acceleration が二つ以上の独立 corpus で held-out score を 10% 以上改善し、
  calibration と安定性を悪化させなければ optional に残す。
- [ ] 両者が用途別に勝つなら constant-twist を全 edge の default、constant-acceleration を
  明示 opt-in とする。
- [ ] constant-acceleration が全 strata で支配されるか、IMU noise を積分して不安定になるなら prune する。
- [ ] 単一の最良モデルへ暗黙切替する auto-selector は、選択誤り自体の uncertainty を表せるまでは実装しない。

### S4. process-noise 表現を確認する

- [ ] 同一連続軌道を異なる sampling rate で評価し、連続時間 \(Q_c\) の結果が許容誤差内で一致することを確認する。
- [ ] 離散 compatibility adapter が sampling rate 依存の結果を生む場合、core public API から prune する。
- [ ] uncertainty growth が実データの held-out coverage を満たさなければ model ごとに再同定し、
  core 共通の「都合のよい inflation」を default にしない。

### S5. core へ残す API の最終判定

- [ ] 結果を `SELECTION_RESULTS.md` に、commit、config hash、corpus hash、表、採否理由付きで記録する。
- [ ] default を package config と API docs の両方で一意にする。
- [ ] `OPTIONAL` は優位になる利用条件と fallback 条件を文書化する。
- [ ] `PRUNE` は実装を削除するか、公開 registry から外す。比較用 fixture と結果は残す。
- [ ] 判定不能は `EXPERIMENTAL` とし、default にせず、必要な追加データを明記する。
- [ ] Grape 側の `TrajectoryParticleSet` を core に昇格するか再判定する。
  **独立した二つ目の application が同じ時系列 sample/weight/provenance 契約を必要とした場合だけ**
  core message/API 化し、それまでは Grape 側に置く。

## 共通の採否原則

1. correctness と calibration を速度より先に判定する。
2. hard gate 通過後は、held-out データでの一標準誤差則により、同等なら単純な候補を default にする。
3. 特定 strata で再現可能な改善がある候補は optional に残し、その適用条件を機械判定できる形にする。
4. 全 strata で品質が悪く、速度・memory・依存関係にも利点がない候補は prune する。
5. 統計的に判別できない候補を勝者扱いしない。`EXPERIMENTAL` のまま追加データを集める。

## この roadmap の完了条件

- [ ] model-based な二つの temporal policy が、未実装 error ではなく上記契約で動作する。
- [ ] backward compatibility、因果性、calibration、bounded prediction の gate が CI で継続実行される。
- [ ] Phase 2 の採否が再現可能な artifact として残り、default と optional の理由を追跡できる。
- [ ] application 固有モデルを core に混入させず、Grape から plugin として利用できる。

この TODO の範囲外にある Bingham 正規化、一般 factor graph、authority policy UI などの既存 backlog は
[`../../../docs/reports/v2-demo-migration_2026-07-13.md`](../../../docs/reports/v2-demo-migration_2026-07-13.md)
を参照する。
