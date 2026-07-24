# 2026-07-24: model-based temporal evaluation 実装報告

## 結論

Grape rosbag の議論から core に切り出した model-based temporal evaluation を、
明示的に opt-in する実験 API として実装した。補間、因果予測、単なる sample
選択、offline smoothing を診断上区別し、時間 horizon、不確実性、依存関係、
model/config/source provenance を query 結果に残す。

一方、凍結評価では全 hard gate を通過した uncertainty backend がない。そのため
package-wide な production `DEFAULT` は選定していない。TODO の削除は
「全候補が production ready になった」という意味ではなく、実装結果と未達の
昇格条件をこの報告へ移したことを意味する。

この報告が対象とする TODO/doc 移行直前の repository HEAD は
`70b551778383639a41da8924daf060c5e53ad20c` である。

## 実装した公開契約

`probtf.temporal` に次を追加した。

- `TemporalEvaluationRequest`
  - requested stamp、policy、anchor、model selector
  - query mode、最大 prediction horizon、最大 age
  - random seed/stream
- `TemporalEvaluationResult`
  - transform distribution、source stamp、evaluation kind、horizon
  - model ID/version、config fingerprint、backend、authority
  - dependency ID、近似、warning、diagnostic、uncertainty 増加量
- `TemporalModel`
  - support、必要履歴数、最大 horizon、model/config fingerprint
  - `interpolate()` と `predict()` の共通 interface
- `TemporalQueryMode`、`TemporalEvaluationKind`、
  `TemporalUncertaintyBackend`、`TemporalDiagnosticCode`

process noise の正規表現は連続時間 spectral density \(Q_c\) とした。legacy の
離散 \(Q_d\) は sample period を必須とする
`adapt_discrete_process_noise()` 経由だけで \(Q_c\) に変換し、adaptation
diagnostic を返す。

## graph と時間評価

`ProbTfGraph` は named temporal model を physical edge と authority の組へ
登録する。未登録 edge を暗黙に外挿せず、複数 binding に selector も local
default もなければ `MODEL_AMBIGUOUS` で fail closed する。binding が一つだけの
場合は selector 省略時にもその model を選ぶが、監査可能性のため application
からは `model_id` の明示を推奨する。`make_default=True` は graph instance
内の local binding であり、package-wide な production 推奨ではない。

時間 policy の意味は次のように固定した。

| policy / mode | 実装した規則 |
|---|---|
| sample selection | exact stamp では model を呼ばず、保存済み record を選ぶ |
| `INTERPOLATE_WITH_MODEL` | requested stamp を厳密に挟む同一 authority の二端点だけを使う |
| `PREDICT_WITH_MODEL` | requested stamp 以下の履歴だけを model へ渡す |
| offline smoothing | 未来 endpoint の利用を明示した別 mode として記録する |
| static edge | requested stamp へ同じ分布を retime し、process noise を加えない |

query は開始時の buffer snapshot を使う。prediction は query、model、anchor age
の三つの上限を満たす必要があり、horizon/support/staleness/uncertainty 超過を
成功値として返さない。`allow_degraded=True` を built-in `bool` で明示した場合
だけ degraded result を許し、path と component の両方へ warning を残す。
既存の `EXACT`、`NEAREST_WITHIN_TOLERANCE`、`LATEST`、`LATEST_COMMON` の
characterization test も維持した。

## reference model と uncertainty backend

同じ API の後ろに次を実装した。

- SE(3) constant-body-twist + \(Q_c\)
- acceleration source/frame metadata を必須とする
  SE(3) constant-body-acceleration + \(Q_c\)
- endpoint-conditioned sample interpolation
- tangent-space moment propagation
- sample-wise propagation

quaternion sign、\(\pi\) 近傍、zero motion、短い \(\Delta t\)、endpoint 一致、
zero-\(Q_c\) limit、正の \(Q_c\) による uncertainty growth を test した。
sample backend は固定 seed/stream、sample ID、weight、dependency ID を保持する。
moment backend が共有依存を厳密に表せない場合は
`DEPENDENCE_APPROXIMATED` を付ける。

source record の dependency ID は inverse、compose、時間評価、path aggregation
へ伝播する。model が返した requested/source stamp、evaluation kind、model ID、
config fingerprint、random stream/seed、approximation、uncertainty claim は graph
側でも再検証し、custom model が過小な uncertainty や不整合な provenance を
主張した場合は reject する。safety flag は Python truthy 値を許さず built-in
`bool` に限定した。

ROS v2 bridge は temporal detail を保持する。標準 TF/TF2 へ代表 pose だけを
射影すると、分布、dependency、model/config、source stamp、horizon、近似 warning
が失われるため、監査用 consumer は ProbTF message を正本とする。

## 凍結選定の結果

protocol、corpus、seed、metric、threshold は
[`../test/temporal_selection.yaml`](../test/temporal_selection.yaml) に固定した。
baseline artifact は
[`../test/temporal_selection_results_2026-07-24.json`](../test/temporal_selection_results_2026-07-24.json)
（SHA-256
`72e61c62e2f65dffc9d9ddacd678476796b61e21202be319b6894613bae97c22`）
である。

safety hardening と result-file self-reference の回帰修正後、同じ protocol を
clean tree から再実行した artifact は
[`../test/temporal_selection_results_2026-07-24_post_hardening.json`](../test/temporal_selection_results_2026-07-24_post_hardening.json)
である。

| post-hardening provenance | 値 |
|---|---|
| run HEAD | `9c326b3210d473783288d97320a6eddf557346fe` |
| artifact SHA-256 | `3410e1a223463a54b779e22fe6132963591c4be7e9758f37149a915b5663e2fd` |
| evaluated source SHA-256 | `e88468ecf32c586d4ba7a396a9b4c9fb5a054c62e7b0f1be6e359fbcb63ea123` |
| worktree | whole/core とも clean |
| conformance | `221 passed` |

accuracy、coverage、uncertainty metric、motion-model one-standard-error
判定、hard gate は baseline と数値まで一致した。主な不通過理由は次である。

- moment backend は bimodal orientation と強い translation-rotation coupling
  で nominal 95% coverage gate を通らなかった。
- sample backend は全 distribution stratum の coverage gate を通らず、
  factor-level joint-sample contract も未実装である。
- sample backend は energy distance を改善したが、moment twist より p50 で
  約 `22.213x` 遅く、correctness gate の失敗を補えない。
- constant acceleration は一つの synthetic corpus では優位だったが、
  `OPTIONAL` に必要な二つの独立 corpus がない。

最終 disposition は次のとおりである。

| candidate | disposition |
|---|---|
| tangent-space moment backend | `EXPERIMENTAL` |
| sample backend | `EXPERIMENTAL` |
| constant body twist | `EXPERIMENTAL` |
| constant body acceleration | `EXPERIMENTAL` |
| endpoint-conditioned sample interpolation | `EXPERIMENTAL` |
| discrete \(Q_d\) compatibility adapter | `OPTIONAL`（migration 専用） |
| automatic model selector | `PRUNE` |

詳細な数値と再実行方法は
[`../SELECTION_RESULTS.md`](../SELECTION_RESULTS.md) と
[`../docs/temporal_evaluation.md`](../docs/temporal_evaluation.md) に残した。

## 検証

- `tests/probtf`: `221 passed`
- `catkin build probtf_core`: success、warning/failure なし
- post-hardening artifact の file/source/config/runner/corpus hash を再計算し一致
- 同じ出力 path に既存 result JSON があっても evaluated source hash へ混入しない
  回帰 test を追加
- `git diff --check`: pass

## 昇格・実証待ち ledger

次は実装を隠す TODO ではなく、状態を `EXPERIMENTAL` から昇格させるための
evidence ledger とする。

1. 全 distribution stratum の calibration hard gate を通る backend がない。
2. 選定 corpus は一つの synthetic/anonymous corpus だけで、独立した Grape
   application corpus による再現がない。
3. sample backend は共有 calibration factor と独立 residual を factor level
   で分離する joint-sample contract を持たない。
4. moment backend の表現不能な cross-time dependence は診断付き近似であり、
   厳密な joint posterior ではない。
5. core の `TemporalModel` を使う Grape-specific application plugin/example は
   未実装であり、PC+MCU exact replay と組み合わせた calibration と per-time
   integration も未実証である。
6. 標準 TF2 だけを保存する経路は distribution/provenance を lossless に保持
   できない。
7. `TrajectoryParticleSet` は二つ目の独立 application が同じ契約を必要とする
   まで Grape application 側に置き、core message へ昇格させない。

したがって、production 利用では application corpus で calibration gate を
再実行し、明示的な model/backend binding と failure handling を設定する必要がある。
