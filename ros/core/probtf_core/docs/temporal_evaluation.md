# Model-based temporal evaluation

## 公開契約

model-based query は、履歴 sample の単なる穴埋めではない。呼び出しごとに
`TemporalEvaluationRequest` を構成し、`TemporalEvaluationResult` に評価済みの
分布と provenance を返す。

結果には少なくとも次が含まれる。

- requested stamp、source stamp 群、evaluation kind、horizon
- model ID/version、config fingerprint、authority、uncertainty backend
- source record の dependency ID 群
- random seed/stream、diagnostic code、warning
- uncertainty trace と、制限超過を許可した場合の degraded detail

graph は custom model の返す requested stamp、source、model、config、
evaluation kind が request と整合するかを検査し、不整合な結果を fail closed
にする。一つの path query は buffer record の snapshot を保持するため、query
中の更新で参照元が入れ替わらない。

## query policy と因果性

### Sample selection

`EXACT`、`NEAREST_WITHIN_TOLERANCE`、`LATEST`、`LATEST_COMMON` は保存済み sample
の選択 policy である。model-based policy でも requested stamp と sample stamp
が完全一致すれば model を呼ばず、`sample_selection` evaluation を返す。
この exact-sample 経路には temporal model の登録を要求しない。

### Model interpolation

`INTERPOLATE_WITH_MODEL` は requested stamp を厳密に挟む同一 authority の二端点
だけを model に渡す。右端点は requested stamp より未来なので、
`query_mode=TemporalQueryMode.OFFLINE_SMOOTHING` を明示しなければ拒否する。
端点外の外挿、異なる authority の混在、model support 外の分布は成功扱いに
しない。

### Causal prediction

`PREDICT_WITH_MODEL` は requested stamp 以下の record だけからなる因果的な履歴
suffix を渡す。次の三つの上限をすべて満たす必要がある。

1. query の `max_prediction_horizon`
2. model instance の `maximum_horizon`
3. query の `max_age` による anchor freshness

future record を後から挿入しても同じ online prediction が変化しないことを
conformance test で固定している。必要履歴数、orientation kind、acceleration
metadata などが model support を満たさない場合も明示 error になる。

### Offline smoothing

offline smoothing は未来 evidence の利用を認める mode であり、online prediction
とは因果性の主張が異なる。現在 core が公開する endpoint-conditioned
interpolation はこの mode の一部である。Grape のように pose、velocity、
acceleration、IMU bias、control をまたぐ joint trajectory posterior は
application estimator/plugin の責務であり、独立な時刻 marginal を差分して
代用してはならない。

## model registration と選択

model は physical edge と authority の組に bind し、`model_id` で識別する。
binding が一つだけなら query の `model_id` は省略でき、その唯一の model が
選択される。監査可能性のため application からは selector の明示を推奨する。
同じ組に複数 model を `make_default=False` で登録した場合、query は
`model_id` を明示しなければ `MODEL_AMBIGUOUS` になる。未登録 edge の暗黙の
外挿や、名前なし global model は存在しない。

`make_default=True` は利用者が一つの graph instance 内で行う明示的な binding
選択であり、この package の評価済み production default を意味しない。
2026-07-24 の選定では production `DEFAULT` は未選定なので、本番導入前に
application corpus で calibration gate を再実行すること。

## process noise

canonical parameter は continuous-time spectral density \(Q_c\) である。
model の SE(3) convention は

\[
T(t+h)=T(t)\operatorname{Exp}(h\,\xi_\mathrm{body}),
\qquad \xi=[\rho,\phi].
\]

したがって process noise はこの 6 次元接空間の単位と frame convention に
合わせる。legacy の per-step covariance \(Q_d\) は sampling period
\(\Delta t\) と不可分であり、

\[
Q_c = Q_d / \Delta t
\]

として `adapt_discrete_process_noise(Qd, sample_period)` でのみ移行する。
adapter は `DISCRETE_PROCESS_NOISE_ADAPTED` diagnostic を返す。canonical config
に `Qd` を直接保存したり、sample period を暗黙値にしたりしない。

凍結評価では 0.05 s と 0.1 s の二レートで adapter の予測等価性が
絶対誤差 \(10^{-9}\) 以内を満たした。この adapter の判定だけが `OPTIONAL` で
あり、temporal model/backend 自体の production default を意味しない。

## uncertainty backend

### Tangent-space moment

代表 pose と接空間 covariance を伝播する高速近似である。source endpoint 間の
cross-time covariance を表現できない場合は `DEPENDENCE_APPROXIMATED` と warning
を付ける。2026-07-24 run では高速だったが、distribution stratum ごとの 95%
coverage gate をすべては通過しなかったため `EXPERIMENTAL` である。

### Sample-wise

固定 seed/stream と sample ID を用い、非線形・非 Gaussian な伝播の参照 oracle
として使える。評価では pose/covariance/energy metric を改善したが、現在の
marginal endpoint record だけでは、共有 calibration factor と独立 measurement
residual を factor level で分離できない。joint-sample contract がないまま
共有 latent を厳密に扱ったと主張してはならず、状態は `EXPERIMENTAL` である。

## failure と degraded result

主要な `GraphErrorCode` は次の通り。

| code | 意味 |
|---|---|
| `MODEL_NOT_REGISTERED` | edge/authority/model ID に binding がない |
| `MODEL_AMBIGUOUS` | 複数 model があり selector がない |
| `MODEL_SUPPORT_EXCEEDED` | interpolation bracket、分布 kind、metadata などが support 外 |
| `PREDICTION_HORIZON_EXCEEDED` | query または model の horizon 上限を超えた |
| `INSUFFICIENT_HISTORY` | model の最小履歴数を満たさない |
| `NON_CAUSAL_INPUT_REJECTED` | online mode が未来 endpoint を必要とした |
| `TEMPORAL_STALE` | anchor が `max_age` より古い |
| `UNCERTAINTY_LIMIT_EXCEEDED` | propagated uncertainty trace が上限を超えた |

`max_uncertainty_trace` 超過時は既定で reject する。`allow_degraded=True` を
明示した場合だけ数値を返し、path-level と各 component の temporal detail の
両方に degraded diagnostic、requested stamp、warning を残す。consumer は
warning を無視して通常値として扱ってはならない。

## ROS と TF2 の境界

ProbTF v2 bridge は evaluation detail を ROS message と往復できる。標準 TF/TF2
が持つのは時刻付き代表 transform であり、次は保持できない。

- transform distribution、mixture component/weight
- dependency ID と共有 latent の関係
- model/config fingerprint、backend、random stream
- source stamp 群、prediction horizon、approximation/degraded warning

したがって TF2 publish は lossless serialization ではない。ProbTF message を
正本として同時保存し、TF2 は既存 consumer・可視化向けの射影として扱う。

## Grape adapter の境界

core は Grape dynamics、PID/controller、actuator calibration、wrench likelihood、
成功条件を import しない。Grape application plugin は次を担当する。

1. mocap/IMU/controller/actuator evidence の時刻同期と source provenance
2. joint trajectory/control posterior の保持
3. 必要な pose marginal の `TransformDistributionStamped` 化
4. application corpus による model/backend calibration と安全 gate

core selection corpus は匿名化した synthetic fixture だけである。Grape の
PC controller と MCU firmware を同一 provenance で結合した exact replay、
および独立した Grape corpus での再現性は、この run では実証していない。
`TrajectoryParticleSet` のような application contract は、二つ目の独立
application が同じ sample/weight/provenance 契約を必要とするまで core API や
message に昇格させない。

## 凍結評価の再実行

repository root で workspace を setup し、source checkout から実行する。
runner は conformance suite `tests/probtf` と git provenance を参照するため、
install space だけをコピーした環境は再現 run とみなさない。

```bash
source /home/leus/catkin_ws/devel/setup.bash
export PYTHONPATH=/tmp/probtf_test_deps:ros/core/probtf_core/src:${PYTHONPATH}
python3 ros/core/probtf_core/test/run_temporal_selection.py \
  --output ros/core/probtf_core/test/temporal_selection_results_2026-07-24.json
```

`/tmp/probtf_test_deps` はこの workspace で Bingham 正規化 test が使う
評価用 dependency の場所であり、環境に同じ dependency が通常 install されて
いれば省略できる。runner は開始時に corpus hash と train/held-out split を
検査し、実行環境、seed、git tree、source/config/runner/corpus hash、
conformance 出力を JSON に保存する。

記録済み artifact:

- [temporal_selection_results_2026-07-24.json](../test/temporal_selection_results_2026-07-24.json)
- SHA-256:
  `72e61c62e2f65dffc9d9ddacd678476796b61e21202be319b6894613bae97c22`
- repository HEAD at run:
  `ccef72f10dd1762cfb340d7f458d62f1d9ddda3f`

判定の要約は [SELECTION_RESULTS.md](../SELECTION_RESULTS.md) を参照する。
