# `probtf_core`

`probtf_core` は、確率分布を持つ transform の履歴を保持し、明示的に登録した
時間モデルによる補間・予測と、その不確実性・依存関係・provenance を扱う。
ROS に依存しない Python core、ROS bridge、C++ runtime helper を同じ package に
収録している。

## 時間モデルの選定状態

2026-07-24 の凍結評価では、すべての hard correctness gate を通過した
uncertainty backend が存在しなかった。このため、package-wide な production
`DEFAULT` は**未選定**である。constant-body-twist、constant-body-acceleration、
moment backend、sample backend、endpoint-conditioned interpolation は
`EXPERIMENTAL` であり、利用者が edge/authority へ named model を明示登録して
opt-in する必要がある。監査可能性のため query でも `model_id` の明示を推奨するが、
その edge/authority に binding が一つだけなら省略時にもその model が選択される。
複数 binding があり、local default も query selector もない場合は
`MODEL_AMBIGUOUS` で fail closed する。

`ProbTfGraph.register_temporal_model(..., make_default=True)` の `default` は、
利用者がその graph instance の特定 edge/authority に対して選ぶローカルな
binding を意味する。評価済みの package-wide recommendation ではない。

測定値、判定理由、再現情報は [SELECTION_RESULTS.md](SELECTION_RESULTS.md) を
参照する。API と安全契約の詳細は
[docs/temporal_evaluation.md](docs/temporal_evaluation.md) に記載している。
入力 protocol を変えずに provenance/boolean 検証追加後の 221-test suite で
再実行した post-hardening run でも、この disposition は変わらなかった。

## 四つの時間処理を区別する

| 処理 | 入力 | 未来の観測 | 用途と provenance |
|---|---|---:|---|
| sample selection | 保存済み sample | 不要 | sample をそのまま選択する。model policy で stamp が完全一致した場合も `sample_selection` になる |
| model interpolation | query stamp を厳密に挟む二端点 | 使用する | endpoint-conditioned な補間。`offline_smoothing` 専用 |
| causal prediction | query stamp 以下の新しい履歴 | 使用しない | bounded horizon の online 予測。`model_prediction` として記録する |
| offline smoothing | 過去と未来の evidence | 使用可能 | 後処理・解析用の query mode。online prediction と同じ因果的主張をしない |

`INTERPOLATE_WITH_MODEL` を既定の `ONLINE` mode で呼ぶと
`NON_CAUSAL_INPUT_REJECTED` になる。`PREDICT_WITH_MODEL` は
`max_prediction_horizon` と `max_age` の両方を必須とし、query stamp より未来の
record を model へ渡さない。

## 明示的な登録と因果予測の最小例

次の例では、`graph` に `edge_id="mocap_body"`、`authority="mocap"` の
少なくとも二つの履歴 sample が既に `insert` されているものとする。
`Qc` は model の 6 次元接空間 convention における連続時間 process-noise
spectral density であり、離散 per-step covariance `Qd` ではない。

```python
import numpy as np

from probtf.graph import (
    GraphErrorCode,
    ProbTfGraph,
    TemporalResolutionError,
)
from probtf.temporal import (
    ConstantBodyTwistModel,
    TemporalPolicy,
)

graph = ProbTfGraph()
# graph.insert(record_at_t0)
# graph.insert(record_at_t1)

Qc = np.diag([2e-4, 2e-4, 2e-4, 1e-4, 1e-4, 1e-4])
model = ConstantBodyTwistModel(
    process_noise_spectral_density=Qc,
    maximum_horizon=0.25,
    model_id="mocap_cv_v1",
)
graph.register_temporal_model(
    "mocap_body",
    "mocap",
    model,
    make_default=False,
)

try:
    path = graph.lookup_path(
        target_frame="world",
        source_frame="body",
        stamp=12.10,
        policy=TemporalPolicy.PREDICT_WITH_MODEL,
        model_id="mocap_cv_v1",
        max_prediction_horizon=0.20,
        max_age=0.10,
        max_uncertainty_trace=0.08,
        allow_degraded=False,
    )
except TemporalResolutionError as error:
    if error.code in {
        GraphErrorCode.MODEL_NOT_REGISTERED,
        GraphErrorCode.MODEL_SUPPORT_EXCEEDED,
        GraphErrorCode.PREDICTION_HORIZON_EXCEEDED,
        GraphErrorCode.INSUFFICIENT_HISTORY,
        GraphErrorCode.TEMPORAL_STALE,
        GraphErrorCode.UNCERTAINTY_LIMIT_EXCEEDED,
    }:
        # drop, hold, or route to an application-specific fallback
        raise
    raise

evaluation = path.edge_evaluations[0]
assert evaluation.requested_stamp == 12.10
print(evaluation.model_id, evaluation.horizon, evaluation.diagnostics)
```

legacy の `Qd` を移行する場合だけ、sample period を明示して compatibility
adapter を使う。

```python
from probtf.temporal import adapt_discrete_process_noise

adapted = adapt_discrete_process_noise(Qd, sample_period=0.01)
model = ConstantBodyTwistModel(
    adapted.spectral_density,
    maximum_horizon=0.25,
    model_id="legacy_qd_migrated_v1",
)
# adapted.diagnostic を application 側の設定 provenance にも保存する。
```

## ROS/TF2 と application の境界

ProbTF v2 の ROS message/bridge は temporal detail を保持する。一方、標準 TF/TF2
へ代表 pose だけを publish すると、分布、mixture weight、dependency ID、
source stamp、model/config fingerprint、予測 horizon、近似 warning は表現できず
失われる。監査や再推定に必要な consumer は ProbTF message を正本とし、TF2 を
可視化・互換出力として扱う。

Grape の controller、actuator、plant、trajectory particle の意味論は core に
取り込まない。Grape 側 adapter/plugin が時系列 evidence を
`TransformDistributionStamped` と明示的な `TemporalModel` 契約へ変換し、Grape
固有の joint trajectory、control、wrench provenance は application 側で保持する。
今回の core 選定 run は、Grape の PC+MCU を結合した exact replay も、独立した
Grape application corpus も実証していない。

## 開発時の検証

source workspace を setup した後に core conformance suite を実行する。

```bash
source /home/leus/catkin_ws/devel/setup.bash
PYTHONPATH=ros/core/probtf_core/src python3 -m pytest -q tests/probtf
```

凍結選定 run の再実行方法と artifact 検証方法は
[docs/temporal_evaluation.md](docs/temporal_evaluation.md) を参照する。
