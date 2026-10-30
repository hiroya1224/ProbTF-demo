# 正解ゲイン位置の併記

2026年10月30日のユーザー指定により、33・65格子の3D図とP–D断面図へ正解ゲインの位置を追加した。既存の各PDF・SVG・PNGを更新する。

青い菱形の `Success ref.`／`position only` は、成功記録の各群の(P,I,D)座標を示す。紫の印は引き続き、全12ゲインが当該失敗記録の値である実失敗制御器を示す。

| 群 | 正解ゲイン(P,I,D) |
|---|---|
| Roll/Pitch | (13,1,20) |
| Yaw | (6,1,2) |
| XY | (4,0.1,2) |
| Z | (5,1,2.5) |

背景の色は、他9ゲインを失敗記録の値へ固定したまま計算済みの安定割合である。青い座標は、全12ゲインを正解値にした制御器の評価点とは異なる。青い印には安定率を付けず、成功100%の主張も行わない。

成功記録から取り出すのは指令ゲインだけであり、その推定機体パラメータ、旧安定判定、旧条件付き分母は使用しない。全マップの完了後に表示参照として読み込む。出所とSHA256を `successful_gain_reference.json` に残す。`figure_index.json` には表示参照の出所と役割、計算済み領域、表示領域を区別して記録する。

Roll/PitchのD=20は計算済みD上限16の外であるため、表示枠だけを広げる。格子外は空白であり、外挿や補間、追加の安定計算は行わない。Zは正解・失敗の座標が重なるので、紫の輪と青い菱形を同一座標に描き、注釈だけを分ける。点の座標はずらさない。今回の全群で正解・失敗のIは一致しており、P–D断面のIも変えない。

再生成はリポジトリルートから、以下を33と65に対して実行する。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /usr/bin/python3 ros/examples/grape-param-estim/minimal/first-order-lag/automation/render_failure_publication.py \
  --input-root ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029/data \
  --output ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029/figures/grid_65 \
  --settings ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029/figure_settings.json \
  --grid 65 \
  --success-gains-json ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029/successful_gain_reference.json
```

保存した数値データ96ファイルのSHA256は追加前後で一致する。色の4096エントリも前回修正版と同一である。成功座標の表示は、候補標本の条件付けや他9ゲインの置換に使わない。
