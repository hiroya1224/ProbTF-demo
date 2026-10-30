# 正解ゲイン位置の併記

2026年10月30日のユーザー指定により、33・65格子の3D図とP–D断面図へ正解ゲインの位置を追加した。既存の各PDF・SVG・PNGを更新する。

青い菱形のラベルは `Success ref.`、成功記録の各群の(P,I,D)座標、独立に計算した `rate` を示す。紫の印は引き続き、全12ゲインが当該失敗記録の値である実失敗制御器を示す。

| 群 | 正解ゲイン(P,I,D) |
|---|---|
| Roll/Pitch | (13,1,20) |
| Yaw | (6,1,2) |
| XY | (4,0.1,2) |
| Z | (5,1,2.5) |

背景の色は、他9ゲインを失敗記録の値へ固定したまま計算済みの安定割合である。青の `rate` は、それとは別に、全12ゲインを正解値にした制御器を元の全標本で評価した割合である。青い座標を背景から読み取った値ではないことを、図の説明とmetadataに明記する。100%になるような条件付けや値の変更は行わない。

独立比較には、当該failureの保存済み128標本を元の順序でそのまま使用する。機体モデル、推定値、actuatorの時定数、controller周期も当該failureの計算時のものを使う。failure指令を参照とする別の評価器で全12成功ゲインを一度評価し、各標本のスペクトル半径、安定数、有効数、非有限数、標本同一性と入力の出所を `successful_controller_comparison.json` に保存する。安定数は有限なスペクトル半径が1未満の数、分母は未解決標本も含む全128である。

| 使用したfailureの標本 | 全12成功ゲインの安定数／全標本数 | 図のSuccess rate | 非有限数 |
|---|---:|---:|---:|
| failure1 | 127/128 | 99.2% | 0 |
| failure2 | 128/128 | 100.0% | 0 |

同じケースの4群および33・65格子では、同じ全12ゲインの制御器を同じ128標本で比較するため、Success rateは共通になる。

参照用にfailure制御器も再評価し、両ケースとも保存済み全128標本のスペクトル半径とビット単位で一致した。数値計算に使う133ソースファイルも、元マップ計算時の凍結版と一致する。これは有限標本上の局所線形安定割合であり、実飛行の成功確率ではない。

成功記録から取り出すのは指令ゲインだけであり、その推定機体パラメータ、旧安定判定、旧条件付き分母は使用しない。全マップの完了後に表示参照と独立比較へ使う。出所とSHA256を `successful_gain_reference.json` に残す。`figure_index.json` には表示参照の出所と役割、計算済み領域、表示領域、背景とは別の全12ゲイン比較を区別して記録する。

Roll/PitchのD=20は計算済みD上限16の外であるため、表示枠だけを広げる。格子外は空白であり、背景値の外挿・補間・格子への追加計算は行わない。Zは正解・失敗の座標が重なるので、紫の輪と青い菱形を同一座標に描き、注釈だけを分ける。点の座標はずらさない。今回の全群で正解・失敗のIは一致しており、P–D断面のIも変えない。

独立比較を再計算するには、リポジトリルートから以下を実行する。既存の格子データは読み取りだけで使う。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /usr/bin/python3 ros/examples/grape-param-estim/minimal/first-order-lag/automation/evaluate_success_reference.py \
  --publication ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029 \
  --reference ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029/successful_gain_reference.json \
  --output ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029/successful_controller_comparison.json
```

図の再生成はリポジトリルートから、以下を33と65に対して実行する。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /usr/bin/python3 ros/examples/grape-param-estim/minimal/first-order-lag/automation/render_failure_publication.py \
  --input-root ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029/data \
  --output ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029/figures/grid_65 \
  --settings ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029/figure_settings.json \
  --grid 65 \
  --success-gains-json ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029/successful_gain_reference.json \
  --success-comparison-json ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029/successful_controller_comparison.json
```

保存した数値データ96ファイルのSHA256は追加前後で一致する。色の4096エントリも前回修正版と同一である。成功座標の表示は、候補標本の条件付けや他9ゲインの置換に使わない。
