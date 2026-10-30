# 正解ゲイン位置の併記

2026年10月30日のユーザー指定により、33・65格子の3D図とP–D断面図へ正解ゲインの位置を追加した。既存の各PDF・SVG・PNGを更新する。

青い菱形は `Success ref.` と成功記録の各群の(P,I,D)座標、紫の印は `Failure` と失敗記録の各群の(P,I,D)座標を示す。最新指定では、両方の点に残り9ゲインの二条件を併記する。

各点の注釈には、中央揃えの `rest PID` 見出しの下に、`success: rate = xx.x%` と `failure: rate = yy.y%` を表示する。`rest PID`は図の3軸で示すP/I/D以外の9ゲインである。点自身のP/I/Dを固定したまま、この9ゲインの出所だけを変える。`rest PID`の行は描画機能で中央揃えにし、空白文字による位置調節は行わない。既存の33・65格子の図を同じファイル名で更新する。

| 群 | 正解ゲイン(P,I,D) |
|---|---|
| Roll/Pitch | (13,1,20) |
| Yaw | (6,1,2) |
| XY | (4,0.1,2) |
| Z | (5,1,2.5) |

群をj、点のP/I/Dをa、rest PIDの出所をbとすれば、評価する全12ゲインは、j群だけをa、他の群をbの記録値としたものである。Failure点・Success点の両方でb=success/failureを評価する。背景の色は従来どおりb=failureであり、b=successの値を背景から補間したり、背景の固定値を変えたりしない。成功標本による条件付けや100%になるような値の変更は行わない。

独立比較には、当該failureの保存済み128標本を元の順序でそのまま使用する。機体モデル、推定値、actuatorの時定数、controller周期も当該failureの計算時のものを使う。failure指令だけを初期参照とする別の評価器で、2ケース×4群×2点×2条件の32行を評価する。同一の全12ゲインは再計算せず、各ケース8個の異なる制御器の結果を共有する。実際の全12ゲイン、各標本のスペクトル半径、安定数、有効数、非有限数、標本同一性と入力の出所は `rest_pid_comparison.json` に保存する。安定数は有限なスペクトル半径が1未満の数、分母は未解決標本も含む全128である。

| 使用したfailureの標本 | 全12成功ゲインの安定数／全標本数 | Success点・rest PID=successのrate | 非有限数 |
|---|---:|---:|---:|
| failure1 | 127/128 | 99.2% | 0 |
| failure2 | 128/128 | 100.0% | 0 |

上表は以前の全12成功ゲイン比較 `successful_controller_comparison.json` と一致する部分である。Success点・rest PID=successは全群で同じ全12成功ゲイン、Failure点・rest PID=failureは全群で同じ全12失敗ゲインになる。一方、交差する二つの組合せは群ごとに異なるため、rateも群によって異なり得る。

参照用にfailure制御器も再評価し、両ケースとも保存済み全128標本のスペクトル半径とビット単位で一致した。数値計算に使う133ソースファイルも、元マップ計算時の凍結版と一致する。これは有限標本上の局所線形安定割合であり、実飛行の成功確率ではない。

成功記録から取り出すのは指令ゲインだけであり、その推定機体パラメータ、旧安定判定、旧条件付き分母は使用しない。全マップの完了後に表示参照と今回指定された有限個の点比較へ使う。出所とSHA256を `successful_gain_reference.json` に残す。`figure_index.json` には各点・各contextの実際の全12ゲインとrate、および計算済み領域・表示領域を区別して記録する。

Roll/PitchのD=20は計算済みD上限16の外であるため、表示枠だけを広げる。格子外は空白であり、背景値の外挿・補間・格子への追加計算は行わない。Zは正解・失敗の座標が重なるので、紫の輪と青い菱形を同一座標に描き、注釈だけを分ける。点の座標はずらさない。今回の全群で正解・失敗のIは一致しており、P–D断面のIも変えない。

独立比較を再計算するには、リポジトリルートから以下を実行する。既存の格子データは読み取りだけで使う。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /usr/bin/python3 ros/examples/grape-param-estim/minimal/first-order-lag/automation/evaluate_rest_pid_reference.py \
  --publication ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029 \
  --reference ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029/successful_gain_reference.json \
  --output ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029/rest_pid_comparison.json
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
  --rest-pid-comparison-json ros/examples/grape-param-estim/minimal/first-order-lag/publication/pid_gain_voxels_failure_only_33_65_20261029/rest_pid_comparison.json
```

保存した数値データ96ファイルのSHA256は追加前後で一致する。色の4096エントリも前回修正版と同一である。今回の有限個の点比較を、候補標本の条件付けや背景の他9ゲインの置換へ戻さない。


各点・各条件の表示値は以下のとおり。すべて分母は128。

| ケース | 群 | Failure点 rest=success | Failure点 rest=failure | Success点 rest=success | Success点 rest=failure |
|---|---|---:|---:|---:|---:|
| failure1 | roll_pitch | 55.5% | 48.4% | 99.2% | 96.1% |
| failure1 | yaw | 96.1% | 48.4% | 99.2% | 50.8% |
| failure1 | xy | 99.2% | 48.4% | 99.2% | 53.1% |
| failure1 | z | 99.2% | 48.4% | 99.2% | 48.4% |
| failure2 | roll_pitch | 14.1% | 11.7% | 100.0% | 98.4% |
| failure2 | yaw | 98.4% | 11.7% | 100.0% | 14.8% |
| failure2 | xy | 100.0% | 11.7% | 100.0% | 11.7% |
| failure2 | z | 100.0% | 11.7% | 100.0% | 11.7% |
