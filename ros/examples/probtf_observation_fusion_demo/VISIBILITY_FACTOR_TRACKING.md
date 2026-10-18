# Running Gradient Trackingへの視野因子（2026年10月18日）

2026年10月18日追加訂正: 新しい既定のview因子は10度の中央許容円錐内で零になる `tolerance_cone` とした。旧常時中央化は `always_center` の比較用として残す。weight=2e6、task J、身体/目標lawは保持する。局所曲率と新しい実測は [LOCAL_MODEL_TRACKING.md](LOCAL_MODEL_TRACKING.md) を参照する。以下の旧因子の実測を新しいsourceの到達性能と混同しない。

指定された[指示書原文](VISIBILITY_FACTOR_INSTRUCTIONS_20261018.md)を保存した。変更前は`a8f1bac5acac3323e1a1e4fb948d8f712594b22b`。今回の追加は指令選定だけに作用する。[従来方式の説明と実測](RUNNING_GRADIENT_TRACKING.md)は、視野因子追加前の履歴として残す。

共通30秒の実測では、検出時間率がmarkerで25.36%→96.06%、IMU併用で23.12%→96.10%になった。ただし到達は未達で、IMU版は剛性evidenceの数値判定で途中停止した。視野維持の効果と、未解決の推定遅延・停止を分けて以下へ記録する。

## 分離した二つの因子

既存の完全タスク規準を$J_t(c)$、タスク因子を$\Lambda_{\mathrm{task},t}(c)=\exp[-J_t(c)]$とする。Gaussianの全共分散・正規化項、Bingham正規化項、全mixture・log weight、身体と目標の交差を変更しない。

各成分$k$について、同じ身体候補と凍結した剛性・平衡branchを試行指令$c$へ輸送し、$T_{CM,k}(c)=T_{BC}(c)^{-1}T_{BM,k}(c)$をcoreへ照会する。対象は保持したマーカー中央であり、ボタンへの取付offsetを含めない。光学座標の前方は$+z$である。その位置平均を$p_k(c)$として、

$\phi(p)=1-p_z/\|p\|$、$V_t(c)=\sum_k\bar w_k\phi(p_k(c))$、$\bar w_k=\exp(\log w_k-\operatorname{LSE}_j\log w_j)$とする。

全成分を個別に評価してから寄与を足す。一点平均へまとめず、平均・全共分散・重み・log weight・出自を保存する。後方の成分も削除しない。$\|p\|\le10^{-9}$ mや非有限値では使用不能の理由を記録し、零勾配を作らない。正面と真後ろの軸上には方向規準の停留点がある。

選定レイヤだけで$\Lambda_{\mathrm{select}}=\Lambda_{\mathrm{task}}\Lambda_{\mathrm{view}}$、$\log\Lambda_{\mathrm{view}}=-\lambda V$と合成する。実行へ渡す符号は、

$g_{\mathrm{select}}=\nabla_cJ+\lambda\nabla_cV=-\nabla_c\log\Lambda_{\mathrm{select}}$である。

`task_gradient`は$\nabla J$、`view_gradient`は係数を掛ける前の$\nabla V$、`weighted_view_gradient`は$\lambda\nabla V$である。三つのlog因子も別々に保存する。視野因子を身体や剛性の尤度、BFS evidenceへ入れない。可視確率、四隅の可視性や遮蔽の保証とも定義しない。

## 計算と実行

`visibility_factor.py`は照会結果を読む因子、`selection_gradient.py`は二つの因子を合成する選定レイヤである。既存のタスク差分評価にobserverを設け、同じ試行で既に輸送した共同状態から視野を照会する。指令作用を二重計算せず、タスクと同じ中心差分または片側差分の幅・許容域を使う。モンテカルロ、modal姿勢だけのJacobianへの置換、旧visual接近、位置s制約、端点IKは使用しない。

`enabled=false`または`weight=0`ではobserverと視野照会を通さない。従来のJ、差分勾配をそのまま返し、単位因子のlog値は0とする。未計算のVとその勾配はnullであり、計測した0と表示しない。

不変packetへ両因子・勾配・同じ身体/目標ID・剛性密度ID/K・評価指令/時刻・元の身体観測時刻/保持目標画像時刻を保存する。表示snapshotも同じlaw・Kと一体で採用する。視野評価が使用不能なら新packetを棄却し、最後の受理済み入力・law・K・表示を保持する。初回に受理済み入力がなければ停止したまま待つ。

実行は従来の$\dot u=v,\ \dot v=-\gamma v-\eta g_{\mathrm{select}}$、`eta=1e-6, gamma=8.0`のままである。新しい勾配や画像、Kの到着で$u,v$をリセットしない。画像喪失中も取得済み保持マーカーとIMU更新・実指令ledgerの身体予測から両因子を評価する。未取得マーカーを捏造せず、raw検出だけで停止・探索・再初期化を開始しない。取得目標120/40/30/12 Hz、物理・推定モデル、coreとRViz設定を保持する。

## 重みの尺度

同じ初期観測・共同候補で、マーカー単独では$\|\nabla J\|=391317.987$、$\|\nabla V\|=0.2145623$、両方向のcosは$-0.746913$。IMU併用では$448059.779$、$0.2138436$、$-0.822290$だった。両項は競合し、重み1では視野側が約200万分の1になる。

両ケース共通の初期設定を`weight=2000000.0`とする。同じ初期点では選定方向$-g_{\mathrm{select}}$とVの勾配の内積はそれぞれ$-29361.621$、$-12670.683$であり、J側とも降下方向を共有した。初速0なら最初の$\dot V$は0で、この値は初期加速方向の局所診断である。全時間の降下や可視性、到達を保証しない。重み0を同じ設定・時間で比較する。

## 再現

```bash
source ~/catkin_ws/devel/setup.bash
roslaunch probtf_observation_fusion_demo marker.launch
# 別masterでIMU併用。
ROS_MASTER_URI=http://localhost:11854 roslaunch -p 11854 probtf_observation_fusion_demo imu_marker.launch
```

重み0・90秒は次のwrapperを`/tmp/visibility-zero.launch`へ保存し、`roslaunch /tmp/visibility-zero.launch`で起動する。IMU併用では`marker.launch`を`imu_marker.launch`へ替える。今回の比較wrapperは同じmarker.rviz由来のコピーを使用し、ウィンドウ位置以外の全設定を一致させた。

```xml
<launch>
  <include file="$(find probtf_observation_fusion_demo)/launch/marker.launch">
    <arg name="output_dir" value="/tmp/probtf-visibility-zero-marker" />
  </include>
  <param name="/observation_fusion_demo/async_duration_s" value="90.0" type="double" />
  <rosparam param="/observation_fusion_demo/visibility_factor_options">{weight: 0.0}</rosparam>
</launch>
```

```bash
rosrun probtf_observation_fusion_demo evaluate_running_gradient.py RUN_DIR \
  --near-threshold 0.006 --output /tmp/visibility-evaluation.json
```

評価CLIは検出frame率と取得stampの左holdによる時間率を分け、連続喪失と再取得、未復帰の打切り、予測角度・画素、勾配の競合、同一source、接続残差を保存する。ボード四隅の物理座標は保存済み取得要求を読む評価側だけで使い、制御や推定へ渡さない。瞬時姿勢と露光中の7姿勢、黒領域と白土台を区別する。未記録値を0や成功へ補完しない。

## 今回の検証結果

結果と凍結source、比較の出力先は`validation/visibility_factor_20261018.json`へ記録する。大容量のraw画像、SQLite、途中ログ、動画はGitへ追加しない。

重み0の2本は60秒完了した。正重みは最初の90秒試行がIMU側66.846522秒で停止し、60秒再試行もIMU側37.067380秒で停止した。各回で相手側のmarker実行も終了した。正重みを60秒・90秒完走したとは扱わない。以下は全4本が実際に進んだ共通の先頭30秒だけの比較であり、元ログや取得時刻、未完了の状態は変更していない。

| ケース・重み | 検出時間率 | 最長連続ロスト | 白土台四隅の画像内frame率 | 最小物理距離 | 最後の物理距離 |
| --- | ---: | ---: | ---: | ---: | ---: |
| marker・0 | 25.36% | 21.333秒以上・未復帰 | 5.00% | 354.974 mm | 401.926 mm |
| marker・2,000,000 | 96.06% | 0.754秒・再取得 | 100% | 344.129 mm | 351.056 mm |
| IMU＋marker・0 | 23.12% | 22.001秒以上・未復帰 | 5.00% | 357.682 mm | 476.930 mm |
| IMU＋marker・2,000,000 | 96.10% | 0.752秒・再取得 | 100% | 344.788 mm | 344.792 mm |

検出frame率はそれぞれ25.28%、96.10%、23.06%、96.11%。検出時間率は画像取得stamp間の左holdだけで求め、30秒境界まで未観測の末尾約0.07〜0.08秒を延長していない。物理距離も30秒以下の最後の実サンプル（29.9937〜29.9992秒）を使う。未復帰のロストは右側打切りの下限であり、再取得時間とは扱わない。正重みは各3回再取得し、比較窓の末尾に未復帰はなかった。

黒い領域の四隅も同じ5%→100%で、露光中の7姿勢の全四隅でも同じ結果だった。これは評価側の保存物理姿勢を投影した幾何で、検出結果とは別である。部分的なマーカーが検出できるため、四隅の画像内率と検出率は一致しない。正重みの予測光軸角度は平均1.380°／1.266°だった。重み0ではVを計算していないため、その予測角度はnullとした。全ケースで6 mm近傍滞在は0で、距離条件のランプ点灯もなく、ロスト軽減を到達成功とは説明しない。

3回の実行群で、各ケースの初回有効候補の身体・目標ID、K・剛性密度ID、J・task勾配は完全一致した。全4本の実行module hashも一致した。ただし壁時計の取得tick・非同期完了時刻は一致せず、同一の実行軌道や同一の観測列ではない。単回ずつの比較であり、統計的な性能保証ではない。

視野差分照会の追加時間は共通窓で平均22.42／21.88 ms/勾配だった。taskと重み付きviewの勾配方向が逆向きだった割合は90.32%／94.68%。両者の競合は残る。実物理サンプルは正重み107.77／108.04 Hz、重み0は109.47／109.66 Hzで、目標120 Hzは達成していない。画像取得は約12 Hz。物理実行を推定待ちへ変更していない。

### 剛性推定による停止と未処理データ

以下は旧sourceの停止記録である。その後の原因調査で、2件ともKeep枝の有限な結果に対するstationarity監査で止まったことが判明した。全処理共通の新方針と修正・再生結果は[数値監査と早期終了の扱い](NUMERICAL_DIAGNOSTICS.md)を参照する。元の未完走ログを完走済みへ変更していない。

最初の正重み試行では取得0.291659秒の`imu:00000033`で、剛性evidenceのBFGSが14回後にprecision lossで終了した。mode scoreの最大絶対値は`2.60019e-4`で既定`1e-6`を満たさず、物理66.846522秒で明示停止した。再試行では取得0.240355秒の`imu:00000029`、13回、`1.19361e-4`で物理37.067380秒に停止した。どちらもHessian計算前の失敗である。Keep/Resetのどちらの積分で失敗したかは既存診断に識別子がなく、断定しない。

直前の剛性密度と全因子は保持され、新因子はpendingで、新しい事後・K代表は存在しない。許容値や剛性更新式を変更せず、失敗を零情報の更新や成功へ置き換えなかった。停止時の観測ageは約66.55／36.83秒で、全時刻順推定の大きな遅延が残る。raw検出の継続と、後段で新画像の推定が完了したことは区別する。今回の視野因子はこの遅延と数値停止を解消していない。

重み0の終了記録では全取得4,728／4,714件を保存し、未処理は4,542／4,688件だった。推定drain済みとは記録しない。正重み試行の保存数・中断イベント・未処理数もJSONへ残す。失敗群のmarker側は比較相手の停止に伴う終了であり、marker推定そのものの失敗と混同しない。物理距離の最終値はmanifestの最終書込みより後の実サンプルがあれば、その保存物理ログを優先する。

中断したmarker側では、90秒試行の取得journal 5,210件に対しDB 5,207件、60秒再試行では2,974件に対し2,969件で、停止直前の3／5要求がDBへ保存されていない。元journalは保持し、この2本を全取得DB保存完了とは扱わない。IMU側と重み0の両側はjournal件数とDB件数が一致した。比較窓30秒の画像要求とraw previewは全4本とも一致している。元DBや失敗ログへ後から生成値を足して差を隠していない。

### 数値・同一性検証と再集計

回帰713件、追加のselection/async 13件、評価CLI最終9件が通過し、収集した727件すべてをカバーした（CLIの8件は回帰と重複）。ROSビルドは成功し、既存のBoost警告とhppfcl非推奨警告がある。重み0/無効時のJ・勾配・非零初速と不規則dtでの実行u/vのバイト一致、同じtrial stateの共有、全crossと推定履歴の不変、欠落中IMU更新、再観測一回処理、使用不能packetの拒否を確認した。

全6本の34,041実区間では、保存したu/vの厳密hold式からの残差と採用前後の状態差は0だった。これは指令状態の保存値の検査で、実物理の到達や連続時間精度の証明ではない。core142ファイル・RViz43ファイルを含む保護対象214ファイルと、native物理ライブラリのhashは一致した。

保存した1,047 snapshotの46,180点照会から63,002個のROS markerと8,482,160粒子点を再生成し、wireの全バイトが一致した。採用済packet1,006件・初回目標待ちの棄却40件、全task/view出自と615件の有効view合成・392件の単位因子を照合した。最後の制御計算1件は採用前の終了である。これは保存したcore点照会・packet・表示データの同一性検査で、rawからの全推定の独立再実行や全joint密度の再伝播ではない。

これとは別に、重み0のIMU側の終了時にJSON未保存の孤立wireが1件ある。その1件は数値照合不能で、公開・採用・一致を主張しない。JSONとcontrol記録がある1,047件の合格と、保存された全wire artifactが検証可能であることを区別する。

実装コミットは`47d4699788cdecbbcc3263472f1119029ce31214`。実行時に保存した全module・設定・評価CLIのhashを、このコミットの内容と照合した。検証JSONと本追記は後続の記録コミットで保存する。

元の全記録を読み、時刻条件で評価だけを選別するhelperと、表示の全wireを再生成するhelperを保存する。これらは制御側では使用しない。

```bash
cd ~/catkin_ws/src/ProbTF-demo/ros/examples/probtf_observation_fusion_demo
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 \
  validation/tools/visibility_common_prefix.py --duration 30 \
  --zero-root /tmp/probtf_visibility_zero60_20261018_0827 \
  --positive-root /tmp/probtf_visibility_positive60_20261018_0826 \
  --output /tmp/visibility-common30.json
source ~/catkin_ws/devel/setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 \
  validation/tools/visibility_wire_identity.py \
  /tmp/probtf_visibility_zero60_20261018_0827/marker \
  /tmp/probtf_visibility_positive60_20261018_0826/marker \
  --output /tmp/visibility-wire.json
```
