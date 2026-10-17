# 最新制御での設定（2026年10月18日）

以下は`7046be5`の240 Hz化の履歴である。その後ユーザーが指定した[統合指示書](UNIFIED_RUNNING_GRADIENT_THEORY_20261018.md)の120/40/30/12 Hzへ変更し、[Running Gradient Tracking](RUNNING_GRADIENT_TRACKING.md)で新しく測定する。全リンク変換コピーの省略とC++動力学は保持する。

## 物理更新の240 Hz化（2026年10月18日・旧source）

ユーザーの「物理更新は110Hzよりあげられませんか」に従い、既定の外側物理更新目標を120 Hzから240 Hzへ引き上げた。設定は `moving_estimation.yaml` の `async_execution.physics_hz` である。指令通知40 Hz・IMU30 Hz・画像12 Hzと、各データの全取得・時刻順処理は維持する。動作の速度上限・加速度・0.5秒の位置参照を速める変更ではない。

前の約110 Hzは、120 Hz目標に対する外側の更新実測だった。C++の内部積分は既に最大1 ms刻みであり、110 Hzが積分器の計算限界だったわけではない。前の10秒記録ではtick間隔の中央値は8.333 ms、95%点15.59 ms、最大88.0 msで、散発的な遅延が平均周波数を下げていた。

## 計算を減らした箇所

物理側の固定根元変換とframe IDを一度だけ解決してcacheする。根元frameが固定jointに属することを検査する。内部積分のカメラ履歴を作る際は、必要なカメラ変換だけをコピーし、毎回11リンク分の行列と辞書を構築しない。Pinocchioの順運動学自体と、その結果の変換値は同じである。

全リンクを描画する終点では全リンク姿勢を作る。内部刻みの移動した実カメラ姿勢と時刻はすべて保持し、露光0.09秒・7時刻のブラーへ渡す。間引きによる高速化ではない。独立に計算した全リンク・カメラ変換とのbitwise一致と、露光履歴の保持をテストした。

C++積分、内部刻み上限、Newton法、減衰2.0、干渉判定一時OFF、推定・IKの補助平衡モデルや候補PDFには変更を加えていない。物理の状態や真のKを推定へ渡す変更もない。

比較用 `benchmark_physics.py` では、単独native呼出しにも `elapsed-previous_elapsed` を使い、sceneと全く同じ浮動小数のdtを渡すよう直した。名目periodとの差が丸めによって内部の分割数を変え、異なる離散化同士を比較していた問題への修正であり、積分器の許容誤差を緩めたものではない。

## CPU計測

240 Hz相当の規定指令を2秒・480フレーム実行した。画像描画・ROS・RVizを含まない、sleepを入れない計算時間の測定である。

| 処理 | 平均 | 95%点 |
|---|---:|---:|
| native積分呼出し | 0.111 ms | 0.134 ms |
| 物理frame全体 | 0.421 ms | 0.502 ms |

別のcProfile付き診断では、同じ240 Hz相当の物理frameが最適化前0.842 ms、最適化後0.529 msだった。計測のinstrumentationが異なるため、上表の0.421 msとは区別する。別の推定済みKを使う指令作用計算も、通常launchと同じBLAS1スレッドでは平均0.279 msだった。

## ROS実測と並列設定

通常の `common.launch` は変更前から `OPENBLAS_NUM_THREADS=1` と `OMP_NUM_THREADS=1` を指定している。最終比較は、この既存launchと同じ環境で行う。

| 条件 | 動作期間 | 物理更新 | 画像取得 |
|---|---:|---:|---:|
| no sensor | 10秒 | 219.20 Hz | 12.00 fps |
| IMU・推定Kを全実行区間で使用 | 1秒 | 214.02 Hz | 12.03 fps |

IMU試行は、実際のIMU更新でKが選ばれ、そのKの完全規準IKが採用されるまで静止して待ってから開始する測定である。この待機は上表の1秒と物理Hzに含めない。通常のautostartの仕様を変えたものではない。各実行区間の凍結Kと実指令ledgerを照合する。推定は動作後も全取得を時刻順に処理する。この短時間測定から、IMU条件の長時間性能や到達成功を主張しない。

IMU試行の全215実行区間で選定済みKを使った。両試行の取得は814件・84件で、すべてDBへ保存・処理し、同じsourceの分布wireと保存済み制御snapshotの対応も照合した。IMUの1秒動作が終わった後、全観測を処理し終えるまで約179秒かかった。物理更新の高速化と推定処理の速度は別であり、この変更で推定backlogが解消したとは説明しない。推定・IKの最終metadataにある完了Hzはdrain期間も含むため、物理動作中のHzとして転用しない。

最初に環境変数なしで直接Pythonを実行した診断も保存した。NumPy・SciPyが各64スレッドを作る条件で、no sensorは10秒218.57 Hz、推定K使用のIMUは0.5秒97.21 Hzだった。後者は短い測定窓に散発的な長い遅延が入っている。同じKの隔離計測では1スレッド化で計算時間・jitterが減ったが、60〜70 msの遅延を単独再現したわけではなく、BLASだけが唯一の原因とは断定しない。実測値にはOS負荷・IPC・ログ出力・他のprocessとの競合が含まれる。

240 Hzは目標であり、全tickの期限を保証する値ではない。画像・IMU・Jointの独立取得周期と、推定・IKの完了周波数を混同しない。500 Hz相当のCPU benchmarkも通ったが、外側の既定を500 Hzにしたり、ROSで500 Hz動いたと説明したりはしない。

## 検証と起動

全623件のテストが通過した（233.01秒、既存hpp-fclの非推奨警告1件）。240 tickを進める間に推定workerを待たせても、1秒内のJoint41件・IMU31件・画像13件（それぞれ初期取得を含む）を全保存する既存テストも更新した。RViz37ファイル、作業前からのユーザーファイル28件、core142ファイルはhash一致した。rawや動画・DBはGitへ追加しない。

既存の起動コマンドで240 Hz目標を使う。

```bash
source ~/catkin_ws/devel/setup.bash
roslaunch probtf_observation_fusion_demo imu_marker.launch
```

CPU benchmarkは測定周期を明示して実行できる。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  rosrun probtf_observation_fusion_demo benchmark_physics.py \
  --physics-hz 240 --output /tmp/probtf_physics_240_new
```

source・全取得・実指令・配信照合の結果は [検証記録](validation/physics_rate_20261018.json) に保存する。以前の120 Hz sourceの検証は [動力学導入時の記録](DYNAMIC_PHYSICS.md) として区別する。
