# symaware_grasp

対象物の軸対称 Bingham 尤度を deterministic FK の IK cost へ直接加える、pointwise
symmetry-aware grasp demo です。joint noise から手先分布を作る確率伝播や、手先分布と目標分布の
Bhattacharyya 距離最小化は行いません。

## 実行

```bash
roslaunch symaware_grasp probabilistic_tf_demo.launch
```

この launch は robot/object producer、visualizer、RViz までを起動し、IK は実行しません。
別端末から明示的に一回だけ解きます。

```bash
rosrun symaware_grasp symmetry_aware_ik_node.py
```

IK node の出力は通常の関節指令と `IKResult` だけです。手先 ProbTF や選択 target 分布を
publish・表示せず、IK 後は RobotModel の parallel gripper が半透明の cylinder を囲む形で表示されます。

## ボトルへの接近・指の開閉を記録する試作

```bash
rosrun symaware_grasp run_bottle_grasp_demo.py compare
```

軸対称分布と単一姿勢を中心とする分布で、同じ位置・初期関節角・ボトル座標系・把持手順を一回ずつ記録する。
ボトル座標系のZ軸回転は `3π/8`（67.5°）。Binghamの集中度はそれぞれ `[5000, 5000, 0]` と
`[5000, 5000, 5000]` で、後者では手首を約90°曲げて把持する。
角度は `--object-yaw-pi 3/8` で有理数として指定できる。

独立したROS masterで物体の観測、IK計算、上方を経由した12 cm手前への移動、
把持位置への直線接近、指の閉鎖、5秒の静止を実行する。
最終把持姿勢は元のBingham尤度で選び、接近経路では選んだ向きを維持する。
ボトルは既存の直径100 mm、高さ180 mmの円筒で近似し、表示用の首とキャップを添える。
指の間隔は185 mmから106 mmへ閉じる。取付部はパーム内部で止まり、把持空間に入り込まない。
生成したURDFのアームと動く指のcollision形状で経路を確認し、bagから全期間の隙間を評価する。
ロボットの円筒は包含する直方体で評価し、ボトル本体・首・キャップは円筒のまま分離軸で隙間の下限を求める。
下限が正なら交差しない。不確かな配置は経路として採用しない。接触力・摩擦・保持力は計算しない。

斜めと真上の2つのRVizを同じROS状態で表示し、両方で手先座標系 `tool0` のX・Y・Z軸を表示する。
1つの画面キャプチャを分岐して個別映像と横並び映像を作るので、2視点の各フレームは同期する。

記録先はこのパッケージの `outputs/bottle_grasp_distribution_comparison/`。
条件ごとに `axisymmetric/`、`single_pose/` へ保存する。既存の記録を上書きせず、
別の保存先は `--output /abs/path/new_directory` で指定する。録画を省く場合は `--no-video`。
録画にはX11の `DISPLAY` と `ffmpeg`、`ffprobe`、`xwininfo`、`xprop` が必要。
新しいスクリプトの初回利用時は `catkin build symaware_grasp --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3` でビルドする。

- `rviz_oblique.mp4`、`rviz_top.mp4`、`rviz_side_by_side.mp4`: 斜め・真上・同期した横並び映像
- `grasp_preview.png`、`grasp_oblique.png`、`grasp_top.png`: 最終場面
- `experiment.bag`: 物体・把持目標の分布、IKResult、関節状態・指の間隔・指令、TF、段階イベント
- `samples.csv`、`overview.png` / `.pdf`、`joint_states.png` / `.pdf`: 手先と物体中心の距離、軸の傾き、モードからの姿勢誤差、指の間隔、隙間の下限、関節角度
- `summary.json`: IKのコスト、干渉の有無、最小隙間、最終姿勢の評価と `kinematic_grasp_reached`
- `manifest.json`、`parameters.yaml`、URDF、`grasp_library.yaml`: 試験の設定とモデル

記録したbagから評価値とグラフを作り直す。

```bash
rosrun symaware_grasp run_bottle_grasp_demo.py analyze \
  "$(rospack find symaware_grasp)/outputs/bottle_grasp_distribution_comparison/single_pose" \
  --output /tmp/bottle_grasp_analysis
```

片方の条件だけを実行する場合は `run --distribution axisymmetric` または `run --distribution single_pose` を使う。

収集用の実行ではIK nodeの `publish_command` を `false` にし、IK結果を確認してから
接近経路を送る。通常の `symmetry_aware_ik_node.py` は従来どおり解いた姿勢を直接指令する。
動く指のURDFは元のモデルから試験時に生成する。アーム6軸とTCPの位置は維持する。

## アーム各リンクの分布表示

static arm link の point moments と pointcloud だけを確認する場合:

```bash
roslaunch symaware_grasp prob_tf_link_cloud.launch
```

この launch は RViz と 6 関節のスライダを起動する。スライダの `joint_states` はロボットの
通常 TF と revolute joint の native ProbTF v2 record の両方へ反映されるため、姿勢と
pointcloud は一緒に更新される。GUI を起動しない headless 実行は次のとおり。

```bash
roslaunch symaware_grasp prob_tf_link_cloud.launch rviz:=false
```

## 主要 topic

- `/probtf`: dynamic native v2 transform records
- `/probtf_static`: latched static native v2 transform set
- `/symaware_grasp/object_belief`: `ObjectBelief`
- `/symaware_grasp/grasp_targets`: `GraspTargetArray`
- `/symaware_grasp/symmetry_aware_ik_result`: pointwise IK の `IKResult`
- `/symaware_grasp/object_geometry`: cylinder の表示用 `Marker`
- `/symaware_grasp/link_pointcloud`: link axis endpoint pointcloud

application message は完全な `probtf_msgs/ProbabilisticTransformStamped` を内包する。runtime consumer
は `RosProbTfListener` の lookup API を使い、app message は用途 metadata と lookup trigger として扱う。

詳細は [lectures/symaware_grasp_demo.md](lectures/symaware_grasp_demo.md) と
[lectures/probTF.md](lectures/probTF.md) を参照してください。

## 箱の四成分混合分布と単一姿勢分布の比較

箱の姿勢を四成分のBingham混合分布で表す条件と、同じ基準モードの単一Bingham分布で表す条件を記録する。
箱の大きさは100 × 80 × 180 mm、中心は `[0.78, -0.12, 0.46]` m、真の姿勢のヨー角は `3π/8`（67.5°）とする。
三辺の長さが異なる箱の向きを保つ四つの正回転 `D2` に対応して、基準姿勢と物体座標系のX・Y・Z軸周りの180°回転を四つのモードにする。
四モードはすべて、真の箱と同じ八つの角を持つ。
混合条件の重みは各0.25、各成分の集中度は `[5000, 5000, 5000]`、単一姿勢条件は同じ基準モードと集中度を使う。

把持ライブラリには一つの候補を登録し、混合条件ではその一つの目標が四成分すべてを保持する。
成分ごとに別のIKを解いて候補を選ぶ処理は行わない。
物体の分布と把持目標は完全なnative ProbTF recordで受け渡し、表示用の真の箱はその代表成分から独立した固定姿勢の `Marker.CUBE` として配信する。

IKの向き項は、正規化したBingham混合密度の負の対数とする。
手先の単位四元数を `q_h(θ)`、各成分のパラメータ行列を `A_i`、正規化定数を `C(A_i)` とすると、

\[
L_{\mathrm{orientation}}(\theta) := -\log\!\left(\sum_{i=1}^{K} w_{i} \frac{\exp(q_{h}(\theta)^\top A_{i} q_{h}(\theta))}{C(A_{i})}\right), \qquad \sum_{i=1}^{K} w_{i} = 1.
\]

ここで `K=4` が混合条件、`K=1` が単一姿勢条件であり、四元数の成分順は `wxyz` とする。
この設定では `A_i = 5000 (q_i q_i^T - I_4)` となる。
実装ではlog-sum-expで評価し、Binghamの正規化定数を含める。
位置・移動・関節限界のコスト、数値最適化の手順、初期関節角、制御器、関節速度制限は既存のボトルデモの設定を維持する。
IKで得た姿勢を確認した後、同じ `plan_approach` と補間処理で上方を経由し、12 cm手前から直線接近して指を185 mmから106 mmへ閉じ、5秒間静止する。
URDFの各collision形状と固定した真の箱の間を、15本の分離軸による直方体同士の検査で確認する。
計画経路では2 mm、記録後の判定では0.5 mmの隙間の下限を要求する。
この実行は運動学による可視化であり、接触力・摩擦・保持力は計算しない。

```bash
cd ~/catkin_ws
catkin build symaware_grasp --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3
source ~/catkin_ws/devel/setup.bash
rosrun symaware_grasp run_box_grasp_demo.py compare \
  --object-yaw-pi 3/8 \
  --output /tmp/box_grasp_distribution_comparison
```

指定した保存先に `mixture/` と `single_pose/` を作り、条件ごとに独立したROS masterで順に実行する。
既存の保存先は上書きしない。
録画にはX11の `DISPLAY` と `ffmpeg`、`ffprobe`、`xwininfo`、`xprop` が必要で、録画を省く場合は `--no-video` を指定する。
一つのX11フレームを分岐して斜めと真上のRViz viewportを切り出すため、二視点は同期する。
録画へ見出しや進行字幕を追加せず、軸・物体形状・分布の点群をRVizの表示のまま保存する。
両条件に同じ物体姿勢と把持手順を使い、得られた解と記録された分布を比較する。
この例だけから把持性能の優位性は判断しない。

- `rviz_oblique.mp4`、`rviz_top.mp4`、`rviz_side_by_side.mp4`: 同期した斜め・真上・横並び映像
- `grasp_preview.png`、`grasp_oblique.png`、`grasp_top.png`: 映像から抽出した同時刻の最終場面
- `experiment.bag`: 物体・目標のnative分布、IKResult、関節状態、指の間隔・指令、TF、段階イベント
- `samples.csv`、`summary.json`、`events.json`: 手先と中心の距離、許容モードへの角度誤差、手首角度、指の間隔、箱に対する隙間の下限、最終保持区間の判定
- `manifest.json`、`parameters.yaml`、URDF、`grasp_library.yaml`、RViz設定: モデルと設定、ソースのSHA-256、publisherから照合・取得した実際の成分行列・重み・モード
- `comparison.json`: 両条件の評価結果

一条件だけの実行と記録後の再評価は次のように指定する。

```bash
source ~/catkin_ws/devel/setup.bash
rosrun symaware_grasp run_box_grasp_demo.py run \
  --distribution mixture --output /tmp/box_grasp_mixture
rosrun symaware_grasp run_box_grasp_demo.py analyze \
  /tmp/box_grasp_mixture --output /tmp/box_grasp_mixture_analysis
```

再評価ではbag内の目標が一件のまま混合成分を保持していることと、モードがmanifestの把持姿勢に対応することを確認する。
最終静止区間の手先位置・向き・指の間隔と、全記録中の箱との隙間を数値で検査し、`kinematic_grasp_reached` に結果を保存する。
解析結果と元の記録を共有する際は、映像・bagの容量を確認し、必要な記録を選択する。
