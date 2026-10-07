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
