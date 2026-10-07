# 干渉を避けたボトル把持：分布の比較

2026年10月8日に実行した2条件の記録。ボトルの位置、初期関節角、ボトル座標系、グリッパ、
経路の作り方と速度設定をそろえ、姿勢分布の集中度だけを変えた。
ボトル座標系はZ軸周りに `3π/8`（67.5°）回転させている。

| 条件 | Bingham集中度 | 最終手首曲げ角 | 記録中の最小隙間の下限 | 横並び動画 |
| --- | --- | --- | --- | --- |
| 軸対称 | `[5000, 5000, 0]` | 22.18° | 2.996 mm | [斜め・真上を同期再生](axisymmetric/rviz_side_by_side.mp4) |
| 単一姿勢 | `[5000, 5000, 5000]` | 92.37° | 2.992 mm | [斜め・真上を同期再生](single_pose/rviz_side_by_side.mp4) |

両条件とも手先座標系 `tool0` のX・Y・Z軸を表示している。
1つのX11キャプチャを2視点へ分岐し、個別動画と合成動画を同時にエンコードした。
解像度は各視点1168×860、横並び2336×860、20 fps。条件内で3動画のフレーム数は一致する。
軸対称は約23.1秒、単一姿勢は約23.4秒で、IK計算時間には差がある。

| 記録 | 軸対称 | 単一姿勢 |
| --- | --- | --- |
| 斜め動画 | [MP4](axisymmetric/rviz_oblique.mp4) | [MP4](single_pose/rviz_oblique.mp4) |
| 真上動画 | [MP4](axisymmetric/rviz_top.mp4) | [MP4](single_pose/rviz_top.mp4) |
| 最終場面 | [PNG](axisymmetric/grasp_preview.png) | [PNG](single_pose/grasp_preview.png) |
| rosbag | [bag](axisymmetric/experiment.bag) | [bag](single_pose/experiment.bag) |
| 測定値 | [CSV](axisymmetric/samples.csv) | [CSV](single_pose/samples.csv) |
| 距離・姿勢・指の間隔・隙間 | [グラフ](axisymmetric/overview.png) | [グラフ](single_pose/overview.png) |
| 関節角度 | [グラフ](axisymmetric/joint_states.png) | [グラフ](single_pose/joint_states.png) |
| 評価結果 | [JSON](axisymmetric/summary.json) | [JSON](single_pose/summary.json) |
| 設定・モデル | [manifest](axisymmetric/manifest.json) | [manifest](single_pose/manifest.json) |

両条件とも `kinematic_grasp_reached` と `collision_free` は `true`。
取付部が把持中心まで伸びていた干渉を修正し、上方を経由して12 cm手前へ移動してから直線接近した。
指の間隔は185 mmから106 mmへ閉じ、5秒間静止した。
隙間はURDFのcollision形状とボトル本体・首・キャップの円筒に対して求めた。
ロボットの円筒を包含する直方体で囲い、分離軸で評価するため、掲載値は保守的な下限である。
経路を補間して確認し、実行中とbagの各関節状態でも確認している。
接触力・摩擦・保持力は計算していない。

軸対称分布のモードは複数あるので、ボトル座標系の表示には指定した基準姿勢を使っている。
IKには各条件の分布の尤度をそのまま使う。単一姿勢ではZ軸周りにも集中度を持たせるため、
手首を大きく曲げて基準姿勢へ合わせる。

bagから解析を作り直す場合は、条件名を指定する。

```bash
source /home/leus/catkin_ws/devel/setup.bash
rosrun symaware_grasp run_bottle_grasp_demo.py analyze \
  "$(rospack find symaware_grasp)/outputs/bottle_grasp_distribution_comparison/single_pose" \
  --output /tmp/bottle_grasp_reanalysis
```

新しい2条件の記録は保存先を変えて実行する。

```bash
rosrun symaware_grasp run_bottle_grasp_demo.py compare \
  --object-yaw-pi 3/8 --output /tmp/bottle_grasp_next_comparison
```
