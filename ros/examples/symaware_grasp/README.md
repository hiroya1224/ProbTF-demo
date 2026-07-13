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
