# symaware_grasp

Native ProbTF v2 の component/mixture semantics を保持した symmetry-aware grasp demo です。

## 実行

```bash
roslaunch symaware_grasp probabilistic_tf_demo.launch
```

static arm link の point moments と pointcloud だけを確認する場合:

```bash
roslaunch symaware_grasp prob_tf_link_cloud.launch
```

## 主要 topic

- `/probtf`: dynamic native v2 transform records
- `/probtf_static`: latched static native v2 transform set
- `/symaware_grasp/object_belief`: `ObjectBelief`
- `/symaware_grasp/hand_belief`: `HandBelief`
- `/symaware_grasp/grasp_targets`: `GraspTargetArray`
- `/symaware_grasp/selected_target`: `SelectedGraspTarget`
- `/symaware_grasp/link_pointcloud`: link axis endpoint pointcloud

application message は完全な `probtf_msgs/ProbabilisticTransformStamped` を内包する。runtime consumer
は `RosProbTfListener` の lookup API を使い、app message は用途 metadata と lookup trigger として扱う。

詳細は [lectures/symaware_grasp_demo.md](lectures/symaware_grasp_demo.md) と
[lectures/probTF.md](lectures/probTF.md) を参照してください。
