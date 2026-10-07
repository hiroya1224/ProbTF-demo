```
roslaunch deflecomp_ros deflecomp_frames.launch model:=/home/leus/catkin_ws/src/nejineji-urdfs/yamaguchi_arm_nejineji/urdf/yamaguchi_6axis_arm_nejineji.urdf imu_config:=/home/leus/catkin_ws/src/nejineji-urdfs/yamaguchi_arm_nejineji/config/deflecomp_imu_frames.yaml  viewer:=true
```

```
rosrun deflecomp_sim apply_frame_load.py _frame:=module5_gripper_dummy_link _mass_kg:=0.5
```

RViz の `Applied External Force` は `base_link` 座標で表示される。既定の
payload force は作用点を始点として world -Z 方向を向く。

姿勢変更・静止・荷重追加・除去を自動実行して、RViz動画、rosbag、解析グラフを残す。

```bash
rosrun deflecomp_debug run_convergence_demo.py run --live-plot
```

ゼロ姿勢→肩−π/3→肘π/5と動かし、荷重の追加・予測収束待ち・解除を録画する。

```bash
rosrun deflecomp_debug run_convergence_demo.py run \
  --scenario "$(rospack find deflecomp_debug)/config/tip_load_step.yaml"
```

記録済みのbagから解析結果を作り直す。

```bash
rosrun deflecomp_debug run_convergence_demo.py analyze /abs/path/to/experiment_directory
```

保存済みの成功例 [tip_load_prediction_recovery](../outputs/tip_load_prediction_recovery/README.md) を再解析する。

```bash
rosrun deflecomp_debug run_convergence_demo.py analyze \
  "$(rospack find deflecomp_debug)/../outputs/tip_load_prediction_recovery" \
  --output /tmp/tip_load_prediction_recovery_analysis
```
