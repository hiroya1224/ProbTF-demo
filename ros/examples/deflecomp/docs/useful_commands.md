```
roslaunch deflecomp_ros deflecomp_frames.launch model:=/home/leus/catkin_ws/src/nejineji-urdfs/yamaguchi_arm_nejineji/urdf/yamaguchi_6axis_arm_nejineji.urdf imu_config:=/home/leus/catkin_ws/src/nejineji-urdfs/yamaguchi_arm_nejineji/config/deflecomp_imu_frames.yaml  viewer:=true
```

```
rosrun deflecomp_sim apply_frame_load.py _frame:=module5_gripper_dummy_link _mass_kg:=0.5
```

RViz の `Applied External Force` は `base_link` 座標で表示される。既定の
payload force は作用点を始点として world -Z 方向を向く。
