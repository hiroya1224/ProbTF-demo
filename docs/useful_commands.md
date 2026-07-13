# 実行コマンド集

- 軸対称 cylinder の pointwise Bingham IK（2端末で起動）
```
roslaunch symaware_grasp probabilistic_tf_demo.launch
```
```
rosrun symaware_grasp symmetry_aware_ik_node.py
```

- 誤差が伝播している様子のビジュアライズ
```
roslaunch symaware_grasp prob_tf_link_cloud.launch
```

- たわみ補償の rviz launch
```
roslaunch deflecomp_ros deflecomp_frames.launch model:=/home/leus/catkin_ws/src/nejineji-urdfs/yamaguchi_arm_nejineji/urdf/yamaguchi_6axis_arm_nejineji.urdf imu_config:=/home/leus/catkin_ws/src/nejineji-urdfs/yamaguchi_arm_nejineji/config/deflecomp_imu_frames.yaml  viewer:=true
```

  - 手先に重力方向に力を加える
    ```
    rosrun deflecomp_sim apply_frame_load.py _frame:=module5_gripper_dummy_link _mass_kg:=0.5
    ```
