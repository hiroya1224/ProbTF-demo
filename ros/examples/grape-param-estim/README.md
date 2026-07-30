# grape_param_estim

Grape の ROS 1 bag に記録された command で、観測状態から始まる短区間の
剛体運動を replay する最小ツールです。旧 effective-gain estimator、report、
Tk GUI との互換性はありません。

Phase 0–1 では次を実装しています。

- 指定 directory 以下の `.bag` scan
- thrust command、gimbal target / measured joint、IMU、CoG / baselink
  odometry、flight state の直接読込
- CoG odometry 時刻への stream 整列
- bag 全体の位置・姿勢・速度・flight state overview からの解析区間選択
- 解析区間の短時間 segment への分割
- 選択した全データの Plotly 表示
- GUI で設定した内容の YAML 保存
- 4 rotor thrust と gimbal angle から body wrench への変換
- nominal mass / inertia を持つ 6-DoF 剛体 model
- 各 segment の観測初期状態からの open-loop replay
- observed / nominal pose、`T_nominal^-1 T_observed`、segment 別
  `SE(3)` residual の Plotly 表示
- observed trajectory だけを基準に centre / scale を決める 3-D view

controller は再実行しません。particle 推定、parameter posterior、次回 parameter
提案もこの phase には含みません。

## 準備

ROS package を build し、Streamlit と Plotly を user environment へ入れます。

```bash
cd /home/leus/catkin_ws
catkin build grape_param_estim --no-deps
source devel/setup.bash
python3 -m pip install --user \
  -r src/ProbTF-demo/ros/examples/grape-param-estim/requirements.txt
```

## GUI

```bash
source /home/leus/catkin_ws/devel/setup.bash
rosrun grape_param_estim grape_param_estim_app.py
```

別の初期設定を読み込む場合は `--config` を指定します。

```bash
rosrun grape_param_estim grape_param_estim_app.py \
  --config ~/.ros/grape_param_estim/analysis.yaml
```

sidebar で bag の検索 root と file、topic、segment 長、nominal mass、対角
inertia を指定できます。main view の whole-bag overview 上を横方向に box
select するか、その直下の range slider を動かして bag-local の解析区間を
指定します。bag を切り替えたときは、airborne かつ低速な区間が初期候補に
なります。`Save analysis YAML` は画面上の値を `Save YAML path` へ保存します。

`Selected data` tab は推定へ渡す native data、`Nominal replay` tab は observed /
nominal pose、補正変換、segment-local residual を表示します。表示中の点線は
segment 境界です。選択区間の large residual や接触後データを自動的に
除外しません。

3-D plot は記録軌道を中心に同じ physical scale の立方体を作ります。nominal が
枠外へ出た場合は mouse wheel で zoom out できます。このため nominal の発散で
重要な記録軌道が極端に小さくなることはありません。

各 segment の nominal state は、先頭の観測 `(p, R, v, omega)` へ reset されます。
表示する residual は
`Log(T_observed,relative^-1 T_nominal,relative)`、補正変換は
`T_nominal^-1 T_observed` です。

replay の body frame は、原点を `/uav/cog/odom` の実測 CoG、軸を
`/uav/baselink/odom` の物理 baselink に合わせた frame です。以前の実装は、
姿勢に controller の可変 desired-CoG frame を使いながら、wrench と gyro を
baselink frame のまま積分していました。desired coordinate を変更した bag で
nominal force が横向きになる原因だったため、この組合せは使用しません。
gimbal target は表示用、剛体への入力方向には `/joint_states` の実測角を使います。

## 設定

既定値は `config/default.yaml` にあります。

```yaml
schema: grape_param_estim/phase1
data:
  bag_directory: /path/to/bags
  bag_path: /path/to/example.bag
  topics:
    thrust_command: /gimbalrotor/four_axes/command
    gimbal_command: /gimbalrotor/gimbals_ctrl
    gimbal_state: /gimbalrotor/joint_states
    imu: /gimbalrotor/sensor_plugin/imu1/ros_converted
    cog_odometry: /gimbalrotor/uav/cog/odom
    body_odometry: /gimbalrotor/uav/baselink/odom
    flight_state: /gimbalrotor/flight_state
analysis:
  start_time: 57.466
  end_time: 62.747
  segment_duration: 0.75
model:
  nominal:
    mass: 2.351557590812377
    inertia_diagonal: [0.0649940671, 0.0649466618, 0.1289801290]
    force_scale: 1.0
    torque_scale: 1.0
output:
  config_path: ~/.ros/grape_param_estim/analysis.yaml
```

既定 mass / inertia は現在 checkout の Grape URDF を zero-joint で集約した値です。
model は固定 CoG、固定 inertia、剛体、無遅延、無減衰を仮定します。実 bag と
現在 checkout の機体構成が異なる場合は GUI で nominal 値を変更してください。

PID で安定飛行していることは、記録 command を nominal open-loop plant に入れた
軌道が一致することを意味しません。feedback command には、実 CoG、actuator
scale、外力など nominal に無い差を打ち消す成分も含まれるためです。Phase 1 は
座標系と入力を物理的に一貫させ、その残差を見える状態にするところまでです。
mass / force 比、inertia / torque 比、不定な CoG 変換を posterior として推定
するのは Phase 2 です。

## テスト

```bash
cd /home/leus/catkin_ws/src/ProbTF-demo
source /home/leus/catkin_ws/devel/setup.bash
PYTHONPATH=ros/examples/grape-param-estim/src:${PYTHONPATH} \
  /usr/bin/python3 -m unittest discover \
  -s tests/grape_param_estim -p 'test_*.py'
```
