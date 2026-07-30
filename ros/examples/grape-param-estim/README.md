# grape_param_estim

Grape の ROS 1 bag から、軌道 replay に使う観測状態と記録 command を選ぶための
最小ツールです。旧 effective-gain estimator、report、Tk GUI との互換性は
ありません。

Phase 0 では次だけを実装しています。

- 指定 directory 以下の `.bag` scan
- command、gimbal、IMU、odometry、flight state の直接読込
- odometry 時刻への stream 整列
- 解析区間の選択と短時間 segment への分割
- 選択した全データの Plotly 表示
- GUI で設定した内容の YAML 保存

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

sidebar から bag directory、bag、topic、bag-local の解析開始・終了時刻、
segment 長を指定できます。`Save analysis YAML` は画面上の値を
`Save YAML path` へ保存します。

表示中の点線は segment 境界です。選択区間の large residual や接触後データを
自動的に除外しません。

## 設定

既定値は `config/default.yaml` にあります。

```yaml
schema: grape_param_estim/phase0
data:
  bag_directory: /path/to/bags
  bag_path: /path/to/example.bag
  topics:
    command: /gimbalrotor/four_axes/command
    gimbal: /gimbalrotor/gimbals_ctrl
    imu: /gimbalrotor/sensor_plugin/imu1/ros_converted
    odometry: /gimbalrotor/uav/cog/odom
    flight_state: /gimbalrotor/flight_state
analysis:
  start_time: 20.0
  end_time: 25.28
  segment_duration: 0.75
output:
  config_path: ~/.ros/grape_param_estim/analysis.yaml
```

## テスト

```bash
cd /home/leus/catkin_ws/src/ProbTF-demo
source /home/leus/catkin_ws/devel/setup.bash
PYTHONPATH=ros/examples/grape-param-estim/src \
  /usr/bin/python3 -m unittest discover \
  -s tests/grape_param_estim -p 'test_*.py'
```
