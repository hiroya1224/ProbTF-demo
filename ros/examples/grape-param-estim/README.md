# grape_param_estim

失敗を含む Grape の ROS 1 bag から、飛行制御が有効な episode と liftoff を
自動検出し、実効 command-response parameter と失敗診断区間を抽出する
オフライン解析 package です。ROS master や `rosbag play` は不要です。

```text
one or more rosbag files
  -> flight_state から controller-active episode を抽出
  -> episode 直前の静止 z から支持面を推定
  -> 相対高度と上昇継続から liftoff を検出
  -> command / gimbal / IMU / odometry を時刻合わせ
  -> fit_mask と failure_diagnostic_mask を分離
  -> effective gain / velocity feedback / bias を robust 回帰
  -> JSON と自己完結 browser report を生成
```

## ビルド

```bash
cd /home/leus/catkin_ws
catkin build grape_param_estim --no-deps
source devel/setup.bash
```

## 自動解析

bag 4 を固定時刻の指定なしで解析する例です。

```bash
rosrun grape_param_estim analyze_failure_bags.py \
  --bag /home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/20260612_grape_hovering_4_2026-06-12-17-33-59.bag \
  --output-dir /tmp/grape_failure_auto
```

複数 bag は試行順に一度に指定できます。

```bash
rosrun grape_param_estim analyze_failure_bags.py \
  --bag \
    /path/to/trial_01.bag \
    /path/to/trial_02.bag \
    /path/to/trial_03.bag \
  --output-dir /tmp/grape_failure_trials
```

出力は次の2ファイルです。

- `analysis.json`: bag SHA-256、自動抽出 episode、支持面、liftoff、
  fit/diagnostic interval、推定値、累積 parameter 推移
- `report.html`: 時系列、parameter 推移、残差、区間ラベルを含む
  外部 JavaScript 不要の browser report

`report.html` は通常の browser で直接開けます。高度、速度、IMU specific
force、angular rate、vertical command、flight state を同じ時刻軸に表示します。
青い領域が parameter fit、赤い領域が失敗診断、緑の点線が liftoff です。
既存出力を置換するときだけ `--force` を追加します。

## 自動抽出の意味

`config/automatic_failure_analysis.yaml` の既定値では、
TAKEOFF `3`、LAND `4`、HOVER `5`、FORCE_LANDING `17` を
controller-active state とします。ただし state だけで空中とはみなしません。

各 active episode の直前3秒から静止 sample を選び、その z 中央値をその試行の
支持面とします。支持面からの相対上昇が `0.02 m` 以上となり、静止時ノイズに
対して有意な上昇が一定時間続いた点を liftoff とします。このため、台や mocap
原点が変わっても絶対 z には依存しません。

次の sample は parameter fit から分離し、JSON と GUI には残します。

- liftoff 前の controller-active / supported 区間
- FORCE_LANDING `17`
- command heartbeat が欠けた区間
- episode 内の preliminary fit に対して大きな残差が持続する区間

自由飛行の有効 sample が不足する episode は、無理に係数を返さず
`not_identifiable` と記録します。一つの bag に複数 episode があれば個別に
抽出します。

複数 bag は GUI の `sequence_time` 上で連続した試行として並びますが、初期版は
異なる実験の parameter を暗黙に一つへ同化しません。各 episode を独立に fit
するため、実験条件が異なる場合にも過剰な共通性を仮定しません。

## 推定値の解釈

記録された4 rotor thrust と gimbal angle は、固定した source geometry により
6軸 command wrench へ変換されます。common alignment lag を探索した後、各軸で
次の実効 model を Huber 回帰します。

```text
response = bias + effective_gain * command + velocity_feedback * state
```

並進 response は IMU specific force、回転 response は gyro から得た角加速度
です。最終係数の区間は moving-block bootstrap、時間推移は選択済み sample を
順次加えた cumulative robust fit です。

出力する量は `effective command-to-motion gain` です。command の物理単位が
未校正で、記録 command は closed loop 内で生成されているため、physical mass、
inertia、因果的 actuator gain、純粋な transport delay とは断定できません。
block-bootstrap 95%区間も Bayesian posterior ではありません。

代表的な parameter 名は次です。

- `specific_force_{x,y,z}_gain`
- `angular_acceleration_{roll,pitch,yaw}_gain`
- 各軸の `_velocity_feedback` と `_bias`

`informative` は、その episode と実効 model の中で gain の符号と大きさに
情報があることだけを意味します。

## 固定区間の最小 CLI

比較・監査用として、従来の単一 bag / 固定区間 CLI も残しています。

```bash
rosrun grape_param_estim estimate_failure_parameters.py \
  --bag /path/to/failure.bag \
  --config /path/to/failure_estimator.yaml \
  --output /tmp/failure_parameters.json
```

こちらは `failure_estimator.yaml` に bag SHA-256、`start_offset_s`、
`end_offset_s` を明示する方式です。通常の新規解析には自動 CLI を使用します。

## 構成

```text
config/
  automatic_failure_analysis.yaml
  failure_estimator.yaml
scripts/
  analyze_failure_bags.py
  estimate_failure_parameters.py
src/grape_param_estim/
  automatic_analysis.py
  browser_report.py
  controller_sample.py
  effective_estimator.py
  episode_detection.py
  failure_bag.py
```

原理と仮定の詳細は `lectures/2026-07-30_automatic_failure_episode_analysis.md`
に記録しています。

## テスト

```bash
cd /home/leus/catkin_ws/src/ProbTF-demo
source /home/leus/catkin_ws/devel/setup.bash
PYTHONPATH=ros/examples/grape-param-estim/src \
  /usr/bin/python3 -m unittest discover \
  -s tests/grape_param_estim -p 'test_*.py'
```
