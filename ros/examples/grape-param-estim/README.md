# grape_param_estim

失敗した Grape の rosbag から、記録済み actuator command と機体応答の間に
含まれる有用な情報だけを取り出す最小オフライン推定器です。

この package が行うことは一つだけです。

```text
failed rosbag
  -> recorded thrust + gimbal command を6軸のcommand wrenchへ変換
  -> IMU specific force / angular accelerationと時刻合わせ
  -> common alignment lagを探索
  -> 各軸の実効gain、velocity feedback、biasをrobust回帰
  -> block bootstrap 95%区間をJSONへ保存
```

## 非目標

- `jsk_aerial_robot` は変更しません。
- live controller、exact replay、closed-loop simulation は実装しません。
- controller gain の推薦や実機 parameter の書き換えは行いません。
- 未校正の command から physical mass / inertia を断定しません。
- 旧 pipeline との後方互換性は提供しません。

出力する gain は、固定した source geometry と記録 command の単位に条件づけた
`effective command-to-motion gain` です。95%区間も、指定区間と回帰modelに
条件づけた block-bootstrap interval であり、機体 parameter の客観的な
Bayesian posterior ではありません。また記録commandはclosed-loop内で生成
されているため、回帰係数とalignment lagは診断的な関連量であり、actuatorの
因果gainや純粋なtransport delayとは断定しません。

## 構成

```text
config/failure_estimator.yaml
scripts/estimate_failure_parameters.py
src/grape_param_estim/
  controller_sample.py
  effective_estimator.py
  failure_bag.py
```

`controller_sample.py` は、記録された4 rotor thrust と gimbal angle を
body wrench に変換する小さな参考実装と、独立した PID の例だけを持ちます。
実機 controller の複製や exact 実装ではありません。

## ビルド

```bash
cd /home/leus/catkin_ws
catkin build grape_param_estim --no-deps
source devel/setup.bash
```

## 最小実行例

bag 4 の失敗直前区間を推定する command は次だけです。ROS master や
`rosbag play` は不要です。

```bash
rosrun grape_param_estim estimate_failure_parameters.py \
  --bag /home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/20260612_grape_hovering_4_2026-06-12-17-33-59.bag \
  --output /tmp/grape_failure_04_parameters.json
```

既存 output を置換するときだけ `--force` を追加します。別の failure bag を
使う場合は `config/failure_estimator.yaml` をコピーし、bag SHA-256、
`start_offset_s`、`end_offset_s` を明示的に変更してください。

## 出力

一つの JSON に次を保存します。

- input bag、config、解析区間の SHA-256
- 選択された common alignment lag
- 各軸の response gain、velocity feedback、bias
- 各係数の block-bootstrap 95% interval
- RMSE、R²、input excitation、`informative / weak / not_excited` 判定

代表的な gain 名は次です。

- `specific_force_gain_x/y/z`
- `angular_acceleration_gain_roll/pitch/yaw`

`informative` は「この失敗区間と実効modelの中で符号と大きさに情報がある」
ことだけを意味します。physical mass / inertia と読み替えないでください。

## テスト

```bash
cd /home/leus/catkin_ws/src/ProbTF-demo
PYTHONPATH=ros/examples/grape-param-estim/src \
  /usr/bin/python3 -m unittest discover \
  -s tests/grape_param_estim -p 'test_*.py'
```
