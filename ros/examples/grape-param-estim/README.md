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
  -> 記録 PID と controller model の識別不能リッジを保持
  -> 不確実性を考慮した PID / model の初回修正値を提案
  -> GUI で対話的に確認し、JSON を自動保存
```

## ビルド

```bash
cd /home/leus/catkin_ws
catkin build grape_param_estim --no-deps
source devel/setup.bash
```

## 対話 GUI（推奨）

ファイル名や出力先をコマンドラインへ入力せず、次の一つのコマンドで GUI を
起動できます。

```bash
rosrun grape_param_estim failure_analysis_gui.py
```

GUI の基本操作は次のとおりです。

1. `bag を追加…` で複数ファイルを選ぶか、`フォルダを追加…` でフォルダ直下の
   `.bag` をまとめて一覧へ加える。
2. 一覧の `解析` 欄をクリックし、今回使う bag だけを `[x]` にする。
   `未解析をすべて選択` と `選択を解除` も利用できる。
3. `解析を実行` を押し、実進捗率と `残り約` の時間表示を確認しながら待つ。
4. 完了後、右側の `PID・モデル提案`、時系列、実効ゲイン、推定値タブを確認する。

進捗は indeterminate animation ではありません。bag 内の読み込み位置、
SHA-256 計算量、alignment 探索、bootstrap、parameter trace の完了量を集約した
determinate progress です。ETA はファイルサイズから得る初期値を、実測経過時間で
解析中に補正します。

時系列とパラメータ推移には Matplotlib の対話 canvas を直接埋め込んでいます。
下部ツールバーから pan、矩形 zoom、表示範囲の戻る／進む、画像保存を行えます。
HTML への変換や browser の起動は必要ありません。

GUI を閉じずに別の bag を追加して再び `解析を実行` すると、解析済み bag は
再計算せず、新規 bag の結果だけを現在の試行時系列へ追加します。各 bag の解析が
終わるたび、累積結果を次へ自動保存します。

```text
~/.ros/grape_param_estim/failure_analysis/YYYYMMDD-HHMMSS/analysis.json
```

実際の保存先は GUI 左下へ常時表示され、`結果保存先を開く` から直接開けます。
bag の内容自体は変更しません。解析中に一つの bag が失敗しても、完了済みの結果は
保持され、エラーになった bag だけを次回の実行で再試行できます。

## 自動解析

GUI を使用しない batch 処理では、従来の CLI も利用できます。bag 4 を固定時刻の
指定なしで解析する例です。

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
  fit/diagnostic interval、推定値、累積 parameter 推移、PID / model 提案、
  識別不能リッジ
- `report.html`: 時系列、parameter 推移、残差、区間ラベル、PID / model 提案を
  含む外部 JavaScript 不要の browser report

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
- liftoff 後に支持面付近へ戻り、一定時間続いた `support_contact` 区間
- FORCE_LANDING `17`
- command heartbeat が欠けた区間
- episode 内の preliminary fit に対して大きな残差が持続する区間

接地 sample が posterior の中で自然に消えることは仮定しません。支持面相対高度で
明示的に fit から外した上で、失敗診断区間として残します。episode 直前の非制御
静止区間が bag にない場合だけ、controller-active episode 冒頭の静止区間を
支持面推定へ使い、その出所も `support.source` に記録します。

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

## PID・controller model 提案の解釈

bag 内の `/gimbalrotor/debug/pose/pid` から各軸の total、P、I、D term を、
4個の dynamic-reconfigure `parameter_updates` topic から現在の
`xy / z / roll_pitch / yaw` gain を読みます。PID total は選択済み alignment
lag だけ前の時刻へ合わせ、並進成分を odometry 姿勢で body / CoG 座標へ変換して
自由飛行 sample だけを使います。

観測から直接得る応答倍率を `r`、actuator scale を `alpha`、物理 parameter の
controller model に対する比を `rho` とすると、一意に識別できるのは次の比です。

```text
r = alpha / rho
```

したがって質量または慣性と actuator scale を一点へ潰さず、複数の `rho` に対する
`alpha` と block-bootstrap 95% 区間をリッジとして JSON と GUI に保持します。
記録 command wrench と PID desired acceleration の回帰から、現在 controller が
使った mass / inertia 相当値も復元します。これは source geometry 上の
controller model 相当値であり、独立に校正した真の質量・慣性ではありません。

PID の P/I/D 比は bag だけから個別には決め直せないため、3項を同じ倍率
`s_pid` で動かします。controller model 倍率を `s_model` として、
`r * s_pid * s_model = 1` を満たす中で両倍率の対数変更量が等しくなる点を
初回候補とし、各倍率を一度に `0.8` から `1.2` までへ制限します。
95%区間が1を含む場合は `現状維持`、回帰または PID term の根拠が弱い場合は
`根拠不足` とし、変更値を勧めません。PID の feedforward は変更せず、記録 total
に対して無視できない場合も根拠を弱めます。

これらは閉ループ log から得た「次に試す一手」であり、自動 tuning や安定性保証
ではありません。値は機体へ書き込まれません。係留・安全設備下で一項目ずつ変更し、
新しい bag で再評価してください。

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
  failure_analysis_gui.py
src/grape_param_estim/
  analysis_session.py
  automatic_analysis.py
  browser_report.py
  controller_advice.py
  controller_sample.py
  effective_estimator.py
  episode_detection.py
  failure_analysis_gui.py
  failure_bag.py
  interactive_plots.py
```

原理と仮定の詳細は次に記録しています。

- `lectures/2026-07-30_automatic_failure_episode_analysis.md`
- `lectures/2026-07-30_pid_ridge_tuning_advice.md`

## テスト

```bash
cd /home/leus/catkin_ws/src/ProbTF-demo
source /home/leus/catkin_ws/devel/setup.bash
PYTHONPATH=ros/examples/grape-param-estim/src \
  /usr/bin/python3 -m unittest discover \
  -s tests/grape_param_estim -p 'test_*.py'
```
