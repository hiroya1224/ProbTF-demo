# 手先荷重の追加・解除と予測の再収束

山口アームを曲げて手先に下向きの荷重を加え、予測が落ち着いたら荷重を解除した成功例。
2026年10月8日（日本時間）に記録した。録画は約24秒、1500×950、20 fps。

- [RViz動画](rviz.mp4)
- [荷重をかけている場面](load_preview.png) / [解除後の場面](recovery_preview.png)
- [予測・追従誤差のグラフ](overview.png)
- [関節角度のグラフ](joint_states.png) / [有効剛性のグラフ](stiffness.png)
- [rosbag](experiment.bag) / [評価値CSV](samples.csv) / [判定結果](summary.json)

## 動作と荷重

関節の順序は `joint1`、`module1_joint1`、`module2_joint1`、`module3_joint1`、
`module4_joint1`、`module4_joint2`、`module5_gripper_dummy_joint`。
元の指定値−1.0 radを−π/3、0.6 radをπ/5に調整した。グリッパ角度は0に固定した。

| 開始時刻 | 動作 | 目標角度 [rad] |
| --- | --- | --- |
| 0秒 | ゼロ姿勢で1秒静止 | `(0, 0, 0, 0, 0, 0, 0)` |
| 1秒 | 0.5秒で肩を曲げ、1秒静止 | `(0, -π/3, 0, 0, 0, 0, 0)` |
| 2.5秒 | 0.5秒で肘を曲げ、1秒静止 | `(0, -π/3, π/5, 0, 0, 0, 0)` |
| 約4秒 | 手先に荷重を加え、予測の収束まで静止 | 同じ姿勢 |
| 約13.49秒 | 荷重を解除し、予測の再収束まで静止 | 同じ姿勢 |
| 約22.97秒 | 観測終了 | 同じ姿勢 |

荷重は `module5_gripper_dummy_link` に加えた0.5 kg相当の力（world −Z方向へ4.905 N）。
RVizは曲げる平面を横から見る画角。緑の球が目標、ピンクが実際の手先、青が予測位置。
`Cmd` のロボット表示はばね基準角であり、物理姿勢は `Equil` で確認する。

収束の目安は、関節RMS予測誤差が0.5度以下、手先位置予測誤差が5 mm以下の状態が
3秒続くこと。オンライン判定では記録のサンプル時刻との差をカバーする0.5秒も加えた。
荷重追加・解除後はそれぞれ最低5秒観測し、追加後180秒、解除後90秒を待機上限とした。
この記録ではどちらも上限前に収束条件を満たした。1秒の静止段階は全体の収束判定に含めない。

## 結果

bagから再計算した両荷重段階の `prediction_stable_at_end` は `true`。

| 段階 | 収束確認までの待機時間 | 終盤の関節RMS予測誤差 | 終盤の手先位置予測誤差 |
| --- | --- | --- | --- |
| 荷重追加後 | 約9.48秒 | 0.058度 | 0.747 mm |
| 荷重解除後 | 約9.48秒 | 0.139度 | 0.590 mm |

終盤の誤差は、`summary.json` の `tail_median_*`、すなわち最終評価窓の中央値。
予測精度と剛性値の安定は別々に判定する。荷重解除後も予測は条件を満たすが、
剛性の更新は続いている。`K_est` は外乱を吸収した有効剛性であり、材料剛性の同定値ではない。
実姿勢や真の剛性は評価専用で、推定器への入力には使っていない。
時間軸はROSの記録時刻であり、simulator内部の積分時間とは一致を保証しない。

## 再解析

catkin workspaceの `devel/setup.bash` をsourceした環境で実行する。
必要なパッケージのビルド方法は [deflecompのREADME](../../README.md) を参照。
再解析には同封のbag、URDF、パラメータを使い、録画時の絶対パスにあるモデルを読み直す必要はない。

```bash
tip_load_recording="$(rospack find deflecomp_debug)/../outputs/tip_load_prediction_recovery"
rosrun deflecomp_debug run_convergence_demo.py analyze "$tip_load_recording" \
  --output /tmp/tip_load_prediction_recovery_analysis
```

`samples.csv`、`estimation_gates.csv`、`events.json`、`summary.json`、3種類のグラフのPNG/PDFを
出力先に作り直す。`--require-convergence` は剛性値の安定も含む判定なので、この記録では終了code 2になる。
予測の収束は `summary.json` の `all_predictions_stable_at_end` と各段階の
`prediction_stable_at_end` で確認できる。

## 同じ動作を再実行

```bash
rosrun deflecomp_debug run_convergence_demo.py run \
  --scenario "$(rospack find deflecomp_debug)/../outputs/tip_load_prediction_recovery/scenario.yaml"
```

新しい動画・bagは別の保存先に作成される。モデルとIMU設定はworkspaceの山口アームを使う。
別の配置では `--model` と `--imu-config` を指定する。

`manifest.json` は実行時刻・録画情報・実際に使ったシナリオ、`parameters.yaml` はROSパラメータ、
`model.urdf` と `imu_frames.yaml` はモデルとIMU設定のsnapshot。
`events.jsonl` とbag内の `/deflecomp_demo/event` に、荷重変更と収束確認の時刻・理由を残している。
