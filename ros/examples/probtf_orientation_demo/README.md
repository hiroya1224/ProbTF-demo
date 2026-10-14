# 角速度伝播と地磁気の三条件比較

`compare_rolling_bag.py`は、実測された角速度を一標本ずつ伝播し、地磁気の使い方を変えた三条件をモーションキャプチャと比較する。ROS masterを起動せず、bagを読み取り専用で開く。

| 条件 | 現在時刻の出力 | 次時刻へ保持する状態 |
| --- | --- | --- |
| 角速度のみ | 角速度による予測分布 | 同じ予測分布 |
| 地磁気を後処理で合成 | 角速度だけの予測と現在の地磁気因子の積 | 地磁気を加える前の予測分布 |
| 地磁気を次時刻へ帰還 | 予測と現在の地磁気因子の積 | 地磁気を加えた事後分布 |

加速度による重力方向の更新は使わない。計算には`probtf_estimators.orientation_imu`の6点cubature、`probtf.bingham`の四元数積の二次モーメントとBinghamモーメント整合、地磁気の方向因子を使う。モーメント整合の初期値には前時刻の固有値を使い、二次モーメント残差を検査する。deprecatedのパッケージは読み込まない。

ProbTF-demoのルートで次を実行する。Python環境にはROS Noeticの`rosbag`、NumPy、SciPy、Matplotlib、numpy-quaternionを、動画生成には`ffmpeg`を用意する。ビジュアライザは同梱の`third_party/BinghamNLL`から読む。

```bash
/usr/bin/python3 ros/examples/probtf_orientation_demo/scripts/compare_rolling_bag.py \
  --bag ~/rosbags/20250310_delta_linear_rolling_1_2025-03-10-16-18-36.bag \
  --output ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling
```

数値計算とグラフだけを作る場合は`--skip-video`を加える。保存済みの結果から動画を作り直す場合は同じ出力先と`--render-only`を指定する。短い区間の確認には`--duration 1.0`を使える。

入力は`/delta/imu`の`spinal/Imu`に含まれる`gyro_data`と`mag_data`、および`/delta/mocap/pose`である。各メッセージの計測時刻を使い、モーションキャプチャを四元数の球面線形補間で対応付ける。共通の記録範囲外へは外挿しない。角速度は右端標本を時刻間隔中の値として用い、すべての標本を伝播する。

冒頭の1.0秒で世界の磁場基準方向を決め、その後の最初の参照方位から三条件共通の事前分布を作る。モーションキャプチャはこの初期設定と評価にだけ使い、その後の時間再帰に入力しない。角速度のバイアスは差し引かない。全方向の磁気較正も追加しない。

| 設定 | 既定値 |
| --- | --- |
| 初期Bingham集中度 | 100.0 |
| 角速度標本の等方的な作業分散 | 0.0001 rad²/s² |
| 地磁気方向の集中度 | 20.0 |
| Bingham正規化定数の数値計算点数 | 120 |
| 動画の表示頻度 | 8 frames/s |
| 各分布の表示標本数 | 180 |

これらの尺度は三条件の比較に固定したものであり、参照誤差を最小化して選んだ値ではない。分布の被覆率を較正した値とも扱わない。

出力先に`summary.json`、`timeseries.npz`、`timeseries.csv`、RPY・四元数・回転誤差と磁場残差のPDF/SVG/PNG、`bingham_comparison.mp4`、動画中の5時刻のPNGを保存する。四元数の並びは`wxyz`で、符号は時刻方向に連続にそろえる。RPYの角度は参照時系列に最も近い枝へそろえる。この表示上の枝の選択は時間再帰へ影響しない。

動画では、既存の`bingham.visualize.SO3s.draw_so3s`でBingham標本の軸端点とモードを描く。黒破線の軸がモーションキャプチャであり、右端には参照方位だけを表示する。下段の回転誤差とカーソルを同じ時刻にそろえる。

2025年3月10日の記録では、初期設定後の85.446秒、17,088標本に対する回転誤差の二乗平均平方根は、角速度のみが7.35度、後処理の地磁気合成が6.15度、地磁気を次時刻へ帰還した場合が13.61度であった。初期基準方向に対する地磁気方向の残差は最大52.45度であり、この記録では帰還によって誤差が増えた。較正・環境磁場・取付・参照計測の影響をこの比較だけで個別に分離してはいない。

回帰検査は次で行なう。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 tests/probtf/test_rolling_orientation_comparison.py -v
```
