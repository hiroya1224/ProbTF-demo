# 角速度・加速度・地磁気による5条件の方位比較

`compare_rolling_bag.py`は、実測の角速度・加速度・地磁気を用いる5条件をモーションキャプチャと比較する。ROS masterを起動せず、bagを読み取り専用で開く。

| 条件 | 時間発展 | 観測の融合 |
| --- | --- | --- |
| 角速度の時間発展のみ | 角速度による予測 | なし |
| 角速度と加速度の融合 | 前時刻の事後分布を角速度で予測 | 現在の加速度方向 |
| 角速度と地磁気の融合 | 前時刻の事後分布を角速度で予測 | 現在の地磁気方向 |
| 地磁気と加速度の融合 | なし。各時刻に一様事前分布から構成 | 現在の加速度方向と地磁気方向 |
| 9軸の全情報の融合 | 前時刻の事後分布を角速度で予測 | 現在の加速度方向と地磁気方向 |

候補の密度を掛ける処理を「融合」と呼び、四元数積による回転の「合成」と区別する。条件名と順序は上記に統一する。角速度を使う観測融合の3条件では、更新した事後分布を次時刻の予測に用いる。加速度・地磁気だけの条件は、現在の2方向だけで構成し、過去の観測や初期方位を蓄積しない。

ProbTF-demoのルートで次を実行する。Python環境にはROS Noeticの`rosbag`、NumPy、SciPy、Matplotlib、numpy-quaternionを、動画生成には`ffmpeg`を用意する。既存のビジュアライザは同梱の`third_party/BinghamNLL`から読む。

```bash
/usr/bin/python3 ros/examples/probtf_orientation_demo/scripts/compare_rolling_bag.py \
  --bag ~/rosbags/20250310_delta_linear_rolling_1_2025-03-10-16-18-36.bag \
  --output ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_five_conditions
```

数値計算とグラフだけを作る場合は`--skip-video`を加える。保存済みの5条件の結果から図や動画を作り直す場合は`--render-only`を指定する。静的な図だけを作り直す場合は`--render-only --skip-video`を指定する。短い確認には`--duration 1.0`を使える。以前の3条件の記録にはこの版を使わず、gitの`c99fd7d`にある計算版を用いる。

入力は`/delta/imu`の`gyro_data`、`acc_data`、`mag_data`と、`/delta/mocap/pose`である。各メッセージの計測時刻を使い、モーションキャプチャを四元数の球面線形補間で対応付ける。共通の記録範囲外へ外挿しない。角速度は右端標本を時刻間隔中の値として用い、すべての標本を伝播する。

冒頭の1.0秒で世界の磁場基準方向を決め、その後の最初の参照方位で角速度を使う4条件の共通の事前分布を作る。モーションキャプチャはこの初期設定と評価にだけ使い、その後の時間再帰に入力しない。「地磁気と加速度の融合」は各時刻に一様事前分布から構成し、初期方位の事前分布を加えない。

加速度と地磁気は単位方向へ正規化する。加速度は静止時の比力がmocapのworld座標系の+z方向に対応するモデルを用いる。これは加速度の向きを重力方向の観測として利用する近似であり、運動中の並進加速度は分離していない。ノルムによる除外・重み変更、平滑化、バイアス除去、全方向の磁気較正は加えない。この仮定は動的な転がり動作では常に成立するとは限らない。

| 設定 | 既定値 |
| --- | --- |
| 時間再帰の初期Bingham集中度 | 100.0 |
| 角速度標本の等方的な作業分散 | 0.0001 rad²/s² |
| 加速度方向の集中度 | 20.0 |
| 地磁気方向の集中度 | 20.0 |
| Bingham正規化定数の数値計算点数 | 120 |
| 動画の表示頻度 | 8 frames/s |
| 各分布の表示標本数 | 180 |

これらの尺度は5条件の比較中に固定する。参照誤差を最小化して選んだ値ではなく、分布の被覆率を較正した値でもない。6点積分、四元数積の2次モーメント、Binghamモーメント整合と2種類の方向因子はProbTFの公開関数を使う。deprecatedのパッケージは変更も読み込みも行なわない。

出力先に`summary.json`、`timeseries.npz`、`timeseries.csv`、以下のPDF/SVG/PNG、`bingham_comparison.mp4`、動画中の5時刻のPNGを保存する。

| 図 | 内容 |
| --- | --- |
| `rpy` | 5条件のロール・ピッチ・ヨーとmocap |
| `quaternion` | 5条件の四元数成分とmocap |
| `axis_errors` | ロール・ピッチ・ヨー各軸の誤差と3軸の合成誤差の4段 |
| `axis_errors_semilog` | 同じ4段の図で、縦軸だけを底10の対数に変更 |

誤差図では各段に5条件を重ねる。各軸は推定角と参照角の差を±180度へ折り返した絶対値、3軸の合成誤差は参照と推定の相対回転角とする。ピッチが±90度に近いとロール・ヨーの差が大きくなるため、下段と併せて読む。片対数図では0の標本を表示せず、微小な正数へ置き換えない。4段の構成、条件の色、時刻範囲は線形図と同じである。

四元数はwxyzの順で隣接符号を連続にそろえる。RPYは参照時系列に最も近い角度の枝を選ぶ表示であり、推定の時間再帰に参照を入力しない。各軸の誤差は`rpy_error_deg`、3軸の合成誤差は`error_deg`、入力加速度は`acceleration_m_s2`へ保存する。統計は各条件の`statistics`に保存する。`moment_fit_residual`の4列は`propagation_case_ids`に記録した角速度を使う条件に対応する。

動画は上段・中段の3列に5条件とmocapを順に並べ、下段に3軸の合成誤差を表示する。既存の`bingham.visualize.SO3s.draw_so3s`でBingham標本の軸端点とモードを描き、黒破線をmocapの参照方位とする。時系列とカーソルを同じ時刻にそろえる。

回帰検査は次で行なう。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 tests/probtf/test_rolling_orientation_comparison.py -v
```

2025年3月10日の記録の85.446秒、17,088時刻に対する二乗平均平方根を度で示す。

| 条件 | ロール | ピッチ | ヨー | 3軸の合成誤差 |
| --- | ---: | ---: | ---: | ---: |
| 角速度の時間発展のみ | 10.06 | 6.25 | 9.48 | 7.35 |
| 角速度と加速度の融合 | 6.99 | 2.86 | 6.00 | 4.91 |
| 角速度と地磁気の融合 | 16.02 | 5.00 | 11.12 | 13.61 |
| 地磁気と加速度の融合 | 13.34 | 5.83 | 31.08 | 32.82 |
| 9軸の全情報の融合 | 6.48 | 3.00 | 18.57 | 20.39 |

この固定設定では「角速度と加速度の融合」で方位全体の誤差が小さくなった。地磁気を含む条件で誤差が大きくなり、9軸の利用によって精度が一律に改善する結果ではない。較正・座標系の対応・動的加速度・観測尺度と時間相関の影響は個別に分離していない。

[5条件の実測比較と最終精査](../../../docs/reports/orientation-rolling-five-conditions_2026-10-15.md)に、数値の照合、図・動画の確認と残る検証事項を記録した。

## 9軸の全情報の融合とMadgwick（9軸）の比較

角速度・加速度・地磁気をすべて使う「9軸の全情報の融合」と、固定係数beta=0.1のMadgwick（9軸）を比較する。図は図5.3・5.4と同じ4段の構成で、各段にこの2系列だけを重ねる。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 ros/examples/probtf_orientation_demo/scripts/compare_madgwick_rolling.py \
  --render-only \
  --input ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_madgwick \
  --output ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_nine_axis_madgwick
```

[線形図](outputs/20250310_delta_linear_rolling_nine_axis_madgwick/method_comparison.pdf)と[片対数図](outputs/20250310_delta_linear_rolling_nine_axis_madgwick/method_comparison_semilog.pdf)は、上からロール・ピッチ・ヨーの各軸と3軸の合成誤差を示す。片対数版は同じ配置で縦軸だけを対数にする。PDF/SVG/PNG、集計、2系列の時系列を保存する。

`--render-only`は保存済みの9軸の2系列を取り出し、推定値を保持して図を作る。新たにMadgwickを計算する場合はこの指定を省き、`--input`に既存の5条件の結果を指定する。既定の入力先はその5条件である。磁北とmocapの座標対応は冒頭の磁場基準だけから固定する。

3軸の合成誤差のRMSは「9軸の全情報の融合」が20.39度、Madgwick（9軸）が28.17度である。[比較条件と精査](../../../docs/reports/orientation-rolling-nine-axis-madgwick_2026-10-15.md)を参照。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 tests/probtf/test_madgwick_rolling.py -v
```
