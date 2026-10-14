# Madgwickフィルタによる実測比較と地磁気入力の確認

同じ2025年3月10日の転がり動作に、Madgwickの6軸版と9軸版を適用した。既存の5条件の推定結果と図は保持し、新しい7系列の図と、地磁気の有無を分けた4段2列の比較図を追加した。本文5.3.3に比較条件、結果の表、線形・片対数の比較図、地磁気由来のヨーを確認する図と説明を追加した。

## 基本形と評価条件

[Madgwickの2010年報告](https://x-io.co.uk/downloads/madgwick_internal_report.pdf)と、著者が公開する[MATLAB実装](https://x-io.co.uk/downloads/madgwick_algorithm_matlab.zip)の正規化勾配の式に基づく。基本形は、角速度による四元数の微分から観測方向の二乗残差の勾配を正規化した項を引き、Forward Eulerで積分して四元数を正規化する。9軸版の磁場基準は、前時刻の推定方位で現在の磁場を写した水平成分の大きさと鉛直成分から構成する。

主比較の固定係数はbeta=0.1とした。この値は著者の公開MATLAB例とC版の既定値にある。バイアスの適応推定、入力の除外、平滑化、係数の時間変更は加えていない。並進加速度の分離と全方向の磁気較正も追加していない。6軸版と9軸版の係数を同じにし、mocapの評価誤差で選んだ値は使わない。

入力bagは`/home/leus/rosbags/20250310_delta_linear_rolling_1_2025-03-10-16-18-36.bag`である。既存の5条件に保存した加速度、地磁気、角速度、計測時刻、参照方位をそのまま入力し、同じ85.446秒、17,088時刻を評価する。右端の角速度標本と、計測時刻から求めた各区間の長さを用いる。

Madgwickの内部では磁場の水平成分が正のx方向に沿うため、mocap世界の鉛直をzに保ったまま、冒頭1秒の固定磁場の水平方位119.7143度から内部座標系を決める。初期参照方位を内部座標系へ写して開始し、出力をmocap世界へ戻す。この変換を省くと、磁北基準とmocap基準の差を推定誤差へ混入させてしまう。評価開始後のmocapは、両フィルタの更新へ入力しない。

## 数値結果

区間全体の誤差の二乗平均平方根を度で示す。各軸はEuler角同士の差を±180度へ折り返した絶対値、3軸は参照と推定の相対回転角である。

| 条件 | ロール | ピッチ | ヨー | 3軸の合成誤差 |
| --- | ---: | ---: | ---: | ---: |
| 角速度の時間発展のみ | 10.0603 | 6.2544 | 9.4837 | 7.3461 |
| 角速度と加速度の融合 | 6.9859 | 2.8587 | 5.9978 | 4.9082 |
| 9軸の全情報の融合 | 6.4791 | 3.0014 | 18.5729 | 20.3905 |
| Madgwick（6軸） | 5.9002 | 1.2955 | 5.0608 | 3.6384 |
| Madgwick（9軸） | 12.3425 | 4.8189 | 29.6862 | 28.1676 |

Madgwickでも、地磁気を加えると3軸の合成誤差が増えた。最初の60秒のヨーRMSは6軸版0.5994度、9軸版28.9694度である。この区間では参照ピッチが−14.45〜20.29度にあるため、特異点付近のEuler角の差だけでは説明できない。

固定係数を変えた補助比較でも、すべての指定値で9軸版の誤差が大きい。これらの値の中から主比較の係数を選ぶことはしていない。

| beta | Madgwick（6軸）の3軸RMS | Madgwick（9軸）の3軸RMS |
| ---: | ---: | ---: |
| 0.01 | 3.3135 | 20.3820 |
| 0.033 | 3.3576 | 27.7052 |
| 0.041 | 3.3888 | 27.8334 |
| 0.1 | 3.6384 | 28.1676 |

## 融合手法を介さない入力確認

地磁気を各時刻のmocap方位で世界へ写すと、水平方位は冒頭の基準から−73.96〜63.91度変動した。参照の傾きで補償して地磁気からヨーを求めると、そのRMSは全区間32.2367度、最初の60秒30.1906度である。固定世界磁場の仮定に沿う観測なら、機体の回転を戻した磁場の水平方位は固定基準と一致する。この確認にはgyroやBingham、Madgwickの推定結果を用いていない。

傾きの補償には評価区間のmocapを用いるので、この図は入力確認として扱う。参照を初期設定と評価にだけ使う方位フィルタと同列の推定器として数値を比較しない。

今回の誤差増大はBinghamの処理にだけ現れるものではなく、入力と固定磁場の仮定の整合性にも問題がある。ただし、磁気センサの故障を意味するとは限らない。較正、センサ軸・時刻の対応、周辺磁場の変動、鉛直に近い磁場の水平成分への誤差の影響を、この記録だけでは分離していない。

## 成果物と再生成

[方式を比較する線形図](../../ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_madgwick/method_comparison.pdf)と[片対数図](../../ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_madgwick/method_comparison_semilog.pdf)は4段2列で、左列に地磁気なし、右列に地磁気ありを置く。各段の縦軸を左右で共通にし、右列の大きな誤差を含む範囲にする。0の標本は対数軸にだけ表示しない。[地磁気からのヨーの確認](../../ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_madgwick/magnetic_heading_control.pdf)は別図にした。

既存の5条件にMadgwickを加えた[全7系列の誤差図](../../ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_madgwick/axis_errors.pdf)、[その片対数図](../../ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_madgwick/axis_errors_semilog.pdf)、[RPY](../../ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_madgwick/rpy.pdf)、[四元数](../../ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_madgwick/quaternion.pdf)も保存した。いずれもPDF/SVG/PNGがある。既存のBingham動画は5条件の版を保持する。

実装はProbTF-demoの`ros/examples/probtf_orientation_demo/scripts/compare_madgwick_rolling.py`、出力は`outputs/20250310_delta_linear_rolling_madgwick/`である。再生成はProbTF-demoのルートで行なう。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 ros/examples/probtf_orientation_demo/scripts/compare_madgwick_rolling.py \
  --output ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_madgwick
```

`--input`には既存の5条件の出力を指定できる。主比較の係数は`--beta`、補助比較の係数は`--sensitivity-gains`で指定する。入力と同じ出力先への上書きは拒否する。集計には入力の集計・時系列と使用したコードのSHA-256を記録する。論文側には数値とコードを`references/imu-orientation-filter/20250310-delta-linear-rolling-madgwick/`へ保存した。

独立に数値微分した方向残差のJacobi行列、静止時の方位回復、鉛直周りの6軸版の不変性、単位の影響、座標対応、右端標本と可変区間、評価中の参照を使わないこと、入力確認の式、左右の図で曲線が切れないことを11件の検査で確認した。
