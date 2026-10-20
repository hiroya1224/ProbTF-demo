# 円軌道飛行の6条件とMadgwick フィルタの比較

本文5.3.3に、2025年3月3日の円軌道飛行の6条件、全情報3条件とMadgwick フィルタの比較、位置・角速度・加速度・地磁気の時系列を載せた。表示・評価するBingham分布と次時刻へ渡す分布を分け、Algorithmにも同じ手順を記した。

| 条件 | 出力へ融合する観測 | 次の予測へ渡す観測 |
| --- | --- | --- |
| 角速度の時間発展のみ | なし | なし |
| 角速度と加速度（加速度は予測に使わない） | 加速度 | なし |
| 角速度と加速度（加速度も予測に使う） | 加速度 | 加速度 |
| すべての情報（加速度と地磁気は予測に使わない） | 加速度・地磁気 | なし |
| すべての情報（加速度のみ予測に使う） | 加速度・地磁気 | 加速度 |
| すべての情報（加速度も地磁気も予測に使う） | 加速度・地磁気 | 加速度・地磁気 |

全条件で回転増分は角速度から求める。「予測に使う」は、観測を融合した分布を次時刻の角速度予測の初期値へ渡す意味である。現在の出力にだけ融合する観測は再帰分布へ加えない。候補の密度の積を「融合」と呼ぶ。

## 数値と設定

初期設定の1.0秒、200標本で世界の磁場基準を固定した。その後の87.171135秒、17,434時刻を評価する。モーションキャプチャは初期設定と評価だけに用い、評価中の方位を予測へ入力しない。初期集中度100、角速度共分散0.0001 I3 rad²/s²、加速度・地磁気の集中度20、数値積分点数120を用いた。Madgwick フィルタは正規化勾配の基本形で、固定係数beta=0.1である。

入力は同じbagの角速度・加速度・地磁気・モーションキャプチャである。予測は全標本を時刻順に伝播し、計測時刻間隔と右端の角速度を用いる。参照は四元数の球面線形補間により対応付ける。図の経過時間0は全系列で共通の評価開始時刻を表す。

各軸はEuler角の差を±180度へ折り返した絶対値、3軸の合成誤差は相対回転角である。全時系列の二乗平均平方根を度で示す。

| 条件 | ロール | ピッチ | ヨー | 3軸の合成誤差 |
| --- | ---: | ---: | ---: | ---: |
| 角速度の時間発展のみ | 8.25 | 14.88 | 9.52 | 18.77 |
| 角速度と加速度（加速度は予測に使わない） | 5.53 | 10.75 | 8.75 | 14.52 |
| 角速度と加速度（加速度も予測に使う） | 1.53 | 5.34 | 7.95 | 9.64 |
| すべての情報（加速度と地磁気は予測に使わない） | 13.17 | 11.83 | 7.31 | 18.64 |
| すべての情報（加速度のみ予測に使う） | 1.53 | 5.34 | 4.62 | 7.18 |
| すべての情報（加速度も地磁気も予測に使う） | 23.31 | 6.71 | 17.35 | 30.29 |

| 全情報の比較 | ロール | ピッチ | ヨー | 3軸の合成誤差 |
| --- | ---: | ---: | ---: | ---: |
| すべての情報（加速度と地磁気は予測に使わない） | 13.17 | 11.83 | 7.31 | 18.64 |
| すべての情報（加速度のみ予測に使う） | 1.53 | 5.34 | 4.62 | 7.18 |
| すべての情報（加速度も地磁気も予測に使う） | 23.31 | 6.71 | 17.35 | 30.29 |
| Madgwick フィルタ（9軸） | 1.47 | 0.52 | 25.06 | 25.12 |

加速度を引き継ぎ、地磁気を出力時だけに融合する条件では、加速度だけを用いる条件に比べてヨーのRMSが7.95度から4.62度、合成誤差が9.64度から7.18度へ小さくなった。全情報の2条件の合成誤差18.64度と7.18度はMadgwick フィルタの25.12度より小さく、両観測を引き継ぐ条件の30.29度は大きい。センサの較正、動的加速度、観測尺度と時間相関の影響は個別に分離していない。

## 表示

- [6条件の誤差](outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions/axis_errors.pdf)、[全情報3条件とMadgwick フィルタ](outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions/madgwick_comparison/method_comparison.pdf)。
- [ロール・ピッチ・ヨー](outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions/rpy.pdf)。
- [位置](outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions/position.pdf)、[角速度](outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions/gyro.pdf)、[加速度](outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions/acc.pdf)、[地磁気](outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions/mag.pdf)。
- [Bingham分布の6条件の比較動画](outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions/bingham_comparison.mp4)。
- [設定・時系列・使用コード](../../../docs/reports/orientation-six-feedback_2026-10-15.md)。

Madgwick フィルタは名称を省略せず黒色にした。6条件は青・水色・緑・黄・紫・赤を使い、実線・破線・一点鎖線の対応を保つ。線幅は1.35ポイント、破線は14:7、一点鎖線は14:5:2:5の長さにした。丸・四角・上向き三角・菱形・下向き三角・十字のマーカーを3 [s]間隔で実際の標本に付け、条件ごとに開始時刻をずらした。Madgwick フィルタは黒点線と星形を使う。全標本を描画し、推定値や誤差には平滑化・間引きを加えない。

論文全体で座標成分の色はx＝赤、y＝緑、z＝青とする。4つの取得値の図は3成分を横に並べて高さを抑え、曲線にこの色を使う。手法比較の曲線の色は条件を示す。グラフのラベル・目盛文字・凡例の文字はすべて黒色とし、単位は角括弧に囲む。加速度の縦軸上限は12 [m/s²]で、保存値と方位推定への入力は変更しない。位置はworld座標系、他の3図はセンサ座標系の取得値であり、正規化や平滑化を表示値へ加えない。

## 再生成

描画には[SciencePlots](https://github.com/garrettj403/SciencePlots)の`science`と`no-latex`を使用した。日本語の表示にはIPAexGothicを使う。生値の図も同じスタイルで描き、四元数の成分図は出力しない。

ProbTF-demoのルートで次を実行する。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 ros/examples/probtf_orientation_demo/scripts/compare_feedback_orientation.py \
  --bag ~/rosbags/20250303_delta_circle_trajectory_flight_5_2025-03-03-13-20-43.bag \
  --output ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions
```

`--duration 0.12`は短い実行、`--render-only`は保存済みの推定値から図を再描画する指定である。出力は7図のPDF/SVG/PNG、6条件と比較4系列の集計・時系列である。読み込み、予測、Madgwick フィルタの演算と使用コードはこの出力に対応させた。

動画は同じ6条件の分布行列とモードを用い、既存のSO(3)ビジュアライザで軸端点の点群を描く。各球面にモーションキャプチャの方位を黒破線で重ね、下段には6条件の合成誤差と現在時刻を表示する。4フレーム/秒、各分布180標本で生成する。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 ros/examples/probtf_orientation_demo/scripts/render_feedback_video.py \
  --input ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions \
  --output ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions
```

次の8件に、再帰分布の独立性、部分的な観測の引き継ぎ、共通時刻の読み込みと初期設定、加速度の表示上限を確認する検査を用意した。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 tests/probtf/test_orientation_feedback_comparison.py -v
```

従来の方位計算9件とMadgwick フィルタ11件も通過した。分布の被覆率と、較正・観測誤差の時間相関を含む観測モデルの検証は本文のTODOに残している。

[全体の最終精査](../../../docs/reports/orientation-six-feedback-final_2026-10-15.md)に、本文の配置、座標色、数値と動画の対応を記した。

## 成功飛行の静止ノイズ校正

`compare_success_yaw_window.py` は、2026年6月13日の成功飛行の冒頭1 [s]の静止200標本から、角速度・加速度・地磁気の3×3標本共分散をそれぞれ求める。分母は N−1であり、平均値の共分散にするための N 除算はしない。角速度は軸間の共分散を含む全3×3行列を使い、加速度・地磁気の方向観測だけを `trace(covariance)/3` で等方化する。元の観測ベクトルから静止平均を引かない。

角速度の共分散は `[[6.47814413e-7,-5.35653150e-9,-1.23244457e-8],[-5.35653150e-9,3.51899478e-7,4.52832111e-8],[-1.23244457e-8,4.52832111e-8,3.48278089e-7]] [rad²/s²]` である。旧固定値 `1e-4 I` を置き換え、同じ行列を再帰3群の全角速度予測へ渡す。実測 dt は既存の四元数増分計算の内部で適用し、入力共分散へ dt や標本数による追加の係数を掛けない。`static_noise_calibration.json` の `angular_velocity` と `summary.json` の `gyro_sample_covariance_rad2_s2` に同じ行列を保存する。

生ベクトルの等方 Gaussian 観測モデルから、各時刻の集中度を `norm(static_mean)*norm(raw_measurement)/isotropic_variance` として既存の Bingham 方向尤度へ渡す。κを20へ固定せず、加速度・地磁気で別の校正値を使う。ノイズの校正値は `static_noise_calibration.json`、標本ごとの集中度は `timeseries.npz`、その範囲は `summary.json` に保存する。右列の加速度のみ・地磁気のみの分布も、推定と同じノイズ項で描く。

このデータの加速度集中度は約256万になる。全3群でデモ内の `bingham_moment_inverse.py` を使い、同じ core の2次モーメント・解析的なモーメント Jacobian・四元数積の伝播式を保って、上限のない3自由固有値のモーメント方程式を Newton 法で解く。目標固有モーメントによる数値スケール調整と残差での backtracking を使い、微小な角速度ノイズが逆算の収束判定で失われないようにする。core の分布法則、ノイズ、目標モーメントは変更せず、有限の反復残差を理由に次の更新を停止しない。Madgwick フィルタの計算は変更しない。

静止ノイズだけでは飛行中の並進加速度、磁場の変化、バイアスを説明しない。静止系列にも時間相関があるため、全標本の独立性まで検証できたという意味ではない。静止平均・参照・ノイズ共分散は固定した推定値として扱う。

実測角速度共分散の異方性によって、加速度を事前へ戻す条件では異常なヨー変化が再現する。逆算で微小ノイズを取りこぼす不具合は修正したが、この変化は未解消である。単一Binghamへのモーメント近似がヨー方向の密度形状を保たない反例と、数値不具合の修正を[調査報告](../../../docs/reports/imu-gyro-covariance-yaw-jump_2026-10-21.md)に区別して記録した。

```bash
source /home/leus/catkin_ws/devel/setup.bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 \
  ros/examples/probtf_orientation_demo/scripts/compare_success_yaw_window.py \
  --selection-json /home/leus/dthesis/references/imu-orientation-filter/20260613-success-yaw-63-123s/selection.json \
  --output /tmp/imu-success-yaw-calibrated-noise
```
