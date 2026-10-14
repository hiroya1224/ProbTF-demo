# 円軌道飛行の6条件とMadgwick比較

2025年3月3日の円軌道飛行について、角速度に加速度と地磁気を融合する方法と、融合結果を次時刻へ渡す方法を変えた6条件に本文5.3.3を更新した。「すべての情報」の3条件と基本形のMadgwick（9軸）の比較も、同じ4段の誤差図にまとめた。

| 条件名 | 出力へ融合する観測 | 次時刻へ引き継ぐ観測 |
| --- | --- | --- |
| 角速度の時間発展のみ | なし | なし |
| 角速度と加速度（加速度は予測に使わない） | 加速度 | なし |
| 角速度と加速度（加速度も予測に使う） | 加速度 | 加速度 |
| すべての情報（加速度と地磁気は予測に使わない） | 加速度・地磁気 | なし |
| すべての情報（加速度のみ予測に使う） | 加速度・地磁気 | 加速度 |
| すべての情報（加速度も地磁気も予測に使う） | 加速度・地磁気 | 加速度・地磁気 |

「予測に使う」は、観測を融合した分布を次時刻の角速度による予測の初期値に使うという意味である。回転増分は全条件で角速度から求める。加速度と地磁気は方向因子として融合する。候補の密度の積を「融合」と呼び、回転の「合成」と区別する。

表示・評価する行列Aと次時刻へ渡す行列Bを分けた。3種類のB（角速度だけ、加速度まで、両観測まで）を伝播し、現在の方向因子を各条件のAへ加える。出力専用の因子を引き継がず、同じ観測因子を同一時刻で2度加えることもない。本文の式とAlgorithmをこの手順に改めた。

## 入力と固定設定

入力bagは`/home/leus/rosbags/20250303_delta_circle_trajectory_flight_5_2025-03-03-13-20-43.bag`、SHA-256は`6e3c5eb7a945a04e363acfaa14d3a32e0b36cc90a76a1d52bffd55549fd7bcaf`である。冒頭1.0秒、200標本で磁場基準を固定し、その後の90.361694秒、18,072標本を評価した。モーションキャプチャは初期設定と評価にだけ使う。全標本を時刻順に予測し、右端の角速度標本を用いる。

6条件の初期集中度100、角速度標本の共分散0.0001 I3 rad²/s²、加速度・地磁気の方向集中度20、数値積分点数120を固定した。Madgwickは正規化勾配の基本形、固定係数beta=0.1である。磁北座標系とmocap世界座標系の対応は冒頭だけで固定した115.743399度の鉛直周り回転を使う。バイアス補正、平滑化、動的加速度の分離、観測の除外・重み変更は追加していない。

## 数値

各軸は推定と参照のEuler角の差を±180度へ折り返した絶対値、3軸の合成誤差は方位間の相対回転角である。二乗平均平方根を度で示す。

| 条件 | ロール | ピッチ | ヨー | 3軸の合成誤差 |
| --- | ---: | ---: | ---: | ---: |
| 角速度の時間発展のみ | 8.30 | 14.50 | 9.35 | 18.42 |
| 角速度と加速度（加速度は予測に使わない） | 5.57 | 10.47 | 8.58 | 14.24 |
| 角速度と加速度（加速度も予測に使う） | 1.51 | 5.33 | 7.75 | 9.47 |
| すべての情報（加速度と地磁気は予測に使わない） | 13.99 | 11.61 | 7.20 | 19.03 |
| すべての情報（加速度のみ予測に使う） | 1.52 | 5.33 | 4.43 | 7.05 |
| すべての情報（加速度も地磁気も予測に使う） | 23.72 | 6.56 | 21.70 | 33.35 |

同じ全9軸情報による4系列の比較は次の通りである。

| 条件 | ロール | ピッチ | ヨー | 3軸の合成誤差 |
| --- | ---: | ---: | ---: | ---: |
| すべての情報（加速度と地磁気は予測に使わない） | 13.99 | 11.61 | 7.20 | 19.03 |
| すべての情報（加速度のみ予測に使う） | 1.52 | 5.33 | 4.43 | 7.05 |
| すべての情報（加速度も地磁気も予測に使う） | 23.72 | 6.56 | 21.70 | 33.35 |
| Madgwick（9軸） | 1.45 | 0.53 | 30.73 | 30.79 |

加速度を引き継いで地磁気を出力時だけに融合する条件は、加速度だけを使う条件と比べてヨーが7.75度から4.43度、方位全体が9.47度から7.05度へ小さくなった。地磁気も引き継ぐ条件は33.35度となった。Madgwickはロール・ピッチの誤差が最も小さく、ヨーは全情報の3条件より大きかった。較正・方向モデル・時間相関などの原因はこの比較では個別に分離していない。

## 図と保存した資料

- 6条件とmocap：[ロール・ピッチ・ヨー](../../ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions/rpy.pdf)、[四元数](../../ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions/quaternion.pdf)。
- 6条件の4段誤差図：[線形](../../ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions/axis_errors.pdf)、[片対数](../../ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions/axis_errors_semilog.pdf)。
- 全情報3条件とMadgwickの4段誤差図：[線形](../../ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions/madgwick_comparison/method_comparison.pdf)、[片対数](../../ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions/madgwick_comparison/method_comparison_semilog.pdf)。
- [6条件の集計・時系列・コード](../../ros/examples/probtf_orientation_demo/README.md)。
- 直前の3月3日5条件と2系列比較：[5条件の図と本文](orientation-five-conditions-before-feedback_2026-10-15.md)、[2系列比較の図](orientation-five-conditions-before-feedback_2026-10-15.md)。
- [既存のBingham動画](../../ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_five_conditions/bingham_comparison.mp4)は以前の5条件を記録した資料として保持した。今回の6条件図とは条件が異なる。

線形図と片対数図は同じ系列・4段・時刻範囲で、片対数は縦軸だけを変えた。誤差0の標本は対数軸で非表示とし、小さな正数へ置き換えていない。図中には系列名と軸ラベルだけを置いた。x,y,zおよびgyro/acc/magの確認用PDFと12パターンのプレビューは変更していない。

## 再生成

ProbTF-demoのルートからbagを使う計算は次で行なう。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 ros/examples/probtf_orientation_demo/scripts/compare_feedback_orientation.py \
  --bag ~/rosbags/20250303_delta_circle_trajectory_flight_5_2025-03-03-13-20-43.bag \
  --output ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions
```

保存済みの3種類の再帰分布とMadgwick時系列を使う場合は次である。元の5条件のうち、角速度のみ、加速度の全引き継ぎ、加速度・地磁気の全引き継ぎの分布と一致するため、これらを再利用し、残る出力分布を現在の観測因子から構成した。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 ros/examples/probtf_orientation_demo/scripts/compare_feedback_orientation.py \
  --input-five ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_five_conditions \
  --madgwick-input ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_nine_axis_madgwick \
  --output ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions
```

実装はProbTF-demoの`ros/examples/probtf_orientation_demo/scripts/compare_feedback_orientation.py`に置いた。旧スクリプトと旧出力は変更せず、deprecatedパッケージは変更も読み込みも行なっていない。新規の5検査は、過去の加速度・地磁気が各条件へ残るか、出力専用因子が蓄積されないか、保存分布と独立の再計算が一致するか、角速度の右端標本と時刻の検査を扱う。

分布の被覆率、較正、観測誤差の時間相関を含む観測モデルの検証は本文のTODOを維持した。全体の最終精査は別途記録する。
