# 円軌道飛行の6条件とMadgwick比較

`compare_feedback_orientation.py`は、全条件を角速度で時間発展させ、現在の加速度・地磁気を出力に融合するか、融合結果を次の予測にも引き継ぐかを比較する。ROS masterを起動せずbagを読み取り専用で開く。

| 条件名 | 出力への融合 | 次の予測へ渡す因子 |
| --- | --- | --- |
| 角速度の時間発展のみ | なし | なし |
| 角速度と加速度（加速度は予測に使わない） | 加速度 | なし |
| 角速度と加速度（加速度も予測に使う） | 加速度 | 加速度 |
| すべての情報（加速度と地磁気は予測に使わない） | 加速度・地磁気 | なし |
| すべての情報（加速度のみ予測に使う） | 加速度・地磁気 | 加速度 |
| すべての情報（加速度も地磁気も予測に使う） | 加速度・地磁気 | 加速度・地磁気 |

「予測に使う」は観測を融合したBingham分布を次時刻の角速度予測の初期値に使う意味である。加速度を積分して回転増分を作る意味ではない。候補密度の積を「融合」、四元数積による回転を「合成」とする。

## 再生成

ProbTF-demoのルートから実行する。Python環境はROS Noeticのrosbag、NumPy、SciPy、Matplotlib、numpy-quaternionを使う。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 ros/examples/probtf_orientation_demo/scripts/compare_feedback_orientation.py \
  --bag ~/rosbags/20250303_delta_circle_trajectory_flight_5_2025-03-03-13-20-43.bag \
  --output ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions
```

短い実行には`--duration 0.12`を加える。全記録では右端の角速度標本と計測時刻間隔を使い、間引かずに伝播する。mocapは冒頭の磁場基準の固定、共通初期方位とその後の評価にだけ使う。時系列は共通の記録範囲内で四元数の球面線形補間により対応付け、範囲外へ外挿しない。

保存済みの5条件のうち3つの等価な再帰分布とMadgwickを使う場合は次である。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 ros/examples/probtf_orientation_demo/scripts/compare_feedback_orientation.py \
  --input-five ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_five_conditions \
  --madgwick-input ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_nine_axis_madgwick \
  --output ros/examples/probtf_orientation_demo/outputs/20250303_delta_circle_trajectory_flight_5_six_feedback_conditions
```

新スクリプトは元の5条件の識別子とMadgwickの係数、時刻・入力値・参照の一致を検査する。`--duration`は保存結果の再利用と併用しない。読み込み・伝播は従来の`compare_rolling_bag.py`、基本形Madgwickは`compare_madgwick_rolling.py`の関数を使う。deprecatedパッケージは読み込まない。

初期集中度100、角速度の共分散0.0001 I3 rad²/s²、加速度・地磁気の方向集中度20、数値積分点数120、Madgwickの固定係数beta=0.1を用いる。加速度は世界の鉛直上向きと比力方向を対応させる。動的加速度の分離、バイアス除去、全方向の磁気較正、平滑化、観測除外・重み変更は行なわない。

## 出力

| ファイル | 内容 |
| --- | --- |
| `rpy`、`quaternion` | 6条件とmocapの成分の図、PDF/SVG/PNG |
| `axis_errors`、`axis_errors_semilog` | 各軸と3軸の合成誤差の4段、6条件、線形と同じ配置の片対数 |
| `summary.json`、`timeseries.npz` | 設定、RMS、出力分布6列、引き継ぐ分布3列とモード・誤差・入力 |
| `madgwick_comparison/method_comparison`、同`_semilog` | 全情報3条件とMadgwick（9軸）の同じ4段の誤差図 |
| `madgwick_comparison/summary.json`、同`timeseries.npz` | 比較4系列の設定、RMS、モード・誤差・入力 |

各軸はEuler角の差を±180度へ折り返した絶対値、3軸の合成誤差は相対回転角を用いる。片対数軸では0を非表示とし微小数を足さない。表示のRPYと四元数の枝・符号の選択は推定の再帰へ入力しない。

`carried_parameter`と`moment_fit_residual`の3列は、角速度のみ、加速度を引き継ぐ、両観測を引き継ぐ、の順である。`case_carried_groups`で6条件との対応、`output_and_retained_evidence_coefficients`で出力と引き継ぎの各因子を確認できる。出力専用の地磁気は次時刻の分布に加えない。

## 検証と数値

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 tests/probtf/test_orientation_feedback_comparison.py -v
```

90.361694秒、18,072時刻のRMSを度で示す。

| 条件 | ロール | ピッチ | ヨー | 3軸の合成誤差 |
| --- | ---: | ---: | ---: | ---: |
| 角速度の時間発展のみ | 8.30 | 14.50 | 9.35 | 18.42 |
| 角速度と加速度（加速度は予測に使わない） | 5.57 | 10.47 | 8.58 | 14.24 |
| 角速度と加速度（加速度も予測に使う） | 1.51 | 5.33 | 7.75 | 9.47 |
| すべての情報（加速度と地磁気は予測に使わない） | 13.99 | 11.61 | 7.20 | 19.03 |
| すべての情報（加速度のみ予測に使う） | 1.52 | 5.33 | 4.43 | 7.05 |
| すべての情報（加速度も地磁気も予測に使う） | 23.72 | 6.56 | 21.70 | 33.35 |

Madgwick（9軸）は1.45、0.53、30.73、30.79度である。加速度を引き継いで地磁気を出力時だけに融合する条件は、加速度だけを使う条件よりヨーと方位全体の誤差が小さい。この記録と固定した設定での比較であり、較正、分布の被覆率、観測の時間相関は別途検証を要する。

## 保存した以前の比較

3月3日の`outputs/20250303_delta_circle_trajectory_flight_5_five_conditions/`と`..._nine_axis_madgwick/`、3月10日の全出力を保持した。既存動画は以前の5条件の記録として保持しており、今回の6条件とは条件が異なる。

[今回の作業記録](../../../docs/reports/orientation-six-feedback_2026-10-15.md)と[保存した5条件の再生成案内](../../../docs/reports/orientation-five-conditions-before-feedback_2026-10-15.md)を参照する。
