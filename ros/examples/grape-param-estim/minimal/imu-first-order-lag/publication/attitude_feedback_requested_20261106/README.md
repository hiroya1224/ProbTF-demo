# 実IMU：姿勢更新を動力学へ戻す比較

[五種類の要求図をまとめたPDF（15ページ）](requested_sensor_trajectory_torque_comparison.pdf)

failure1・failure2・successの各記録について、生gyro／生accel対モデル＋余剰レンチのセンサ応答、二方式のxyz、二方式のrpy、実使用の余剰トルクだけを掲載する。最新指定によりxyzのモデル共分散帯は表示から省き、平均を示す。rpyの右列はIMU姿勢推定の同じ曲線と±2σの全幅。青は重力だけの加速度観測、橙は並進加速度を含む観測、黒は原時刻のmocap参照。

| 記録 | 元IMU時刻 [s] | 時刻数 | PNG |
|---|---:|---:|---|
| failure1 | 19.002181–24.997109 | 1200 | [gyro](failure1/gyro_reconstruction.png) / [accel](failure1/accel_reconstruction.png) / [xyz](failure1/position_reconstruction.png) / [rpy](failure1/attitude_reconstruction.png) / [余剰トルク](failure1/applied_residual_torque.png) |
| failure2 | 25.504054–30.998216 | 1100 | [gyro](failure2/gyro_reconstruction.png) / [accel](failure2/accel_reconstruction.png) / [xyz](failure2/position_reconstruction.png) / [rpy](failure2/attitude_reconstruction.png) / [余剰トルク](failure2/applied_residual_torque.png) |
| success | 65.001922–74.998157 | 2000 | [gyro](success/gyro_reconstruction.png) / [accel](success/accel_reconstruction.png) / [xyz](success/position_reconstruction.png) / [rpy](success/attitude_reconstruction.png) / [余剰トルク](success/applied_residual_torque.png) |

旧同定値・全共同共分散・三つの時定数を固定し、再同定せず計算した。モデル＋実使用余剰レンチの回転増分を、各方式の現在の再帰姿勢候補へ作用させる。その回転モーメントで物理並進を伝播する。地磁気は新規取得時の出力融合だけに使い、次の事前へ戻さない。mocapの継続補正はない。

位置の共分散は、保持したパラメータcrossと未知な姿勢・時刻相関の上界を合わせた**保守的な共分散上界**であり、計算・保存は継続する。姿勢の表示帯は各出力lawから得た局所座標proxyであり、Euler角の厳密な分散ではない。95%被覆保証とはしない。姿勢帯の切捨て、元の共分散の縮小、観測出力の跳びの平滑化は行わない。有限射影、数値積分、区間増分の独立近似については[定義と再実行手順](../../ATTITUDE_FEEDBACK_COMPARISON.md)を参照。

`<case>/figure_data.npz` は全描画曲線・標準偏差・元時刻の参照列。`trajectory_marginal_covariance.npz` は同じ位置とbody右回転の全3×3周辺共分散をコピーしたもので、位置・方位の共同事後を捏造したものではない。`observer_numerical_diagnostics.npz` は有限射影残差・求積誤差の診断と実更新mask。未収束の大きさで後続観測を止める処理はない。元入力、同定値、全パラメータ共分散は[旧publication](../imu_native_20261031/)を参照し、重複してコピーしない。

`run_manifest.json` に入力・同定値・計算実行ソースhash、`render_manifest.json` に後から指定された表示専用変更のsourceと描画前後の全配列不変の検証、`test_regression_summary.json` に58件＋27件の回帰結果、`saved_array_integrity_check.json` に原時計と両方式の共通入力の検証、`plotted_array_validation.json` に描画値・姿勢帯・15ページの独立照合を残した。これらは検証記録であり、推定の受理条件ではない。

粗い固定求積で判明した誤差を補うため、数値精細化を有効にして全記録を再計算した。GH7／GH9の19特徴・正規化定数・回転平均の差を確認し、必要なときは粗いMqqの固有基底でHaar求積を配置して32×32²／40×48²へ精細化する。候補law・モデル・ノイズは変更しない。残る二則間の差、有限3回の射影残差、精細則の使用maskを保存する。未収束でも有限な結果を使って観測更新を継続し、真値を見て設定や結果を選び直していない。固定lawの独立Haar収束値に対する回帰では、回転平均の最大誤差1.65×10⁻⁶、logZ誤差2.20×10⁻⁶だった。これは全時刻の厳密な誤差保証ではない。

**残る数値精度の制約：** successの再帰側では最終二則間の特徴モーメント差が最大4.61×10⁻⁴、logZ差が1.60×10⁻³。地磁気を加えた出力側では最大0.00539／0.626となり、全時刻で数値許容誤差1e-5を達成した結果ではない。地磁気の出力姿勢の揺れを観測モデルだけの効果とは断定しない。この出力は次の再帰事前やCoGの並進へ戻していない。二則間の差は数値診断であり、厳密な誤差上界ではない。これらの値と全maskは保存し、未達を理由に観測を捨てたり計算結果を選別したりしていない。

全law・条件付き加速度・source因子を含む中間NPZと分割PDFは実行先 `/tmp/grape-attitude-feedback-refined-20261106` に残し、Gitには含めない。公開物の容量とhashは `published_artifacts.json` に記録する。旧結果・既定方式・coreは変更していない。
