# 0.5 Hzジャイロと非線形回転運動によるデッドレコニング

2026-11-13。第4章の一定角速度による増分写像を、制御入力を含む非線形運動方程式の流れに置き換えた独立比較。既存の同定値・図・coreは変更していない。

結果は [4条件の一覧図](publication/bag8_50_55s/noise_overview.pdf) と [数値表CSV](publication/bag8_50_55s/comparison_table.csv)、以下の別ファイルに保存する。各PDFはRPY比較・角速度・伝播幅と姿勢誤差の3ページである。

| 各軸の角速度標準偏差 | 個別PDF | RPY図 |
|---:|---|---|
| 0.1 deg/s | [PDF](publication/bag8_50_55s/sigma_0p1_deg_s/comparison.pdf) | [PNG](publication/bag8_50_55s/sigma_0p1_deg_s/rpy.png) |
| 0.5 deg/s | [PDF](publication/bag8_50_55s/sigma_0p5_deg_s/comparison.pdf) | [PNG](publication/bag8_50_55s/sigma_0p5_deg_s/rpy.png) |
| 2 deg/s | [PDF](publication/bag8_50_55s/sigma_2_deg_s/comparison.pdf) | [PNG](publication/bag8_50_55s/sigma_2_deg_s/rpy.png) |
| 5 deg/s | [PDF](publication/bag8_50_55s/sigma_5_deg_s/comparison.pdf) | [PNG](publication/bag8_50_55s/sigma_5_deg_s/rpy.png) |

## 使用した飛行記録

元bagは `20260612_grape_hovering_8_2026-06-12-17-46-41.bag`。mocapのZから、初期位置より約0.60 m高い位置を前半から維持していることを確認した記録である。[ゲイン・高度の確認記録](../bag_gain_comparison_20261106/README.md)を参照。

誤差を見て区間を選び直さず、前半のbag時刻50〜55秒を使った。この間のゲインはXY=(3,0.1,1)、Z=(5,1,2.5)、Roll/Pitch=(12,1,12)、Yaw=(4,1,2)。XY・Zは既存のfailure1・failure2と共通だが、全12ゲインが同じという意味ではない。

原IMU header時計で選んだ最初の観測は50.000988619秒、続く観測は52.000961632秒と54.000961632秒。終端は最初の観測からちょうど5秒後である。原測定を平均化せず選び、stamp・record時刻・source indexを保存した。FCとmain_bodyの軸はbagの `/tf_static` の恒等回転を照合している。

mocap姿勢は開始時の一度だけSLERPで整合し、初期姿勢はDiracとした。それ以後のmocapは比較専用。加速度・地磁気による姿勢更新はこの実験には入れていない。0.5 Hzの取得間隔の影響を見るため、評価区間の高密度gyroによる補間・SG微分・遅れ補正を予測に使わない。原gyroを物理角速度の候補平均とみなす近似を置き、gyroセンサ遅れ自体は逆補償していない。

## 今までの処理との違い

既存の `imu-first-order-lag` には次の処理がある。

1. 姿勢フィルタは、区間内一定の角速度候補を6点求積し、四元数積の二次モーメントを計算してBinghamへ再近似する。gyro予測とaccel更新を次の事前へ渡し、地磁気は出力だけに融合する。
2. モデル再生は、機体・アクチュエータの非線形6DoFをRK4積分し、同じ評価区間の高密度IMUから算出した余剰レンチを平均入力へ加えていた。
3. 直近の姿勢更新を動力学へ戻す比較では、このモデル＋余剰レンチ姿勢列から右回転増分を作り、現在の姿勢候補へ作用させていた。

今回は、6個それぞれの角速度候補を運動方程式へ直接通して増分を作る。既存の高密度余剰トルクをそのまま使うと、0.5 Hzで取得していない間のgyro情報が予測へ漏れるため、評価前の40〜45秒だけで余剰トルク平均を求め、評価中は固定する。

## 計算した法則

既知固定の公称慣性を \(J\)、body・重心まわりの制御トルクを \(\tau_{\rm act}(t)\)、過去から固定した余剰トルク平均を \(\bar\tau_{\rm res}\) とする。各観測区間で

\[
W_k\sim\mathcal N(\bar\omega_k,\sigma_\omega^2 I_3),\qquad
w_{i,\pm}=\bar\omega_k\pm\sqrt3\,\sigma_\omega e_i,
\qquad \lambda_{i,\pm}=1/6.
\]

\(\sigma_\omega\)は一回の角速度観測の標準偏差であり、連続時間白色雑音のPSDではない。ユーザーが依頼したノイズ感度比較として、こちらで上記4種類を事前に設定した。同じ実測平均を使い、乱数で測定値を変えていない。

各点から、右Hamilton積・wxyz順で

\[
\dot\omega_s=J^{-1}\left[\tau_{\rm act}(t)+\bar\tau_{\rm res}
                  -\omega_s\times(J\omega_s)\right],
\qquad
\dot D_s=\tfrac12D_s\odot(0,\omega_s),\qquad D_s(t_k)=1
\]

を同時RK4積分する。同じ6個の初期角速度を2秒間保持し、中間表示のたびに新しい雑音を加えない。各中間時刻で

\[
\widehat M_{D,k}(t)=\sum_s\lambda_sD_s(t)D_s(t)^\top,
\qquad
M_Q(t)=\operatorname{ProductMoment}\left(M_Q(t_k),\widehat M_{D,k}(t)\right)
\]

を既存coreの四元数積二次モーメント式で計算する。区間開始からの全増分を一度合成し、同じ全増分を中間時刻ごとに繰り返し掛けていない。

次の角速度取得時には、現在の姿勢周辺法則を保ち、新しい観測に基づく独立なGaussian角速度入力へ置く。これは第4章の入力独立性の仮定を採用したもので、物理的な姿勢・角速度の共同Bayes更新ではない。区間内部の依存は6本の同時積分に残すが、共有gyroバイアスや観測間の時間相関はモデル化していない。

RPYの表示では同じ6点求積法則の積を列挙する。3観測で6、36、216個になる有限な積法則であり、モンテカルロ標本ではない。その重み付きRPY平均・全共分散を直接計算するので、Binghamへの再近似は今回は不要である。図の帯は同じ法則の平均±2標準偏差であり、Gaussianや95%被覆を保証する領域とは説明しない。角度分岐は予測自身から選び、mocapを使って予測の巻き戻しや再中心化を行っていない。

6点求積は非線形流れに対する近似である。RK4刻み・トルクの補間にも有限誤差がある。四元数積式は、この有限かつ独立な積法則についての二次モーメントを計算する。元の連続Gaussianの全密度を厳密に積分したとは扱わない。

## 固定した力学モデルと入力

公称慣性は `minimal/grape_vehicle_model.json` の全行列を使用する。対角化や推定値への差替えはしていない。

\[
J=\begin{bmatrix}
0.0650000615&-0.0000007279&0.0000190151\\
-0.0000007279&0.0649526563&0.0000000592\\
0.0000190151&0.0000000592&0.1289921107
\end{bmatrix}\ {\rm kg\,m^2}.
\]

実測gimbal角と指令前史全体から、既存のロータ幾何・反トルク・公称CoGを保ったレンチを作る。推力は既存のZOH一次遅れモデルを使い、時定数0.1977120296秒を以前の別記録のIMU同定結果から固定使用した。この値を公称値や今回のbagから同定した値とは扱わない。推力飽和1.5〜27.6145 Nも既存モデルと共通である。

40〜45秒の過去gyroだけで \(J\dot\omega+\omega\times J\omega-\tau_{\rm act}\) の時間加重平均を求め、

\[
\bar\tau_{\rm res}=(-0.0750638571,-0.1172160690,-0.0117518022)^\top\ {\rm N\,m}
\]

を全予測へ固定して加えた。高密度データでの微分は完了済みの過去区間だけで使い、50〜55秒のgyroやmocapからトルクを合わせ直していない。過去残差scatterは診断として保存し、今回の帯には混ぜない。したがって帯は、固定J・固定アクチュエータモデル・記録入力・固定した過去余剰トルク平均の下での、角速度候補に由来する条件付きの広がりである。

指令列は全ノイズ条件で同じものを再生する。ノイズを変えて制御器を閉ループで再実行した実験ではない。実測gimbalの補間も既知入力として扱う。

## 比較対象と結果

- `constant`：既存の一定角速度増分。取得間は観測角速度を保持し、トルクモデルを使わない。
- `linear_forced`：同じ制御トルクと過去余剰トルク平均を使うが、\(\omega\times J\omega\)だけを省略。
- `nonlinear`：制御入力とジャイロ項を含む上記の運動方程式。

全方式で初期姿勢、取得した3件のgyro、角速度候補の分散、比較するmocap列は共通。原gyroの高密度曲線は角速度図へ参照専用で表示する。

この固定モデルでは、制御トルクを使う予測の平均がmocapから大きく外れた。ノイズを広げても、この平均のずれは解消しない。一定角速度・線形強制・非線形のそれぞれを残し、結果が良くなるように区間・トルク平均・ノイズを選び直していない。詳しい数値は下記の表と保存ファイルを参照。これは非線形伝播を使えば実測再現が改善する、という結果ではない。

入力の変換について、`four_axes/command` は4ロータ分のfull thrustであり、8個の仮想推力の前半だけを読んでいたわけではないことを元C++と照合した。40〜55秒の `/rpy/pid` 148件で記録されたRP/Yaw PID追加項はすべて零で、`joint_states` はservo_bridgeからの実測だった。未記録のジャイロ補償、電圧・PWM変換、飽和等まで実測したわけではない。ここで計算しているのは、公称モデルによる指令トルクであって、実現トルクの直接測定ではない。

比較専用の高密度gyroから、公称Jで必要トルクを逆算した[入力モデル診断図](publication/bag8_50_55s/input_model_diagnostic.png)も保存した。予測に加えたトルクと、この診断トルクの差の平均は(0.081913, -0.009168, -0.019855) N mで、roll方向の大きな平均ずれが残る。これは未使用gyroからの事後診断であり、予測へ戻していない。実現トルクの真値や外力事後とも扱わない。非線形項を省いた対照も同様に大きく外れることと合わせ、公称モデル・入力・固定した余剰平均の不整合がこの比較の主要な制約になっている。

[4ノイズ条件の数値表](publication/bag8_50_55s/RESULTS.md)に、各方式のSO(3) RMSEと終端の2σ幅を併記した。角度の全数値はCSVと各条件のJSONに保存している。

RPY誤差は周期性を考慮して[-180,180]度へ折り返して集計し、SO(3)測地角も併記した。図は連続な予測角度を表示するため、一回転を超える場合がある。角度表示の折返しによる見かけだけで大きな誤差が生じたのではない。

17件の回帰検証で、自由回転の保存則、球対称慣性の既知解、元の一定角速度6点則、全有限積とcoreの二次モーメント・RPY共分散の一致、姿勢をリセットしない更新、高密度評価gyro・初期以外のmocapの非混入を確認した。σ=2 deg/sでRK4最大刻みを2 msから1 msへ半減した最大姿勢差は0.000232度だった。[刻み感度](publication/bag8_50_55s/rk4_step_audit.json)は有限な数値検査であり、モデル・求積・独立性の近似誤差が小さいことまで保証しない。

## 保存と再実行

公開先の `input.npz/json` は小容量の抽出入力で、生bagではない。全原時計とsource index、初期整合、過去較正データ、指令前史、比較専用列を区別して保存する。各条件の `*.npz` は各時刻の四元数二次モーメント、同じ法則のRPY平均・全共分散、6本の区間増分と角速度、区間開始の親法則を含む。`metrics.json`、`summary.json`、`torque_profile.json` に仮定・出典・hashと誤差を保存する。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/nonlinear-gyro-mpl /usr/bin/python3 \
  ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/run.py \
  --input ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/publication/bag8_50_55s/input.npz \
  --output /tmp/nonlinear-gyro-rerun --sigma-deg-s 0.1 0.5 2 5
```

元bagからの抽出は `extract.py --output <新しい出力先>`、検算は同フォルダの `tests` を `python3 -m unittest discover -s .../tests -v` で実行する。

## 追加のSG微分・再積分検査

同じ区間のmocap姿勢からbody角速度をSG微分で求め、原ジャイロとの時間ずれを調べた。また、そのSG角速度を再積分する往復検査も別に行った。[説明・二つのPDF・数値結果](gyro_mocap_sg_check/README.md)を参照。元の予測やトルクには戻していない。
