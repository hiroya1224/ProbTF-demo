# 滑らかな合成gyroによる線形・非線形の比較

2026-11-13。実測や機体同定の結果と分け、既知の慣性・既知の入力トルクの下で、Euler式のジャイロ項の違いを比較した。

3周波数・0.5 Hz・sigma=5 deg/sの例は、[日本語音声付きの詳細解説動画](explainer/publication/INDEX.md)にもまとめた。960×540 px、音声2倍速。保存済みの6候補、RK4の実計算段、回転モーメント、全共分散、観測前後をアニメーションで説明する。[全文台本と再現手順](explainer/README.md)も保存した。

最新指定の[0.5 Hz・sigma=5 deg/s・2×3図PDF](publication/three_frequency_0p5hz_sigma5/attitude_and_gyro.pdf)／[PNG](publication/three_frequency_0p5hz_sigma5/attitude_and_gyro.png)は、各軸に振幅のある3周波数成分を使って再計算した。上段はroll・pitch・yaw、下段はomega_x・omega_y・omega_z。[係数と保存値](publication/three_frequency_0p5hz_sigma5/INDEX.md)を別保存し、以下の旧2周波数の比較も保持している。

3周波数版の再現は次のコマンド。生成noise・bias・既知Jと推定法は従来と同じで、3周波数に対応するgyroと入力トルクを生成し直す。

```bash
MPLCONFIGDIR=/tmp/synthetic-rotation-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/focus.py --output /tmp/synthetic-three-frequency-rerun
```

[周波数ごとの一覧PDF](publication/smooth_rotation_5s/rate_overview_sigma_2p5.pdf)、[生成した運動・gyro・トルク](publication/smooth_rotation_5s/generated_motion.pdf)、[全18条件の個別PDF](publication/smooth_rotation_5s/INDEX.md)、[数値CSV](publication/smooth_rotation_5s/comparison_table.csv)を保存した。

追加した[角速度の平均・±2σと幅の一覧PDF](publication/smooth_rotation_5s/gyro_progress_sigma_2p5.pdf)では、観測の直前・直後を同じ時刻に両方表示する。[幅の一覧PNG](publication/smooth_rotation_5s/gyro_width_sigma_2p5.png)も保存した。各条件の`gyro_progress.pdf`は、角速度、±2σ半幅、角速度誤差の3ページ。保存済みの同じ6点軌道から得る平均・全共分散は、元の保存値とbitwiseで一致し、推定は再実行していない。

現在の再取得は、新しい独立Gaussian角速度入力への置換であり、事前分布とのBayes条件付けではない。非線形側は伝播で幅が変化するため、観測時に狭まる軸も広がる軸もある。0.5 Hz・sigma=2.5の4秒時点では、±2σ半幅が約(6.710,5.701,5.993)から(5,5,5) deg/sへ狭まる。線形側は`delta omega`が区間内で一定なので角速度分散も一定で、くびれはない。見栄えに合わせた分散操作はしていない。

## 生成条件

5秒間の滑らかなroll・pitch・yawを、それぞれ2本の正弦波の和で与えた。結果を見て軌道・慣性・ノイズを選び直していない。

| 軸 | 振幅 [deg] | 周波数 [Hz] | 位相 [rad] |
|---|---|---|---|
| roll | 35, 6 | 0.42, 0.13 | 0.2, 0 |
| pitch | 25, 5 | 0.31, 0.19 | −0.4, 0.6 |
| yaw | 50, 10 | 0.37, 0.11 | 0.5, −0.2 |

姿勢は (R=R_z(\psi)R_y(\theta)R_x(\phi))。body角速度を

\[
\omega=\begin{bmatrix}
\dot\phi-\dot\psi\sin\theta\\
\dot\theta\cos\phi+\dot\psi\sin\phi\cos\theta\\
-\dot\theta\sin\phi+\dot\psi\cos\phi\cos\theta
\end{bmatrix}
\]

として解析的に計算し、さらに微分してbody角加速度を得る。RPYの微分をそのままgyroにしていない。慣性は三主慣性が異なる既知の (J=\operatorname{diag}(0.045,0.070,0.100)\ {\rm kg\,m^2})。物理的な慣性の三角不等式も満たす。

生成側で

\[
\tau(t)=J\dot\omega(t)+\omega(t)\times J\omega(t)
\]

を計算し、1 kHzの既知入力として保存する。両方式が受け取るトルクは同一で、予測中に真の角速度を使って補正しない。これは処方した姿勢運動による回転動力学の試験であり、並進・推進器・空力まで含む6自由度の飛行シミュレーションではない。

gyro生成は200 Hz、

\[
y_k=\omega(t_k)+b+\epsilon_k,\quad
b=[0.03,-0.02,0.04]^T\ {\rm deg/s},\quad
\epsilon_k\sim\mathcal N(0,(0.2\ {\rm deg/s})^2I).
\]

seedは20261113。乱数はこの測定生成だけに使う。実現したノイズ・測定値・真値・生成共分散・整数ns時計を[input.npz](publication/smooth_rotation_5s/input.npz)へ保存した。固定biasは推定せず、差し引かない。

## 再構成

同じ原測定列を0.5／2／5 Hzで取得する。原200 Hzのindexは、それぞれ0,400,800／100件刻み／40件刻み。取得の選択は整数ns時計で行い、浮動小数点の丸めで意図しない1 tick遅れを作らない。

入力角速度の仮定sigmaは0.1／0.5／1／2／2.5／5 deg/s。**この6設定は推定に使う候補Gaussianの幅であり、生成側の0.2 deg/sとは別**。測定値を設定ごとに生成し直さない。

比較する式は

\[
\begin{aligned}
\text{非線形：}\quad&\dot\omega=J^{-1}(\tau-\omega\times J\omega),\\
\text{線形：}\quad&\dot\omega=J^{-1}\tau.
\end{aligned}
\]

四元数の運動学 (\dot q=\tfrac12q\odot(0,\omega)\) は共通。線形側で省くのは角速度方程式のジャイロ項だけであり、姿勢運動学全体を線形化したという意味ではない。既知の同じJ・トルク・初期姿勢・取得gyroを使う。

初期姿勢を一度だけ真値にそろえ、その後の真値姿勢は評価専用。予測器へ渡すデータは区間、初期姿勢、取得gyro時刻、取得gyro値の4項目だけ。生成bias・真の途中角速度・未取得gyroは含めない。

既存の6点決定論的求積とRK4を使い、四元数二次モーメントと回転行列の平均・全9×9共分散を伝播する。各取得で新しい独立角速度Gaussianを与える仮定は従来と同じ。Monte Carlo推定、粒子の枝棄却、分散floorは使わず、coreも変更していない。

RPYの中心と帯は、同じ回転モーメントからの (f(E[R])) とdelta-method共分散による表示近似。帯は±2sigmaであり、保証付き95%領域ではない。J・トルク・固定した測定列に条件付けた不確かさで、未知のbiasの事後を含むものではない。

## 結果

仮定sigma=2.5 deg/s、同じ501時刻での姿勢中心のSO(3) RMSE：

| gyro取得 | 非線形Euler | ジャイロ項を省いた角速度動力学 |
|---:|---:|---:|
| 0.5 Hz | 0.6715° | 102.8546° |
| 2 Hz | 0.4224° | 30.3497° |
| 5 Hz | 0.2459° | 12.5940° |

今回はジャイロ項のトルクRMSが慣性トルクRMSの約20%／57%／7%になる三軸運動であり、その項を省く影響が現れている。非線形側にも小さい観測ノイズ・biasと有限求積による誤差は残る。既知入力に整合する合成例での比較であり、実機モデルの同定誤差まで解消した結果ではない。

別の無ノイズ・無bias検査では、非線形再構成の最大姿勢誤差が0.5／2／5 Hzで約0.000262／0.0000239／0.0000102°。主実験の測定列へこの理想値を代入していない。解析的body角速度・角加速度と回転行列／角速度の数値微分、トルク式、同じ測定列の保存を含む9テストも成功した。[検証数値](publication/smooth_rotation_5s/audit.json)を保存している。

## 再現

リポジトリのルートから実行する。

```bash
MPLCONFIGDIR=/tmp/synthetic-rotation-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/run.py --output /tmp/synthetic-controlled-rotation-rerun
python3 -m unittest discover -s ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/tests -v
```

角速度の途中経過は、上の再現コマンドの出力先を指定して追加できる。元の推定ファイル・測定列を更新せず、観測前後の全共分散と表示用数値を保存する。

```bash
MPLCONFIGDIR=/tmp/synthetic-rotation-mpl python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/gyro_progress.py --publication /tmp/synthetic-controlled-rotation-rerun
```

各条件のPDFと分布モーメントを別フォルダへ出力する。小容量の生成入力・図・描画数値・再現コードだけを保存し、以前の実測結果は上書きしていない。
