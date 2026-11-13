# mocap微分でrefineした非線形回転モデル

2026-11-13。公称値を初期値として、mocapから得たbody角速度・角加速度と実指令から、回転モデルを最小二乗で調整した。旧実験・同定済み機体・ProbTF coreは変更していない。

[結果とパラメータ表](publication/bag8_40_45_fit_50_55_predict/RESULTS.md)、[周波数一覧PDF](publication/bag8_40_45_fit_50_55_predict/rate_overview_sigma_2p5.pdf)、[全18条件の個別PDF](publication/bag8_40_45_fit_50_55_predict/INDEX.md)、[動力学照合PDF](publication/bag8_40_45_fit_50_55_predict/dynamics_fit.pdf)、[同定値PDF](publication/bag8_40_45_fit_50_55_predict/fit_report.pdf)を保存した。

## データの分離

同じ一日目#8を使い、調整はbag時刻40〜45秒のmocap 600件、予測評価は従来と同じ50.000988619〜55.000988619秒とした。調整区間は評価誤差を見る前に固定した。中心SGは0.32秒・3次で、実際に参照するmocapは39.848068129〜45.155978265秒。全て評価開始より前である。

原header/record時刻とindexを保持した既存の[制御・gyro入力](../publication/bag8_50_55s/input.json)と[mocap入力](../gyro_mocap_sg_check/publication/bag8_50_55s/input.json)を再利用する。新たにbagを複製していない。最適化に渡す指令は学習終端まで、実測gimbalは補間の右側1件までに切り出す。

評価開始の姿勢だけをmocapに一度整合する。その後の姿勢予測は、取得した原gyroと、記録された指令・gimbalによるトルクに条件付ける。評価区間のmocap・高密度gyroを使ってトルクやパラメータを合わせない。加速度・地磁気の観測更新、gyroバイアス・時刻ずれの補正は加えていない。

## 角速度・角加速度

符号を連続にしたxyzw四元数の各成分を、原時刻で局所多項式に当てはめる。多項式を (u(t)) として、

\[
q(t)=u(t)/\|u(t)\|,\quad
\omega=2\operatorname{vec}(\overline q\odot\dot q),\quad
\dot\omega=2\operatorname{vec}(\overline q\odot\ddot q).
\]

正規化の一次・二次微分を含める。RPYの微分をbody角速度と同一視しない。回転はbody→world、Hamilton積。中心SGによるオフラインの有限窓近似であり、実時間の因果推定ではない。

## 公称値からの最小二乗

原指令のZOHと既存の推力一次応答で各ロータの推力 (f_i(t;\tau_f)) を作る。実測gimbalから方向 (d_i(t))、動く推力原点 (r_i(t)) を求める。推力係数を (k_i>0)、重心を (c)、既知の反動トルク比と回転方向の積を (kappa_i) として、

\[
F_i=k_i f_i d_i,\qquad
\tau_{\rm act}(t;\theta)
=\sum_i\{(r_i-c)\times F_i+\kappa_i F_i\}.
\]

実アクチュエータの既存飽和1.5〜27.6145 Nはそのまま使う。共分散のclipではない。実推力の測定値ではなく、記録された発行指令に対する応答モデルである。

調整する式は

\[
\min_\theta\sum_n w_n
\left\|J_\theta\dot\omega_n+
\omega_n\times(J_\theta\omega_n)-\tau_{\rm act}(t_n;\theta)\right\|^2,
\]

で、(w_n)は原時刻の台形積分重みを和1にしたもの。線形項だけへの置換、外れ値除去、ロバスト損失、事前正則化は使わない。

回転式だけでは (J,k_i) の共通倍率が定まらないため、

\[
\operatorname{tr}J=\operatorname{tr}J_{\rm nominal}
=0.258944828487908\ {\rm kg\,m^2}
\]

を固定する。物理的な慣性を保つため、(C_0=\operatorname{tr}(J_0)I/2-J_0)から

\[
\widetilde C=C_0^{1/2}\exp\left(\sum_{a=1}^{5}z_a B_a\right)C_0^{1/2},\quad
C=\frac{\operatorname{tr}J_0}{2\operatorname{tr}\widetilde C}\widetilde C,\quad
J=\frac{\operatorname{tr}J_0}{2}I-C
\]

とする。(B_a)は対称trace-free行列の基底。自由座標は慣性形状5、重心3、log推力係数4、log推力時定数1の13次元。全慣性交差を残す。ロータ幾何と反動トルク比は公称値に固定し、質量・並進運動は同定しない。

物理値は全て公称値から開始する。公称JSONには推力時定数がないため、その数値初期値だけは原指令周期の中央値0.004983183秒とした。過去の別bagの推定時定数を公称値として流用しない。解析Jacobian付きの通常のtrust-region最小二乗を1初期値で実行し、乱数・multistartを使わない。有限の最適化残差や反復上限は記録し、品質監査による後続予測の打切りに使わない。

物理値を調整した後で、学習区間の

\[
\bar\tau_{\rm res}=\sum_n w_n
\{J\dot\omega_n+\omega_n\times J\omega_n-\tau_{\rm act}(t_n)\}
\]

だけを計算し、予測中の平均入力へ固定して加える。余剰を自由な最適化変数にはしていない。公称初期モデルの対照も、同じ学習区間から自身の余剰平均を求める。

## 周波数・ノイズ比較

取得頻度は0.5／2／5 Hz、各軸の標準偏差は0.1／0.5／1／2／2.5／5 deg/s。各条件を別フォルダ・別PDFにした。ノイズを測定値へ乱数で加えず、同じ原gyroを平均とする入力Gaussianの幅だけを変える。これは1観測あたりの分散であり、連続時間PSDではない。

6点から非線形Euler式と四元数を同時RK4積分する。

\[
\dot\omega_s=J^{-1}\{\tau_{\rm act}(t)+\bar\tau_{\rm res}
-\omega_s\times J\omega_s\},\quad
\dot D_s=\tfrac12D_s\odot(0,\omega_s).
\]

公称モデル・調整後モデルの両方で非線形項を残す。次の取得では姿勢候補を保持して新たな独立角速度入力へ置く。区間内で同じ初期6点を使い、中間表示ごとに独立化しない。

既存の[固定サイズの回転モーメント伝播](../rate_comparison_5hz/engine.py)を変更せず使う。同じ6点増分因子列に対して四元数二次モーメント、回転行列の平均・全9×9共分散を保持する。RPYの中心 (f(E[R])) と (Df\operatorname{Cov}(\operatorname{vec}R)Df^T) は表示だけのdelta-method近似で、状態へ戻さない。6点求積・RK4最大刻み0.002秒・トルク時間格子0.001秒にも有限誤差がある。Monte Carloは使わない。

## 結果の読み方

sigma=2.5 deg/sの同じ501時刻におけるSO(3)姿勢RMSE：

| 取得頻度 | 公称初期モデル | 調整後の非線形モデル |
|---:|---:|---:|
| 0.5 Hz | 103.5340° | 57.9916° |
| 2 Hz | 51.4672° | 14.8143° |
| 5 Hz | 22.3489° | 6.7866° |

refineで公称モデルより改善したが、ずれは依然大きい。評価時の角加速度のベクトルRMSEも2.4432→0.9927 rad/s²に減ったが、残る偏りを積分すると姿勢がずれる。小さな観測ノイズの帯にmocapが入らないのは、この条件付き帯がモデル誤差や将来の余剰変動を含まないためでもある。

ユーザーの追加質問に基づき、新しい図・表は非線形モデルだけに絞った。[ずれの分解](publication/bag8_40_45_fit_50_55_predict/DRIFT_ANALYSIS.md)では、モデルのトルク誤差・非線形項・取得時のgyro誤差・数値積分誤差を分けている。今回の最小二乗は (r=J(\dot\omega_{\rm obs}-\dot\omega_{\rm model})) を最小化しており、(J)の調整とともに角加速度誤差の重み (J^TJ) まで変わる。トルク残差の減少を、角加速度や姿勢予測の適合と同一視できない。評価区間には角加速度の平均偏り ((+0.2834,+0.1518,-0.2322)\) rad/s²が残る。RK4刻みを半減したRPY変化は最大約0.00002度であり、現在の多度のずれを説明しない。

調整後は慣性が平面剛体の境界へ近づいた。(C)の最小固有値は約2.80e-12 kg m²。log chartのJacobianは数値rank12だが、直接のtrace-free慣性方向では局所rank13であり、追加の物理パラメータ非同定と解釈しない。境界近くの局所Gaussian曲率や、重なるSG窓の誤差を独立とした共分散を、校正済み事後分布とは扱わない。全特異値・全交差曲率・基底・境界診断を保存した。

回転式だけの調整なので、推力の絶対校正や並進力の整合は保証しない。調整後の学習平均推力zは12.976 Nに対し、公称質量の重量は23.061 Nである。これをそのまま6自由度の実機同定値とすることはできない。

このホバリングでは、真の運動に近いmocap参照での非線形項自体は小さい。公称慣性で学習区間の (omega\times J\omega) RMSはroll約0.000485 Nm、(J\dot\omega)の約2.5%。今回の改善は慣性・重心・係数・応答の同時調整の結果であり、非線形項だけの優位性の実証ではない。

## 再現と検証

リポジトリのルートから実行する。

```bash
MPLCONFIGDIR=/tmp/mocap-refined-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/mocap_refined_dynamics/run.py --output /tmp/mocap-refined-dynamics-rerun
python3 -m unittest discover -s ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/mocap_refined_dynamics/tests -v
```

解析的な多軸回転でのSG二階微分、全13列の解析Jacobianの差分照合、人工の既知パラメータ回復、公称アクチュエータ式との一致、有限反復上限後の継続など14テストを実施した。加えて実データの5 Hz・sigma=2.5条件で、未取得gyro・評価mocapなどをNaNへ置換して予測し直し、全保存回転モーメントがbitwise同一であることを検証する。これは入力経路の検査であり、推定を受理・棄却するゲートではない。

公開対象は完成した図、描画・分布モーメント数値、同定・識別診断と再現コード。生bag、最適化途中、smoke試行、キャッシュは含めない。
