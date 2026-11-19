# 10 Hz予測へ等方角速度プロセスノイズを追加する

ユーザー指定のモデル側標準偏差を `sigma_Q = 3*pi/180 rad/s` とし、各0.1秒の予測終端に独立な角速度増分を加える。観測側は従来どおり `sigma_R = 5 deg/s`。元の人工真値、生gyro、白色ノイズ、スパイク、取得201件、J、既知トルク、初期姿勢は再生成・変更しない。[Q付きの図](process_noise_publication/INDEX.md)と[元のQ=0](rate_10hz_publication/INDEX.md)を別保存する。

共通201時刻の観測直後で比較した実行結果は次のとおり。

| モデル側の角速度増分σ | 非線形Euler姿勢RMSE [deg] | ジャイロ項なし姿勢RMSE [deg] |
|---|---:|---:|
| 0 | 18.8948 | 19.4696 |
| 3 deg/s / 0.1秒区間 | 20.0769 | 20.0328 |

今回のスパイク入り観測では姿勢誤差は少し増え、角速度は観測へ強く追従する。非線形側の終端角速度の2σ半幅は約0.71から6.68 deg/sへ広がった。比較のためにRやスパイクを再調整していない。

実際の全20秒・201観測について、線形角速度のKalman解析解との最大平均差は次数5/7/9/11で約0.936/0.246/0.0335/0.00224 deg/s。9→11の非線形RPY中心の最大差は約0.0302 deg。別の極端な交互スパイクでは低次数の誤差が大きく非単調になるため、その測定列と誤差も `tail_stress_diagnostic.json` に区別して保存する。一様な尾部精度を達成したとは主張しない。

## 追加する確率モデル

元のEuler／四元数式を区間内で積分した写像を、角速度について $f_k$、右回転増分について $\delta_k$ と書く。

\[
\begin{aligned}
q_k^- &=q_{k-1}^+\odot\delta_k(\omega_{k-1}^+),\\
\omega_k^- &=f_k(\omega_{k-1}^+)+\xi_k,
&\xi_k\sim\mathcal N(0,Q_\omega),\\
Q_\omega&=(3\pi/180)^2I_3.
\end{aligned}
\]

$Q_\omega$ の単位は rad²/s²、数値は約0.00274155678 I₃。1区間の角速度増分の分散であり、連続時間PSD、角加速度の分散、RK4各内部刻みの分散ではない。初期時刻には追加せず、201観測の間に200回適用する。終端で瞬間的に加わった増分はその時点のqを変えず、次の区間で姿勢へ作用する。この離散モデルは、連続白色トルク拡散を同時にqへ積分するモデルとは区別する。

各観測で $y_k=\omega_k^-+v_k$、$v_k\sim\mathcal N(0,R)$ として共同分布をBayes更新する。初回観測で $\omega_0\sim\mathcal N(y_0,R)$ を一度構成し、初期姿勢は既知のDirac測度。最初の観測を再融合しない。

## 初期角速度だけの法則から、Markov遷移へ

Q=0では全不確かさを初期 $W_0$ に置けたが、新しい独立増分が各時刻に入るため、現在の分布を保持して次へ伝える必要がある。単に表示共分散にQを足す処理は行わない。

本例では既知のbodyトルクと角速度の運動がqに依存しない。この構造を利用し、現在角速度の連続Gaussian mixtureと、各成分の条件付き四元数二次モーメント・全回転共分散を保持する。姿勢と角速度の依存は成分間・角速度条件付きのモーメントに残る。回転を条件付けずに独立なq分布へ分離しない。

親角速度の決定論的求積点をuᵢとすると、物理伝播後の成分は角速度 $\mathcal N(f_k(u_i),Q_\omega)$ と、同じuᵢで右回転した姿勢の条件付き法則を持つ。各成分へのgyro更新は解析的に計算できる。

\[
\begin{aligned}
\bar a_i&\propto a_i\mathcal N(y_k;f_k(u_i),Q_\omega+R),\\
G&=Q_\omega(Q_\omega+R)^{-1},\\
\mu_i&=f_k(u_i)+G(y_k-f_k(u_i)),\\
V&=(I-G)Q_\omega(I-G)^\top+GRG^\top.
\end{aligned}
\]

共同モーメントは、この重み・角速度平均・成分内共分散V・姿勢条件付きモーメントを全て使って計算する。例えば角速度共分散は $V+\sum_i\bar a_i(\mu_i-\bar\mu)(\mu_i-\bar\mu)^\top$。姿勢–角速度の交差も同じ成分対応で保持する。

## 数値積分の近似と限界

遷移積分は現在角速度3次元上の決定論的Gauss–Hermite重要度求積で近似する。Gaussian proposalは点の配置にだけ使い、実際のmixture密度／proposal密度で補正する。現在の観測が大きいときは観測を利用して親角速度の積分領域を移すが、観測前の出力は前時刻の法則だけで計算する。事後を単一Gaussianへ置換しない。

角速度を条件とした姿勢モーメントの回転・正重みによる混合は、このモデルで必要なモーメントを閉じて伝える。一方、有限Gaussian-mixture求積は密度の遠い裾を完全には表現できない。これは全計算の厳密解ではない。予定した5・7・9・11点/軸の精細化、実際の全201観測についての線形Kalman解析解との差、極端な交互スパイクの診断を保存する。極端なスパイクでの精度は一様に保証しない。

回転共分散は条件付き共分散＋平均差のGram行列として計算し、生の二次モーメントから大きい平均外積を引くことによる桁落ちを避ける。共分散floor、jitter、固有値の切落としは加えない。有限の残差や求積差で観測を棄却・更新停止せず、その数値を報告する。Monte Carlo・外れ値除去・ロバスト尤度・core変更は行わない。

表示には保存した同じ共同モーメントだけを使う。RPY中心と帯は従来どおり回転平均と全共分散からの局所表示近似であり、保証付き95%領域ではない。Q追加はモデル不確かさの変更で、観測スパイクを識別する仕組みを追加したものではない。

## 再現

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/synthetic-process-noise-mpl \
python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/process_noise_run.py \
  --output /tmp/synthetic-process-noise-comparison-20261120

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 \
  ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/process_noise_verify.py \
  /tmp/synthetic-process-noise-comparison-20261120

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover \
  -s ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/tests -v
```

設定は `process_noise_config.json`。`cache/` は途中次数の検算用でGitへ入れない。完成図・最終共同モーメント・最後の連続mixture・小容量の数値診断・再現コードだけを通常commit/pushする。
