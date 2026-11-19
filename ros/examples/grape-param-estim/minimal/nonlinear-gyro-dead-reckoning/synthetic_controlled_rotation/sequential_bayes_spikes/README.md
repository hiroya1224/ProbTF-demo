# 合成回転：時系列Bayes更新とスパイク観測

旧独立置換との違いとスパイクの影響を、[結果の図・動画・数値](publication/INDEX.md)へ保存する。対象は元の3周波数軌道、5秒、0.5 Hz取得、仮定σ=5 deg/s。以前の図・解説動画・推定コードとProbTF coreは上書きしない。

## 更新する法則

既知の慣性行列 \(J\)、既知のbody重心まわり入力トルク \(\tau(t)\)、一度だけ与える初期姿勢 \(q_0\)を使う。未知量を初期body角速度 \(W_0\)とする。各候補 \(w\)を、同じ非線形運動方程式へ通す。

\[
\dot\omega=J^{-1}\{\tau-\omega\times(J\omega)\},\qquad
\dot q=\tfrac12 q\odot(0,\omega),\qquad
(q(0),\omega(0))=(q_0,w).
\]

この流れを \(\Phi_t(w)=(q_t(w),\omega_t(w))\) と書く。姿勢と角速度は同じ \(w\) に依存し、観測区間の境界でも独立化しない。ここでは慣性と入力が既知の決定論的モデルなので、途中の過程雑音は追加しない。未知の入力誤差やbiasの推定まで解決するモデルではない。

最初のgyroを一度だけ使い、

\[
\pi_0(dw)=\mathcal N(w;y_0,R)\,dw,\qquad
R=(5\pi/180)^2 I_3
\]

と初期化する。これは初期角速度の拡散的な事前から最初のGaussian観測を得た場合の事後に対応する。独立した情報のある事前を別途与えたとは扱わず、\(y_0\) の尤度を再度掛けない。

後続の取得時刻 \(t_k\) では、

\[
L_k(w)=\mathcal N(y_k;\omega_{t_k}(w),R),\qquad
\pi_k(dw)=\frac{L_k(w)\pi_{k-1}(dw)}{\int L_k(v)\pi_{k-1}(dv)}.
\]

従って、

\[
\pi_k(dw)\propto\mathcal N(w;y_0,R)
\prod_{j=1}^{k}\mathcal N(y_j;\omega_{t_j}(w),R)\,dw.
\]

前回の候補へ今回の因子を掛けるため、過去情報は残る。一般にこの事後はGaussianではなく、平均・共分散だけへ置き換えて再帰する方式でもない。前回の共分散は保持した同じ法則から伝播され、今回の尤度で更新される。

取得後から次の取得直前までの共同法則は

\[
\mu_t=(\Phi_t)_*\pi_k,\qquad t_k\le t<t_{k+1}
\]

である。取得直前・直後を同じ原時刻に別々に保存する。後の観測を使って以前に記録したオンライン出力を修正する平滑化は行わない。

## 決定論的な計算方法

初期Gaussianの3次元積分をtensor Gauss–Hermite求積で近似する。1軸の次数を15、21、31と精細化し、主結果は31³=29,791点。点は固定の求積点であり、乱数サンプルや粒子フィルタではない。旧6点則と物理モデルを変えたのではなく、Bayes事後の積分に使う決定論的数値表現を精細化した。元の6点だけでは、同じ半径の候補へ中心のGaussian尤度を掛けても全重みが等しいままになるなど、事後の縮小を表現しにくい。

各点は一度だけ、角速度と四元数を同じRK4段で積分する。最大刻みは0.002秒、トルクは元の1 kHz保存列から各段で補間する。四元数の有限刻みのnorm誤差を正規化するが、候補分散や重みへfloorを加えない。

観測時は対数重みへGaussian対数尤度を一度加え、log-sum-expで正規化する。全候補を保持し、濃度やinnovationが大きいことを理由に候補・観測を棄却しない。有限求積に関しては、これがその離散法則のBayes更新である。連続法則に対する求積誤差と有限RK4誤差は残る。

同じ更新後の重み \(a_s\) から、

\[
M_q=\sum_s a_s q_s q_s^\top,\quad
\bar\omega=\sum_s a_s\omega_s,\quad
P_\omega=\sum_s a_s(\omega_s-\bar\omega)(\omega_s-\bar\omega)^\top
\]

を計算する。\(M_q\) から回転行列の平均を求める部分は既存coreを再利用する。同じ候補から \(\operatorname{Cov}(\operatorname{vec}R)\) の全9×9、\(\operatorname{Cov}(\operatorname{vec}R,\omega)\) の全9×3も保存する。観測は角速度だけだが、同じ候補の重みを変えるため、姿勢も条件付けられる。独立な四元数積の式を、依存のある姿勢・角速度へ誤用しない。

図のRPY中心は \(f(E[R])\)、帯は同じ回転全共分散をJacobianで写すdelta-methodの±2σ。姿勢自体をGaussianとした厳密な95%領域ではなく、表示近似を推定へ戻さない。

固定事前中心の求積は、極端に細い事後や遠い裾を十分に解像しない場合がある。この実験の3観測では次数間の全時刻の平均・共分散差を保存する。将来、観測を大量に増やしても31点／軸で十分だとは主張しない。積分差を状態・観測の受理基準へ転用しない。

## 白色ノイズとスパイク

白色対照は、以前保存した200 Hzのgyro実現値をそのまま読む。元の生成白色ノイズは等方σ=0.2 deg/s、固定biasは[0.03,−0.02,0.04] deg/s。推定の仮定σ=5 deg/sとは別であり、biasを教えたり差し引いたりしない。

新しいスパイクはseed=20261120で、初期化後の取得時刻から1件を一様に選ぶ。方向は等方、ベクトル長は20〜60 deg/sの一様乱数とし、選んだ原センサindexの観測へ加える。これは「スパイクを取得した場合」を調べる条件付きストレス試験であり、自然な200 Hzセンサの発生頻度を再現したという意味ではない。結果を見てseedを選び直さない。

白色とスパイクの推定へ渡すのはそれぞれのgyro値だけで、共通の時刻・初期姿勢・既知J・既知トルクを使用する。スパイクフラグと真値は描画・診断専用。推定は両方とも同じGaussian観測モデルのままとし、ロバスト尤度・外れ値除去・BFSは追加しない。

そのため、Bayesであってもスパイクを正しい情報として取り込む可能性がある。誤差と幅がどう変わったかを結果で確認する。Bayes化をスパイク耐性の保証とは扱わない。

## 再現と保存

リポジトリのルートから実行する。NumPy、SciPy、Matplotlib、Pillow、既存ProbTF core、動画用FFmpegを使う。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/synthetic-bayes-mpl \
python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/run.py \
  --output /tmp/synthetic-sequential-bayes-spikes-20261120

python3 -m unittest discover \
  -s ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/tests -v
```

`config.json`でσ、取得頻度、求積次数、スパイクseed・個数・大きさを指定する。主結果と異なる設定は別の出力へ保存する。`acquisitions.npz`に原時刻・index・両測定列とスパイク実現値、`bayes_*.npz`に同じオンライン事後の全モーメントと時刻を保存する。初期求積点と観測ごとの対数重みも保存するため、各更新の対応が追える。`events.json`には実際の観測前後と予測尤度、`quadrature_refinement.json`には数値積分差を記録する。

元の推定ファイル・core・解説動画を更新せず、完成図・小容量の数値・再現コードだけを保存する。動画のフレームや中間キャッシュはGitへ入れない。
