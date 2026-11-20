# Student-t観測で二つの回転モデルを比較する

[結果・図](student_t_publication/INDEX.md)。今回の比較はStudent-t観測を使う非線形Eulerとジャイロ項なしの二つだけである。

## 固定した入力

`omega3_q0_publication` の `source_input.npz`、`sensor_input.npz`、`acquisitions.npz` を再生成せず、同じ値・原時計・indexで使う。Q=0、body角速度3倍、スパイク振幅0.5倍、20秒、生gyro200 Hz、取得10 Hz、201観測。白色生成σ=2 deg/s、固定biasとスパイクの実現値、慣性、初期姿勢、保存1 kHzトルクもそのままである。

推定へ渡すものは、初期姿勢、初回を含む取得gyro、取得時刻、既知慣性と入力トルクだけである。途中の真値、生成bias、実現noise、スパイクフラグを使わない。

## 二モデルと候補法則

状態は $(\omega,q)$、四元数はHamilton・bodyからworldへの回転とする。

\[
\begin{aligned}
\dot\omega_{\mathrm{NL}}&=J^{-1}\{\tau-\omega\times(J\omega)\},\\
\dot\omega_{\mathrm{no\ gyro}}&=J^{-1}\tau,\\
\dot q&=\tfrac12 q\odot(0,\omega).
\end{aligned}
\]

四元数式は両者で共通。初期姿勢 $q_0$ が既知で、入力トルクも既知なので、Q=0では各初期角速度 $w$ から時刻 $t_j$ の $(\omega_m(t_j;w),q_m(t_j;w))$ が決まる。

初期候補は従来と同じ

\[
\pi_0(w)=\mathcal N(w;y_0,R),\qquad R=(5\pi/180)^2I_3
\]

とする。初回gyroはこの初期化に一度だけ使い、以降の尤度に重複して掛けない。したがって初期候補までStudent-tにした比較ではない。変更したのは、その後の全200観測の尤度である。

## 3変量Student-t観測

自由度は実行前に $\nu=4$ と固定した。これはPichéら(2012) §2.1の推奨値を使用したもので、今回の真値や誤差を見て選び直した値ではない。三軸に独立のt分布を掛けるのではなく、三軸共通の精度を持つ3変量分布を使う。

Student-tの尺度行列 $\Sigma$ と周辺共分散 $R$ は異なる。$\nu>2$ では共分散が $\nu\Sigma/(\nu-2)$ なので、従来の周辺標準偏差5 deg/sを保つため、

\[
\Sigma=\frac{\nu-2}{\nu}R=\frac12R
\]

とする。尺度の対角平方根は約3.5355 deg/sであり、これを周辺標準偏差5 deg/sと混同しない。

$e=y-\omega$、$a=e^\top\Sigma^{-1}e$ とすると、正規化済み尤度は

\[
L(y\mid\omega)=
\frac{\Gamma((\nu+3)/2)}{\Gamma(\nu/2)(\nu\pi)^{3/2}|\Sigma|^{1/2}}
\left(1+\frac a\nu\right)^{-(\nu+3)/2}.
\]

これは

\[
\lambda\sim\operatorname{Gamma}(\nu/2,\nu/2),\qquad
y\mid\omega,\lambda\sim\mathcal N(\omega,\Sigma/\lambda)
\]

（Gammaの第2引数はrate）から $\lambda$ を積分した密度である。ここではこの積分を済ませた対数密度を直接評価する。精度に関する乱数や粒子は発生させない。

大きい残差に対して、負の対数尤度の勾配は

\[
\nabla_e[-\log L]=\frac{\nu+3}{\nu+a}\Sigma^{-1}e
\]

となる。残差に比例して無制限に増えるGaussianの勾配と異なり、大きな残差の影響が弱まる。これは尤度自体の性質であり、観測を閾値で棄却したり、表示用の共分散を操作したりする処理ではない。

## 全取得尤度によるBayes更新

各モデルで評価する法則は

\[
\pi_{m,k}(w)\propto
\mathcal N(w;y_0,R)
\prod_{j=1}^{k}
t_\nu\!\left(y_j;\omega_m(t_j;w),\Sigma\right).
\]

各観測を取得順に一度ずつ反映する。Gaussianの初期候補と、Gaussianの数値積分用proposalを、Student-t観測尤度と区別する。観測後分布をproposalのGaussianへ置換しない。

求積点 $w_s$、proposal密度 $g_k$、Gaussian求積重み $a_s$ に対して

\[
\widetilde w_s=
a_s\frac{\mathcal N(w_s;y_0,R)}{g_k(w_s)}
\prod_{j=1}^{k}t_\nu(y_j;\omega_m(t_j;w_s),\Sigma)
\]

を正規化し、同じ点と重みで姿勢・角速度・全交差のモーメントを計算する。過去因子を新しい積分点で再評価することは、観測を重複融合することではない。過去のオンライン出力へ後着観測を戻さない。

proposalは元のGaussian初期候補と全Student-t因子の負の対数へ合わせて配置する。最小二乗の各観測残差は、白色化した3次元残差 $z$ に

\[
\widetilde z=
\sqrt{\frac{(\nu+3)\log(1+\|z\|^2/\nu)}{\|z\|^2}}\,z
\]

を使い、その二乗ノルムがStudent-tの対数尤度項と一致するようにした。$z=0$ は係数の解析的な極限 $\sqrt{(\nu+3)/\nu}$ を使う。ここから得るGauss–Newton行列は求積proposalだけに使い、推定共分散へ代入しない。

各軸7点と11点の決定論的求積で精細化の差を記録する。有限な未収束・求積差は診断値として保存し、観測棄却や更新停止の条件にしない。coreと物理式は変更しない。

## 論文の方式との区別

参照：[Piché, Särkkä, Hartikainen (2012), Recursive Outlier-Robust Filtering and Smoothing for Nonlinear Systems Using the Multivariate Student-t Distribution](https://users.aalto.fi/~ssarkka/pub/studentFiltering_MLSP.pdf), DOI: [10.1109/MLSP.2012.6349794](https://doi.org/10.1109/MLSP.2012.6349794)。

採用したのは同論文のStudent-t観測モデルとGamma精度による解釈である。同論文は状態と精度の事後を変分分解し、Gaussian状態近似を反復更新する。本実装は既存の候補測度・共同伝播を維持するため、精度を解析的に周辺化した尤度を直接掛ける。同論文のVBアルゴリズムそのものを実装したとは説明しない。

## 描画・検証・再現

上段RPY、中段角速度平均±2σ、下段角速度2σ半幅の3×3図と、二モデルの誤差2×3図を作る。Student-tなしの方式は含めない。観測前後401行の保存モーメントをそのまま使用し、全回転9×9共分散・角速度3×3共分散・回転角速度交差を保持する。RPY中心と帯は回転モーメントからの局所表示近似である。

独立検証ではSciPyのStudent-t密度とnative prefixを比較し、元事前×全Student-t尤度／proposalから最終共同モーメントを再構成する。Student-t更新にGaussianの解析共分散R/Nを誤って適用しない。共通する201個の保存済み観測直後で誤差を評価する。

リポジトリrootで実行する。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/probtf-student-t-mpl python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/student_t_run.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/student_t_verify.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/tests -v
```

初期候補・入力・設定・source hash・求積差・全出力モーメント・図を保存する。Gitには小容量の最終成果物だけを追加し、中間求積重みのcache・共有ライブラリ・生成フレームを入れない。
