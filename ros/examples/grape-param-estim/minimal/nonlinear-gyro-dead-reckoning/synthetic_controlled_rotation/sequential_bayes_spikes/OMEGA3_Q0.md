# Q=0・body角速度を3倍にする比較

最新指定の結果は [omega3_q0_publication](omega3_q0_publication/INDEX.md) に保存する。以前の元速度・Q=0と、元速度・Q>0の結果は保持する。

## 運動と観測の変更範囲

実飛行に合わせた元の滑らかな角速度を $\omega_1(t)$ として、同じ時刻で

\[
\omega_3(t)=3\omega_1(t),\qquad
\alpha_3(t)=3\alpha_1(t)
\]

とする。周波数・時間軸を3倍速にはしない。RPYはbody角速度と異なるので、RPYの振幅だけを3倍にはしない。同じ初期姿勢からHamilton四元数（bodyからworld）を

\[
\dot q_3(t)=\tfrac12q_3(t)\odot(0,\omega_3(t))
\]

で積分し、RPYはこの四元数から導出する。生成側の姿勢はDOP853、相対許容誤差 $10^{-11}$、絶対許容誤差 $10^{-13}$ の決定論的数値積分であり、解析解とは呼ばない。元のRPY微分は新しい運動の微分ではないので、新入力へ誤って引き継がない。

慣性 $J=\operatorname{diag}(0.045,0.070,0.100)$ kg m² は固定し、既知入力を

\[
\tau_3(t)=J\alpha_3(t)+\omega_3(t)\times J\omega_3(t)
=3J\alpha_1(t)+9\{\omega_1(t)\times J\omega_1(t)\}
\]

として元の1 kHz時刻に保存する。推定器にはこの保存入力を既知として与える。途中の真値姿勢を戻す処理ではない。

20秒・生gyro200 Hz・取得10 Hzのまま、生成白色標準偏差2 deg/sと固定biasはそのまま再利用する。最新指定により、147個のスパイクは開始時刻・頻度・幅・方向・重なりを維持し、振幅だけを半分にする。単独パルスの長さは20〜60から10〜30 deg/sとなる。新しい乱数は引かない。`omega3_config.json` の `spike_scale=0.5` が加算ベクトルの倍率である。

\[
y_{3,j}=\omega_3(t_j)+b+\epsilon_j+0.5s_j,
\qquad R=(5\pi/180)^2I_3.
\]

角速度3倍をノイズへ適用しない。生成白色の2 deg/sと、推定が仮定する観測Rの5 deg/sは区別する。スパイクの頻度を半減する変更ではない。旧振幅の比較はgitの直前のcommitに残し、現在の図と数値を更新する。

## Q=0の時系列Bayes更新

プロセス増分を加えず、既存 `full_rate_engine.predict_full_rate` を用いる。両モデルの状態は $(\omega,q)$ であり、

\[
\begin{aligned}
\text{非線形: }&\dot\omega=J^{-1}\{\tau_3-\omega\times J\omega\},\\
\text{ジャイロ項なし: }&\dot\omega=J^{-1}\tau_3
\end{aligned}
\]

だけが異なる。両方の四元数運動学、初期姿勢、初回gyro候補、測定列、R、J、入力トルクは同じ。

初回 $y_{3,0}$ は初期候補に一度だけ使い、以降は

\[
\pi_{m,k}(w)\propto
\mathcal N(w;y_{3,0},R)
\prod_{j=1}^{k}\mathcal N\bigl(y_{3,j};\omega_m(t_j;w),R\bigr)
\]

を評価する。201件すべてを取得順に使い、初回を除く200個の尤度を一度ずつ反映する。観測の直前・直後の共同姿勢／角速度モーメント401行を保存する。真値、生成noise、bias、スパイクフラグは推定用データへ渡さない。

積分には適応配置する決定論的Gaussian求積を使う。Gaussianは求積点配置のproposalであり、事後状態への置換ではない。元の初期事前と全取得尤度を評価し、proposal密度を除く。各軸7点と11点の差、有限なproposal残差を保存し、観測の棄却・更新停止には使わない。Monte Carlo、外れ値除去、BFS、分散floor、core変更は追加しない。

## 表示と検証

上段RPY、中段body角速度平均±2σ、下段角速度の2σ半幅を、同じ保存済み共同モーメントから描く。全9×9回転共分散・全3×3角速度共分散・その交差を残す。RPYの中心と帯だけは回転モーメントからの局所表示近似であり、保証付き95%領域ではない。誤差図は新しい真値と二モデルだけを比較し、旧速度の真値を混ぜない。

主要統計は共通する201個の保存済み観測直後で計算し、推定列を補間しない。補助的な501時刻の統計は `summary.json` 内で別名として区別する。

独立検証は、保存四元数の微分と角速度、既知入力による非線形Euler再積分、線形側の厳密な角速度事後共分散 $R/N$、初期事前×全取得尤度からの最終モーメント再構成を確認する。有限な数値差は診断値として記録する。

## 再現

リポジトリrootから実行する。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/omega3_run.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/omega3_verify.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/tests -v
```

`cache/` はGitへ追加しない。公開ケースでは全時刻の共同モーメントと取得データを保持し、再構成可能な中間求積重みだけを除く。最後の求積重み、全proposalのパラメータ、診断、source hashは保存する。
