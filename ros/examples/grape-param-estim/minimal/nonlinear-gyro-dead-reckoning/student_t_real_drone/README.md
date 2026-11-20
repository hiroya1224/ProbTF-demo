# 実ドローンのStudent-t再試行

[全8条件の図・数値](publication/bag8_50_55s/INDEX.md) ／ [主図：5 Hz・σ=2.5 deg/s](publication/bag8_50_55s/rate_5_hz/sigma_2p5_deg_s/attitude_and_gyro.pdf) ／ [結果と原因の整理](publication/bag8_50_55s/RESULTS.md)。

合成例で用いたStudent-t観測の時系列Bayes更新を、以前の実飛行#8へ適用する。比較はジャイロ項あり／なしの二つだけ。合成の全振幅Student-t結果と以前の実測refine結果は変更せず、全181ファイルの保存hashを照合した。合成結果の凍結commitは `5bdd9072` である。

## 保存入力と固定条件

- bag：`20260612_grape_hovering_8_2026-06-12-17-46-41.bag`。
- 評価：原header時計の50.000988619〜55.000988619秒。開始姿勢だけを以前と同じmocap SLERPで一度整合する。
- モデル：40〜45秒のmocap・指令・実測gimbalからrefineした既存の慣性・アクチュエータモデルを凍結する。新しい同定は行わない。
- トルク：保存したrefine後アクチュエータトルクへ、学習区間だけの余剰トルク平均を一度加える。既に合成済みの `torque_applied_body` へ再加算しない。
- 取得頻度：以前の0.5／2／5 Hzと、直近の合成例に合わせた10 Hz。実行前に固定した。
- Student-t：ν=4、周辺標準偏差2.5／5 deg/s、尺度行列 $\Sigma=(\nu-2)R/\nu$。Q=0。初期候補は従来の $\mathcal N(W_0;y_0,R)$ で初回gyroを一度だけ使う。

原データへ人工noiseやspikeを追加しない。bias除去・時刻ずれ補正・座標合わせ直しもしない。原gyroを要求bag時刻以降の最初の原header標本から選び、値・header/record ns・source indexを保存する。bodyとgyroの取付回転は保存済みの恒等回転を使う。

継続mocap・未取得gyro・評価区間のSG角速度を、推定やモデル・余剰トルクの更新へ渡さない。これらは描画・評価専用。公開入力は既存の小容量抽出ファイルを再利用し、生bagを複製・追加しない。

## 二モデルと更新する法則

\[
\begin{aligned}
\dot\omega_{\rm NL}&=J^{-1}\{\tau_{\rm act}(t)+\bar\tau_{\rm res}-\omega\times(J\omega)\},\\
\dot\omega_{\rm no\ gyro}&=J^{-1}\{\tau_{\rm act}(t)+\bar\tau_{\rm res}\},\\
\dot q&=\tfrac12q\odot(0,\omega).
\end{aligned}
\]

J・トルク・初期姿勢・gyro列・観測Rは共通で、異なるのはジャイロ項だけ。全慣性行列を使い、対角化しない。

モデルmの初期角速度wから得る流れを $(\omega_m(t;w),q_m(t;w))$ として、

\[
\pi_{m,k}(w)\propto
\mathcal N(w;y_0,R)
\prod_{j=1}^{k}t_\nu\bigl(y_j;\omega_m(t_j;w),\Sigma\bigr)
\]

を逐次評価する。Gamma観測精度を解析的に周辺化したStudent-t尤度を直接掛け、Gaussian proposalは決定論的求積点の配置だけに使う。全取得尤度を一度ずつ反映し、全姿勢・角速度と交差のモーメントを同じ候補で計算する。Monte Carlo・core変更・外れ値棄却・共分散floor・Gaussian事後への置換は行わない。

Q=0なので、観測更新後の候補も同じ凍結モデルの初期角速度から生成される軌道に限定される。現在角速度が更新されないという意味ではなく、更新は初期角速度候補の重みを通じて行われる。実機との力学のずれを独立な過程増分へ吸収する自由度はない。

## 以前の実測方式との差

以前の実測refine比較は、取得ごとに角速度を新しい独立Gaussian入力へ置き換え、姿勢分布を持ち越す方式だった。今回は合成例と同じ全履歴Bayes更新であり、その独立置換へ戻していない。したがって旧方式との数値比較は、観測密度だけをGaussianからStudent-tへ置換した単一要因の比較ではない。

同一実機を表現しきれないモデルにQ=0を課すと、Student-tが実際の運動との不一致まで低い尤度の観測として扱う可能性がある。外れ値への頑健性だけで、力学の系統誤差を補正できるとは考えない。結果が悪いことを理由にモデル、Q、R、区間、参照の扱いを変更していない。

## 観測間と最後の尾部

実取得数は0.5／2／5／10 Hzで3／10／25／50件。最後の取得後にも約1.0／0.5／0.2／0.1秒の予測区間が残る。既存engineの観測前後の行を完全に保持し、それ以外の0.01秒格子と固定終端を同じ事後候補の物理伝播で追加する。

追加点は `event_side=0`、観測直前／直後は−1／+1。追加点の `acquisition_index` と尤度数は、その時点までに得た最後の観測を示す。未来の観測を使わず、推定値の補間や終端の水平保持をしない。固定した501時刻を全条件で共通に評価する。

## 図・評価・数値精度

上段はmocap姿勢とRPY中心±2σ、中段は角速度推定±2σと原gyro、下段は同じ角速度共分散の2σ半幅。誤差図も作成する。SG由来角速度の点線は過去に保存したオフライン参照であり、推定入力ではない。角速度誤差の基準は生gyroの参照補間なので、角速度真値に対する誤差とは呼ばない。

四元数二次モーメント、全9×9回転共分散、全3×3角速度共分散と交差を保持する。RPY中心と帯は回転モーメントに対する局所表示近似。モデルパラメータと将来の未知余剰トルクの不確かさを含めた共分散ではなく、その被覆を保証しない。

各軸7点→11点の有限求積差、proposal残差、全尤度からの独立再計算差を保存する。有限な診断値を観測の棄却・更新停止へ使わず、連続分布での収束を未検証のまま保証しない。

## 再現

リポジトリrootから実行する。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/probtf-real-student-mpl python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/student_t_real_drone/run.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/probtf-real-student-mpl python3 -m unittest discover -s ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/student_t_real_drone/tests -v
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/probtf-real-student-mpl python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/student_t_real_drone/verify.py
```

`publication/**/cache/` はGitへ追加しない。最終モーメントと図・設定・必要な小容量入力・検証数値だけを保存する。
