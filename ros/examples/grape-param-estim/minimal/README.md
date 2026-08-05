# 最小構成の実機パラメータ推定

このディレクトリの `estimate_recorded_control.py` は deterministic、deterministic-Sobol、deterministic-tempered、deterministic-Q、probabilistic を切り替える共通エントリポイントです。

既定では `deterministic_estimator.py` のベースライン推定法を呼び出します。

ベースライン推定法は、GUI、Q、潜在状態、residual wrench、EM、MCMC を使わず、SciPy の `least_squares` だけで質量、慣性行列、CoG offset、相対 rotor effectiveness を推定します。

固定 Q と 1 iteration の Laplace-EM を実 bag で比較した結果、解釈、残作業は [現時点の比較結果](RESULTS_ja.md) に記録しています。

既定では同梱 rosbag の 19–24 秒を読み、記録された rotor thrust command と gimbal command を固定アクチュエータモデルへ入れます。

最初に観測 gyro の時間微分と IMU specific force を使う局所運動方程式で初期化し、最後に最初の観測状態から一度も状態をリセットしない 5 秒間の open-loop 軌道誤差を直接最小化します。

## 実行

### Deterministic baseline

```bash
source /home/leus/catkin_ws/devel/setup.bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py"
```

推定結果は `minimal/output/result.json` に、軌道と時系列の比較図は `minimal/output/trajectory.pdf` に生成されます。

PDF は rosbag の observed、推定前の nominal-parameter rollout、推定後の estimated-parameter rollout を同じ図へ重ねます。

Observed は青の実線、nominal は橙の破線、estimated は緑の点線で表し、位置、姿勢、速度、角速度、specific force を表示します。

試行時間を短くする場合は `--max-nfev` を小さくし、軌道 refinement を長く続ける場合は大きくします。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" --max-nfev 50
```

### Sobol multi-start deterministic baseline

`--method deterministic_sobol` は、現行物理パラメータをscreeningだけでなく局所最適化seedにも必ず含めた上で、既定では16次元の bounded chart に `2^9=512` 個の scrambled Sobol 点を生成します。

Mass、相対 rotor effectiveness、thrust/gimbal time constant は対数座標、inertia は正の対角を持つ Cholesky 座標、CoG と delay は通常の有界座標です。
Force effectiveness は geometric mean を1に固定する三つの contrast で表し、baseline と同じ common effectiveness/mass scale ambiguity を追加しません。

各点では剛体 inertia の triangle inequality を確認します。Production PID proposal と同じ acceleration-response compensation からPID gainも計算して結果へ記録しますが、既定ではPID範囲による棄却は行いません。
PID範囲を再び有効にする場合は、下限と上限を両方指定します。
通過した点だけを全軌道 recorded-control open-loop simulation で評価し、損失順に並べてから、正規化パラメータ距離が近すぎる点を除外して最大16点を局所最適化します。

これとは別に、nominal 値から局所運動方程式初期化と軌道 refinement を行う従来 deterministic baseline の経路も、同じrun内で必ず実行して incumbent として保持します。最終出力は、制約適合Sobol解がこの incumbent 以下の軌道損失を達成した場合だけSobol解へ置き換えるため、従来baselineより悪化しません。Baseline incumbent がSobol探索boxまたはPID許容帯を外れる場合は、最終出力の `selected_constraint_eligible` を `false` とし、制約内の最良解も別にJSONへ残します。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_sobol \
  --sobol-power 9 \
  --local-start-count 16 \
  --minimum-seed-distance 0.20
```

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_sobol \
  --pid-gain-min-scale 0.8 \
  --pid-gain-max-scale 1.2
```

結果は `minimal/output/deterministic_sobol/` に保存されます。

十分な全軌道損失を事前に定義できる場合は、収束かつ有効で、境界近傍の変数数も許容内の候補が得られた時点で残りを打ち切れます。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_sobol \
  --early-stop-loss 400 \
  --early-stop-max-boundary-hits 0
```

Delay は causal ZOH command に対して滑らかな変数ではないため、Sobol 点では探索しますが各局所 least-squares 中は固定します。
最終選択では、最小 trajectory loss の既定1%以内にある収束候補から、box bound、inertia triangle inequality、PID許容帯の境界近傍成分が最も少ないものを採用します。

各局所 least-squares の15変数 Jacobian は数値差分ではなく、Cholesky/log物理座標、active branch の actuator model、RK4 rigid-body rollout、SO(3)を含む全観測残差を連鎖した解析 forward sensitivity で計算します。Delay は局所solve中に固定されるため、このJacobianには含まれません。

### Local parallel-tempering deterministic initializer

`--method deterministic_tempered` は、nominal-start deterministic baseline の最良点を中心に、局所 Gaussian 内だけを replica exchange で探索します。全物理範囲へ点を撒かず、baseline 点での解析 trajectory Jacobian の SVD から、細い ridge に沿った proposal covariance を構成します。固有値には上下限を置くため、弱同定方向へ無制限には飛びません。

最初の局所 proposal 48点で実際の正の trajectory-loss 増分を測り、その中央値から8 replica の温度を自動較正します。既定では各 replica を96 sweep 動かし、4 sweep ごとに隣接温度間で状態交換します。各 proposal は forward simulation 1回だけで、全履歴の最良点から解析 Jacobian の least-squares を最後に1回だけ実行します。baseline incumbent も最終比較に残すため、出力 trajectory loss は baseline より悪化しません。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_tempered \
  --replica-count 8 \
  --sweeps 96 \
  --workers 4
```

PID gain は常に計算してJSONへ記録しますが、棄却は既定でオフです。有効にする場合だけ上下限を両方指定します。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_tempered \
  --pid-gain-min-scale 0.8 \
  --pid-gain-max-scale 1.2
```

温度を固定したい場合は `--temperature-min` と `--temperature-max`、探索幅を変える場合は `--proposal-scale` と15成分の `--local-prior-scales` を指定できます。結果は `minimal/output/deterministic_tempered/` に保存されます。Delay はこの局所探索でも固定です。

### Deterministic baseline と対角 Q の交互推定

`--method deterministic_q` は deterministic の single-shooting 軌道誤差を維持し、観測 IMU 軌道上の required-minus-modeled body wrench に対する対角 Q 尤度を追加します。

固定 Q のパラメータ最適化と、Mahalanobis 項および `log det(Q/dt)` を含む Gaussian 尤度の閉形式 Q 更新を交互に実行します。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_q \
  --q-iterations 2 \
  --q-max-nfev 8
```

結果は `minimal/output/deterministic_q/` に保存されます。

この手法は latent trajectory を持たず、Q の residual は順方向シミュレーション軌道ではなく観測軌道から計算する inverse-dynamics residual です。

### GUI backend の根幹アルゴリズムとの比較

`--method probabilistic` は GUI、project、worker process、lag profile、MCMC を通さず、GUI 版と同じ full-trajectory factor graph、解析 Jacobian、sparse MAP、Laplace covariance を直接呼び出します。

最初は変更点を latent full-trajectory MAP だけに限定するため、command lag と Q を固定して実行してください。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method probabilistic \
  --q-policy fixed
```

この段階でも static parameter 18 次元に加えて、各 knot の pose、velocity、angular velocity、controller integral、actuator thrust、gimbal angle と bag-local IMU bias を同時に推定します。

時刻ごとの residual wrench は推定変数にせず、隣接 latent state と物理パラメータから得る 6 次元 body-wrench dynamics residual だけを使います。

次の段階では同じ sparse MAP と Laplace covariance を使い、Q の六つの対角成分を generalized Laplace-EM で更新します。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method probabilistic \
  --q-policy laplace_em \
  --q-em-iterations 2
```

固定 Q の結果は `minimal/output/probabilistic/fixed_q/` に生成されます。

Laplace-EM の結果は `minimal/output/probabilistic/laplace_em/` に生成されます。

各 directory の `method_comparison.pdf` は deterministic、probabilistic、nominal、observed の四者を同じ図で比較します。

現在の bridge では deterministic の推定結果を static parameter の初期値かつ prior mean とし、lag は既定の `0.01 s` に固定しています。

Lag profile と MCMC は、固定 Q と Laplace-EM Q のどちらが open-loop 軌道再現性を維持できるか確認した後に追加する TODO です。
